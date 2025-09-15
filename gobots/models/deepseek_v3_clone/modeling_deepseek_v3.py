import torch
import torch.nn as nn
from transformers import GradientCheckpointingLayer, PreTrainedModel
from torchtune.modules import RotaryPositionalEmbeddings
from ..attention_mechanisms import MultiHeadLatentAttention
from ..feedforward_layers import SwiGLUFeedForward
from ..mask_utils import generate_block_mask
from ..mixture_of_experts import MixtureOfExperts
from ..multitoken_prediction_layer import MultiTokenPredictionHead
from .configuration_deepseek_v3 import DeepSeekV3Config


class DeepSeekV3Block(GradientCheckpointingLayer):
    def __init__(self, config: DeepSeekV3Config, index: int):
        super().__init__()
        self.dense_layer = index < config.first_k_dense_replace
        self.input_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attention = MultiHeadLatentAttention(
            d_model=config.hidden_dim,
            num_heads=config.num_attention_heads,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            qk_nope_head_dim=config.qk_nope_head_dim,
            dropout=config.attention_dropout,
            attention_bias=config.attention_bias,
        )
        self.mid_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        if self.dense_layer:
            self.feedforward = SwiGLUFeedForward(
                input_dim=config.hidden_dim,
                intermediary_dim=config.intermediate_dim,
                bias=config.mlp_bias,
            )

        else:
            self.feedforward = MixtureOfExperts(
                n_shared_experts=config.n_shared_experts,
                n_routed_experts=config.n_routed_experts,
                hidden_size=config.hidden_dim,
                intermediate_size=config.moe_intermediate_size,
                num_experts_per_token=config.num_experts_per_tok,
            )

    def forward(self, x, mask=None, pos_embedding=None, input_pos=None):
        skip = x
        x = self.input_norm(x)
        attention_score = self.attention(
            x,
            mask,
            pos_embedding,
            input_pos,
        )

        x = skip + attention_score
        skip = x

        x = self.mid_norm(x)

        auxiliary_losses = None

        if self.dense_layer:
            x = self.feedforward(x)

        else:
            x, auxiliary_losses = self.feedforward(x)

        x = skip + x

        return x, auxiliary_losses


class DeepSeekV3Model(PreTrainedModel):
    _tied_weights_keys = ["embedding_layer.weight", "lm_head.weight"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

        elif isinstance(module, MixtureOfExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.gate_down_proj.data.normal_(mean=0.0, std=std)

    def __init__(self, config: DeepSeekV3Config):
        super().__init__(config)

        self.config = config
        self.pad_token_id = config.pad_token_id
        self.num_nextn_predict_layers = config.num_nextn_predict_layers

        self.embedding_layer = nn.Embedding(
            config.vocab_size, config.hidden_dim, self.pad_token_id
        )

        self.transformer_blocks = nn.ModuleList(
            [
                DeepSeekV3Block(config, index)
                for index in range(config.num_hidden_layers)
            ]
        )
        self.rope = RotaryPositionalEmbeddings(
            dim=config.qk_rope_head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_base,
        )

        self.final_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self._tie_or_clone_weights(self.embedding_layer, self.lm_head)

        if config.num_nextn_predict_layers > 0:
            self.mtp_heads = nn.ModuleList(
                [
                    MultiTokenPredictionHead(config)
                    for _ in range(config.num_nextn_predict_layers)
                ]
            )

        else:
            self.mtp_heads = None

        self.post_init()

    def forward(self, x, mask=None, document_ids=None, input_pos=None):
        b, s = x.shape
        input_ids = x
        x = self.embedding_layer(x)
        auxiliary_losses = []

        if mask is None:
            mask = generate_block_mask(
                batch_size=b,
                query_length=s,
                key_value_length=s,
                document_ids=document_ids,
            )

        for block in self.transformer_blocks:
            x, aux_loss = block(x, mask, self.rope, input_pos)

            if aux_loss is not None:
                auxiliary_losses.append(aux_loss)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        mtp_logits = None

        if self.num_nextn_predict_layers > 0:
            b, s, d = x.shape
            mtp_logits = []

            current_input_ids = input_ids
            current_hidden = x
            current_logits = logits
            shifted_document_ids = document_ids
            shifted_input_pos = input_pos

            for mtp_head in self.mtp_heads:
                # shift document_ids to create new mask
                if shifted_document_ids is not None:
                    shifted_document_ids = torch.nn.functional.pad(
                        shifted_document_ids, (0, 1), mode="replicate"
                    )[:, 1:]

                    mask = generate_block_mask(
                        batch_size=b,
                        query_length=s,
                        key_value_length=s,
                        document_ids=shifted_document_ids,
                    )
                    mask = torch._dynamo.mark_dynamic(mask, index=0)

                # shift the input positional ids:
                if shifted_input_pos is not None:
                    shifted_input_pos = torch.nn.functional.pad(
                        shifted_input_pos, (0, 1), mode="replicate"
                    )
                    shifted_input_pos[:, -1] += 1
                    shifted_input_pos = shifted_document_ids[:, 1:]

                # determine next token:
                next_token = current_logits[:, -1, :].argmax(dim=-1)

                # add to sequence and drop first:
                current_input_ids = torch.cat(
                    [current_input_ids, next_token.view(b, 1)], dim=-1
                )[:, 1:]

                # apply mpt head
                embeds = self.embedding_layer(current_input_ids)
                current_hidden, aux_loss = mtp_head(
                    current_hidden,
                    embeds,
                    mask,
                    self.rope,
                    shifted_document_ids,
                )

                if aux_loss is not None:
                    auxiliary_losses.append(aux_loss)

                # add new logit to output
                current_logits = self.lm_head(current_hidden)
                mtp_logits.append(current_logits)

        return {
            "logits": logits,
            "mtp_logits": mtp_logits,
            "auxiliary_losses": auxiliary_losses,
        }
