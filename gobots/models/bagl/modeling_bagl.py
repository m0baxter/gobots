import torch
import torch.nn as nn
from transformers import GradientCheckpointingLayer, PreTrainedModel
from torchtune.modules import RotaryPositionalEmbeddings
from ..attention_mechanisms import GroupedQueryAttention
from ..feedforward_layers import SwiGLUFeedForward
from ..mask_utils import generate_block_mask
from ..mixture_of_experts import MixtureOfExperts
from ..multitoken_prediction_layer import MultiTokenPredictionHead
from .configuration_bagl import BaGLConfig


class BaGLBlock(GradientCheckpointingLayer):
    def __init__(self, config: BaGLConfig, is_dense: bool):
        super().__init__()
        self.is_dense = is_dense
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = GroupedQueryAttention(
            E_q=config.hidden_size,
            E_k=config.hidden_size,
            E_v=config.hidden_size,
            E_total=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_groups=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            dropout=config.attention_dropout,
            attention_bias=config.attention_bias,
            qk_norm=config.use_qk_norm,
        )
        self.mid_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.is_dense:
            self.feedforward = SwiGLUFeedForward(
                input_dim=config.hidden_size,
                intermediary_dim=config.intermediate_size_mlp,
                bias=config.mlp_bias,
            )

        else:
            self.feedforward = MixtureOfExperts(
                n_shared_experts=config.n_shared_experts,
                n_routed_experts=config.n_routed_experts,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts_per_token=config.num_experts_per_tok,
            )

    def forward(self, x, mask=None, pos_embedding=None, input_pos=None):
        skip = x
        x = self.input_norm(x)
        attention_score = self.attention(
            x,
            x,
            x,
            mask,
            pos_embedding,
            input_pos,
        )

        x = skip + attention_score
        skip = x

        x = self.mid_norm(x)
        auxiliary_losses = None
        z_loss = None

        if self.is_dense:
            x = self.feedforward(x)

        else:
            x, auxiliary_losses, z_loss = self.feedforward(x)

        x = skip + x

        return x, auxiliary_losses, z_loss


class BaGLModel(PreTrainedModel):
    _tied_weights_keys = ["embedding_layer.weight", "lm_head.weight"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            w_fan_in = module.weight.shape[-1]
            std = torch.math.sqrt(self.config.initializer_range / w_fan_in)

            module.weight.data.normal_(mean=0.0, std=std)
            module.weight.data.clamp_(min=-2 * std, max=2 * std)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            w_fan_in = module.weight.shape[-1]
            std = torch.math.sqrt(self.config.initializer_range / w_fan_in)
            module.weight.data.normal_(mean=0.0, std=std)

            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

    def __init__(self, config: BaGLConfig):
        super().__init__(config)

        self.config = config
        self.hidden_size = config.hidden_size
        self.nope_layers = set(config.nope_layers)
        self.dense_layers = set(config.dense_layers)
        self.pad_token_id = config.pad_token_id
        self.num_nextn_predict_layers = config.num_nextn_predict_layers

        self.embedding_layer = nn.Embedding(
            config.vocab_size, config.hidden_size, self.pad_token_id
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BaGLBlock(config, is_dense=idx in self.dense_layers)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.rope = RotaryPositionalEmbeddings(
            dim=config.hidden_size // config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_base,
        )

        self.embed_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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

    def forward(
        self, input_ids, mask=None, document_ids=None, input_pos=None, labels=None
    ):
        b, s = input_ids.shape
        x = self.embed_norm(self.embedding_layer(input_ids))
        auxiliary_losses = []
        z_losses = []

        if mask is None:
            mask = generate_block_mask(
                batch_size=b,
                query_length=s,
                key_value_length=s,
                document_ids=document_ids,
            )

        for i, block in enumerate(self.transformer_blocks):
            pos_embed = None if i in self.nope_layers else self.rope
            x, aux_loss, z_loss = block(x, mask, pos_embed, input_pos)

            if aux_loss is not None:
                auxiliary_losses.append(aux_loss)

            if z_loss is not None:
                z_losses.append(z_loss)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        mtp_logits = []

        if self.num_nextn_predict_layers > 0:
            b, s, d = x.shape

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
                embeds = self.embed_norm(self.embedding_layer(current_input_ids))
                current_hidden, aux_loss, z_loss = mtp_head(
                    current_hidden,
                    embeds,
                    mask,
                    self.rope,
                    shifted_document_ids,
                )

                if aux_loss is not None:
                    auxiliary_losses.append(aux_loss)

                if z_loss is not None:
                    z_losses.append(z_loss)

                # add new logit to output
                current_logits = self.lm_head(self.final_norm(current_hidden))
                mtp_logits.append(current_logits)

        return {
            "logits": logits,
            "mtp_logits": mtp_logits,
            "auxiliary_losses": auxiliary_losses,
            "z_losses": z_losses,
        }
