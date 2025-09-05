import torch
import torch.nn as nn
from transformers import PreTrainedModel
from torchtune.modules import RotaryPositionalEmbeddings
from ..attention_mechanisms import MultiHeadLatentAttention
from ..feedforward_layers import SwiGLUFeedForward
from ..mixture_of_experts import MixtureOfExperts
from ..multitoken_prediction_layer import MultiTokenPredictionHead
from .configuration_deepseek_v3 import DeepSeekV3Config


class DeepSeekV3Block(nn.Module):
    def __init__(self, config: DeepSeekV3Config, index: int):
        super().__init__()
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

        if index < config.first_k_dense_replace:
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

    def forward(self, x, mask=None, pos_embedding=None):
        skip = x
        x = self.input_norm(x)
        attention_score = self.attention(
            x=x,
            attn_mask=mask,
            pos_embedding=pos_embedding,
            is_causal=mask is None,
        )

        x = skip + attention_score
        skip = x

        x = self.mid_norm(x)
        x = self.feedforward(x)

        x = skip + x

        return x


class DeepSeekV3Model(PreTrainedModel):
    _tied_weights_keys = ["embedding_layer.weight", "lm_head.weight"]

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

    def forward(self, x, mask=None):
        input_ids = x
        x = self.embedding_layer(x)

        for block in self.transformer_blocks:
            x = block(x, mask, self.rope)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if self.num_nextn_predict_layers > 0:
            b, s, d = x.shape
            mtp_logits = []

            current_input_ids = input_ids
            current_hidden = x
            current_logits = logits

            for mtp_head in self.mtp_heads:
                # determine next token:
                next_token = current_logits[:, -1, :].argmax(dim=-1)

                # add to sequence and drop first:
                current_input_ids = torch.cat(
                    [current_input_ids, next_token.view(b, 1)], dim=-1
                )[:, 1:]

                # apply mpt head
                embeds = self.embedding_layer(current_input_ids)
                current_hidden = mtp_head(current_hidden, embeds)

                # add new logit to output
                current_logits = self.lm_head(current_hidden)
                mtp_logits.append(current_logits)

            return logits, mtp_logits

        return logits
