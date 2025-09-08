import torch.nn as nn
from transformers import GradientCheckpointingLayer, PreTrainedModel
from torchtune.modules import RotaryPositionalEmbeddings
from ..attention_mechanisms import GroupedQueryAttention
from ..feedforward_layers import SwiGLUFeedForward
from .configuration_llama3 import Llama3Config


class LlamaBlock(GradientCheckpointingLayer):
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.input_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attention = GroupedQueryAttention(
            E_q=config.hidden_dim,
            E_k=config.hidden_dim,
            E_v=config.hidden_dim,
            E_total=config.hidden_dim,
            num_heads=config.num_attention_heads,
            num_kv_groups=config.num_key_value_heads,
            qk_norm=False,
            rms_norm_eps=config.rms_norm_eps,
            dropout=config.attention_dropout,
            attention_bias=config.attention_bias,
        )
        self.mid_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.feedforward = SwiGLUFeedForward(
            input_dim=config.hidden_dim,
            intermediary_dim=config.intermediate_dim,
            bias=config.mlp_bias,
        )

    def forward(self, x, mask=None, pos_embedding=None):
        skip = x
        x = self.input_norm(x)
        attention_score = self.attention(
            query=x,
            key=x,
            value=x,
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


class Llama3Model(PreTrainedModel):
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

    def __init__(self, config: Llama3Config):
        super().__init__(config)
        self.config = config
        self.pad_token_id = config.pad_token_id

        self.embedding_layer = nn.Embedding(
            config.vocab_size, config.hidden_dim, config.pad_token_id
        )

        self.transformer_blocks = nn.ModuleList(
            [LlamaBlock(config) for _ in range(config.num_hidden_layers)]
        )

        self.rope = RotaryPositionalEmbeddings(
            dim=config.hidden_dim // config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_base,
        )

        self.final_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self._tie_or_clone_weights(self.embedding_layer, self.lm_head)

        self.post_init()

    def forward(self, x, mask=None):
        x = self.embedding_layer(x)

        for block in self.transformer_blocks:
            x = block(x, mask, self.rope)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits
