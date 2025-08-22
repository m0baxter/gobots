import torch.nn as nn
from transformers import PreTrainedModel
from torchtune.modules import RotaryPositionalEmbeddings
from ..attention_mechanisms import GroupedQueryAttention
from ..feedforward_layers import SwiGLUFeedForward
from .configuration_llama3 import Llama3Config


class LlamaBlock(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.input_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attention = GroupedQueryAttention(
            E_q=config.hidden_dim,
            E_k=config.hidden_dim,
            E_v=config.hidden_dim,
            E_total=config.hidden_dim,
            nheads=config.num_attention_heads,
            num_kv_groups=config.num_key_value_heads,
            qk_norm=False,
            rms_norm_eps=config.rms_norm_eps,
            dropout=config.attention_dropout,
            bias=config.attention_bias,
        )
        self.mid_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.feedforward = SwiGLUFeedForward(
            input_dim=config.hidden_dim,
            intermediary_dim=config.intermediate_dim,
            bias=config.mlp_bias,
        )

    def forward(self, x, mask, pos_embedding):
        skip = x
        x = self.input_norm(x)
        attention_score = self.attention(
            query=x,
            key=x,
            value=x,
            attn_mask=mask,
            pos_embedding=pos_embedding,
            is_causal=True,
        )

        x = skip + attention_score
        skip = x

        x = self.mid_norm(x)
        x = self.feedforward(x)

        x = skip + x

        return x


class Llama3Model(PreTrainedModel):
    def __init__(self, config: Llama3Config):
        super().__init__(config)
        self.config = config

        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.transformer_blocks = nn.ModuleList(
            [LlamaBlock(config) for _ in range(config.num_hidden_layers)]
        )

        self.rope = RotaryPositionalEmbeddings(
            dim=config.hidden_dim // config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_base,
        )

        self.final_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.output_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, x, mask):
        x = self.embedding_layer(x)

        for block in self.transformer_blocks:
            x = block(x, mask, self.rope)

        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits
