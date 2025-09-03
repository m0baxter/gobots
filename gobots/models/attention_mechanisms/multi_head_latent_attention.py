import torch
import torch.nn as nn
import torch.nn.functional as F


# add qk-norm option
class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        dropout: float = 0.0,
        attention_bias: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.dropout = dropout

        self.q_down = nn.Linear(d_model, q_lora_rank, bias=attention_bias)
        self.q_up_nope_rope = nn.Linear(
            q_lora_rank,
            self.num_heads * self.q_head_dim,
            bias=attention_bias,
        )

        # Low-rank compression layers for key-value (KV)
        self.kv_down = nn.Linear(
            d_model,
            kv_lora_rank + qk_rope_head_dim,
            bias=attention_bias,
        )
        self.kv_up = nn.Linear(
            kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=attention_bias,
        )

        # Output projection layer
        self.output_proj = nn.Linear(
            self.num_heads * self.v_head_dim, d_model, bias=attention_bias
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask=None,
        pos_embedding=None,
        is_causal=False,
    ):
        batch, seq_len, d_model = x.shape

        # query:
        q = self.q_up_nope_rope(self.q_down(x))
        q = q.view(batch, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # key/value:
        compressed_kv = self.kv_down(x)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(batch, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = self.kv_up(compressed_kv)
        kv = kv.view(
            batch, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        if pos_embedding:
            q_pe = pos_embedding(q_pe.transpose(1, 2)).transpose(1, 2)
            k_pe = pos_embedding(k_pe.transpose(1, 2)).transpose(1, 2)

        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat(
            [k_nope, k_pe.repeat_interleave(self.num_heads, dim=1)], dim=-1
        )

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=is_causal,
            enable_gqa=False,
        )
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        output = self.output_proj(attn_output)

        return output
