import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // num_heads
        num_heads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        attention_bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_bias=True,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v

        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(
                E_q, E_total * 3, bias=attention_bias, **factory_kwargs
            )
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=attention_bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=attention_bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=attention_bias, **factory_kwargs)

        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=attention_bias, **factory_kwargs)
        assert E_total % num_heads == 0, "Embedding dim is not divisible by num_heads"
        self.E_head = E_total // num_heads

    @torch.compile(mode="max-autotune")
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, num_heads, E_head) -> (N, num_heads, L_t, E_head)
        query = query.unflatten(-1, [self.num_heads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, num_heads, E_head) -> (N, num_heads, L_s, E_head)
        key = key.unflatten(-1, [self.num_heads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, num_heads, E_head) -> (N, num_heads, L_s, E_head)
        value = value.unflatten(-1, [self.num_heads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, num_heads, L_t, E_head)
        attn_output = flex_attention(
            query,
            key,
            value,
            block_mask=attn_mask,
        )
        # (N, num_heads, L_t, E_head) -> (N, L_t, num_heads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output
