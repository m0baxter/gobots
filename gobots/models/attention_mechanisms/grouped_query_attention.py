import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
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
        num_kv_groups: int,
        qk_norm: bool = False,
        rms_norm_eps: float = 1.0e-6,
        dropout: float = 0.0,
        attention_bias: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=attention_bias, **factory_kwargs)
        assert E_total % num_heads == 0, "Embedding dim is not divisible by num_heads"
        assert num_heads % num_kv_groups == 0, (
            "num_heads must be divisible by num_kv_groups"
        )
        self.E_head = E_total // num_heads
        self.q_norm, self.k_norm = None, None

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.E_head, eps=rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.E_head, eps=rms_norm_eps)

        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.q_proj = nn.Linear(E_q, E_total, bias=attention_bias, **factory_kwargs)
        self.k_proj = nn.Linear(
            E_k, num_kv_groups * self.E_head, bias=attention_bias, **factory_kwargs
        )
        self.v_proj = nn.Linear(
            E_v, num_kv_groups * self.E_head, bias=attention_bias, **factory_kwargs
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        pos_embedding=None,
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
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, num_heads, E_head) -> (N, num_heads, L_t, E_head)
        query = query.unflatten(-1, [self.num_heads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, num_heads, E_head) -> (N, num_heads, L_s, E_head)
        key = key.unflatten(-1, [self.num_kv_groups, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, num_heads, E_head) -> (N, num_heads, L_s, E_head)
        value = value.unflatten(-1, [self.num_kv_groups, self.E_head]).transpose(1, 2)

        if self.q_norm:
            query = self.q_norm(query)
            key = self.q_norm(key)

        if pos_embedding:
            query = pos_embedding(query.transpose(1, 2)).transpose(1, 2)
            key = pos_embedding(key.transpose(1, 2)).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, num_heads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=is_causal,
            enable_gqa=True,
        )
        # (N, num_heads, L_t, E_head) -> (N, L_t, num_heads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output
