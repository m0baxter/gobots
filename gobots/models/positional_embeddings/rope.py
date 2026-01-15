import torch
import torch.nn as nn

DEVICE_COUNT = torch.cuda.device_count()


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Dynamic RoPE we first set it to a max of 4 * 8192 tokens then we iteratively grow this
        self.current_rope_size = min(4 * 8192, self.max_position_embeddings)
        self.multi_gpu_cache = [None] * DEVICE_COUNT

        # Build here to make `torch.jit.trace` work.
        for device_idx in range(DEVICE_COUNT):
            self._set_cos_sin_cache(
                seq_len=self.current_rope_size,
                device=torch.device(device_idx),
                dtype=torch.get_default_dtype(),
            )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Note: on the original Llama codebase, these tensors are created on the target device (and not on CPU) and
        # in FP32. They are applied (multiplied) in FP32 as well.
        self.current_rope_size = seq_len
        seq_idx = torch.arange(
            seq_len,
            dtype=torch.int64,
            device="cpu",
        )
        theta = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64)[
                    : (self.dim // 2)
                ].float()
                / self.dim
            )
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1).to(
            device=device, non_blocking=True
        )
        self.multi_gpu_cache[device.index] = cache

    def forward(
        self,
        x,
        position_ids=None,
    ):
        # x: [bs, seq_len, num_attention_heads, head_size]
        seq_len = x.size(1)
        device_index = x.device.index

        if seq_len > self.current_rope_size:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = (
            self.multi_gpu_cache[device_index][:seq_len]
            if position_ids is None
            else self.multi_gpu_cache[device_index][position_ids]
        )

        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)

        return x_out.type_as(x)
