import torch
import torch.nn as nn
from transformers import GradientCheckpointingLayer, PretrainedConfig
from ..attention_mechanisms import (
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiHeadLatentAttention,
)
from ..feedforward_layers import SwiGLUFeedForward
from ..mixture_of_experts import MixtureOfExperts

_attention_mechanisms = {
    "grouped_query_attention": GroupedQueryAttention,
    "multi_head_attention": MultiHeadAttention,
    "multi_head_latent_attention": MultiHeadLatentAttention,
}
_feedforward_layers = {
    "dense": SwiGLUFeedForward,
    "moe": MixtureOfExperts,
}


class MultiTokenPredictionHead(GradientCheckpointingLayer):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.feedforward_type = config.mtp_config["feedforward_type"]

        # Combine previous hidden state with future token embedding
        self.combine_proj = nn.Linear(
            2 * config.hidden_dim, config.hidden_dim, bias=config.attention_bias
        )

        self.norm1 = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        self.attention = _attention_mechanisms[config.mtp_config["attention_type"]](
            **config.mtp_config
        )

        self.feedforward = _feedforward_layers[config.mtp_config["feedforward_type"]](
            **config.mtp_config
        )

        self.attn_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.mlp_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

    def forward(self, prev_hidden, future_token_embed):
        # Normalize inputs
        prev_norm = self.norm1(prev_hidden)
        future_norm = self.norm2(future_token_embed)

        # Combine representations
        combined = torch.cat([prev_norm, future_norm], dim=-1)
        hidden = self.combine_proj(combined)

        # Process through transformer components
        hidden = hidden + self.attention(self.attn_norm(hidden))

        auxiliary_losses = None

        if self.feedforward_type == "moe":
            output, auxiliary_losses = self.feedforward(self.mlp_norm(hidden))
            hidden = hidden + output

        else:
            hidden = hidden + self.feedforward(self.mlp_norm(hidden))

        return hidden, auxiliary_losses
