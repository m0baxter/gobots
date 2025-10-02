import torch
import torch.nn as nn
from ..feedforward_layers import SwiGLUFeedForward


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        n_shared_experts: int = 1,
        n_routed_experts: int = 128,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_experts_per_token: int = 8,
        expert_bias_update_rate: float = 0.001,
        **kwargs,
    ):
        super().__init__()
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_token = num_experts_per_token
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

        self.shared_experts = nn.ModuleList(
            [
                SwiGLUFeedForward(
                    input_dim=hidden_size,
                    intermediary_dim=intermediate_size,
                    bias=False,
                )
                for _ in range(n_shared_experts)
            ]
        )
        self.routed_experts = nn.ModuleList(
            [
                SwiGLUFeedForward(
                    input_dim=hidden_size,
                    intermediary_dim=intermediate_size,
                    bias=False,
                )
                for _ in range(n_routed_experts)
            ]
        )

        self.router = nn.Linear(hidden_size, n_routed_experts, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer("expert_bias", torch.zeros(n_routed_experts))
        self.expert_bias_update_rate = expert_bias_update_rate

    @torch.compiler.disable(recursive=False)
    def forward(self, x, calculate_auxiliary_loss: bool = True):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        input_shape = x.shape
        shared_output = x

        for shared_expert in self.shared_experts:
            shared_output = shared_output + shared_expert(shared_output)

        with torch.autocast(device_type=device_type, enabled=False):
            pre_sigmoid = self.router(x.to(dtype=torch.float32))
            router_logits = self.sigmoid(pre_sigmoid)

            _, selected_experts = torch.topk(
                router_logits + self.expert_bias, self.num_experts_per_token, dim=-1
            )

            # z-loss:
            z_loss = (
                torch.logsumexp(pre_sigmoid.gather(2, selected_experts), dim=-1) ** 2.0
            )
            z_loss = torch.mean(z_loss)

            routing_weights = router_logits.gather(2, selected_experts)
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )

            flat_x = x.view(-1, self.hidden_size)  # (B*T, d_hidden_size)
            flat_router_weights = routing_weights.view(
                -1, self.num_experts_per_token
            )  # (B*T, num_experts_per_token)
            flat_selected_experts = selected_experts.view(
                -1, self.num_experts_per_token
            )  # (B*T, num_experts_per_token)

            routed_output = torch.zeros_like(flat_x)
            expert_load = torch.zeros(self.n_routed_experts).to(self.expert_bias.device)
            auxiliary_loss = None

            # calculate axiliary loss:
            if calculate_auxiliary_loss:
                avg_expert_prob = router_logits.mean(axis=1)

                indicator = torch.zeros_like(router_logits)
                indicator.scatter_(2, selected_experts, 1)
                expert_fraction = indicator.sum(dim=1) / input_shape[1]

                auxiliary_loss = (expert_fraction * avg_expert_prob).sum(axis=-1)

        for expert_idx in range(self.n_routed_experts):
            expert = self.routed_experts[expert_idx]
            token_indices, weight_indices = torch.where(
                flat_selected_experts == expert_idx
            )
            expert_weights = flat_router_weights[token_indices, weight_indices]

            if token_indices.numel() > 0:
                expert_input = flat_x[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                routed_output.index_add_(
                    0, token_indices, weighted_output.to(dtype=routed_output.dtype)
                )

                expert_load[expert_idx] = token_indices.numel()

        routed_output = routed_output.view(input_shape)

        # perform auxiliary-loss free load balancing:
        if self.training:
            with torch.no_grad():
                mean_load = expert_load.mean()
                self.expert_bias += self.expert_bias_update_rate * torch.sign(
                    mean_load - expert_load
                )

        return x + shared_output + routed_output, auxiliary_loss, z_loss
