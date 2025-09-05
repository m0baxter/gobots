import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.n_routed_experts, self.hidden_size, 2 * self.intermediate_size
            )
        )
        self.gate_down_proj = nn.Parameter(
            torch.empty(
                (self.n_routed_experts, self.intermediate_size, self.hidden_size)
            )
        )
        self.activation = nn.SiLU()

        self.router = nn.Linear(hidden_size, n_routed_experts, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer("expert_bias", torch.zeros(n_routed_experts))
        self.expert_bias_update_rate = expert_bias_update_rate

    def forward(self, x):
        input_shape = x.shape
        shared_output = x

        for shared_expert in self.shared_experts:
            shared_output = shared_output + shared_expert(shared_output)

        router_logits = self.sigmoid(self.router(x))
        _, selected_experts = torch.topk(
            router_logits + self.expert_bias, self.num_experts_per_token, dim=-1
        )
        routing_weights = router_logits.gather(2, selected_experts)
        routing_weights = F.softmax(routing_weights, dim=-1)

        flat_x = x.view(-1, self.hidden_size)  # (B*T, d_hidden_size)
        flat_router_weights = routing_weights.view(
            -1, self.num_experts_per_token
        )  # (B*T, num_experts_per_token)
        flat_selected_experts = selected_experts.view(
            -1, self.num_experts_per_token
        )  # (B*T, num_experts_per_token)

        routed_output = torch.zeros_like(x)
        expert_load = torch.zeros(self.n_routed_experts).to(self.expert_bias.device)

        for k in range(self.num_experts_per_token):
            expert_idx = flat_selected_experts[
                :, k
            ]  # Indices of the k-th best expert for each token (B*T)
            expert_load += torch.bincount(expert_idx, minlength=self.n_routed_experts)

            # Get weights for the selected experts
            gate_up_w_k = self.gate_up_proj[
                expert_idx
            ]  # (B*T, d_model, 2 * expert_dim)
            down_w_k = self.gate_down_proj[expert_idx]  # (B*T, expert_dim, d_model)

            # Perform expert calculations using bmm
            # Input needs shape (B*T, 1, d_model) for bmm with (B*T, d_model, 2*expert_dim)
            expert_input_k = flat_x.unsqueeze(1)  # (B*T, 1, d_model)
            gate_up_out_k = torch.bmm(
                expert_input_k, gate_up_w_k
            )  # (B*T, 1, 2 * expert_dim)

            # Split gate and up projections
            gate_k, up_k = gate_up_out_k.chunk(2, dim=-1)  # Each (B*T, 1, expert_dim)

            # Apply activation and gating
            activated_up_k = self.activation(gate_k) * up_k  # (B*T, 1, expert_dim)

            # Down projection
            # Input needs shape (B*T, 1, expert_dim) for bmm with (B*T, expert_dim, d_model)
            expert_output_k = torch.bmm(activated_up_k, down_w_k)  # (B*T, 1, d_model)
            expert_output_k = expert_output_k.squeeze(1)  # (B*T, d_model)

            # Weight the expert output
            expert_output_weighted_k = expert_output_k * flat_router_weights[
                :, k
            ].unsqueeze(1)
            routed_output = routed_output + expert_output_weighted_k.view(input_shape)

        if self.training:
            with torch.no_grad():
                mean_load = expert_load.mean()
                self.expert_bias += self.expert_bias_update_rate * torch.sign(
                    mean_load - expert_load
                )

                print(expert_load, mean_load)

        return x + shared_output + routed_output
