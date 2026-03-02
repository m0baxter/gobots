import torch
import torch.nn as nn
from ..feedforward_layers import SwiGLUFeedForward
from .grouped_gemm.interface import grouped_gemm


class Experts(nn.Module):
    def __init__(
        self,
        n_routed_experts: int = 128,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = n_routed_experts
        self.intermediate_size = intermediate_size

        self.fc1 = nn.Parameter(
            torch.empty(n_routed_experts, intermediate_size * 2, hidden_size)
        )
        self.fc2 = nn.Parameter(
            torch.empty((n_routed_experts, hidden_size, intermediate_size))
        )
        self.swish = nn.SiLU()

    def apply_activation(self, fc1_output):
        path1 = fc1_output[..., : self.intermediate_size]
        path2 = fc1_output[..., self.intermediate_size :]

        return self.swish(path1) * path2

    def forward(self, x, m_sizes, topk, gather_indices):
        fc1_output = grouped_gemm(
            X=x,
            W=self.fc1,
            m_sizes=m_sizes,
            topk=topk,
            gather_indices=gather_indices,
            permute_x=True,
            permute_y=False,
            fuse_mul_post=False,
            autotune=True,
            is_first_gemm=True,
        )
        fc2_input = self.apply_activation(fc1_output)
        fc2_output = grouped_gemm(
            X=fc2_input,
            W=self.fc2,
            m_sizes=m_sizes,
            topk=topk,
            gather_indices=gather_indices,
            permute_x=False,
            permute_y=True,
            fuse_mul_post=False,
            autotune=True,
            is_first_gemm=False,
        )

        return fc2_output


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        shared_expert: bool | None = None,
        n_routed_experts: int = 128,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_experts_per_token: int = 8,
        expert_bias_update_rate: float = 0.001,
        calculate_z_loss: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.shared_expert = None
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_token = num_experts_per_token
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.calculate_z_loss = calculate_z_loss

        if shared_expert:
            self.shared_expert = SwiGLUFeedForward(
                input_dim=hidden_size,
                intermediary_dim=intermediate_size,
                bias=False,
            )
        self.routed_experts = Experts(
            n_routed_experts=n_routed_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        self.router = nn.Linear(
            hidden_size,
            n_routed_experts,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

        self.register_buffer("expert_bias", torch.zeros(n_routed_experts))
        self.expert_bias_update_rate = expert_bias_update_rate

    @torch.no_grad()
    def get_routing_indices(self, selected_experts):
        token_counts_by_expert = torch.histc(
            selected_experts.view(-1),
            bins=self.n_routed_experts,
            min=0,
            max=self.n_routed_experts,
        )
        gather_indices = torch.argsort(selected_experts.view(-1), stable=True)

        return token_counts_by_expert, gather_indices

    def forward(self, x, calculate_auxiliary_loss: bool = True):
        input_shape = x.shape
        num_tokens = input_shape[0] * input_shape[1]
        x_flat = x.view(-1, input_shape[2])

        router_logits = self.router(x_flat)
        router_logits = (
            router_logits - router_logits.mean(dim=-1, keepdim=True)
        ) / router_logits.std(dim=-1, keepdim=True)

        _, selected_experts = torch.topk(
            router_logits + self.expert_bias, self.num_experts_per_token, dim=-1
        )
        router_top_value = router_logits.gather(1, selected_experts)
        router_top_value = self.sigmoid(router_top_value.float()).to(x.dtype)

        token_counts_by_expert, gather_indices = self.get_routing_indices(
            selected_experts
        )

        routed_out = self.routed_experts(
            x_flat, token_counts_by_expert, self.num_experts_per_token, gather_indices
        ).view(num_tokens, self.num_experts_per_token, -1)
        routed_out = routed_out * router_top_value.view(
            num_tokens, self.num_experts_per_token, 1
        )
        routed_out = routed_out.sum(dim=1)

        if self.shared_expert:
            out = self.shared_expert(x_flat)
            out.add_(routed_out)

        else:
            out = routed_out

        out = out.view(input_shape)

        auxiliary_loss = None
        z_loss = None

        if self.training:
            # z-loss:
            # if self.calculate_z_loss:
            #    z_loss = torch.logsumexp(router_top_value, dim=-1) ** 2.0
            #    z_loss = torch.mean(z_loss)

            # calculate axiliary loss:
            router_logits = router_logits.view(
                input_shape[0], input_shape[1], self.n_routed_experts
            )
            avg_expert_prob = router_logits.mean(axis=1)

            indicator = torch.zeros_like(router_logits)
            indicator.scatter_(
                2, selected_experts.view(input_shape[0], input_shape[1], -1), 1
            )
            expert_fraction = indicator.sum(dim=1) / input_shape[1]

            auxiliary_loss = (expert_fraction * avg_expert_prob).sum(axis=-1)

            # perform auxiliary-loss free load balancing:
            with torch.no_grad():
                mean_load = token_counts_by_expert.float().mean()
                self.expert_bias += self.expert_bias_update_rate * torch.sign(
                    mean_load - token_counts_by_expert
                )

        return out, auxiliary_loss, z_loss
