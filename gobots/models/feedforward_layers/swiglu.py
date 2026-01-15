import torch.nn as nn


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        intermediary_dim: int,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.intermediary_dim = intermediary_dim
        self.fc1 = nn.Linear(input_dim, intermediary_dim * 2, bias=bias)
        self.fc2 = nn.Linear(intermediary_dim, input_dim, bias=bias)
        self.swish = nn.SiLU()

    def forward(self, x):
        fc1_output = self.fc1(x)
        path1 = fc1_output[..., : self.intermediary_dim]
        path2 = fc1_output[..., self.intermediary_dim :]
        gate = self.swish(path1) * path2

        return self.fc2(gate)
