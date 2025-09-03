import torch.nn as nn


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self, input_dim: int, intermediary_dim: int, bias: bool = False, **kwargs
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, intermediary_dim, bias=bias)
        self.fc2 = nn.Linear(input_dim, intermediary_dim, bias=bias)
        self.fc3 = nn.Linear(intermediary_dim, input_dim, bias=bias)

        self.swish = nn.SiLU()

    def forward(self, x):
        path1 = self.fc1(x)
        path2 = self.fc2(x)
        gate = self.swish(path1) * path2

        return self.fc3(gate)
