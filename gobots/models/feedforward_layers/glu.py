import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.linear1(x) * self.sigmoid(self.linear2(x))
