import torch.nn as nn


class SwiGLU(nn.module):
    def __init__(self, dim):
        super().__init__()

        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

        self.swish = nn.SiLU()

    def forward(self, x):
        output1 = self.linear1(x)
        output2 = self.linear2(x)

        return self.swish(output1) * output2
