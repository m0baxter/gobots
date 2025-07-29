import torch.nn as nn


class SwiGLUFeedForward(nn.module):
    def __init__(self, dimension):
        super().__init__()

        self.fc1 = nn.Linear(dimension, dimension)
        self.fc2 = nn.Linear(dimension, dimension)
        self.fc3 = nn.Linear(dimension, dimension)

        self.swish = nn.SiLU()

    def forward(self, x):
        path1 = self.fc1(x)
        path2 = self.fc2(x)
        gate = self.swish(path1) * path2

        return self.fc3(gate)
