import torch as th
import torch.nn as nn
import torch.nn.functional as F

class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(1, embed_dim)
        self.silu = nn.SiLU(inplace=True)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.silu(x)
        x = self.linear2(x)
        return x

class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(num_classes, embed_dim)
        self.silu = nn.SiLU(inplace=True)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.silu(x)
        x = self.linear2(x)
        return x