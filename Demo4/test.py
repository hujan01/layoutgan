import torch
from torch import nn
m = nn.MaxPool1d(3)
input = torch.randn(20, 16, 15)
output = m(input)
print(output.shape)