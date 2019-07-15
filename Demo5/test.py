import torch
from torch import nn

torch.manual_seed(2) #设置随机数种子
input = torch.rand(3, 3) #均匀分布

m = nn.Sigmoid()
a = torch.Tensor([-1])
print(m(a))
print(torch.sigmoid(a))
print(m(input))
target = torch.FloatTensor([[0, 1, 1],
                            [0, 0, 1],
                            [1, 0, 1]])

loss = nn.BCELoss()
print(loss(input, target))

loss1 = nn.BCEWithLogitsLoss()
print(loss1(input, target))