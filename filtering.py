import torch

y = torch.arange(0, 5)
x = torch.Tensor([True, False, True, False, True]).to(torch.bool)
print(~x)
print(y[x])
