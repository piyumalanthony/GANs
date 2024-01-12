import torch

x = torch.randn(3, requires_grad=False)

print(x)

y = x * 2
z = y + 2
a = x.mean()
b = y.mean()
print(a)
print(b)
print(y)
print(z)



torch.optim.ASGD