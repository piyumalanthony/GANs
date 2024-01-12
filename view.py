import torch

# a = torch.randn(10, 5)
# print(a)
# b = a.view(50, -1)
# print(b)
# print(b.shape)
# print(torch.squeeze(b))
#
# x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
# print(x.shape)
# print(x)
# y = x.view(3, 2)
# print(y.shape)
# print(y)
# print(x.t())


# a = torch.randn(3, 2)
# b = a.view(3, 2, 1, 1)
# print("a:", a)
# print("b:", b)

# a = torch.arange(1,10)
# b = torch.randn(10)
# print(a)
# c = torch.tensor(torch.cat((a,b), dim=0), dtype=float)
# print(c)

x = torch.randn(2, 3)
print(x)
x = x[:, :, None, None]
x = x.repeat(1, 1, 4, 3)
print(x)
