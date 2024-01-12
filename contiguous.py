import torch

non_contiguous_tensor = torch.randn(2, 3)
print(non_contiguous_tensor.is_contiguous())

sliced_tensor = non_contiguous_tensor[:, 1:]
print(sliced_tensor.is_contiguous())

contiguous_tensor = sliced_tensor.contiguous()
print(contiguous_tensor.is_contiguous())
