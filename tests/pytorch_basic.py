""" This contains pytorch code that intentionally uses non-deterministic
functions, and is used to exercise the linter.

These examples are copied from

https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
"""
import torch

torch.randn(10, device='cpu').kthvalue(1)

torch.nn.AvgPool3d(1)(torch.randn(3, 4, 5, 6, requires_grad=True)).sum().backward()

