""" This contains pytorch code that intentionally uses non-deterministic
functions, and is used as to exercise the linter.
"""

import torch

torch.randn(10, device='cuda').kthvalue(1)

torch.nn.AvgPool3d(1)(torch.randn(3, 4, 5, 6, requires_grad=True).cuda()).sum().backward()

