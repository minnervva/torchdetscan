""" This contains pytorch code that intentionally uses non-deterministic
functions, and is used to exercise the linter.

"""
import torch


src = torch.tensor([[4, 3, 5],
                    [6, 7, 8]])

src.put_(torch.tensor([1, 3]),
         torch.tensor([9, 10]),
         accumulate=True)

src.put_(torch.tensor([1, 3]),
         torch.tensor([9, 10]),
         accumulate=False)

src.put_(torch.tensor([1, 3]),
         torch.tensor([9, 10]))