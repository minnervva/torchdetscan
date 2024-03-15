""" This contains pytorch code that intentionally uses non-deterministic
functions, and is used to exercise the linter.

"""
import torch

input = torch.tensor([1., 2., 3., 4.])
input.scatter_reduce(0, index, src, reduce="sum")
input.scatter_reduce(0, index, src, reduce="prod")
input.scatter_reduce(0, index, src, reduce="mean")
input.scatter_reduce(0, index, src, reduce="amax")
input.scatter_reduce(0, index, src, reduce="amin")

