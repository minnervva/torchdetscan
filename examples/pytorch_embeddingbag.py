""" This contains pytorch code that intentionally uses non-deterministic
functions, and is used to exercise the linter.

"""
import torch.nn as nn

embedding_sum = nn.EmbeddingBag(10, 3, mode='sum')
embedding_sum = nn.EmbeddingBag(10, 3, mode='max')
embedding_sum = nn.EmbeddingBag(10, 3)
