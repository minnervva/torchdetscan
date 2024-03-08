""" This contains pytorch code that intentionally uses non-deterministic
functions, and is used to exercise the linter.

Tests for:

torch.nn.functional.interpolate() when attempting to differentiate a CUDA
tensor and one of the following modes is used:
- linear
- bilinear
- bicubic
- trilinear
"""
import torch

import torch.nn.functional as F

# Assume input is a 1x1x5x5 image (batch size x channels x height x width)
input = torch.randn(1, 1, 5, 5)

# Resize the image to 8x8
output = F.interpolate(input, size=(8, 8), mode='linear', align_corners=False)
output = F.interpolate(input, size=(8, 8), mode='bilinear', align_corners=False)
output = F.interpolate(input, size=(8, 8), mode='nearest', align_corners=False)
output = F.interpolate(input, size=(8, 8), mode='bicubic', align_corners=False)
output = F.interpolate(input, size=(8, 8), mode='trilinear', align_corners=False)
output = F.interpolate(input, size=(8, 8), mode='area', align_corners=False)
output = F.interpolate(input, size=(8, 8), mode='nearest-exact', align_corners=False)
output = F.interpolate(input, size=(8, 8), align_corners=False)
