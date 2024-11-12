import torch
from dataclasses import dataclass
from torchdetscan.kernels.utils import (
    HyperParamLoop,
    HyperParams,
    LoopParams,
    Params,
    initialise_weights,
)
from typing import List, Tuple


@dataclass
class MaxPoolHyperParams(HyperParams):
    kernel_size: Tuple
    stride: int
    padding: int
    dilation: int
    ceil_mode: bool


@dataclass
class MaxPoolLoop(HyperParamLoop):
    kernel_size: List[Tuple]
    stride: List[int]
    padding: List[int]
    dilation: List[int]
    ceil_mode: List[bool]


@dataclass
class MaxPoolDim(Params):
    batch_size: int
    dim: Tuple


@dataclass
class MaxPoolDimLoop(LoopParams):
    batch_size: List[int]
    dim: List[Tuple]


def max_pool_loop(nnmodule, max_pool_loop, data_loop):
    for params in max_pool_loop:
        max_pool_params = MaxPoolHyperParams(*params)
        make_pool_model = max_pool_params.asdict()
        make_pool_model.pop("device")
        make_pool_model.pop("dtype")
        make_pool_model.pop("distribution")
        pool_model = nnmodule(**make_pool_model)
        initialise_weights(pool_model, max_pool_params.distribution)
        pool_model = pool_model.to(max_pool_params.dtype)
        pool_model = pool_model.to(max_pool_params.device)

        for d_params in data_loop:
            dim_params = MaxPoolDim(*d_params)
            dims = [dim_params.batch_size, *dim_params.dim]
            pool_input = torch.randn(dims, dtype=max_pool_params.dtype)
            pool_input = pool_input.to(max_pool_params.device)
            yield pool_model, pool_input, max_pool_params, dim_params
