import torch
from dataclasses import dataclass
from torchdet.kernels.utils import (
    HyperParamLoop,
    HyperParams,
    LoopParams,
    Params,
    initialise_weights,
)
from typing import List, Tuple


@dataclass
class MaxUnpoolHyperParams(HyperParams):
    kernel_size: Tuple
    stride: int
    padding: int


@dataclass
class MaxUnpoolLoop(HyperParamLoop):
    kernel_size: List[Tuple]
    stride: List[int]
    padding: List[int]


@dataclass
class MaxUnpoolDim(Params):
    batch_size: int
    dim: Tuple


@dataclass
class MaxUnpoolDimLoop(LoopParams):
    batch_size: List[int]
    dim: List[Tuple]


def max_unpool_loop(nnmodule, max_pool_loop, data_loop):
    for params in max_pool_loop:
        max_unpool_params = MaxUnpoolHyperParams(*params)
        make_unpool_model = max_unpool_params.asdict()
        make_unpool_model.pop("device")
        make_unpool_model.pop("dtype")
        make_unpool_model.pop("distribution")
        unpool_model = nnmodule(**make_unpool_model)
        initialise_weights(unpool_model, max_unpool_params.distribution)
        unpool_model = unpool_model.to(max_unpool_params.dtype)
        unpool_model = unpool_model.to(max_unpool_params.device)

        # MaxPools use same args as unpool
        pool_model = None
        if isinstance(unpool_model, torch.nn.MaxUnpool1d):
            pool_model = torch.nn.MaxPool1d(**make_unpool_model, return_indices=True)
        elif isinstance(unpool_model, torch.nn.MaxUnpool2d):
            pool_model = torch.nn.MaxPool2d(**make_unpool_model, return_indices=True)
        elif isinstance(unpool_model, torch.nn.MaxUnpool3d):
            pool_model = torch.nn.MaxPool3d(**make_unpool_model, return_indices=True)
        initialise_weights(pool_model, max_unpool_params.distribution)
        pool_model = pool_model.to(max_unpool_params.dtype)
        pool_model = pool_model.to(max_unpool_params.device)

        for d_params in data_loop:
            dim_params = MaxUnpoolDim(*d_params)
            dims = [dim_params.batch_size, *dim_params.dim]
            pool_input = torch.randn(dims, dtype=max_unpool_params.dtype)
            pool_input = pool_input.to(max_unpool_params.device)
            input, indices = pool_model(pool_input)
            yield unpool_model.forward, {
                "input": input,
                "indices": indices,
                "output_size": pool_input.size(),
            }, max_unpool_params, dim_params
