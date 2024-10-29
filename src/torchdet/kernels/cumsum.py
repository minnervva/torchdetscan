import torch
from dataclasses import dataclass
from torchdet.kernels.utils import HyperParamLoop, HyperParams, LoopParams, Params
from typing import List, Tuple

@dataclass
class CumSumHyperParams(HyperParams):
    dim: int
@dataclass
class CumSumLoop(HyperParamLoop):
    dim: List[int]

@dataclass
class CumSumDim(Params):
    size: int

@dataclass
class CumSumDimLoop(LoopParams):
    size: List[int]
    # reduction_ratio: List[float]


def cum_sum_loop(func_name:str, cum_sum_loop, data_loop):
    """ This is a generator function called by benchmark.func_benchmark
    to generate random parameters for the torch.cumsum kernel.

    :param func_name: function name; unused since we know it's torch.bincount
    :param cum_sum_loop: Contains the ranges for the hyperparameters of the torch.cumsum kernel.
    :param data_loop:
    :yields: A tuple containing the cumsum kernel, the parameters for the kernel, the hyperparameters, and the dimension parameters.
    """
    for params in cum_sum_loop:
        cum_sum_params = CumSumHyperParams(*params)

        for d_params in data_loop:
            dim_params = CumSumDim(*d_params)

            weights = (
                cum_sum_params.distribution(torch.zeros(dim_params.size))
                .to(cum_sum_params.dtype)
                .to(cum_sum_params.device)
            )

            yield torch.cumsum, {
                "input":weights,
                "dim":0,
            }, cum_sum_params, dim_params
