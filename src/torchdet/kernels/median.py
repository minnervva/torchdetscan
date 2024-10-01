""" Benchmark for median kernel. """
import torch
from dataclasses import dataclass
from torchdet.kernels.utils import HyperParamLoop, HyperParams, LoopParams, Params
from typing import List, Tuple


@dataclass
class MedianHyperParams(HyperParams):
    dim: int


@dataclass
class MedianLoop(HyperParamLoop):
    dim: List[int]


@dataclass
class MedianDim(Params):
    input_dim: Tuple
    # reduction_ratio: float


@dataclass
class MedianDimLoop(LoopParams):
    input_dim: List[Tuple]
    # reduction_ratio: List[float]


def median_loop(func_name, median_loop, data_loop):
    """ This is a generator function called by benchmark.func_benchmark
    to generate random parameters for the median kernel.

    :param func_name: function name; unused since we know it's torch.median
    :param median_loop: Contains the ranges for the hyperparameters of the median kernel.
    :param data_loop:
    :yields: A tuple containing the median kernel, the parameters for the kernel, the hyperparameters, and the dimension parameters.
    """
    for params in median_loop:
        median_params = MedianHyperParams(*params)

        for d_params in data_loop:
            dim_params = MedianDim(*d_params)
            input = (
                median_params.distribution(torch.zeros(dim_params.input_dim))
                .to(median_params.dtype)
                .to(median_params.device)
            )

            yield torch.median, {
                "input": input,
                # "dim": median_params.input_dim,
                # "keepdim": median_params.keepdim,
            }, median_params, dim_params
