import torch
from dataclasses import dataclass
from torchdet.kernels.utils import HyperParamLoop, HyperParams, LoopParams, Params
from typing import List, Tuple

@dataclass
class BinCountHyperParams(HyperParams):
    dim: int
@dataclass
class BinCountLoop(HyperParamLoop):
    dim: List[int]

@dataclass
class BinCountDim(Params):
    input_range: Tuple
    size: int

@dataclass
class BinCountDimLoop(LoopParams):
    input_range: List[Tuple]
    size: List[int]
    # reduction_ratio: List[float]


def bin_count_loop(func_name:str, bin_count_loop, data_loop):
    """ This is a generator function called by benchmark.func_benchmark
    to generate random parameters for the tensor.put_ kernel.

    :param func_name: function name; unused since we know it's torch.bincount
    :param tensor_put_loop: Contains the ranges for the hyperparameters of the torch.bincount kernel.
    :param data_loop:
    :yields: A tuple containing the histc kernel, the parameters for the kernel, the hyperparameters, and the dimension parameters.
    """
    for params in bin_count_loop:
        bin_count_params = BinCountHyperParams(*params)

        for d_params in data_loop:
            dim_params = BinCountDim(*d_params)
            input = (torch.randint(dim_params.input_range[0], dim_params.input_range[1], (dim_params.size,))
                .to(bin_count_params.dtype)
                .to(bin_count_params.device)
            )

            weights = (torch.linspace(0., 1.0, steps=dim_params.size)
                .to(torch.float32)
                .to(bin_count_params.device)
            )

            yield torch.bincount, {
                "input":input,
                "weights": weights
            }, bin_count_params, dim_params
