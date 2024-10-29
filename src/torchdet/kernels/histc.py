import torch
from dataclasses import dataclass
from torchdet.kernels.utils import HyperParamLoop, HyperParams, LoopParams, Params
from typing import List, Tuple

@dataclass
class HistcHyperParams(HyperParams):
    dim: int
@dataclass
class HistcLoop(HyperParamLoop):
    dim: List[int]
@dataclass
class HistcDim(Params):
    input_dim: Tuple

@dataclass
class HistcDimLoop(LoopParams):
    input_dim: List[Tuple]
    # reduction_ratio: List[float]


def histc_loop(func_name, histc_loop, data_loop):
    """ This is a generator function called by benchmark.func_benchmark
    to generate random parameters for the histc kernel.

    :param func_name: function name; unused since we know it's torch.histc
    :param histc_loop: Contains the ranges for the hyperparameters of the histc kernel.
    :param data_loop:
    :yields: A tuple containing the histc kernel, the parameters for the kernel, the hyperparameters, and the dimension parameters.
    """
    for params in histc_loop:
        histc_params = HistcHyperParams(*params)

        for d_params in data_loop:
            dim_params = HistcDim(*d_params)
            input = (
                histc_params.distribution(torch.zeros(dim_params.input_dim[0]))
                .to(histc_params.dtype)
                .to(histc_params.device)
            )

            yield torch.histc, {
                "input": input,
                "bins": dim_params.input_dim[1],
                "min": dim_params.input_dim[2],
                "max": dim_params.input_dim[3],
            }, histc_params, dim_params
