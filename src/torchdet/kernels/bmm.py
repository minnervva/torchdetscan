import torch
from dataclasses import dataclass
from torchdet.kernels.utils import HyperParamLoop, HyperParams, LoopParams, Params
from typing import List, Tuple

@dataclass
class BmmHyperParams(HyperParams):
    dim: int
@dataclass
class BmmLoop(HyperParamLoop):
    dim: List[int]
@dataclass
class BmmDim(Params):
    input_dim: Tuple

@dataclass
class BmmDimLoop(LoopParams):
    input_dim: List[Tuple]
    # reduction_ratio: List[float]


def bmm_loop(func_name, bmm_loop, data_loop):
    """ This is a generator function called by benchmark.func_benchmark
    to generate random parameters for the bmm kernel.

    :param func_name: function name; unused since we know it's torch.bmm
    :param bmm_loop: Contains the ranges for the hyperparameters of the bmm kernel.
    :param data_loop:
    :yields: A tuple containing the bmm kernel, the parameters for the kernel, the hyperparameters, and the dimension parameters.
    """
    for params in bmm_loop:
        bmm_params = BmmHyperParams(*params)

        for d_params in data_loop:
            dim_params = BmmDim(*d_params)
            a_size = (dim_params.input_dim[0], dim_params.input_dim[1], dim_params.input_dim[2])
            b_size = (dim_params.input_dim[0], dim_params.input_dim[2], dim_params.input_dim[3])
            a_matrix = (
                bmm_params.distribution(torch.zeros(a_size))
                .to(bmm_params.dtype)
                .to(bmm_params.device)
            )
            b_matrix = (
                bmm_params.distribution(torch.zeros(b_size))
                .to(bmm_params.dtype)
                .to(bmm_params.device)
            )

            yield torch.bmm, {
                "input": a_matrix,
                "mat2": b_matrix,
            }, bmm_params, dim_params
