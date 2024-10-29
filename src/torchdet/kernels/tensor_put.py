import torch
from dataclasses import dataclass
from torchdet.kernels.utils import HyperParamLoop, HyperParams, LoopParams, Params
from typing import List, Tuple

@dataclass
class TensorPutHyperParams(HyperParams):
    accumulate: bool
@dataclass
class TensorPutLoop(HyperParamLoop):
    accumulate: List[int]
@dataclass
class TensorPutDim(Params):
    input_dim: Tuple
    index_list_size: int
@dataclass
class TensorPutDimLoop(LoopParams):
    input_dim: List[Tuple]
    index_list_size: List[int]
    # reduction_ratio: List[float]


def tensor_put_loop(func_name:str, tensor_put_loop, data_loop):
    """ This is a generator function called by benchmark.func_benchmark
    to generate random parameters for the tensor.put_ kernel.

    :param func_name: function name; unused since we know it's tensor.put_
    :param tensor_put_loop: Contains the ranges for the hyperparameters of the tensor.put_ kernel.
    :param data_loop:
    :yields: A tuple containing the histc kernel, the parameters for the kernel, the hyperparameters, and the dimension parameters.
    """
    for params in tensor_put_loop:
        tensor_put_params = TensorPutHyperParams(*params)

        for d_params in data_loop:
            dim_params = TensorPutDim(*d_params)
            input = (
                tensor_put_params.distribution(torch.zeros(dim_params.input_dim))
                .to(tensor_put_params.dtype)
                .to(tensor_put_params.device)
            )

            index_list = (torch.randperm(torch.numel(input), device=tensor_put_params.device)[:dim_params.index_list_size]
                )

            value_list = (
                tensor_put_params.distribution(torch.zeros(dim_params.index_list_size))
                .to(tensor_put_params.dtype)
                .to(tensor_put_params.device)
            )
            yield getattr(input, func_name), {
                "index":index_list,
                "source":value_list,
                "accumulate":tensor_put_params.accumulate
            }, tensor_put_params, dim_params
