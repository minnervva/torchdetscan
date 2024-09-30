import torch
from dataclasses import dataclass
from torchdet.kernels.utils import HyperParamLoop, HyperParams, LoopParams, Params
from typing import List


@dataclass
class NLLLossHyperParams(HyperParams):
    weight: torch.tensor
    ignore_index: int
    reduction: str


@dataclass
class NLLLossHyperParamLoop(HyperParamLoop):
    weight: List[torch.tensor]
    ignore_index: List[int]
    reduction: List[str]


@dataclass
class NLLLossDim(Params):
    batch_size: int
    classes: int


@dataclass
class NLLLossDimLoop(LoopParams):
    batch_size: List[int]
    classes: List[int]


def nll_loss_loop(hyper_param_loop, data_loop):
    for params in hyper_param_loop:
        nll_loss_params = NLLLossHyperParams(*params)

        for d_params in data_loop:
            dim_params = NLLLossDim(*d_params)

            loss_fn = torch.nn.NLLLoss(
                nll_loss_params.weight,
                nll_loss_params.ignore_index,
                reduction=nll_loss_params.reduction,
            )
            N = loss_fn.batch_size
            C = loss_fn.classes
            data = torch.randn(N)
            target = torch.empty(N).random_(0, C)

            yield loss_fn, {
                "input": data,
                "target": target,
            }, nll_loss_params, dim_params
