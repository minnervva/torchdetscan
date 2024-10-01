import torch
from dataclasses import dataclass
from torchdet.kernels.utils import HyperParamLoop, HyperParams, LoopParams, Params
from typing import List


@dataclass
class CTCLossParams(HyperParams):
    reduction: str
    zero_infinity: bool


@dataclass
class CTCLossLoop(HyperParamLoop):
    reduction: List[str]
    zero_infinity: List[bool]


@dataclass
class CTCLossDim(Params):
    input_length: int
    batch_size: int
    classes: int


@dataclass
class CTCLossDimLoop(LoopParams):
    input_length: List[int]
    batch_size: List[int]
    classes: List[int]


def ctc_loss_loop(func_name: str, hyper_param_loop, data_loop):
    for params in hyper_param_loop:
        nll_loss_params = CTCLossParams(*params)

        ctc_loss = torch.nn.CTCLoss(
            reduction=nll_loss_params.reduction,
            zero_infinity=nll_loss_params.zero_infinity,
        )
        for d_params in data_loop:
            dim_params = CTCLossDim(*d_params)

            T = dim_params.input_length
            C = dim_params.classes
            N = dim_params.batch_size

            input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
            target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
            target = torch.randint(
                low=1, high=C, size=(sum(target_lengths),), dtype=torch.long
            )

            yield ctc_loss, {
                "log_probs": input,
                "targets": target,
                "input_lengths": input_lengths,
                "target_lengths": target_lengths,
            }, nll_loss_params, dim_params
