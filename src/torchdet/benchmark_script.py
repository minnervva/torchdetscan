import argparse
import pandas as pd
import torch

from .kernels.benchmark import *
from .kernels.avg_pool import *
from .kernels.conv import *
from .kernels.scatter import *
from .kernels.utils import *

def benchmark_avg_pool(niterations):
    gpu = torch.device("cuda")
    avg_pool_params = AvgPoolLoop(
        kernel_size=[(3, 3, 3), (5, 5, 5), (7, 7, 7)],
        stride=[1, 3, 5],
        padding=[0, 1],
        ceil_mode=[True, False],
        count_include_pad=[True, False],
        device=[gpu],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    avg_pool_dims = BatchDimLoop(batch_size=[1, 3], dim=[(1, 16, 16, 16), (3, 64, 64, 64)])
    nn_benchmark(avg_pool_params, avg_pool_dims, avg_pool_loop, torch.nn.AvgPool3d, niterations)

def benchmark_conv1d(niterations):
    gpu = torch.device("cuda")
    conv_params = ConvLoop(
        in_channels=[3, 7],
        out_channels=[3, 7],
        kernel_size=[(3,), (5,), (9,)],
        stride=[1, 3, 5],
        padding=[0, 1],
        dilation=[1, 2, 4],
        groups=[1, 3],
        device=[gpu],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    data_loop = BatchDimLoop(batch_size=[1, 3], dim=[(100,), (1000,), (10000,)])
    nn_benchmark(conv_params, data_loop, convolution_loop, torch.nn.Conv1d, niterations)
    nn_benchmark(conv_params, data_loop, convolution_loop, torch.nn.ConvTranspose1d, niterations)

# Mapping function names to their benchmark functions
benchmark_map = {
    'AvgPool3d': benchmark_avg_pool,
    'Conv1d': benchmark_conv1d,
    # Add additional function mappings here...
}

def prompt_user_selection(functions):
    print("The scan found the following non-deterministic functions, would you like to benchmark any?")
    for idx, func in enumerate(functions, start=1):
        print(f"{idx}. {func}")
    
    selected_indices = input("Enter the numbers of the functions to benchmark (comma-separated): ")
    selected_funcs = [functions[int(i) - 1] for i in selected_indices.split(",") if i.isdigit()]
    return selected_funcs
