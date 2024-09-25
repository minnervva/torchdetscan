import argparse
import pandas as pd
import torch

from kernels.benchmark import *
from kernels.avg_pool import *
from kernels.conv import *
from kernels.scatter import *
from kernels.utils import *

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
    
def benchmark_conv2d(niterations):
    gpu = torch.device("cuda")
    conv_params = ConvLoop(
        in_channels=[3, 7],
        out_channels=[3, 7],
        kernel_size=[(3, 3), (5, 5), (9, 9)],
        stride=[1, 3, 5],
        padding=[0, 1],
        dilation=[1, 2, 4],
        groups=[1, 3],
        device=[gpu],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    data_loop = BatchDimLoop(batch_size=[1, 3], dim=[(100, 100)])
    nn_benchmark(conv_params, data_loop, convolution_loop, torch.nn.Conv2d, niterations)
    nn_benchmark(conv_params, data_loop, convolution_loop, torch.nn.ConvTranspose2d, niterations)


def benchmark_scatter(niterations):
    gpu = torch.device("cuda")
    scatter_params = ScatterLoop(
        dim=[0],
        reduce=["add", "multiply"],
        device=[gpu],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    scatter_dims = ScatterDimLoop(
        input_dim=[(100,), (500,), (1_000,), (10_000,), (100, 100), (500, 500), (1_000, 1_000)],
        reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    func_benchmark(scatter_params, scatter_dims, scatter_loop, "scatter", niterations)


def benchmark_scatter_reduce(niterations):
    gpu = torch.device("cuda")
    scatter_params = ScatterReduceLoop(
        dim=[0],
        reduce=["sum", "mean"],
        include_self=[True, False],
        device=[gpu],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    scatter_dims = ScatterReduceDimLoop(
        input_dim=[(100,), (500,), (1_000,), (10_000,), (100, 100), (500, 500), (1_000, 1_000)],
        reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    func_benchmark(scatter_params, scatter_dims, scatter_reduce_loop, "scatter_reduce", niterations)


def benchmark_gather(niterations):
    gpu = torch.device("cuda")
    gather_params = GatherLoop(
        dim=[0],
        device=[gpu],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    gather_dims = GatherDimLoop(
        input_dim=[(100,), (1_000,)],
        reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    func_benchmark(gather_params, gather_dims, gather_loop, "gather", niterations)


def benchmark_index_add(niterations):
    gpu = torch.device("cuda")
    index_add_params = IndexAddLoop(
        dim=[0],
        device=[gpu],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    index_add_dims = IndexAddDimLoop(input_dim=[(100, 100), (1_000, 1_000)], reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    func_benchmark(index_add_params, index_add_dims, index_add_loop, "index_add", niterations)


def benchmark_index_copy(niterations):
    gpu = torch.device("cuda")
    index_copy_params = IndexAddLoop(
        dim=[0],
        device=[gpu],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    index_copy_dims = IndexAddDimLoop(input_dim=[(100, 100), (1_000, 1_000)], reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    func_benchmark(index_copy_params, index_copy_dims, index_copy_loop, "index_copy", niterations)


def benchmark_index_put(niterations):
    gpu = torch.device("cuda")
    index_put_params = IndexPutLoop(
        accumulate=[True, False],
        device=[gpu],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    index_put_dims = IndexPutDimLoop(input_dim=[(100, 100), (1_000, 1_000)], reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    func_benchmark(index_put_params, index_put_dims, index_put_loop, "index_put", niterations)


# Mapping function names to their benchmark functions
benchmark_map = {
    'AvgPool3d': benchmark_avg_pool,
    'Conv1d': benchmark_conv1d,
    'Conv2d': benchmark_conv2d,
    'Scatter': benchmark_scatter,
    'ScatterReduce': benchmark_scatter_reduce,
    'Gather': benchmark_gather,
    'IndexAdd': benchmark_index_add,
    'IndexCopy': benchmark_index_copy,
    'IndexPut': benchmark_index_put,
    # Add additional function mappings here...
}

def prompt_user_selection(functions):
    print("The scan found the following non-deterministic functions, would you like to benchmark any?")
    for idx, func in enumerate(functions, start=1):
        print(f"{idx}. {func}")
    
    selected_indices = input("Enter the numbers of the functions to benchmark (comma-separated): ")
    selected_funcs = [functions[int(i) - 1] for i in selected_indices.split(",") if i.isdigit()]
    return selected_funcs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark specified functions.")
    parser.add_argument('csv', help="Path to CSV file containing functions to benchmark")
    parser.add_argument('--iterations', type=int, default=100, help="Number of iterations for benchmarking")
    
    args = parser.parse_args()
    
    # Assuming the functions are read from the CSV file
    df = pd.read_csv(args.csv)
    functions = df['function'].tolist()

    # Prompt the user for selection
    selected_functions = prompt_user_selection(functions)

    # Benchmark the selected functions
    for func in selected_functions:
        if func in benchmark_map:
            print(f"Benchmarking {func}...")
            benchmark_map[func](args.iterations)
        else:
            print(f"Warning: Function '{func}' is not recognized.")
