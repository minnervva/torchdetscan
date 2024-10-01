"""Functions for running benchmarks."""

import warnings
import torch
from . import kernels as kn

# Suppress scipy warning
warnings.filterwarnings("ignore",
                        message="Precision loss occurred in moment calculation "
                                "due to catastrophic cancellation")

PYTORCH_DEVICE = "cpu"  # use `cpu` or `cuda` for this string


def benchmark_avg_pool(niterations, outdir):
    device = torch.device(PYTORCH_DEVICE)
    avg_pool_params = kn.AvgPoolLoop(
        kernel_size=[(3, 3, 3), (5, 5, 5), (7, 7, 7)],
        stride=[1, 3, 5],
        padding=[0, 1],
        ceil_mode=[True, False],
        count_include_pad=[True, False],
        device=[device],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    avg_pool_dims = kn.BatchDimLoop(batch_size=[1, 3], dim=[(1, 16, 16, 16), (3, 64, 64, 64)])
    kn.nn_benchmark(
        avg_pool_params, avg_pool_dims, kn.avg_pool_loop, torch.nn.AvgPool3d, niterations, outdir
    )


def benchmark_conv1d(niterations, outdir):
    device = torch.device(PYTORCH_DEVICE)
    conv_params = kn.ConvLoop(
        in_channels=[3, 7],
        out_channels=[3, 7],
        kernel_size=[(3,), (5,), (9,)],
        stride=[1, 3, 5],
        padding=[0, 1],
        dilation=[1, 2, 4],
        groups=[1, 3],
        device=[device],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    data_loop = kn.BatchDimLoop(batch_size=[1, 3], dim=[(100,), (1000,), (10000,)])
    kn.nn_benchmark(conv_params, data_loop, kn.convolution_loop, torch.nn.Conv1d, niterations, outdir)
    kn.nn_benchmark(
        conv_params, data_loop, kn.convolution_loop, torch.nn.ConvTranspose1d, niterations, outdir
    )


def benchmark_conv2d(niterations, outdir):
    device = torch.device(PYTORCH_DEVICE)
    conv_params = kn.ConvLoop(
        in_channels=[3, 7],
        out_channels=[3, 7],
        kernel_size=[(3, 3), (5, 5), (9, 9)],
        stride=[1, 3, 5],
        padding=[0, 1],
        dilation=[1, 2, 4],
        groups=[1, 3],
        device=[device],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    data_loop = kn.BatchDimLoop(batch_size=[1, 3], dim=[(100, 100)])
    kn.nn_benchmark(conv_params, data_loop, kn.convolution_loop, torch.nn.Conv2d, niterations, outdir)
    kn.nn_benchmark(
        conv_params, data_loop, kn.convolution_loop, torch.nn.ConvTranspose2d, niterations, outdir
    )


def benchmark_scatter(niterations, outdir):
    device = torch.device(PYTORCH_DEVICE)
    scatter_params = kn.ScatterLoop(
        dim=[0],
        reduce=["add", "multiply"],
        device=[device],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    scatter_dims = kn.ScatterDimLoop(
        input_dim=[(100,), (500,), (1_000,), (10_000,), (100, 100), (500, 500), (1_000, 1_000)],
        reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    kn.func_benchmark(scatter_params, scatter_dims, kn.scatter_loop, "scatter", niterations, outdir)


def benchmark_scatter_reduce(niterations, outdir):
    device = torch.device(PYTORCH_DEVICE)
    scatter_params = kn.ScatterReduceLoop(
        dim=[0],
        reduce=["sum", "mean"],
        include_self=[True, False],
        device=[device],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    scatter_dims = kn.ScatterReduceDimLoop(
        input_dim=[(100,), (500,), (1_000,), (10_000,), (100, 100), (500, 500), (1_000, 1_000)],
        reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    kn.func_benchmark(
        scatter_params, scatter_dims, kn.scatter_reduce_loop, "scatter_reduce", niterations, outdir
    )


def benchmark_gather(niterations, outdir):
    device = torch.device(PYTORCH_DEVICE)
    gather_params = kn.GatherLoop(
        dim=[0],
        device=[device],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    gather_dims = kn.GatherDimLoop(
        input_dim=[(100,), (1_000,)],
        reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    kn.func_benchmark(gather_params, gather_dims, kn.gather_loop, "gather", niterations, outdir)


def benchmark_index_add(niterations, outdir):
    device = torch.device(PYTORCH_DEVICE)
    index_add_params = kn.IndexAddLoop(
        dim=[0],
        device=[device],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    index_add_dims = kn.IndexAddDimLoop(
        input_dim=[(100, 100), (1_000, 1_000)],
        reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    kn.func_benchmark(index_add_params, index_add_dims, kn.index_add_loop, "index_add", niterations, outdir)


def benchmark_index_copy(niterations, outdir):
    device = torch.device(PYTORCH_DEVICE)
    index_copy_params = kn.IndexAddLoop(
        dim=[0],
        device=[device],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    index_copy_dims = kn.IndexAddDimLoop(
        input_dim=[(100, 100), (1_000, 1_000)],
        reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    kn.func_benchmark(
        index_copy_params, index_copy_dims, kn.index_copy_loop, "index_copy", niterations, outdir
    )


def benchmark_index_put(niterations, outdir):
    device = torch.device(PYTORCH_DEVICE)
    index_put_params = kn.IndexPutLoop(
        accumulate=[True, False],
        device=[device],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    index_put_dims = kn.IndexPutDimLoop(
        input_dim=[(100, 100), (1_000, 1_000)],
        reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    kn.func_benchmark(index_put_params, index_put_dims, kn.index_put_loop, "index_put", niterations, outdir)


def benchmark_median(niterations, outdir):
    device = torch.device(PYTORCH_DEVICE)
    median_params = kn.MedianLoop(keepdim=[True, False],
                                  device=[device],
                                  dtype=[torch.float32],
                                  distribution=[torch.nn.init.normal_], )
    median_dims = kn.MedianDimLoop(input_dim=[(100,), (1_000,)],
            # reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    kn.func_benchmark(median_params, median_dims, kn.median_loop, "median",
                      niterations, outdir)


# Mapping function names to their benchmark functions
benchmark_map = {
    "AvgPool3d": benchmark_avg_pool,
    "Conv1d": benchmark_conv1d,
    "Conv2d": benchmark_conv2d,
    "Scatter": benchmark_scatter,
    "ScatterReduce": benchmark_scatter_reduce,
    "Gather": benchmark_gather,
    "IndexAdd": benchmark_index_add,
    "IndexCopy": benchmark_index_copy,
    "IndexPut": benchmark_index_put,
    # Add additional function mappings here...
    "Median" : benchmark_median
}
