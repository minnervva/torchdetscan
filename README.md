
# Documentation for `torchdet` CLI Tool

`torchdet` is a command-line interface (CLI) tool for identifying non-deterministic operations in PyTorch code. It supports two primary functionalities: scanning for non-deterministic code patterns and testing PyTorch functions for performance benchmarking.

---

## Table of Contents
- [Overview](#overview)
- [CLI Subcommands](#cli-subcommands)
  - [torchdet scan](#torchdet-scan)
  - [torchdet test](#torchdet-test)
- [Benchmarking Functions](#benchmarking-functions)
- [Helper Classes](#helper-classes)
  - [Params and HyperParams](#params-and-hyperparams)
  - [LoopParams and HyperParamLoop](#loopparams-and-hyperparamloop)
- [Example Function: Scatter](#example-function-scatter)
- [Initialization Helpers](#initialization-helpers)
- [Usage Examples](#usage-examples)

---

## Overview

`torchdet` helps users analyze and test PyTorch code by:

- Scanning code to identify functions that may produce non-deterministic behavior.
- Testing functions by benchmarking their performance, either directly or by reading a list of functions from a CSV file.

This tool consists of two primary subcommands:
1. `torchdet scan`: Scans a directory or file to detect non-deterministic PyTorch operations.
2. `torchdet test`: Benchmarks selected PyTorch functions.

---

## Installation

From your terminal, clone this repository and move into the project directory as shown here:

```text
git clone https://github.com/minnervva/torchdetscan.git
cd torchdetscan
```

Install Python from https://www.python.org or by using various other methods. After installing Python, create and activate a virtual environment within the project directory as follows:

```text
python -m venv .venv
source .venv/bin/activate
```

Install the torchdet package and its dependencies into the virtual environment:

```text
pip install .
```

Check installation by running the `torchdet --help` command. This should output the following text in your terminal:

```text
usage: torchdet [-h] [--verbose] {scan,test} ...

Find non-deterministic functions in your PyTorch code

options:
  -h, --help     show this help message and exit
  --verbose, -v  enable chatty output

subcommands:
  {scan,test}    valid subcommands
    scan         run the linter
    test         run the testing tool
```

---

## CLI Subcommands

### torchdet scan

The `torchdet scan` subcommand scans the specified file or directory for non-deterministic PyTorch operations based on a given PyTorch version. This can help identify functions that may introduce randomness into the computation, which is particularly useful for ensuring reproducibility in machine learning experiments.

#### Arguments:
- **`path`** (required): The file or directory to recursively scan for non-deterministic functions.
- **`--pytorch-version`** (`-ptv`, optional): The version of PyTorch used for checking. Defaults to `2.3`.
- **`--csv`** (optional): If provided, the output will be saved in a CSV format.

#### Usage Example:
```bash
torchdet scan --pytorch-version 2.3 ./my_project
```

---

### torchdet test

The `torchdet test` subcommand benchmarks the performance of selected PyTorch functions by running them multiple times and saving the results in a specified output directory. You can pass either a list of functions directly or a CSV file containing the function names.

#### Arguments:
- **`function`**: A list of function names or a CSV file containing a column named `function` with function names.
- **`--iterations`**: The number of times to run each function for benchmarking. Defaults to `100`.
- **`--valid`**: Displays a list of valid function names from `benchmark_map` without running the test.
- **`--select`**: Allows the user to select specific functions by index from the CSV file.
- **`--outdir`**: The output directory where the benchmarking results will be saved. Defaults to `./data`.

#### Usage Example:
```bash
torchdet test Scatter --iterations 50 --outdir results/
```

---

## Benchmarking Functions

The benchmarking process is defined in the `run_test` function. It either accepts a list of functions directly or reads them from a CSV file. The following logic is used:

- If the `--valid` flag is provided, the tool lists all valid functions and exits.
- If the input to the `function` argument is a CSV file, the tool loads the CSV, selects the specified functions, and runs the benchmarking.
- Otherwise, the function is benchmarked by iterating the specified number of times.

Results are saved in the output directory as pickle files (`.pkl`), where each file represents a function and contains a pandas DataFrame.

---

## Helper Classes

### Params and HyperParams

- **`Params`**: The base class for holding and converting parameters into a dictionary using the `asdict()` method.
- **`HyperParams`**: Inherits from `Params` and holds specific hyperparameters required for benchmarking. This includes the device (`torch.device`), data type (`torch.dtype`), and the distribution used for generating data.

#### Example:
```python
@dataclass
class HyperParams(Params):
    device: torch.device
    dtype: torch.dtype
    distribution: Callable
```

### LoopParams and HyperParamLoop

- **`LoopParams`**: Defines a structure for iterating over combinations of hyperparameters using permutations of field values.
- **`HyperParamLoop`**: Inherits from `LoopParams` and holds lists of devices, data types, and distributions, allowing for iteration over various combinations during benchmarking.

#### Example:
```python
@dataclass
class HyperParamLoop(LoopParams):
    device: List[torch.device]
    dtype: List[torch.dtype]
    distribution: List[Callable]
```

These classes are used to generate combinations of parameters for running tests across different configurations.

---

## Example Function: Scatter

The `kernels/scatter.py` file demonstrates how benchmarking is performed for the `scatter` function.

### Main Components:
- **`ScatterHyperParams`**: Defines the hyperparameters for the `scatter` function (e.g., dimension, reduction operation).
- **`ScatterLoop`**: Defines multiple sets of hyperparameters to loop over.
- **`scatter_loop`**: Iterates over combinations of parameters and applies the `scatter` function on randomly generated input tensors.

The `scatter_loop` function runs a scatter operation on PyTorch tensors by generating random data, random indices, and selecting different dimensions for reduction.

#### Example:
```python
for params in scatter_loop:
    scatter_params = ScatterHyperParams(*params)
    # Further processing for benchmarking...
```

---

## Initialization Helpers

The `initialise_weights` function in `kernels/utils.py` initializes weights and biases for certain PyTorch layers (e.g., Conv2d, Conv3d). It applies a given distribution function to both weights and biases.

#### Example:
```python
def initialise_weights(module: torch.nn.Module, weight_dist: Callable):
    if module.__class__.__name__ in weights_and_biases:
        weight_dist(module.weight)
        if module.bias is not None:
            weight_dist(module.bias)
```

---

## Usage Examples

### Example 1: Scanning for Non-Deterministic Code
```bash
torchdet scan --pytorch-version 2.3 ./my_project
```
This command scans `./my_project` for non-deterministic functions using PyTorch version 2.3.

### Example 2: Benchmarking Functions
```bash
torchdet test Scatter --iterations 50 --outdir ./results
```
This command benchmarks the `Scatter` function for 50 iterations and stores the results in the `./results` directory.

### Example 3: Testing with CSV Input
```bash
torchdet test functions.csv --iterations 100 --select 1,3
```
This command reads functions from `functions.csv`, selects the first and third functions for benchmarking, and runs 100 iterations for each.

---

This documentation should provide a clear understanding of how the `torchdet` tool is structured and how its functions interact for testing and benchmarking purposes in PyTorch workflows.


## Adding a New Benchmark

To add a new benchmark function to `torchdet`, follow these steps. We'll use the `scatter` function as an example. Please refer to the PyTorch ```torch.scatter``` [documentation](https://pytorch.org/docs/stable/generated/torch.scatter.html) to understand the operation and hyperparameters in more detail.

### Step 1: Define Benchmark Logic
In `benchmarks.py`, define the logic for the benchmark function using the provided utilities from the `kernels` module.

Example: `benchmark_scatter` function
```python
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
```

### Step 2: Add the Function to `benchmark_map`
Next, add the new benchmark function to the `benchmark_map` dictionary in `benchmarks.py`. This allows the function to be selected and executed during the benchmarking process.

Example:
```python
benchmark_map = {
    "Scatter": benchmark_scatter,
    # Add additional function mappings here...
}
```

### Step 3: Testing the New Benchmark
To run the new benchmark, use the following command:
```bash
torchdet test Scatter --iterations 50 --outdir results/
```
This will execute the `Scatter` benchmark function for 50 iterations and save the results in the `results/` directory.
