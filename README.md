# TorchDet

A static linter and runtime testing platform for assessing deep learning non-determinism, randomness and reproducibility with applications to scientific computing and simulation within the PyTorch framework.

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

## Usage

Run the scan command line tool:

```text
torchdet scan examples/pytorch_basic.py
```

Run the test command line tool:

```text
torchdet test convolve2D
```

Use the `--help` option with each subcommand to see more information about that command.
