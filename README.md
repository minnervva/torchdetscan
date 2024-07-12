# torchdetscan: rooting out non-determinism in pytorch code
A linter for deep learning non-determinism, randomness and 
reproducibility with applications to scientific computing and simulation.

## Usage

```
usage: torchdetscan [-h] [--verbose] [--pytorch-version {1.6.0,1.7.0,1.7.1,1.8.0,1.8.1,1.9.0,1.9.1,1.10,1.11,1.12,1.13,2.0,2.1,2.2,2.3}] path

torchdetscan is a linter for finding non-deterministic functions in pytorch code.

positional arguments:
  path                  Path to the file or directory in which to recursively lint

options:
  -h, --help            show this help message and exit
  --verbose, -v         Enable chatty output
  --pytorch-version {1.6.0,1.7.0,1.7.1,1.8.0,1.8.1,1.9.0,1.9.1,1.10,1.11,1.12,1.13,2.0,2.1,2.2,2.3}, -ptv {1.6.0,1.7.0,1.7.1,1.8.0,1.8.1,1.9.0,1.9.1,1.10,1.11,1.12,1.13,2.0,2.1,2.2,2.3}
                        Version of Pytorch to use for checking
```
