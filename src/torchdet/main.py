"""Main driver for torchdet command line interface."""

import argparse
from pathlib import Path
from rich.console import Console

from .file_linter import lint_file
from .find_functions import deterministic_registry
from .benchmark_script import benchmark_map, prompt_user_selection

import pandas as pd

def run_scan(args: argparse.Namespace):
    """Run the scan functionality to lint a file.

    This is invoked from the CLI using `torchdet scan`.

    Parameters
    ----------
    args
        Namespace arguments from argparse.
    """
    console = Console()
    ptv = args.pytorch_version

    if args.verbose:
        print(f"Checking against Pytorch version {ptv}")

    if args.path.is_file():
        lint_file(args.path, ptv, args.verbose, args.csv)
    elif args.path.is_dir():
        if args.verbose:
            console.print(f"[cyan]Linting directory: {args.path.absolute()!s}[/cyan]\n")
        for file in args.path.rglob("*.py"):
            lint_file(file, ptv, args.verbose, args.csv)
    else:
        console.print(f":X: [red]Path does not exist: {args.path}[/red]")

def run_test(args: argparse.Namespace):
    """Run the test functionality.

    This is invoked from the CLI using `torchdet test`.

    Parameters
    ----------
    args
        Namespace arguments from argparse.
    """
    function: str = args.function
    iterations: int = args.iterations

    if function[0].endswith(".csv"):
        # The input argument is a path to a CSV file
        # The CSV file contains a list of function names
        csv = function[0]
        print(f"Path to CSV file is {csv}")
        df = pd.read_csv(csv)
        # We explicitly look for the 'function column'. Any csv with this column will work
        functions = df['function'].tolist()
        # Prompt the user for selection
        selected_functions = prompt_user_selection(functions)
            
    else:
        # The input argument is the name of a function
        selected_functions = function
        
    # Benchmark the selected functions
    for func in selected_functions:
        if func in benchmark_map:
            print(f"Benchmarking {func}...")
            benchmark_map[func](iterations)
        else:
            print(f"Warning: Function '{func}' is not recognized.")


def main():
    """Command line interface (CLI) for torchdet."""
    parser = argparse.ArgumentParser(
        description="Find non-deterministic functions in your PyTorch code"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="enable chatty output")

    subparsers = parser.add_subparsers(required=True, title="subcommands", help="valid subcommands")

    # Create parser and arguments for the `torchdet scan` subcommand
    parser_scan = subparsers.add_parser("scan", help="run the linter")
    parser_scan.set_defaults(func=run_scan)

    parser_scan.add_argument("--csv", "-c", action="store_true", help="output in csv file format")

    parser_scan.add_argument(
        "--pytorch-version",
        "-ptv",
        default="2.3",
        choices=deterministic_registry.keys(),
        help="Version of PyTorch to use for checking",
    )

    parser_scan.add_argument(
        "path",
        type=Path,
        help="Path to the file or directory in which to recursively lint",
    )

    # Create parser and arguments for the `torchdet test` subcommand
    parser_test = subparsers.add_parser("test", help="run the testing tool")
    parser_test.set_defaults(func=run_test)

    parser_test.add_argument("function", type=str, nargs='+', help="function name(s) or path to csv file")
    parser.add_argument('--iterations', type=int, default=100, help="Number of iterations for benchmarking")
    
    # Get arguments and run appropriate subcommand function
    args = parser.parse_args()
    args.func(args)
