"""Main driver for torchdet command line interface."""

import argparse
import pandas as pd
from pathlib import Path
from rich.console import Console

from .benchmarks import benchmark_map
from .find_functions import deterministic_registry
from .linter import lint_file


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
        print(f"Checking against PyTorch version {ptv}")

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
    function: list[str] = args.function
    iterations: int = args.iterations

    if args.valid:
        print("Valid function names are:")
        for key in benchmark_map:
            print(f"- {key}")
        return
    elif not args.function:
        # If --valid is not provided, and functions are missing, raise an error
        print("the following arguments are required: function")
        return

    if function[0].endswith(".csv"):
        # The input argument is a path to a csv file which contains list of functions
        csv = function[0]
        print(f"Path to the CSV file is {csv}")

        # Look for the 'function column', any csv with this column should work
        df = pd.read_csv(csv)
        functions = df["function"].tolist()

        if args.select:
            # Get user selected functions
            selected_functions = [functions[int(i) - 1] for i in args.select.split(",") if i.isdigit()]
        else:
            selected_functions = functions
    else:
        # The input argument is the name of a function
        selected_functions = function

    # Benchmark the selected functions
    for func in selected_functions:
        if func in benchmark_map:
            print(f"Benchmarking {func}...")
            benchmark_map[func](iterations)
        else:
            print(f"Warning: function '{func}' is not recognized.")
            print(f"Run 'torchdet test {func} --valid' for valid function names.")


def main():
    """Command line interface (CLI) for torchdet."""
    parser = argparse.ArgumentParser(
        description="Find non-deterministic functions in your PyTorch code."
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="enable chatty output")

    subparsers = parser.add_subparsers(required=True, title="subcommands", help="valid subcommands")

    ##
    # Create parser and arguments for the `torchdet scan` subcommand
    ##
    parser_scan = subparsers.add_parser("scan", help="run the linter tool")
    parser_scan.set_defaults(func=run_scan)

    parser_scan.add_argument("--csv", action="store_true", help="output in csv file format")

    parser_scan.add_argument(
        "--pytorch-version",
        "-ptv",
        default="2.3",
        choices=deterministic_registry.keys(),
        help="version of pytorch to use for checking",
    )

    parser_scan.add_argument(
        "path", type=Path,
            help="path to file or directory in which to recursively lint"
    )

    ##
    # Create parser and arguments for the `torchdet test` subcommand
    ##
    parser_test = subparsers.add_parser("test",
                                        help="run the testing tool")
    parser_test.set_defaults(func=run_test)

    parser_test.add_argument(
        "function", type=str,
            nargs="*", # '*' instead of '?' to allow --valid to override need
            help="function name(s) or path to csv file"
    )

    parser_test.add_argument(
        "--iterations", type=int, default=100,
            help="number of iterations for benchmarking"
    )

    parser_test.add_argument(
        "--valid", action="store_true",
            help="show valid function names then abort test"
    )

    parser_test.add_argument("--select", type=str,
                             help="comma-separated list of functions in csv")

    parser_test.add_argument('--outfile', type=str,
                             help='Output for benchmark results. Defaults to '
                                  'a pickle file of a pandas dataframe with the '
                                  'named "<function name>.pkl"')

    # Get arguments and run appropriate subcommand function
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
