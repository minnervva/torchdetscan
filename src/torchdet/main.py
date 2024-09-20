"""here."""

import argparse
from pathlib import Path
from rich.console import Console

from .file_linter import lint_file
from .find_functions import deterministic_registry


def run_scan(args: argparse.Namespace):
    """here."""
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
    """here."""
    function: str = args.function

    if function.endswith(".csv"):
        # The input argument is a path to a CSV file
        # The CSV file contains a list of function names
        print(f"Path to CSV file is {function}")
    else:
        # The input argument is the name of a function
        print(f"Function name is {function}")


def main():
    """here."""
    parser = argparse.ArgumentParser(
        description="Find non-deterministic functions in your PyTorch code"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable chatty output"
    )

    subparsers = parser.add_subparsers(required=True, help="subparsers help")

    # Create parser and arguments for the `torchdet scan` subcommand
    parser_scan = subparsers.add_parser("scan", help="scan help")

    parser_scan.add_argument(
        "--csv", "-c", action="store_true", help="Output in CSV format"
    )

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

    parser_scan.set_defaults(func=run_scan)

    # Create parser and arguments for the `torchdet test` subcommand
    parser_test = subparsers.add_parser("test", help="test help")

    parser_test.add_argument(
        "function", type=str, help="function name or path to CSV file"
    )

    parser_test.set_defaults(func=run_test)

    # Get arguments and run appropriate subcommand function
    args = parser.parse_args()
    args.func(args)
