#!/usr/bin/env python3
""" MINNERVVA

Linter for finding non-deterministic functions in pytorch code.

usage: torchdetscan.py [-h] [--verbose VERBOSE] path

`path` can be a file or a directory

TODO need to add support for __get_item__
"""
import argparse
import ast
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.torchdetscan.findfuncs import (FindNondeterministicFunctionsCSV, \
                                        FindNondeterministicFunctionsTable,
                                        deterministic_registry,
                                        nondeterministic_registry)

console = Console()

DESCRIPTION = """
MINNERVA is a linter for finding non-deterministic functions in pytorch code.
"""

def lint_file(path: Path, pytorch_version: str = '2.3',
              verbose: bool = False, csv_output: bool = False):
    """Lint a single file.

    :param path: The path to the file to lint.
    :param pytorch_version: The version of Pytorch to check against.
    :param verbose: Whether to enable chatty output.
    :param csv_output: True if we want CSV output instead of a table
    :returns: None
    """
    with open(path, 'r') as file:
        source = file.read()

    tree = ast.parse(source)

    if csv_output:
        finder = FindNondeterministicFunctionsCSV(path, pytorch_version, verbose)
    else:
        table = Table(title=str(path.absolute()))
        finder = FindNondeterministicFunctionsTable(pytorch_version,
                                                     table, verbose)
    finder.visit(tree)

    if finder.count == 0:
        console.print(
            f':white_check_mark: {path}: No non-deterministic functions '
            f'found\n')
    else:
        if not csv_output:
            console.print(finder.table)
            console.print('\n')


def main():
    global always_nondeterministic, conditionally_nondeterministic
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable chatty output')
    parser.add_argument('--csv', '-c', action='store_true',
                        help='Output in CSV format')
    parser.add_argument('--pytorch-version', '-ptv', default='2.3',
                        choices=deterministic_registry.keys(),
                        help='Version of Pytorch to use for checking')
    parser.add_argument('path', type=Path,
                        help='Path to the file or directory in which to '
                             'recursively lint')

    args = parser.parse_args()

    ptv = args.pytorch_version


    if args.verbose:
        print(f"Checking against Pytorch version {ptv}")



    if args.path.is_file():
        lint_file(args.path, ptv, args.verbose, args.csv)
    elif args.path.is_dir():
        if args.verbose:
            console.print(
                f'[cyan]Linting directory: {args.path.absolute()!s}[/cyan]\n')
        for file in args.path.rglob('*.py'):
            lint_file(file, ptv, args.verbose, args.csv)
    else:
        console.print(f':X: [red]Path does not exist: {args.path}[/red]')


if __name__ == '__main__':
    main()
