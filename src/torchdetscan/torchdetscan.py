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

from src.torchdetscan.findfuncs import FindNondeterministicFunctions, \
    FindNondeterministicFunctionsTable, deterministic_registry

console = Console()

DESCRIPTION = """
MINNERVA is a linter for finding non-deterministic functions in pytorch code.
"""

def lint_file(path: Path, verbose: bool = False, csv_output: bool = False):
    """Lint a single file.

    :param path: The path to the file to lint.
    :param verbose: Whether to enable chatty output.
    :param csv_output: True if we want CSV output instead of a table
    :returns: None
    """
    with open(path, 'r') as file:
        source = file.read()

    tree = ast.parse(source)

    table = Table(title=str(path.absolute()))

    visitor = FindNondeterministicFunctionsTable(table, verbose)
    visitor.visit(tree)

    if len(visitor.table.rows) == 0:
        console.print(
            f':white_check_mark: {path}: No non-deterministic functions '
            f'found\n')
    else:
        console.print(visitor.table)
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

    always_nondeterministic = nondeterministic_registry[ptv]
    conditionally_nondeterministic = deterministic_registry[ptv]

    if args.path.is_file():
        lint_file(args.path, args.verbose, args.csv)
    elif args.path.is_dir():
        if args.verbose:
            console.print(
                f'[cyan]Linting directory: {args.path.absolute()!s}[/cyan]\n')
        for file in args.path.rglob('*.py'):
            lint_file(file, args.verbose, args.csv)
    else:
        console.print(f':X: [red]Path does not exist: {args.path}[/red]')


if __name__ == '__main__':
    main()
