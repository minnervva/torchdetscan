#!/usr/bin/env python3
""" MINNERVVA

Linter for finding non-deterministic functions in pytorch code.

usage: minnervva.py [-h] [--verbose VERBOSE] path
"""
import argparse
from pathlib import Path
import ast

DESCRIPTION = \
"""
MINNERVA is a linter for finding non-deterministic functions in pytorch code.
"""

forbidden_functions = set(['AvgPool3D',
                           'AdaptiveAvgPool2d',
                           'AdaptiveAvgPool3d',
                           'MaxPool3d',
                           'AdaptiveMaxPool2d',
                           'FractionalMaxPool2d',
                           'FractionalMaxPool3d',
                           'MaxUnpool1d',
                           'MaxUnpool2d',
                           'MaxUnpool3d',
                           'interpolate',
                           'ReflectionPad1d',
                           'ReflectionPad2d',
                           'ReflectionPad3d',
                           'ReplicationPad1d',
                           'ReplicationPad3d',
                           'NLLLoss',
                           'CTCLoss',
                           'EmbeddingBag',
                           'put_',
                           'histc',
                           'bincount',
                           'kthvalue',
                           'median',
                           'grid_sample',
                           'cumsum',
                           'scatter_reduce',
                           'resize_'])

def find_function_calls(node):
    """ Recursively find all function calls in a node.

        :param node: The node to search for function calls.
        :returns: A list of all function calls in the node.
    """
    function_calls = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Call):
            function_calls.append(child)
        else:
            function_calls.extend(find_function_calls(child))
    return function_calls


def lint_file(path: Path, verbose: bool = False):
    """Lint a single file.

    :param path: The path to the file to lint.
    :param verbose: Whether to enable chatty output.
    :returns: None
    """
    if verbose:
        print(f'Linting file: {path}')

    with open(path, 'r') as file:
        source = file.read()

    tree = ast.parse(source)

    function_calls = find_function_calls(tree)
    for call in function_calls:
        if hasattr(call.func, 'id'):
            print(call.func.id)
        elif hasattr(call.func, 'attr'):
            print(call.func.attr)

def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Enable chatty output')
    parser.add_argument('path', type=Path,
                        help='Path to the file or directory to lint')

    args = parser.parse_args()

    if args.path.is_file():
        lint_file(args.path, args.verbose)
    elif args.path.is_dir():
        if args.verbose:
            print(f'Linting directory: {args.path}')
        for file in args.path.rglob('*.py'):
            lint_file(file, args.verbose)
    else:
        print(f'Path does not exist: {args.path}')

    print('Done')

if __name__ == '__main__':
    main()