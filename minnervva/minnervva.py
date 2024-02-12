#!/usr/bin/env python3
""" MINNERVVA

Linter for finding non-deterministic functions in pytorch code.

"""
import argparse
from pathlib import Path
import ast

DESCRIPTION = \
"""
MINNERVA is a linter for finding non-deterministic functions in pytorch code.
"""

def find_function_calls(node):
    """ Recusively find all function calls in a node.

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


def lint_file(path: Path):
    """Lint a single file.

    :param path: The path to the file to lint.
    :returns: None
    """

    with open(path, 'r') as file:
        source = file.read()

    tree = ast.parse(source)

    function_calls = find_function_calls(tree)
    for call in function_calls:
        print(call.func.id)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--verbose', '-v',
                        help='Enable chatty output')
    parser.add_argument('path', type=Path,
                        help='Path to the file or directory to lint')

    args = parser.parse_args()

    if args.path.is_file():
        print(f'Linting file: {args.path}')
    elif args.path.is_dir():
        print(f'Linting directory: {args.path}')
    else:
        print(f'Path does not exist: {args.path}')

    print('Done')

if __name__ == '__main__':
    main()