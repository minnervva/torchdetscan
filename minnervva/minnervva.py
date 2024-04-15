#!/usr/bin/env python3
""" MINNERVVA

Linter for finding non-deterministic functions in pytorch code.

usage: minnervva.py [-h] [--verbose VERBOSE] path

`path` can be a file or a directory
"""
import argparse
from pathlib import Path
import ast

from rich.console import Console
from rich.table import Table

console = Console()


DESCRIPTION = """
MINNERVA is a linter for finding non-deterministic functions in pytorch code.
"""

always_nondeterministic = {'AvgPool3d', 'AdaptiveAvgPool2d',
                           'AdaptiveAvgPool3d', 'MaxPool3d',
                           'AdaptiveMaxPool2d', 'FractionalMaxPool2d',
                           'FractionalMaxPool3d', 'MaxUnpool1d', 'MaxUnpool2d',
                           'MaxUnpool3d', 'interpolate', 'ReflectionPad1d',
                           'ReflectionPad2d', 'ReflectionPad3d',
                           'ReplicationPad1d', 'ReplicationPad3d', 'NLLLoss',
                           'CTCLoss', 'EmbeddingBag', 'put_', 'histc',
                           'bincount', 'kthvalue', 'median', 'grid_sample',
                           'cumsum', 'scatter_reduce', 'resize_'}

# These ARE deterministic iff torch.use_deterministic_algorithms(True)
# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch-use-deterministic-algorithms
conditionally_nondeterministic = {'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d',
                                  'ConvTranspose2d', 'ConvTranspose3d', 'ReplicationPad2d',
                                  'bmm', 'index_put', 'put_', 'scatter_add_', 'gather',
                                  'index_add', 'index_select', 'repeat_interleave',
                                  'index_copy', 'scatter', 'scatter_reduce'}




class FindNondetermnisticFunctions(ast.NodeVisitor):
    """ This Visitor class is used to find non-deterministic functions in
        pytorch code.
    """
    interpolate_nondeterministic_keywords = {'linear', 'bilinear', 'bicubic',
                                             'trilinear'}

    def __init__(self, table, verbose = False):
        """ Initialize the visitor.
            :param table: Rich table for reporting non-determ. funcs
            :param verbose: Whether to enable chatty output.
        """
        super().__init__()

        self.table = table
        self.verbose = verbose

        # Initially we assume that all functions are non-deterministic; we will
        # remove the `conditionally_nondeterministic` from the set of
        # non-deterministic functions iff we encounter a call to
        # torch.use_deterministic_algorithms(True).
        self.non_deterministic_funcs = always_nondeterministic | conditionally_nondeterministic

    def report_nondetermninism(self, function_name, line, column, argument=None):
        """ This function is called when a non-deterministic function is found.

            :param line: The line number where the non-deterministic function was
                found.
            :param column: The column number where the non-deterministic function
                was found.
            :param function_name: The name of the non-deterministic function.
            :param argument: The optional offending argument to the
                non-deterministic function.
            :returns: None
        """
        if argument is None:
            self.table.add_row(function_name, str(line), str(column), '')
            # print(f"Found non-deterministic function {function_name} at "
            #       f"line {line}, column {column}")
        else:
            self.table.add_row(function_name, str(line), str(column), argument)
            # print(f"Found non-deterministic function '{function_name}' with argument "
            #       f"'{argument}' that makes it nondeterministic at "
            #       f"line {line}, column {column}")

    def handle_use_deterministic_algorithms(self):
        """ If we are using deterministic algorithms, then we can remove the
            `conditionally_nondeterministic` functions from the set of
            non-deterministic functions.

            TODO add check for True not False.
        """
        if self.verbose:
            print('Found call to torch.use_deterministic_algorithms(True)')

        # Just remove the set of conditionally non-deterministic functions
        # from the overall set of non-deterministic functions.
        self.non_deterministic_funcs -= conditionally_nondeterministic


    def handle_interpolate(self, node):
        """ This function is called when the visitor finds an `interpolate`
            function call.
        """
        for kw in node.keywords:
            # Check if there's a forbidden keyword argument
            if kw.arg == 'mode' and isinstance(kw.value, ast.Constant):
                if kw.value.value in \
                        FindNondetermnisticFunctions.interpolate_nondeterministic_keywords:
                    self.report_nondetermninism('interpolate',
                                                node.lineno, node.col_offset,
                                            kw.value.value)


    def handle_put_(self, node):
        """ This function is called when the visitor finds a `put_` function
            call.
        """
        for kw in node.keywords:
            # Check if there's a forbidden keyword argument
            if kw.arg == 'accumulate' and isinstance(kw.value, ast.Constant):
                if kw.value.value:
                    self.report_nondetermninism('put_', node.lineno, node.col_offset,
                                                'accumulate=True will be non-deterministic if used with a CUDA tensor')
                    # print(f'Found non-deterministic function put_ at line {node.lineno}, '
                    #       f'column {node.col_offset} with accumulate=True that '
                    #       f'will be non-deterministic if used with a CUDA tensor')
                    break
                else:
                    self.report_nondetermninism('put_',
                                                node.lineno, node.col_offset,
                                                'accumulate=False')
                    print(f'Found non-deterministic function put_ at line {node.lineno}, '
                          f'column {node.col_offset} because accumulate=False')
                    break
        else:
            self.report_nondetermninism('put_', node.lineno, node.col_offset,
                                        'accumulate will be False by default and therefore non-deterministic')
            # print(f'Found non-deterministic function put_ at line {node.lineno}, '
            #       f'column {node.col_offset} because accumulate keyword '
            #       f'argument will be False by default and therefore '
            #       f'non-deterministic.')

    def handle_embeddedbag(self, node):
        """ This function is called when the visitor finds an `EmbeddingBag`
            function call.
        """
        for kw in node.keywords:
            # Check if there's a forbidden keyword argument
            if kw.arg == 'mode' and isinstance(kw.value, ast.Constant):
                if kw.value.value == 'max':
                    self.report_nondetermninism('EmbeddingBag', node.lineno, node.col_offset,
                                                'mode=max will be non-deterministic if used with a CUDA tensor')
                    # print(
                    #     f'Found non-deterministic function EmbeddingBag at line {node.lineno}, '
                    #     f'column {node.col_offset} with mode=max that '
                    #     f'will be non-deterministic if used with a CUDA tensor')
                    break

    def handle_scatter_reduce(self, node):
        """ This function is called when the visitor finds a `scatter_reduce`
            function call.
        """
        for kw in node.keywords:
            # Check if there's a forbidden keyword argument
            if kw.arg == 'reduce' and isinstance(kw.value, ast.Constant):
                if kw.value.value == 'prod':
                    self.report_nondetermninism('scatter_reduce', node.lineno, node.col_offset,
                                                f'reduce={kw.value.value} will be non-deterministic if used with a CUDA tensor')
                    # print(f'Found non-deterministic function scatter_reduce at line {node.lineno}, '
                    #       f'column {node.col_offset} with reduce={kw.value.value} that '
                    #       f'will be non-deterministic if used with a CUDA tensor')
                    break

    def visit_Call(self, node):
        # Check if the function being called is non-deterministic
        if (isinstance(node.func, ast.Attribute)):
            if node.func.attr == 'use_deterministic_algorithms':
                self.handle_use_deterministic_algorithms()

            if node.func.attr in self.non_deterministic_funcs:
                if node.func.attr == 'interpolate':
                    # Check to see if the keyword arguments are non-deterministic
                    self.handle_interpolate(node)
                elif node.func.attr == 'put_':
                    self.handle_put_(node)
                elif node.func.attr == 'EmbeddingBag':
                    self.handle_embeddedbag(node)
                elif node.func.attr == 'scatter_reduce':
                    self.handle_scatter_reduce(node)
                else:
                    if hasattr(node.func, 'id'):
                        self.report_nondetermninism(node.func.id,
                                                    node.lineno, node.col_offset,
                                               )
                    elif hasattr(node.func, 'attr'):
                        self.report_nondetermninism(node.func.attr,
                                                    node.lineno, node.col_offset,
                                               )
                    else:
                        # Welp, dunno how to get the name of the function
                        raise ValueError('Unknown function type')

        # Continue searching the tree
        self.generic_visit(node)


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

    table = Table(title=path.name)
    table.add_column('Function', justify='left', style='cyan')
    table.add_column('Line', justify='right', style='magenta')
    table.add_column('Column', justify='right', style='green')
    table.add_column('Optional Arguments', justify='left', style='purple')

    visitor = FindNondetermnisticFunctions(table, verbose)
    visitor.visit(tree)

    console.print(visitor.table)
    console.print('\n')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable chatty output')
    parser.add_argument('path', type=Path,
                        help='Path to the file or directory to lint')

    args = parser.parse_args()

    if args.path.is_file():
        lint_file(args.path, args.verbose)
    elif args.path.is_dir():
        if args.verbose:
            print(f'Linting directory: {args.path.absolute()}')
        for file in args.path.rglob('*.py'):
            lint_file(file, args.verbose)
    else:
        print(f'Path does not exist: {args.path}')

    print('Done')


if __name__ == '__main__':
    main()
