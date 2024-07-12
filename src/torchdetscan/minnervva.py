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

console = Console()

DESCRIPTION = """
MINNERVA is a linter for finding non-deterministic functions in pytorch code.
"""

# Entries in this dicionary, keyed on Pytorch version, contains
# operations that ARE deterministic iff torch.use_deterministic_algorithms(True)
# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
deterministic_registry = {}

# Entries in this dictionary, keyed on Pytorch version, contains 
# normally-noneterministic operations that will throw a RuntimeError 
# when torch.use_deterministic_algorithms(True)
# Some may have other conditions for triggering the RuntimeError
nondeterministic_registry = {}

#
# 1.6.0
#
# From https://pytorch.org/docs/1.6.0/_modules/torch.html
# #use_deterministic_algorithms
#  This feature is experimental and not complete. The above docstring 
#     represents what the future behavior is intended to be. Right now,
#    `_set_deterministic` will only affect `torch.bmm` and convolution
#    operators.
# TODO: the following entries are from 1.7.0, check validity for 1.6.0

deterministic_registry["1.6.0"] = {"Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "bmm", }
nondeterministic_registry["1.6.0"] = {"AvgPool3d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "MaxPool3d", "AdaptiveMaxPool2d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "interpolate",
    "ReflectionPad1d", "ReflectionPad2d", "ReplicationPad1d",
    "ReplicationPad2d", "ReplicationPad3d", "NLLLoss", "CTCLoss",
    "EmbeddingBag", "scatter_add_", "index_add_", "index_select",
    "repeat_interleave", "histc", "bincount", }

#
# 1.7.0
# From: https://pytorch.org/docs/1.7.0/generated/torch.set_deterministic.html
#
deterministic_registry["1.7.0"] = {"Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "bmm", }
nondeterministic_registry["1.7.0"] = {"AvgPool3d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "MaxPool3d", "AdaptiveMaxPool2d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "interpolate",
    "ReflectionPad1d", "ReflectionPad2d", "ReplicationPad1d",
    "ReplicationPad2d", "ReplicationPad3d", "NLLLoss", "CTCLoss",
    "EmbeddingBag", "scatter_add_", "index_add_", "index_select",
    "repeat_interleave", "histc", "bincount", }
#
# 1.7.1
# From: https://pytorch.org/docs/1.7.1/generated/torch.set_deterministic.html
#
deterministic_registry["1.7.1"] = {"Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "bmm", }
nondeterministic_registry["1.7.1"] = {"AvgPool3d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "MaxPool3d", "AdaptiveMaxPool2d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "interpolate",
    "ReflectionPad1d", "ReflectionPad2d", "ReplicationPad1d",
    "ReplicationPad2d", "ReplicationPad3d", "NLLLoss", "CTCLoss",
    "EmbeddingBag", "scatter_add_", "index_add_", "index_select",
    "repeat_interleave", "histc", "bincount", }
#
# 1.8.0
# From: https://pytorch.org/docs/1.8.0/generated/torch.use_deterministic_algorithms.html
#
deterministic_registry["1.8.0"] = {"Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "bmm",
    "index_put", }
nondeterministic_registry["1.8.0"] = {"AvgPool3d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "MaxPool3d", "AdaptiveMaxPool2d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "interpolate",
    "ReflectionPad1d", "ReflectionPad2d", "ReplicationPad1d",
    "ReplicationPad2d", "ReplicationPad3d", "NLLLoss", "CTCLoss",
    "EmbeddingBag", "scatter_add_", "index_add_", "index_copy", "index_select",
    "repeat_interleave", "histc", "bincount", "kthvalue", "median", }
#
# 1.8.1
# From: https://pytorch.org/docs/1.8.1/generated/torch.use_deterministic_algorithms.html
#
deterministic_registry["1.8.1"] = {"Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "bmm",
    "index_put", }
nondeterministic_registry["1.8.1"] = {"AvgPool3d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "MaxPool3d", "AdaptiveMaxPool2d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "interpolate",
    "ReflectionPad1d", "ReflectionPad2d", "ReplicationPad1d",
    "ReplicationPad2d", "ReplicationPad3d", "NLLLoss", "CTCLoss",
    "EmbeddingBag", "scatter_add_", "index_add_", "index_copy", "index_select",
    "repeat_interleave", "histc", "bincount", "kthvalue", "median", }
#
# 1.9.0
# From: https://pytorch.org/docs/1.9.0/generated/torch.use_deterministic_algorithms.html
#
deterministic_registry["1.9.0"] = {"Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "bmm", "index_put",
    "index_put", "put_", "gather", "index_add", "index_select",
    "repeat_interleave", "index_copy", }
nondeterministic_registry["1.9.0"] = {"AvgPool3d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "MaxPool3d", "AdaptiveMaxPool2d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "interpolate",
    "ReflectionPad1d", "ReflectionPad2d", "ReplicationPad1d",
    "ReplicationPad2d", "ReplicationPad3d", "NLLLoss", "CTCLoss",
    "EmbeddingBag", "scatter_add_", "put_", "put_", "histc", "bincount",
    "kthvalue", "median", "gather", "grid_sample", }

#
# 1.9.1
# From: https://pytorch.org/docs/1.9.1/generated/torch.use_deterministic_algorithms.html
#
deterministic_registry["1.9.1"] = {"Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "bmm", "index_put",
    "index_put", "put_", "gather", "index_add", "index_select",
    "repeat_interleave", "index_copy", }

nondeterministic_registry["1.9.1"] = {"AvgPool3d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "MaxPool3d", "AdaptiveMaxPool2d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "interpolate",
    "ReflectionPad1d", "ReflectionPad2d", "ReplicationPad1d",
    "ReplicationPad2d", "ReplicationPad3d", "NLLLoss", "CTCLoss",
    "EmbeddingBag", "scatter_add_", "put_", "put_", "histc", "bincount",
    "kthvalue", "median", "gather", "grid_sample", }

#
# 1.10
# From: https://pytorch.org/docs/1.10/generated/torch.use_deterministic_algorithms.html
#
deterministic_registry["1.10"] = {"Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "bmm", "index_put",
    "index_put", "put_", "scatter_add_", "gather", "index_add", "index_select",
    "repeat_interleave", "index_copy", }
nondeterministic_registry["1.10"] = {"AvgPool3d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "MaxPool3d", "AdaptiveMaxPool2d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "interpolate",
    "ReflectionPad1d", "ReflectionPad2d", "ReflectionPad3d", "ReplicationPad1d",
    "ReplicationPad2d", "ReplicationPad3d", "NLLLoss", "CTCLoss",
    "EmbeddingBag", "scatter_add_", "gather", "put_", "put_", "histc",
    "bincount", "kthvalue", "median", "grid_sample", }

#
# 1.11
# From: https://pytorch.org/docs/1.11/generated/torch.use_deterministic_algorithms.html
#
deterministic_registry["1.11"] = {"Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "bmm", "index_put",
    "index_put", "put_", "scatter_add_", "gather", "index_add", "index_select",
    "repeat_interleave", "index_copy", }
nondeterministic_registry["1.11"] = {"AvgPool3d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "MaxPool3d", "AdaptiveMaxPool2d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "interpolate",
    "ReflectionPad1d", "ReflectionPad2d", "ReflectionPad3d", "ReplicationPad1d",
    "ReplicationPad2d", "ReplicationPad3d", "NLLLoss", "CTCLoss",
    "EmbeddingBag", "scatter_add_", "gather", "put_", "put_", "histc",
    "bincount", "kthvalue", "median", "grid_sample", }
#
# 1.12
# From: https://pytorch.org/docs/1.12/generated/torch.use_deterministic_algorithms.html
#
deterministic_registry["1.12"] = {"Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "bmm", "index_put",
    "put_", "scatter_add_", "gather", "index_add", "index_select",
    "repeat_interleave", "index_copy", }
nondeterministic_registry["1.12"] = {"AvgPool3d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "MaxPool3d", "AdaptiveMaxPool2d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "interpolate",
    "ReflectionPad1d", "ReflectionPad2d", "ReflectionPad3d", "ReplicationPad1d",
    "ReplicationPad2d", "ReplicationPad3d", "NLLLoss", "CTCLoss",
    "EmbeddingBag", "scatter_add_", "gather", "put_", "put_", "histc",
    "bincount", "kthvalue", "median", "grid_sample", }
#
# 1.13
# From : https://pytorch.org/docs/1.13/generated/torch.use_deterministic_algorithms.html
#
deterministic_registry["1.13"] = {"Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "bmm", "index_put",
    "put_", "scatter_add_", "gather", "index_add", "index_select",
    "repeat_interleave", "index_copy", }
nondeterministic_registry["1.13"] = {"AvgPool3d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "MaxPool3d", "AdaptiveMaxPool2d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "MaxUnpool1d", "MaxUnpool2d",
    "MaxUnpool3d", "interpolate", "ReflectionPad1d", "ReflectionPad2d",
    "ReflectionPad3d", "ReplicationPad1d", "ReplicationPad2d",
    "ReplicationPad3d", "NLLLoss", "CTCLoss", "EmbeddingBag", "put_", "put_",
    "histc", "bincount", "kthvalue", "median", "grid_sample", "cumsum", }
#
# 2.0
# From: https://pytorch.org/docs/2.0/generated/torch.use_deterministic_algorithms.html
#
deterministic_registry["2.0"] = {"Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "bmm", "index_put",
    "put_", "scatter_add_", "gather", "index_add", "index_select",
    "repeat_interleave", "index_copy", }
nondeterministic_registry["2.0"] = {"AvgPool3d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "MaxPool3d", "AdaptiveMaxPool2d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "MaxUnpool1d", "MaxUnpool2d",
    "MaxUnpool3d", "interpolate", "ReflectionPad1d", "ReflectionPad2d",
    "ReflectionPad3d", "ReplicationPad1d", "ReplicationPad2d",
    "ReplicationPad3d", "NLLLoss", "CTCLoss", "EmbeddingBag", "put_", "put_",
    "histc", "bincount", "kthvalue", "median", "grid_sample", "cumsum", }
#
# 2.1
# From: https://pytorch.org/docs/2.1/generated/torch.use_deterministic_algorithms.html
#
deterministic_registry["2.1"] = {"Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "bmm", "index_put",
    "put_", "scatter_add_", "gather", "index_add", "index_select",
    "repeat_interleave", "index_copy", "scatter", "scatter_reduce", "resize_",
    "empty", "empty_permuted", }
nondeterministic_registry["2.1"] = {"AvgPool3d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "MaxPool3d", "AdaptiveMaxPool2d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "MaxUnpool1d", "MaxUnpool2d",
    "MaxUnpool3d", "interpolate", "ReflectionPad1d", "ReflectionPad2d",
    "ReflectionPad3d", "ReplicationPad1d", "ReplicationPad2d",
    "ReplicationPad3d", "NLLLoss", "CTCLoss", "EmbeddingBag", "put_", "put_",
    "histc", "bincount", "kthvalue", "median", "grid_sample", "cumsum",
    "scatter_reduce", "resize_", }
#
# 2.2
# From: https://pytorch.org/docs/2.2/generated/torch.use_deterministic_algorithms.html
#
deterministic_registry["2.2"] = {"Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "ReplicationPad2d",
    "bmm", "index_put", "put_", "scatter_add_", "gather", "index_add",
    "index_select", "repeat_interleave", "index_copy", "scatter",
    "scatter_reduce", }
nondeterministic_registry["2.2"] = {"AvgPool3d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "MaxPool3d", "AdaptiveMaxPool2d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "MaxUnpool1d", "MaxUnpool2d",
    "MaxUnpool3d", "interpolate", "ReflectionPad1d", "ReflectionPad2d",
    "ReflectionPad3d", "ReplicationPad1d", "ReplicationPad3d", "NLLLoss",
    "CTCLoss", "EmbeddingBag", "put_", "histc", "bincount", "kthvalue",
    "median", "grid_sample", "cumsum", "scatter_reduce", "resize_", }

#
# 2.3
# From : https://pytorch.org/docs/2.3/generated/torch.use_deterministic_algorithms.html
#
deterministic_registry["2.3"] = {"Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "ReplicationPad2d",
    "bmm", "index_put", "put_", "scatter_add_", "gather", "index_add",
    "index_select", "repeat_interleave", "index_copy", "scatter",
    "scatter_reduce", }

nondeterministic_registry["2.3"] = {"AvgPool3d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "MaxPool3d", "AdaptiveMaxPool2d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "MaxUnpool1d", "MaxUnpool2d",
    "MaxUnpool3d", "interpolate", "ReflectionPad1d", "ReflectionPad2d",
    "ReflectionPad3d", "ReplicationPad1d", "ReplicationPad3d", "NLLLoss",
    "CTCLoss", "EmbeddingBag", "put_", "histc", "bincount", "kthvalue",
    "median", "grid_sample", "cumsum", "scatter_reduce", "resize_", }



class FindNondetermnisticFunctions(ast.NodeVisitor):
    """ This Visitor class is used to find non-deterministic functions in
        pytorch code.
    """
    interpolate_nondeterministic_keywords = {'linear', 'bilinear', 'bicubic',
                                             'trilinear'}

    def __init__(self, table, verbose=False):
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
        self.non_deterministic_funcs = (always_nondeterministic |
                                        conditionally_nondeterministic)

    def report_nondetermninism(self,
                               function_name,
                               line,
                               column,
                               argument='',
                               notes=''):
        """ This function is called when a non-deterministic function is found.

            :param line: The line number where the non-deterministic function
            was
                found.
            :param column: The column number where the non-deterministic
            function
                was found.
            :param function_name: The name of the non-deterministic function.
            :param argument: The optional offending argument to the
                non-deterministic function.
            :param notes: Optional ancillary notes
            :returns: None
        """
        self.table.add_row(function_name, str(line), str(column), argument,
                           notes)

    def handle_use_deterministic_algorithms(self, node):
        """ If we are using deterministic algorithms, then we can remove the
            `conditionally_nondeterministic` functions from the set of
            non-deterministic functions.
        """
        if len(node.args) == 1 and isinstance(node.args[0], ast.Constant):

            if node.args[0].value:
                self.table.add_row('use_deterministic_algorithms',
                                   str(node.lineno), str(node.col_offset), '',
                                   f'[red] use deterministic algorithms '
                                   f'turned [yellow]ON[/yellow][/red]')

                # Just remove the set of conditionally non-deterministic
                # functions
                # from the overall set of non-deterministic functions.
                self.non_deterministic_funcs -= conditionally_nondeterministic
            else:
                self.table.add_row('use_deterministic_algorithms',
                                   str(node.lineno), str(node.col_offset), '',
                                   f'[red] use deterministic algorithms '
                                   f'turned [yellow]OFF[/yellow][/red]')

                # Add the set of conditionally non-deterministic functions
                # from the overall set of non-deterministic functions. Even if
                # they're already there, it doesn't hurt to try to add them.
                self.non_deterministic_funcs |= conditionally_nondeterministic

    def handle_interpolate(self, node):
        """ This function is called when the visitor finds an `interpolate`
            function call.
        """
        for kw in node.keywords:
            # Check if there's a forbidden keyword argument
            if kw.arg == 'mode' and isinstance(kw.value, ast.Constant):
                details = (f'because mode={kw.value.value} will be '
                           f'non-deterministic')
                if (kw.value.value in
                        FindNondetermnisticFunctions.interpolate_nondeterministic_keywords):
                    self.report_nondetermninism('interpolate', node.lineno,
                                                node.col_offset, details)

    def handle_put_(self, node):
        """ This function is called when the visitor finds a `put_` function
            call.
        """
        for kw in node.keywords:
            # Check if there's a forbidden keyword argument
            if kw.arg == 'accumulate' and isinstance(kw.value, ast.Constant):
                if kw.value.value:
                    self.report_nondetermninism('put_', node.lineno,
                                                node.col_offset,
                                                'accumulate=True will be '
                                                'non-deterministic if used '
                                                'with a CUDA tensor')
                    break
                else:
                    self.report_nondetermninism('put_', node.lineno,
                                                node.col_offset,
                                                'accumulate=False')
                    break
        else:
            self.report_nondetermninism('put_',
                                        node.lineno,
                                        node.col_offset,
                                        'accumulate will be False by default '
                                        'and therefore non-deterministic')

    def handle_embeddedbag(self, node):
        """ This function is called when the visitor finds an `EmbeddingBag`
            function call.
        """
        for kw in node.keywords:
            # Check if there's a forbidden keyword argument
            if kw.arg == 'mode' and isinstance(kw.value, ast.Constant):
                if kw.value.value == 'max':
                    self.report_nondetermninism('EmbeddingBag',
                                                node.lineno,
                                                node.col_offset,
                                                'mode=max will be '
                                                'non-deterministic if used '
                                                'with a CUDA tensor')
                    break

    def handle_scatter_reduce(self, node):
        """ This function is called when the visitor finds a `scatter_reduce`
            function call.
        """
        for kw in node.keywords:
            # Check if there's a forbidden keyword argument
            if kw.arg == 'reduce' and isinstance(kw.value, ast.Constant):
                if kw.value.value == 'prod':
                    self.report_nondetermninism('scatter_reduce',
                                                node.lineno,
                                                node.col_offset,
                                                f'reduce={kw.value.value} '
                                                f'will be non-deterministic '
                                                f'if used with a CUDA tensor')
                    break

    def visit_Call(self, node):
        # Check if the function being called is non-deterministic
        if (isinstance(node.func, ast.Attribute)):
            if node.func.attr == 'use_deterministic_algorithms':
                self.handle_use_deterministic_algorithms(node)

            if node.func.attr in self.non_deterministic_funcs:
                if node.func.attr == 'interpolate':
                    # Check to see if the keyword arguments are
                    # non-deterministic
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
                                                    node.lineno,
                                                    node.col_offset, )
                    elif hasattr(node.func, 'attr'):
                        self.report_nondetermninism(node.func.attr,
                                                    node.lineno,
                                                    node.col_offset, )
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
    with open(path, 'r') as file:
        source = file.read()

    tree = ast.parse(source)

    table = Table(title=str(path.absolute()))
    table.add_column('Function', justify='left', style='cyan')
    table.add_column('Line', justify='right', style='magenta')
    table.add_column('Column', justify='right', style='green')
    table.add_column('Optional Arguments', justify='left', max_width=25,
                     style='purple')
    table.add_column('Notes', justify='left', max_width=25, style='blue')

    visitor = FindNondetermnisticFunctions(table, verbose)
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
        lint_file(args.path, args.verbose)
    elif args.path.is_dir():
        if args.verbose:
            console.print(
                f'[cyan]Linting directory: {args.path.absolute()!s}[/cyan]\n')
        for file in args.path.rglob('*.py'):
            lint_file(file, args.verbose)
    else:
        console.print(f':X: [red]Path does not exist: {args.path}[/red]')


if __name__ == '__main__':
    main()
