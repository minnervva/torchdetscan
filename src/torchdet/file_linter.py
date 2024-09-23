"""here."""

import ast
from pathlib import Path
from rich.table import Table
from rich.console import Console

from .find_functions import FindNondeterministicFunctionsCSV
from .find_functions import FindNondeterministicFunctionsTable


def lint_file(
    path: Path,
    pytorch_version: str = "2.3",
    verbose: bool = False,
    csv_output: bool = False,
) -> None:
    """Lint a single file.

    Parameters
    ----------
    path
        The path to the file to lint.
    pytorch_version
        The version of Pytorch to check against.
    verbose
        Whether to enable chatty output.
    csv_output
        True if we want CSV output instead of a table.
    """
    with open(path, "r") as file:
        source = file.read()

    tree = ast.parse(source)

    if csv_output:
        finder = FindNondeterministicFunctionsCSV(path, pytorch_version, verbose)
    else:
        table = Table(title=str(path.absolute()))
        finder = FindNondeterministicFunctionsTable(pytorch_version, table, verbose)

    finder.visit(tree)
    console = Console()

    if finder.count == 0:
        msg = f":white_check_mark: {path}: No non-deterministic functions found\n"
        console.print(msg)
    else:
        if not csv_output:
            console.print(finder.table)
            console.print("\n")
