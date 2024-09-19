"""Test pytorch functions for non-deterministic vs. deterministic behavior."""

import argparse

# TODO add guts of non-deterministic testing here


def main():
    """Command line interface for torchdettest.

    Run from command line and provide a function name:
    python src/torchdettest/torchdettest.py ConvTranspose1d

    Run from command line and provide path to CSV file:
    python src/torchdettest/torchdettest.py /user/home/myfuncs.csv
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--verbose", type=bool, default=False, help="Print verbose output"
    )

    parser.add_argument(
        "function", type=str, help="Provide function name or path to CSV file"
    )

    args = parser.parse_args()

    function: str = args.function

    if function.endswith(".csv"):
        # The input argument is a path to a CSV file
        # The CSV file contains a list of function names
        print(f"Path to CSV file is {function}")
    else:
        # The input argument is the name of a function
        print(f"Function name is {function}")


if __name__ == "__main__":
    main()
