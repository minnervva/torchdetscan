#!/usr/bin/env python3
""" MINNERVVA

Linter for finding non-deterministic functions in pytorch code.

"""
import argparse
from pathlib import Path

DESCRIPTION = \
"""
MINNERVA is a linter for finding non-deterministic functions in pytorch code.
"""

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