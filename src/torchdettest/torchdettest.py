#!/usr/bin/env python3
"""
    For testing pytorch functions for non-deterministic vs. deterministic
    behavior.
"""
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', type=bool, default=False,
                        help='Print verbose output')
    args = parser.parse_args()

    pass