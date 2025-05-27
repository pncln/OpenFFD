#!/usr/bin/env python3
"""
Benchmarking script for OpenFFD.

This script allows benchmarking the parallel processing capabilities
of the OpenFFD library with various configurations.
"""

import sys
from openffd.utils.benchmark import parse_arguments, run_benchmark

if __name__ == "__main__":
    args = parse_arguments()
    run_benchmark(args)
