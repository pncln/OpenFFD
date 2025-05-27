"""Utility functions for OpenFFD."""

from openffd.utils.parallel import (
    get_optimal_chunk_size,
    chunk_array,
    parallel_process,
    is_parallelizable,
    ParallelConfig
)

__all__ = [
    "get_optimal_chunk_size", 
    "chunk_array", 
    "parallel_process",
    "is_parallelizable",
    "ParallelConfig"
]
