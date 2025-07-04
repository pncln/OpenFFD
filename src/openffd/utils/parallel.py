"""
Parallel processing utilities for OpenFFD.

This module provides functions and classes for parallel processing
of large datasets, such as mesh points and control points.
"""

import logging
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

# Thread-local storage to track nested parallelization
_parallel_context = threading.local()


class ParallelConfig:
    """Configuration for parallel processing.
    
    This class provides configuration options for parallel processing,
    including enabling/disabling, method (thread/process), and worker count.
    """
    
    def __init__(
        self,
        enabled: bool = False,  # Default to False for better performance with small datasets
        method: str = "process",
        max_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        threshold: int = 500000,  # Much higher threshold to avoid parallelization overhead
    ):
        """Initialize parallel processing configuration.
        
        Args:
            enabled: Whether parallel processing is enabled
            method: Method for parallelization ('process' or 'thread')
            max_workers: Maximum number of worker processes/threads
            chunk_size: Size of data chunks for parallel processing
            threshold: Minimum data size to trigger parallelization
        """
        self.enabled = enabled
        self.method = method
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.threshold = threshold


def is_parallelizable(data_size: int, config: Optional[ParallelConfig] = None) -> bool:
    """Determine if a task is parallelizable based on data size and configuration.
    
    Args:
        data_size: Size of the data to process
        config: Parallel processing configuration
        
    Returns:
        True if the task should be parallelized, False otherwise
    """
    if config is None:
        config = ParallelConfig()
    
    # Check if we're already in a parallel context to prevent nested parallelization
    if hasattr(_parallel_context, 'in_parallel') and _parallel_context.in_parallel:
        return False
    
    # Early optimization: don't parallelize very small datasets regardless of config
    if data_size < 100000:  # Much higher minimum threshold to avoid overhead
        return False
    
    # Only parallelize if enabled and data size exceeds threshold
    return config.enabled and data_size >= config.threshold


def get_optimal_chunk_size(total_size: int, target_chunks: int = None) -> int:
    """Calculate optimal chunk size for parallel processing.
    
    This function determines the optimal chunk size based on the total data size
    and the target number of chunks or available CPU cores.
    
    Args:
        total_size: Total size of the data to be processed
        target_chunks: Target number of chunks (defaults to CPU count)
        
    Returns:
        Chunk size to use for parallel processing
    """
    if target_chunks is None:
        # Use a reasonable number of chunks based on CPU count
        cpu_count = mp.cpu_count()
        # Use fewer chunks for smaller datasets to reduce overhead
        if total_size < 5000000:  # 5M elements
            target_chunks = max(2, min(4, cpu_count))  # Use 2-4 chunks for smaller datasets
        else:
            target_chunks = min(8, cpu_count)  # Cap at 8 to avoid excessive overhead
    
    # Calculate base chunk size with a minimum to avoid overhead of too many small chunks
    chunk_size = max(50000, total_size // target_chunks)
    
    return chunk_size


def chunk_array(array: np.ndarray, chunk_size: int) -> List[np.ndarray]:
    """Split an array into chunks of the specified size.
    
    Args:
        array: Array to split
        chunk_size: Size of each chunk
        
    Returns:
        List of array chunks
    """
    if len(array) <= chunk_size:
        return [array]
        
    return [array[i:i+chunk_size] for i in range(0, len(array), chunk_size)]


def parallel_process(func: Callable[[T], R], items: List[T], config: Optional[ParallelConfig] = None, **kwargs) -> List[R]:
    """Process items in parallel using the given function.
    
    This function is a convenience wrapper around ParallelExecutor.map().
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        config: Parallel processing configuration
        **kwargs: Additional arguments to pass to the function
        
    Returns:
        List of results
    """
    executor = ParallelExecutor(config)
    return executor.map(func, items, **kwargs)


class ParallelExecutor:
    """Class for executing tasks in parallel.
    
    This class provides a simple interface for executing tasks in parallel
    using either processes or threads.
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """Initialize the executor with the given configuration.
        
        Args:
            config: Parallel processing configuration
        """
        self.config = config if config is not None else ParallelConfig()
        
        # Determine the number of workers
        if self.config.max_workers is None:
            # More intelligent worker selection based on data characteristics
            cpu_count = mp.cpu_count()
            # For most systems, using n-1 cores is optimal to leave resources for OS
            self.workers = max(2, min(cpu_count - 1 if cpu_count > 2 else cpu_count, 16))
        else:
            self.workers = self.config.max_workers
    
    def map(self, func: Callable[[T], R], items: List[T], **kwargs) -> List[R]:
        """Execute a function on multiple items in parallel.
        
        Args:
            func: Function to execute on each item
            items: List of items to process
            **kwargs: Additional arguments to pass to the function
            
        Returns:
            List of results
        """
        # Handle empty lists
        if not items:
            return []
            
        # Fall back to serial execution if parallel processing is disabled or dataset is too small
        if not self.config.enabled or not is_parallelizable(len(items), self.config):
            return [func(item, **kwargs) for item in items]
        
        # Calculate chunk size if not specified
        chunk_size = self.config.chunk_size
        if chunk_size is None:
            chunk_size = get_optimal_chunk_size(len(items))
        
        # Split items into chunks - ensure chunks aren't too small
        chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
        
        # If we have very few chunks, just run in serial to avoid parallelization overhead
        if len(chunks) <= 1:
            return [func(item, **kwargs) for item in items]
        
        # For very small jobs, reduce worker count to avoid overhead
        max_workers = min(self.workers, len(chunks))
        
        # Define a wrapper function that accepts a chunk of items
        def process_chunk(chunk):
            return [func(item, **kwargs) for item in chunk]
        
        # Choose executor based on method
        executor_class = ProcessPoolExecutor if self.config.method == 'process' else ThreadPoolExecutor
        
        # Process chunks in parallel
        start_time = time.time()
        try:
            # Mark that we're entering a parallel context
            _parallel_context.in_parallel = True
            
            with executor_class(max_workers=max_workers) as executor:
                # Use simpler, more efficient executor.map pattern
                chunk_results = list(executor.map(process_chunk, chunks))
                
            # Flatten results
            results = []
            for chunk_result in chunk_results:
                results.extend(chunk_result)
                
            end_time = time.time()
            logger.debug(f"Parallel processing completed in {end_time - start_time:.2f} seconds")
            
            return results
        except Exception as e:
            logger.error(f"Error in parallel execution: {e}")
            # Fall back to sequential execution if parallel fails
            logger.info("Falling back to sequential execution due to error")
            return [func(item, **kwargs) for item in items]
        finally:
            # Clear parallel context when exiting
            _parallel_context.in_parallel = False


def process_array_in_chunks(
    func: Callable[[np.ndarray], np.ndarray],
    array: np.ndarray,
    chunk_size: int,
    config: Optional[ParallelConfig] = None,
    **kwargs
) -> np.ndarray:
    """Process a NumPy array in chunks and combine the results.
    
    This function is specifically designed for operations on large NumPy arrays,
    where the operation can be split into independent chunks.
    
    Args:
        func: Function to apply to each chunk
        array: NumPy array to process
        chunk_size: Size of each chunk
        config: Parallel processing configuration
        **kwargs: Additional arguments to pass to the function
        
    Returns:
        Processed array
    """
    # If the array is small, process it directly
    if len(array) <= chunk_size:
        return func(array, **kwargs)
    
    # Split the array into chunks
    chunks = chunk_array(array, chunk_size)
    
    # Process chunks in parallel
    executor = ParallelExecutor(config)
    processed_chunks = executor.map(func, chunks, **kwargs)
    
    # Combine the processed chunks
    return np.vstack(processed_chunks)
