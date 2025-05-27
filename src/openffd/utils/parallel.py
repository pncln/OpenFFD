"""
Parallel processing utilities for OpenFFD.

This module provides functions and classes for parallel processing of large
mesh data and FFD operations using modern multiprocessing techniques.
"""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, Iterator

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic type hints
T = TypeVar('T')
R = TypeVar('R')


@dataclass
class ParallelConfig:
    """Configuration for parallel processing.
    
    Attributes:
        enabled: Whether parallel processing is enabled
        method: The method to use for parallelization ('process' or 'thread')
        max_workers: Maximum number of worker processes/threads (None = auto)
        chunk_size: Size of data chunks for parallel processing (None = auto)
        threshold: Minimum data size to trigger parallelization
    """
    
    enabled: bool = True
    method: str = 'process'  # 'process' or 'thread'
    max_workers: Optional[int] = None  # None = auto-detect based on CPU count
    chunk_size: Optional[int] = None  # None = auto-calculate
    threshold: int = 10000  # Minimum data size to trigger parallelization


def is_parallelizable(array_size: int, config: Optional[ParallelConfig] = None) -> bool:
    """Determine if an operation should be parallelized based on array size and configuration.
    
    Args:
        array_size: Size of the array to process
        config: Parallel processing configuration
        
    Returns:
        bool: True if the operation should be parallelized
    """
    if config is None:
        config = ParallelConfig()
        
    if not config.enabled:
        return False
        
    return array_size >= config.threshold


def get_optimal_chunk_size(
    array_size: int, 
    max_workers: Optional[int] = None, 
    min_chunk_size: int = 1000
) -> int:
    """Calculate optimal chunk size for parallel processing.
    
    Args:
        array_size: Size of the array to process
        max_workers: Maximum number of worker processes/threads
        min_chunk_size: Minimum chunk size
        
    Returns:
        int: Optimal chunk size
    """
    if max_workers is None:
        # Use available CPU cores, but leave one for system tasks
        max_workers = max(1, os.cpu_count() or 4 - 1)
    
    # Calculate base chunk size
    chunk_size = max(min_chunk_size, array_size // max_workers)
    
    # Ensure we don't create more chunks than workers
    if array_size // chunk_size > max_workers:
        chunk_size = max(min_chunk_size, array_size // max_workers)
    
    return chunk_size


def chunk_array(array: np.ndarray, chunk_size: Optional[int] = None) -> List[np.ndarray]:
    """Split an array into chunks for parallel processing.
    
    Args:
        array: Array to split
        chunk_size: Size of each chunk (None = auto-calculate)
        
    Returns:
        List of array chunks
    """
    if array.size == 0:
        return [array]
        
    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(len(array))
    
    # Split array into chunks
    return np.array_split(array, max(1, len(array) // chunk_size))


def parallel_process(
    func: Callable[[T], R],
    items: List[T],
    config: Optional[ParallelConfig] = None,
    **kwargs
) -> List[R]:
    """Process items in parallel using either ProcessPoolExecutor or ThreadPoolExecutor.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        config: Parallel processing configuration
        **kwargs: Additional arguments to pass to the function
        
    Returns:
        List of results
    """
    if not items:
        return []
        
    if config is None:
        config = ParallelConfig()
    
    # If parallelization is disabled or below threshold, process sequentially
    if not config.enabled or len(items) < config.threshold:
        return [func(item, **kwargs) for item in items]
    
    # Determine max_workers
    max_workers = config.max_workers
    if max_workers is None:
        # Use available CPU cores, but leave one for system tasks
        max_workers = max(1, os.cpu_count() or 4 - 1)
    
    # Choose executor based on method
    executor_class = ProcessPoolExecutor if config.method == 'process' else ThreadPoolExecutor
    
    results = []
    start_time = time.time()
    
    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(func, item, **kwargs): i for i, item in enumerate(items)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_item):
            item_index = future_to_item[future]
            try:
                result = future.result()
                results.append((item_index, result))
            except Exception as e:
                logger.error(f"Error processing item {item_index}: {e}")
                # Optionally re-raise or handle the error
                raise
    
    # Sort results by original index
    results.sort(key=lambda x: x[0])
    
    # Extract just the results
    sorted_results = [result for _, result in results]
    
    end_time = time.time()
    logger.debug(f"Parallel processing completed in {end_time - start_time:.2f} seconds")
    
    return sorted_results


def parallelize(config: Optional[ParallelConfig] = None):
    """Decorator to parallelize a function that processes arrays.
    
    This decorator checks if the input array size exceeds the threshold for
    parallelization, and if so, splits the array into chunks and processes
    them in parallel.
    
    Args:
        config: Parallel processing configuration
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(array, *args, **kwargs):
            # Check if we should parallelize
            if not is_parallelizable(len(array), config):
                return func(array, *args, **kwargs)
            
            # Determine chunk size
            chunk_size = config.chunk_size if config and config.chunk_size else get_optimal_chunk_size(len(array))
            
            # Split array into chunks
            chunks = chunk_array(array, chunk_size)
            
            # Process chunks in parallel
            results = parallel_process(func, chunks, config, *args, **kwargs)
            
            # Combine results (the combination method depends on the function)
            try:
                # Most common case: result is a numpy array
                return np.concatenate(results)
            except (ValueError, TypeError):
                # If concatenation fails, return list of results
                return results
        
        return wrapper
    
    return decorator


class ParallelExecutor:
    """Class for executing tasks in parallel with a consistent configuration.
    
    This class provides a convenient interface for parallel processing with
    a reusable configuration, progress tracking, and exception handling.
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """Initialize the parallel executor.
        
        Args:
            config: Parallel processing configuration
        """
        self.config = config or ParallelConfig()
        self._start_time = None
        self._task_count = 0
        self._completed_tasks = 0
    
    def map(self, func: Callable[[T], R], items: List[T], **kwargs) -> List[R]:
        """Execute a function on multiple items in parallel.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            **kwargs: Additional arguments to pass to the function
            
        Returns:
            List of results
        """
        self._start_time = time.time()
        self._task_count = len(items)
        self._completed_tasks = 0
        
        if not items:
            return []
            
        # If parallelization is disabled or below threshold, process sequentially
        if not self.config.enabled or len(items) < self.config.threshold:
            results = []
            for item in items:
                results.append(func(item, **kwargs))
                self._completed_tasks += 1
            return results
        
        # Determine max_workers
        max_workers = self.config.max_workers
        if max_workers is None:
            # Use available CPU cores, but leave one for system tasks
            max_workers = max(1, os.cpu_count() or 4 - 1)
        
        # Choose executor based on method
        executor_class = ProcessPoolExecutor if self.config.method == 'process' else ThreadPoolExecutor
        
        results = []
        
        with executor_class(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(func, item, **kwargs): i for i, item in enumerate(items)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_item):
                item_index = future_to_item[future]
                try:
                    result = future.result()
                    results.append((item_index, result))
                except Exception as e:
                    logger.error(f"Error processing item {item_index}: {e}")
                    # Optionally re-raise or handle the error
                    raise
                
                self._completed_tasks += 1
        
        # Sort results by original index
        results.sort(key=lambda x: x[0])
        
        # Extract just the results
        sorted_results = [result for _, result in results]
        
        end_time = time.time()
        logger.debug(f"Parallel processing completed in {end_time - self._start_time:.2f} seconds")
        
        return sorted_results
    
    def process_array_in_chunks(
        self, 
        func: Callable[[np.ndarray], np.ndarray], 
        array: np.ndarray, 
        chunk_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Process a large array in parallel by splitting it into chunks.
        
        Args:
            func: Function to apply to each chunk
            array: Array to process
            chunk_size: Size of each chunk (None = auto-calculate)
            **kwargs: Additional arguments to pass to the function
            
        Returns:
            Processed array
        """
        if array.size == 0:
            return array
            
        if not self.config.enabled or array.shape[0] < self.config.threshold:
            return func(array, **kwargs)
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        if chunk_size is None:
            chunk_size = get_optimal_chunk_size(array.shape[0])
        
        # Split array into chunks
        chunks = chunk_array(array, chunk_size)
        
        # Process chunks in parallel
        results = self.map(func, chunks, **kwargs)
        
        # Combine results
        try:
            return np.concatenate(results)
        except ValueError:
            logger.warning("Could not concatenate results, returning as list")
            return results
    
    def get_progress(self) -> Tuple[int, int, float, Optional[float]]:
        """Get the current progress of parallel processing.
        
        Returns:
            Tuple containing:
            - Number of completed tasks
            - Total number of tasks
            - Percentage complete
            - Estimated time remaining in seconds (None if not available)
        """
        if self._task_count == 0:
            return 0, 0, 0.0, None
            
        percent = 100.0 * self._completed_tasks / self._task_count
        
        # Calculate estimated time remaining
        if self._start_time is not None and self._completed_tasks > 0:
            elapsed = time.time() - self._start_time
            time_per_task = elapsed / self._completed_tasks
            remaining = time_per_task * (self._task_count - self._completed_tasks)
        else:
            remaining = None
            
        return self._completed_tasks, self._task_count, percent, remaining
