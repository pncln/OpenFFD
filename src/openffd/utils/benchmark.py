"""
Benchmark utilities for OpenFFD.

This module provides benchmarking tools to measure the performance of
parallel processing versus sequential processing for FFD operations.
"""

import argparse
import logging
import time
from typing import Optional, Tuple

import numpy as np

from openffd.core.control_box import create_ffd_box
from openffd.utils.parallel import ParallelConfig

# Configure logging
logger = logging.getLogger(__name__)


def generate_random_mesh(num_points: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random mesh with the specified number of points.

    Args:
        num_points: Number of points to generate
        seed: Random seed for reproducibility

    Returns:
        Numpy array of point coordinates with shape (num_points, 3)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random points in a unit cube
    return np.random.rand(num_points, 3)


def benchmark_ffd_creation(
    mesh_points: np.ndarray,
    control_dim: Tuple[int, int, int] = (4, 4, 4),
    margin: float = 0.1,
    parallel_config: Optional[ParallelConfig] = None,
    compare_sequential: bool = True
) -> dict:
    """Benchmark FFD control box creation with or without parallel processing.

    Args:
        mesh_points: Mesh points to use for the benchmark
        control_dim: Dimensions of the control lattice
        margin: Margin around the bounding box
        parallel_config: Parallel processing configuration
        compare_sequential: Whether to also run sequential processing for comparison

    Returns:
        Dictionary of benchmark results including timings and speedup
    """
    results = {}
    
    # Run with parallel processing
    if parallel_config is None:
        parallel_config = ParallelConfig(enabled=True)
    
    # Ensure parallel processing is enabled
    parallel_config.enabled = True
    
    # Record start time
    start_time = time.time()
    
    # Create FFD box with parallel processing
    control_points, bbox = create_ffd_box(
        mesh_points, control_dim, margin, None, parallel_config
    )
    
    # Record end time
    parallel_time = time.time() - start_time
    
    results["parallel_time"] = parallel_time
    results["num_points"] = len(mesh_points)
    results["control_dim"] = control_dim
    results["num_control_points"] = len(control_points)
    results["parallel_config"] = {
        "method": parallel_config.method,
        "workers": parallel_config.max_workers,
        "chunk_size": parallel_config.chunk_size,
        "threshold": parallel_config.threshold
    }
    
    # Run with sequential processing if requested
    if compare_sequential:
        # Create a new config with parallel processing disabled
        sequential_config = ParallelConfig(enabled=False)
        
        # Record start time
        start_time = time.time()
        
        # Create FFD box without parallel processing
        control_points, bbox = create_ffd_box(
            mesh_points, control_dim, margin, None, sequential_config
        )
        
        # Record end time
        sequential_time = time.time() - start_time
        
        results["sequential_time"] = sequential_time
        results["speedup"] = sequential_time / parallel_time if parallel_time > 0 else 0
    
    return results


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the benchmark with the given arguments.

    Args:
        args: Command-line arguments
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate mesh points
    logger.info(f"Generating random mesh with {args.num_points} points...")
    mesh_points = generate_random_mesh(args.num_points, args.seed)
    
    # Create parallel configuration
    parallel_config = ParallelConfig(
        enabled=True,
        method=args.method,
        max_workers=args.workers,
        chunk_size=args.chunk_size,
        threshold=args.threshold
    )
    
    # Run benchmark
    logger.info("Running benchmark...")
    results = benchmark_ffd_creation(
        mesh_points,
        tuple(args.dims),
        args.margin,
        parallel_config,
        args.compare
    )
    
    # Print results
    logger.info("Benchmark results:")
    logger.info(f"  Number of mesh points: {results['num_points']}")
    logger.info(f"  Control lattice dimensions: {results['control_dim']}")
    logger.info(f"  Number of control points: {results['num_control_points']}")
    logger.info(f"  Parallel processing method: {results['parallel_config']['method']}")
    
    if results['parallel_config']['workers'] is not None:
        logger.info(f"  Number of workers: {results['parallel_config']['workers']}")
    else:
        logger.info("  Number of workers: auto-detected")
    
    logger.info(f"  Parallel processing time: {results['parallel_time']:.4f} seconds")
    
    if args.compare:
        logger.info(f"  Sequential processing time: {results['sequential_time']:.4f} seconds")
        logger.info(f"  Speedup: {results['speedup']:.2f}x")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark.

    Returns:
        Command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Benchmark parallel processing in OpenFFD'
    )
    
    parser.add_argument(
        '--num-points', type=int, default=100000,
        help='Number of mesh points to generate (default: 100000)'
    )
    parser.add_argument(
        '--dims', type=int, nargs=3, default=[4, 4, 4],
        help='Dimensions of the control lattice (default: 4 4 4)'
    )
    parser.add_argument(
        '--margin', type=float, default=0.1,
        help='Margin around the bounding box (default: 0.1)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility (default: None)'
    )
    parser.add_argument(
        '--method', choices=['process', 'thread'], default='process',
        help='Parallelization method (default: process)'
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of worker processes/threads (default: auto-detect)'
    )
    parser.add_argument(
        '--chunk-size', type=int, default=None,
        help='Size of data chunks for parallel processing (default: auto-calculate)'
    )
    parser.add_argument(
        '--threshold', type=int, default=10000,
        help='Minimum data size to trigger parallelization (default: 10000)'
    )
    parser.add_argument(
        '--compare', action='store_true', default=True,
        help='Compare with sequential processing (default: True)'
    )
    parser.add_argument(
        '--no-compare', action='store_false', dest='compare',
        help='Do not compare with sequential processing'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_benchmark(args)
