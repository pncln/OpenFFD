"""
Parallel processing implementations for core FFD operations.

This module provides parallelized versions of computationally intensive
operations in the FFD control box creation and manipulation process.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np

from openffd.utils.parallel import (
    ParallelConfig,
    ParallelExecutor,
    chunk_array,
    get_optimal_chunk_size,
    is_parallelizable
)

# Configure logging
logger = logging.getLogger(__name__)


def compute_bounding_box_parallel(
    points: np.ndarray,
    margin: float = 0.0,
    config: Optional[ParallelConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the bounding box of a point cloud in parallel.
    
    This computes the min and max coordinates of a point cloud with an optional margin.
    For very large point clouds, this can be significantly faster than using numpy's
    min and max functions directly.
    
    Args:
        points: Numpy array of point coordinates with shape (n, 3)
        margin: Margin to add around the bounding box
        config: Parallel processing configuration
        
    Returns:
        Tuple of (min_coords, max_coords)
    """
    if not is_parallelizable(len(points), config):
        # Use standard numpy operations for small arrays
        min_coords = np.min(points, axis=0) - margin
        max_coords = np.max(points, axis=0) + margin
        return min_coords, max_coords
    
    # Use parallel processing for large arrays
    executor = ParallelExecutor(config)
    
    # Function to compute min/max of a chunk
    def compute_minmax_chunk(chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.min(chunk, axis=0), np.max(chunk, axis=0)
    
    # Determine chunk size
    chunk_size = config.chunk_size if config and config.chunk_size else get_optimal_chunk_size(len(points))
    
    # Split array into chunks
    chunks = chunk_array(points, chunk_size)
    
    # Process chunks in parallel
    results = executor.map(compute_minmax_chunk, chunks)
    
    # Combine results
    all_mins, all_maxs = zip(*results)
    
    # Find global min/max
    min_coords = np.min(all_mins, axis=0) - margin
    max_coords = np.max(all_maxs, axis=0) + margin
    
    return min_coords, max_coords


def create_control_points_parallel(
    min_coords: np.ndarray,
    max_coords: np.ndarray,
    dims: Tuple[int, int, int],
    config: Optional[ParallelConfig] = None
) -> np.ndarray:
    """Create FFD control points in parallel for large control lattices.
    
    This creates a grid of control points for the FFD box. For large lattices,
    parallel processing can significantly improve performance.
    
    Args:
        min_coords: Minimum coordinates (x_min, y_min, z_min)
        max_coords: Maximum coordinates (x_max, y_max, z_max)
        dims: Dimensions of the control lattice (nx, ny, nz)
        config: Parallel processing configuration
        
    Returns:
        Numpy array of control point coordinates with shape (nx*ny*nz, 3)
    """
    nx, ny, nz = dims
    total_points = nx * ny * nz
    
    # Create arrays of coordinates along each axis
    xs = np.linspace(min_coords[0], max_coords[0], nx)
    ys = np.linspace(min_coords[1], max_coords[1], ny)
    zs = np.linspace(min_coords[2], max_coords[2], nz)
    
    # If the total number of points is small, use standard approach
    if not is_parallelizable(total_points, config):
        control_points = np.zeros((total_points, 3))
        idx = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    control_points[idx] = [xs[i], ys[j], zs[k]]
                    idx += 1
        return control_points
    
    # For large lattices, use parallel processing
    executor = ParallelExecutor(config)
    
    # Create a function to generate a portion of the control points
    def create_control_points_chunk(chunk_indices: List[int]) -> np.ndarray:
        chunk_size = len(chunk_indices)
        chunk_points = np.zeros((chunk_size, 3))
        
        for local_idx, global_idx in enumerate(chunk_indices):
            # Convert flat index to 3D indices
            i = global_idx // (ny * nz)
            remainder = global_idx % (ny * nz)
            j = remainder // nz
            k = remainder % nz
            
            # Set coordinates
            chunk_points[local_idx] = [xs[i], ys[j], zs[k]]
            
        return chunk_points
    
    # Create list of all indices
    all_indices = list(range(total_points))
    
    # Determine chunk size
    chunk_size = config.chunk_size if config and config.chunk_size else get_optimal_chunk_size(total_points)
    
    # Split indices into chunks
    index_chunks = [all_indices[i:i + chunk_size] for i in range(0, len(all_indices), chunk_size)]
    
    # Process chunks in parallel
    logger.info(f"Creating {total_points} control points in parallel using {len(index_chunks)} chunks")
    chunk_results = executor.map(create_control_points_chunk, index_chunks)
    
    # Combine results
    return np.vstack(chunk_results)


def compute_control_point_distances_parallel(
    control_points: np.ndarray,
    mesh_points: np.ndarray,
    config: Optional[ParallelConfig] = None
) -> np.ndarray:
    """Compute distances from mesh points to control points in parallel.
    
    This computes the Euclidean distance from each mesh point to each control point.
    For large meshes, this operation can be very computationally intensive.
    
    Args:
        control_points: Numpy array of control point coordinates with shape (m, 3)
        mesh_points: Numpy array of mesh point coordinates with shape (n, 3)
        config: Parallel processing configuration
        
    Returns:
        Numpy array of distances with shape (n, m)
    """
    # If either array is small, use standard approach
    if not is_parallelizable(len(mesh_points), config):
        # Use standard numpy operations
        # For each mesh point, compute distance to all control points
        distances = np.zeros((len(mesh_points), len(control_points)))
        for i, point in enumerate(mesh_points):
            # Vectorized distance calculation
            distances[i] = np.sqrt(np.sum((control_points - point) ** 2, axis=1))
        return distances
    
    # For large meshes, use parallel processing
    executor = ParallelExecutor(config)
    
    # Function to compute distances for a chunk of mesh points
    def compute_distances_chunk(mesh_chunk: np.ndarray) -> np.ndarray:
        chunk_distances = np.zeros((len(mesh_chunk), len(control_points)))
        for i, point in enumerate(mesh_chunk):
            # Vectorized distance calculation
            chunk_distances[i] = np.sqrt(np.sum((control_points - point) ** 2, axis=1))
        return chunk_distances
    
    # Determine chunk size
    chunk_size = config.chunk_size if config and config.chunk_size else get_optimal_chunk_size(len(mesh_points))
    
    # Split mesh points into chunks
    mesh_chunks = chunk_array(mesh_points, chunk_size)
    
    # Process chunks in parallel
    logger.info(f"Computing distances for {len(mesh_points)} mesh points in parallel using {len(mesh_chunks)} chunks")
    chunk_results = executor.map(compute_distances_chunk, mesh_chunks)
    
    # Combine results
    return np.vstack(chunk_results)


def find_nearest_control_points_parallel(
    mesh_points: np.ndarray,
    control_points: np.ndarray,
    config: Optional[ParallelConfig] = None
) -> np.ndarray:
    """Find the nearest control points for each mesh point in parallel.
    
    This identifies the index of the nearest control point for each mesh point,
    which is useful for FFD deformation and weight calculation.
    
    Args:
        mesh_points: Numpy array of mesh point coordinates with shape (n, 3)
        control_points: Numpy array of control point coordinates with shape (m, 3)
        config: Parallel processing configuration
        
    Returns:
        Numpy array of indices with shape (n,)
    """
    # If the mesh is small, use standard approach
    if not is_parallelizable(len(mesh_points), config):
        nearest_indices = np.zeros(len(mesh_points), dtype=np.int32)
        for i, point in enumerate(mesh_points):
            # Compute distances to all control points
            distances = np.sum((control_points - point) ** 2, axis=1)
            # Find index of minimum distance
            nearest_indices[i] = np.argmin(distances)
        return nearest_indices
    
    # For large meshes, use parallel processing
    executor = ParallelExecutor(config)
    
    # Function to find nearest control points for a chunk of mesh points
    def find_nearest_chunk(mesh_chunk: np.ndarray) -> np.ndarray:
        chunk_indices = np.zeros(len(mesh_chunk), dtype=np.int32)
        for i, point in enumerate(mesh_chunk):
            # Compute distances to all control points
            distances = np.sum((control_points - point) ** 2, axis=1)
            # Find index of minimum distance
            chunk_indices[i] = np.argmin(distances)
        return chunk_indices
    
    # Determine chunk size
    chunk_size = config.chunk_size if config and config.chunk_size else get_optimal_chunk_size(len(mesh_points))
    
    # Split mesh points into chunks
    mesh_chunks = chunk_array(mesh_points, chunk_size)
    
    # Process chunks in parallel
    logger.info(f"Finding nearest control points for {len(mesh_points)} mesh points in parallel")
    chunk_results = executor.map(find_nearest_chunk, mesh_chunks)
    
    # Combine results
    return np.concatenate(chunk_results)


def compute_ffd_weights_parallel(
    mesh_points: np.ndarray,
    control_points: np.ndarray,
    dims: Tuple[int, int, int],
    config: Optional[ParallelConfig] = None
) -> np.ndarray:
    """Compute FFD weights for mesh points in parallel.
    
    This computes the weights of each control point for each mesh point
    using trilinear interpolation or another weighting scheme.
    
    Args:
        mesh_points: Numpy array of mesh point coordinates with shape (n, 3)
        control_points: Numpy array of control point coordinates with shape (m, 3)
        dims: Dimensions of the control lattice (nx, ny, nz)
        config: Parallel processing configuration
        
    Returns:
        Numpy array of weights with shape (n, m)
    """
    nx, ny, nz = dims
    
    # Reshape control points for easier indexing
    cp_reshaped = control_points.reshape(nx, ny, nz, 3)
    
    # If the mesh is small, use standard approach
    if not is_parallelizable(len(mesh_points), config):
        weights = np.zeros((len(mesh_points), len(control_points)))
        
        # For each mesh point
        for i, point in enumerate(mesh_points):
            # Find the control lattice cell containing this point
            # (implementation depends on your FFD parameterization method)
            # This is a simplified example using nearest neighbors
            distances = np.sum((control_points - point) ** 2, axis=1)
            nearest_idx = np.argmin(distances)
            
            # Set weight for nearest control point
            weights[i, nearest_idx] = 1.0
            
        return weights
    
    # For large meshes, use parallel processing
    executor = ParallelExecutor(config)
    
    # Function to compute weights for a chunk of mesh points
    def compute_weights_chunk(mesh_chunk: np.ndarray) -> np.ndarray:
        chunk_weights = np.zeros((len(mesh_chunk), len(control_points)))
        
        # For each mesh point in the chunk
        for i, point in enumerate(mesh_chunk):
            # Find the control lattice cell containing this point
            # (implementation depends on your FFD parameterization method)
            # This is a simplified example using nearest neighbors
            distances = np.sum((control_points - point) ** 2, axis=1)
            nearest_idx = np.argmin(distances)
            
            # Set weight for nearest control point
            chunk_weights[i, nearest_idx] = 1.0
            
        return chunk_weights
    
    # Determine chunk size
    chunk_size = config.chunk_size if config and config.chunk_size else get_optimal_chunk_size(len(mesh_points))
    
    # Split mesh points into chunks
    mesh_chunks = chunk_array(mesh_points, chunk_size)
    
    # Process chunks in parallel
    logger.info(f"Computing FFD weights for {len(mesh_points)} mesh points in parallel")
    chunk_results = executor.map(compute_weights_chunk, mesh_chunks)
    
    # Combine results
    return np.vstack(chunk_results)
