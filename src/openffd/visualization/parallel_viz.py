"""
Parallel visualization utilities for OpenFFD.

This module provides parallel processing implementations for visualization tasks,
which can significantly improve performance when working with large meshes.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np

from openffd.utils.parallel import (
    ParallelConfig,
    parallel_process,
    chunk_array,
    is_parallelizable
)

# Configure logging
logger = logging.getLogger(__name__)

# Try importing PyVista for enhanced visualization
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    logger.warning("PyVista not available. Install with 'pip install pyvista' for enhanced visualization.")


def process_point_cloud_parallel(
    points: np.ndarray,
    process_func: Callable[[np.ndarray], Any],
    config: Optional[ParallelConfig] = None,
    **kwargs
) -> List[Any]:
    """Process a point cloud in parallel.
    
    Args:
        points: Array of point coordinates with shape (n, 3)
        process_func: Function to apply to each chunk of points
        config: Parallel processing configuration
        **kwargs: Additional arguments to pass to the processing function
        
    Returns:
        List of processing results for each chunk
    """
    if config is None:
        config = ParallelConfig()
    
    # Only use parallel processing for large point clouds
    if not is_parallelizable(len(points), config):
        return [process_func(points, **kwargs)]
    
    # Calculate chunk size
    from openffd.utils.parallel import get_optimal_chunk_size
    chunk_size = config.chunk_size if config.chunk_size else get_optimal_chunk_size(len(points))
    
    # Split points into chunks
    point_chunks = chunk_array(points, chunk_size)
    
    # Process chunks in parallel
    logger.info(f"Processing {len(points)} points in parallel using {len(point_chunks)} chunks")
    return parallel_process(process_func, point_chunks, config, **kwargs)


def subsample_points_parallel(
    points: np.ndarray,
    max_points: int,
    config: Optional[ParallelConfig] = None
) -> np.ndarray:
    """Subsample points in parallel for visualization.
    
    This function subsamples a large point cloud to a manageable size
    for visualization, using parallel processing for better performance.
    
    Args:
        points: Array of point coordinates with shape (n, 3)
        max_points: Maximum number of points to keep
        config: Parallel processing configuration
        
    Returns:
        Subsampled array of points
    """
    # If already small enough, return as is
    if len(points) <= max_points:
        return points
    
    # Use uniform sampling to reduce the number of points
    def subsample_chunk(chunk: np.ndarray) -> np.ndarray:
        # Calculate sampling rate based on total points and max allowed
        chunk_max = max(1, int(max_points * (len(chunk) / len(points))))
        
        # Uniform sampling
        indices = np.linspace(0, len(chunk) - 1, chunk_max, dtype=int)
        return chunk[indices]
    
    # Process in parallel
    results = process_point_cloud_parallel(points, subsample_chunk, config)
    
    # Combine results
    return np.vstack(results)


def create_mesh_chunks_parallel(
    points: np.ndarray,
    faces: np.ndarray,
    max_triangles: int,
    config: Optional[ParallelConfig] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split a large mesh into manageable chunks for visualization.
    
    Args:
        points: Array of point coordinates with shape (n, 3)
        faces: Array of face indices with shape (m, 3)
        max_triangles: Maximum number of triangles per chunk
        config: Parallel processing configuration
        
    Returns:
        List of (points, faces) tuples for each chunk
    """
    # If already small enough, return as is
    if len(faces) <= max_triangles:
        return [(points, faces)]
    
    # Calculate number of chunks needed
    num_chunks = max(1, len(faces) // max_triangles)
    
    # Function to process a chunk of faces
    def process_face_chunk(face_chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Get unique points used by this chunk
        unique_indices = np.unique(face_chunk.flatten())
        chunk_points = points[unique_indices]
        
        # Create mapping from global to local indices
        index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(unique_indices)}
        
        # Remap face indices to local indices
        remapped_faces = np.array([
            [index_map[idx] for idx in face]
            for face in face_chunk
        ])
        
        return chunk_points, remapped_faces
    
    # Split faces into chunks
    face_chunks = np.array_split(faces, num_chunks)
    
    # Process in parallel if configured
    if config and config.enabled and len(face_chunks) > 1:
        logger.info(f"Processing {len(faces)} faces in parallel using {len(face_chunks)} chunks")
        return parallel_process(process_face_chunk, face_chunks, config)
    else:
        # Process sequentially
        return [process_face_chunk(chunk) for chunk in face_chunks]


def compute_normals_parallel(
    points: np.ndarray,
    faces: np.ndarray,
    config: Optional[ParallelConfig] = None
) -> np.ndarray:
    """Compute surface normals in parallel.
    
    Args:
        points: Array of point coordinates with shape (n, 3)
        faces: Array of face indices with shape (m, 3)
        config: Parallel processing configuration
        
    Returns:
        Array of face normals with shape (m, 3)
    """
    def compute_face_normals(face_chunk: np.ndarray) -> np.ndarray:
        # Compute normals for each face in the chunk
        normals = np.zeros((len(face_chunk), 3))
        for i, face in enumerate(face_chunk):
            # Get vertices of the face
            v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]
            
            # Compute normal as cross product of edges
            normal = np.cross(v1 - v0, v2 - v0)
            
            # Normalize
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
                
            normals[i] = normal
                
        return normals
    
    # Use parallel processing for large face arrays
    if config is None:
        config = ParallelConfig()
        
    if not is_parallelizable(len(faces), config):
        return compute_face_normals(faces)
    
    # Calculate chunk size
    chunk_size = config.chunk_size if config.chunk_size else get_optimal_chunk_size(len(faces))
    
    # Split faces into chunks
    face_chunks = chunk_array(faces, chunk_size)
    
    # Process chunks in parallel
    logger.info(f"Computing normals for {len(faces)} faces in parallel using {len(face_chunks)} chunks")
    chunk_normals = parallel_process(compute_face_normals, face_chunks, config)
    
    # Combine results
    return np.vstack(chunk_normals)


def extract_mesh_features_parallel(
    points: np.ndarray,
    faces: Optional[np.ndarray] = None,
    compute_bbox: bool = True,
    compute_center: bool = True,
    config: Optional[ParallelConfig] = None
) -> Dict[str, Any]:
    """Extract various features from a mesh in parallel.
    
    Args:
        points: Array of point coordinates with shape (n, 3)
        faces: Optional array of face indices
        compute_bbox: Whether to compute the bounding box
        compute_center: Whether to compute the center point
        config: Parallel processing configuration
        
    Returns:
        Dictionary of extracted features
    """
    result = {}
    
    # Only use parallel processing for large point clouds
    if not is_parallelizable(len(points), config):
        # Sequential computation
        if compute_bbox:
            result['bbox_min'] = np.min(points, axis=0)
            result['bbox_max'] = np.max(points, axis=0)
        
        if compute_center:
            result['center'] = np.mean(points, axis=0)
            
        return result
    
    # Calculate features in parallel
    # Function to compute min/max of a chunk
    def compute_minmax(chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.min(chunk, axis=0), np.max(chunk, axis=0)
    
    # Function to compute sum and count for averaging
    def compute_sum_count(chunk: np.ndarray) -> Tuple[np.ndarray, int]:
        return np.sum(chunk, axis=0), len(chunk)
    
    # Process in parallel
    if compute_bbox:
        # Compute bounding box in parallel
        minmax_results = process_point_cloud_parallel(points, compute_minmax, config)
        mins, maxs = zip(*minmax_results)
        result['bbox_min'] = np.min(mins, axis=0)
        result['bbox_max'] = np.max(maxs, axis=0)
    
    if compute_center:
        # Compute center in parallel
        sum_count_results = process_point_cloud_parallel(points, compute_sum_count, config)
        sums, counts = zip(*sum_count_results)
        result['center'] = np.sum(sums, axis=0) / sum(counts)
    
    return result
