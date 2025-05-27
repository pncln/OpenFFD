"""
Utilities for creating and manipulating FFD control boxes.

This module provides functions for creating Free-Form Deformation (FFD) control boxes
based on mesh point coordinates. FFD is a powerful technique used in shape optimization
and computational design to smoothly deform complex geometries.

This implementation includes parallel processing capabilities for handling very large meshes efficiently.
"""

import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from openffd.core.parallel_processing import (
    compute_bounding_box_parallel,
    create_control_points_parallel
)
from openffd.utils.parallel import ParallelConfig

# Configure logging
logger = logging.getLogger(__name__)


def create_ffd_box(
    mesh_points: np.ndarray,
    control_dim: tuple = (4, 4, 4),
    margin: float = 0.0,
    custom_dims: Optional[List[Optional[Tuple[Optional[float], Optional[float]]]]] = None,
    parallel_config: Optional[ParallelConfig] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Create an FFD control box based on mesh points.
    
    Args:
        mesh_points: Numpy array of point coordinates with shape (n, 3)
        control_dim: Tuple of control point dimensions (nx, ny, nz)
        margin: Margin to add around the bounding box (must be non-negative)
        custom_dims: Optional list of tuples [(min_x, max_x), (min_y, max_y), (min_z, max_z)]
                     specifying custom dimensions for the FFD box. Any None value
                     will be calculated from mesh_points.
        
    Returns:
        Tuple containing:
        - Numpy array of control point coordinates with shape (nx*ny*nz, 3)
        - Tuple of (min_coords, max_coords) for the bounding box
        
    Raises:
        ValueError: If mesh_points is empty or has wrong shape
        ValueError: If control_dim contains non-positive values
        ValueError: If margin is negative
    """
    # Validate inputs
    if not isinstance(mesh_points, np.ndarray):
        raise TypeError("mesh_points must be a numpy array")
        
    if mesh_points.size == 0:
        raise ValueError("No valid mesh points provided to create FFD box")
    
    if len(mesh_points.shape) != 2 or mesh_points.shape[1] != 3:
        raise ValueError(f"mesh_points must have shape (n, 3), got {mesh_points.shape}")
    
    if not all(dim > 1 for dim in control_dim):
        raise ValueError(f"All dimensions in control_dim must be > 1, got {control_dim}")
        
    if len(control_dim) != 3:
        raise ValueError(f"control_dim must have exactly 3 values, got {len(control_dim)}")
        
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    
    # Calculate bounding box with margin using parallel processing if available
    try:
        if parallel_config is None:
            parallel_config = ParallelConfig()
            
        # Use parallel processing for large meshes
        if parallel_config.enabled and len(mesh_points) >= parallel_config.threshold:
            logger.info(f"Using parallel processing for bounding box calculation with {len(mesh_points)} points")
            calc_min_coords, calc_max_coords = compute_bounding_box_parallel(
                mesh_points, margin, parallel_config
            )
        else:
            # Standard approach for smaller meshes
            calc_min_coords = np.min(mesh_points, axis=0) - margin
            calc_max_coords = np.max(mesh_points, axis=0) + margin
        
        # Initialize with calculated values
        min_coords = calc_min_coords.copy()
        max_coords = calc_max_coords.copy()
        
        # Apply custom dimensions if provided
        if custom_dims is not None:
            if not isinstance(custom_dims, list) or len(custom_dims) > 3:
                raise ValueError(f"custom_dims must be a list of up to 3 tuples, got {custom_dims}")
                
            for i, dim_range in enumerate(custom_dims):
                if dim_range is not None:
                    if not isinstance(dim_range, tuple) or len(dim_range) != 2:
                        raise ValueError(f"Each custom dimension must be a tuple (min, max), got {dim_range}")
                    min_val, max_val = dim_range
                    
                    if min_val is not None and max_val is not None and min_val > max_val:
                        raise ValueError(f"Min value {min_val} cannot be greater than max value {max_val}")
                        
                    if min_val is not None:
                        min_coords[i] = min_val
                    if max_val is not None:
                        max_coords[i] = max_val
    except Exception as e:
        logger.error(f"Error calculating bounding box: {e}")
        raise
    
    logger.info(f"Creating FFD box with dimensions {control_dim}")
    logger.debug(f"Bounding box: min={min_coords}, max={max_coords}")
    
    nx, ny, nz = control_dim
    
    # Check for degenerate dimensions
    dimension_names = ['x', 'y', 'z']
    for i in range(3):
        if np.isclose(min_coords[i], max_coords[i]):
            logger.warning(f"Degenerate {dimension_names[i]} dimension detected (min ≈ max)")
            # Add a small offset to prevent degenerate dimensions
            offset = 0.001 * (np.max(max_coords) - np.min(min_coords))
            if offset == 0:
                offset = 0.001
            min_coords[i] -= offset
            max_coords[i] += offset
    
    # Create arrays of coordinates along each axis
    try:
        xs = np.linspace(min_coords[0], max_coords[0], nx)
        ys = np.linspace(min_coords[1], max_coords[1], ny)
        zs = np.linspace(min_coords[2], max_coords[2], nz)
    except Exception as e:
        logger.error(f"Error creating coordinate arrays: {e}")
        raise
    
    # Create control points with correct ordering for FFD
    try:
        # Check if we should use parallel processing
        total_points = nx * ny * nz
        if parallel_config.enabled and total_points >= parallel_config.threshold:
            logger.info(f"Using parallel processing to create {total_points} control points")
            control_points = create_control_points_parallel(
                min_coords, max_coords, (nx, ny, nz), parallel_config
            )
        else:
            # Standard approach for smaller control point grids
            control_points = np.zeros((nx * ny * nz, 3))
            idx = 0
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        control_points[idx] = [xs[i], ys[j], zs[k]]
                        idx += 1
    except Exception as e:
        logger.error(f"Error creating control points: {e}")
        raise
    
    logger.info(f"Created FFD control box with {control_points.shape[0]} control points")
    return control_points, (min_coords, max_coords)


def extract_patch_points(mesh_data: Any, patch_name: str) -> np.ndarray:
    """Extract points belonging to a specified patch.
    
    This function is a wrapper to avoid circular imports.
    
    Args:
        mesh_data: Mesh object (meshio.Mesh or similar)
        patch_name: Name of zone, cell set/patch or Gmsh physical group
        
    Returns:
        np.ndarray: Array of point coordinates
        
    Raises:
        ValueError: If patch_name is not found in the mesh
        ValueError: If the patch contains no points
    """
    if not patch_name or not isinstance(patch_name, str):
        raise ValueError(f"patch_name must be a non-empty string, got {patch_name}")
        
    # Import the function from mesh.general to avoid circular imports
    from openffd.mesh.general import extract_patch_points as _extract_patch_points
    try:
        return _extract_patch_points(mesh_data, patch_name)
    except Exception as e:
        logger.error(f"Error extracting patch points: {e}")
        raise
