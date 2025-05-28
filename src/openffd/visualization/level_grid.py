"""
Utilities for creating grid structures from hierarchical FFD control points.

This module provides functions for reshaping and creating grid structures
from hierarchical FFD control points with varying dimensions.
"""

import logging
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)


def create_level_grid(control_points: np.ndarray, dims: Tuple[int, int, int]) -> np.ndarray:
    """Reshape control points into a 3D grid structure.
    
    Args:
        control_points: Array of control point coordinates with shape (n, 3)
        dims: Dimensions of the control lattice (nx, ny, nz)
        
    Returns:
        Reshaped control points with shape (nx, ny, nz, 3)
        
    Raises:
        ValueError: If control points cannot be reshaped to the specified dimensions
    """
    nx, ny, nz = dims
    expected_size = nx * ny * nz
    
    if len(control_points) != expected_size:
        raise ValueError(f"Cannot reshape control points: expected {expected_size} points for dimensions {dims}, got {len(control_points)}")
    
    return control_points.reshape(nx, ny, nz, 3)


def create_grid_edges(grid: np.ndarray) -> List[np.ndarray]:
    """Create edges for a 3D grid.
    
    Args:
        grid: Reshaped control points with shape (nx, ny, nz, 3)
        
    Returns:
        List of edges, where each edge is a 2x3 array with start and end coordinates
    """
    nx, ny, nz, _ = grid.shape
    edges = []
    
    # X-direction edges
    for i in range(nx-1):
        for j in range(ny):
            for k in range(nz):
                edges.append(np.vstack([grid[i, j, k], grid[i+1, j, k]]))
    
    # Y-direction edges
    for i in range(nx):
        for j in range(ny-1):
            for k in range(nz):
                edges.append(np.vstack([grid[i, j, k], grid[i, j+1, k]]))
    
    # Z-direction edges
    for i in range(nx):
        for j in range(ny):
            for k in range(nz-1):
                edges.append(np.vstack([grid[i, j, k], grid[i, j, k+1]]))
    
    return edges


def create_boundary_edges(grid: np.ndarray) -> List[np.ndarray]:
    """Create edges for the boundary of a 3D grid.
    
    Args:
        grid: Reshaped control points with shape (nx, ny, nz, 3)
        
    Returns:
        List of edges, where each edge is a 2x3 array with start and end coordinates
    """
    nx, ny, nz, _ = grid.shape
    edges = []
    
    # Create a complete 3D grid structure instead of just boundaries
    # This ensures all points are properly connected for all hierarchical levels
    
    # X-direction edges
    for j in range(ny):
        for k in range(nz):
            for i in range(nx-1):
                edges.append(np.vstack([grid[i, j, k], grid[i+1, j, k]]))
    
    # Y-direction edges
    for i in range(nx):
        for k in range(nz):
            for j in range(ny-1):
                edges.append(np.vstack([grid[i, j, k], grid[i, j+1, k]]))
    
    # Z-direction edges
    for i in range(nx):
        for j in range(ny):
            for k in range(nz-1):
                edges.append(np.vstack([grid[i, j, k], grid[i, j, k+1]]))
    
    return edges


def try_create_level_grid(control_points: np.ndarray, dims: Tuple[int, int, int]) -> Optional[np.ndarray]:
    """Try to reshape control points into a 3D grid structure.
    
    Args:
        control_points: Array of control point coordinates with shape (n, 3)
        dims: Dimensions of the control lattice (nx, ny, nz)
        
    Returns:
        Reshaped control points with shape (nx, ny, nz, 3) or None if reshaping fails
    """
    try:
        return create_level_grid(control_points, dims)
    except ValueError as e:
        logger.debug(f"Could not create grid for dimensions {dims}: {e}")
        return None
