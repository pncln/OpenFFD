"""
Visualization utilities for FFD control boxes.

This module provides functions for visualizing FFD control lattices,
bounding boxes, and mesh points using matplotlib's 3D plotting capabilities.

Author: TÜBİTAK
Version: 1.0.0
"""
import numpy as np
import os
import sys
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, Normalize, to_hex
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial import Delaunay, ConvexHull
from typing import List, Dict, Tuple, Set, Optional, Union, Any
import random

# Import PyVista for fast, high-quality 3D visualization
import pyvista as pv

logger = logging.getLogger(__name__)

def visualize_ffd(control_points: np.ndarray, bbox: Tuple[np.ndarray, np.ndarray],
                  mesh_points: Optional[np.ndarray] = None, 
                  show_mesh: bool = False,
                  save_path: Optional[str] = None,
                  show_axes: bool = True,
                  show_grid: bool = True,
                  fig_size: Tuple[int, int] = (12, 10),
                  dpi: int = 300,
                  title: Optional[str] = None,
                  mesh_alpha: float = 0.3,
                  mesh_point_size: float = 2.0,
                  mesh_color: str = 'blue',
                  ffd_alpha: float = 0.7,
                  ffd_color: str = 'red',
                  max_mesh_points: int = 2000,
                  show_surface: bool = True,
                  surface_alpha: float = 0.3,
                  surface_color: str = 'lightblue',
                  show_control_points: bool = True,
                  control_point_size: float = 50.0,
                  lattice_width: float = 1.0,
                  view_angle: Optional[Tuple[float, float]] = None,
                  auto_scale: bool = True,
                  scale_factor: Optional[float] = None):
    """
    Visualize FFD control lattice and bounding box, optionally with the underlying mesh as a surface.
    
    Args:
        control_points: Numpy array of control point coordinates with shape (n, 3)
        bbox: Tuple of (min_coords, max_coords) for the bounding box
        mesh_points: Optional array of mesh points to visualize
        show_mesh: Whether to show the mesh points
        save_path: Optional path to save the figure
        show_axes: Whether to show the coordinate axes
        show_grid: Whether to show the grid lines
        fig_size: Size of the figure as (width, height) in inches
        dpi: Resolution of the figure in dots per inch
        title: Optional custom title for the plot
        mesh_alpha: Transparency of mesh points (0.0-1.0)
        mesh_point_size: Size of mesh points
        mesh_color: Color of mesh points
        ffd_alpha: Transparency of FFD box and control points (0.0-1.0)
        ffd_color: Color of FFD control points
        max_mesh_points: Maximum number of mesh points to display
        show_surface: Whether to show the mesh as a surface
        surface_alpha: Transparency of the mesh surface
        surface_color: Color of the mesh surface
        show_control_points: Whether to show the FFD control points
        control_point_size: Size of control points
        lattice_width: Width of lattice lines
        view_angle: Optional tuple of (elevation, azimuth) angles for the view
        auto_scale: Whether to automatically scale very small geometries for better visualization
        scale_factor: Optional manual scale factor to apply to all coordinates
        
    Raises:
        TypeError: If inputs have incorrect types
        ValueError: If inputs have incorrect shapes or values
        IOError: If the figure cannot be saved to the specified path
    """
    # Validate inputs
    if not isinstance(control_points, np.ndarray):
        raise TypeError("control_points must be a numpy array")
        
    if control_points.size == 0:
        raise ValueError("No control points to visualize")
        
    if len(control_points.shape) != 2 or control_points.shape[1] != 3:
        raise ValueError(f"control_points must have shape (n, 3), got {control_points.shape}")
    
    if not isinstance(bbox, tuple) or len(bbox) != 2:
        raise ValueError("bbox must be a tuple of (min_coords, max_coords)")
        
    min_coords, max_coords = bbox
    if not isinstance(min_coords, np.ndarray) or not isinstance(max_coords, np.ndarray):
        raise TypeError("bbox coordinates must be numpy arrays")
        
    if min_coords.shape != (3,) or max_coords.shape != (3,):
        raise ValueError(f"bbox coordinates must have shape (3,), got {min_coords.shape} and {max_coords.shape}")
        
    # Check if we need to scale the geometry for better visualization
    bbox_size = max_coords - min_coords
    bbox_max_size = np.max(bbox_size)
    bbox_min_size = np.min(bbox_size)
    
    # Determine if we need to scale (if the geometry is very small or has very different scales)
    needs_scaling = False
    if auto_scale:
        if bbox_max_size < 0.1:  # Very small geometry
            needs_scaling = True
            logger.info(f"Geometry is very small (max size: {bbox_max_size:.6f}), applying automatic scaling")
        elif bbox_max_size / max(bbox_min_size, 1e-10) > 1000:  # Very different scales
            needs_scaling = True
            logger.info(f"Geometry has very different scales (max/min: {bbox_max_size/max(bbox_min_size, 1e-10):.1f}), applying automatic scaling")
    
    # Apply scaling if needed or if a manual scale factor is provided
    if needs_scaling or scale_factor is not None:
        if scale_factor is None:
            # Determine automatic scale factor to make the geometry a reasonable size
            if bbox_max_size < 0.1:
                scale_factor = 1.0 / bbox_max_size  # Scale to make max dimension ~1.0
            else:
                scale_factor = 1.0  # Default
            logger.info(f"Automatically determined scale factor: {scale_factor:.1f}")
        
        # Apply scaling to all coordinates
        control_points = control_points * scale_factor
        min_coords = min_coords * scale_factor
        max_coords = max_coords * scale_factor
        if mesh_points is not None:
            mesh_points = mesh_points * scale_factor
        
        # Update bbox with scaled coordinates
        bbox = (min_coords, max_coords)
        logger.info(f"Scaled geometry by factor {scale_factor:.1f} for better visualization")
    
    if mesh_points is not None:
        if not isinstance(mesh_points, np.ndarray):
            raise TypeError("mesh_points must be a numpy array")
            
        if len(mesh_points.shape) != 2 or mesh_points.shape[1] != 3:
            raise ValueError(f"mesh_points must have shape (n, 3), got {mesh_points.shape}")
    
    if save_path is not None and not isinstance(save_path, str):
        raise TypeError("save_path must be a string")
    
    logger.info("Visualizing mesh with patches")
    
    # Check if we should use PyVista (much faster for large meshes)
    if use_pyvista:
        # Use PyVista for high-performance visualization
        return visualize_mesh_with_patches_pyvista(
            mesh_data=mesh_data,
            save_path=save_path,
            title=title,
            point_size=point_size,
            auto_scale=auto_scale,
            scale_factor=scale_factor,
            show_solid=show_solid,
            show_edges=False,
            color_by_zone=True,
            bgcolor='white',
            screenshot_size=(fig_size[0]*dpi, fig_size[1]*dpi),
            off_screen=False if save_path is None else True
        )
    
    # Otherwise fall back to Matplotlib visualization (slower but more compatible)
    # Create figure
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot control points with specified appearance if requested
    if show_control_points:
        try:
            # Get the shape of the control points array
            cp_shape = control_points.shape
            
            # Check if the control points are already in the correct format
            if len(cp_shape) == 2 and cp_shape[1] == 3:  # Already in Nx3 format
                reshaped_cps = control_points
                # Detect dimensions based on number of points
                n_points = cp_shape[0]
                if n_points == 64:  # 4x4x4
                    dims = (4, 4, 4)
                elif n_points == 27:  # 3x3x3
                    dims = (3, 3, 3)
                else:  # Default to a simple structure
                    dims = (int(n_points ** (1/3)), int(n_points ** (1/3)), int(n_points ** (1/3)))
                    logger.info(f"Detected control lattice structure: {dims[0]}x{dims[1]}x{dims[2]}")
            else:  # In i,j,k,3 format
                dims = cp_shape[:-1]  # Get dimensions excluding the last one (which is 3 for x,y,z)
                reshaped_cps = control_points.reshape(-1, 3)  # Flatten to Nx3 array
            
            # Plot all control points
            ax.scatter(reshaped_cps[:,0], reshaped_cps[:,1], reshaped_cps[:,2], 
                      c=ffd_color, marker='o', alpha=ffd_alpha, s=control_point_size)
            
            # If we have the dimensions, plot the control lattice
            if len(dims) == 3:
                # Reshape the control points to the 3D grid format if needed
                if len(cp_shape) == 2:  # If it was originally flat
                    control_grid = reshaped_cps.reshape(dims[0], dims[1], dims[2], 3)
                else:
                    control_grid = control_points
                
                # Plot the control lattice (connections between control points)
                for i in range(dims[0]):
                    for j in range(dims[1]):
                        for k in range(dims[2]):
                            # Connect points along i-direction
                            if i < dims[0] - 1:
                                p1 = control_grid[i, j, k]
                                p2 = control_grid[i+1, j, k]
                                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                                       color=ffd_color, alpha=ffd_alpha, linewidth=lattice_width)
                            
                            # Connect points along j-direction
                            if j < dims[1] - 1:
                                p1 = control_grid[i, j, k]
                                p2 = control_grid[i, j+1, k]
                                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                                       color=ffd_color, alpha=ffd_alpha, linewidth=lattice_width)
                            
                            # Connect points along k-direction
                            if k < dims[2] - 1:
                                p1 = control_grid[i, j, k]
                                p2 = control_grid[i, j, k+1]
                                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                                       color=ffd_color, alpha=ffd_alpha, linewidth=lattice_width)
            
            # Create a simple bounding box from the control points
            try:
                # Get the min and max coordinates of the control points
                min_coords = np.min(reshaped_cps, axis=0)
                max_coords = np.max(reshaped_cps, axis=0)
                
                # Create the corners of the bounding box
                corners = np.array([
                    [min_coords[0], min_coords[1], min_coords[2]],  # 0: min x, min y, min z
                    [min_coords[0], min_coords[1], max_coords[2]],  # 1: min x, min y, max z
                    [min_coords[0], max_coords[1], min_coords[2]],  # 2: min x, max y, min z
                    [min_coords[0], max_coords[1], max_coords[2]],  # 3: min x, max y, max z
                    [max_coords[0], min_coords[1], min_coords[2]],  # 4: max x, min y, min z
                    [max_coords[0], min_coords[1], max_coords[2]],  # 5: max x, min y, max z
                    [max_coords[0], max_coords[1], min_coords[2]],  # 6: max x, max y, min z
                    [max_coords[0], max_coords[1], max_coords[2]]   # 7: max x, max y, max z
                ])
                
                # Define the faces of the bounding box
                faces = [
                    [corners[0], corners[1], corners[3], corners[2]],  # Left face
                    [corners[4], corners[5], corners[7], corners[6]],  # Right face
                    [corners[0], corners[1], corners[5], corners[4]],  # Bottom face
                    [corners[2], corners[3], corners[7], corners[6]],  # Top face
                    [corners[0], corners[2], corners[6], corners[4]],  # Front face
                    [corners[1], corners[3], corners[7], corners[5]]   # Back face
                ]
                
                # Create a very transparent surface for the bounding box
                bbox_poly = Poly3DCollection(faces, alpha=0.1, edgecolor='gray', linewidth=0.5)
                bbox_poly.set_facecolor(ffd_color)  # Explicitly set facecolor
                ax.add_collection3d(bbox_poly)
            except Exception as e:
                logger.warning(f"Error creating bounding box: {e}")
            
            # No need for this section anymore as we're using a simpler approach
        except Exception as e:
            logger.warning(f"Error plotting control points: {e}")
            # Continue with the rest of the visualization
    
    # Plot mesh points and/or surface if provided and requested
    if mesh_points is not None and (show_mesh or show_surface):
        try:
            # Limit number of mesh points to avoid overloading the plot
            if len(mesh_points) > max_mesh_points:
                logger.info(f"Subsetting mesh points from {len(mesh_points)} to {max_mesh_points} for visualization")
                # Use a stratified sampling approach to better preserve the shape
                # First, divide the space into bins
                min_coords = np.min(mesh_points, axis=0)
                max_coords = np.max(mesh_points, axis=0)
                bins = int(np.cbrt(max_mesh_points))  # Cube root for 3D space division
                
                # Create a 3D histogram to count points in each bin
                H, edges = np.histogramdd(mesh_points, bins=[bins, bins, bins],
                                         range=[(min_coords[0], max_coords[0]),
                                                (min_coords[1], max_coords[1]),
                                                (min_coords[2], max_coords[2])])
                
                # Sample points from each bin proportionally
                sampled_points = []
                for i in range(bins):
                    for j in range(bins):
                        for k in range(bins):
                            # Find points in this bin
                            bin_mask = ((mesh_points[:, 0] >= edges[0][i]) & (mesh_points[:, 0] < edges[0][i+1]) &
                                       (mesh_points[:, 1] >= edges[1][j]) & (mesh_points[:, 1] < edges[1][j+1]) &
                                       (mesh_points[:, 2] >= edges[2][k]) & (mesh_points[:, 2] < edges[2][k+1]))
                            bin_points = mesh_points[bin_mask]
                            
                            # Sample points from this bin
                            if len(bin_points) > 0:
                                # Number of points to sample from this bin
                                n_samples = max(1, int(len(bin_points) * max_mesh_points / len(mesh_points)))
                                n_samples = min(n_samples, len(bin_points))  # Can't sample more than we have
                                
                                # Randomly sample points
                                sampled_indices = np.random.choice(len(bin_points), n_samples, replace=False)
                                sampled_points.append(bin_points[sampled_indices])
                
                # Combine all sampled points
                if sampled_points:
                    mesh_subset = np.vstack(sampled_points)
                    # If we still have too many points, subsample randomly
                    if len(mesh_subset) > max_mesh_points:
                        indices = np.random.choice(len(mesh_subset), max_mesh_points, replace=False)
                        mesh_subset = mesh_subset[indices]
                else:
                    # Fallback to simple random sampling if stratified approach fails
                    indices = np.random.choice(len(mesh_points), max_mesh_points, replace=False)
                    mesh_subset = mesh_points[indices]
            else:
                mesh_subset = mesh_points
                
            # Plot the mesh points if requested
            if show_mesh:
                ax.scatter(mesh_subset[:,0], mesh_subset[:,1], mesh_subset[:,2], 
                          c=mesh_color, marker='.', alpha=mesh_alpha, s=mesh_point_size, 
                          label='Mesh Points')
            
            # Add a surface representation if requested
            if show_surface and len(mesh_subset) > 10:
                try:
                    # Try to create a convex hull of the mesh points for visualization
                    hull = ConvexHull(mesh_subset)
                    simplices = hull.simplices
                    
                    # Create a solid surface representation
                    triangles = []
                    for simplex in simplices:
                        triangles.append(mesh_subset[simplex, :3])
                    
                    # Create a solid surface with the triangles
                    poly = Poly3DCollection(triangles, alpha=surface_alpha, 
                                          edgecolor='gray', linewidth=0.2)
                    poly.set_facecolor(surface_color)  # Explicitly set facecolor
                    ax.add_collection3d(poly)
                    
                    logger.info(f"Created mesh surface with {len(triangles)} triangles")
                except Exception as e:
                    logger.warning(f"Could not create mesh surface using ConvexHull: {e}")
                    
                    # Fallback to alpha shapes or other methods if ConvexHull fails
                    try:
                        # Try to create a Delaunay triangulation instead
                        tri = Delaunay(mesh_subset[:, :2])  # Project to 2D for triangulation
                        triangles = []
                        for simplex in tri.simplices:
                            triangles.append(mesh_subset[simplex])
                        
                        # Create a surface with the triangles
                        poly = Poly3DCollection(triangles, alpha=surface_alpha, facecolor=surface_color, 
                                              edgecolor='gray', linewidth=0.2)
                        poly.set_facecolor(surface_color)  # Explicitly set facecolor
                        ax.add_collection3d(poly)
                        
                        logger.info(f"Created mesh surface with {len(triangles)} triangles using Delaunay")
                    except Exception as e:
                        logger.warning(f"Could not create mesh surface using Delaunay: {e}")
                        
                        # Last resort: create a simple convex hull from the bounding box
                        try:
                            min_c, max_c = bbox
                            corners = np.array([
                                [min_c[0], min_c[1], min_c[2]],
                                [min_c[0], min_c[1], max_c[2]],
                                [min_c[0], max_c[1], min_c[2]],
                                [min_c[0], max_c[1], max_c[2]],
                                [max_c[0], min_c[1], min_c[2]],
                                [max_c[0], min_c[1], max_c[2]],
                                [max_c[0], max_c[1], min_c[2]],
                                [max_c[0], max_c[1], max_c[2]]
                            ])
                            
                            # Define the faces of the box
                            faces = [
                                [corners[0], corners[1], corners[3], corners[2]],  # Left face
                                [corners[4], corners[5], corners[7], corners[6]],  # Right face
                                [corners[0], corners[1], corners[5], corners[4]],  # Bottom face
                                [corners[2], corners[3], corners[7], corners[6]],  # Top face
                                [corners[0], corners[2], corners[6], corners[4]],  # Front face
                                [corners[1], corners[3], corners[7], corners[5]]   # Back face
                            ]
                            
                            # Create a surface with the faces
                            poly = Poly3DCollection(faces, alpha=surface_alpha, facecolor=surface_color, 
                                                  edgecolor='gray', linewidth=0.2)
                            poly.set_facecolor(surface_color)  # Explicitly set facecolor
                            ax.add_collection3d(poly)
                            
                            logger.info("Created simplified mesh surface using bounding box")
                        except Exception as e:
                            logger.warning(f"Could not create simplified mesh surface: {e}")
        except Exception as e:
            logger.warning(f"Error plotting mesh: {e}")
            # Continue with the rest of the visualization
    
    # Plot bounding box as a solid box with transparent faces
    try:
        min_c, max_c = bbox
        corners = np.array([
            [min_c[0], min_c[1], min_c[2]],  # 0: min x, min y, min z
            [min_c[0], min_c[1], max_c[2]],  # 1: min x, min y, max z
            [min_c[0], max_c[1], min_c[2]],  # 2: min x, max y, min z
            [min_c[0], max_c[1], max_c[2]],  # 3: min x, max y, max z
            [max_c[0], min_c[1], min_c[2]],  # 4: max x, min y, min z
            [max_c[0], min_c[1], max_c[2]],  # 5: max x, min y, max z
            [max_c[0], max_c[1], min_c[2]],  # 6: max x, max y, min z
            [max_c[0], max_c[1], max_c[2]]   # 7: max x, max y, max z
        ])
        
        # Define the faces of the bounding box
        faces = [
            [corners[0], corners[1], corners[3], corners[2]],  # Left face
            [corners[4], corners[5], corners[7], corners[6]],  # Right face
            [corners[0], corners[1], corners[5], corners[4]],  # Bottom face
            [corners[2], corners[3], corners[7], corners[6]],  # Top face
            [corners[0], corners[2], corners[6], corners[4]],  # Front face
            [corners[1], corners[3], corners[7], corners[5]]   # Back face
        ]
        
        # Create a very transparent surface for the bounding box
        bbox_poly = Poly3DCollection(faces, alpha=0.05, facecolor='lightblue', 
                                   edgecolor='k', linewidth=lattice_width)
        bbox_poly.set_facecolor('lightblue')  # Explicitly set facecolor
        ax.add_collection3d(bbox_poly)
        
        # Define the edges of the bounding box
        lines = [
            (0,1), (0,2), (0,4),  # From corner 0
            (1,3), (1,5),         # From corner 1
            (2,3), (2,6),         # From corner 2
            (3,7),                # From corner 3
            (4,5), (4,6),         # From corner 4
            (5,7),                # From corner 5
            (6,7)                 # From corner 6
        ]
        
        # Plot the edges of the bounding box
        for i, j in lines:
            ax.plot([corners[i,0], corners[j,0]],
                    [corners[i,1], corners[j,1]],
                    [corners[i,2], corners[j,2]], 'k-', alpha=ffd_alpha, linewidth=lattice_width)
            
        # Add corner labels if in debug mode
        if logger.getEffectiveLevel() <= logging.DEBUG:
            for i, corner in enumerate(corners):
                ax.text(corner[0], corner[1], corner[2], f'{i}', color='k', fontsize=8)
    except Exception as e:
        logger.warning(f"Error plotting bounding box: {e}")
        # Continue with the rest of the visualization
    
    # Try to visualize control lattice structure if points follow a regular pattern
    try:
        # Get unique values of coordinates to infer the lattice structure
        x_unique = np.unique(control_points[:, 0])
        y_unique = np.unique(control_points[:, 1])
        z_unique = np.unique(control_points[:, 2])
        
        nx = len(x_unique)
        ny = len(y_unique)
        nz = len(z_unique)
        
        # Check if this is consistent with total number of points
        if nx * ny * nz == len(control_points):
            logger.info(f"Detected control lattice structure: {nx}x{ny}x{nz}")
            
            # Use a more efficient approach to visualize lattice structure
            # Create a 3D grid of indices
            grid = np.zeros((nx, ny, nz), dtype=int)
            
            # Map each control point to its grid position
            for idx, (x, y, z) in enumerate(control_points):
                i = np.where(np.isclose(x_unique, x))[0][0]
                j = np.where(np.isclose(y_unique, y))[0][0]
                k = np.where(np.isclose(z_unique, z))[0][0]
                grid[i, j, k] = idx
            
            # Plot grid lines along each dimension with improved appearance
            # Along x-direction (varying i)
            for j in range(ny):
                for k in range(nz):
                    indices = [grid[i, j, k] for i in range(nx)]
                    pts = control_points[indices]
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'g-', alpha=0.5, linewidth=1)
            
            # Along y-direction (varying j)
            for i in range(nx):
                for k in range(nz):
                    indices = [grid[i, j, k] for j in range(ny)]
                    pts = control_points[indices]
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'g-', alpha=0.5, linewidth=1)
            
            # Along z-direction (varying k)
            for i in range(nx):
                for j in range(ny):
                    indices = [grid[i, j, k] for k in range(nz)]
                    pts = control_points[indices]
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'g-', alpha=0.5, linewidth=1)
        else:
            # Fallback to the original method if dimensions don't match
            logger.warning(f"Lattice structure dimensions ({nx}x{ny}x{nz}={nx*ny*nz}) don't match point count ({len(control_points)})")
            logger.warning("Using alternative lattice visualization method")
            
            # Try to visualize control lattice structure
            # For each x, y pair, connect points along z
            for i in range(nx):
                for j in range(ny):
                    indices = [idx for idx in range(len(control_points)) 
                              if np.isclose(control_points[idx, 0], x_unique[i])
                              and np.isclose(control_points[idx, 1], y_unique[j])]
                    if len(indices) > 1:
                        # Only plot if we have more than one point
                        pts = control_points[indices]
                        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'g-', alpha=0.3)
                        
            # For each x, z pair, connect points along y
            for i in range(nx):
                for k in range(nz):
                    indices = [idx for idx in range(len(control_points)) 
                              if np.isclose(control_points[idx, 0], x_unique[i])
                              and np.isclose(control_points[idx, 2], z_unique[k])]
                    if len(indices) > 1:
                        pts = control_points[indices]
                        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'g-', alpha=0.3)
                        
            # For each y, z pair, connect points along x
            for j in range(ny):
                for k in range(nz):
                    indices = [idx for idx in range(len(control_points)) 
                              if np.isclose(control_points[idx, 1], y_unique[j])
                              and np.isclose(control_points[idx, 2], z_unique[k])]
                    if len(indices) > 1:
                        pts = control_points[indices]
                        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'g-', alpha=0.3)
    except Exception as e:
        logger.warning(f"Error trying to visualize lattice structure: {e}")
        # Continue with the rest of the visualization
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Make sure the aspect ratio is equal for proper 3D visualization
    ax.set_box_aspect([1, 1, 1])
    
    # Enable proper lighting for 3D effect
    ax.view_init(elev=30, azim=45)  # Default view if none specified
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        if mesh_points is not None and show_mesh:
            ax.set_title('Mesh with FFD Control Lattice')
        else:
            ax.set_title('FFD Control Lattice and Bounding Box')
    
    # Equal aspect ratio for better visualization
    ax.set_box_aspect([1.0, 1.0, 1.0])
    
    # Configure grid
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.3)
    else:
        ax.grid(False)
    
    # Add legend if we have multiple data types
    if mesh_points is not None and show_mesh:
        ax.legend()
    
    # Add dimension info to plot
    bbox_dims = max_c - min_c
    dim_text = f"Dimensions: {bbox_dims[0]:.4f} x {bbox_dims[1]:.4f} x {bbox_dims[2]:.4f}"
    ax.text2D(0.05, 0.95, dim_text, transform=ax.transAxes, fontsize=9)
    
    # Add control point count and scaling info
    cp_text = f"Control points: {len(control_points)}"
    ax.text2D(0.05, 0.92, cp_text, transform=ax.transAxes, fontsize=9)
    
    # Add scaling info if applied
    if needs_scaling or scale_factor is not None:
        scale_text = f"Note: Coordinates scaled by {scale_factor:.1f}x for visualization"
        ax.text2D(0.05, 0.89, scale_text, transform=ax.transAxes, fontsize=9, color='red')
    
    # Save figure if requested
    if save_path:
        try:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(os.path.abspath(save_path))
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        except PermissionError:
            logger.error(f"Permission denied when saving to {save_path}")
            raise IOError(f"Permission denied when saving to {save_path}")
        except Exception as e:
            logger.error(f"Error saving figure: {e}")
            raise
    
    # Set the view angle if specified
    if view_angle is not None:
        elevation, azimuth = view_angle
        ax.view_init(elev=elevation, azim=azimuth)
    else:
        # Set a default view angle that shows the 3D structure well
        ax.view_init(elev=30, azim=45)
    
    # Show the plot
    try:
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.warning(f"Error displaying plot: {e}")
        # This is not a critical error, so we don't raise it


def visualize_mesh_with_patches_pyvista(mesh_data: Any, 
                              save_path: Optional[str] = None,
                              title: str = "Mesh with Patches",
                              point_size: float = 5.0,
                              auto_scale: bool = True,
                              scale_factor: Optional[float] = None,
                              show_solid: bool = True,
                              show_edges: bool = False,
                              show_axes: bool = True,
                              color_by_zone: bool = True,
                              opacity: float = 1.0,
                              bgcolor: str = 'white',
                              window_size: Tuple[int, int] = (1024, 768),
                              max_points_per_zone: int = 5000,  # Max points limit for performance
                              skip_internal_zones: bool = True,  # Skip internal zones for faster processing
                              use_original_faces: bool = False,  # Use original face connectivity data for better edges
                              ffd_control_points: Optional[np.ndarray] = None,  # FFD control points to display
                              ffd_color: str = 'red',  # Color for FFD box
                              ffd_opacity: float = 0.7,  # Opacity for FFD box
                              ffd_point_size: float = 10.0,  # Point size for FFD control points
                              show_ffd_mesh: bool = True,  # Whether to show the FFD wireframe mesh
                              suppress_warnings: bool = True,  # Suppress VTK warnings about degenerate triangles
                              zoom_region: Optional[Tuple[float, float, float, float, float, float]] = None,  # Custom region to zoom into (x_min, x_max, y_min, y_max, z_min, z_max)
                              zoom_factor: float = 1.0,  # Zoom factor (>1 zooms in, <1 zooms out)
                              view_axis: Optional[str] = None,  # Align view with specified axis
                              ffd_box_dims: Optional[List[int]] = None):  # Dimensions of the FFD box [ni, nj, nk]
    """
    Visualize the mesh using PyVista for faster, higher-quality rendering.
    
    Args:
        mesh_data: The mesh object (either FluentMeshReader or meshio.Mesh)
        save_path: Optional path to save the figure
        title: Title for the plot
        point_size: Size of points in the visualization
        auto_scale: Whether to automatically scale very small geometries
        scale_factor: Optional manual scale factor to apply to all coordinates
        show_solid: Whether to show solid surfaces for patches
        show_edges: Whether to show edges on the mesh
        show_axes: Whether to show axes
        color_by_zone: Whether to color surfaces by zone
        opacity: Opacity of the surfaces
        bgcolor: Background color for the visualization
        window_size: Size of the window
        max_points_per_zone: Maximum number of points to render per zone for performance
        skip_internal_zones: Whether to skip internal zones for faster rendering
        use_original_faces: Whether to use original face connectivity data for better edges
        ffd_control_points: Optional array of FFD control points to display
        ffd_color: Color for the FFD control points and grid
        ffd_opacity: Opacity for the FFD control points and grid
        ffd_point_size: Size of FFD control points
        show_ffd_mesh: Whether to show the FFD wireframe mesh
        suppress_warnings: Whether to suppress VTK warnings about degenerate triangles
        zoom_region: Custom region to zoom into (x_min, x_max, y_min, y_max, z_min, z_max)
        zoom_factor: Zoom factor (>1 zooms in, <1 zooms out)
        view_axis: Align view with specified axis ('x', 'y', 'z', '-x', '-y', or '-z')
        ffd_box_dims: Dimensions of the FFD box [ni, nj, nk], must match the number of control points
    """
    logger.info("Visualizing mesh with PyVista for high-performance rendering")
    
    # Suppress VTK warnings about degenerate triangles if requested
    if suppress_warnings:
        # Import the logging module to suppress VTK warnings
        import logging
        # Set logging level for VTK to ERROR to suppress warnings
        logging.getLogger('vtk').setLevel(logging.ERROR)
        # Also disable direct output to console
        import vtk
        vtk.vtkObject.GlobalWarningDisplayOff()
        logger.info("VTK warnings about degenerate triangles have been suppressed")
    
    # Create a new plotter
    plotter = pv.Plotter(window_size=window_size, title=title, notebook=False, off_screen=save_path is not None)
    plotter.background_color = bgcolor
    
    # Add a button to save the current view as PNG
    def save_screenshot_callback():
        # Get the current timestamp for filename
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ffd_view_{timestamp}.png"
        
        # Save screenshot
        plotter.screenshot(filename)
        
        # Just print a message to the console instead of showing on screen
        # This avoids thread safety issues with the plotter
        logger.info(f"Screenshot saved to {filename}")
        print(f"\n*** Screenshot saved to {filename} ***\n")
    
    # Add a button in the toolbar to save the current view
    plotter.add_key_event('s', save_screenshot_callback)  # 's' key for screenshot
    
    # Add a menu item to the top menu for saving the view
    if hasattr(plotter, 'add_menu_item'):  # Check if the method exists (newer PyVista versions)
        try:
            plotter.add_menu_item('Save View as PNG', save_screenshot_callback)
        except Exception as e:
            logger.debug(f"Could not add menu item: {e}")
    
    # Set default orientation: Y up, X right, Z towards viewer
    # Need to set the camera position directly to look along negative Z axis toward origin
    # with Y pointing up and X pointing right
    
    # Get bounding box of data (if any actors exist)
    if len(plotter.renderer.actors) > 0:
        bounds = plotter.bounds
    else:
        # Use default bounds if no actors are available yet
        bounds = [-1, 1, -1, 1, -1, 1]
        
    # Calculate center of the scene
    center = [(bounds[0] + bounds[1])/2, 
              (bounds[2] + bounds[3])/2, 
              (bounds[4] + bounds[5])/2]
    
    # Calculate scene size for proper camera distance
    size = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    
    # Set camera position: Looking from positive Z toward origin
    position = [center[0], center[1], center[2] + 2*size]  # Z position is in front
    focal_point = center  # Look at center
    view_up = [0, 1, 0]   # Y axis is up
    
    # Set camera directly
    plotter.camera_position = [position, focal_point, view_up]
    
    if show_axes:
        # Add axes with proper labels to reflect our orientation
        plotter.add_axes(xlabel='X: Right', ylabel='Y: Up', zlabel='Z: Out')
    
    # Add FFD control points if provided
    if ffd_control_points is not None and len(ffd_control_points) > 0:
        # Convert to pyvista PolyData for visualization
        ffd_points = pv.PolyData(ffd_control_points)
        
        # Add FFD control points as spheres
        plotter.add_points(ffd_points, 
                         color=ffd_color, 
                         point_size=ffd_point_size,
                         render_points_as_spheres=True, 
                         opacity=ffd_opacity)
        
        # Add FFD mesh wireframe if requested
        if show_ffd_mesh and len(ffd_control_points) >= 8:
            # Get dimensions - use provided ffd_box_dims if available, otherwise try to infer
            if ffd_box_dims and len(ffd_box_dims) == 3:
                dims = ffd_box_dims
                logger.info(f"Using provided FFD box dimensions: {dims[0]}x{dims[1]}x{dims[2]}")
            else:
                # Try to infer from number of points
                n_points = ffd_control_points.shape[0]
                if n_points == 64:  # 4x4x4
                    dims = [4, 4, 4]
                elif n_points == 27:  # 3x3x3
                    dims = [3, 3, 3]
                elif n_points == 8:  # 2x2x2
                    dims = [2, 2, 2]
                else:
                    # Default to a cube root approximation if we can't determine
                    dim = int(round(n_points ** (1/3)))
                    dims = [dim, dim, dim]
                    logger.info(f"Inferring FFD dimensions as {dim}x{dim}x{dim} from {n_points} points")
            
            # Get the dimensions
            ni, nj, nk = dims
            
            # Check if the dimensions match the number of control points
            expected_points = ni * nj * nk
            actual_points = len(ffd_control_points)
            
            if expected_points != actual_points:
                logger.warning(f"Dimension mismatch: expected {expected_points} control points from dims {dims}, but got {actual_points}")
                logger.warning(f"Cannot create complete FFD grid, showing outline only")
                
                # Just show the outline
                # Calculate the min/max in each dimension to create a box
                min_x, min_y, min_z = np.min(ffd_control_points, axis=0)
                max_x, max_y, max_z = np.max(ffd_control_points, axis=0)
                
                # Create box corners
                corners = np.array([
                    [min_x, min_y, min_z],
                    [max_x, min_y, min_z],
                    [max_x, max_y, min_z],
                    [min_x, max_y, min_z],
                    [min_x, min_y, max_z],
                    [max_x, min_y, max_z],
                    [max_x, max_y, max_z],
                    [min_x, max_y, max_z],
                ])
                
                # Create edges
                box = pv.Box([min_x, max_x, min_y, max_y, min_z, max_z])
                plotter.add_mesh(box.outline(), color=ffd_color, line_width=3.0, opacity=ffd_opacity)
                logger.info("Added FFD box outline")
            else:
                try:
                    # Try to reshape with the given dimensions
                    ffd_points_3d = ffd_control_points.reshape(ni, nj, nk, 3)
                    logger.info(f"Successfully created FFD grid with dimensions {ni}x{nj}x{nk}")
                    
                    # Create a structured grid for better visualization
                    grid = pv.StructuredGrid()
                    grid.points = ffd_control_points
                    grid.dimensions = [ni, nj, nk]
                    
                    # Add the outline (bounding box)
                    outline = grid.outline()
                    plotter.add_mesh(outline, color=ffd_color, line_width=3.0, opacity=ffd_opacity)
                    
                    # Now add all internal grid lines
                    # Create lines along each axis (i, j, k)
                    for i in range(ni):
                        for j in range(nj):
                            # Line along k-axis
                            points = np.array([ffd_points_3d[i, j, k] for k in range(nk)])
                            poly = pv.Spline(points, n_points=nk)
                            plotter.add_mesh(poly, color=ffd_color, line_width=2.0, opacity=ffd_opacity)
                    
                    for i in range(ni):
                        for k in range(nk):
                            # Line along j-axis
                            points = np.array([ffd_points_3d[i, j, k] for j in range(nj)])
                            poly = pv.Spline(points, n_points=nj)
                            plotter.add_mesh(poly, color=ffd_color, line_width=2.0, opacity=ffd_opacity)
                    
                    for j in range(nj):
                        for k in range(nk):
                            # Line along i-axis
                            points = np.array([ffd_points_3d[i, j, k] for i in range(ni)])
                            poly = pv.Spline(points, n_points=ni)
                            plotter.add_mesh(poly, color=ffd_color, line_width=2.0, opacity=ffd_opacity)
                    
                    logger.info(f"Added FFD control grid with dimensions {dims[0]}x{dims[1]}x{dims[2]}")
                except ValueError as e:
                    logger.warning(f"Failed to create FFD wireframe: {str(e)}, displaying control points only")
                    # Just show the outline
                    # Calculate the min/max in each dimension to create a box
                    min_x, min_y, min_z = np.min(ffd_control_points, axis=0)
                    max_x, max_y, max_z = np.max(ffd_control_points, axis=0)
                    box = pv.Box([min_x, max_x, min_y, max_y, min_z, max_z])
                    plotter.add_mesh(box.outline(), color=ffd_color, line_width=3.0, opacity=ffd_opacity)
    
    # Set up the points - these are the global coordinates
    points = np.array(mesh_data.points)
    logger.info(f"Using {len(points)} points from mesh data")
    
    # Scale the points if auto_scale is True
    if auto_scale or scale_factor is not None:
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        z_range = np.max(points[:, 2]) - np.min(points[:, 2])
        
        # Find max and min ranges
        max_range = max(x_range, y_range, z_range)
        min_range = min(filter(lambda x: x > 0, [x_range, y_range, z_range]))
        
        if max_range / min_range > 10:
            logger.info(f"Geometry has very different scales (max/min: {max_range/min_range:.1f}), applying automatic scaling")
            
            if scale_factor is None:
                scale_factor = 1.0
                logger.info(f"Automatically determined scale factor: {scale_factor}")
            
            # Apply the scale factor
            if scale_factor != 1.0:
                points = points * scale_factor
                logger.info(f"Scaled geometry by factor {scale_factor} for better visualization")
    
    # Get zone information
    zones = mesh_data.get_available_zones()
    logger.info(f"Using all {len(zones)} zones for visualization")
    
    # Define colors for known boundary zones
    boundary_color = 'royalblue'  # Royal blue for external boundaries
    
    # Colors for different zone types
    zone_colors_list = [
        'royalblue', 'darkorange', 'forestgreen', 'crimson', 'purple',
        'gold', 'teal', 'navy', 'maroon', 'olive', 'darkturquoise',
        'firebrick', 'dimgray', 'sienna', 'darkviolet', 'steelblue',
        'darkgreen', 'darkgoldenrod', 'indianred', 'mediumorchid'
    ]
    
    # Count successfully visualized zones
    zone_count = 0
    
    # Fast path: Direct visualization of boundary zones only
    # Identify the boundary zones for the rocket launch pad
    boundary_zones = [name for name in zones if any(pattern in name for pattern in ['triangle_8', 'triangle_10', 'quad_9'])]
    logger.info(f"Found {len(boundary_zones)} boundary zones to visualize")
    
    # Process internal zones if requested
    if skip_internal_zones:
        zones_to_process = boundary_zones
        logger.info(f"Skipping internal zones for faster rendering")
    else:
        zones_to_process = zones
        logger.info(f"Processing all {len(zones)} zones for complete visualization")
    
    # Process each zone
    for zone_name in zones_to_process:
        # Check if this is a boundary zone
        is_boundary = any(name in zone_name for name in ['triangle_8', 'triangle_10', 'quad_9'])
        
        try:
            # Get points for this zone
            zone_points = mesh_data.get_zone_points(zone_name)
            
            # Skip if no points
            if len(zone_points) == 0:
                continue
            
            # Smart downsampling if there are too many points (for performance)
            if len(zone_points) > max_points_per_zone:
                logger.info(f"Smart downsampling zone {zone_name} from {len(zone_points)} to approx. {max_points_per_zone} points")
                # Use voxel-based downsampling to preserve shape better than random sampling
                try:
                    cloud = pv.PolyData(zone_points)
                    # Calculate voxel size based on bounding box and desired point count
                    bounds = cloud.bounds
                    x_range = bounds[1] - bounds[0]
                    y_range = bounds[3] - bounds[2]
                    z_range = bounds[5] - bounds[4]
                    avg_range = (x_range + y_range + z_range) / 3
                    voxel_size = avg_range * np.power(len(zone_points) / max_points_per_zone, 1/3)
                    
                    # Use uniform sampling to preserve shape
                    cloud_down = cloud.uniform_remesh(voxel_size)
                    if cloud_down.n_points > 0:
                        # If successful, update zone_points
                        zone_points = cloud_down.points
                        logger.info(f"Shape-preserving downsampling: {len(zone_points)} points")
                    else:
                        # Fallback to random sampling if uniform remeshing fails
                        indices = np.random.choice(len(zone_points), max_points_per_zone, replace=False)
                        zone_points = zone_points[indices]
                        logger.info(f"Fallback to random downsampling: {len(zone_points)} points")
                except Exception as e:
                    # Fallback to random sampling if uniform remeshing fails
                    logger.debug(f"Uniform remeshing failed, using random sampling: {e}")
                    indices = np.random.choice(len(zone_points), max_points_per_zone, replace=False)
                    zone_points = zone_points[indices]
            
            # Determine color for this zone
            if is_boundary:
                color = boundary_color
            elif color_by_zone:
                color = zone_colors_list[zone_count % len(zone_colors_list)]
            else:
                color = 'gray'
            
            # Only use surface reconstruction for boundary zones
            # Use more aggressive surface reconstruction for boundary zones to avoid empty cells
            if show_solid and is_boundary:
                # Create point cloud for surface reconstruction
                cloud = pv.PolyData(zone_points)
                
                # FIRST APPROACH: If use_original_faces is enabled, try using the original mesh faces directly
                if use_original_faces:
                    try:
                        # Get the original faces for this zone
                        faces = mesh_data.get_zone_faces(zone_name)
                        faces_array = np.array([f for f in faces if len(f) >= 3])
                        
                        if len(faces_array) > 0:
                            # For triangular faces, use make_tri_mesh
                            # For specific boundary zones, we know they are triangular faces
                            if is_boundary and len(faces_array[0]) == 3:
                                # Create a mapping from global to local indices
                                point_set = set()
                                for face in faces_array:
                                    for idx in face:
                                        point_set.add(idx)
                                        
                                # Build a global to local index mapping
                                global_to_local = {}
                                local_points = []
                                for i, idx in enumerate(sorted(point_set)):
                                    global_to_local[idx] = i
                                    # Get the actual point coordinates
                                    point_idx = np.where(mesh_data.zones[zone_name] == idx)[0]
                                    if len(point_idx) > 0:
                                        local_idx = point_idx[0]
                                        if local_idx < len(zone_points):
                                            local_points.append(zone_points[local_idx])
                                        else:
                                            # Use original mesh points if local index out of range
                                            local_points.append(mesh_data.points[idx])
                                    else:
                                        # Fallback if point not found in zone
                                        local_points.append(mesh_data.points[idx])
                                
                                # Convert faces to local indices
                                local_faces = []
                                for face in faces_array:
                                    try:
                                        local_face = [global_to_local[idx] for idx in face]
                                        local_faces.append(local_face)
                                    except KeyError:
                                        # Skip faces with missing indices
                                        continue
                                
                                if len(local_faces) > 0 and len(local_points) > 0:
                                    # Create a proper triangular mesh
                                    tri_mesh = pv.make_tri_mesh(np.array(local_points), np.array(local_faces))
                                    if tri_mesh.n_cells > 0:
                                        plotter.add_mesh(tri_mesh, color=color, opacity=opacity,
                                                       show_edges=show_edges, smooth_shading=True)
                                        zone_count += 1
                                        continue
                            
                            # For quads and mixed faces, build a more general polydata
                            # This approach works well for the main body (zone_quad_9)
                            elif 'quad_9' in zone_name:
                                # Directly create a polydata from all points and faces
                                # First, extract all unique point indices used in faces
                                point_indices = set()
                                valid_faces = []
                                for face in faces:
                                    if len(face) >= 3:
                                        valid_faces.append(face)
                                        for idx in face:
                                            point_indices.add(idx)
                                
                                # Create a mapping from global to local indices
                                global_to_local = {}
                                for i, idx in enumerate(sorted(point_indices)):
                                    global_to_local[idx] = i
                                
                                # Create points array using global point indices
                                mesh_points = np.array([mesh_data.points[idx] for idx in sorted(point_indices)])
                                
                                # Convert faces to local indices and to vtk format
                                vtk_faces = []
                                for face in valid_faces:
                                    if all(idx in global_to_local for idx in face):
                                        vtk_face = [len(face)] + [global_to_local[idx] for idx in face]
                                        vtk_faces.extend(vtk_face)
                                
                                if len(vtk_faces) > 0:
                                    cells = np.array(vtk_faces)
                                    surface = pv.PolyData(mesh_points, cells)
                                    plotter.add_mesh(surface, color=color, opacity=opacity,
                                                   show_edges=show_edges, smooth_shading=True)
                                    zone_count += 1
                                    continue
                    except Exception as e:
                        logger.debug(f"Original face approach failed for {zone_name}: {e}")
                
                # SECOND APPROACH: Try advanced surface reconstruction methods
                # that better preserve sharp features
                try:
                    # Try surface reconstruction with Ball Pivoting Algorithm (BPA)
                    # which is excellent at preserving sharp edges
                    # Calculate average nearest neighbor distance for radius
                    if len(zone_points) > 10:
                        kdtree = cloud.cell_centers().extract_points().perform_point_operations("kd-tree")
                        radius_estimate = kdtree.n_nearest_points(5).compute_average_distance() * 2.0
                        
                        surf = cloud.reconstruct_surface(radius=radius_estimate)
                        
                        if surf.n_cells > 0:
                            # Apply smoothing that preserves sharp edges
                            surf = surf.smooth(n_iter=5, feature_smoothing=True, edge_angle=30)
                            plotter.add_mesh(surf, color=color, opacity=opacity,
                                          show_edges=False, smooth_shading=True)
                            zone_count += 1
                            continue
                except Exception as e:
                    logger.debug(f"Advanced surface reconstruction failed for {zone_name}: {e}")
                
                # If advanced method fails, try these alternatives in order
                
                # 1. Use Power Crust algorithm for sharp features (via vtk.js PowerCrustSurfaceReconstruction)
                try:
                    # First try with a more direct surface reconstruction approach
                    # Generate a clean triangulated surface using Poisson surface reconstruction
                    # which is excellent for preserving sharp features
                    surf = None
                    
                    # For the specific zones we know are the important boundary parts:
                    if 'triangle_8' in zone_name or 'triangle_10' in zone_name:
                        # Use specialized parameters for triangle zones (often the deflectors)
                        # Create a cleaner uniform sampling first
                        uniform = cloud.uniform_remesh(0.1)  # Smaller voxel size for sharper features
                        surf = uniform.reconstruct_surface()
                    elif 'quad_9' in zone_name:
                        # For the main body zone - try to get sharp edges by first removing noise
                        filtered = cloud.distance_filtering(0.01, invert=True)  # Remove outliers
                        # Clean any duplicate points
                        filtered = filtered.clean(tolerance=1e-5)
                        surf = filtered.reconstruct_surface()
                    
                    # If surface reconstruction failed, fall back to Alpha Shapes 
                    if surf is None or surf.n_cells == 0:
                        # First clean the point cloud of any duplicates
                        clean_cloud = cloud.clean(tolerance=1e-6)
                        
                        # For triangle zones, they often need specialized handling
                        if 'triangle_8' in zone_name or 'triangle_10' in zone_name:
                            # For deflector parts, use a different approach to fill gaps
                            # Add points to ensure connectivity
                            cloud_size = len(clean_cloud.points)
                            if cloud_size > 10:  # Only process meaningful point clouds
                                # Calculate model dimensions
                                bounds = clean_cloud.bounds
                                bbox_size = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
                                bbox_diag = np.sqrt(np.sum(bbox_size**2))
                                
                                # Use a denser point cloud for better surface coverage
                                # This helps fill gaps between points 
                                cloud_with_normals = clean_cloud.compute_normals()
                                dense_cloud = cloud_with_normals.glyph(geom=pv.Sphere(radius=bbox_diag*0.005, phi_resolution=8, theta_resolution=8),
                                                             scale=False, orient=False)
                                
                                # Use smaller alpha value for better detail preservation
                                alpha = bbox_diag * 0.01
                                surf = dense_cloud.delaunay_3d(alpha=alpha).extract_surface()
                            else:
                                # Use convex hull for very small point clouds as a fallback
                                surf = clean_cloud.delaunay_3d().extract_surface()
                        else:
                            # For the main body (quad_9), use standard alpha shapes with optimized alpha
                            # Compute point cloud density to estimate a good alpha value
                            bounds = clean_cloud.bounds
                            bbox_size = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
                            
                            # Calculate volume for density estimate
                            volume = np.prod(bbox_size) 
                            point_density = len(clean_cloud.points) / max(volume, 1e-10)
                            
                            # Use an alpha that balances detail and filling gaps
                            # Smaller alpha preserves more detail but may leave gaps
                            # Larger alpha fills more gaps but may lose detail
                            alpha = 1.5 / np.cbrt(point_density) * 3.0
                            alpha = min(max(alpha, 0.1), 5.0)  # Reasonable bounds
                            
                            logger.debug(f"Using alpha={alpha:.2f} for zone {zone_name}")
                            
                            # First attempt with normal alpha shape
                            surf = clean_cloud.delaunay_3d(alpha=alpha).extract_surface()
                            
                            # If result has too few triangles, try a different approach
                            if surf.n_cells < 100 and len(clean_cloud.points) > 1000:
                                # Create a denser uniform sampling to ensure surface continuity
                                voxel_size = np.min(bbox_size) * 0.02
                                uniform = clean_cloud.uniform_remesh(voxel_size)
                                surf = uniform.delaunay_3d(alpha=alpha*1.2).extract_surface()
                    
                    if surf.n_cells > 0:
                        # Post-process to ensure watertight mesh without gaps
                        try:
                            # First, apply minimal smoothing to remove artifacts while keeping sharp edges
                            surf = surf.smooth(n_iter=3, feature_smoothing=True, edge_angle=30)
                            
                            # Fill holes to ensure no discontinuities
                            # Calculate appropriate hole size based on model dimensions
                            bounds = surf.bounds
                            diagonal = np.sqrt(np.sum((np.array(bounds[1::2]) - np.array(bounds[::2]))**2))
                            max_hole_size = diagonal * 0.05  # Allow filling holes up to 5% of model size
                            
                            # Use fill_holes to close gaps in the surface
                            # This is crucial for eliminating discontinuities/empty triangles
                            watertight = surf.fill_holes(hole_size=max_hole_size)
                            
                            # If the model has very few triangles, use a fallback approach
                            if watertight.n_cells < 50 and len(cloud.points) > 200:
                                # Create a simplified but complete version using convex hull
                                # This guarantees no empty regions at the cost of some detail
                                clipped_hull = cloud.delaunay_3d().extract_surface()
                                plotter.add_mesh(clipped_hull, color=color, opacity=opacity,
                                               show_edges=False, smooth_shading=True)
                            else:
                                # Use the hole-filled surface for rendering
                                plotter.add_mesh(watertight, color=color, opacity=opacity,
                                               show_edges=False, smooth_shading=True)
                        except Exception as e:
                            # In case of failure with the post-processing, use the original surface
                            logger.debug(f"Post-processing failed for {zone_name}, using original surface: {e}")
                            plotter.add_mesh(surf, color=color, opacity=opacity,
                                           show_edges=False, smooth_shading=True)
                        
                        zone_count += 1
                        continue
                except Exception as e:
                    logger.debug(f"Alpha shape failed for {zone_name}: {e}")
                
                # 2. For planar-like surfaces, try 2D Delaunay with best-fit plane
                try:
                    # Find principal components to determine if surface is planar-like
                    cloud_np = np.array(zone_points)
                    mean = np.mean(cloud_np, axis=0)
                    centered = cloud_np - mean
                    
                    # Get principal components
                    u, s, vh = np.linalg.svd(centered, full_matrices=False)
                    
                    # If the smallest component is much smaller than others, it's planar-like
                    if s[2] < s[0] * 0.2:
                        # Project points to best-fit plane
                        normal = vh[2, :]
                        projected = pv.PolyData(centered.dot(vh.T[:, :2]))
                        surf2d = projected.delaunay_2d()
                        
                        # Map the 2D triangulation back to 3D
                        points2d = np.hstack((projected.points, np.zeros((projected.n_points, 1))))
                        points3d = points2d.dot(vh) + mean
                        surf = pv.make_tri_mesh(points3d, surf2d.faces.reshape(-1, 4)[:, 1:])                
                        
                        plotter.add_mesh(surf, color=color, opacity=opacity,
                                       show_edges=show_edges, smooth_shading=True)
                        zone_count += 1
                        continue
                except Exception as e:
                    logger.debug(f"Best-fit plane triangulation failed for {zone_name}: {e}")
                
                # 3. Try using a guaranteed gap-free approach
                try:
                    # Create a clean, high-quality point cloud to ensure no degenerate triangles
                    # First, remove duplicate points and noise
                    clean_cloud = cloud.clean(tolerance=1e-6)  # Remove duplicates
                    
                    # Remove outliers that could cause degenerate triangles
                    if len(clean_cloud.points) > 20:
                        try:
                            # Statistical outlier removal to eliminate noise
                            clean_cloud = clean_cloud.remove_outliers(nb_points=10)
                        except Exception:
                            # Fallback if statistical removal fails
                            pass
                    
                    # Compute a suitable radius for point densification
                    bounds = clean_cloud.bounds
                    bbox_size = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
                    min_dim = np.min(bbox_size[bbox_size > 0])
                    radius = min_dim * 0.01  # Use 1% of smallest dimension as radius
                    
                    # Create a denser point cloud by adding points
                    # Use smaller spheres with many points to fill any gaps
                    dense_cloud = clean_cloud.glyph(
                        geom=pv.Sphere(radius=radius, phi_resolution=8, theta_resolution=8),
                        scale=False, orient=False)
                    
                    # For very small or sparse point clouds, use an even denser approach
                    if len(clean_cloud.points) < 1000:
                        # Add even more points to ensure no gaps
                        denser_cloud = dense_cloud.glyph(
                            geom=pv.Sphere(radius=radius*0.5, phi_resolution=6, theta_resolution=6),
                            scale=False, orient=False)
                        surf = denser_cloud.delaunay_3d().extract_surface()
                    else:
                        # Use standard approach for larger clouds
                        surf = dense_cloud.delaunay_3d().extract_surface()
                    
                    # If we got a valid surface, add it to the visualization
                    if surf.n_cells > 0:
                        # Apply minimal smoothing to preserve shape but remove artifacts
                        surf = surf.smooth(n_iter=2, feature_smoothing=True, edge_angle=30)
                        
                        # Convert to an explicit triangle mesh to ensure all triangles are rendered
                        # This is similar to what ParaView does internally
                        tri_filter = surf.triangulate()
                        
                        # Add to the scene with ParaView-style rendering parameters
                        plotter.add_mesh(tri_filter, color=color, opacity=opacity,
                                       show_edges=False,  # No edges for smoother appearance
                                       smooth_shading=True,  # Use smooth Phong shading like ParaView
                                       specular=0.5,  # Add specularity like ParaView
                                       ambient=0.3,  # Increase ambient lighting to see all surfaces
                                       diffuse=0.7)  # Adjust diffuse component
                        zone_count += 1
                    else:
                        # As a last resort, use convex hull which guarantees no gaps
                        hull = clean_cloud.delaunay_3d().extract_surface()
                        
                        # Convert to explicit triangles and apply ParaView-style rendering
                        tri_hull = hull.triangulate()
                        plotter.add_mesh(tri_hull, color=color, opacity=opacity,
                                       show_edges=False, smooth_shading=True,
                                       specular=0.5, ambient=0.3, diffuse=0.7)
                        logger.info(f"Using convex hull for {zone_name} to avoid empty cells")
                        zone_count += 1
                except Exception as e:
                    logger.debug(f"Advanced surface methods failed for {zone_name}: {e}")
                    
                    # As an absolute last resort, create a convex hull from the original cloud
                    try:
                        hull = pv.wrap(zone_points).delaunay_3d().extract_surface()
                        
                        # Apply consistent ParaView-style rendering
                        tri_hull = hull.triangulate()
                        plotter.add_mesh(tri_hull, color=color, opacity=opacity,
                                      show_edges=False, smooth_shading=True,
                                      specular=0.5, ambient=0.3, diffuse=0.7)
                        logger.info(f"Using fallback convex hull for {zone_name} to avoid empty cells")
                        zone_count += 1
                    except Exception as e2:
                        # If even the convex hull fails, use points as absolute last resort
                        logger.debug(f"All surface methods failed for {zone_name}: {e2}")
                        plotter.add_points(cloud, color=color, point_size=point_size*2.0, 
                                         render_points_as_spheres=True, opacity=opacity)
                        zone_count += 1
            else:
                # Fast point visualization for internal zones or when solid is not requested
                # Use a simplified point cloud with larger points for better performance
                plotter.add_points(zone_points, color=color, point_size=point_size, 
                                 render_points_as_spheres=True, opacity=opacity)
                zone_count += 1
        except Exception as e:
            logger.warning(f"Error processing zone {zone_name}: {e}")
    
    # Add axes if requested
    if show_axes:
        plotter.add_axes()
        
    # Add a bounding box
    plotter.add_bounding_box()
    
    # Set a good view
    plotter.view_isometric()
    
    # Apply view axis orientation if specified
    if view_axis is not None:
        logger.info(f"Setting view orientation to {view_axis} axis")
        
        # Get the center of the scene
        if len(plotter.renderer.actors) > 0:
            bounds = plotter.bounds
            center = [(bounds[0] + bounds[1])/2, 
                    (bounds[2] + bounds[3])/2, 
                    (bounds[4] + bounds[5])/2]
            size = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        else:
            center = [0, 0, 0]
            size = 1
            
        camera_dist = 2 * size  # Distance to place camera from center
        
        if view_axis == 'x':
            # View from positive X axis (looking towards negative X)
            position = [center[0] + camera_dist, center[1], center[2]]
            focal_point = center
            view_up = [0, 1, 0]  # Y is up
        elif view_axis == '-x':
            # View from negative X axis (looking towards positive X)
            position = [center[0] - camera_dist, center[1], center[2]]
            focal_point = center
            view_up = [0, 1, 0]  # Y is up
        elif view_axis == 'y':
            # View from positive Y axis (looking towards negative Y)
            position = [center[0], center[1] + camera_dist, center[2]]
            focal_point = center
            view_up = [0, 0, 1]  # Z is up when looking along Y
        elif view_axis == '-y':
            # View from negative Y axis (looking towards positive Y)
            position = [center[0], center[1] - camera_dist, center[2]]
            focal_point = center
            view_up = [0, 0, 1]  # Z is up when looking along Y
        elif view_axis == 'z':
            # View from positive Z axis (looking towards negative Z)
            position = [center[0], center[1], center[2] + camera_dist]
            focal_point = center
            view_up = [0, 1, 0]  # Y is up
        elif view_axis == '-z':
            # View from negative Z axis (looking towards positive Z)
            position = [center[0], center[1], center[2] - camera_dist]
            focal_point = center
            view_up = [0, 1, 0]  # Y is up
            
        # Set camera directly
        plotter.camera_position = [position, focal_point, view_up]
    
    # Apply custom zoom to region if specified
    if zoom_region is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = zoom_region
        logger.info(f"Zooming to region: X [{x_min}:{x_max}], Y [{y_min}:{y_max}], Z [{z_min}:{z_max}]")
        
        # Create a box to represent the region
        box_points = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max]
        ])
        box = pv.PolyData(box_points)
        
        # Focus camera on this region
        plotter.reset_camera(bounds=[x_min, x_max, y_min, y_max, z_min, z_max])
        # Add some padding for better visualization
        plotter.camera.zoom(0.9)  # 10% padding
    
    # Apply zoom factor if specified and not 1.0
    if zoom_factor != 1.0:
        logger.info(f"Applying zoom factor: {zoom_factor}")
        # Zoom in (factor > 1) or out (factor < 1)
        plotter.camera.zoom(1.0 / zoom_factor)  # Inverse because zoom works opposite to intuition
    
    # Save or show the visualization
    if save_path:
        plotter.screenshot(save_path, transparent_background=(bgcolor == 'white'))
        logger.info(f"Saved mesh visualization to {save_path}")
    else:
        # Show the visualization
        logger.info(f"Showing interactive visualization with {zone_count} zones")
        plotter.show()
        
    return plotter

def visualize_mesh_with_patches(mesh_data: Any, 
                             save_path: Optional[str] = None,
                             fig_size: Tuple[int, int] = (12, 10),
                             dpi: int = 300,
                             title: str = "Mesh with Patches",
                             point_size: float = 2.0,
                             auto_scale: bool = True,
                             scale_factor: Optional[float] = None,
                             view_angle: Optional[Tuple[float, float]] = None,
                             show_solid: bool = True,
                             alpha: float = 1.0,  # Fully opaque for solid appearance
                             max_faces_per_zone: int = 500000,  # Further increased for better geometry detail
                             max_points_per_patch: int = 50000,  # Increased for better coverage
                             use_multiple_projections: bool = True,
                             preserve_structure: bool = True,
                             remove_artifacts: bool = True,
                             shading_mode: str = 'gouraud',  # Added parameter for shading
                             use_pyvista: bool = True):  # Use PyVista by default for faster visualization
    """
    Visualize the initial mesh geometry with all patches in different colors.
    
    Args:
        mesh_data: The mesh object (either FluentMeshReader or meshio.Mesh)
        save_path: Optional path to save the figure
        fig_size: Size of the figure as (width, height) in inches
        dpi: Resolution of the figure in dots per inch
        title: Title for the plot
        point_size: Size of points in the visualization
        auto_scale: Whether to automatically scale very small geometries
        scale_factor: Optional manual scale factor to apply to all coordinates
        view_angle: Optional tuple of (elevation, azimuth) angles for the view
        show_solid: Whether to show solid surfaces for patches
        alpha: Alpha value for solid surfaces
        max_points_per_patch: Maximum number of points to display per patch
        use_pyvista: Whether to use the faster PyVista renderer (True) or Matplotlib (False)
        
    Raises:
        ValueError: If the mesh has no points or patches
        IOError: If the figure cannot be saved to the specified path
    """
    logger.info("Visualizing mesh with patches")
    
    # Check if we have a valid mesh
    if not hasattr(mesh_data, 'points') or len(mesh_data.points) == 0:
        raise ValueError("Mesh has no points")
    
    # Get all points and determine if we need to scale
    all_points = np.array(mesh_data.points)
    
    # Get bounding box
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    bbox_size = max_coords - min_coords
    bbox_max_size = np.max(bbox_size)
    bbox_min_size = np.min(bbox_size)
    
    # Determine if we need to scale (if the geometry is very small or has very different scales)
    needs_scaling = False
    if auto_scale:
        if bbox_max_size < 0.1:  # Very small geometry
            needs_scaling = True
            logger.info(f"Geometry is very small (max size: {bbox_max_size:.6f}), applying automatic scaling")
        elif bbox_max_size / max(bbox_min_size, 1e-10) > 1000:  # Very different scales
            needs_scaling = True
            logger.info(f"Geometry has very different scales (max/min: {bbox_max_size/max(bbox_min_size, 1e-10):.1f}), applying automatic scaling")
    
    # Apply scaling if needed or if a manual scale factor is provided
    if needs_scaling or scale_factor is not None:
        if scale_factor is None:
            # Determine automatic scale factor to make the geometry a reasonable size
            if bbox_max_size < 0.1:
                scale_factor = 1.0 / bbox_max_size  # Scale to make max dimension ~1.0
            else:
                scale_factor = 1.0  # Default
            logger.info(f"Automatically determined scale factor: {scale_factor:.1f}")
        
        # Apply scaling to all coordinates
        all_points = all_points * scale_factor
        min_coords = min_coords * scale_factor
        max_coords = max_coords * scale_factor
        
        logger.info(f"Scaled geometry by factor {scale_factor:.1f} for better visualization")
    
    # Create figure and 3D axis
    try:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
    except Exception as e:
        logger.error(f"Error creating figure: {e}")
        raise
    
    # Get all available patches/zones
    patches = {}
    
    # Handle different mesh types
    from mesh_readers.fluent_reader import FluentMeshReader
    
    if isinstance(mesh_data, FluentMeshReader):
        # Get all available zones from the mesh
        all_zones = mesh_data.get_available_zones()
        
        # Convert to dictionary if it's returned as a list
        if isinstance(all_zones, list):
            all_zones = {zone_name: 1 for zone_name in all_zones}
            
        # Use all zones for a complete representation
        patches = all_zones
        logger.info(f"Using all {len(patches)} zones for visualization")
        
        # We'll handle the solid appearance through rendering parameters later
        
        # Generate a list of distinct colors for patches
        colors = list(mcolors.TABLEAU_COLORS.values())
        if len(patches) > len(colors):
            # Add more colors if needed
            colors.extend(random.sample(list(mcolors.CSS4_COLORS.values()), 
                                      len(patches) - len(colors)))
        
        # Plot each patch with a different color
        for i, patch_name in enumerate(patches):
            try:
                patch_points = mesh_data.get_zone_points(patch_name)
                
                # Scale points if needed
                if needs_scaling or scale_factor is not None:
                    patch_points = patch_points * scale_factor
                
                # Subsample points if there are too many
                if len(patch_points) > max_points_per_patch:
                    indices = np.linspace(0, len(patch_points)-1, max_points_per_patch, dtype=int)
                    patch_points = patch_points[indices]
                
                # Get color for this patch
                color = colors[i % len(colors)]
                
                # Only show points if we're not showing solid surfaces
                if not show_solid:
                    ax.scatter(patch_points[:,0], patch_points[:,1], patch_points[:,2], 
                              c=[color], marker='.', label=f"{patch_name} ({len(patch_points)} points)", 
                              s=point_size, alpha=0.8)
                else:
                    # Just add a label for the legend without plotting any points
                    ax.plot([], [], [], '.', c=color, markersize=5, alpha=0, label=f"{patch_name}")
                
                # For complex geometries like rocket launch pads, use actual face connectivity
                if show_solid:
                    # Check if the mesh reader has face connectivity information
                    has_face_connectivity = hasattr(mesh_data, 'get_zone_faces') and callable(getattr(mesh_data, 'get_zone_faces'))
                    
                    surface_created = False
                    
                    if has_face_connectivity:
                        try:
                            # Get faces for this zone
                            faces = mesh_data.get_zone_faces(patch_name)
                            
                            if faces is not None and len(faces) > 0:
                                logger.info(f"Found {len(faces)} faces for zone '{patch_name}'")
                                
                                # If we have too many faces, sample a subset
                                if len(faces) > max_faces_per_zone:
                                    logger.info(f"Limiting display to {max_faces_per_zone} faces out of {len(faces)}")
                                    # Sample faces uniformly
                                    indices = np.linspace(0, len(faces)-1, max_faces_per_zone, dtype=int)
                                    faces = [faces[i] for i in indices]
                                
                                # Convert faces to triangles for Poly3DCollection
                                triangulated_faces = []
                                
                                for face in faces:
                                    try:
                                        # Convert face indices to actual 3D coordinates
                                        verts = [patch_points[i] for i in face]
                                        
                                        # Skip problematic faces
                                        if len(verts) < 3:
                                            continue
                                            
                                        # For quads, split into two triangles
                                        if len(verts) == 4:
                                            triangulated_faces.append([verts[0], verts[1], verts[2]])
                                            triangulated_faces.append([verts[0], verts[2], verts[3]])
                                        elif len(verts) == 3:  # Triangle
                                            triangulated_faces.append(verts)
                                        else:  # N-gon
                                            # Simple fan triangulation from first vertex
                                            for i in range(1, len(verts) - 1):
                                                triangulated_faces.append([verts[0], verts[i], verts[i+1]])
                                    except (IndexError, TypeError):
                                        # Skip faces with invalid indices
                                        continue
                                
                                # Use all triangulated faces with minimal filtering
                                filtered_faces = []
                                
                                # Use a very permissive edge length filter - 50% of the domain size
                                max_edge_length = np.max(np.ptp(all_points, axis=0)) * 0.5
                                
                                for face in triangulated_faces:
                                    if len(face) == 3:
                                        # Calculate the longest edge
                                        edges = [
                                            np.linalg.norm(np.array(face[0]) - np.array(face[1])),
                                            np.linalg.norm(np.array(face[1]) - np.array(face[2])),
                                            np.linalg.norm(np.array(face[2]) - np.array(face[0]))
                                        ]
                                        
                                        # Include almost all faces, only filter extreme outliers
                                        if max(edges) < max_edge_length:
                                            filtered_faces.append(face)
                                
                                # Create a solid surface from the filtered faces
                                if filtered_faces:
                                    # Create a Poly3DCollection with enhanced rendering settings
                                    poly = Poly3DCollection(filtered_faces, 
                                                         alpha=alpha,            # Use requested opacity (default 1.0)
                                                         facecolor=color,        # Use zone color
                                                         edgecolor='k',          # Black edges for better definition
                                                         linewidth=0.01,         # Very thin edges
                                                         antialiased=True)       # Smooth appearance
                                    
                                    # Enable proper 3D lighting effects
                                    poly.set_sort_zpos(np.mean([np.mean(f, axis=0)[2] for f in filtered_faces]))
                                    
                                    # Add to plot
                                    ax.add_collection3d(poly)
                                    logger.info(f"Created surface for zone '{patch_name}' using {len(filtered_faces)} triangulated faces")
                                    surface_created = True
                                else:
                                    logger.info(f"No suitable faces found for zone '{patch_name}' after filtering")
                            else:
                                logger.debug(f"No faces found for zone '{patch_name}'")
                        except Exception as e:
                            logger.warning(f"Error creating surface from face connectivity: {e}")
                            # Continue with fallback methods
                    
                    # Backup approaches if no face connectivity or if it failed
                    if not surface_created:
                        # Use a subset of points for performance if there are too many
                        max_vis_points = min(len(patch_points), 5000)
                        if len(patch_points) > max_vis_points:
                            # Use stratified sampling to preserve structure
                            if preserve_structure:
                                # Divide space into bins
                                min_coords = np.min(patch_points, axis=0)
                                max_coords = np.max(patch_points, axis=0)
                                bins = max(2, int(np.cbrt(max_vis_points / 10)))
                                
                                # Sample from each bin
                                sampled_points = []
                                try:
                                    H, edges = np.histogramdd(patch_points, bins=[bins, bins, bins],
                                                            range=[(min_coords[0], max_coords[0]),
                                                                  (min_coords[1], max_coords[1]),
                                                                  (min_coords[2], max_coords[2])])
                                    
                                    for i in range(bins):
                                        for j in range(bins):
                                            for k in range(bins):
                                                # Find points in this bin
                                                bin_mask = ((patch_points[:, 0] >= edges[0][i]) & (patch_points[:, 0] < edges[0][i+1]) &
                                                          (patch_points[:, 1] >= edges[1][j]) & (patch_points[:, 1] < edges[1][j+1]) &
                                                          (patch_points[:, 2] >= edges[2][k]) & (patch_points[:, 2] < edges[2][k+1]))
                                                bin_points = patch_points[bin_mask]
                                                
                                                if len(bin_points) > 0:
                                                    # Sample more points from populated bins
                                                    n_samples = max(1, int(len(bin_points) * max_vis_points / len(patch_points)))
                                                    n_samples = min(n_samples, len(bin_points))
                                                    sampled_indices = np.random.choice(len(bin_points), n_samples, replace=False)
                                                    sampled_points.append(bin_points[sampled_indices])
                                    
                                    if sampled_points:
                                        vis_points = np.vstack(sampled_points)
                                        if len(vis_points) > max_vis_points:
                                            indices = np.random.choice(len(vis_points), max_vis_points, replace=False)
                                            vis_points = vis_points[indices]
                                    else:
                                        # Fallback to random sampling
                                        indices = np.random.choice(len(patch_points), max_vis_points, replace=False)
                                        vis_points = patch_points[indices]
                                except Exception as e:
                                    logger.debug(f"Error in stratified sampling: {e}")
                                    # Fallback to random sampling
                                    indices = np.random.choice(len(patch_points), max_vis_points, replace=False)
                                    vis_points = patch_points[indices]
                            else:
                                # Simple random sampling
                                indices = np.random.choice(len(patch_points), max_vis_points, replace=False)
                                vis_points = patch_points[indices]
                        else:
                            vis_points = patch_points
                        
                        # Approach 2: Multiple projection planes (better for complex geometries)
                        if use_multiple_projections:
                            # Handle zone type recognition explicitly - useful for rocket launch pad geometry
                            is_boundary_zone = False
                            is_internal_zone = False
                            
                            # Identify external boundaries and internal zones by name pattern
                            if any(x in patch_name for x in ['triangle_8', 'triangle_10', 'quad_9']):
                                # These are external boundary zones like walls and inlets/outlets
                                is_boundary_zone = True
                                # Override the color to ensure consistent appearance for all boundary zones
                                # Use a blue-gray color for all external boundaries
                                color = '#4682B4'  # Steel blue
                            elif any(x in patch_name for x in ['quad_0', 'quad_1', 'quad_2', 'quad_3', 'quad_4', 'quad_5', 'quad_6', 'quad_7']):
                                # These are internal zones that should be hidden
                                is_internal_zone = True
                            
                            # Skip internal zones entirely if not boundary zones
                            if is_internal_zone and not is_boundary_zone:
                                # Don't process internal geometry at all
                                continue
                                
                            try:
                                # For each principal plane, create a projection and triangulation
                                # Using all possible projections to ensure complete geometry
                                for projection_dims in [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]:
                                    try:
                                        # Get the two dimensions for projection
                                        points_2d = vis_points[:, projection_dims]
                                        
                                        # Skip if we don't have enough unique points
                                        if len(np.unique(points_2d, axis=0)) <= 3:
                                            continue
                                            
                                        # Create Delaunay triangulation
                                        tri = Delaunay(points_2d)
                                        
                                        # Create triangles in 3D space
                                        triangles = []
                                        for simplex in tri.simplices:
                                            triangles.append(vis_points[simplex])
                                        
                                        # Include all triangles for complete geometry
                                        # Only filter out extremely unreasonable triangles
                                        filtered_triangles = []
                                        # Very permissive max edge length
                                        max_edge_length = np.max(np.ptp(vis_points, axis=0)) * 1.0
                                        
                                        for triangle in triangles:
                                            # Calculate edge lengths
                                            a = np.linalg.norm(triangle[0] - triangle[1])
                                            b = np.linalg.norm(triangle[1] - triangle[2])
                                            c = np.linalg.norm(triangle[2] - triangle[0])
                                            
                                            # Only filter out truly excessive triangles to get full coverage
                                            if max(a, b, c) < max_edge_length:
                                                filtered_triangles.append(triangle)
                                        
                                        # Create a surface if we have triangles
                                        if filtered_triangles:
                                            # Determine if this is boundary or internal zone
                                            is_boundary = False
                                            is_internal = False
                                            
                                            # Check zone type based on name
                                            if any(x in patch_name.lower() for x in ['triangle_8', 'triangle_10', 'quad_9']):
                                                # External boundary zone - fully solid
                                                is_boundary = True
                                                # Use solid blue for all boundary faces
                                                color = '#1E90FF'  # Dodger Blue
                                                zone_alpha = 1.0    # Fully opaque
                                                edge_color = None   # No edges
                                                line_width = 0
                                            elif any(x in patch_name.lower() for x in ['quad_0', 'quad_1', 'quad_2', 'quad_3', 'quad_4', 'quad_5', 'quad_6', 'quad_7']):
                                                # Internal zones - completely hidden
                                                is_internal = True
                                                zone_alpha = 0.0    # Invisible
                                                edge_color = None
                                                line_width = 0
                                            else:
                                                # Any other zones - fully opaque but with original color
                                                zone_alpha = 1.0
                                                edge_color = None
                                                line_width = 0
                                            
                                            # Create the polygon collection with appropriate settings
                                            poly = Poly3DCollection(filtered_triangles, 
                                                                  alpha=zone_alpha,
                                                                  edgecolor=edge_color, 
                                                                  linewidth=line_width)
                                            poly.set_facecolor(color)
                                            ax.add_collection3d(poly)
                                            
                                            logger.info(f"Created mesh surface with {len(filtered_triangles)} triangles for projection {projection_dims}")
                                            surface_created = True
                                    except Exception as e:
                                        logger.debug(f"Error creating triangulation for projection {projection_dims}: {e}")
                            except Exception as e:
                                logger.warning(f"Could not create detailed mesh surface: {e}")
                        
                        # Approach 3: Convex hull (fallback for simpler geometries)
                        if not surface_created:
                            try:
                                if len(vis_points) > 3:
                                    hull = ConvexHull(vis_points)
                                    simplices = hull.simplices
                                    
                                    # Create triangular faces for the convex hull
                                    triangles = []
                                    for simplex in simplices:
                                        triangles.append(vis_points[simplex])
                                    
                                    # Determine zone type based on name for better visualization
                                    is_boundary = False
                                    is_internal = False
                                    
                                    # All external faces should be fully solid with no opacity
                                    if any(x in patch_name.lower() for x in ['triangle_8', 'triangle_10', 'quad_9']):
                                        # External boundary zones - completely solid
                                        is_boundary = True
                                        # Force solid appearance with no transparency
                                        zone_alpha = 1.0  # Fully opaque
                                        edge_color = None  # No edges to avoid lines breaking up the surface
                                        line_width = 0
                                        # Use a consistent color for all boundary zones
                                        color = '#1E90FF'  # Dodger Blue - good for solid rendering
                                    elif any(x in patch_name.lower() for x in ['quad_0', 'quad_1', 'quad_2', 'quad_3', 'quad_4', 'quad_5', 'quad_6', 'quad_7']):
                                        # Internal zones - completely hidden
                                        is_internal = True
                                        zone_alpha = 0.0  # Fully transparent (invisible)
                                        edge_color = None
                                        line_width = 0
                                    else:
                                        # Default for any other zones
                                        zone_alpha = 1.0  # Also fully opaque
                                        edge_color = None
                                        line_width = 0
                                    
                                    # Create the poly collection with appropriate settings
                                    poly = Poly3DCollection(triangles, 
                                                         alpha=zone_alpha,
                                                         edgecolor=edge_color, 
                                                         linewidth=line_width)
                                    poly.set_facecolor(color)
                                    ax.add_collection3d(poly)
                                    
                                    logger.info(f"Created simplified mesh surface with {len(triangles)} triangles using ConvexHull")
                                    surface_created = True
                            except Exception as e:
                                logger.warning(f"Could not create mesh surface using ConvexHull: {e}")
                    
                    # If all else fails, just show the points with higher density
                    if not surface_created:
                        logger.warning("Could not create any surface representation, showing dense point cloud instead")
                        try:
                            # Add more points to better represent the shape
                            more_points = min(len(patch_points), max_vis_points * 3)
                            if len(patch_points) > more_points:
                                indices = np.random.choice(len(patch_points), more_points, replace=False)
                                extra_points = patch_points[indices]
                                ax.scatter(extra_points[:,0], extra_points[:,1], extra_points[:,2], 
                                          c=[color], marker='.', alpha=0.5, s=point_size/2)
                        except Exception as e:
                            logger.warning(f"Could not show additional points: {e}")
                
                logger.info(f"Plotted patch '{patch_name}' with {len(patch_points)} points")
            except Exception as e:
                logger.warning(f"Error plotting patch '{patch_name}': {e}")
    else:
        # For meshio mesh or other types
        try:
            # Try to get patches from cell_sets
            if hasattr(mesh_data, 'cell_sets'):
                patches = list(mesh_data.cell_sets.keys())
            
            # Try to get patches from field_data (for Gmsh physical groups)
            if hasattr(mesh_data, 'field_data'):
                patches.extend(list(mesh_data.field_data.keys()))
            
            # Try to get patches from cell_data
            if hasattr(mesh_data, 'cell_data'):
                for key in mesh_data.cell_data:
                    for data_array in mesh_data.cell_data[key]:
                        if len(data_array) > 0:
                            if isinstance(data_array[0], (str, bytes)):
                                unique_values = np.unique(data_array)
                                patches.extend([f"{key}:{val}" for val in unique_values[:10]])  # Limit to first 10
            
            # Remove duplicates
            patches = list(set(patches))
            
            # Generate colors
            colors = list(mcolors.TABLEAU_COLORS.values())
            if len(patches) > len(colors):
                colors.extend(random.sample(list(mcolors.CSS4_COLORS.values()), 
                                          len(patches) - len(colors)))
            
            # Import the extract_patch_points function
            from mesh_readers.general_reader import extract_patch_points
            
            # Plot each patch
            for i, patch_name in enumerate(patches):
                try:
                    patch_points = extract_patch_points(mesh_data, patch_name)
                    
                    # Scale points if needed
                    if needs_scaling or scale_factor is not None:
                        patch_points = patch_points * scale_factor
                    
                    # Subsample points if there are too many
                    if len(patch_points) > max_points_per_patch:
                        indices = np.linspace(0, len(patch_points)-1, max_points_per_patch, dtype=int)
                        patch_points = patch_points[indices]
                    
                    # Plot the patch
                    color = colors[i % len(colors)]
                    ax.scatter(patch_points[:,0], patch_points[:,1], patch_points[:,2], 
                              c=color, marker='.', s=point_size, label=f"{patch_name} ({len(patch_points)} pts)")
                    
                    logger.info(f"Plotted patch '{patch_name}' with {len(patch_points)} points")
                except Exception as e:
                    logger.warning(f"Error plotting patch '{patch_name}': {e}")
        except Exception as e:
            logger.warning(f"Error identifying patches: {e}")
            
            # If no patches were found or there was an error, just plot all points
            ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2], 
                      c='gray', marker='.', s=point_size, label=f"All points ({len(all_points)} pts)")
            logger.info(f"Plotted all {len(all_points)} points")
    
    # If no patches were plotted, plot all points
    if not patches:
        ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2], 
                  c='gray', marker='.', s=point_size, label=f"All points ({len(all_points)} pts)")
        logger.info(f"No patches found, plotted all {len(all_points)} points")
    
    # Configure plot with improved settings
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Set axis limits to focus on the geometry
    margin = 0.05 * np.max(bbox_size)
    ax.set_xlim([min_coords[0]-margin, max_coords[0]+margin])
    ax.set_ylim([min_coords[1]-margin, max_coords[1]+margin])
    ax.set_zlim([min_coords[2]-margin, max_coords[2]+margin])
    
    # Equal aspect ratio for better visualization
    ax.set_box_aspect([1.0, 1.0, 1.0])
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend if we have multiple patches
    if len(patches) > 1 or not patches:
        # Place legend outside the plot if there are many patches
        if len(patches) > 5:
            ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small')
        else:
            ax.legend(fontsize='small')
    
    # Add dimension info to plot
    bbox_dims = max_coords - min_coords
    dim_text = f"Dimensions: {bbox_dims[0]:.4f} x {bbox_dims[1]:.4f} x {bbox_dims[2]:.4f}"
    ax.text2D(0.05, 0.95, dim_text, transform=ax.transAxes, fontsize=9)
    
    # Add scaling info if applied
    if needs_scaling or scale_factor is not None:
        scale_text = f"Note: Coordinates scaled by {scale_factor:.1f}x for visualization"
        ax.text2D(0.05, 0.92, scale_text, transform=ax.transAxes, fontsize=9, color='red')
    
    # Set the view angle if specified
    if view_angle is not None:
        elevation, azimuth = view_angle
        ax.view_init(elev=elevation, azim=azimuth)
    else:
        # Set a default view angle that shows the 3D structure well
        ax.view_init(elev=30, azim=45)
    
    # Save figure if requested
    if save_path:
        try:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(os.path.abspath(save_path))
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        except PermissionError:
            logger.error(f"Permission denied when saving to {save_path}")
            raise IOError(f"Permission denied when saving to {save_path}")
        except Exception as e:
            logger.error(f"Error saving figure: {e}")
            raise
    
    # Apply view axis orientation if specified
    if view_axis is not None:
        logger.info(f"Setting view orientation to {view_axis} axis")
        if view_axis == 'x':
            ax.view_init(azim=0, elev=0)  # Positive X axis
        elif view_axis == '-x':
            ax.view_init(azim=180, elev=0)  # Negative X axis
        elif view_axis == 'y':
            ax.view_init(azim=90, elev=0)  # Positive Y axis
        elif view_axis == '-y':
            ax.view_init(azim=270, elev=0)  # Negative Y axis
        elif view_axis == 'z':
            ax.view_init(azim=0, elev=90)  # Positive Z axis
        elif view_axis == '-z':
            ax.view_init(azim=0, elev=-90)  # Negative Z axis
    
    # Apply custom zoom to region if specified
    if zoom_region is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = zoom_region
        logger.info(f"Zooming to region: X [{x_min}:{x_max}], Y [{y_min}:{y_max}], Z [{z_min}:{z_max}]")
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
    
    # Apply zoom factor if specified and not 1.0
    if zoom_factor != 1.0:
        logger.info(f"Applying zoom factor: {zoom_factor}")
        ax.set_xlim([x_min/zoom_factor, x_max/zoom_factor])
        ax.set_ylim([y_min/zoom_factor, y_max/zoom_factor])
        ax.set_zlim([z_min/zoom_factor, z_max/zoom_factor])
    
    # Show the plot
    try:
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.warning(f"Error displaying plot: {e}")
        # This is not a critical error, so we don't raise it