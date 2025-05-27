"""
Visualization utilities for FFD control boxes.

This module provides functions for visualizing FFD control lattices,
bounding boxes, and mesh points using matplotlib and PyVista.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Configure logging
logger = logging.getLogger(__name__)

# Try importing PyVista for enhanced visualization
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    logger.warning("PyVista not available. Install with 'pip install pyvista' for enhanced visualization.")


def visualize_ffd(
    control_points: np.ndarray,
    bbox: Tuple[np.ndarray, np.ndarray],
    mesh_points: Optional[np.ndarray] = None,
    mesh_only: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dims: Tuple[int, int, int] = (4, 4, 4),
    ffd_point_size: float = 10.0,
    ffd_color: str = 'b',
    ffd_alpha: float = 0.7,
    mesh_point_size: float = 2.0,
    mesh_alpha: float = 0.3,
    mesh_color: str = 'blue',
    show_full_grid: bool = False,
    lattice_width: float = 1.0,
    control_point_size: float = 30.0,
    hide_control_points: bool = False,
    view_angle: Optional[Tuple[float, float]] = None,
    view_axis: Optional[str] = None,
    auto_scale: bool = True,
    scale_factor: Optional[float] = None,
    zoom_region: Optional[Tuple[float, float, float, float, float, float]] = None,
    zoom_factor: float = 1.0,
    use_pyvista: bool = True
) -> None:
    """Visualize FFD control lattice and bounding box, optionally with the underlying mesh.
    
    Args:
        control_points: Numpy array of control point coordinates with shape (n, 3)
        bbox: Tuple of (min_coords, max_coords) for the bounding box
        mesh_points: Optional array of mesh points to visualize
        mesh_only: Whether to show only the mesh points without the FFD box
        title: Optional custom title for the plot
        save_path: Optional path to save the figure
        dims: Dimensions of the FFD lattice (nx, ny, nz)
        ffd_point_size: Size of FFD control points in PyVista visualization
        ffd_color: Color of FFD control points
        ffd_alpha: Transparency of FFD box and control points (0.0-1.0)
        mesh_point_size: Size of mesh points
        mesh_alpha: Transparency of mesh points (0.0-1.0)
        mesh_color: Color of mesh points
        show_full_grid: Whether to show the complete internal FFD grid lines
        lattice_width: Width of lattice lines
        control_point_size: Size of control points
        hide_control_points: Whether to hide the FFD control points
        view_angle: Optional tuple of (elevation, azimuth) angles for the view
        view_axis: Align view with specified axis ('x', 'y', 'z', '-x', '-y', or '-z')
        auto_scale: Whether to automatically scale very small geometries for better visualization
        scale_factor: Optional manual scale factor to apply to all coordinates
        zoom_region: Custom region to zoom into (x_min, x_max, y_min, y_max, z_min, z_max)
        zoom_factor: Zoom factor (>1 zooms in, <1 zooms out)
        use_pyvista: Whether to use PyVista for visualization (if available)
        
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
        raise TypeError("min_coords and max_coords must be numpy arrays")
        
    if min_coords.shape != (3,) or max_coords.shape != (3,):
        raise ValueError(f"min_coords and max_coords must have shape (3,), got {min_coords.shape} and {max_coords.shape}")
    
    if mesh_points is not None:
        if not isinstance(mesh_points, np.ndarray):
            raise TypeError("mesh_points must be a numpy array")
            
        if len(mesh_points.shape) != 2 or mesh_points.shape[1] != 3:
            raise ValueError(f"mesh_points must have shape (n, 3), got {mesh_points.shape}")
    
    # If mesh_only is specified, just show the mesh points
    if mesh_only and mesh_points is not None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Limit number of mesh points to avoid overloading the plot
        max_points = 2000
        if len(mesh_points) > max_points:
            logger.info(f"Subsetting mesh points from {len(mesh_points)} to {max_points} for visualization")
            indices = np.linspace(0, len(mesh_points)-1, max_points, dtype=int)
            mesh_subset = mesh_points[indices]
        else:
            mesh_subset = mesh_points
        
        ax.scatter(mesh_subset[:,0], mesh_subset[:,1], mesh_subset[:,2], 
                  s=mesh_point_size, c=mesh_color, alpha=mesh_alpha, marker='.')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Mesh Points ({len(mesh_points)} points, showing {len(mesh_subset)})")
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if view_angle:
            ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved visualization to {save_path}")
            except Exception as e:
                logger.error(f"Error saving visualization: {e}")
        
        plt.show()
        return
    
    # Use PyVista if available and not explicitly disabled
    if PYVISTA_AVAILABLE and use_pyvista:
        visualize_ffd_pyvista(
            control_points=control_points,
            bbox=bbox,
            mesh_points=mesh_points,
            title=title,
            save_path=save_path,
            dims=dims,
            ffd_point_size=ffd_point_size,
            ffd_color=ffd_color,
            ffd_alpha=ffd_alpha,
            mesh_point_size=mesh_point_size,
            mesh_alpha=mesh_alpha,
            mesh_color=mesh_color,
            show_full_grid=show_full_grid,
            control_point_size=control_point_size,
            hide_control_points=hide_control_points,
            view_axis=view_axis,
            auto_scale=auto_scale,
            scale_factor=scale_factor,
            zoom_region=zoom_region,
            zoom_factor=zoom_factor
        )
        return
    
    # Fallback to matplotlib visualization
    visualize_ffd_matplotlib(
        control_points=control_points,
        bbox=bbox,
        mesh_points=mesh_points,
        title=title,
        save_path=save_path,
        dims=dims,
        ffd_point_size=ffd_point_size,
        ffd_color=ffd_color,
        ffd_alpha=ffd_alpha,
        mesh_point_size=mesh_point_size,
        mesh_alpha=mesh_alpha,
        mesh_color=mesh_color,
        show_full_grid=show_full_grid,
        control_point_size=control_point_size,
        hide_control_points=hide_control_points,
        view_angle=view_angle,
        auto_scale=auto_scale,
        scale_factor=scale_factor
    )


def visualize_ffd_matplotlib(
    control_points: np.ndarray,
    bbox: Tuple[np.ndarray, np.ndarray],
    mesh_points: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dims: Tuple[int, int, int] = (4, 4, 4),
    ffd_point_size: float = 10.0,
    ffd_color: str = 'b',
    ffd_alpha: float = 0.7,
    mesh_point_size: float = 2.0,
    mesh_alpha: float = 0.3,
    mesh_color: str = 'blue',
    show_full_grid: bool = False,
    control_point_size: float = 30.0,
    hide_control_points: bool = False,
    view_angle: Optional[Tuple[float, float]] = None,
    auto_scale: bool = True,
    scale_factor: Optional[float] = None
) -> None:
    """Visualize FFD control lattice using Matplotlib.
    
    Internal function used by visualize_ffd when PyVista is not available.
    """
    # Scale very small geometries for better visualization
    if auto_scale or scale_factor is not None:
        # Determine if auto-scaling is needed
        bounds_range = max_coords - min_coords
        max_dimension = np.max(bounds_range)
        min_dimension = np.min(bounds_range)
        
        # Apply scaling if explicitly requested or if geometry is very small
        apply_scaling = (scale_factor is not None) or (auto_scale and max_dimension < 0.1)
        
        if apply_scaling:
            # Use provided scale factor or calculate based on geometry size
            actual_scale = scale_factor if scale_factor is not None else (1.0 / max_dimension) * 10
            
            # Apply scaling
            control_points = control_points * actual_scale
            min_coords = min_coords * actual_scale
            max_coords = max_coords * actual_scale
            
            if mesh_points is not None:
                mesh_points = mesh_points * actual_scale
                
            logger.info(f"Applied scaling factor of {actual_scale} to visualization")
    
    # Create figure and axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"FFD Control Box ({dims[0]}×{dims[1]}×{dims[2]}, {control_points.shape[0]} control points)")
    
    # Plot mesh points if provided
    if mesh_points is not None and len(mesh_points) > 0:
        # Limit the number of points for better performance
        max_display_points = 2000
        if len(mesh_points) > max_display_points:
            # Randomly sample points
            indices = np.random.choice(len(mesh_points), max_display_points, replace=False)
            display_points = mesh_points[indices]
            logger.info(f"Displaying {max_display_points} randomly sampled points out of {len(mesh_points)}")
        else:
            display_points = mesh_points
            
        ax.scatter(display_points[:,0], display_points[:,1], display_points[:,2], 
                  s=mesh_point_size, c=mesh_color, alpha=mesh_alpha, marker='.')
    
    # Plot control points
    if not hide_control_points:
        ax.scatter(control_points[:,0], control_points[:,1], control_points[:,2], 
                  s=control_point_size, c=ffd_color, alpha=ffd_alpha, marker='o')
    
    # Get dimensions of control lattice
    nx, ny, nz = dims
    
    # Reshape control points for easier grid creation
    try:
        cp_reshaped = control_points.reshape(nx, ny, nz, 3)
        
        # Draw the FFD box edges
        for i in range(nx):
            for j in range(ny):
                ax.plot(cp_reshaped[i,j,:,0], cp_reshaped[i,j,:,1], cp_reshaped[i,j,:,2], 
                      color=ffd_color, alpha=ffd_alpha, linewidth=1.5)
                      
        for i in range(nx):
            for k in range(nz):
                ax.plot(cp_reshaped[i,:,k,0], cp_reshaped[i,:,k,1], cp_reshaped[i,:,k,2], 
                      color=ffd_color, alpha=ffd_alpha, linewidth=1.5)
                      
        for j in range(ny):
            for k in range(nz):
                ax.plot(cp_reshaped[:,j,k,0], cp_reshaped[:,j,k,1], cp_reshaped[:,j,k,2], 
                      color=ffd_color, alpha=ffd_alpha, linewidth=1.5)
                      
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Could not reshape control points to dimensions {dims}: {e}")
        logger.warning("Visualizing control points without grid structure")
    
    # Set aspect ratio to equal
    ax.set_box_aspect([1, 1, 1])
    
    # Set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set view angle if specified
    if view_angle:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Tight layout for better visualization
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        try:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(os.path.abspath(save_path))
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                logger.info(f"Created directory: {save_dir}")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
            raise IOError(f"Error saving visualization: {e}")
    
    # Show the plot
    plt.show()


def visualize_ffd_pyvista(
    control_points: np.ndarray,
    bbox: Tuple[np.ndarray, np.ndarray],
    mesh_points: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dims: Tuple[int, int, int] = (4, 4, 4),
    ffd_point_size: float = 10.0,
    ffd_color: str = 'b',
    ffd_alpha: float = 0.7,
    mesh_point_size: float = 2.0,
    mesh_alpha: float = 0.3,
    mesh_color: str = 'blue',
    show_full_grid: bool = False,
    control_point_size: float = 30.0,
    hide_control_points: bool = False,
    view_axis: Optional[str] = None,
    auto_scale: bool = True,
    scale_factor: Optional[float] = None,
    zoom_region: Optional[Tuple[float, float, float, float, float, float]] = None,
    zoom_factor: float = 1.0
) -> None:
    """Visualize FFD control lattice using PyVista.
    
    Internal function used by visualize_ffd when PyVista is available.
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for this visualization method")
    
    min_coords, max_coords = bbox
    
    # Scale very small geometries for better visualization
    if auto_scale or scale_factor is not None:
        # Determine if auto-scaling is needed
        bounds_range = max_coords - min_coords
        max_dimension = np.max(bounds_range)
        min_dimension = np.min(bounds_range)
        
        # Apply scaling if explicitly requested or if geometry is very small
        apply_scaling = (scale_factor is not None) or (auto_scale and max_dimension < 0.1)
        
        if apply_scaling:
            # Use provided scale factor or calculate based on geometry size
            actual_scale = scale_factor if scale_factor is not None else (1.0 / max_dimension) * 10
            
            # Apply scaling
            control_points = control_points * actual_scale
            min_coords = min_coords * actual_scale
            max_coords = max_coords * actual_scale
            
            if mesh_points is not None:
                mesh_points = mesh_points * actual_scale
                
            logger.info(f"Applied scaling factor of {actual_scale} to visualization")
    
    # Create a PyVista plotter
    plotter = pv.Plotter(window_size=(1024, 768))
    
    # Set background color and title
    plotter.set_background('white')
    if title:
        plotter.add_title(title, font_size=16)
    else:
        plotter.add_title(f"FFD Control Box ({dims[0]}×{dims[1]}×{dims[2]}, {control_points.shape[0]} control points)", font_size=16)
    
    # Plot mesh points if provided
    if mesh_points is not None and len(mesh_points) > 0:
        # Convert to PyVista PolyData for more efficient rendering
        mesh_cloud = pv.PolyData(mesh_points)
        
        # Limit the number of points for better performance
        max_display_points = 10000
        if len(mesh_points) > max_display_points:
            # Randomly sample points
            indices = np.random.choice(len(mesh_points), max_display_points, replace=False)
            mesh_cloud = pv.PolyData(mesh_points[indices])
            logger.info(f"Displaying {max_display_points} randomly sampled points out of {len(mesh_points)}")
        
        # Add points to the plotter
        plotter.add_mesh(
            mesh_cloud, 
            color=mesh_color, 
            opacity=mesh_alpha, 
            point_size=mesh_point_size, 
            render_points_as_spheres=True
        )
    
    # Get dimensions of control lattice
    nx, ny, nz = dims
    
    # Plot control points
    if not hide_control_points:
        # Create a PyVista PolyData for control points
        cp_cloud = pv.PolyData(control_points)
        
        # Add control points to the plotter
        plotter.add_mesh(
            cp_cloud, 
            color=ffd_color, 
            opacity=ffd_alpha, 
            point_size=control_point_size, 
            render_points_as_spheres=True
        )
    
    # Draw the FFD box grid lines
    try:
        # Reshape control points for easier grid creation
        cp_reshaped = control_points.reshape(nx, ny, nz, 3)
        
        # Create grid lines
        for i in range(nx):
            for j in range(ny):
                # Create a line through control points
                line = pv.Line(cp_reshaped[i,j,0], cp_reshaped[i,j,-1], resolution=nz-1)
                # Add the line to the plotter
                plotter.add_mesh(line, color=ffd_color, opacity=ffd_alpha, line_width=ffd_point_size/2)
                
        for i in range(nx):
            for k in range(nz):
                # Create a line through control points
                line = pv.Line(cp_reshaped[i,0,k], cp_reshaped[i,-1,k], resolution=ny-1)
                # Add the line to the plotter
                plotter.add_mesh(line, color=ffd_color, opacity=ffd_alpha, line_width=ffd_point_size/2)
                
        for j in range(ny):
            for k in range(nz):
                # Create a line through control points
                line = pv.Line(cp_reshaped[0,j,k], cp_reshaped[-1,j,k], resolution=nx-1)
                # Add the line to the plotter
                plotter.add_mesh(line, color=ffd_color, opacity=ffd_alpha, line_width=ffd_point_size/2)
        
        # If requested, show full grid with internal lines
        if show_full_grid:
            # Create a structured grid
            grid = pv.StructuredGrid()
            grid.points = control_points.reshape((nx*ny*nz, 3))
            grid.dimensions = [nx, ny, nz]
            
            # Add grid to the plotter
            plotter.add_mesh(grid, style='wireframe', color=ffd_color, opacity=ffd_alpha*0.5, line_width=ffd_point_size/3)
    
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Could not reshape control points to dimensions {dims}: {e}")
        logger.warning("Visualizing control points without grid structure")
    
    # Set camera position based on view_axis if specified
    if view_axis:
        if view_axis == 'x':
            plotter.view_xz()
        elif view_axis == 'y':
            plotter.view_yz()
        elif view_axis == 'z':
            plotter.view_xy()
        elif view_axis == '-x':
            plotter.view_xz(negative=True)
        elif view_axis == '-y':
            plotter.view_yz(negative=True)
        elif view_axis == '-z':
            plotter.view_xy(negative=True)
    
    # Apply zoom if specified
    if zoom_region:
        x_min, x_max, y_min, y_max, z_min, z_max = zoom_region
        # Set bounds explicitly
        plotter.camera.SetClippingRange((0.01, 1000))
        plotter.set_focus([x_min + (x_max - x_min)/2, 
                          y_min + (y_max - y_min)/2, 
                          z_min + (z_max - z_min)/2])
        plotter.set_position([x_min - (x_max - x_min), 
                             y_min - (y_max - y_min), 
                             z_min - (z_max - z_min)])
    elif zoom_factor != 1.0:
        # Zoom in/out by factor
        plotter.camera.Zoom(zoom_factor)
    
    # Save screenshot if path is provided
    if save_path:
        try:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(os.path.abspath(save_path))
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                logger.info(f"Created directory: {save_dir}")
            
            plotter.screenshot(save_path, transparent_background=False)
            logger.info(f"Saved visualization to {save_path}")
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
            raise IOError(f"Error saving visualization: {e}")
    
    # Show the plot
    plotter.show()
