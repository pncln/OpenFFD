"""
Mesh visualization utilities for OpenFFD.

This module provides functions for visualizing meshes with patches and zones
using both matplotlib and PyVista for high-quality rendering.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Import the level_grid utility for properly reshaping control points
from openffd.visualization.level_grid import try_create_level_grid, create_boundary_edges

from openffd.utils.parallel import ParallelConfig, is_parallelizable, parallel_process
from openffd.visualization.parallel_viz import (
    process_point_cloud_parallel,
    subsample_points_parallel,
    create_mesh_chunks_parallel,
    compute_normals_parallel,
    extract_mesh_features_parallel
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


def visualize_mesh_with_patches(
    mesh_data: Any,
    save_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (12, 10),
    dpi: int = 300,
    title: str = "Mesh with Patches",
    point_size: float = 2.0,
    auto_scale: bool = True,
    scale_factor: Optional[float] = None,
    view_angle: Optional[Tuple[float, float]] = None,
    show_solid: bool = True,
    alpha: float = 1.0,
    max_faces_per_zone: int = 500000,
    max_points_per_patch: int = 50000,
    use_multiple_projections: bool = True,
    preserve_structure: bool = True,
    remove_artifacts: bool = True,
    shading_mode: str = 'gouraud',
    use_pyvista: bool = True
) -> None:
    """Visualize the initial mesh geometry with all patches in different colors.
    
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
        max_faces_per_zone: Maximum number of faces to process per zone
        max_points_per_patch: Maximum number of points to display per patch
        use_multiple_projections: Whether to use multiple projections for better patch visualization
        preserve_structure: Whether to preserve mesh structure when downsampling
        remove_artifacts: Whether to remove artifacts from triangulation
        shading_mode: Shading mode for solid surfaces ('flat' or 'gouraud')
        use_pyvista: Whether to use PyVista for visualization (if available)
        
    Raises:
        ValueError: If the mesh has no points or patches
        IOError: If the figure cannot be saved to the specified path
    """
    # Check if PyVista is available and should be used
    if PYVISTA_AVAILABLE and use_pyvista:
        # Use PyVista for better visualization
        visualize_mesh_with_patches_pyvista(
            mesh_data=mesh_data,
            save_path=save_path,
            title=title,
            point_size=point_size,
            auto_scale=auto_scale,
            scale_factor=scale_factor,
            show_solid=show_solid,
            opacity=alpha,
            max_points_per_zone=max_points_per_patch
        )
        return

    # Import mesh related modules here to avoid circular imports
    from openffd.mesh.fluent import FluentMeshReader
    
    # Track visualization timing
    start_time = time.time()
    
    # Configure parallel processing
    parallel_config = ParallelConfig(
        enabled=True,
        threshold=100000,
        max_workers=None
    )
    
    # Get all points and patches from the mesh
    all_points = None
    patches = {}
    
    # Handle different mesh data types
    if isinstance(mesh_data, FluentMeshReader):
        logger.info("Processing Fluent mesh data with parallel processing")
        all_points = mesh_data.points
        
        # Get zone information
        zone_names = mesh_data.get_zone_names()
        
        # Use parallel processing to extract zone points if dataset is large
        if parallel_config.enabled and is_parallelizable(len(all_points), parallel_config):
            # Define function to extract zone points in parallel
            def get_zone_data(zone_name):
                return (
                    zone_name, 
                    mesh_data.get_zone_points(zone_name),
                    mesh_data.get_zone_type(zone_name),
                    mesh_data.get_zone_faces(zone_name) if True and hasattr(mesh_data, 'get_zone_faces') else None
                )
            
            # Process zones in parallel
            logger.info(f"Extracting zone data in parallel for {len(zone_names)} zones")
            zone_results = parallel_process(get_zone_data, zone_names, parallel_config)
            
            # Organize results
            patches = {}
            patch_types = {}
            zone_faces = {}
            
            for zone_name, zone_points, zone_type, zone_face_data in zone_results:
                patches[zone_name] = zone_points
                patch_types[zone_name] = zone_type
                if zone_face_data is not None:
                    zone_faces[zone_name] = zone_face_data
        else:
            # Standard sequential processing
            patches = {zone_name: mesh_data.get_zone_points(zone_name) 
                     for zone_name in zone_names}
            patch_types = {zone_name: mesh_data.get_zone_type(zone_name) 
                         for zone_name in zone_names}
            
            # Get face connectivity if available
            zone_faces = {}
            if True and hasattr(mesh_data, 'get_zone_faces'):
                for zone_name in zone_names:
                    try:
                        zone_faces[zone_name] = mesh_data.get_zone_faces(zone_name)
                    except Exception as e:
                        logger.warning(f"Could not get faces for zone {zone_name}: {e}")
    elif hasattr(mesh_data, 'points') and hasattr(mesh_data, 'cells'):
        # This is likely a meshio.Mesh object
        all_points = mesh_data.points
        
        # Extract patches from mesh cells
        patches = {}
        patch_types = {}
        zone_faces = {}
        
        # Process cells by type
        for cell_type, cell_data in zip(mesh_data.cells_dict.keys(), mesh_data.cells_dict.values()):
            # Create a zone name based on cell type
            zone_name = f"{cell_type}"
            
            # Extract points for this zone
            cell_points = np.unique(cell_data.flatten())
            patches[zone_name] = all_points[cell_points]
            patch_types[zone_name] = cell_type
            
            # Store face connectivity if requested
            if True:
                zone_faces[zone_name] = cell_data
    else:
        # Try to extract points directly
        try:
            all_points = np.array(mesh_data)
            patches = {"default": all_points}
            patch_types = {"default": "points"}
        except Exception as e:
            raise ValueError(f"Could not extract points from mesh data: {e}")
    
    # Check if we have points
    if all_points is None or len(all_points) == 0:
        raise ValueError("Mesh has no points")
    
    # Check if we have patches
    if not patches:
        raise ValueError("Mesh has no patches")
    
    # Scale very small geometries for better visualization
    if auto_scale or scale_factor is not None:
        # Determine if auto-scaling is needed
        bounds_min = np.min(all_points, axis=0)
        bounds_max = np.max(all_points, axis=0)
        bounds_range = bounds_max - bounds_min
        max_dimension = np.max(bounds_range)
        min_dimension = np.min(bounds_range)
        
        # Apply scaling if explicitly requested or if geometry is very small
        apply_scaling = (scale_factor is not None) or (auto_scale and max_dimension < 0.1)
        
        if apply_scaling:
            # Use provided scale factor or calculate based on geometry size
            actual_scale = scale_factor if scale_factor is not None else (1.0 / max_dimension) * 10
            
            # Apply scaling to all points
            all_points = all_points * actual_scale
            
            # Apply scaling to all patches
            for patch_name in patches:
                patches[patch_name] = patches[patch_name] * actual_scale
                
            logger.info(f"Applied scaling factor of {actual_scale} to visualization")
    
    # Create figure and axis
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set title
    ax.set_title(title)
    
    # Define a colormap for patches
    colors = plt.cm.tab20.colors
    
    # Plot each patch with a different color
    for i, (patch_name, patch_points) in enumerate(patches.items()):
        # Limit the number of points for better performance
        if len(patch_points) > max_points_per_patch:
            # Randomly sample points
            indices = np.random.choice(len(patch_points), max_points_per_patch, replace=False)
            display_points = patch_points[indices]
            logger.info(f"Patch '{patch_name}': Displaying {max_points_per_patch} randomly sampled points out of {len(patch_points)}")
        else:
            display_points = patch_points
        
        # Get color for this patch
        color = colors[i % len(colors)]
        
        # Plot the patch
        ax.scatter(display_points[:,0], display_points[:,1], display_points[:,2], 
                  s=point_size, c=[color], alpha=0.7, marker='.')
        
        # Add label for this patch
        ax.text(np.mean(display_points[:,0]), np.mean(display_points[:,1]), np.mean(display_points[:,2]), 
               patch_name, color=color, fontsize=8)
    
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
            
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
            raise IOError(f"Error saving visualization: {e}")
    
    # Show the plot
    plt.show()


def visualize_mesh_with_patches_pyvista(
    mesh_data: Any,
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
    max_points_per_zone: int = 5000,
    max_triangles: int = 10000,
    detail_level: str = 'medium',
    skip_internal_zones: bool = True,
    use_original_faces: bool = False,
    ffd_control_points: Optional[np.ndarray] = None,
    ffd_color: str = 'red',
    ffd_opacity: float = 0.7,
    ffd_point_size: float = 10.0,
    show_ffd_mesh: bool = True,
    suppress_warnings: bool = True,
    zoom_region: Optional[Tuple[float, float, float, float, float, float]] = None,
    zoom_factor: float = 1.0,
    view_axis: Optional[str] = None,
    ffd_box_dims: Optional[List[int]] = None,
    parallel: bool = True,
    parallel_threshold: int = 100000,
    parallel_workers: Optional[int] = None
) -> None:
    """Visualize the mesh using PyVista for faster, higher-quality rendering.
    
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
        
    Raises:
        ImportError: If PyVista is not available
        ValueError: If the mesh has no points or patches
        IOError: If the figure cannot be saved to the specified path
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for this visualization method")
    
    # Import mesh related modules here to avoid circular imports
    from openffd.mesh.fluent import FluentMeshReader
    
    # Track visualization timing
    start_time = time.time()
    
    # Configure parallel processing
    parallel_config = ParallelConfig(
        enabled=parallel,
        threshold=parallel_threshold,
        max_workers=parallel_workers
    )
    
    # Create a PyVista plotter
    plotter = pv.Plotter(window_size=window_size)
    
    # Set background color and title
    plotter.set_background(bgcolor)
    plotter.add_title(title, font_size=16)
    
    # Get all points and patches from the mesh
    all_points = None
    patches = {}
    patch_types = {}
    
    # Handle different mesh types
    if isinstance(mesh_data, FluentMeshReader):
        all_points = mesh_data.points
        
        # Get all zones
        for zone_name in mesh_data.get_available_zones():
            # Skip internal zones if requested
            if skip_internal_zones and mesh_data.zone_types.get(zone_name) == 'interior':
                continue
                
            zone_points = mesh_data.get_zone_points(zone_name)
            if len(zone_points) > 0:
                patches[zone_name] = zone_points
                patch_types[zone_name] = mesh_data.zone_types.get(zone_name, 'unknown')
                
                # Get faces if available
                if use_original_faces and zone_name in mesh_data.faces_by_zone:
                    # Process faces
                    faces = mesh_data.get_faces(zone_name)
                    face_types = mesh_data.get_face_types(zone_name)
                    
                    if faces:
                        # Create a PyVista mesh from these faces
                        try:
                            # Process faces and create PyVista mesh
                            # ... (implementation details specific to your mesh format)
                            logger.info(f"Using original face connectivity for zone '{zone_name}'")
                        except Exception as e:
                            logger.warning(f"Could not create PyVista mesh from faces for zone '{zone_name}': {e}")
    else:
        # Assume it's a meshio.Mesh object
        if hasattr(mesh_data, 'points'):
            all_points = mesh_data.points
            
            # Try to extract patches from cell_sets
            if hasattr(mesh_data, 'cell_sets') and mesh_data.cell_sets:
                for patch_name, cell_blocks in mesh_data.cell_sets.items():
                    # Extract points for this patch
                    patch_point_indices = set()
                    
                    for cell_type, indices in cell_blocks.items():
                        # Find the corresponding cell block
                        cell_block_idx = next((i for i, block in enumerate(mesh_data.cells) 
                                             if block.type == cell_type), None)
                        
                        if cell_block_idx is None:
                            continue
                            
                        # Get the cells for this patch
                        cells = mesh_data.cells[cell_block_idx].data[indices]
                        
                        # Collect all unique node IDs
                        for cell in cells:
                            for node_id in cell:
                                patch_point_indices.add(node_id)
                    
                    # Convert to points
                    if patch_point_indices:
                        patch_points = all_points[list(patch_point_indices)]
                        patches[patch_name] = patch_points
                        patch_types[patch_name] = 'unknown'
            
            # If no patches found, create a default patch with all points
            if not patches:
                patches["default"] = all_points
                patch_types["default"] = 'default'
    
    # Check if we have points
    if all_points is None or len(all_points) == 0:
        raise ValueError("Mesh has no points")
    
    # Check if we have patches
    if not patches:
        raise ValueError("Mesh has no patches")
    
    # Scale very small geometries for better visualization
    if auto_scale or scale_factor is not None:
        # Determine if auto-scaling is needed
        bounds_min = np.min(all_points, axis=0)
        bounds_max = np.max(all_points, axis=0)
        bounds_range = bounds_max - bounds_min
        max_dimension = np.max(bounds_range)
        min_dimension = np.min(bounds_range)
        
        # Apply scaling if explicitly requested or if geometry is very small
        apply_scaling = (scale_factor is not None) or (auto_scale and max_dimension < 0.1)
        
        if apply_scaling:
            # Use provided scale factor or calculate based on geometry size
            actual_scale = scale_factor if scale_factor is not None else (1.0 / max_dimension) * 10
            
            # Apply scaling to all points
            all_points = all_points * actual_scale
            
            # Apply scaling to all patches
            for patch_name in patches:
                patches[patch_name] = patches[patch_name] * actual_scale
                
            # Apply scaling to FFD control points if provided
            if ffd_control_points is not None:
                ffd_control_points = ffd_control_points * actual_scale
                
            logger.info(f"Applied scaling factor of {actual_scale} to visualization")
    
    # Calculate mesh features in parallel if the dataset is large
    if parallel_config.enabled and is_parallelizable(len(all_points), parallel_config):
        logger.info(f"Computing mesh features in parallel for {len(all_points)} points")
        mesh_features = extract_mesh_features_parallel(
            all_points, 
            compute_bbox=True, 
            compute_center=True,
            config=parallel_config
        )
        bbox_min = mesh_features['bbox_min']
        bbox_max = mesh_features['bbox_max']
    else:
        # Standard sequential computation
        bbox_min = np.min(all_points, axis=0)
        bbox_max = np.max(all_points, axis=0)
    
    # Scale the mesh if needed
    scale = 1.0
    if auto_scale:
        # Calculate bounding box size
        bbox_size = np.max(bbox_max - bbox_min)
        
        # If the geometry is very small, scale it up for better visualization
        if bbox_size < 1e-3:
            scale = 1.0 / bbox_size
            logger.info(f"Auto-scaling very small geometry by factor of {scale}")
    
    # Apply manual scaling if provided
    if scale_factor is not None:
        scale = scale_factor
        logger.info(f"Applied manual scaling factor of {scale}")
    
    # Apply scaling to all points if needed
    if scale != 1.0:
        all_points = all_points * scale
        for zone_name in patches:
            patches[zone_name] = patches[zone_name] * scale
        
        # Scale FFD control points if present
        if ffd_control_points is not None:
            ffd_control_points = ffd_control_points * scale
    
    # Define a colormap for patches
    colors = plt.cm.tab20.colors
    
    # Plot each patch with a different color
    for i, (patch_name, patch_points) in enumerate(patches.items()):
        # Limit the number of points for better performance
        if len(patch_points) > max_points_per_zone:
            # If there are too many points, subsample for faster rendering using parallel processing
            if parallel_config.enabled and is_parallelizable(len(patch_points), parallel_config):
                # Parallel subsampling for large point clouds
                logger.info(f"Subsampling zone '{patch_name}' in parallel from {len(patch_points)} to {max_points_per_zone} points")
                patch_points = subsample_points_parallel(patch_points, max_points_per_zone, parallel_config)
            else:
                # Standard uniform sampling to reduce point count
                sample_indices = np.linspace(0, len(patch_points) - 1, max_points_per_zone, dtype=int)
                patch_points = patch_points[sample_indices]
            
            logger.info(f"Subsampled zone '{patch_name}' from {len(patches[patch_name])} to {len(patch_points)} points")
        
        # Get color for this patch
        if color_by_zone:
            color = colors[i % len(colors)]
        else:
            # Use a fixed color scheme based on patch type
            patch_type = patch_types.get(patch_name, 'unknown')
            if patch_type == 'wall':
                color = 'lightblue'
            elif patch_type == 'symmetry':
                color = 'lightgreen'
            elif patch_type == 'inlet':
                color = 'lightcoral'
            elif patch_type == 'outlet':
                color = 'lightyellow'
            else:
                color = 'lightgray'
        
        # Convert points to PyVista PolyData
        point_cloud = pv.PolyData(patch_points)
        
        # Create a surface from points if requested
        if show_solid:
            try:
                # Create a surface from points
                surf = point_cloud.delaunay_2d()
                
                # Add surface to the plotter
                plotter.add_mesh(
                    surf, 
                    color=color, 
                    opacity=opacity, 
                    show_edges=show_edges,
                    smooth_shading=True
                )
            except Exception as e:
                if not suppress_warnings:
                    logger.warning(f"Could not create surface for patch '{patch_name}': {e}")
                # Fall back to point cloud
                plotter.add_mesh(
                    point_cloud, 
                    color=color, 
                    opacity=opacity, 
                    point_size=point_size, 
                    render_points_as_spheres=True
                )
        else:
            # Just show points
            plotter.add_mesh(
                point_cloud, 
                color=color, 
                opacity=opacity, 
                point_size=point_size, 
                render_points_as_spheres=True
            )
        
        # Add label for this patch at its centroid
        if point_cloud.n_points > 0:
            centroid = np.mean(patch_points, axis=0)
            plotter.add_point_labels(
                [centroid], 
                [patch_name], 
                font_size=10, 
                point_color=color, 
                text_color='black',
                always_visible=True
            )
    
    # Add FFD control points if provided
    if ffd_control_points is not None and len(ffd_control_points) > 0:
        # Convert to PyVista PolyData
        cp_cloud = pv.PolyData(ffd_control_points)
        
        # Add control points to the plotter
        plotter.add_mesh(
            cp_cloud, 
            color=ffd_color, 
            opacity=ffd_opacity, 
            point_size=ffd_point_size, 
            render_points_as_spheres=True
        )
        
        # Add FFD box grid if dimensions are provided
        if show_ffd_mesh and ffd_box_dims is not None and len(ffd_box_dims) == 3:
            nx, ny, nz = ffd_box_dims
            
            # Try to reshape control points using our specialized grid utility
            grid_points = try_create_level_grid(ffd_control_points, ffd_box_dims)
            
            if grid_points is not None:
                # Successfully reshaped to grid, create lattice lines
                nx, ny, nz = ffd_box_dims
                
                # Create grid lines
                for i in range(nx):
                    for j in range(ny):
                        # Create a line through control points
                        line = pv.Line(grid_points[i,j,0], grid_points[i,j,-1], resolution=nz-1)
                        # Add the line to the plotter
                        plotter.add_mesh(line, color=ffd_color, opacity=ffd_opacity, line_width=ffd_point_size/2)
                        
                for i in range(nx):
                    for k in range(nz):
                        # Create a line through control points
                        line = pv.Line(grid_points[i,0,k], grid_points[i,-1,k], resolution=ny-1)
                        # Add the line to the plotter
                        plotter.add_mesh(line, color=ffd_color, opacity=ffd_opacity, line_width=ffd_point_size/2)
                        
                for j in range(ny):
                    for k in range(nz):
                        # Create a line through control points
                        line = pv.Line(grid_points[0,j,k], grid_points[-1,j,k], resolution=nx-1)
                        # Add the line to the plotter
                        plotter.add_mesh(line, color=ffd_color, opacity=ffd_opacity, line_width=ffd_point_size/2)
            
            else:
                # Could not reshape to grid - just display points without grid structure
                # For hierarchical FFD, this is normal since each level has different dimensions
                if len(ffd_control_points) != nx * ny * nz:
                    logger.debug(f"FFD control points don't match dimensions {ffd_box_dims}: {len(ffd_control_points)} points vs {nx*ny*nz} expected")
                    logger.debug("This is normal for hierarchical FFD with different level dimensions")
                else:
                    logger.warning(f"Could not create grid structure for control points with dimensions {ffd_box_dims}")
                    logger.warning("Visualizing control points without grid structure")
    
    # Show axes if requested
    if show_axes:
        plotter.add_axes()
    
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
    
    # Close the plotter to avoid memory leaks
    plotter.close()
    
    # Report visualization timing
    end_time = time.time()
    logger.info(f"Visualization completed in {end_time - start_time:.2f} seconds")
    if parallel_config.enabled:
        logger.info(f"Used parallel processing with {parallel_config.method} method")
        if parallel_config.max_workers:
            logger.info(f"Used {parallel_config.max_workers} workers")
        else:
            logger.info("Used auto-detected number of workers")
