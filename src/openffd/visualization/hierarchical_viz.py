"""
Visualization utilities for Hierarchical FFD.

This module provides functions for visualizing hierarchical FFD control boxes
with multiple resolution levels of influence.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Try importing PyVista for enhanced visualization
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    logging.warning("PyVista not available. Install with 'pip install pyvista' for enhanced visualization.")

from openffd.core.hierarchical import HierarchicalFFD, HierarchicalLevel
from openffd.utils.parallel import ParallelConfig, is_parallelizable
from openffd.visualization.level_grid import try_create_level_grid, create_boundary_edges

# Configure logging
logger = logging.getLogger(__name__)


def visualize_hierarchical_ffd_matplotlib(
    hffd: HierarchicalFFD,
    show_levels: Optional[List[int]] = None,
    title: str = "Hierarchical FFD Control Lattice",
    save_path: Optional[str] = None,
    mesh_points: Optional[np.ndarray] = None,
    show_mesh: bool = False,
    mesh_size: float = 1.0,
    mesh_alpha: float = 0.3,
    mesh_color: str = 'gray',
    point_size: float = 10.0,
    line_width: float = 1.0,
    color_by_level: bool = True,
    view_angle: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """Visualize a hierarchical FFD control lattice using Matplotlib.
    
    Args:
        hffd: Hierarchical FFD object
        show_levels: List of level IDs to show (None for all)
        title: Title for the plot
        save_path: Optional path to save the figure
        mesh_points: Optional mesh points to show
        show_mesh: Whether to show mesh points
        mesh_size: Size of mesh points
        mesh_alpha: Opacity of mesh points
        mesh_color: Color of mesh points
        point_size: Size of control points
        line_width: Width of control lattice lines
        color_by_level: Whether to color by level
        view_angle: Optional view angle (elevation, azimuth)
        figsize: Figure size
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Show mesh points if requested
    if show_mesh and mesh_points is not None:
        ax.scatter(
            mesh_points[:, 0],
            mesh_points[:, 1],
            mesh_points[:, 2],
            s=mesh_size,
            alpha=mesh_alpha,
            c=mesh_color,
            marker='.'
        )
    
    # Determine which levels to show
    if show_levels is None:
        show_levels = list(hffd.levels.keys())
    
    # Color map for levels
    cmap = plt.cm.viridis
    
    # Show each level
    for i, level_id in enumerate(show_levels):
        level = hffd.levels[level_id]
        
        # Determine color based on level depth
        if color_by_level:
            color = cmap(level.depth / max(1, max(l.depth for l in hffd.levels.values())))
        else:
            color = cmap(i / len(show_levels))
        
        # Get control points for this level
        control_points = level.control_points
        
        # Plot control points
        ax.scatter(
            control_points[:, 0],
            control_points[:, 1],
            control_points[:, 2],
            s=point_size * (1.0 - 0.2 * level.depth),  # Size decreases with depth
            alpha=0.7,
            c=[color],
            label=f"Level {level_id} (weight: {level.weight_factor:.2f})"
        )
        
        # TODO: Plot control lattice lines
        # This would require reconstructing the lattice connectivity
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set view angle if specified
    if view_angle is not None:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Show or save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    
    plt.tight_layout()
    plt.show()


def visualize_hierarchical_ffd_pyvista(
    hffd: HierarchicalFFD,
    show_levels: Optional[List[int]] = None,
    title: str = "Hierarchical FFD Control Lattice",
    save_path: Optional[str] = None,
    mesh_points: Optional[np.ndarray] = None,
    show_mesh: bool = False,
    mesh_size: float = 5.0,
    mesh_alpha: float = 0.3,
    mesh_color: str = 'gray',
    point_size: float = 10.0,
    line_width: float = 2.0,
    color_by_level: bool = True,
    bgcolor: str = 'white',
    window_size: Tuple[int, int] = (1024, 768),
    show_influence: bool = False,
    off_screen: bool = False,
    view_axis: Optional[str] = None,
    parallel_config: Optional[ParallelConfig] = None
) -> None:
    """Visualize a hierarchical FFD control lattice using PyVista.
    
    Args:
        hffd: Hierarchical FFD object
        show_levels: List of level IDs to show (None for all)
        title: Title for the plot
        save_path: Optional path to save the figure
        mesh_points: Optional mesh points to show
        show_mesh: Whether to show mesh points
        mesh_size: Size of mesh points
        mesh_alpha: Opacity of mesh points
        mesh_color: Color of mesh points
        point_size: Size of control points
        line_width: Width of control lattice lines
        color_by_level: Whether to color by level
        bgcolor: Background color
        window_size: Window size
        show_influence: Whether to show influence regions
        off_screen: Whether to render off-screen
        view_axis: View axis ('x', 'y', 'z', '-x', '-y', '-z')
        parallel_config: Configuration for parallel processing
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for this visualization method")
    
    # Create plotter
    plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)
    
    # Set background color and title
    plotter.set_background(bgcolor)
    plotter.add_title(title, font_size=16)
    
    # Show mesh points if requested
    if show_mesh and mesh_points is not None:
        # Create point cloud
        point_cloud = pv.PolyData(mesh_points)
        plotter.add_points(
            point_cloud,
            color=mesh_color,
            point_size=mesh_size,
            opacity=mesh_alpha
        )
    
    # Determine which levels to show
    if show_levels is None:
        show_levels = list(hffd.levels.keys())
    
    # Color map for levels
    cmap = plt.cm.viridis
    
    # Parallel config
    if parallel_config is None:
        parallel_config = ParallelConfig()
    
    # Show each level
    for i, level_id in enumerate(sorted(show_levels)):
        level = hffd.levels[level_id]
        
        # Determine color based on level depth
        if color_by_level:
            color = cmap(level.depth / max(1, max(l.depth for l in hffd.levels.values())))
            color = (color[0], color[1], color[2])  # Convert to RGB tuple
        else:
            color = cmap(i / len(show_levels))
            color = (color[0], color[1], color[2])  # Convert to RGB tuple
        
        # Get control points for this level
        control_points = level.control_points
        
        # Create point cloud for control points
        cp_cloud = pv.PolyData(control_points)
        
        # Add control points to plotter
        plotter.add_points(
            cp_cloud,
            color=color,
            point_size=point_size * (1.0 - 0.1 * level.depth),  # Size decreases with depth
            render_points_as_spheres=True,
            label=f"Level {level_id} (weight: {level.weight_factor:.2f})"
        )
        
        # Try to create a grid structure for this level
        nx, ny, nz = level.dims
        grid_points = try_create_level_grid(control_points, level.dims)
        
        if grid_points is not None:
            # Create lattice lines using the grid structure
            logger.info(f"Creating grid structure for level {level_id} with dimensions {level.dims}")
            edges = create_boundary_edges(grid_points)
            
            # Create a single polydata for all edges for better rendering performance
            lines = pv.MultiBlock()
            for edge in edges:
                lines.append(pv.Line(edge[0], edge[1]))
            
            # Combine all lines into a single mesh for better performance
            combined_lines = lines.combine()
            
            # Add all edges as a single mesh for better performance
            plotter.add_mesh(
                combined_lines, 
                color=color,
                line_width=line_width * (1.0 - 0.1 * level.depth),  # Width decreases with depth
                opacity=0.7,
                label=f"Grid Level {level_id}"
            )
        else:
            logger.warning(f"Could not create grid structure for level {level_id} with dimensions {level.dims} - showing points only")
        
        # Show influence regions if requested
        if show_influence and mesh_points is not None:
            # Calculate influence of this level on mesh points
            influence = level.get_influence(mesh_points)
            
            # Create point cloud colored by influence
            if is_parallelizable(len(mesh_points), parallel_config):
                # Use parallel processing for large meshes
                logger.info(f"Computing influence for {len(mesh_points)} points in parallel")
                
                # Function to process a chunk of points
                def process_chunk(points_chunk):
                    return level.get_influence(points_chunk)
                
                # Calculate influence in parallel
                from openffd.utils.parallel import parallel_process, chunk_array
                
                # Chunk the mesh points
                chunk_size = parallel_config.chunk_size if parallel_config.chunk_size else min(10000, len(mesh_points))
                point_chunks = chunk_array(mesh_points, chunk_size)
                
                # Process chunks in parallel
                influence_chunks = parallel_process(process_chunk, point_chunks, parallel_config)
                
                # Combine results
                influence = np.concatenate(influence_chunks)
            
            # Create a separate point cloud for each influence level
            # to avoid having too many points in a single cloud
            if len(mesh_points) > 10000:
                # For large meshes, sample points
                sample_rate = max(1, len(mesh_points) // 10000)
                sample_indices = np.arange(0, len(mesh_points), sample_rate)
                sample_points = mesh_points[sample_indices]
                sample_influence = influence[sample_indices]
                
                # Create point cloud
                influence_cloud = pv.PolyData(sample_points)
                influence_cloud["influence"] = sample_influence
            else:
                # Create point cloud
                influence_cloud = pv.PolyData(mesh_points)
                influence_cloud["influence"] = influence
            
            # Add to plotter
            plotter.add_points(
                influence_cloud,
                scalars="influence",
                cmap="coolwarm",
                point_size=mesh_size * 0.8,
                opacity=0.5,
                render_points_as_spheres=True,
                show_scalar_bar=True,
                scalar_bar_args={"title": f"Level {level_id} Influence"}
            )
        
        # Grid structure is now handled by the try_create_level_grid and create_boundary_edges functions
    
    # Add legend
    plotter.add_legend()
    
    # Set view axis if specified
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
    
    # Save if requested
    if save_path:
        plotter.screenshot(save_path, transparent_background=False)
        logger.info(f"Saved visualization to {save_path}")
    
    # Show the plot
    plotter.show()


def visualize_influence_distribution(
    hffd: HierarchicalFFD,
    mesh_points: np.ndarray,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """Visualize the influence distribution of hierarchical levels.
    
    Args:
        hffd: Hierarchical FFD object
        mesh_points: Mesh points to calculate influence for
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Calculate influence for each level
    level_influences = {}
    for level_id, level in hffd.levels.items():
        level_influences[level_id] = level.get_influence(mesh_points)
    
    # Plot histogram of influence values
    for level_id, influence in level_influences.items():
        ax1.hist(
            influence,
            bins=50,
            alpha=0.7,
            label=f"Level {level_id} (depth {hffd.levels[level_id].depth})"
        )
    
    ax1.set_xlabel("Influence Value")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Level Influence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot level weights vs depth
    depths = [level.depth for level in hffd.levels.values()]
    weights = [level.weight_factor for level in hffd.levels.values()]
    ids = [level.level_id for level in hffd.levels.values()]
    
    # Sort by depth
    sorted_indices = np.argsort(depths)
    depths = [depths[i] for i in sorted_indices]
    weights = [weights[i] for i in sorted_indices]
    ids = [ids[i] for i in sorted_indices]
    
    ax2.bar(range(len(depths)), weights, alpha=0.7)
    ax2.set_xticks(range(len(depths)))
    ax2.set_xticklabels([f"L{i} (d={d})" for i, d in zip(ids, depths)])
    ax2.set_xlabel("Level (depth)")
    ax2.set_ylabel("Weight Factor")
    ax2.set_title("Level Weights by Depth")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved influence distribution to {save_path}")
    
    plt.show()


def visualize_hierarchical_deformation(
    hffd: HierarchicalFFD,
    deformed_control_points: Dict[int, np.ndarray],
    mesh_points: Optional[np.ndarray] = None,
    show_levels: Optional[List[int]] = None,
    title: str = "Hierarchical FFD Deformation",
    save_path: Optional[str] = None,
    show_original: bool = True,
    original_alpha: float = 0.3,
    original_color: str = 'gray',
    deformed_color: str = 'blue',
    window_size: Tuple[int, int] = (1024, 768),
    view_axis: Optional[str] = None,
    off_screen: bool = False
) -> None:
    """Visualize a hierarchical FFD deformation using PyVista.
    
    Args:
        hffd: Hierarchical FFD object
        deformed_control_points: Dictionary mapping level_id to deformed control points
        mesh_points: Optional mesh points to deform (defaults to hffd.mesh_points)
        show_levels: List of level IDs to show (None for all)
        title: Title for the plot
        save_path: Optional path to save the figure
        show_original: Whether to show original mesh
        original_alpha: Opacity of original mesh
        original_color: Color of original mesh
        deformed_color: Color of deformed mesh
        window_size: Window size
        view_axis: View axis ('x', 'y', 'z', '-x', '-y', '-z')
        off_screen: Whether to render off-screen
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for this visualization method")
    
    # Get mesh points to deform
    if mesh_points is None:
        mesh_points = hffd.mesh_points
    
    # Deform mesh points
    deformed_points = hffd.deform_mesh(deformed_control_points, mesh_points)
    
    # Create plotter
    plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)
    
    # Set background color and title
    plotter.set_background('white')
    plotter.add_title(title, font_size=16)
    
    # Show original mesh if requested
    if show_original:
        original_cloud = pv.PolyData(mesh_points)
        plotter.add_points(
            original_cloud,
            color=original_color,
            point_size=5.0,
            opacity=original_alpha,
            label="Original Mesh"
        )
    
    # Show deformed mesh
    deformed_cloud = pv.PolyData(deformed_points)
    plotter.add_points(
        deformed_cloud,
        color=deformed_color,
        point_size=5.0,
        label="Deformed Mesh"
    )
    
    # Determine which levels to show
    if show_levels is None:
        show_levels = list(deformed_control_points.keys())
    
    # Color map for levels
    cmap = plt.cm.viridis
    
    # Show control points for each level
    for i, level_id in enumerate(sorted(show_levels)):
        if level_id not in deformed_control_points:
            continue
        
        level = hffd.levels[level_id]
        
        # Determine color based on level depth
        color = cmap(level.depth / max(1, max(l.depth for l in hffd.levels.values())))
        color = (color[0], color[1], color[2])  # Convert to RGB tuple
        
        # Get original and deformed control points
        original_cp = level.control_points
        deformed_cp = deformed_control_points[level_id]
        
        # Show original control points
        cp_cloud = pv.PolyData(original_cp)
        plotter.add_points(
            cp_cloud,
            color=original_color,
            point_size=8.0 * (1.0 - 0.1 * level.depth),
            opacity=0.5,
            render_points_as_spheres=True
        )
        
        # Show deformed control points
        deformed_cp_cloud = pv.PolyData(deformed_cp)
        plotter.add_points(
            deformed_cp_cloud,
            color=color,
            point_size=10.0 * (1.0 - 0.1 * level.depth),
            render_points_as_spheres=True,
            label=f"Level {level_id} (depth {level.depth})"
        )
        
        # Show displacement vectors
        displacement = deformed_cp - original_cp
        
        # Skip very small displacements
        mask = np.linalg.norm(displacement, axis=1) > 1e-6
        if np.any(mask):
            # Create arrows for displacement vectors
            arrows = pv.Arrow()
            for j in range(len(original_cp)):
                if mask[j]:
                    # Create arrow for this displacement
                    start = original_cp[j]
                    direction = displacement[j]
                    
                    # Scale arrow by displacement magnitude
                    mag = np.linalg.norm(direction)
                    if mag > 1e-6:
                        arrow = arrows.copy()
                        arrow.translate(start)
                        arrow.scale(mag * 0.5)  # Scale arrow by displacement
                        arrow.rotate_vector(direction, [1, 0, 0])
                        
                        # Add arrow to plotter
                        plotter.add_mesh(
                            arrow,
                            color=color,
                            opacity=0.7
                        )
    
    # Add legend
    plotter.add_legend()
    
    # Set view axis if specified
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
    
    # Save if requested
    if save_path:
        plotter.screenshot(save_path, transparent_background=False)
        logger.info(f"Saved deformation visualization to {save_path}")
    
    # Show the plot
    plotter.show()
