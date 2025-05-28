"""
Visualization utilities for mesh zones.

This module provides functionality for visualizing mesh zones
extracted by the ZoneExtractor.
"""

import logging
import os
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import numpy as np

from openffd.mesh.zone_extractor import ZoneExtractor, ZoneType, ZoneInfo

logger = logging.getLogger(__name__)

# Check for visualization libraries
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Install with 'pip install matplotlib' for basic visualization.")

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    logger.warning("pyvista not available. Install with 'pip install pyvista' for advanced visualization.")


def visualize_zones_matplotlib(
    extractor: ZoneExtractor,
    zone_names: Optional[List[str]] = None,
    zone_colors: Optional[Dict[str, str]] = None,
    point_size: float = 5.0,
    alpha: float = 0.7,
    figure_size: Tuple[int, int] = (10, 8),
    view_angle: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> Any:
    """Visualize mesh zones using matplotlib.
    
    Args:
        extractor: ZoneExtractor instance
        zone_names: List of zone names to visualize (default: all zones)
        zone_colors: Dictionary mapping zone names to colors (default: auto-generate)
        point_size: Size of points in the visualization
        alpha: Transparency of points (0.0-1.0)
        figure_size: Size of the figure (width, height) in inches
        view_angle: Optional view angle as (elevation, azimuth)
        save_path: Optional path to save the visualization
        show: Whether to show the visualization
        
    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")
        
    # Get all zones if none specified
    if zone_names is None:
        zone_names = extractor.get_zone_names()
        
    # Create a default color map if none provided
    if zone_colors is None:
        # Use a color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        zone_colors = {name: colors[i % len(colors)] for i, name in enumerate(zone_names)}
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract and plot each zone
    for zone_name in zone_names:
        try:
            # Extract zone points
            zone_mesh = extractor.extract_zone_mesh(zone_name)
            points = zone_mesh.points
            
            # Get zone info
            zone_info = extractor.get_zone_info(zone_name)
            
            # Determine marker based on zone type
            marker = 'o'  # default
            if zone_info.zone_type == ZoneType.VOLUME:
                marker = 'o'
            elif zone_info.zone_type == ZoneType.BOUNDARY:
                marker = 's'
            elif zone_info.zone_type == ZoneType.INTERFACE:
                marker = 'd'
                
            # Plot the zone points
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                s=point_size,
                c=zone_colors.get(zone_name, 'blue'),
                marker=marker,
                alpha=alpha,
                label=f"{zone_name} ({zone_info.zone_type.name})"
            )
            
        except Exception as e:
            logger.warning(f"Error visualizing zone '{zone_name}': {e}")
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set view angle if provided
    if view_angle is not None:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Add legend
    ax.legend()
    
    # Add title
    plt.title(f"Mesh Zones: {os.path.basename(extractor.mesh_file)}")
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
        
    return fig


def visualize_zones_pyvista(
    extractor: ZoneExtractor,
    zone_names: Optional[List[str]] = None,
    zone_colors: Optional[Dict[str, str]] = None,
    point_size: float = 5.0,
    alpha: float = 0.7,
    show_edges: bool = True,
    show_zones_as_surfaces: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    background_color: str = 'white',
    window_size: Tuple[int, int] = (1024, 768)
) -> Any:
    """Visualize mesh zones using PyVista for advanced visualization.
    
    Args:
        extractor: ZoneExtractor instance
        zone_names: List of zone names to visualize (default: all zones)
        zone_colors: Dictionary mapping zone names to colors (default: auto-generate)
        point_size: Size of points in the visualization
        alpha: Transparency of points/surfaces (0.0-1.0)
        show_edges: Whether to show mesh edges
        show_zones_as_surfaces: Whether to show zones as surfaces (if possible)
        save_path: Optional path to save the visualization
        show: Whether to show the visualization
        background_color: Background color of the visualization
        window_size: Size of the rendering window
        
    Returns:
        PyVista plotter
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError("pyvista is required for advanced visualization")
        
    # Get all zones if none specified
    if zone_names is None:
        zone_names = extractor.get_zone_names()
        
    # Create a default color map if none provided
    if zone_colors is None:
        # Generate colors using HSV color space for better differentiation
        import colorsys
        n_zones = len(zone_names)
        zone_colors = {}
        for i, name in enumerate(zone_names):
            hue = i / n_zones
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            zone_colors[name] = rgb
    
    # Create a plotter
    plotter = pv.Plotter(window_size=window_size)
    plotter.set_background(background_color)
    
    # Add a legend
    plotter.add_legend(size=(0.2, 0.2), face='rectangle')
    
    # Extract and plot each zone
    for zone_name in zone_names:
        try:
            # Extract zone mesh
            zone_mesh = extractor.extract_zone_mesh(zone_name)
            
            # Get zone info
            zone_info = extractor.get_zone_info(zone_name)
            
            # Convert meshio mesh to PyVista mesh
            pv_mesh = _meshio_to_pyvista(zone_mesh)
            
            if pv_mesh is None:
                logger.warning(f"Failed to convert zone '{zone_name}' to PyVista mesh")
                continue
                
            # Determine how to display based on zone type and mesh content
            if show_zones_as_surfaces and len(pv_mesh.faces) > 0:
                # Show as surface with edges
                plotter.add_mesh(
                    pv_mesh,
                    color=zone_colors.get(zone_name, 'blue'),
                    opacity=alpha,
                    edge_color='black' if show_edges else None,
                    show_edges=show_edges,
                    label=f"{zone_name} ({zone_info.zone_type.name})"
                )
            else:
                # Show as points
                plotter.add_mesh(
                    pv_mesh.points,
                    color=zone_colors.get(zone_name, 'blue'),
                    opacity=alpha,
                    point_size=point_size,
                    render_points_as_spheres=True,
                    label=f"{zone_name} ({zone_info.zone_type.name})"
                )
                
        except Exception as e:
            logger.warning(f"Error visualizing zone '{zone_name}': {e}")
    
    # Add a title
    plotter.add_title(f"Mesh Zones: {os.path.basename(extractor.mesh_file)}")
    
    # Save if requested
    if save_path:
        plotter.screenshot(save_path, transparent_background=(background_color == 'white'))
    
    # Show if requested
    if show:
        plotter.show()
        
    return plotter


def visualize_zone_comparison(
    extractor: ZoneExtractor,
    zone1_name: str,
    zone2_name: str,
    point_size: float = 5.0,
    alpha: float = 0.7,
    color1: str = 'blue',
    color2: str = 'red',
    figure_size: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    show: bool = True,
    use_pyvista: bool = True
) -> Any:
    """Visualize a comparison between two zones.
    
    Args:
        extractor: ZoneExtractor instance
        zone1_name: Name of the first zone
        zone2_name: Name of the second zone
        point_size: Size of points in the visualization
        alpha: Transparency of points/surfaces (0.0-1.0)
        color1: Color for the first zone
        color2: Color for the second zone
        figure_size: Size of the figure (width, height) in inches
        save_path: Optional path to save the visualization
        show: Whether to show the visualization
        use_pyvista: Whether to use PyVista for visualization (if available)
        
    Returns:
        Visualization plotter/figure
    """
    # Use PyVista if requested and available
    if use_pyvista and PYVISTA_AVAILABLE:
        return visualize_zones_pyvista(
            extractor,
            zone_names=[zone1_name, zone2_name],
            zone_colors={zone1_name: color1, zone2_name: color2},
            point_size=point_size,
            alpha=alpha,
            save_path=save_path,
            show=show
        )
        
    # Fall back to matplotlib
    return visualize_zones_matplotlib(
        extractor,
        zone_names=[zone1_name, zone2_name],
        zone_colors={zone1_name: color1, zone2_name: color2},
        point_size=point_size,
        alpha=alpha,
        figure_size=figure_size,
        save_path=save_path,
        show=show
    )


def visualize_zone_distribution(
    extractor: ZoneExtractor,
    property_name: str = 'point_count',
    zone_type: Optional[ZoneType] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    figure_size: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> Any:
    """Visualize the distribution of a property across zones.
    
    Args:
        extractor: ZoneExtractor instance
        property_name: Name of the property to visualize ('point_count', 'cell_count')
        zone_type: Optional filter for zone type
        min_value: Optional minimum value for filtering
        max_value: Optional maximum value for filtering
        figure_size: Size of the figure (width, height) in inches
        save_path: Optional path to save the visualization
        show: Whether to show the visualization
        
    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")
        
    # Get all zones
    zones = extractor.get_zone_info()
    
    # Filter by zone type if specified
    if zone_type is not None:
        zones = {name: info for name, info in zones.items() if info.zone_type == zone_type}
    
    # Extract property values
    zone_names = []
    property_values = []
    
    for name, info in zones.items():
        value = getattr(info, property_name, 0)
        
        # Apply filters
        if min_value is not None and value < min_value:
            continue
        if max_value is not None and value > max_value:
            continue
            
        zone_names.append(name)
        property_values.append(value)
    
    # Sort by property value
    sorted_indices = np.argsort(property_values)
    zone_names = [zone_names[i] for i in sorted_indices]
    property_values = [property_values[i] for i in sorted_indices]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Create bar chart
    bars = ax.barh(zone_names, property_values)
    
    # Color bars by zone type
    for i, name in enumerate(zone_names):
        zone_type = zones[name].zone_type
        color = 'blue'  # default
        if zone_type == ZoneType.VOLUME:
            color = 'royalblue'
        elif zone_type == ZoneType.BOUNDARY:
            color = 'forestgreen'
        elif zone_type == ZoneType.INTERFACE:
            color = 'darkorange'
            
        bars[i].set_color(color)
    
    # Add a legend for zone types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='royalblue', label='Volume'),
        Patch(facecolor='forestgreen', label='Boundary'),
        Patch(facecolor='darkorange', label='Interface')
    ]
    ax.legend(handles=legend_elements)
    
    # Add labels and title
    ax.set_xlabel(property_name.replace('_', ' ').title())
    ax.set_title(f'Zone {property_name.replace("_", " ").title()} Distribution')
    
    # Add value labels
    for i, v in enumerate(property_values):
        ax.text(v + 0.1, i, str(v), va='center')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
        
    return fig


def _meshio_to_pyvista(meshio_mesh: Any) -> Any:
    """Convert a meshio mesh to a PyVista mesh.
    
    Args:
        meshio_mesh: meshio Mesh object
        
    Returns:
        PyVista mesh or None if conversion failed
    """
    if not PYVISTA_AVAILABLE:
        return None
        
    try:
        # Check if the mesh has cells
        if not meshio_mesh.cells:
            # Create point cloud mesh
            return pv.PolyData(meshio_mesh.points)
            
        # Try to use PyVista's built-in conversion
        try:
            return pv.from_meshio(meshio_mesh)
        except Exception:
            # If that fails, try manual conversion
            pass
            
        # Manual conversion for common cell types
        points = meshio_mesh.points
        cells = []
        cell_types = []
        
        # Map meshio cell types to VTK cell types
        type_map = {
            'vertex': 1,
            'line': 3,
            'triangle': 5,
            'quad': 9,
            'tetra': 10,
            'hexahedron': 12,
            'wedge': 13,
            'pyramid': 14
        }
        
        # Process each cell block
        offset = 0
        for block in meshio_mesh.cells:
            vtk_type = type_map.get(block.type)
            if vtk_type is None:
                continue
                
            # Create connectivity array
            for cell in block.data:
                n_points = len(cell)
                conn = np.hstack(([n_points], cell))
                cells.extend(conn)
                cell_types.append(vtk_type)
                offset += n_points + 1
        
        # Create mesh
        if cells:
            cells = np.array(cells)
            return pv.UnstructuredGrid(cells, np.array(cell_types), points)
        else:
            # Fall back to point cloud
            return pv.PolyData(points)
            
    except Exception as e:
        logger.warning(f"Error converting meshio mesh to PyVista: {e}")
        return None


def visualize_mesh_with_zones(
    mesh_file: str,
    zone_names: Optional[List[str]] = None,
    use_pyvista: bool = True,
    **kwargs
) -> Any:
    """Convenience function to visualize a mesh with its zones.
    
    Args:
        mesh_file: Path to the mesh file
        zone_names: Optional list of zone names to visualize
        use_pyvista: Whether to use PyVista for visualization
        **kwargs: Additional arguments for the visualization function
        
    Returns:
        Visualization figure/plotter
    """
    # Create zone extractor
    extractor = ZoneExtractor(mesh_file)
    
    # Use the appropriate visualization function
    if use_pyvista and PYVISTA_AVAILABLE:
        return visualize_zones_pyvista(extractor, zone_names, **kwargs)
    else:
        return visualize_zones_matplotlib(extractor, zone_names, **kwargs)
