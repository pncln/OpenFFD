"""Configuration module for OpenFFD.

This module provides configuration classes for the FFD control box generation process.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any


@dataclass
class FFDConfig:
    """Configuration for FFD control box generation.
    
    Attributes:
        dims: Dimensions of the FFD control lattice (Nx, Ny, Nz)
        margin: Margin padding around the mesh
        custom_bounds: Optional custom bounds for the FFD box
            List of tuples (min, max) for each dimension, or None for automatic
        output_file: Output filename
        export_xyz: Whether to also export in XYZ format
        visualize: Whether to visualize the FFD box
        visualization_options: Options for visualization
        debug: Whether to enable debug output
        force_ascii: Whether to force ASCII reading for Fluent mesh
        force_binary: Whether to force binary reading for Fluent mesh
    """
    
    dims: Tuple[int, int, int]
    margin: float = 0.0
    custom_bounds: Optional[List[Optional[Tuple[Optional[float], Optional[float]]]]] = None
    output_file: str = "ffd_box.3df"
    export_xyz: bool = False
    visualize: bool = False
    visualization_options: Dict[str, Any] = field(default_factory=dict)
    debug: bool = False
    force_ascii: bool = False
    force_binary: bool = False
    
    def __post_init__(self) -> None:
        """Validate the configuration after initialization."""
        if len(self.dims) != 3:
            raise ValueError("Dimensions must be a tuple of 3 integers")
        
        if self.margin < 0:
            raise ValueError("Margin must be non-negative")
        
        if self.custom_bounds is not None and len(self.custom_bounds) != 3:
            raise ValueError("Custom bounds must be a list of 3 tuples or None")
            
        # Set default visualization options if not provided
        default_viz_options = {
            'show_mesh': False,
            'mesh_only': False,
            'mesh_size': 2.0,
            'mesh_alpha': 0.3,
            'mesh_color': 'blue',
            'ffd_point_size': 10.0,
            'ffd_alpha': 1.0,
            'ffd_color': 'b',
            'show_full_ffd_grid': False,
            'ffd_line_width': 3.0,
            'show_surface': True,
            'show_mesh_edges': False,
            'surface_alpha': 1.0,
            'surface_color': 'gray',
            'hide_control_points': False,
            'control_point_size': 30.0,
            'lattice_width': 1.5,
            'view_angle': None,
            'view_axis': None,
            'auto_scale': True,
            'scale_factor': None,
            'zoom_region': None,
            'zoom_factor': 1.0,
            'show_original_mesh': False,
            'point_size': 2.0,
            'detail_level': 'medium',
            'max_triangles': 10000,
            'max_points': 5000,
        }
        
        # Update with provided options
        for key, value in default_viz_options.items():
            if key not in self.visualization_options:
                self.visualization_options[key] = value
