"""Visualization component for the OpenFFD GUI.

This module provides the 3D visualization widget for rendering meshes and FFD control boxes.
"""

import logging
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QComboBox, QLabel
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QIcon

# Import PyVista for 3D visualization
import pyvista as pv

# Set PyQt6 as the backend before importing pyvista.qt
import os
os.environ['PYQT_API'] = 'pyqt6'

# Import PyVista Qt components - these are required for the visualization
import pyvistaqt
from pyvistaqt import QtInteractor
import pyvista as pv

# Import the existing CLI visualization logic
from openffd.visualization.ffd_viz import visualize_ffd_pyvista
from openffd.visualization.mesh_viz import visualize_mesh_with_patches_pyvista
from openffd.visualization.level_grid import create_level_grid, try_create_level_grid, create_grid_edges, create_boundary_edges
from openffd.visualization.zone_viz import visualize_zones_pyvista
from openffd.visualization.hierarchical_viz import visualize_hierarchical_ffd_pyvista

# Configure logging
logger = logging.getLogger(__name__)


class FFDVisualizationWidget(QWidget):
    """Widget for 3D visualization of meshes and FFD control boxes."""
    
    def __init__(self, parent=None):
        """Initialize the visualization widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.mesh_data = None
        self.mesh_points = None
        self.ffd_control_points = None
        self.control_dim = None
        self.mesh_actor = None
        self.ffd_actor = None
        self.ffd_points_actor = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toolbar with view options
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(20, 20))
        
        # View presets
        view_label = QLabel("View:")
        toolbar.addWidget(view_label)
        
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Isometric", "Top", "Front", "Right", "Bottom", "Back", "Left"])
        self.view_combo.currentTextChanged.connect(self._on_view_changed)
        toolbar.addWidget(self.view_combo)
        
        toolbar.addSeparator()
        
        # Action - Reset camera
        reset_action = QAction("Reset View", self)
        reset_action.triggered.connect(self.reset_camera)
        toolbar.addAction(reset_action)
        
        # Action - Toggle axes
        axes_action = QAction("Show Axes", self)
        axes_action.setCheckable(True)
        axes_action.setChecked(True)
        axes_action.triggered.connect(self._toggle_axes)
        toolbar.addAction(axes_action)
        
        # Add toolbar to layout
        layout.addWidget(toolbar)
        
        # Create the PyVista visualization component
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter)
        
        # Set up the plotter
        self.plotter.set_background("white")
        self.plotter.add_axes()
    
    def set_mesh(self, mesh_data: Any, mesh_points: np.ndarray):
        """Set the mesh to visualize.
        
        Args:
            mesh_data: The mesh data object
            mesh_points: Numpy array of mesh point coordinates
        """
        self.mesh_data = mesh_data
        self.mesh_points = mesh_points
        
        # Clear existing actors
        if self.mesh_actor is not None:
            self.plotter.remove_actor(self.mesh_actor)
            self.mesh_actor = None
        
        if mesh_data is not None:
            try:
                # Create a PyVista mesh from the points
                if hasattr(mesh_data, 'points') and hasattr(mesh_data, 'cells'):
                    # Use mesh directly if compatible
                    if hasattr(mesh_data, 'to_pyvista'):
                        pv_mesh = mesh_data.to_pyvista()
                    else:
                        # Create a point cloud from the points
                        pv_mesh = pv.PolyData(mesh_points)
                else:
                    # Create a point cloud from the mesh_points
                    pv_mesh = pv.PolyData(mesh_points)
                
                # Add mesh to plotter
                self.mesh_actor = self.plotter.add_mesh(
                    pv_mesh, 
                    style='surface', 
                    color='lightblue', 
                    opacity=0.7, 
                    show_edges=True
                )
                
                # Reset camera to focus on mesh
                self.plotter.reset_camera()
                logger.info(f"Mesh visualization updated with {len(mesh_points)} points")
            except Exception as e:
                logger.error(f"Error visualizing mesh: {str(e)}")
    
    def set_ffd(self, control_points: np.ndarray, control_dim: Tuple[int, int, int]):
        """Set the FFD control box to visualize.
        
        Args:
            control_points: Numpy array of control point coordinates
            control_dim: Tuple of control point dimensions (nx, ny, nz)
        """
        self.ffd_control_points = control_points
        self.control_dim = control_dim
        
        # Clear the plotter completely
        self.plotter.clear()
        
        # Calculate the bounding box from the control points
        if control_points is not None and len(control_points) > 0:
            min_coords = np.min(control_points, axis=0)
            max_coords = np.max(control_points, axis=0)
            bbox = (min_coords, max_coords)
            
            try:
                # Set up visualization using the proper FFD grid approach from CLI
                if self.mesh_points is not None:
                    # Visualize both mesh and FFD
                    self._setup_ffd_with_proper_grid(control_points, bbox, control_dim, True)
                else:
                    # Visualize only the FFD
                    self._setup_ffd_with_proper_grid(control_points, bbox, control_dim, False)
                    
                # Reset camera to show the scene
                self.plotter.reset_camera()
                logger.info(f"FFD visualization updated with {len(control_points)} control points")
            except Exception as e:
                logger.error(f"Error visualizing FFD box: {str(e)}")
                
    def _setup_ffd_with_proper_grid(self, control_points, bbox, dims, show_mesh=False):
        """Set up FFD visualization with proper grid connectivity using the CLI modules.
        
        Args:
            control_points: Array of control point coordinates
            bbox: Tuple of (min_coords, max_coords)
            dims: Dimensions of the FFD lattice (nx, ny, nz)
            show_mesh: Whether to show the mesh points
        """
        nx, ny, nz = dims
        
        # First, reshape control points using the level_grid module
        try:
            cp_grid = try_create_level_grid(control_points, dims)
            if cp_grid is None:
                # If reshaping failed, fallback to direct reshape
                logger.warning("Could not reshape control points using level_grid, falling back to direct reshape")
                cp_grid = control_points.reshape(nx, ny, nz, 3)
                
            # Add mesh points if available and requested
            if show_mesh and self.mesh_points is not None:
                # Create and add point cloud for mesh points
                mesh_cloud = pv.PolyData(self.mesh_points)
                self.mesh_actor = self.plotter.add_mesh(
                    mesh_cloud,
                    style='surface',
                    color='lightblue',
                    opacity=0.7,
                    point_size=3,
                    render_points_as_spheres=True
                )
            
            # Add control points
            cp_cloud = pv.PolyData(control_points)
            self.ffd_points_actor = self.plotter.add_points(
                cp_cloud,
                color='red',
                point_size=8,
                render_points_as_spheres=True
            )
            
            # Generate the grid edges based on the shaped control points
            edges = create_grid_edges(cp_grid)
            
            # Add each edge as a line to the plotter
            for edge in edges:
                start, end = edge[0], edge[1]
                line = pv.Line(start, end)
                self.plotter.add_mesh(line, color='red', line_width=2)
                
            # Add plot title
            self.plotter.add_text(f"FFD Control Box ({nx}×{ny}×{nz})\n{len(control_points)} control points", font_size=12)
            
        except Exception as e:
            logger.error(f"Error creating proper grid visualization: {e}")
            # Fall back to simple structured grid as last resort
            logger.warning("Falling back to simple structured grid")
            grid = pv.StructuredGrid()
            grid.points = control_points
            grid.dimensions = [nx, ny, nz]
            self.ffd_actor = self.plotter.add_mesh(grid, style='wireframe', line_width=2, color='red')
            self.ffd_points_actor = self.plotter.add_points(control_points, color='red', point_size=8, render_points_as_spheres=True)
    
    def toggle_mesh_visibility(self, visible: bool):
        """Toggle the visibility of the mesh.
        
        Args:
            visible: Whether the mesh should be visible
        """
        if self.mesh_actor is not None:
            self.plotter.set_actor_visibility(self.mesh_actor, visible)
    
    def toggle_ffd_visibility(self, visible: bool):
        """Toggle the visibility of the FFD control box.
        
        Args:
            visible: Whether the FFD control box should be visible
        """
        if self.ffd_actor is not None:
            self.plotter.set_actor_visibility(self.ffd_actor, visible)
    
    def reset_camera(self):
        """Reset the camera to focus on the scene."""
        self.plotter.reset_camera()
    
    def _toggle_axes(self, checked: bool):
        """Toggle the visibility of the axes.
        
        Args:
            checked: Whether the axes should be visible
        """
        if checked:
            self.plotter.add_axes()
        else:
            self.plotter.hide_axes()
    
    def _on_view_changed(self, view_type: str):
        """Handle view type changes.
        
        Args:
            view_type: The type of view to set
        """
        if view_type == "Isometric":
            self.plotter.view_isometric()
        elif view_type == "Top":
            self.plotter.view_xy()
        elif view_type == "Front":
            self.plotter.view_yz()
        elif view_type == "Right":
            self.plotter.view_xz()
        elif view_type == "Bottom":
            self.plotter.view_xy(negative=True)
        elif view_type == "Back":
            self.plotter.view_yz(negative=True)
        elif view_type == "Left":
            self.plotter.view_xz(negative=True)
