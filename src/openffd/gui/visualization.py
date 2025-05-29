"""Visualization component for the OpenFFD GUI.

This module provides the 3D visualization widget for rendering meshes and FFD control boxes.
"""

import logging
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QAction, QComboBox
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QIcon

# Import PyVista for 3D visualization
import pyvista as pv
from pyvista.qt import QtInteractor

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
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create PyVista plotter
        self.plotter = QtInteractor(self)
        
        # Create toolbar with view options
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(20, 20))
        
        # View presets
        view_label = QToolBar("View:")
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
        
        # Add toolbar and plotter to layout
        layout.addWidget(toolbar)
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
        
        # Clear existing FFD actor
        if self.ffd_actor is not None:
            self.plotter.remove_actor(self.ffd_actor)
            self.ffd_actor = None
        
        if control_points is not None and control_dim is not None:
            try:
                nx, ny, nz = control_dim
                
                # Create a structured grid for the FFD control box
                grid = pv.StructuredGrid()
                
                # Reshape control points for structured grid
                points_3d = np.zeros((nx, ny, nz, 3))
                idx = 0
                for i in range(nx):
                    for j in range(ny):
                        for k in range(nz):
                            points_3d[i, j, k] = control_points[idx]
                            idx += 1
                
                # Set grid dimensions and points
                grid.points = control_points
                grid.dimensions = [nx, ny, nz]
                
                # Add control points and grid to plotter
                self.ffd_actor = self.plotter.add_mesh(
                    grid, 
                    style='wireframe',
                    line_width=2, 
                    color='red'
                )
                
                # Add control points
                self.plotter.add_points(
                    control_points,
                    color='red',
                    point_size=8,
                    render_points_as_spheres=True
                )
                
                logger.info(f"FFD visualization updated with {len(control_points)} control points")
            except Exception as e:
                logger.error(f"Error visualizing FFD box: {str(e)}")
    
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
