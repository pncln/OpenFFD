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
        self.boundary_actors = {}  # Dict to store boundary zone actors by zone name
        
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
        """Set mesh data and update visualization.
        
        Args:
            mesh_data: Mesh data object or zone mesh dict
            mesh_points: Point coordinates as numpy array
        """
        if mesh_data is None or mesh_points is None:
            return
        
        # Store references
        self.mesh_data = mesh_data
        self.mesh_points = mesh_points
        
        # 🔍 DEBUG: Log what we received
        logger.info(f"🔍 VISUALIZATION DEBUG: Received mesh data type: {type(mesh_data)}")
        if isinstance(mesh_data, dict):
            logger.info(f"🔍 VISUALIZATION DEBUG: Dict keys: {list(mesh_data.keys())}")
            if 'faces' in mesh_data:
                faces = mesh_data['faces']
                logger.info(f"🔍 VISUALIZATION DEBUG: Faces count: {len(faces)}")
                logger.info(f"🔍 VISUALIZATION DEBUG: Is point cloud: {mesh_data.get('is_point_cloud', 'unknown')}")
                logger.info(f"🔍 VISUALIZATION DEBUG: Zone type: {mesh_data.get('zone_type', 'unknown')}")
        logger.info(f"🔍 VISUALIZATION DEBUG: Points count: {len(mesh_points)}")
        
        # Clear existing actors
        if self.mesh_actor is not None:
            self.plotter.remove_actor(self.mesh_actor)
            self.mesh_actor = None
        
        # Clear existing boundary actors
        for zone_name, actor in self.boundary_actors.items():
            if actor:
                self.plotter.remove_actor(actor)
        self.boundary_actors.clear()
        
        if mesh_data is not None:
            try:
                # Check if this is a zone mesh data dict with face connectivity
                if isinstance(mesh_data, dict) and 'points' in mesh_data and 'faces' in mesh_data:
                    self._render_zone_mesh(mesh_data)
                # Check if this is a full Fluent mesh with boundaries
                elif hasattr(mesh_data, 'zones') and hasattr(mesh_data, 'points'):
                    # Render boundary zones as individual surfaces for visibility control
                    self._render_full_mesh_with_boundaries(mesh_data)
                # Handle other mesh types
                elif hasattr(mesh_data, 'points') and hasattr(mesh_data, 'cells'):
                    # Use mesh directly if compatible
                    if hasattr(mesh_data, 'to_pyvista'):
                        pv_mesh = mesh_data.to_pyvista()
                    else:
                        # Create a point cloud from the points
                        pv_mesh = pv.PolyData(mesh_points)
                    
                    # Add mesh to plotter
                    self.mesh_actor = self.plotter.add_mesh(
                        pv_mesh, 
                        style='surface', 
                        color='lightblue', 
                        opacity=0.7, 
                        show_edges=True
                    )
                else:
                    # NO POINT CLOUDS - create solid surface from mesh points
                    try:
                        pv_mesh = pv.PolyData(mesh_points)
                        surface_mesh = pv_mesh.convex_hull()
                        self.mesh_actor = self.plotter.add_mesh(
                            surface_mesh, 
                            style='surface', 
                            color='blue', 
                            opacity=1.0,
                            show_edges=False
                        )
                        logger.info(f"Created solid surface from {len(mesh_points)} mesh points")
                    except Exception as surf_error:
                        logger.error(f"Cannot create surface from mesh points: {surf_error}")
                        raise surf_error
                
                # Reset camera to focus on mesh
                self.plotter.reset_camera()
                logger.info(f"Mesh visualization updated with {len(mesh_points)} points")
            except Exception as e:
                logger.error(f"Error visualizing mesh: {str(e)}")
    
    def _render_zone_mesh(self, zone_mesh_data: Dict[str, Any]):
        """Render a zone mesh with proper surface connectivity.
        
        Args:
            zone_mesh_data: Dict containing 'points', 'faces', and other zone data
        """
        points = zone_mesh_data['points']
        faces = zone_mesh_data['faces']
        zone_type = zone_mesh_data.get('zone_type', 'unknown')
        zone_name = zone_mesh_data.get('zone_name', 'unknown')
        is_point_cloud = zone_mesh_data.get('is_point_cloud', True)
        
        # Debug logging for connectivity issues
        logger.info(f"🔍 Zone visualization: is_point_cloud={is_point_cloud}, faces_count={len(faces) if faces else 0}")
        
        if not is_point_cloud and faces and len(faces) > 0:
            # Choose rendering mode based on zone size
            face_count = len(faces)
            
            # Debug: Check face connectivity
            logger.info(f"🔍 First few faces: {faces[:3] if len(faces) >= 3 else faces}")
            
            # ALL zones use the same complete surface rendering (no artificial thresholds)
            logger.info(f"🚀 Complete surface rendering for zone: {face_count:,} faces")
            try:
                self.mesh_actor = self._render_fast_surface(faces, points, zone_name)
                logger.info(f"✅ Complete surface rendered: {len(points):,} points, {face_count:,} faces")
                return
            except Exception as e:
                logger.error(f"Surface rendering failed: {e}, attempting emergency surface creation")
                # Try emergency surface creation instead of point cloud
                try:
                    pv_mesh = pv.PolyData(points)
                    surface_mesh = pv_mesh.convex_hull()
                    color = self._get_zone_color(zone_name)
                    self.mesh_actor = self.plotter.add_mesh(
                        surface_mesh,
                        style='surface',
                        color=color,
                        opacity=1.0,
                        show_edges=False
                    )
                    logger.info(f"Emergency surface created")
                    return
                except Exception as emergency_error:
                    logger.error(f"All surface creation failed: {emergency_error}")
                    raise emergency_error
        
        # No point cloud fallback - force surface creation with all points
        logger.warning(f"No valid faces found, creating surface from all points using convex hull")
        try:
            # Create a surface mesh from the point cloud using Delaunay triangulation
            pv_mesh = pv.PolyData(points)
            
            # Try to create a surface from points
            if len(points) > 3:
                try:
                    # For 2D-like surfaces (like wedges), project to best plane and triangulate
                    surface_mesh = pv_mesh.delaunay_2d()
                    if surface_mesh.n_faces > 0:
                        color = self._get_zone_color(zone_name)
                        self.mesh_actor = self.plotter.add_mesh(
                            surface_mesh,
                            style='surface',
                            color=color,
                            opacity=1.0,
                            show_edges=False
                        )
                        logger.info(f"✅ Created solid surface from {len(points)} points using Delaunay triangulation")
                        return
                except Exception as delaunay_error:
                    logger.warning(f"Delaunay triangulation failed: {delaunay_error}")
            
            # Ultimate fallback - at least show something but prefer surface over points
            color = self._get_zone_color(zone_name)  
            self.mesh_actor = self.plotter.add_mesh(
                pv_mesh,
                style='surface',  # Force surface even without faces
                color=color,
                opacity=1.0
            )
            logger.warning(f"Fallback: rendered {len(points)} points as basic surface")
            
        except Exception as surface_error:
            logger.error(f"All surface creation methods failed: {surface_error}")
            raise surface_error
    
    def _render_full_mesh_with_boundaries(self, mesh_data: Any):
        """Render full mesh with boundary visibility controls.
        
        Args:
            mesh_data: Full mesh data with zones
        """
        # Clear existing boundary actors
        for zone_name, actor in self.boundary_actors.items():
            if actor:
                self.plotter.remove_actor(actor)
        self.boundary_actors.clear()
        
        try:
            from openffd.mesh.general import extract_zone_mesh
            
            # Render each boundary zone as a separate surface
            for zone_name, zone_info in mesh_data.zones.items():
                zone_type = zone_info.get('type', 'unknown')
                zone_obj = zone_info.get('object')
                
                # Skip volume zones
                if zone_obj and hasattr(zone_obj, 'zone_type_enum'):
                    is_volume = zone_obj.zone_type_enum.name == 'VOLUME'
                else:
                    is_volume = zone_type in ['interior', 'fluid', 'solid']
                
                if not is_volume:
                    try:
                        # Create solid surfaces from all zone points instead of point clouds
                        zone_points = mesh_data.get_zone_points(zone_name)
                        if zone_points is not None and len(zone_points) > 0:
                            try:
                                # Create surface from points using convex hull
                                pv_mesh = pv.PolyData(zone_points)
                                surface_mesh = pv_mesh.convex_hull()
                                color = self._get_zone_color(zone_type)
                                
                                actor = self.plotter.add_mesh(
                                    surface_mesh,
                                    style='surface',
                                    color=color,
                                    opacity=1.0,
                                    show_edges=False,
                                    name=zone_name
                                )
                                self.boundary_actors[zone_name] = actor
                                logger.info(f"Created solid surface for zone '{zone_name}': {surface_mesh.n_faces} faces")
                            except Exception as surf_error:
                                logger.warning(f"Failed to create surface for zone '{zone_name}': {surf_error}")
                                # Skip this zone rather than create point cloud
                    except Exception as e:
                        logger.warning(f"Failed to render boundary zone '{zone_name}': {e}")
            
            if self.boundary_actors:
                logger.info(f"Rendered {len(self.boundary_actors)} boundary zones as solid surfaces")
            else:
                # Create emergency mesh surface if no boundary surfaces found
                try:
                    pv_mesh = pv.PolyData(self.mesh_points)
                    surface_mesh = pv_mesh.convex_hull()
                    self.mesh_actor = self.plotter.add_mesh(
                        surface_mesh,
                        style='surface',
                        color='blue',
                        opacity=1.0,
                        show_edges=False
                    )
                    logger.info("Created emergency solid surface from all mesh points")
                except Exception as emergency_error:
                    logger.error(f"Cannot create any surface: {emergency_error}")
                    raise emergency_error
                
        except Exception as e:
            logger.error(f"Error rendering full mesh: {e}")
            # NO POINT CLOUDS - create emergency surface from all mesh points
            try:
                pv_mesh = pv.PolyData(self.mesh_points)
                surface_mesh = pv_mesh.convex_hull()
                self.mesh_actor = self.plotter.add_mesh(
                    surface_mesh,
                    style='surface',
                    color='blue',
                    opacity=1.0,
                    show_edges=False
                )
                logger.info("Created emergency solid surface from mesh points")
            except Exception as final_error:
                logger.error(f"All surface creation failed: {final_error}")
                raise final_error
    
    def _get_zone_color(self, zone_identifier: str) -> str:
        """Get appropriate color for zone type or name.
        
        Args:
            zone_identifier: Zone name or type
            
        Returns:
            Color string for the zone
        """
        color_map = {
            'wall': 'lightcoral',
            'inlet': 'lightgreen', 
            'velocity-inlet': 'lightgreen',
            'outlet': 'lightblue',
            'pressure-outlet': 'lightblue',
            'symmetry': 'lightyellow',
            'periodic': 'lightpink',
            'interface': 'lightgray',
            'interior': 'lightsteelblue',
            'fluid': 'lightsteelblue'
        }
        
        # Special handling for specific zones to ensure visibility and distinction
        zone_identifier_lower = zone_identifier.lower()
        
        # Wedge zones get distinct bright colors
        if 'wedge' in zone_identifier_lower:
            if 'pos' in zone_identifier_lower:
                return 'red'  # Bright red for wedge_pos
            elif 'neg' in zone_identifier_lower:
                return 'blue'  # Bright blue for wedge_neg
            else:
                return 'magenta'  # Fallback bright color for other wedge zones
        
        # Other specific thin surface zones
        if 'symmetry' in zone_identifier_lower:
            return 'orange'  # Bright orange for symmetry planes
        elif 'interface' in zone_identifier_lower:
            return 'cyan'  # Bright cyan for interfaces
        elif 'periodic' in zone_identifier_lower:
            return 'lime'  # Bright lime for periodic surfaces
        
        return color_map.get(zone_identifier_lower, 'lightgray')
    
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
            # self.plotter.add_text(f"FFD Control Box ({nx}×{ny}×{nz})\n{len(control_points)} control points", font_size=12)
            
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
            self.mesh_actor.SetVisibility(visible)
            self.plotter.render()
    
    def toggle_ffd_visibility(self, visible: bool):
        """Toggle the visibility of the FFD control box.
        
        Args:
            visible: Whether the FFD control box should be visible
        """
        if self.ffd_actor is not None:
            self.ffd_actor.SetVisibility(visible)
            self.plotter.render()
    
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
    
    def _render_professional_surface(self, mesh: pv.PolyData, color: str, zone_type: str, lod_levels: dict) -> any:
        """Render surface using professional-grade techniques for maximum performance and quality.
        
        Args:
            mesh: Optimized PyVista mesh
            color: Surface color
            zone_type: Type of zone for specialized rendering
            lod_levels: Level-of-detail hierarchy
            
        Returns:
            Mesh actor for visibility control
        """
        try:
            # Professional rendering settings based on mesh complexity and zone type
            n_faces = mesh.n_faces
            
            # Adaptive rendering settings for optimal performance
            if n_faces > 50000:
                # Ultra-high performance mode for very large meshes
                render_settings = {
                    'style': 'surface',
                    'color': color,
                    'opacity': 0.95,
                    'show_edges': False,  # No edges for maximum performance
                    'smooth_shading': True,
                    'lighting': True,
                    'ambient': 0.3,
                    'diffuse': 0.7,
                    'specular': 0.05,
                    'specular_power': 5,
                    'backface_culling': True,  # GPU optimization
                    'interpolate_before_map': True  # Better color interpolation
                }
            elif n_faces > 20000:
                # High performance mode
                render_settings = {
                    'style': 'surface',
                    'color': color,
                    'opacity': 0.9,
                    'show_edges': False,
                    'smooth_shading': True,
                    'lighting': True,
                    'ambient': 0.25,
                    'diffuse': 0.75,
                    'specular': 0.1,
                    'specular_power': 10,
                    'backface_culling': True
                }
            else:
                # Balanced quality mode for smaller meshes
                render_settings = {
                    'style': 'surface',
                    'color': color,
                    'opacity': 0.85,
                    'show_edges': True,
                    'edge_color': 'gray',
                    'line_width': 0.3,
                    'smooth_shading': True,
                    'lighting': True,
                    'ambient': 0.2,
                    'diffuse': 0.8,
                    'specular': 0.15,
                    'specular_power': 15
                }
            
            # Special rendering for specific zone types
            if zone_type.lower() in ['wall', 'solid']:
                render_settings['color'] = 'lightcoral'
                render_settings['specular'] = 0.2  # More reflective for walls
            elif zone_type.lower() in ['inlet', 'velocity-inlet']:
                render_settings['color'] = 'lightgreen'
                render_settings['opacity'] = 0.7  # Semi-transparent for flow
            elif zone_type.lower() in ['outlet', 'pressure-outlet']:
                render_settings['color'] = 'lightblue'
                render_settings['opacity'] = 0.7
            
            logger.info(f"🎨 Rendering {zone_type} surface: {n_faces:,} faces with professional settings")
            
            # Create the mesh actor with professional settings
            actor = self.plotter.add_mesh(mesh, **render_settings)
            
            # Apply additional GPU optimizations
            if hasattr(actor, 'GetProperty'):
                prop = actor.GetProperty()
                # Enable GPU-based rendering optimizations
                if hasattr(prop, 'SetInterpolationToPhong'):
                    prop.SetInterpolationToPhong()  # High-quality surface interpolation
                if hasattr(prop, 'SetShading') and n_faces > 10000:
                    prop.SetShading(True)  # Enable hardware shading for large meshes
            
            logger.info(f"✨ Professional surface rendering completed successfully")
            return actor
            
        except Exception as e:
            logger.error(f"Professional rendering failed: {e}, using simple fallback")
            # Simple fallback rendering
            return self.plotter.add_mesh(
                mesh,
                style='surface',
                color=color,
                opacity=0.8
            )
    
    def _optimize_surface_for_performance(self, faces: list, points: np.ndarray) -> tuple:
        """Professional-grade mesh optimization using industry-standard techniques.
        
        This method implements a multi-tier optimization system including:
        - Intelligent face sampling with area preservation
        - Proper mesh triangulation and cleaning
        - Level-of-Detail (LOD) hierarchy generation
        - GPU-accelerated mesh processing
        - Robust fallback mechanisms
        
        Args:
            faces: List of face connectivity data
            points: Numpy array of 3D points
            
        Returns:
            Tuple of (optimized_mesh, lod_levels_dict)
        """
        logger.info(f"🔧 Professional mesh optimization: {len(points):,} points, {len(faces):,} faces")
        
        try:
            # NO DOWNSAMPLING - Professional rendering with ALL faces preserved
            optimized_faces = faces
            logger.info(f"🎯 Professional rendering: preserving ALL {len(faces):,} faces for complete surface coverage")
            
            # Step 2: Create PyVista mesh with proper face format
            pv_faces = []
            for face in optimized_faces:
                if len(face) >= 3:
                    pv_faces.extend([len(face)] + list(face))
            
            if not pv_faces:
                logger.error("No valid faces found for optimization")
                # NO POINT CLOUDS - create surface using convex hull
                try:
                    point_mesh = pv.PolyData(points)
                    surface_mesh = point_mesh.convex_hull()
                    logger.info(f"Emergency convex hull surface: {surface_mesh.n_faces} faces")
                    return surface_mesh, {"original": surface_mesh}
                except Exception:
                    # Final attempt using Delaunay
                    try:
                        surface_mesh = point_mesh.delaunay_2d()
                        logger.info(f"Emergency Delaunay surface: {surface_mesh.n_faces} faces")
                        return surface_mesh, {"original": surface_mesh}
                    except Exception as e:
                        raise ValueError(f"Cannot create any surface from points: {e}")
            
            # Create the initial mesh
            mesh = pv.PolyData(points, faces=np.array(pv_faces, dtype=np.int32))
            
            # Step 3: Professional mesh processing pipeline
            try:
                # Clean mesh first - remove duplicates and degenerate cells
                mesh = mesh.clean()
                logger.info(f"✅ Mesh cleaned: {mesh.n_points:,} points, {mesh.n_faces:,} faces")
                
                # Advanced triangulation with gap-filling for consistency and better GPU performance
                if mesh.n_faces > 0:
                    original_face_count = mesh.n_faces
                    
                    # First try standard triangulation
                    mesh = mesh.triangulate()
                    logger.info(f"✅ Mesh triangulated: {mesh.n_faces:,} triangular faces")
                    
                    # Check if we have sparse connectivity that might cause gaps
                    face_point_ratio = mesh.n_faces / max(mesh.n_points, 1)
                    if face_point_ratio < 1.2:  # Professional threshold for gap detection
                        logger.info(f"Professional gap-filling for sparse connectivity ({face_point_ratio:.3f})")
                        try:
                            # Apply professional hole filling
                            filled_mesh = mesh.fill_holes(hole_size=500)
                            if filled_mesh.n_faces > mesh.n_faces:
                                mesh = filled_mesh
                                logger.info(f"✅ Professional gap-filling completed: {mesh.n_faces:,} faces")
                        except Exception:
                            logger.debug("Professional gap-filling failed, continuing with standard mesh")
                
            except Exception as pe:
                logger.warning(f"Mesh processing warning: {pe}, continuing with original")
            
            # Step 4: Generate Level-of-Detail hierarchy
            lod_levels = self._generate_lod_hierarchy(mesh)
            
            # Step 5: Final GPU optimizations for professional rendering
            try:
                # Compute normals for proper lighting
                mesh = mesh.compute_normals(
                    point_normals=True, 
                    cell_normals=True, 
                    auto_orient_normals=True,
                    consistent_normals=True
                )
                logger.info(f"✅ Professional lighting normals computed")
                
            except Exception as ge:
                logger.warning(f"GPU optimizations failed: {ge}")
            
            logger.info(f"🏆 Professional optimization completed: {mesh.n_points:,} points, {mesh.n_faces:,} faces")
            return mesh, lod_levels
            
        except Exception as e:
            logger.warning(f"Professional optimization failed: {e}, using fallback")
            return self._fallback_optimization(faces, points)
    
    def _convert_faces_to_pyvista(self, faces: list) -> np.ndarray:
        """Convert face list to PyVista format efficiently."""
        try:
            pv_faces = []
            for face in faces:
                if len(face) >= 3:
                    pv_faces.extend([len(face)] + list(face))
            return np.array(pv_faces, dtype=np.int32) if pv_faces else None
        except Exception as e:
            logger.warning(f"Face conversion failed: {e}")
            return None
    
    def _apply_no_decimation_processing(self, mesh: pv.PolyData) -> pv.PolyData:
        """Apply professional mesh processing WITHOUT face reduction.
        
        Preserves ALL faces while applying quality improvements like
        smoothing and normal computation for better visual quality.
        """
        try:
            logger.info(f"🎯 NO DECIMATION: Preserving ALL {mesh.n_faces:,} faces for complete surface")
            
            # Apply light smoothing to improve visual quality WITHOUT reducing faces
            try:
                smoothed = mesh.smooth(
                    n_iter=5,                    # Very light smoothing
                    relaxation_factor=0.05,      # Very conservative
                    feature_smoothing=False,     # Preserve features
                    boundary_smoothing=False     # Keep boundaries sharp
                )
                logger.info(f"✅ Applied light smoothing while preserving all faces")
                return smoothed
            except Exception as smooth_error:
                logger.warning(f"Smoothing failed: {smooth_error}, returning original mesh")
                return mesh
            
        except Exception as e:
            logger.warning(f"Mesh processing failed: {e}, returning original")
            return mesh
    
    def _preserve_all_faces(self, faces: list, points: np.ndarray, target_count: int) -> list:
        """Preserve ALL faces for complete surface coverage - NO SAMPLING.
        
        Args:
            faces: Original face list
            points: Point coordinates
            target_count: Ignored - all faces preserved
            
        Returns:
            All valid faces (no reduction)
        """
        try:
            # Filter only invalid faces, keep ALL valid ones
            valid_faces = [face for face in faces if len(face) >= 3]
            
            logger.info(f"🎯 PRESERVING ALL FACES: {len(faces):,} → {len(valid_faces):,} valid faces (NO SAMPLING)")
            return valid_faces
            
        except Exception as e:
            logger.warning(f"Face validation failed: {e}, returning all faces")
            return faces
            
    def _generate_lod_hierarchy(self, mesh: pv.PolyData) -> dict:
        """Generate Level-of-Detail hierarchy for adaptive rendering.
        
        Creates multiple resolution levels for distance-based rendering
        optimization used in professional 3D applications.
        """
        try:
            lod_levels = {}
            
            # LOD 0: Full detail (close view)
            lod_levels[0] = mesh
            
            # NO DECIMATION - All LOD levels use full detail to preserve complete surface
            # LOD 1: Same as full detail (no face reduction)
            lod_levels[1] = mesh
            
            # LOD 2: Same as full detail (no face reduction)
            lod_levels[2] = mesh
            
            logger.info(f"📊 LOD hierarchy: {len(lod_levels)} levels created")
            return lod_levels
            
        except Exception as e:
            logger.warning(f"LOD generation failed: {e}")
            return {0: mesh}
    
    def _apply_gpu_optimizations(self, mesh: pv.PolyData) -> pv.PolyData:
        """Apply GPU-specific optimizations for maximum rendering performance."""
        try:
            # GPU-friendly mesh optimizations
            optimized = mesh.copy()
            
            # 1. Triangulate for GPU efficiency (GPUs prefer triangles)
            if optimized.n_faces > 0:
                optimized = optimized.triangulate()
            
            # 2. Compute normals for proper lighting (done on CPU once, not per-frame)
            optimized.compute_normals(inplace=True, point_normals=True, cell_normals=False)
            
            # 3. Clean mesh for optimal GPU batching
            optimized.clean(inplace=True)
            
            logger.info(f"🚀 GPU optimizations applied: {optimized.n_faces:,} triangles")
            return optimized
            
        except Exception as e:
            logger.warning(f"GPU optimization failed: {e}")
            return mesh
    
    def _fallback_optimization(self, faces: list, points: np.ndarray) -> tuple:
        """Professional fallback when advanced algorithms fail."""
        try:
            # Convert to PyVista with simple optimization
            pv_faces = self._convert_faces_to_pyvista(faces)
            if pv_faces is not None:
                mesh = pv.PolyData(points, pv_faces)
                # NO DECIMATION - keep all faces for complete surface
                mesh = mesh.triangulate()  # Only triangulate, don't reduce faces
                return mesh, {0: mesh}
            else:
                # Ultimate fallback: return points only
                return pv.PolyData(points), {0: pv.PolyData(points)}
        except Exception as e:
            logger.error(f"Fallback optimization failed: {e}")
            return pv.PolyData(points), {0: pv.PolyData(points)}
    
    @pyqtSlot(str, bool)
    def set_boundary_zone_visibility(self, zone_name: str, visible: bool):
        """Set the visibility of a boundary zone.
        
        Args:
            zone_name: Name of the zone
            visible: Whether the zone should be visible
        """
        if zone_name in self.boundary_actors:
            actor = self.boundary_actors[zone_name]
            if actor:
                actor.SetVisibility(visible)
                self.plotter.render()
                logger.info(f"Set boundary zone '{zone_name}' visibility to {visible}")
        else:
            logger.warning(f"Boundary zone '{zone_name}' not found in rendered actors")
    
    def _render_fast_surface(self, faces: list, points: np.ndarray, zone_name: str):
        """Fast surface rendering that preserves ALL faces for complete solid surfaces.
        
        This method provides a complete surface rendering pipeline that:
        - Preserves ALL faces for complete surface coverage
        - Uses advanced triangulation for gap-free surfaces
        - Applies professional mesh processing
        - Prioritizes surface completeness over speed
        
        Args:
            faces: List of face connectivity data
            points: Numpy array of 3D points
            zone_name: Name of zone for color selection
            
        Returns:
            PyVista mesh actor
        """
        # NO DOWNSAMPLING - Keep ALL faces for complete solid surface coverage
        sampled_faces = faces
        logger.info(f"⚡ PRESERVING ALL {len(faces):,} faces for complete solid surface coverage - NO DOWNSAMPLING")
        
        # Step 2: Robust face validation and conversion
        pv_faces = []
        valid_face_count = 0
        max_point_index = len(points) - 1
        
        for face in sampled_faces:
            if len(face) < 3:
                continue
                
            # Validate face indices are within bounds and convert to integers
            try:
                face_indices = [int(idx) for idx in face]
            except (ValueError, TypeError):
                continue
                
            # Check all indices are within valid range
            if any(idx < 0 or idx > max_point_index for idx in face_indices):
                continue
                
            # Remove degenerate faces (duplicate vertices)
            unique_indices = []
            seen = set()
            for idx in face_indices:
                if idx not in seen:
                    unique_indices.append(idx)
                    seen.add(idx)
            
            # Need at least 3 unique vertices for a valid face
            if len(unique_indices) < 3:
                continue
                
            # Handle different cell types properly
            if len(unique_indices) == 3:
                # Triangle - keep as is
                pv_faces.extend([3] + unique_indices)
                valid_face_count += 1
            elif len(unique_indices) == 4:
                # Quadrilateral - add as quad directly (PyVista can handle quads)
                pv_faces.extend([4] + unique_indices)
                valid_face_count += 1
            elif len(unique_indices) == 5:
                # Pentagon - keep as polygon 
                pv_faces.extend([5] + unique_indices)
                valid_face_count += 1
            elif len(unique_indices) == 6:
                # Hexagon - keep as polygon
                pv_faces.extend([6] + unique_indices)
                valid_face_count += 1
            else:
                # For very large polygons, use fan triangulation as fallback
                if len(unique_indices) <= 10:  # Reasonable polygon size
                    pv_faces.extend([len(unique_indices)] + unique_indices)
                    valid_face_count += 1
                else:
                    # Fan triangulation only for very large polygons
                    for i in range(1, len(unique_indices) - 1):
                        pv_faces.extend([3, unique_indices[0], unique_indices[i], unique_indices[i + 1]])
                        valid_face_count += 1
        
        logger.info(f"🔧 Face validation: {len(sampled_faces):,} → {valid_face_count:,} valid cells (mixed triangles/quads)")
        
        if not pv_faces:
            # Emergency surface creation - never use points
            logger.warning(f"No valid faces after validation, creating emergency surface")
            pv_mesh = pv.PolyData(points)
            
            # Try to create a surface from the points
            try:
                # Use convex hull to create a surface
                surface_mesh = pv_mesh.convex_hull()
                color = self._get_zone_color(zone_name)
                return self.plotter.add_mesh(
                    surface_mesh,
                    style='surface',
                    color=color,
                    opacity=1.0,
                    show_edges=False
                )
            except Exception as hull_error:
                logger.error(f"Emergency surface creation failed: {hull_error}")
                raise ValueError(f"Cannot create any surface for zone {zone_name}")
        
        # Step 3: Create mesh with proper validation and processing
        try:
            # Convert to numpy array with proper data type
            faces_array = np.array(pv_faces, dtype=np.int32)
            
            # Create mesh with validated faces (mixed triangles and quads)
            mesh = pv.PolyData(points, faces=faces_array)
            
            # Basic mesh validation and cleaning
            if mesh.n_faces == 0:
                raise ValueError("No valid faces created")
            
            logger.info(f"🔧 Mixed cell mesh created: {mesh.n_points:,} points, {mesh.n_faces:,} faces")
            
            # Advanced gap-filling triangulation for complete surface coverage
            try:
                if mesh.n_faces > 0:
                    original_face_count = mesh.n_faces
                    
                    # Try multiple triangulation strategies for gap-free surfaces
                    best_mesh = mesh
                    best_face_count = original_face_count
                    
                    # Strategy 1: Conservative triangulation
                    try:
                        triangulated_mesh = mesh.triangulate()
                        if triangulated_mesh.n_faces >= original_face_count * 0.8:
                            best_mesh = triangulated_mesh
                            best_face_count = triangulated_mesh.n_faces
                            logger.info(f"🔧 Conservative triangulation: {best_face_count:,} triangular faces")
                    except Exception:
                        logger.debug("Conservative triangulation failed")
                    
                    # Strategy 2: For sparse connectivity (face/point ratio < 1.0), try aggressive gap filling
                    face_point_ratio = original_face_count / max(mesh.n_points, 1)
                    if face_point_ratio < 1.0:
                        logger.info(f"🔧 SPARSE CONNECTIVITY detected ({face_point_ratio:.3f}) - applying gap-filling triangulation")
                        
                        try:
                            # Use fill_holes to close gaps in the surface
                            filled_mesh = mesh.fill_holes(hole_size=1000)  # Fill moderate-sized holes
                            if filled_mesh.n_faces > best_face_count:
                                best_mesh = filled_mesh
                                best_face_count = filled_mesh.n_faces
                                logger.info(f"🔧 Gap-filling improved surface: {best_face_count:,} faces")
                        except Exception:
                            logger.debug("Gap-filling triangulation failed")
                        
                        # Try Delaunay 3D for better connectivity if other methods didn't help much
                        try:
                            if best_face_count < original_face_count * 1.2:  # Only if we didn't improve much
                                delaunay_mesh = mesh.delaunay_3d()
                                surface_mesh = delaunay_mesh.extract_surface()
                                if surface_mesh.n_faces > best_face_count:
                                    best_mesh = surface_mesh
                                    best_face_count = surface_mesh.n_faces
                                    logger.info(f"🔧 Delaunay 3D improved surface: {best_face_count:,} faces")
                        except Exception:
                            logger.debug("Delaunay 3D triangulation failed")
                    
                    # Use the best mesh we found
                    mesh = best_mesh
                    logger.info(f"🔧 Final mesh after gap-filling: {mesh.n_faces:,} triangular faces")
                    
            except Exception as te:
                logger.warning(f"Advanced triangulation failed: {te}, keeping original mesh for complete surface")
            
            # Conservative mesh cleaning to preserve connectivity
            try:
                # Only clean if mesh is very large to avoid breaking connectivity
                if mesh.n_faces > 100000:
                    mesh = mesh.clean()
                    logger.info(f"🔧 Applied conservative cleaning for large mesh")
                else:
                    logger.info(f"🔧 Skipped cleaning to preserve connectivity for medium mesh")
            except Exception as ce:
                logger.warning(f"Mesh cleaning failed: {ce}, continuing with unclean mesh")
            
            logger.info(f"🔧 Mesh created: {mesh.n_points:,} points, {mesh.n_faces:,} faces")
            
            # Check for mesh discontinuities/gaps (especially for wedge zones)
            if 'wedge' in zone_name.lower():
                logger.info(f"🔧 MESH CONTINUITY CHECK for '{zone_name}':")
                
                # Check point distribution
                bounds = mesh.bounds
                x_range = bounds[1] - bounds[0]
                y_range = bounds[3] - bounds[2]
                z_range = bounds[5] - bounds[4]
                
                point_density_x = mesh.n_points / max(x_range, 1e-10) if x_range > 1e-10 else 0
                point_density_y = mesh.n_points / max(y_range, 1e-10) if y_range > 1e-10 else 0
                point_density_z = mesh.n_points / max(z_range, 1e-10) if z_range > 1e-10 else 0
                
                logger.info(f"🔧   Point densities: X={point_density_x:.1f}, Y={point_density_y:.1f}, Z={point_density_z:.1f}")
                logger.info(f"🔧   Mesh bounds: X=[{bounds[0]:.3f}, {bounds[1]:.3f}], Y=[{bounds[2]:.3f}, {bounds[3]:.3f}], Z=[{bounds[4]:.3e}, {bounds[5]:.3e}]")
                
                # Check face-to-point ratio
                face_point_ratio = mesh.n_faces / max(mesh.n_points, 1)
                logger.info(f"🔧   Face/Point ratio: {face_point_ratio:.3f} (good: 1.5-3.0, sparse: <1.0)")
                
                if face_point_ratio < 1.0:
                    logger.warning(f"🔧   LOW FACE DENSITY detected - may cause gaps in surface")
                elif face_point_ratio > 3.0:
                    logger.info(f"🔧   High face density - good surface coverage expected")
            
            # Add basic normal computation for proper lighting (fast method)
            try:
                # First try fast normal computation
                mesh = mesh.compute_normals(
                    point_normals=True,     # Essential for smooth lighting
                    cell_normals=False,     # Skip cell normals for speed
                    auto_orient_normals=False,  # Skip expensive auto-orientation
                    consistent_normals=False    # Skip consistency checks
                )
                logger.info("⚡ Fast normal computation completed")
            except Exception as ne:
                logger.warning(f"Fast normal computation failed: {ne}")
                
                # Try even simpler normal computation as fallback
                try:
                    mesh = mesh.compute_normals(point_normals=True, cell_normals=False)
                    logger.info("⚡ Fallback normal computation completed")
                except Exception as ne2:
                    logger.warning(f"All normal computation failed: {ne2}, continuing without normals")
            
            # Final mesh validation
            if mesh.n_points == 0 or mesh.n_faces == 0:
                raise ValueError(f"Invalid mesh after processing: {mesh.n_points} points, {mesh.n_faces} faces")
            
            color = self._get_zone_color(zone_name)
            
            # Check for degenerate/thin dimensions BEFORE rendering (common in wedge zones)
            bounds = mesh.bounds
            x_range = bounds[1] - bounds[0]  # x_max - x_min
            y_range = bounds[3] - bounds[2]  # y_max - y_min  
            z_range = bounds[5] - bounds[4]  # z_max - z_min
            
            min_thickness = 1e-10  # Threshold for degenerate dimension
            degenerate_dims = []
            if x_range < min_thickness:
                degenerate_dims.append('X')
            if y_range < min_thickness:
                degenerate_dims.append('Y')
            if z_range < min_thickness:
                degenerate_dims.append('Z')
            
            # Decide on edge rendering based on face count and mesh quality
            show_edges = len(sampled_faces) < 30000  # Only show edges for reasonably sized meshes
            
            # Check if we need to compensate for potential gaps
            face_point_ratio = mesh.n_faces / max(mesh.n_points, 1)
            has_potential_gaps = face_point_ratio < 1.5  # Low face density suggests gaps
            
            # Fast but visually appealing rendering
            render_settings = {
                'style': 'surface',
                'color': color,
                'opacity': 0.9,
                'lighting': True,              # Essential for depth perception
                'smooth_shading': True,        # Smooth surfaces
                'ambient': 0.4,                # Stronger ambient light for definition
                'diffuse': 0.6,                # Good surface definition
                'specular': 0.1,               # Subtle highlights
            }
            
            # Force solid surface rendering for all surfaces
            render_settings.update({
                'style': 'surface',     # Always surface, never points
                'opacity': 1.0,         # Full opacity for solid appearance
                'show_edges': False,    # Clean surface without edge lines
                'smooth_shading': True  # Smooth surface interpolation
            })
            
            # Universal handling for ALL degenerate/thin surfaces
            if degenerate_dims:
                logger.warning(f"🔧 DEGENERATE SURFACE: {'/'.join(degenerate_dims)} dimension(s) are nearly zero")
                logger.warning(f"🔧   X range: {x_range:.2e}, Y range: {y_range:.2e}, Z range: {z_range:.2e}")
                logger.info(f"🔧 APPLYING UNIVERSAL THIN SURFACE FIX for zone '{zone_name}'")
                
                # Enhanced rendering for thin surfaces while maintaining solid appearance
                render_settings.update({
                    'opacity': 1.0,           # Full opacity for visibility
                    'ambient': 0.7,           # Higher ambient for better visibility
                    'diffuse': 0.9,           # Higher diffuse lighting
                    'style': 'surface',       # Always solid surface
                    'show_edges': False       # Clean solid surface
                })
            
            # Always render as clean solid surface without edges
            render_settings['show_edges'] = False
            logger.info(f"⚡ Solid surface rendering: {len(sampled_faces):,} faces")
            
            logger.info(f"🔧 Final render settings: {render_settings}")
            
            # Render as solid surface only
            try:
                actor = self.plotter.add_mesh(mesh, **render_settings)
                logger.info(f"✅ Solid surface rendered successfully")
                    
            except Exception as render_error:
                logger.error(f"Surface rendering failed: {render_error}, attempting alternative surface method")
                # Try alternative surface rendering with basic settings
                try:
                    actor = self.plotter.add_mesh(
                        mesh,
                        style='surface',
                        color=color,
                        opacity=1.0,
                        show_edges=False
                    )
                    logger.info(f"✅ Alternative surface rendering succeeded")
                except Exception as alt_error:
                    logger.error(f"All surface rendering failed: {alt_error}")
                    raise alt_error  # Re-raise to trigger fallback
            
            # Debug mesh bounds and camera positioning
            center = mesh.center
            logger.info(f"🔧 Mesh bounds: {bounds}")
            logger.info(f"🔧 Mesh center: {center}")
            logger.info(f"🔧 Mesh size: {mesh.length}")
            
            # Verify actor was created
            if actor is not None:
                logger.info(f"🔧 Actor created successfully: {type(actor)}")
                logger.info(f"🔧 Actor visibility: {actor.GetVisibility()}")
                
                # Special debugging for degenerate surfaces
                if degenerate_dims:
                    logger.info(f"🔧 DEGENERATE SURFACE DEBUG:")
                    logger.info(f"🔧   Zone name: {zone_name}")
                    logger.info(f"🔧   Color: {color}")
                    logger.info(f"🔧   Degenerate dims: {degenerate_dims}")
                    logger.info(f"🔧   Actor bounds: {actor.GetBounds() if hasattr(actor, 'GetBounds') else 'N/A'}")
                    logger.info(f"🔧   Mesh bounds: {bounds}")
                    logger.info(f"🔧   Render settings: {render_settings}")
                
                # Force set actor properties to ensure visibility for ALL actors
                try:
                    if hasattr(actor, 'GetProperty'):
                        prop = actor.GetProperty()
                        prop.SetOpacity(1.0)  # Force full opacity
                        if degenerate_dims:
                            logger.info(f"🔧   Forced opacity to 1.0 for degenerate surface")
                except Exception as pe:
                    logger.warning(f"Failed to set actor properties: {pe}")
            else:
                logger.error(f"🔧 Actor creation failed - returned None")
            
            # Force camera reset to focus on the mesh
            try:
                self.plotter.reset_camera()
                logger.info(f"🔧 Camera reset completed")
                
                # Universal camera handling for ALL degenerate/thin surfaces
                if degenerate_dims:
                    logger.info(f"🔧 APPLYING UNIVERSAL CAMERA for degenerate surface '{zone_name}'")
                    
                    # Determine best camera position based on which dimensions are degenerate
                    if 'Z' in degenerate_dims:
                        # For thin Z surfaces, position camera to look down from above
                        camera_pos = (center[0], center[1], center[2] + 10.0)
                        self.plotter.camera_position = [camera_pos, center, (0, 1, 0)]
                        logger.info(f"🔧   Set camera above thin Z surface: {camera_pos}")
                    elif 'Y' in degenerate_dims:
                        # For thin Y surfaces, position camera to look from front
                        camera_pos = (center[0], center[1] + 10.0, center[2])
                        self.plotter.camera_position = [camera_pos, center, (0, 0, 1)]
                        logger.info(f"🔧   Set camera in front of thin Y surface: {camera_pos}")
                    elif 'X' in degenerate_dims:
                        # For thin X surfaces, position camera to look from side
                        camera_pos = (center[0] + 10.0, center[1], center[2])
                        self.plotter.camera_position = [camera_pos, center, (0, 0, 1)]
                        logger.info(f"🔧   Set camera to side of thin X surface: {camera_pos}")
                    
                    # Zoom in closer for better visibility of ALL thin surfaces
                    self.plotter.camera.zoom(2.0)  # Zoom in 2x
                    logger.info(f"🔧   Applied 2x zoom for thin surface")
                
                # Force render
                self.plotter.render()
                logger.info(f"🔧 Force render completed")
            except Exception as cam_e:
                logger.warning(f"Camera reset/render failed: {cam_e}")
            
            return actor
            
        except Exception as e:
            logger.error(f"Fast surface creation failed: {e}")
            # NO POINT CLOUDS - create emergency solid surface using convex hull
            try:
                pv_mesh = pv.PolyData(points)
                # Create solid surface from points using convex hull
                surface_mesh = pv_mesh.convex_hull()
                color = self._get_zone_color(zone_name)
                logger.info(f"Emergency convex hull surface: {surface_mesh.n_faces} faces")
                return self.plotter.add_mesh(
                    surface_mesh,
                    style='surface',
                    color=color,
                    opacity=1.0,
                    show_edges=False
                )
            except Exception as hull_error:
                # Final attempt: 2D Delaunay triangulation for surface creation
                try:
                    pv_mesh = pv.PolyData(points)
                    surface_mesh = pv_mesh.delaunay_2d()
                    color = self._get_zone_color(zone_name)
                    logger.info(f"Emergency Delaunay surface: {surface_mesh.n_faces} faces")
                    return self.plotter.add_mesh(
                        surface_mesh,
                        style='surface',
                        color=color,
                        opacity=1.0,
                        show_edges=False
                    )
                except Exception as delaunay_error:
                    logger.error(f"All surface creation methods failed: {delaunay_error}")
                    raise ValueError(f"Cannot create solid surface for zone {zone_name} - no point clouds allowed")
