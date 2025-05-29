"""Main GUI application for OpenFFD.

This module provides the main entry point for the OpenFFD GUI application.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QPushButton, QFileDialog, QComboBox, QLabel, QSpinBox, 
    QDoubleSpinBox, QTabWidget, QSplitter, QMessageBox, QGroupBox,
    QScrollArea, QFormLayout, QCheckBox, QLineEdit, QProgressBar,
    QStatusBar, QToolBar, QSizePolicy
)
from PyQt6.QtGui import QIcon, QAction, QPixmap, QFont, QColor
from PyQt6.QtCore import Qt, QSize, QSettings, QThread, pyqtSignal, pyqtSlot

# Import visualization components
from openffd.gui.visualization import FFDVisualizationWidget
from openffd.gui.mesh_panel import MeshPanel
from openffd.gui.ffd_panel import FFDPanel
from openffd.gui.solver_panel import SolverPanel
from openffd.gui.settings_dialog import SettingsDialog
from openffd.gui.utils import setup_logger, get_icon_path, show_error_dialog

# Import advanced components
from openffd.gui.zone_extraction import ZoneExtractionPanel
from openffd.gui.deformation_view import DeformationWidget
from openffd.gui.sensitivity_mapper import SensitivityMapper
from openffd.gui.hierarchical_panel import HierarchicalFFDPanel

# Import core OpenFFD functionality
from openffd.core.config import FFDConfig
from openffd.utils.parallel import ParallelConfig
from openffd.mesh.general import read_general_mesh, is_fluent_mesh
from openffd.core.control_box import create_ffd_box
from openffd.core.hierarchical import HierarchicalFFD
from openffd.io.export import write_ffd_3df, write_ffd_xyz

# Configure logging
logger = logging.getLogger(__name__)

class OpenFFDMainWindow(QMainWindow):
    """Main window for the OpenFFD GUI application."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("OpenFFD - Free-Form Deformation Tool")
        self.setMinimumSize(1200, 800)
        
        # Initialize state variables
        self.current_mesh = None
        self.mesh_points = None
        self.ffd_control_points = None
        self.bounding_box = None
        self.ffd_config = FFDConfig(dims=(4, 4, 4))  # Default dimensions
        self.hierarchical_ffd = None  # For hierarchical FFD
        self.parallel_config = ParallelConfig()
        self.settings = QSettings("OpenFFD", "GUI")
        
        # Setup UI components
        self._setup_ui()
        self._create_menu()
        self._create_toolbar()
        self._restore_settings()
        
        # Connect signals and slots
        self._connect_signals()
        
        # Set status
        self.statusBar().showMessage("Ready")
        logger.info("OpenFFD GUI initialized")
    
    def _setup_ui(self):
        """Setup the user interface layout."""
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create a splitter for resizable panels
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Tab widget for organizing controls
        self.tabs = QTabWidget()
        
        # Mesh tab
        self.mesh_panel = MeshPanel()
        self.tabs.addTab(self.mesh_panel, "Mesh")
        
        # FFD tab
        self.ffd_panel = FFDPanel()
        self.tabs.addTab(self.ffd_panel, "FFD")
        
        # Solver tab
        self.solver_panel = SolverPanel()
        self.tabs.addTab(self.solver_panel, "Solver")
        
        # Advanced tab for zone extraction and optimization
        self.advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_tab)
        
        # Create a tab widget for advanced features
        advanced_tabs = QTabWidget()
        
        # Zone extraction panel
        self.zone_panel = ZoneExtractionPanel()
        advanced_tabs.addTab(self.zone_panel, "Zone Extraction")
        
        # Deformation panel
        self.deformation_panel = DeformationWidget()
        advanced_tabs.addTab(self.deformation_panel, "Deformation")
        
        # Hierarchical FFD panel
        self.hierarchical_panel = HierarchicalFFDPanel()
        advanced_tabs.addTab(self.hierarchical_panel, "Hierarchical FFD")
        
        advanced_layout.addWidget(advanced_tabs)
        self.tabs.addTab(self.advanced_tab, "Advanced")
        
        left_layout.addWidget(self.tabs)
        
        # Action buttons section
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        
        self.generate_ffd_btn = QPushButton("Generate FFD Box")
        self.generate_ffd_btn.setEnabled(False)
        
        self.export_ffd_btn = QPushButton("Export FFD Box")
        self.export_ffd_btn.setEnabled(False)
        
        action_layout.addWidget(self.generate_ffd_btn)
        action_layout.addWidget(self.export_ffd_btn)
        action_group.setLayout(action_layout)
        
        left_layout.addWidget(action_group)
        
        # Right panel - Visualization
        self.visualization = FFDVisualizationWidget()
        
        # Add widgets to splitter
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(self.visualization)
        self.main_splitter.setSizes([300, 900])  # Default sizes
        
        # Add splitter to main layout
        main_layout.addWidget(self.main_splitter)
        
        # Status bar
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusbar.addPermanentWidget(self.progress_bar)
    
    def _create_menu(self):
        """Create the main menu."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open Mesh...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.on_open_mesh)
        file_menu.addAction(open_action)
        
        export_action = QAction("&Export FFD...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.on_export_ffd)
        export_action.setEnabled(False)
        self.export_action = export_action
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        settings_action = QAction("&Settings...", self)
        settings_action.triggered.connect(self.on_show_settings)
        edit_menu.addAction(settings_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        reset_view_action = QAction("Reset &View", self)
        reset_view_action.triggered.connect(self.visualization.reset_camera)
        view_menu.addAction(reset_view_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About OpenFFD", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)
        
    def _create_toolbar(self):
        """Create the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("mainToolBar")  # Set an object name for state saving
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        
        # Open mesh action
        open_mesh_action = QAction("Open Mesh", self)
        open_mesh_action.triggered.connect(self.on_open_mesh)
        toolbar.addAction(open_mesh_action)
        
        # Generate FFD action
        generate_action = QAction("Generate FFD", self)
        generate_action.triggered.connect(self.on_generate_ffd)
        generate_action.setEnabled(False)
        self.generate_action = generate_action
        toolbar.addAction(generate_action)
        
        # Export FFD action
        export_action = QAction("Export FFD", self)
        export_action.triggered.connect(self.on_export_ffd)
        export_action.setEnabled(False)
        toolbar.addAction(export_action)
        
        toolbar.addSeparator()
        
        # View options
        view_mesh_action = QAction("Toggle Mesh", self)
        view_mesh_action.setCheckable(True)
        view_mesh_action.setChecked(True)
        view_mesh_action.triggered.connect(self.visualization.toggle_mesh_visibility)
        toolbar.addAction(view_mesh_action)
        
        view_ffd_action = QAction("Toggle FFD", self)
        view_ffd_action.setCheckable(True)
        view_ffd_action.setChecked(True)
        view_ffd_action.triggered.connect(self.visualization.toggle_ffd_visibility)
        toolbar.addAction(view_ffd_action)
    
    def _connect_signals(self):
        """Connect signals and slots."""
        # Mesh panel signals
        self.mesh_panel.mesh_loaded.connect(self.on_mesh_loaded)
            
        # FFD panel signals
        self.ffd_panel.ffd_parameters_changed.connect(self.on_ffd_parameters_changed)
        self.generate_ffd_btn.clicked.connect(self.on_generate_ffd)
        self.export_ffd_btn.clicked.connect(self.on_export_ffd)
            
        # Zone extraction panel signals
        # Connect mesh loaded signal from main mesh panel to the zone handler
        self.mesh_panel.mesh_loaded.connect(self.on_mesh_for_zones_loaded)
        self.zone_panel.zone_extracted.connect(self.on_zone_extracted)
        
        # Solver panel signals
        self.solver_panel.simulation_completed.connect(self.on_simulation_completed)
        
        # Deformation panel signals
        self.deformation_panel.deformation_changed.connect(self.on_deformation_changed)
        
        # Hierarchical FFD panel signals
        self.hierarchical_panel.hierarchical_ffd_updated.connect(self.on_hierarchical_ffd_updated)
        
    def _restore_settings(self):
        """Restore application settings."""
        # Restore window geometry and state if available
        if self.settings.contains("MainWindow/geometry"):
            self.restoreGeometry(self.settings.value("MainWindow/geometry"))
            
        if self.settings.contains("MainWindow/state"):
            self.restoreState(self.settings.value("MainWindow/state"))
            
        if self.settings.contains("MainWindow/splitter") and hasattr(self, "main_splitter"):
            self.main_splitter.restoreState(self.settings.value("MainWindow/splitter"))
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Save settings
        self.settings.setValue("MainWindow/geometry", self.saveGeometry())
        self.settings.setValue("MainWindow/state", self.saveState())
        self.settings.setValue("MainWindow/splitter", self.main_splitter.saveState())
        super().closeEvent(event)
    
    @pyqtSlot()
    def on_open_mesh(self):
        """Open a mesh file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Mesh File",
            "",
            "Mesh Files (*.cas *.msh *.vtk *.stl *.obj);;All Files (*)"
        )
        
        if file_path:
            self.mesh_panel.load_mesh(file_path)
    
    @pyqtSlot(object, object)
    def on_mesh_loaded(self, mesh_data, mesh_points):
        """Handle mesh loaded signal."""
        self.current_mesh = mesh_data
        self.mesh_points = mesh_points
        
        # Enable FFD generation
        self.generate_ffd_btn.setEnabled(True)
        self.generate_action.setEnabled(True)
        
        # Update visualization
        self.visualization.set_mesh(mesh_data, mesh_points)
        self.statusBar().showMessage(f"Mesh loaded: {len(mesh_points)} points")
        
        # Update FFD panel bounds
        if mesh_points is not None and len(mesh_points) > 0:
            min_coords = mesh_points.min(axis=0)
            max_coords = mesh_points.max(axis=0)
            self.ffd_panel.update_bounds(min_coords, max_coords)
            
            # Update hierarchical FFD panel with mesh points
            self.hierarchical_panel.set_mesh_points(mesh_points)
    
    @pyqtSlot()
    def on_ffd_parameters_changed(self):
        """Handle FFD parameter changes."""
        # If we already have FFD control points, we may want to update them
        if self.ffd_control_points is not None and self.mesh_points is not None:
            # Could implement auto-update here if desired
            pass
    
    @pyqtSlot()
    def on_generate_ffd(self):
        """Generate FFD control box."""
        if self.mesh_points is None:
            show_error_dialog("No mesh loaded", "Please load a mesh first.")
            return
        
        try:
            self.statusBar().showMessage("Generating FFD control box...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            
            # Get FFD parameters from panel
            control_dim = self.ffd_panel.get_control_dimensions()
            margin = self.ffd_panel.get_margin()
            custom_dims = self.ffd_panel.get_custom_bounds()
            
            # Generate FFD control box
            self.ffd_control_points, self.bounding_box = create_ffd_box(
                self.mesh_points,
                control_dim=control_dim,
                margin=margin,
                custom_dims=custom_dims,
                parallel_config=self.parallel_config
            )
            
            self.progress_bar.setValue(90)
            
            # Update visualization
            self.visualization.set_ffd(self.ffd_control_points, control_dim)
            
            # Enable export
            self.export_ffd_btn.setEnabled(True)
            self.export_action.setEnabled(True)
            
            self.statusBar().showMessage(f"FFD control box generated: {len(self.ffd_control_points)} control points")
        except Exception as e:
            logger.error(f"Error generating FFD box: {str(e)}")
            show_error_dialog("FFD Generation Error", str(e))
        finally:
            self.progress_bar.setVisible(False)
    
    @pyqtSlot()
    def on_export_ffd(self):
        """Export FFD control box."""
        if self.ffd_control_points is None:
            show_error_dialog("No FFD Box", "Please generate an FFD box first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export FFD Box",
            "",
            "FFD Files (*.3df);;XYZ Files (*.xyz);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            self.statusBar().showMessage(f"Exporting FFD to {file_path}...")
            
            control_dim = self.ffd_panel.get_control_dimensions()
            nx, ny, nz = control_dim
            
            # Export based on file extension
            if file_path.lower().endswith(".3df"):
                write_ffd_3df(file_path, self.ffd_control_points, nx, ny, nz)
            elif file_path.lower().endswith(".xyz"):
                write_ffd_xyz(file_path, self.ffd_control_points, nx, ny, nz)
            else:
                # Add extension if needed
                if "." not in os.path.basename(file_path):
                    file_path += ".3df"
                    write_ffd_3df(file_path, self.ffd_control_points, nx, ny, nz)
            
            self.statusBar().showMessage(f"FFD box exported to {file_path}")
        except Exception as e:
            logger.error(f"Error exporting FFD box: {str(e)}")
            show_error_dialog("Export Error", str(e))
    
    @pyqtSlot()
    def on_show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self.parallel_config, self)
        if dialog.exec():
            # Update settings
            self.parallel_config = dialog.get_parallel_config()
            logger.debug(f"Updated parallel config: {self.parallel_config}")
    
    @pyqtSlot()
    def on_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About OpenFFD",
            """<h1>OpenFFD</h1>
            <p>An open-source FFD (Free-Form Deformation) control box generator 
            for computational meshes</p>
            <p>Version 1.0.0</p>
            <p>&copy; 2023</p>"""
        )
        
    @pyqtSlot(object, object)
    def on_mesh_for_zones_loaded(self, mesh_data, mesh_points):
        """Handle mesh loaded signal for zone extraction.
        
        Args:
            mesh_data: The mesh data object
            mesh_points: Numpy array of mesh point coordinates
        """
        # Pass the mesh to the zone extraction panel
        if mesh_data is not None:
            mesh_file_path = self.mesh_panel.mesh_file_path
            self.zone_panel.set_mesh(mesh_file_path, mesh_data)
            
    @pyqtSlot(object, object, str)
    def on_zone_extracted(self, mesh_data, zone_points, zone_name):
        """Handle zone extraction completion.
        
        Args:
            mesh_data: The mesh data object
            zone_points: Numpy array of extracted zone point coordinates
            zone_name: Name of the extracted zone
        """
        # Update the visualization with extracted zone points
        self.visualization.set_mesh(mesh_data, zone_points)
        
        # Update the mesh points for FFD generation
        self.mesh_points = zone_points
        
        # Enable FFD generation for the zone
        self.generate_ffd_btn.setEnabled(True)
        self.generate_action.setEnabled(True)
        
        # Update FFD panel with new bounds
        if zone_points is not None and len(zone_points) > 0:
            min_coords = zone_points.min(axis=0)
            max_coords = zone_points.max(axis=0)
            self.ffd_panel.update_bounds(min_coords, max_coords)
            
        # Show a message in the status bar
        self.statusBar().showMessage(f"Extracted zone '{zone_name}' with {len(zone_points)} points")
        
    @pyqtSlot(bool, str)
    def on_simulation_completed(self, success, message):
        """Handle simulation completion from the solver panel.
        
        Args:
            success: Whether the simulation was successful
            message: Result message
        """
        if not success:
            show_error_dialog("Simulation Error", message)
            return
            
        # Check if this was an adjoint simulation that produced sensitivities
        sensitivity_file = self.solver_panel.get_sensitivity_file()
        if sensitivity_file and os.path.exists(sensitivity_file) and self.ffd_control_points is not None:
            self.statusBar().showMessage("Loading sensitivities from adjoint simulation...")
            
            try:
                # Create sensitivity mapper
                mapper = SensitivityMapper(self.parallel_config)
                
                # Load sensitivities
                mesh_sensitivities = mapper.load_sensitivities(sensitivity_file)
                
                # Map sensitivities to control points
                control_dims = self.ffd_panel.get_control_dimensions()
                control_sens = mapper.map_to_control_points(
                    self.mesh_points, 
                    mesh_sensitivities, 
                    self.ffd_control_points,
                    control_dims
                )
                
                # Set sensitivities in deformation panel
                self.deformation_panel.set_sensitivity_data(
                    control_sens, 
                    self.ffd_control_points,
                    control_dims
                )
                
                # Switch to the advanced tab and deformation panel
                self.tabs.setCurrentIndex(3)  # Advanced tab
                
                self.statusBar().showMessage("Sensitivities loaded and mapped to control points")
            except Exception as e:
                logger.error(f"Error loading sensitivities: {str(e)}")
                show_error_dialog("Sensitivity Error", str(e))
        else:
            self.statusBar().showMessage(message)
    
    @pyqtSlot(object, float)
    def on_deformation_changed(self, deformed_points, scale):
        """Handle deformation changes from the deformation panel.
        
        Args:
            deformed_points: The deformed control points
            scale: Scale factor applied to the deformation
        """
        if deformed_points is None:
            return
            
        # Update the visualization with deformed control points
        control_dims = self.ffd_panel.get_control_dimensions()
        self.visualization.set_ffd(deformed_points, control_dims)
        
        # Update status bar
        self.statusBar().showMessage(f"Deformation applied with scale factor {scale:.2f}")
    
    @pyqtSlot(object)
    def on_hierarchical_ffd_updated(self, hierarchical_ffd):
        """Handle hierarchical FFD updates from the hierarchical panel.
        
        Args:
            hierarchical_ffd: The HierarchicalFFD object
        """
        if hierarchical_ffd is None:
            return
        
        # Store the hierarchical FFD
        self.hierarchical_ffd = hierarchical_ffd
        
        # Clear the visualization
        self.visualization.plotter.clear()
        
        try:
            # Get visualization parameters
            show_mesh = self.hierarchical_panel.show_mesh_cb.isChecked()
            color_by_level = self.hierarchical_panel.color_by_level_cb.isChecked()
            show_influence = self.hierarchical_panel.show_influence_cb.isChecked()
            
            # Create plotter for the visualization
            plotter = self.visualization.plotter
            
            # Use mesh points if available and requested
            mesh_points = self.mesh_points if show_mesh else None
            
            # Display all levels
            level_info = hierarchical_ffd.get_level_info()
            all_levels = [level['level_id'] for level in level_info]
            
            # Add hierarchical FFD visualization directly to the plotter
            for level_id in all_levels:
                level = hierarchical_ffd.levels[level_id]
                
                # Get control points for this level
                cp = level.control_points
                dims = level.dims
                
                # Create a color based on the level
                if color_by_level:
                    # Use a different color for each level (red -> blue gradient)
                    level_fraction = level_id / max(all_levels)
                    r = 1.0 - level_fraction
                    b = level_fraction
                    color = (r, 0.2, b)
                else:
                    color = 'red'  # Default color
                
                # Try to create a proper grid for this level
                try:
                    from openffd.visualization.level_grid import try_create_level_grid, create_grid_edges
                    
                    # Reshape the control points to a 3D grid
                    cp_grid = try_create_level_grid(cp, dims)
                    
                    if cp_grid is not None:
                        # Generate grid edges
                        edges = create_grid_edges(cp_grid)
                        
                        # Add each edge as a line
                        for edge in edges:
                            start, end = edge[0], edge[1]
                            line = pv.Line(start, end)
                            plotter.add_mesh(line, color=color, line_width=2)
                        
                        # Add the control points
                        plotter.add_points(cp, color=color, point_size=8, render_points_as_spheres=True)
                except Exception as e:
                    logger.error(f"Error creating grid for level {level_id}: {str(e)}")
                    
                    # Fallback to simple point cloud
                    plotter.add_points(cp, color=color, point_size=8, render_points_as_spheres=True)
            
            # Add mesh points if requested
            if show_mesh and self.mesh_points is not None:
                mesh_cloud = pv.PolyData(self.mesh_points)
                plotter.add_mesh(mesh_cloud, color='lightblue', opacity=0.5, point_size=3)
            
            # Reset camera to show the scene
            plotter.reset_camera()
            
            logger.info(f"Updated hierarchical FFD visualization with {len(all_levels)} levels")
            self.statusBar().showMessage(f"Hierarchical FFD updated with {len(all_levels)} levels", 3000)
            
        except Exception as e:
            logger.error(f"Error visualizing hierarchical FFD: {str(e)}")
            self.statusBar().showMessage(f"Error visualizing hierarchical FFD: {str(e)}", 5000)
    

def launch_gui():
    """Launch the OpenFFD GUI application."""
    setup_logger()
    
    # Configure high DPI scaling before creating the application
    # In PyQt6, high DPI scaling is enabled by default
    # But we can still ensure high DPI pixmaps are used
    import PyQt6.QtCore
    if hasattr(PyQt6.QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        PyQt6.QtCore.QCoreApplication.setAttribute(PyQt6.QtCore.Qt.AA_UseHighDpiPixmaps, True)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern cross-platform style
    
    # Create and show main window
    window = OpenFFDMainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(launch_gui())
