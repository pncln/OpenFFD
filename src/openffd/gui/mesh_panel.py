"""Mesh panel for OpenFFD GUI.

This module provides the UI panel for mesh loading and configuration.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QComboBox, QFileDialog,
    QMessageBox, QListWidget, QTextEdit, QCheckBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread

from openffd.mesh.general import read_general_mesh, is_fluent_mesh
from openffd.mesh.fluent_reader import FluentMeshReader
from openffd.gui.utils import show_error_dialog

# Configure logging
logger = logging.getLogger(__name__)


class MeshPanel(QWidget):
    """Panel for mesh loading and configuration."""
    
    # Signal emitted when a mesh is loaded
    mesh_loaded = pyqtSignal(object, object)
    
    # Signal emitted when boundary zone visibility changes
    boundary_visibility_changed = pyqtSignal(str, bool)
    
    def __init__(self, parent=None):
        """Initialize the mesh panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.mesh_file_path = None
        self.mesh_data = None
        self.mesh_points = None
        self.available_zones = []
        self.extraction_thread = None  # For background zone extraction
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Mesh file section
        mesh_group = QGroupBox("Mesh File")
        mesh_layout = QVBoxLayout()
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_mesh_file)
        
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(browse_button, 0)
        mesh_layout.addLayout(file_layout)
        
        # Format options
        format_layout = QFormLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Auto Detect", "Fluent", "VTK", "STL", "OBJ", "Other"])
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        
        format_layout.addRow("Format:", self.format_combo)
        mesh_layout.addLayout(format_layout)
        
        # Fluent-specific options
        self.fluent_options = QWidget()
        fluent_layout = QFormLayout(self.fluent_options)
        
        self.binary_checkbox = QCheckBox("Use binary format")
        self.binary_checkbox.setChecked(True)
        fluent_layout.addRow("Read mode:", self.binary_checkbox)
        
        self.fluent_options.setVisible(False)
        mesh_layout.addWidget(self.fluent_options)
        
        # Load button
        self.load_button = QPushButton("Load Mesh")
        self.load_button.clicked.connect(self._load_mesh)
        self.load_button.setEnabled(False)
        mesh_layout.addWidget(self.load_button)
        
        mesh_group.setLayout(mesh_layout)
        layout.addWidget(mesh_group)
        

        
        # Zone selection section
        self.zone_group = QGroupBox("Zone Selection")
        zone_layout = QVBoxLayout()
        
        # Zone combo
        zone_form_layout = QFormLayout()
        self.zone_combo = QComboBox()
        self.zone_combo.setEnabled(False)
        self.zone_combo.currentIndexChanged.connect(self._on_zone_changed)
        
        zone_form_layout.addRow("Selected Zone:", self.zone_combo)
        zone_layout.addLayout(zone_form_layout)
        
        # Zone list button
        self.list_zones_button = QPushButton("List Available Zones")
        self.list_zones_button.clicked.connect(self._list_zones)
        self.list_zones_button.setEnabled(False)
        zone_layout.addWidget(self.list_zones_button)
        
        # Zone operation buttons
        zone_ops_layout = QHBoxLayout()
        
        self.extract_zone_button = QPushButton("Extract Zone")
        self.extract_zone_button.clicked.connect(self._extract_zone)
        self.extract_zone_button.setEnabled(False)
        
        self.save_boundary_button = QPushButton("Save Boundary")
        self.save_boundary_button.clicked.connect(self._save_boundary)
        self.save_boundary_button.setEnabled(False)
        
        zone_ops_layout.addWidget(self.extract_zone_button)
        zone_ops_layout.addWidget(self.save_boundary_button)
        zone_layout.addLayout(zone_ops_layout)
        
        self.zone_group.setLayout(zone_layout)
        layout.addWidget(self.zone_group)
        
        # Boundary zone visibility controls
        self.boundary_group = QGroupBox("Boundary Zone Visibility")
        self.boundary_layout = QVBoxLayout()
        self.boundary_group.setLayout(self.boundary_layout)
        self.boundary_group.setVisible(False)  # Hidden until mesh is loaded
        
        self.boundary_checkboxes = {}  # Store zone checkboxes
        
        # Show all/hide all buttons
        boundary_buttons_layout = QHBoxLayout()
        self.show_all_boundaries_button = QPushButton("Show All")
        self.show_all_boundaries_button.clicked.connect(self._show_all_boundaries)
        self.hide_all_boundaries_button = QPushButton("Hide All")
        self.hide_all_boundaries_button.clicked.connect(self._hide_all_boundaries)
        
        boundary_buttons_layout.addWidget(self.show_all_boundaries_button)
        boundary_buttons_layout.addWidget(self.hide_all_boundaries_button)
        self.boundary_layout.addLayout(boundary_buttons_layout)
        
        # Scroll area for checkboxes
        self.boundary_scroll = QScrollArea()
        self.boundary_scroll.setWidgetResizable(True)
        self.boundary_scroll.setMaximumHeight(200)
        self.boundary_scroll_widget = QWidget()
        self.boundary_scroll_layout = QVBoxLayout()
        self.boundary_scroll_widget.setLayout(self.boundary_scroll_layout)
        self.boundary_scroll.setWidget(self.boundary_scroll_widget)
        self.boundary_layout.addWidget(self.boundary_scroll)
        
        layout.addWidget(self.boundary_group)
        
        # Mesh stats section
        stats_group = QGroupBox("Mesh Statistics")
        stats_layout = QFormLayout()
        
        self.point_count_label = QLabel("0")
        self.cell_count_label = QLabel("0")
        self.bounds_label = QLabel("N/A")
        
        stats_layout.addRow("Number of Points:", self.point_count_label)
        stats_layout.addRow("Number of Cells:", self.cell_count_label)
        stats_layout.addRow("Bounds:", self.bounds_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Add stretch to push everything to the top
        layout.addStretch(1)
    
    def _browse_mesh_file(self):
        """Open file dialog to select a mesh file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Mesh File",
            "",
            "Mesh Files (*.cas *.msh *.vtk *.stl *.obj);;All Files (*)"
        )
        
        if file_path:
            self.mesh_file_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.load_button.setEnabled(True)
            
            # Try to auto-detect format
            if file_path.lower().endswith(('.cas', '.msh')):
                self.format_combo.setCurrentText("Fluent")
            elif file_path.lower().endswith('.vtk'):
                self.format_combo.setCurrentText("VTK")
            elif file_path.lower().endswith('.stl'):
                self.format_combo.setCurrentText("STL")
            elif file_path.lower().endswith('.obj'):
                self.format_combo.setCurrentText("OBJ")
            else:
                self.format_combo.setCurrentText("Auto Detect")
    
    def _on_format_changed(self, index):
        """Handle format selection change."""
        format_text = self.format_combo.currentText()
        self.fluent_options.setVisible(format_text == "Fluent")
    
    def _load_mesh(self):
        """Load the selected mesh file."""
        if not self.mesh_file_path:
            return
        
        try:
            # Get format and options
            format_text = self.format_combo.currentText()
            use_binary = self.binary_checkbox.isChecked()
            
            if format_text == "Auto Detect":
                # Try to detect Fluent mesh first
                if is_fluent_mesh(self.mesh_file_path):
                    # Force ASCII mode to use native parser
                    self._load_fluent_mesh(use_binary=False)
                else:
                    # Generic mesh reader
                    self._load_generic_mesh()
            elif format_text == "Fluent":
                # Force ASCII mode to use native parser
                self._load_fluent_mesh(use_binary=False)
            else:
                # Generic mesh reader for other formats
                self._load_generic_mesh()
                
            # Update UI
            self._update_mesh_statistics()
            self._update_zone_list()
            self.zone_group.setVisible(True)
            
            # Setup boundary zone visibility controls
            self._setup_boundary_controls()
            
            # Emit signal with mesh data and points
            self.mesh_loaded.emit(self.mesh_data, self.mesh_points)
            
            logger.info(f"Mesh loaded: {self.mesh_file_path}")
        except Exception as e:
            logger.error(f"Error loading mesh: {str(e)}")
            show_error_dialog("Error Loading Mesh", str(e))
            
    
    def _load_fluent_mesh(self, use_binary=True):
        """Load a Fluent mesh file.
        
        Args:
            use_binary: Whether to use binary format
        """
        # Initialize the reader with the filename and binary option
        reader = FluentMeshReader(filename=self.mesh_file_path, force_binary=use_binary, force_ascii=not use_binary)
        # Call read without binary parameter
        self.mesh_data = reader.read()
        
        if hasattr(self.mesh_data, 'points'):
            self.mesh_points = np.array(self.mesh_data.points)
        else:
            raise ValueError("Loaded mesh does not have points attribute")
    
    def _load_generic_mesh(self):
        """Load a generic mesh file."""
        self.mesh_data = read_general_mesh(self.mesh_file_path)
        
        if hasattr(self.mesh_data, 'points'):
            self.mesh_points = np.array(self.mesh_data.points)
        else:
            raise ValueError("Loaded mesh does not have points attribute")
    
    def _update_mesh_statistics(self):
        """Update the mesh statistics display."""
        if self.mesh_points is not None:
            # Update point count
            self.point_count_label.setText(str(len(self.mesh_points)))
            
            # Update cell count if available
            cell_count = 0
            if hasattr(self.mesh_data, 'cells') and self.mesh_data.cells is not None:
                if isinstance(self.mesh_data.cells, list):
                    for cells_block in self.mesh_data.cells:
                        if isinstance(cells_block, tuple) and len(cells_block) >= 2:
                            cell_count += len(cells_block[1])
                        else:
                            cell_count += len(cells_block)
                else:
                    cell_count = len(self.mesh_data.cells)
            self.cell_count_label.setText(str(cell_count))
            
            # Update bounds
            min_coords = np.min(self.mesh_points, axis=0)
            max_coords = np.max(self.mesh_points, axis=0)
            bounds_str = f"X: [{min_coords[0]:.2f}, {max_coords[0]:.2f}], "
            bounds_str += f"Y: [{min_coords[1]:.2f}, {max_coords[1]:.2f}], "
            bounds_str += f"Z: [{min_coords[2]:.2f}, {max_coords[2]:.2f}]"
            self.bounds_label.setText(bounds_str)
    
    def _update_zone_list(self):
        """Update the list of available zones."""
        self.zone_combo.clear()
        self.available_zones = []
        
        try:
            # Try to get zones from the mesh
            if hasattr(self.mesh_data, 'zones') and self.mesh_data.zones:
                self.available_zones = list(self.mesh_data.zones.keys())
            elif hasattr(self.mesh_data, 'cell_sets') and self.mesh_data.cell_sets:
                self.available_zones = list(self.mesh_data.cell_sets.keys())
            elif hasattr(self.mesh_data, 'cell_data') and self.mesh_data.cell_data:
                self.available_zones = list(self.mesh_data.cell_data.keys())
            
            if self.available_zones:
                # Add zones with type indicators
                zone_display_items = []
                for zone_name in self.available_zones:
                    zone_type = "unknown"
                    zone_icon = ""
                    
                    if hasattr(self.mesh_data, 'zones') and zone_name in self.mesh_data.zones:
                        zone_info = self.mesh_data.zones[zone_name]
                        zone_type = zone_info.get('type', 'unknown')
                        zone_obj = zone_info.get('object')
                        
                        # Determine if it's a volume or boundary zone
                        if zone_obj and hasattr(zone_obj, 'zone_type_enum'):
                            is_volume = zone_obj.zone_type_enum.name == 'VOLUME'
                        else:
                            is_volume = zone_type in ['interior', 'fluid', 'solid']
                        
                        if is_volume:
                            zone_icon = "🔵"  # Blue circle for volume zones
                        else:
                            zone_icon = "🟢"  # Green circle for boundary zones
                    
                    # Create display text with icon and type
                    display_text = f"{zone_icon} {zone_name} ({zone_type})"
                    zone_display_items.append(display_text)
                
                self.zone_combo.addItems(zone_display_items)
                self.zone_combo.setEnabled(True)
                self.extract_zone_button.setEnabled(True)
                self.save_boundary_button.setEnabled(True)
            else:
                self.zone_combo.addItem("No zones found")
                self.zone_combo.setEnabled(False)
        except Exception as e:
            logger.error(f"Error updating zone list: {str(e)}")
            self.zone_combo.addItem("Error retrieving zones")
            self.zone_combo.setEnabled(False)
    
    def _list_zones(self):
        """List all available zones."""
        if not self.available_zones:
            QMessageBox.information(
                self,
                "Zone Information",
                "No zones found in the mesh file."
            )
            return
        
        # Create a message with all zone names
        zone_text = "Available zones:\n\n"
        for i, zone in enumerate(self.available_zones, 1):
            zone_text += f"{i}. {zone}\n"
        
        QMessageBox.information(
            self,
            "Zone Information",
            zone_text
        )
    
    def _on_zone_changed(self, index):
        """Handle zone selection change."""
        if index >= 0 and index < len(self.available_zones):
            logger.info(f"Selected zone: {self.available_zones[index]}")
    
    def _extract_zone(self):
        """Extract the selected zone."""
        if not self.available_zones:
            return
        
        zone_idx = self.zone_combo.currentIndex()
        if zone_idx < 0 or zone_idx >= len(self.available_zones):
            return
        
        zone_name = self.available_zones[zone_idx]
        
        # Check zone size first to determine if we need background processing
        try:
            zone_info = self.mesh_data.zones.get(zone_name, {})
            zone_obj = zone_info.get('object')
            
            # Estimate zone size for performance decision
            needs_background = False
            if zone_obj and hasattr(zone_obj, 'faces') and hasattr(zone_obj.faces, '__len__'):
                face_count = len(zone_obj.faces)
                if face_count > 10000:  # Large zone threshold
                    needs_background = True
                    logger.info(f"Zone '{zone_name}' has {face_count:,} faces - using background processing")
            
            if needs_background:
                self._extract_zone_background(zone_name)
            else:
                self._extract_zone_direct(zone_name)
                
        except Exception as e:
            logger.error(f"Error during zone extraction: {str(e)}")
            show_error_dialog("Zone Extraction Error", str(e))
    
    def _extract_zone_direct(self, zone_name: str):
        """Extract zone directly on main thread (for small zones)."""
        try:
            # Import extraction functionality here to avoid circular imports
            from openffd.mesh.general import extract_zone_mesh
            
            # Extract zone mesh with proper face connectivity
            zone_mesh_data = extract_zone_mesh(self.mesh_data, zone_name)
            
            if zone_mesh_data is not None:
                self._process_extracted_zone(zone_name, zone_mesh_data)
            else:
                self._handle_empty_zone(zone_name)
                
        except Exception as e:
            logger.error(f"Error extracting zone: {str(e)}")
            show_error_dialog("Zone Extraction Error", str(e))
    
    def _extract_zone_background(self, zone_name: str):
        """Extract zone in background thread (for large zones)."""
        # Disable extraction button during processing
        self.extract_zone_button.setEnabled(False)
        self.extract_zone_button.setText("Extracting...")
        
        # Create and start background thread
        self.extraction_thread = ZoneExtractionThread(self.mesh_data, zone_name)
        self.extraction_thread.extraction_completed.connect(self._on_background_extraction_completed)
        self.extraction_thread.extraction_failed.connect(self._on_background_extraction_failed)
        self.extraction_thread.start()
        
        logger.info(f"Started background extraction for zone '{zone_name}'")
    
    def _on_background_extraction_completed(self, zone_name: str, zone_mesh_data: dict):
        """Handle completed background zone extraction."""
        try:
            self._process_extracted_zone(zone_name, zone_mesh_data)
        except Exception as e:
            logger.error(f"Error processing background extraction result: {str(e)}")
            show_error_dialog("Zone Processing Error", str(e))
        finally:
            # Re-enable extraction button
            self.extract_zone_button.setEnabled(True)
            self.extract_zone_button.setText("Extract Zone")
    
    def _on_background_extraction_failed(self, zone_name: str, error_message: str):
        """Handle failed background zone extraction."""
        logger.error(f"Background extraction failed for zone '{zone_name}': {error_message}")
        show_error_dialog("Zone Extraction Failed", f"Failed to extract zone '{zone_name}': {error_message}")
        
        # Re-enable extraction button
        self.extract_zone_button.setEnabled(True)
        self.extract_zone_button.setText("Extract Zone")
    
    def _process_extracted_zone(self, zone_name: str, zone_mesh_data: dict):
        """Process successfully extracted zone data."""
        zone_points = zone_mesh_data['points']
        zone_faces = zone_mesh_data['faces']
        zone_type = zone_mesh_data['zone_type']
        is_point_cloud = zone_mesh_data['is_point_cloud']
        
        # Update mesh points to use only the selected zone
        self.mesh_points = zone_points
        
        # Update statistics
        self._update_mesh_statistics()
        
        # Notify about the change with zone mesh data for proper surface rendering
        self.mesh_loaded.emit(zone_mesh_data, zone_points)
        
        # Provide informative success message
        face_info = f", {len(zone_faces)} faces" if zone_faces else " (point cloud)"
        surface_info = "proper surface mesh" if not is_point_cloud else "point cloud (no face connectivity)"
        
        QMessageBox.information(
            self,
            "Zone Extraction Complete",
            f"Successfully extracted zone '{zone_name}' with {len(zone_points):,} points{face_info}.\n\n"
            f"Zone type: {zone_type}\n"
            f"Surface data: {surface_info}\n\n"
            f"This boundary zone is ready for FFD generation."
        )
        
        logger.info(f"Extracted zone '{zone_name}' with {len(zone_points)} points and {len(zone_faces)} faces")
    
    def _handle_empty_zone(self, zone_name: str):
        """Handle case where zone extraction returns no data."""
        # Check if it's a volume zone
        zone_type = "unknown"
        is_volume_zone = False
        if hasattr(self.mesh_data, 'zones') and zone_name in self.mesh_data.zones:
            zone_info = self.mesh_data.zones[zone_name]
            zone_type = zone_info.get('type', 'unknown')
            zone_obj = zone_info.get('object')
            if zone_obj and hasattr(zone_obj, 'zone_type_enum'):
                is_volume_zone = zone_obj.zone_type_enum.name == 'VOLUME'
            else:
                is_volume_zone = zone_type in ['interior', 'fluid', 'solid']
        
        # Provide different messages for volume vs boundary zones
        if is_volume_zone:
            QMessageBox.information(
                self,
                "Volume Zone Selected",
                f"Zone '{zone_name}' is a volume zone ({zone_type}).\n\n"
                f"Volume zones define 3D fluid domains and don't have extractable surface points.\n\n"
                f"For FFD generation, please select a boundary zone instead:\n"
                f"• Wall zones (rocket, launchpad, deflector)\n"
                f"• Inlet/outlet zones\n"
                f"• Symmetry zones"
            )
        else:
            QMessageBox.warning(
                self,
                "Zone Extraction",
                f"No mesh data found in zone '{zone_name}' ({zone_type}).\n\n"
                f"This boundary zone may be empty or have connectivity issues."
            )
    
    def _save_boundary(self):
        """Save the selected boundary/zone to a file."""
        if not self.available_zones:
            return
        
        zone_idx = self.zone_combo.currentIndex()
        if zone_idx < 0 or zone_idx >= len(self.available_zones):
            return
        
        zone_name = self.available_zones[zone_idx]
        
        try:
            # Get the file path to save to
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Boundary Mesh",
                f"{zone_name}.vtk",
                "VTK Files (*.vtk);;STL Files (*.stl);;OBJ Files (*.obj);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Import zone extraction functionality
            from openffd.cli.zone_extractor import extract_and_save_boundary
            
            # Extract and save the boundary
            extract_and_save_boundary(
                mesh_file=self.mesh_file_path,
                boundary_name=zone_name,
                output_file=file_path
            )
            
            QMessageBox.information(
                self,
                "Save Boundary",
                f"Successfully saved boundary '{zone_name}' to {file_path}."
            )
            
            logger.info(f"Saved boundary '{zone_name}' to {file_path}")
        except Exception as e:
            logger.error(f"Error saving boundary: {str(e)}")
    
    def _setup_boundary_controls(self):
        """Set up boundary zone visibility controls."""
        if not self.mesh_data or not hasattr(self.mesh_data, 'zones'):
            return
        
        # Clear existing checkboxes
        for checkbox in self.boundary_checkboxes.values():
            checkbox.deleteLater()
        self.boundary_checkboxes.clear()
        
        # Add checkboxes for boundary zones
        boundary_zones = []
        for zone_name, zone_info in self.mesh_data.zones.items():
            zone_type = zone_info.get('type', 'unknown')
            zone_obj = zone_info.get('object')
            
            # Skip volume zones
            if zone_obj and hasattr(zone_obj, 'zone_type_enum'):
                is_volume = zone_obj.zone_type_enum.name == 'VOLUME'
            else:
                is_volume = zone_type in ['interior', 'fluid', 'solid']
            
            if not is_volume:
                boundary_zones.append((zone_name, zone_type))
        
        # Create checkboxes for boundary zones
        for zone_name, zone_type in boundary_zones:
            checkbox = QCheckBox(f"{zone_name} ({zone_type})")
            checkbox.setChecked(True)  # Initially visible
            checkbox.stateChanged.connect(lambda state, name=zone_name: self._on_boundary_visibility_changed(name, state == Qt.CheckState.Checked))
            
            self.boundary_scroll_layout.addWidget(checkbox)
            self.boundary_checkboxes[zone_name] = checkbox
        
        # Show the boundary group if we have boundary zones
        if boundary_zones:
            self.boundary_group.setVisible(True)
            logger.info(f"Created visibility controls for {len(boundary_zones)} boundary zones")
        else:
            self.boundary_group.setVisible(False)
    
    def _show_all_boundaries(self):
        """Show all boundary zones."""
        for checkbox in self.boundary_checkboxes.values():
            checkbox.setChecked(True)
    
    def _hide_all_boundaries(self):
        """Hide all boundary zones."""
        for checkbox in self.boundary_checkboxes.values():
            checkbox.setChecked(False)
    
    def _on_boundary_visibility_changed(self, zone_name: str, visible: bool):
        """Handle boundary zone visibility change.
        
        Args:
            zone_name: Name of the zone
            visible: Whether the zone should be visible
        """
        # Emit signal to update visualization
        logger.info(f"Boundary zone '{zone_name}' visibility changed to {visible}")
        self.boundary_visibility_changed.emit(zone_name, visible)
    
    def load_mesh(self, file_path):
        """Public method to load a mesh from a file path.
        
        Args:
            file_path: Path to the mesh file
        """
        self.mesh_file_path = file_path
        self.file_label.setText(os.path.basename(file_path))
        self.load_button.setEnabled(True)
        self._load_mesh()


class ZoneExtractionThread(QThread):
    """Background thread for zone extraction."""
    
    extraction_completed = pyqtSignal(str, dict)
    extraction_failed = pyqtSignal(str, str)
    
    def __init__(self, mesh_data, zone_name):
        super().__init__()
        self.mesh_data = mesh_data
        self.zone_name = zone_name
    
    def run(self):
        """Run zone extraction in background."""
        try:
            from openffd.mesh.general import extract_zone_mesh
            
            # Extract zone mesh with progress logging
            logger.info(f"Background thread: Starting extraction of zone '{self.zone_name}'")
            zone_mesh_data = extract_zone_mesh(self.mesh_data, self.zone_name)
            
            if zone_mesh_data is not None:
                logger.info(f"Background thread: Extraction completed for zone '{self.zone_name}'")
                self.extraction_completed.emit(self.zone_name, zone_mesh_data)
            else:
                self.extraction_failed.emit(self.zone_name, "No mesh data extracted")
                
        except Exception as e:
            logger.error(f"Background extraction error: {str(e)}")
            self.extraction_failed.emit(self.zone_name, str(e))
