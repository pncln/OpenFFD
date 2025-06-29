"""Mesh panel for OpenFFD GUI.

This module provides the UI panel for mesh loading and configuration.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QGroupBox, QCheckBox, QLineEdit, QSpinBox,
    QScrollArea, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

from openffd.mesh.general import read_general_mesh, is_fluent_mesh
from openffd.mesh.fluent_reader import FluentMeshReader
from openffd.gui.utils import show_error_dialog

# Configure logging
logger = logging.getLogger(__name__)


class MeshPanel(QWidget):
    """Panel for mesh loading and configuration."""
    
    # Signal emitted when a mesh is loaded
    mesh_loaded = pyqtSignal(object, object)
    
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
                    self._load_fluent_mesh(use_binary)
                else:
                    # Generic mesh reader
                    self._load_generic_mesh()
            elif format_text == "Fluent":
                self._load_fluent_mesh(use_binary)
            else:
                # Generic mesh reader for other formats
                self._load_generic_mesh()
            
            # Update UI with mesh information
            self._update_mesh_statistics()
            self._update_zone_list()
            
            # Enable zone-related buttons
            self.list_zones_button.setEnabled(True)
            
            # Emit mesh loaded signal
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
                self.zone_combo.addItems(self.available_zones)
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
        
        try:
            # Import extraction functionality here to avoid circular imports
            from openffd.mesh.general import extract_patch_points
            
            # Extract points for the selected zone
            zone_points = extract_patch_points(self.mesh_data, zone_name)
            
            if zone_points is not None and len(zone_points) > 0:
                # Update mesh points to use only the selected zone
                self.mesh_points = zone_points
                
                # Update statistics
                self._update_mesh_statistics()
                
                # Notify about the change
                self.mesh_loaded.emit(self.mesh_data, self.mesh_points)
                
                QMessageBox.information(
                    self,
                    "Zone Extraction",
                    f"Successfully extracted zone '{zone_name}' with {len(zone_points)} points."
                )
                
                logger.info(f"Extracted zone '{zone_name}' with {len(zone_points)} points")
            else:
                QMessageBox.warning(
                    self,
                    "Zone Extraction",
                    f"No points found in zone '{zone_name}'."
                )
        except Exception as e:
            logger.error(f"Error extracting zone: {str(e)}")
            show_error_dialog("Zone Extraction Error", str(e))
    
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
            show_error_dialog("Save Boundary Error", str(e))
    
    def load_mesh(self, file_path):
        """Public method to load a mesh from a file path.
        
        Args:
            file_path: Path to the mesh file
        """
        self.mesh_file_path = file_path
        self.file_label.setText(os.path.basename(file_path))
        self.load_button.setEnabled(True)
        self._load_mesh()
