"""Zone extraction panel for OpenFFD GUI.

This module provides a dedicated panel for the advanced zone extraction system,
building on the functionality previously implemented in the CLI.
"""

import logging
import os
from typing import Optional, List, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QScrollArea,
    QComboBox, QGroupBox, QCheckBox, QLineEdit, QFileDialog, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QFont, QColor

from openffd.gui.utils import show_error_dialog, show_info_dialog
from openffd.mesh.fluent_reader import FluentMeshReader

# Configure logging
logger = logging.getLogger(__name__)


class ZoneExtractionPanel(QWidget):
    """Panel for advanced zone extraction."""
    
    # Signal emitted when zone extraction is complete
    zone_extracted = pyqtSignal(object, object, str)  # mesh_data, points, zone_name
    
    def __init__(self, parent=None):
        """Initialize the zone extraction panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.mesh_file_path = None
        self.mesh_data = None
        self.zones = []
        self.boundaries = []
        self.volumes = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Create a splitter for zones and zone details
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Zone listing section
        zone_group = QGroupBox("Available Zones")
        zone_layout = QVBoxLayout()
        
        # Table for zones with properties
        self.zone_table = QTableWidget()
        self.zone_table.setColumnCount(3)
        self.zone_table.setHorizontalHeaderLabels(["Zone Name", "Type", "Element Count"])
        self.zone_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.zone_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.zone_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.zone_table.verticalHeader().setVisible(False)
        self.zone_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.zone_table.itemSelectionChanged.connect(self._on_zone_selected)
        
        zone_layout.addWidget(self.zone_table)
        
        # Zone filtering options
        filter_layout = QHBoxLayout()
        
        # Zone type filtering
        self.type_combo = QComboBox()
        self.type_combo.addItems(["All Zones", "Boundaries Only", "Volumes Only"])
        self.type_combo.currentIndexChanged.connect(self._filter_zones)
        
        # Search box
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search zones...")
        self.search_edit.textChanged.connect(self._filter_zones)
        
        filter_layout.addWidget(QLabel("Filter:"))
        filter_layout.addWidget(self.type_combo)
        filter_layout.addWidget(self.search_edit, 1)
        
        zone_layout.addLayout(filter_layout)
        
        zone_group.setLayout(zone_layout)
        splitter.addWidget(zone_group)
        
        # Zone details section
        details_group = QGroupBox("Zone Details")
        details_layout = QVBoxLayout()
        
        # Properties table
        self.properties_table = QTableWidget()
        self.properties_table.setColumnCount(2)
        self.properties_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.properties_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.properties_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.properties_table.verticalHeader().setVisible(False)
        
        details_layout.addWidget(self.properties_table)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.extract_btn = QPushButton("Extract Selected Zone")
        self.extract_btn.clicked.connect(self._extract_zone)
        self.extract_btn.setEnabled(False)
        
        self.save_btn = QPushButton("Save Zone to File")
        self.save_btn.clicked.connect(self._save_zone)
        self.save_btn.setEnabled(False)
        
        button_layout.addWidget(self.extract_btn)
        button_layout.addWidget(self.save_btn)
        
        details_layout.addLayout(button_layout)
        
        details_group.setLayout(details_layout)
        splitter.addWidget(details_group)
        
        # Set initial splitter sizes
        splitter.setSizes([200, 150])
        
        layout.addWidget(splitter)
    
    def set_mesh(self, mesh_file_path: str, mesh_data: Any):
        """Set the mesh for zone extraction.
        
        Args:
            mesh_file_path: Path to the mesh file
            mesh_data: Loaded mesh data
        """
        self.mesh_file_path = mesh_file_path
        self.mesh_data = mesh_data
        
        # Extract zones from mesh
        self._extract_zones_from_mesh()
        
        # Update UI with zone information
        self._update_zone_list()
    
    def _extract_zones_from_mesh(self):
        """Extract zones from the loaded mesh."""
        if self.mesh_data is None:
            return
        
        try:
            self.zones = []
            self.boundaries = []
            self.volumes = []
            
            # Check for zones in different possible attributes
            if hasattr(self.mesh_data, 'zones') and self.mesh_data.zones:
                # Extract from dedicated zones attribute
                for zone_name, zone_info in self.mesh_data.zones.items():
                    zone_type = zone_info.get('type', 'unknown')
                    element_count = zone_info.get('element_count', 0)
                    
                    zone = {
                        'name': zone_name,
                        'type': zone_type,
                        'element_count': element_count,
                        'properties': zone_info
                    }
                    
                    self.zones.append(zone)
                    
                    # Categorize zones
                    if zone_type in ['wall', 'boundary', 'inlet', 'outlet', 'symmetry']:
                        self.boundaries.append(zone)
                    else:
                        self.volumes.append(zone)
            
            # Check for cell_sets (common in some formats)
            elif hasattr(self.mesh_data, 'cell_sets') and self.mesh_data.cell_sets:
                for zone_name, cells in self.mesh_data.cell_sets.items():
                    zone = {
                        'name': zone_name,
                        'type': 'cell_set',
                        'element_count': len(cells) if hasattr(cells, '__len__') else 0,
                        'properties': {'cell_count': len(cells) if hasattr(cells, '__len__') else 0}
                    }
                    
                    self.zones.append(zone)
                    self.volumes.append(zone)
            
            # Fallback for other formats
            elif hasattr(self.mesh_data, 'cell_data') and self.mesh_data.cell_data:
                for data_name in self.mesh_data.cell_data.keys():
                    zone = {
                        'name': data_name,
                        'type': 'cell_data',
                        'element_count': len(self.mesh_data.cell_data[data_name]),
                        'properties': {'data_type': str(type(self.mesh_data.cell_data[data_name]))}
                    }
                    
                    self.zones.append(zone)
            
            logger.info(f"Found {len(self.zones)} zones: {len(self.boundaries)} boundaries, {len(self.volumes)} volumes")
        except Exception as e:
            logger.error(f"Error extracting zones: {str(e)}")
            show_error_dialog("Error Extracting Zones", str(e))
    
    def _update_zone_list(self):
        """Update the zone list in the UI."""
        self.zone_table.setRowCount(0)  # Clear existing rows
        
        if not self.zones:
            return
        
        # Filter zones based on current settings
        filtered_zones = self._get_filtered_zones()
        
        # Add zones to table
        self.zone_table.setRowCount(len(filtered_zones))
        
        for i, zone in enumerate(filtered_zones):
            # Zone name
            name_item = QTableWidgetItem(zone['name'])
            self.zone_table.setItem(i, 0, name_item)
            
            # Zone type
            type_item = QTableWidgetItem(zone['type'])
            if zone in self.boundaries:
                type_item.setForeground(QColor(0, 120, 215))  # Blue for boundaries
            elif zone in self.volumes:
                type_item.setForeground(QColor(0, 140, 0))  # Green for volumes
            self.zone_table.setItem(i, 1, type_item)
            
            # Element count
            count_item = QTableWidgetItem(str(zone['element_count']))
            self.zone_table.setItem(i, 2, count_item)
    
    def _filter_zones(self):
        """Filter zones based on current filter settings."""
        self._update_zone_list()
    
    def _get_filtered_zones(self) -> List[Dict[str, Any]]:
        """Get zones filtered by current settings.
        
        Returns:
            List of filtered zones
        """
        filter_type = self.type_combo.currentText()
        search_text = self.search_edit.text().lower()
        
        if filter_type == "Boundaries Only":
            zones = self.boundaries
        elif filter_type == "Volumes Only":
            zones = self.volumes
        else:
            zones = self.zones
        
        if search_text:
            return [zone for zone in zones if search_text in zone['name'].lower()]
        else:
            return zones
    
    def _on_zone_selected(self):
        """Handle zone selection in the table."""
        selected_rows = self.zone_table.selectedIndexes()
        
        if not selected_rows:
            self.extract_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self._clear_properties_table()
            return
        
        row = selected_rows[0].row()
        zone_name = self.zone_table.item(row, 0).text()
        
        # Find the selected zone
        selected_zone = None
        for zone in self.zones:
            if zone['name'] == zone_name:
                selected_zone = zone
                break
        
        if selected_zone:
            # Update properties table
            self._update_properties_table(selected_zone)
            
            # Enable action buttons
            self.extract_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
    
    def _update_properties_table(self, zone: Dict[str, Any]):
        """Update the properties table with zone details.
        
        Args:
            zone: Zone dictionary
        """
        self.properties_table.setRowCount(0)  # Clear existing rows
        
        # Add basic properties
        self._add_property("Name", zone['name'])
        self._add_property("Type", zone['type'])
        self._add_property("Element Count", str(zone['element_count']))
        
        # Add additional properties
        if 'properties' in zone and isinstance(zone['properties'], dict):
            for key, value in zone['properties'].items():
                if key not in ['name', 'type', 'element_count']:
                    self._add_property(key, str(value))
    
    def _add_property(self, name: str, value: str):
        """Add a property to the properties table.
        
        Args:
            name: Property name
            value: Property value
        """
        row = self.properties_table.rowCount()
        self.properties_table.insertRow(row)
        
        name_item = QTableWidgetItem(name)
        value_item = QTableWidgetItem(value)
        
        self.properties_table.setItem(row, 0, name_item)
        self.properties_table.setItem(row, 1, value_item)
    
    def _clear_properties_table(self):
        """Clear the properties table."""
        self.properties_table.setRowCount(0)
    
    def _extract_zone(self):
        """Extract the selected zone."""
        selected_rows = self.zone_table.selectedIndexes()
        
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        zone_name = self.zone_table.item(row, 0).text()
        
        try:
            # Import extraction functionality
            from openffd.mesh.general import extract_patch_points
            
            # Extract zone points
            zone_points = extract_patch_points(self.mesh_data, zone_name)
            
            if zone_points is None or len(zone_points) == 0:
                show_error_dialog("Zone Extraction", f"No points found in zone '{zone_name}'")
                return
            
            # Emit signal with extracted zone
            self.zone_extracted.emit(self.mesh_data, zone_points, zone_name)
            
            show_info_dialog("Zone Extraction", f"Successfully extracted zone '{zone_name}' with {len(zone_points)} points")
            
            logger.info(f"Extracted zone '{zone_name}' with {len(zone_points)} points")
        except Exception as e:
            logger.error(f"Error extracting zone: {str(e)}")
            show_error_dialog("Zone Extraction Error", str(e))
    
    def _save_zone(self):
        """Save the selected zone to a file."""
        selected_rows = self.zone_table.selectedIndexes()
        
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        zone_name = self.zone_table.item(row, 0).text()
        
        try:
            # Ask for save location
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Zone",
                f"{zone_name}.vtk",
                "VTK Files (*.vtk);;STL Files (*.stl);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Import zone extraction functionality
            from openffd.cli.zone_extractor import extract_and_save_boundary
            
            # Extract and save the zone
            extract_and_save_boundary(
                mesh_file=self.mesh_file_path,
                boundary_name=zone_name,
                output_file=file_path
            )
            
            show_info_dialog("Save Zone", f"Successfully saved zone '{zone_name}' to {file_path}")
            
            logger.info(f"Saved zone '{zone_name}' to {file_path}")
        except Exception as e:
            logger.error(f"Error saving zone: {str(e)}")
            show_error_dialog("Save Zone Error", str(e))
