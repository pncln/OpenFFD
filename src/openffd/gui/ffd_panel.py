"""FFD panel for OpenFFD GUI.

This module provides the UI panel for FFD box configuration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QGroupBox, QCheckBox, QLineEdit,
    QRadioButton, QButtonGroup, QTreeWidget, QTreeWidgetItem, QSplitter, QComboBox,
    QScrollArea
)

from openffd.core.hierarchical import HierarchicalFFD, HierarchicalLevel, create_hierarchical_ffd
from openffd.utils.parallel import ParallelConfig

# Configure logging
logger = logging.getLogger(__name__)


class FFDPanel(QWidget):
    """Panel for FFD box configuration."""
    
    # Signal emitted when FFD parameters are changed
    ffd_parameters_changed = pyqtSignal()
    
    # Signal emitted when hierarchical FFD is updated
    hierarchical_ffd_updated = pyqtSignal(object)  # Emits HierarchicalFFD object
    
    def __init__(self, parent=None):
        """Initialize the FFD panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.mesh_min_coords = None
        self.mesh_max_coords = None
        self.mesh_points = None
        self.hierarchical_ffd = None
        self.ffd_mode = "standard"  # "standard" or "hierarchical"
        self.ffd_shape_mode = "box"  # "box", "convex", or "surface"
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        
        # FFD Mode selection
        mode_group = QGroupBox("FFD Mode")
        mode_layout = QHBoxLayout(mode_group)
        
        # Radio buttons for FFD mode
        self.standard_mode_radio = QRadioButton("Standard FFD")
        self.standard_mode_radio.setChecked(True)
        self.hierarchical_mode_radio = QRadioButton("Hierarchical FFD")
        
        # Group radio buttons
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.standard_mode_radio, 1)
        self.mode_group.addButton(self.hierarchical_mode_radio, 2)
        self.mode_group.buttonClicked.connect(self._on_mode_changed)
        
        mode_layout.addWidget(self.standard_mode_radio)
        mode_layout.addWidget(self.hierarchical_mode_radio)
        main_layout.addWidget(mode_group)
        
        # FFD Generation Shape Mode selection
        shape_group = QGroupBox("FFD Generation Shape")
        shape_layout = QHBoxLayout(shape_group)
        
        # Radio buttons for FFD generation shape
        self.box_shape_radio = QRadioButton("Box (Rectangular)")
        self.box_shape_radio.setChecked(True)
        self.box_shape_radio.setToolTip("Traditional rectangular FFD box that fits the domain")
        
        self.convex_shape_radio = QRadioButton("Convex Hull")
        self.convex_shape_radio.setToolTip("FFD that tightly encloses the domain with no empty space")
        
        self.surface_shape_radio = QRadioButton("Surface-Fitted")
        self.surface_shape_radio.setToolTip("FFD that follows the geometry surface contours (e.g., wing-shaped)")
        
        # Group radio buttons for shape mode
        self.shape_group = QButtonGroup(self)
        self.shape_group.addButton(self.box_shape_radio, 1)
        self.shape_group.addButton(self.convex_shape_radio, 2)
        self.shape_group.addButton(self.surface_shape_radio, 3)
        self.shape_group.buttonClicked.connect(self._on_shape_mode_changed)
        
        shape_layout.addWidget(self.box_shape_radio)
        shape_layout.addWidget(self.convex_shape_radio)
        shape_layout.addWidget(self.surface_shape_radio)
        main_layout.addWidget(shape_group)
        
        # Create a scroll area for the content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create content widget that will contain both standard and hierarchical UIs
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # Stack for different mode UIs
        self.standard_widget = QWidget()
        self.hierarchical_widget = QWidget()
        
        # Setup standard FFD UI
        self._setup_standard_ui()
        
        # Setup hierarchical FFD UI
        self._setup_hierarchical_ui()
        
        # Add both widgets to layout
        layout.addWidget(self.standard_widget)
        layout.addWidget(self.hierarchical_widget)
        
        # Set the content widget as the scroll area's widget
        scroll_area.setWidget(content_widget)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        
        # Show the appropriate widget based on initial mode
        self._update_mode_visibility()
    
    def _setup_hierarchical_ui(self):
        """Setup UI for hierarchical FFD mode."""
        layout = QVBoxLayout(self.hierarchical_widget)
        
        # Base dimensions section
        dims_group = QGroupBox("Base Control Dimensions")
        dims_layout = QHBoxLayout()
        dims_layout.addWidget(QLabel("Base Dimensions:"))
        
        self.base_dims_nx = QSpinBox()
        self.base_dims_nx.setRange(2, 20)
        self.base_dims_nx.setValue(4)
        self.base_dims_nx.setToolTip("Number of control points in X direction")
        
        self.base_dims_ny = QSpinBox()
        self.base_dims_ny.setRange(2, 20)
        self.base_dims_ny.setValue(4)
        self.base_dims_ny.setToolTip("Number of control points in Y direction")
        
        self.base_dims_nz = QSpinBox()
        self.base_dims_nz.setRange(2, 20)
        self.base_dims_nz.setValue(4)
        self.base_dims_nz.setToolTip("Number of control points in Z direction")
        
        dims_layout.addWidget(self.base_dims_nx)
        dims_layout.addWidget(self.base_dims_ny)
        dims_layout.addWidget(self.base_dims_nz)
        
        dims_group.setLayout(dims_layout)
        layout.addWidget(dims_group)
        
        # Hierarchy options
        hierarchy_group = QGroupBox("Hierarchy Options")
        hierarchy_layout = QFormLayout(hierarchy_group)
        
        # Max depth
        self.max_depth = QSpinBox()
        self.max_depth.setRange(1, 5)
        self.max_depth.setValue(3)
        self.max_depth.setToolTip("Maximum depth of the hierarchical FFD")
        hierarchy_layout.addRow("Max Depth:", self.max_depth)
        
        # Subdivision factor
        self.subdiv_factor = QSpinBox()
        self.subdiv_factor.setRange(2, 4)
        self.subdiv_factor.setValue(2)
        self.subdiv_factor.setToolTip("Subdivision factor between hierarchy levels")
        hierarchy_layout.addRow("Subdivision:", self.subdiv_factor)
        
        # Margin
        self.h_margin = QDoubleSpinBox()
        self.h_margin.setRange(0.0, 0.5)
        self.h_margin.setValue(0.05)
        self.h_margin.setSingleStep(0.01)
        self.h_margin.setToolTip("Margin around mesh for control box")
        hierarchy_layout.addRow("Margin:", self.h_margin)
        
        layout.addWidget(hierarchy_group)
        
        # Custom bounds for hierarchical FFD
        h_bounds_group = QGroupBox("Custom Bounds")
        h_bounds_layout = QVBoxLayout()
        
        # X bounds
        h_x_bounds_layout = QFormLayout()
        self.h_x_min_check = QCheckBox()
        self.h_x_min_check.stateChanged.connect(self._on_bounds_check_changed)
        self.h_x_min_spin = QDoubleSpinBox()
        self.h_x_min_spin.setEnabled(False)
        self.h_x_min_spin.setRange(-1000000, 1000000)
        self.h_x_min_spin.setDecimals(3)
        self.h_x_min_spin.setSingleStep(0.1)
        self.h_x_min_spin.valueChanged.connect(self._on_params_changed)
        
        self.h_x_max_check = QCheckBox()
        self.h_x_max_check.stateChanged.connect(self._on_bounds_check_changed)
        self.h_x_max_spin = QDoubleSpinBox()
        self.h_x_max_spin.setEnabled(False)
        self.h_x_max_spin.setRange(-1000000, 1000000)
        self.h_x_max_spin.setDecimals(3)
        self.h_x_max_spin.setSingleStep(0.1)
        self.h_x_max_spin.valueChanged.connect(self._on_params_changed)
        
        h_x_min_layout = QHBoxLayout()
        h_x_min_layout.addWidget(self.h_x_min_check)
        h_x_min_layout.addWidget(self.h_x_min_spin)
        
        h_x_max_layout = QHBoxLayout()
        h_x_max_layout.addWidget(self.h_x_max_check)
        h_x_max_layout.addWidget(self.h_x_max_spin)
        
        h_x_bounds_layout.addRow("X Min:", h_x_min_layout)
        h_x_bounds_layout.addRow("X Max:", h_x_max_layout)
        h_bounds_layout.addLayout(h_x_bounds_layout)
        
        # Y bounds
        h_y_bounds_layout = QFormLayout()
        self.h_y_min_check = QCheckBox()
        self.h_y_min_check.stateChanged.connect(self._on_bounds_check_changed)
        self.h_y_min_spin = QDoubleSpinBox()
        self.h_y_min_spin.setEnabled(False)
        self.h_y_min_spin.setRange(-1000000, 1000000)
        self.h_y_min_spin.setDecimals(3)
        self.h_y_min_spin.setSingleStep(0.1)
        self.h_y_min_spin.valueChanged.connect(self._on_params_changed)
        
        self.h_y_max_check = QCheckBox()
        self.h_y_max_check.stateChanged.connect(self._on_bounds_check_changed)
        self.h_y_max_spin = QDoubleSpinBox()
        self.h_y_max_spin.setEnabled(False)
        self.h_y_max_spin.setRange(-1000000, 1000000)
        self.h_y_max_spin.setDecimals(3)
        self.h_y_max_spin.setSingleStep(0.1)
        self.h_y_max_spin.valueChanged.connect(self._on_params_changed)
        
        h_y_min_layout = QHBoxLayout()
        h_y_min_layout.addWidget(self.h_y_min_check)
        h_y_min_layout.addWidget(self.h_y_min_spin)
        
        h_y_max_layout = QHBoxLayout()
        h_y_max_layout.addWidget(self.h_y_max_check)
        h_y_max_layout.addWidget(self.h_y_max_spin)
        
        h_y_bounds_layout.addRow("Y Min:", h_y_min_layout)
        h_y_bounds_layout.addRow("Y Max:", h_y_max_layout)
        h_bounds_layout.addLayout(h_y_bounds_layout)
        
        # Z bounds
        h_z_bounds_layout = QFormLayout()
        self.h_z_min_check = QCheckBox()
        self.h_z_min_check.stateChanged.connect(self._on_bounds_check_changed)
        self.h_z_min_spin = QDoubleSpinBox()
        self.h_z_min_spin.setEnabled(False)
        self.h_z_min_spin.setRange(-1000000, 1000000)
        self.h_z_min_spin.setDecimals(3)
        self.h_z_min_spin.setSingleStep(0.1)
        self.h_z_min_spin.valueChanged.connect(self._on_params_changed)
        
        self.h_z_max_check = QCheckBox()
        self.h_z_max_check.stateChanged.connect(self._on_bounds_check_changed)
        self.h_z_max_spin = QDoubleSpinBox()
        self.h_z_max_spin.setEnabled(False)
        self.h_z_max_spin.setRange(-1000000, 1000000)
        self.h_z_max_spin.setDecimals(3)
        self.h_z_max_spin.setSingleStep(0.1)
        self.h_z_max_spin.valueChanged.connect(self._on_params_changed)
        
        h_z_min_layout = QHBoxLayout()
        h_z_min_layout.addWidget(self.h_z_min_check)
        h_z_min_layout.addWidget(self.h_z_min_spin)
        
        h_z_max_layout = QHBoxLayout()
        h_z_max_layout.addWidget(self.h_z_max_check)
        h_z_max_layout.addWidget(self.h_z_max_spin)
        
        h_z_bounds_layout.addRow("Z Min:", h_z_min_layout)
        h_z_bounds_layout.addRow("Z Max:", h_z_max_layout)
        h_bounds_layout.addLayout(h_z_bounds_layout)
        
        h_bounds_group.setLayout(h_bounds_layout)
        layout.addWidget(h_bounds_group)
        
        # Levels tree
        levels_group = QGroupBox("FFD Levels")
        levels_layout = QVBoxLayout(levels_group)
        
        # Tree for showing levels
        self.levels_tree = QTreeWidget()
        self.levels_tree.setHeaderLabels(["Level", "Dimensions", "Weight"])
        self.levels_tree.setColumnWidth(0, 80)
        self.levels_tree.setColumnWidth(1, 100)
        levels_layout.addWidget(self.levels_tree)
        
        layout.addWidget(levels_group)
        
        # Visualization options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QVBoxLayout(viz_group)
        
        # Show mesh checkbox
        self.show_mesh_cb = QCheckBox("Show Mesh")
        self.show_mesh_cb.setChecked(True)
        viz_layout.addWidget(self.show_mesh_cb)
        
        # Color by level checkbox
        self.color_by_level_cb = QCheckBox("Color by Level")
        self.color_by_level_cb.setChecked(True)
        viz_layout.addWidget(self.color_by_level_cb)
        
        layout.addWidget(viz_group)
    
    def _update_mode_visibility(self):
        """Update UI visibility based on selected FFD mode."""
        if self.ffd_mode == "standard":
            self.standard_widget.setVisible(True)
            self.hierarchical_widget.setVisible(False)
        else:  # hierarchical mode
            self.standard_widget.setVisible(False)
            self.hierarchical_widget.setVisible(True)
    
    def _on_mode_changed(self, button):
        """Handle FFD mode change."""
        if button == self.standard_mode_radio:
            self.ffd_mode = "standard"
        else:
            self.ffd_mode = "hierarchical"
        
        self._update_mode_visibility()
        self.ffd_parameters_changed.emit()
    
    def _on_shape_mode_changed(self, button):
        """Handle FFD shape mode change."""
        if button == self.box_shape_radio:
            self.ffd_shape_mode = "box"
        elif button == self.convex_shape_radio:
            self.ffd_shape_mode = "convex"
        else:  # surface_shape_radio
            self.ffd_shape_mode = "surface"
        
        logger.info(f"FFD shape mode changed to: {self.ffd_shape_mode}")
        self.ffd_parameters_changed.emit()
    
    def set_mesh_points(self, mesh_points: np.ndarray):
        """Set the mesh points for FFD generation.
        
        Args:
            mesh_points: Numpy array of mesh point coordinates
        """
        self.mesh_points = mesh_points
    
    def get_ffd_mode(self):
        """Get the current FFD mode.
        
        Returns:
            String indicating the FFD mode ("standard" or "hierarchical")
        """
        return self.ffd_mode
    
    def get_ffd_shape_mode(self):
        """Get the current FFD shape generation mode.
        
        Returns:
            String indicating the FFD shape mode ("box", "convex", or "surface")
        """
        return self.ffd_shape_mode
    
    def get_hierarchical_custom_bounds(self):
        """Get custom bounds for hierarchical FFD if specified.
        
        Returns:
            List of tuples in the format expected by create_ffd_box:
            [(min_x, max_x), (min_y, max_y), (min_z, max_z)]
            where any unspecified value will be None
        """
        # Initialize with None values for unspecified bounds
        x_min = self.h_x_min_spin.value() if self.h_x_min_check.isChecked() else None
        x_max = self.h_x_max_spin.value() if self.h_x_max_check.isChecked() else None
        y_min = self.h_y_min_spin.value() if self.h_y_min_check.isChecked() else None
        y_max = self.h_y_max_spin.value() if self.h_y_max_check.isChecked() else None
        z_min = self.h_z_min_spin.value() if self.h_z_min_check.isChecked() else None
        z_max = self.h_z_max_spin.value() if self.h_z_max_check.isChecked() else None
        
        # Debug: Log the values from spinboxes
        logger.debug(f"Hierarchical FFD bounds from UI: X: [{x_min}, {x_max}], Y: [{y_min}, {y_max}], Z: [{z_min}, {z_max}]")
        logger.debug(f"Checkbox states: X: [{self.h_x_min_check.isChecked()}, {self.h_x_max_check.isChecked()}], "
                     f"Y: [{self.h_y_min_check.isChecked()}, {self.h_y_max_check.isChecked()}], "
                     f"Z: [{self.h_z_min_check.isChecked()}, {self.h_z_max_check.isChecked()}]")
        
        # Validate that min values are less than max values when both are specified
        if x_min is not None and x_max is not None and x_min >= x_max:
            logger.warning(f"X min value {x_min} must be less than X max value {x_max}, ignoring these bounds")
            x_min = None
            x_max = None
            
        if y_min is not None and y_max is not None and y_min >= y_max:
            logger.warning(f"Y min value {y_min} must be less than Y max value {y_max}, ignoring these bounds")
            y_min = None
            y_max = None
            
        if z_min is not None and z_max is not None and z_min >= z_max:
            logger.warning(f"Z min value {z_min} must be less than Z max value {z_max}, ignoring these bounds")
            z_min = None
            z_max = None
        
        # Create the custom bounds list with exactly 3 dimension tuples as expected by create_ffd_box
        # This is critically important - the order matters and we need all 3 dimensions
        custom_bounds = [
            (x_min, x_max),
            (y_min, y_max),
            (z_min, z_max)
        ]
        
        # Only return custom bounds if at least one value is specified
        has_custom_bound = any(val is not None for tup in custom_bounds for val in tup)
        
        logger.debug(f"Final hierarchical FFD custom bounds: {custom_bounds if has_custom_bound else None}")
        return custom_bounds if has_custom_bound else None
    
    def create_hierarchical_ffd(self):
        """Create a hierarchical FFD from current settings.
        
        Returns:
            Tuple of (hierarchical_ffd, success)
        """
        if self.mesh_points is None or len(self.mesh_points) == 0:
            logger.error("No mesh points available for hierarchical FFD")
            return None, False
        
        try:
            # Get configuration from UI
            base_dims = (
                self.base_dims_nx.value(),
                self.base_dims_ny.value(),
                self.base_dims_nz.value()
            )
            max_depth = self.max_depth.value()
            subdivision_factor = self.subdiv_factor.value()
            margin = self.h_margin.value()
            
            # Get custom bounds if specified
            custom_bounds = self.get_hierarchical_custom_bounds()
            
            # Create the hierarchical FFD
            logger.info(f"Creating hierarchical FFD with base dims {base_dims}, depth {max_depth}, subdivision {subdivision_factor}")
            hierarchical_ffd = create_hierarchical_ffd(
                mesh_points=self.mesh_points,
                base_dims=base_dims,
                max_depth=max_depth,
                subdivision_factor=subdivision_factor,
                margin=margin,
                custom_dims=custom_bounds
            )
            
            # Update UI
            self._update_levels_tree(hierarchical_ffd)
            
            # Store the hierarchical FFD
            self.hierarchical_ffd = hierarchical_ffd
            
            return hierarchical_ffd, True
            
        except Exception as e:
            logger.error(f"Error creating hierarchical FFD: {str(e)}")
            return None, False
    
    def _update_levels_tree(self, hierarchical_ffd):
        """Update the levels tree with the current hierarchical FFD.
        
        Args:
            hierarchical_ffd: The HierarchicalFFD object
        """
        self.levels_tree.clear()
        
        if hierarchical_ffd is None:
            return
        
        try:
            # Get level information
            levels = hierarchical_ffd.levels
            
            # Add levels to tree
            for level_id, level in levels.items():
                item = QTreeWidgetItem(self.levels_tree)
                item.setText(0, f"Level {level_id}")
                item.setText(1, f"{level.dims[0]}×{level.dims[1]}×{level.dims[2]}")
                item.setText(2, f"{level.weight_factor:.2f}" if hasattr(level, 'weight_factor') else "1.00")
                
                # Store level_id as data
                item.setData(0, Qt.ItemDataRole.UserRole, level_id)
        except Exception as e:
            logger.error(f"Error updating levels tree: {str(e)}")
    
    def _setup_standard_ui(self):
        """Setup UI for standard FFD mode."""
        layout = QVBoxLayout(self.standard_widget)
        
        # Control dimensions section
        dim_group = QGroupBox("Control Point Dimensions")
        dim_layout = QFormLayout(dim_group)
        
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(2, 100)
        self.nx_spin.setValue(4)
        self.nx_spin.valueChanged.connect(self._on_params_changed)
        
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(2, 100)
        self.ny_spin.setValue(4)
        self.ny_spin.valueChanged.connect(self._on_params_changed)
        
        self.nz_spin = QSpinBox()
        self.nz_spin.setRange(2, 100)
        self.nz_spin.setValue(4)
        self.nz_spin.valueChanged.connect(self._on_params_changed)
        
        dim_layout.addRow("Number of X control points:", self.nx_spin)
        dim_layout.addRow("Number of Y control points:", self.ny_spin)
        dim_layout.addRow("Number of Z control points:", self.nz_spin)
        
        dim_group.setLayout(dim_layout)
        layout.addWidget(dim_group)
        
        # Margin section
        margin_group = QGroupBox("Box Margin")
        margin_layout = QFormLayout()
        
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 100.0)
        self.margin_spin.setValue(0.0)
        self.margin_spin.setSingleStep(0.1)
        self.margin_spin.setDecimals(3)
        self.margin_spin.valueChanged.connect(self._on_params_changed)
        
        margin_layout.addRow("Margin padding:", self.margin_spin)
        
        margin_group.setLayout(margin_layout)
        layout.addWidget(margin_group)
        
        # Custom bounds section
        bounds_group = QGroupBox("Custom Bounds")
        bounds_layout = QVBoxLayout()
        
        # X bounds
        x_bounds_layout = QFormLayout()
        self.x_min_check = QCheckBox()
        self.x_min_check.stateChanged.connect(self._on_bounds_check_changed)
        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setEnabled(False)
        self.x_min_spin.setRange(-1000000, 1000000)
        self.x_min_spin.setDecimals(3)
        self.x_min_spin.setSingleStep(0.1)
        self.x_min_spin.valueChanged.connect(self._on_params_changed)
        
        self.x_max_check = QCheckBox()
        self.x_max_check.stateChanged.connect(self._on_bounds_check_changed)
        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setEnabled(False)
        self.x_max_spin.setRange(-1000000, 1000000)
        self.x_max_spin.setDecimals(3)
        self.x_max_spin.setSingleStep(0.1)
        self.x_max_spin.valueChanged.connect(self._on_params_changed)
        
        x_min_layout = QHBoxLayout()
        x_min_layout.addWidget(self.x_min_check)
        x_min_layout.addWidget(self.x_min_spin)
        
        x_max_layout = QHBoxLayout()
        x_max_layout.addWidget(self.x_max_check)
        x_max_layout.addWidget(self.x_max_spin)
        
        x_bounds_layout.addRow("X Min:", x_min_layout)
        x_bounds_layout.addRow("X Max:", x_max_layout)
        bounds_layout.addLayout(x_bounds_layout)
        
        # Y bounds
        y_bounds_layout = QFormLayout()
        self.y_min_check = QCheckBox()
        self.y_min_check.stateChanged.connect(self._on_bounds_check_changed)
        self.y_min_spin = QDoubleSpinBox()
        self.y_min_spin.setEnabled(False)
        self.y_min_spin.setRange(-1000000, 1000000)
        self.y_min_spin.setDecimals(3)
        self.y_min_spin.setSingleStep(0.1)
        self.y_min_spin.valueChanged.connect(self._on_params_changed)
        
        self.y_max_check = QCheckBox()
        self.y_max_check.stateChanged.connect(self._on_bounds_check_changed)
        self.y_max_spin = QDoubleSpinBox()
        self.y_max_spin.setEnabled(False)
        self.y_max_spin.setRange(-1000000, 1000000)
        self.y_max_spin.setDecimals(3)
        self.y_max_spin.setSingleStep(0.1)
        self.y_max_spin.valueChanged.connect(self._on_params_changed)
        
        y_min_layout = QHBoxLayout()
        y_min_layout.addWidget(self.y_min_check)
        y_min_layout.addWidget(self.y_min_spin)
        
        y_max_layout = QHBoxLayout()
        y_max_layout.addWidget(self.y_max_check)
        y_max_layout.addWidget(self.y_max_spin)
        
        y_bounds_layout.addRow("Y Min:", y_min_layout)
        y_bounds_layout.addRow("Y Max:", y_max_layout)
        bounds_layout.addLayout(y_bounds_layout)
        
        # Z bounds
        z_bounds_layout = QFormLayout()
        self.z_min_check = QCheckBox()
        self.z_min_check.stateChanged.connect(self._on_bounds_check_changed)
        self.z_min_spin = QDoubleSpinBox()
        self.z_min_spin.setEnabled(False)
        self.z_min_spin.setRange(-1000000, 1000000)
        self.z_min_spin.setDecimals(3)
        self.z_min_spin.setSingleStep(0.1)
        self.z_min_spin.valueChanged.connect(self._on_params_changed)
        
        self.z_max_check = QCheckBox()
        self.z_max_check.stateChanged.connect(self._on_bounds_check_changed)
        self.z_max_spin = QDoubleSpinBox()
        self.z_max_spin.setEnabled(False)
        self.z_max_spin.setRange(-1000000, 1000000)
        self.z_max_spin.setDecimals(3)
        self.z_max_spin.setSingleStep(0.1)
        self.z_max_spin.valueChanged.connect(self._on_params_changed)
        
        z_min_layout = QHBoxLayout()
        z_min_layout.addWidget(self.z_min_check)
        z_min_layout.addWidget(self.z_min_spin)
        
        z_max_layout = QHBoxLayout()
        z_max_layout.addWidget(self.z_max_check)
        z_max_layout.addWidget(self.z_max_spin)
        
        z_bounds_layout.addRow("Z Min:", z_min_layout)
        z_bounds_layout.addRow("Z Max:", z_max_layout)
        bounds_layout.addLayout(z_bounds_layout)
        
        bounds_group.setLayout(bounds_layout)
        layout.addWidget(bounds_group)
        
        # Advanced options section
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout()
        
        # Hierarchical FFD options
        self.hierarchical_check = QCheckBox("Enable Hierarchical FFD")
        self.hierarchical_check.setToolTip("Use hierarchical FFD for more complex deformations")
        self.hierarchical_check.stateChanged.connect(self._on_hierarchical_changed)
        
        hierarchical_form = QFormLayout()
        
        self.hierarchy_levels_spin = QSpinBox()
        self.hierarchy_levels_spin.setRange(1, 5)
        self.hierarchy_levels_spin.setValue(2)
        self.hierarchy_levels_spin.setEnabled(False)
        self.hierarchy_levels_spin.valueChanged.connect(self._on_params_changed)
        
        hierarchical_form.addRow("Number of levels:", self.hierarchy_levels_spin)
        
        advanced_layout.addWidget(self.hierarchical_check)
        advanced_layout.addLayout(hierarchical_form)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # Add stretch to push everything to the top
        layout.addStretch(1)
    
    def update_bounds(self, min_coords, max_coords):
        """Update the bounds controls with mesh min/max coordinates.
        
        Args:
            min_coords: Minimum coordinates of the mesh [x_min, y_min, z_min]
            max_coords: Maximum coordinates of the mesh [x_max, y_max, z_max]
        """
        self.mesh_min_coords = min_coords
        self.mesh_max_coords = max_coords
        
        # Set bounds without triggering signals for standard FFD
        self.x_min_spin.blockSignals(True)
        self.x_max_spin.blockSignals(True)
        self.y_min_spin.blockSignals(True)
        self.y_max_spin.blockSignals(True)
        self.z_min_spin.blockSignals(True)
        self.z_max_spin.blockSignals(True)
        
        # Set bounds without triggering signals for hierarchical FFD
        self.h_x_min_spin.blockSignals(True)
        self.h_x_max_spin.blockSignals(True)
        self.h_y_min_spin.blockSignals(True)
        self.h_y_max_spin.blockSignals(True)
        self.h_z_min_spin.blockSignals(True)
        self.h_z_max_spin.blockSignals(True)
        
        # Update standard FFD spin box values
        self.x_min_spin.setValue(min_coords[0])
        self.x_max_spin.setValue(max_coords[0])
        self.y_min_spin.setValue(min_coords[1])
        self.y_max_spin.setValue(max_coords[1])
        self.z_min_spin.setValue(min_coords[2])
        self.z_max_spin.setValue(max_coords[2])
        
        # Update hierarchical FFD spin box values
        self.h_x_min_spin.setValue(min_coords[0])
        self.h_x_max_spin.setValue(max_coords[0])
        self.h_y_min_spin.setValue(min_coords[1])
        self.h_y_max_spin.setValue(max_coords[1])
        self.h_z_min_spin.setValue(min_coords[2])
        self.h_z_max_spin.setValue(max_coords[2])
        
        # Re-enable signals for standard FFD
        self.x_min_spin.blockSignals(False)
        self.x_max_spin.blockSignals(False)
        self.y_min_spin.blockSignals(False)
        self.y_max_spin.blockSignals(False)
        self.z_min_spin.blockSignals(False)
        self.z_max_spin.blockSignals(False)
        
        # Re-enable signals for hierarchical FFD
        self.h_x_min_spin.blockSignals(False)
        self.h_x_max_spin.blockSignals(False)
        self.h_y_min_spin.blockSignals(False)
        self.h_y_max_spin.blockSignals(False)
        self.h_z_min_spin.blockSignals(False)
        self.h_z_max_spin.blockSignals(False)
        
        logger.debug(f"Updated bounds: min={min_coords}, max={max_coords}")
    
    def get_control_dimensions(self) -> Tuple[int, int, int]:
        """Get the current control point dimensions.
        
        Returns:
            Tuple of (nx, ny, nz) control point dimensions
        """
        return (
            self.nx_spin.value(),
            self.ny_spin.value(),
            self.nz_spin.value()
        )
    
    def get_margin(self) -> float:
        """Get the current margin value.
        
        Returns:
            Margin value
        """
        return self.margin_spin.value()
    
    def get_custom_bounds(self) -> List[Optional[Tuple[Optional[float], Optional[float]]]]:
        """Get the custom bounds if specified.
        
        Returns:
            List of tuples [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
            where any value can be None if not specified
        """
        custom_dims = [None, None, None]
        
        # X bounds
        x_min = self.x_min_spin.value() if self.x_min_check.isChecked() else None
        x_max = self.x_max_spin.value() if self.x_max_check.isChecked() else None
        if x_min is not None or x_max is not None:
            custom_dims[0] = (x_min, x_max)
        
        # Y bounds
        y_min = self.y_min_spin.value() if self.y_min_check.isChecked() else None
        y_max = self.y_max_spin.value() if self.y_max_check.isChecked() else None
        if y_min is not None or y_max is not None:
            custom_dims[1] = (y_min, y_max)
        
        # Z bounds
        z_min = self.z_min_spin.value() if self.z_min_check.isChecked() else None
        z_max = self.z_max_spin.value() if self.z_max_check.isChecked() else None
        if z_min is not None or z_max is not None:
            custom_dims[2] = (z_min, z_max)
        
        return custom_dims
    
    def get_hierarchical_options(self) -> Dict[str, Any]:
        """Get hierarchical FFD options if enabled.
        
        Returns:
            Dictionary of hierarchical FFD options or None if not enabled
        """
        if not self.hierarchical_check.isChecked():
            return None
        
        return {
            'enabled': True,
            'levels': self.hierarchy_levels_spin.value()
        }
    
    @pyqtSlot(int)
    def _on_bounds_check_changed(self, state):
        """Handle bounds checkbox state changes."""
        sender = self.sender()
        
        # Standard FFD bounds
        if sender == self.x_min_check:
            self.x_min_spin.setEnabled(state == Qt.CheckState.Checked.value)
        elif sender == self.x_max_check:
            self.x_max_spin.setEnabled(state == Qt.CheckState.Checked.value)
        elif sender == self.y_min_check:
            self.y_min_spin.setEnabled(state == Qt.CheckState.Checked.value)
        elif sender == self.y_max_check:
            self.y_max_spin.setEnabled(state == Qt.CheckState.Checked.value)
        elif sender == self.z_min_check:
            self.z_min_spin.setEnabled(state == Qt.CheckState.Checked.value)
        elif sender == self.z_max_check:
            self.z_max_spin.setEnabled(state == Qt.CheckState.Checked.value)
        # Hierarchical FFD bounds
        elif sender == self.h_x_min_check:
            self.h_x_min_spin.setEnabled(state == Qt.CheckState.Checked.value)
        elif sender == self.h_x_max_check:
            self.h_x_max_spin.setEnabled(state == Qt.CheckState.Checked.value)
        elif sender == self.h_y_min_check:
            self.h_y_min_spin.setEnabled(state == Qt.CheckState.Checked.value)
        elif sender == self.h_y_max_check:
            self.h_y_max_spin.setEnabled(state == Qt.CheckState.Checked.value)
        elif sender == self.h_z_min_check:
            self.h_z_min_spin.setEnabled(state == Qt.CheckState.Checked.value)
        elif sender == self.h_z_max_check:
            self.h_z_max_spin.setEnabled(state == Qt.CheckState.Checked.value)
        
        self.ffd_parameters_changed.emit()
    
    @pyqtSlot()
    def _on_params_changed(self):
        """Handle parameter value changes."""
        self.ffd_parameters_changed.emit()
    
    @pyqtSlot(int)
    def _on_hierarchical_changed(self, state):
        """Handle hierarchical FFD checkbox state changes."""
        enabled = state == Qt.CheckState.Checked.value
        self.hierarchy_levels_spin.setEnabled(enabled)
        self.ffd_parameters_changed.emit()
