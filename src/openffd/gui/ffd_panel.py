"""FFD panel for OpenFFD GUI.

This module provides the UI panel for FFD box configuration.
"""

import logging
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QGroupBox, QCheckBox, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

# Configure logging
logger = logging.getLogger(__name__)


class FFDPanel(QWidget):
    """Panel for FFD box configuration."""
    
    # Signal emitted when FFD parameters are changed
    ffd_parameters_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the FFD panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.mesh_min_coords = None
        self.mesh_max_coords = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Control dimensions section
        dim_group = QGroupBox("Control Point Dimensions")
        dim_layout = QFormLayout()
        
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
        
        # Set bounds without triggering signals
        self.x_min_spin.blockSignals(True)
        self.x_max_spin.blockSignals(True)
        self.y_min_spin.blockSignals(True)
        self.y_max_spin.blockSignals(True)
        self.z_min_spin.blockSignals(True)
        self.z_max_spin.blockSignals(True)
        
        # Update spin box ranges and values
        self.x_min_spin.setValue(min_coords[0])
        self.x_max_spin.setValue(max_coords[0])
        self.y_min_spin.setValue(min_coords[1])
        self.y_max_spin.setValue(max_coords[1])
        self.z_min_spin.setValue(min_coords[2])
        self.z_max_spin.setValue(max_coords[2])
        
        # Unblock signals
        self.x_min_spin.blockSignals(False)
        self.x_max_spin.blockSignals(False)
        self.y_min_spin.blockSignals(False)
        self.y_max_spin.blockSignals(False)
        self.z_min_spin.blockSignals(False)
        self.z_max_spin.blockSignals(False)
        
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
