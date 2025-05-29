"""Deformation visualization for OpenFFD GUI.

This module provides components to visualize mesh deformations and sensitivities.
"""

import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

import pyvista as pv
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

from openffd.visualization.ffd_viz import visualize_ffd
from openffd.gui.utils import show_error_dialog

# Configure logging
logger = logging.getLogger(__name__)


class DeformationWidget(QWidget):
    """Widget for visualizing mesh deformations and sensitivities."""
    
    # Signal emitted when deformation parameters change
    deformation_changed = pyqtSignal(np.ndarray, float)
    
    def __init__(self, parent=None):
        """Initialize the deformation visualization widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.sensitivity_data = None
        self.control_points = None
        self.control_dims = None
        self.scale_factor = 1.0
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Deformation controls
        deform_group = QGroupBox("Deformation Controls")
        deform_layout = QVBoxLayout()
        
        # Scale slider
        scale_layout = QHBoxLayout()
        scale_label = QLabel("Scale Factor:")
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setMinimum(-100)
        self.scale_slider.setMaximum(100)
        self.scale_slider.setValue(10)
        self.scale_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.scale_slider.setTickInterval(10)
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(-10.0, 10.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setValue(1.0)
        self.scale_spin.valueChanged.connect(self._on_spin_changed)
        
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.scale_slider, 1)
        scale_layout.addWidget(self.scale_spin)
        
        deform_layout.addLayout(scale_layout)
        
        # View options
        view_layout = QFormLayout()
        
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Original + Deformed", "Original Only", "Deformed Only", "Sensitivities"])
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "jet", "rainbow", "coolwarm", "RdBu"])
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        
        view_layout.addRow("Display mode:", self.view_combo)
        view_layout.addRow("Color scheme:", self.colormap_combo)
        
        deform_layout.addLayout(view_layout)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset Deformation")
        self.reset_btn.clicked.connect(self._on_reset)
        
        self.apply_btn = QPushButton("Apply Deformation")
        self.apply_btn.clicked.connect(self._on_apply_deformation)
        
        buttons_layout.addWidget(self.reset_btn)
        buttons_layout.addWidget(self.apply_btn)
        
        deform_layout.addLayout(buttons_layout)
        
        deform_group.setLayout(deform_layout)
        layout.addWidget(deform_group)
        
        # Sensitivity stats
        stats_group = QGroupBox("Sensitivity Statistics")
        stats_layout = QFormLayout()
        
        self.min_sens_label = QLabel("N/A")
        self.max_sens_label = QLabel("N/A")
        self.avg_sens_label = QLabel("N/A")
        
        stats_layout.addRow("Min Sensitivity:", self.min_sens_label)
        stats_layout.addRow("Max Sensitivity:", self.max_sens_label)
        stats_layout.addRow("Average Sensitivity:", self.avg_sens_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Set initial enabled state
        self._update_ui_state()
    
    def set_sensitivity_data(self, sensitivity_data: np.ndarray, control_points: np.ndarray, control_dims: Tuple[int, int, int]):
        """Set the sensitivity data for visualization.
        
        Args:
            sensitivity_data: Sensitivity values for each control point
            control_points: Control point coordinates
            control_dims: Control point dimensions (nx, ny, nz)
        """
        self.sensitivity_data = sensitivity_data
        self.control_points = control_points
        self.control_dims = control_dims
        
        # Update sensitivity statistics
        if sensitivity_data is not None:
            self.min_sens_label.setText(f"{np.min(sensitivity_data):.6f}")
            self.max_sens_label.setText(f"{np.max(sensitivity_data):.6f}")
            self.avg_sens_label.setText(f"{np.mean(sensitivity_data):.6f}")
        
        self._update_ui_state()
    
    def get_deformed_control_points(self) -> Optional[np.ndarray]:
        """Get the deformed control points based on sensitivities and scale factor.
        
        Returns:
            Numpy array of deformed control point coordinates or None if no sensitivity data
        """
        if self.sensitivity_data is None or self.control_points is None:
            return None
        
        # Apply deformation based on sensitivities and scale factor
        deformation = self.sensitivity_data.reshape(-1, 3) * self.scale_factor
        return self.control_points + deformation
    
    def _update_ui_state(self):
        """Update the UI state based on available data."""
        has_data = self.sensitivity_data is not None
        
        self.scale_slider.setEnabled(has_data)
        self.scale_spin.setEnabled(has_data)
        self.view_combo.setEnabled(has_data)
        self.colormap_combo.setEnabled(has_data)
        self.reset_btn.setEnabled(has_data)
        self.apply_btn.setEnabled(has_data)
    
    @pyqtSlot(int)
    def _on_scale_changed(self, value):
        """Handle scale slider value changes."""
        # Convert slider value to scale factor (-10.0 to 10.0)
        scale = value / 10.0
        
        # Block signals to avoid recursion
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(scale)
        self.scale_spin.blockSignals(False)
        
        self.scale_factor = scale
        
        # Emit signal that deformation has changed
        if self.sensitivity_data is not None:
            self.deformation_changed.emit(self.get_deformed_control_points(), scale)
    
    @pyqtSlot(float)
    def _on_spin_changed(self, value):
        """Handle scale spinbox value changes."""
        # Convert scale factor to slider value (-100 to 100)
        slider_value = int(value * 10.0)
        
        # Block signals to avoid recursion
        self.scale_slider.blockSignals(True)
        self.scale_slider.setValue(slider_value)
        self.scale_slider.blockSignals(False)
        
        self.scale_factor = value
        
        # Emit signal that deformation has changed
        if self.sensitivity_data is not None:
            self.deformation_changed.emit(self.get_deformed_control_points(), value)
    
    @pyqtSlot(int)
    def _on_view_changed(self, index):
        """Handle view mode changes."""
        # View mode logic would be handled by the visualization component
        pass
    
    @pyqtSlot(str)
    def _on_colormap_changed(self, colormap):
        """Handle colormap changes."""
        # Colormap logic would be handled by the visualization component
        pass
    
    @pyqtSlot()
    def _on_reset(self):
        """Reset the deformation to zero."""
        self.scale_spin.setValue(0.0)
    
    @pyqtSlot()
    def _on_apply_deformation(self):
        """Apply the current deformation to the FFD control box."""
        if self.sensitivity_data is None or self.control_points is None:
            return
        
        try:
            # Get the deformed control points
            deformed_points = self.get_deformed_control_points()
            
            # In a real implementation, we would update the FFD control points
            # and propagate the changes to the mesh
            
            logger.info(f"Applied deformation with scale factor: {self.scale_factor}")
        except Exception as e:
            logger.error(f"Error applying deformation: {str(e)}")
            show_error_dialog("Deformation Error", str(e))
