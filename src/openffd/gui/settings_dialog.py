"""Settings dialog for OpenFFD GUI.

This module provides a dialog for configuring global settings like parallel processing.
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QDialogButtonBox, QGroupBox, QCheckBox, QSpinBox, QComboBox, QTabWidget
)
from PyQt6.QtCore import Qt

from openffd.utils.parallel import ParallelConfig

# Configure logging
logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """Dialog for configuring OpenFFD settings."""
    
    def __init__(self, parallel_config: Optional[ParallelConfig] = None, parent=None):
        """Initialize the settings dialog.
        
        Args:
            parallel_config: Current parallel processing configuration
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("OpenFFD Settings")
        self.setMinimumWidth(500)
        
        # Initialize with current config or default
        self.parallel_config = parallel_config or ParallelConfig()
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Parallel processing tab
        parallel_tab = QWidget()
        parallel_layout = QVBoxLayout(parallel_tab)
        
        # Parallel processing group
        parallel_group = QGroupBox("Parallel Processing")
        parallel_form = QFormLayout()
        
        # Enable parallel processing
        self.enable_parallel = QCheckBox("Enable parallel processing")
        self.enable_parallel.setChecked(self.parallel_config.enabled)
        self.enable_parallel.stateChanged.connect(self._on_parallel_toggled)
        
        # Method selection
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Process-based", "Thread-based"])
        current_method = "Process-based" if self.parallel_config.method == "process" else "Thread-based"
        self.method_combo.setCurrentText(current_method)
        
        # Worker count
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setValue(self.parallel_config.workers or 4)
        self.workers_spin.setSpecialValueText("Auto")
        
        # Chunk size
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(0, 1000000)
        self.chunk_spin.setValue(self.parallel_config.chunk_size or 0)
        self.chunk_spin.setSpecialValueText("Auto")
        
        # Threshold for parallelization
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1000, 1000000)
        self.threshold_spin.setValue(self.parallel_config.threshold)
        self.threshold_spin.setSingleStep(1000)
        
        # Advanced options
        self.enable_viz_parallel = QCheckBox("Use parallelization for visualization")
        self.enable_viz_parallel.setChecked(self.parallel_config.viz_parallel)
        
        # Add to form layout
        parallel_form.addRow("", self.enable_parallel)
        parallel_form.addRow("Parallelization method:", self.method_combo)
        parallel_form.addRow("Number of workers:", self.workers_spin)
        parallel_form.addRow("Chunk size:", self.chunk_spin)
        parallel_form.addRow("Parallelization threshold:", self.threshold_spin)
        parallel_form.addRow("", self.enable_viz_parallel)
        
        # Add form to group
        parallel_group.setLayout(parallel_form)
        
        # Add to tab layout
        parallel_layout.addWidget(parallel_group)
        
        # Update enabled state
        self._on_parallel_toggled(self.enable_parallel.checkState())
        
        # Add tab to tab widget
        tabs.addTab(parallel_tab, "Parallel Processing")
        
        # Add tab widget to main layout
        layout.addWidget(tabs)
        
        # Add button box
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _on_parallel_toggled(self, state):
        """Handle parallel processing toggle."""
        enabled = state == Qt.CheckState.Checked.value
        self.method_combo.setEnabled(enabled)
        self.workers_spin.setEnabled(enabled)
        self.chunk_spin.setEnabled(enabled)
        self.threshold_spin.setEnabled(enabled)
        self.enable_viz_parallel.setEnabled(enabled)
    
    def get_parallel_config(self) -> ParallelConfig:
        """Get the parallel configuration from the dialog.
        
        Returns:
            ParallelConfig object with updated settings
        """
        # Create a new config with the dialog settings
        return ParallelConfig(
            enabled=self.enable_parallel.isChecked(),
            method="process" if self.method_combo.currentText() == "Process-based" else "thread",
            workers=None if self.workers_spin.value() == 1 else self.workers_spin.value(),
            chunk_size=None if self.chunk_spin.value() == 0 else self.chunk_spin.value(),
            threshold=self.threshold_spin.value(),
            viz_parallel=self.enable_viz_parallel.isChecked()
        )
