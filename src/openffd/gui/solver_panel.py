"""Solver panel for OpenFFD GUI.

This module provides the UI panel for solver configuration and integration.
"""

import os
import logging
from typing import Optional, List, Tuple, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QCheckBox, QLineEdit, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QMessageBox, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

from openffd.gui.utils import show_error_dialog

# Configure logging
logger = logging.getLogger(__name__)


class SolverPanel(QWidget):
    """Panel for solver configuration and integration."""
    
    # Signal emitted when solver configuration is changed
    solver_config_changed = pyqtSignal()
    
    # Signal emitted when a simulation is started
    simulation_started = pyqtSignal()
    
    # Signal emitted when a simulation is completed
    simulation_completed = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, parent=None):
        """Initialize the solver panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.openfoam_path = None
        self.case_directory = None
        self.current_solver_interface = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Solver selection
        solver_group = QGroupBox("Solver Selection")
        solver_layout = QFormLayout()
        
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["sonicFoam", "sonicFoamAdjoint"])
        self.solver_combo.currentIndexChanged.connect(self._on_solver_changed)
        
        solver_layout.addRow("Solver:", self.solver_combo)
        
        self.adjoint_option = QCheckBox("Use adjoint solver for optimization")
        self.adjoint_option.setToolTip("Enable adjoint-based optimization for faster convergence")
        self.adjoint_option.setChecked(self.solver_combo.currentText() == "sonicFoamAdjoint")
        self.adjoint_option.stateChanged.connect(self._on_adjoint_option_changed)
        
        solver_layout.addRow("", self.adjoint_option)
        
        solver_group.setLayout(solver_layout)
        layout.addWidget(solver_group)
        
        # OpenFOAM settings
        openfoam_group = QGroupBox("OpenFOAM Settings")
        openfoam_layout = QVBoxLayout()
        
        # OpenFOAM path
        path_layout = QHBoxLayout()
        self.path_label = QLabel("Path to OpenFOAM:")
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("OpenFOAM installation directory")
        self.path_browse = QPushButton("Browse...")
        self.path_browse.clicked.connect(self._browse_openfoam_path)
        
        path_layout.addWidget(self.path_label)
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(self.path_browse)
        
        openfoam_layout.addLayout(path_layout)
        
        # Case directory
        case_layout = QHBoxLayout()
        self.case_label = QLabel("Case directory:")
        self.case_edit = QLineEdit()
        self.case_edit.setPlaceholderText("OpenFOAM case directory")
        self.case_browse = QPushButton("Browse...")
        self.case_browse.clicked.connect(self._browse_case_directory)
        
        case_layout.addWidget(self.case_label)
        case_layout.addWidget(self.case_edit, 1)
        case_layout.addWidget(self.case_browse)
        
        openfoam_layout.addLayout(case_layout)
        
        openfoam_group.setLayout(openfoam_layout)
        layout.addWidget(openfoam_group)
        
        # Create a tab widget for different solver settings
        self.solver_tabs = QTabWidget()
        
        # sonicFoam tab
        self.sonic_tab = QWidget()
        sonic_layout = QVBoxLayout(self.sonic_tab)
        
        # Simulation parameters
        sim_group = QGroupBox("Simulation Parameters")
        sim_form = QFormLayout()
        
        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setRange(0.01, 1000.0)
        self.end_time_spin.setValue(0.1)
        self.end_time_spin.setDecimals(5)
        self.end_time_spin.setSingleStep(0.01)
        
        self.delta_t_spin = QDoubleSpinBox()
        self.delta_t_spin.setRange(0.000001, 1.0)
        self.delta_t_spin.setValue(0.0001)
        self.delta_t_spin.setDecimals(7)
        self.delta_t_spin.setSingleStep(0.0001)
        
        self.write_interval_spin = QSpinBox()
        self.write_interval_spin.setRange(1, 1000)
        self.write_interval_spin.setValue(100)
        
        sim_form.addRow("End time:", self.end_time_spin)
        sim_form.addRow("Time step (Δt):", self.delta_t_spin)
        sim_form.addRow("Write interval:", self.write_interval_spin)
        
        sim_group.setLayout(sim_form)
        sonic_layout.addWidget(sim_group)
        
        # Boundary conditions
        bc_group = QGroupBox("Boundary Conditions")
        bc_form = QFormLayout()
        
        self.bc_combo = QComboBox()
        self.bc_combo.addItems(["inlet", "outlet", "wall", "symmetry"])
        
        self.field_combo = QComboBox()
        self.field_combo.addItems(["U", "p", "T", "rho"])
        
        bc_form.addRow("Boundary:", self.bc_combo)
        bc_form.addRow("Field:", self.field_combo)
        
        bc_group.setLayout(bc_form)
        sonic_layout.addWidget(bc_group)
        
        # Add to tabs
        self.solver_tabs.addTab(self.sonic_tab, "sonicFoam")
        
        # sonicFoamAdjoint tab
        self.adjoint_tab = QWidget()
        adjoint_layout = QVBoxLayout(self.adjoint_tab)
        
        # Objective function group
        obj_group = QGroupBox("Objective Function")
        obj_form = QFormLayout()
        
        self.obj_combo = QComboBox()
        self.obj_combo.addItems(["drag", "lift", "moment", "pressure_uniformity"])
        
        obj_form.addRow("Objective:", self.obj_combo)
        
        # Add sensitivity options
        self.sensitivity_check = QCheckBox("Calculate surface sensitivities")
        self.sensitivity_check.setChecked(True)
        obj_form.addRow("", self.sensitivity_check)
        
        obj_group.setLayout(obj_form)
        adjoint_layout.addWidget(obj_group)
        
        # Optimization settings
        opt_group = QGroupBox("Optimization Settings")
        opt_form = QFormLayout()
        
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 1000)
        self.max_iter_spin.setValue(10)
        
        self.conv_tol_spin = QDoubleSpinBox()
        self.conv_tol_spin.setRange(1e-10, 1e-1)
        self.conv_tol_spin.setValue(1e-5)
        self.conv_tol_spin.setDecimals(10)
        
        opt_form.addRow("Max iterations:", self.max_iter_spin)
        opt_form.addRow("Convergence tolerance:", self.conv_tol_spin)
        
        opt_group.setLayout(opt_form)
        adjoint_layout.addWidget(opt_group)
        
        # Add to tabs
        self.solver_tabs.addTab(self.adjoint_tab, "sonicFoamAdjoint")
        
        layout.addWidget(self.solver_tabs)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self._on_run_simulation)
        
        self.optimize_button = QPushButton("Run Optimization")
        self.optimize_button.clicked.connect(self._on_run_optimization)
        
        action_layout.addWidget(self.run_button)
        action_layout.addWidget(self.optimize_button)
        
        layout.addLayout(action_layout)
        
        # Console output
        console_group = QGroupBox("Console Output")
        console_layout = QVBoxLayout()
        
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setMinimumHeight(100)
        
        console_layout.addWidget(self.console_text)
        console_group.setLayout(console_layout)
        
        layout.addWidget(console_group)
        
        # Set initial state
        self._on_solver_changed(0)
        self._sync_solver_gui_state()
    
    @pyqtSlot()
    def _browse_openfoam_path(self):
        """Browse for OpenFOAM installation directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select OpenFOAM Installation Directory",
            os.path.expanduser("~")
        )
        
        if dir_path:
            self.openfoam_path = dir_path
            self.path_edit.setText(dir_path)
            self.solver_config_changed.emit()
    
    @pyqtSlot()
    def _browse_case_directory(self):
        """Browse for OpenFOAM case directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select OpenFOAM Case Directory",
            os.path.expanduser("~")
        )
        
        if dir_path:
            self.case_directory = dir_path
            self.case_edit.setText(dir_path)
            self.solver_config_changed.emit()
    
    @pyqtSlot(int)
    def _on_solver_changed(self, index):
        """Handle solver selection changes."""
        solver_name = self.solver_combo.currentText()
        
        # Sync adjoint option with solver
        self.adjoint_option.blockSignals(True)
        self.adjoint_option.setChecked(solver_name == "sonicFoamAdjoint")
        self.adjoint_option.blockSignals(False)
        
        # Switch to appropriate tab
        self.solver_tabs.setCurrentIndex(index)
        
        # Update button states
        self.optimize_button.setEnabled(solver_name == "sonicFoamAdjoint")
        
        self.solver_config_changed.emit()
    
    @pyqtSlot(int)
    def _on_adjoint_option_changed(self, state):
        """Handle adjoint option changes."""
        use_adjoint = state == Qt.CheckState.Checked.value
        
        # Update solver selection to match
        self.solver_combo.blockSignals(True)
        self.solver_combo.setCurrentText("sonicFoamAdjoint" if use_adjoint else "sonicFoam")
        self.solver_combo.blockSignals(False)
        
        # Set the appropriate tab
        self.solver_tabs.setCurrentIndex(1 if use_adjoint else 0)
        
        # Update button states
        self.optimize_button.setEnabled(use_adjoint)
        
        self._sync_solver_gui_state()
        
        self.solver_config_changed.emit()
    
    def _sync_solver_gui_state(self):
        """Sync GUI state based on current solver selection."""
        use_adjoint = self.adjoint_option.isChecked()
        
        # Update tab visibility
        self.solver_tabs.setTabEnabled(0, not use_adjoint)
        self.solver_tabs.setTabEnabled(1, use_adjoint)
    
    @pyqtSlot()
    def _on_run_simulation(self):
        """Handle run simulation button click."""
        # Check for required fields
        if not self._validate_input():
            return
        
        try:
            # Update console
            self.console_text.append("Starting simulation...")
            self.console_text.append(f"Solver: {self.solver_combo.currentText()}")
            self.console_text.append(f"Case directory: {self.case_directory}")
            
            # Get the appropriate solver interface
            solver_type = self.solver_combo.currentText()
            
            if solver_type == "sonicFoam":
                self._create_sonic_foam_interface()
            else:
                self._create_sonic_adjoint_interface()
            
            if self.current_solver_interface is None:
                raise RuntimeError("Failed to create solver interface")
            
            # Collect solver parameters
            params = self._collect_solver_parameters()
            
            # Mock simulation run (in a real implementation, this would run in a thread)
            self.console_text.append("Setting up solver parameters...")
            self.console_text.append("Running simulation...")
            
            # Signal that simulation has started
            self.simulation_started.emit()
            
            # In a real implementation, the simulation would run here
            # and we'd update the console with progress.
            
            # Mock completion
            self.console_text.append("Simulation completed successfully.")
            
            # Signal completion
            self.simulation_completed.emit(True, "Simulation completed successfully.")
            
        except Exception as e:
            logger.error(f"Error running simulation: {str(e)}")
            show_error_dialog("Simulation Error", str(e))
            self.console_text.append(f"ERROR: {str(e)}")
            self.simulation_completed.emit(False, str(e))
    
    @pyqtSlot()
    def _on_run_optimization(self):
        """Handle run optimization button click."""
        # Check for required fields
        if not self._validate_input() or not self.adjoint_option.isChecked():
            return
        
        try:
            # Update console
            self.console_text.append("Starting optimization...")
            self.console_text.append(f"Solver: {self.solver_combo.currentText()}")
            self.console_text.append(f"Case directory: {self.case_directory}")
            self.console_text.append(f"Objective function: {self.obj_combo.currentText()}")
            
            # Create adjoint solver interface
            self._create_sonic_adjoint_interface()
            
            if self.current_solver_interface is None:
                raise RuntimeError("Failed to create solver interface")
            
            # Collect solver parameters
            params = self._collect_solver_parameters()
            params.update({
                "objective": self.obj_combo.currentText(),
                "max_iterations": self.max_iter_spin.value(),
                "convergence_tolerance": self.conv_tol_spin.value(),
                "calculate_sensitivities": self.sensitivity_check.isChecked()
            })
            
            # Mock optimization run (in a real implementation, this would run in a thread)
            self.console_text.append("Setting up optimization parameters...")
            self.console_text.append("Running forward simulation...")
            self.console_text.append("Running adjoint simulation...")
            self.console_text.append("Calculating sensitivities...")
            
            # Signal that simulation has started
            self.simulation_started.emit()
            
            # In a real implementation, the optimization would run here
            # and we'd update the console with progress.
            
            # Mock completion
            self.console_text.append("Optimization completed successfully.")
            
            # Signal completion
            self.simulation_completed.emit(True, "Optimization completed successfully.")
            
        except Exception as e:
            logger.error(f"Error running optimization: {str(e)}")
            show_error_dialog("Optimization Error", str(e))
            self.console_text.append(f"ERROR: {str(e)}")
            self.simulation_completed.emit(False, str(e))
    
    def _validate_input(self) -> bool:
        """Validate user input.
        
        Returns:
            True if input is valid, False otherwise
        """
        # Check for OpenFOAM path
        if not self.path_edit.text().strip():
            show_error_dialog("Missing OpenFOAM Path", "Please specify the OpenFOAM installation directory.")
            return False
        
        # Check for case directory
        if not self.case_edit.text().strip():
            show_error_dialog("Missing Case Directory", "Please specify the OpenFOAM case directory.")
            return False
        
        return True
    
    def _collect_solver_parameters(self) -> Dict[str, Any]:
        """Collect solver parameters from UI.
        
        Returns:
            Dictionary of solver parameters
        """
        params = {
            "openfoam_path": self.path_edit.text().strip(),
            "case_directory": self.case_edit.text().strip(),
            "solver_type": self.solver_combo.currentText(),
            "end_time": self.end_time_spin.value(),
            "delta_t": self.delta_t_spin.value(),
            "write_interval": self.write_interval_spin.value()
        }
        
        return params
    
    def _create_sonic_foam_interface(self):
        """Create a SonicFoamInterface instance."""
        try:
            from openffd.solvers.openfoam.sonic_foam_interface import SonicFoamInterface
            
            self.current_solver_interface = SonicFoamInterface(
                openfoam_path=self.path_edit.text().strip(),
                case_dir=self.case_edit.text().strip()
            )
            
            logger.info("Created SonicFoamInterface")
        except Exception as e:
            logger.error(f"Error creating SonicFoamInterface: {str(e)}")
            self.current_solver_interface = None
            raise
    
    def _create_sonic_adjoint_interface(self):
        """Create a SonicAdjointInterface instance."""
        try:
            from openffd.solvers.openfoam.sonic_adjoint import SonicAdjointInterface
            
            self.current_solver_interface = SonicAdjointInterface(
                openfoam_path=self.path_edit.text().strip(),
                case_dir=self.case_edit.text().strip()
            )
            
            logger.info("Created SonicAdjointInterface")
        except Exception as e:
            logger.error(f"Error creating SonicAdjointInterface: {str(e)}")
            self.current_solver_interface = None
            raise
