"""
Abstract Base Classes for CFD Solver Interface

This module defines the abstract interfaces and data structures that all CFD solvers
must implement to integrate with the OpenFFD optimization framework.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)

class SolverStatus(Enum):
    """CFD solver execution status."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    CONVERGED = "converged"
    DIVERGED = "diverged"
    FAILED = "failed"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class ObjectiveFunction(Enum):
    """Available objective functions for optimization."""
    DRAG_COEFFICIENT = "drag_coefficient"
    LIFT_COEFFICIENT = "lift_coefficient"
    PRESSURE_LOSS = "pressure_loss"
    HEAT_TRANSFER = "heat_transfer"
    MASS_FLOW_RATE = "mass_flow_rate"
    MOMENT_COEFFICIENT = "moment_coefficient"
    EFFICIENCY = "efficiency"
    POWER_CONSUMPTION = "power_consumption"
    TEMPERATURE_UNIFORMITY = "temperature_uniformity"
    VELOCITY_UNIFORMITY = "velocity_uniformity"
    CUSTOM = "custom"

@dataclass
class ConvergenceData:
    """CFD solver convergence monitoring data."""
    iteration: int
    time_step: float
    residuals: Dict[str, float]
    objective_values: Dict[str, float]
    wall_time: float
    cpu_time: float
    memory_usage: float
    is_converged: bool
    convergence_criteria: Dict[str, float]
    
    def __post_init__(self):
        if self.residuals is None:
            self.residuals = {}
        if self.objective_values is None:
            self.objective_values = {}
        if self.convergence_criteria is None:
            self.convergence_criteria = {}

@dataclass
class CFDResults:
    """Base class for CFD simulation results."""
    status: SolverStatus
    convergence_data: List[ConvergenceData]
    execution_time: float
    iterations: int
    final_residuals: Dict[str, float]
    objective_values: Dict[str, float]
    field_data: Dict[str, np.ndarray] = field(default_factory=dict)
    surface_data: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    forces: Dict[str, np.ndarray] = field(default_factory=dict)
    moments: Dict[str, np.ndarray] = field(default_factory=dict)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_converged(self) -> bool:
        """Check if simulation converged."""
        return self.status == SolverStatus.CONVERGED
    
    @property
    def is_successful(self) -> bool:
        """Check if simulation completed successfully."""
        return self.status in [SolverStatus.CONVERGED, SolverStatus.COMPLETED]
    
    def get_objective_value(self, objective: ObjectiveFunction) -> Optional[float]:
        """Get specific objective function value."""
        return self.objective_values.get(objective.value)
    
    def get_field(self, field_name: str) -> Optional[np.ndarray]:
        """Get field data by name."""
        return self.field_data.get(field_name)
    
    def get_surface_data(self, patch_name: str, field_name: str) -> Optional[np.ndarray]:
        """Get surface field data."""
        if patch_name in self.surface_data:
            return self.surface_data[patch_name].get(field_name)
        return None

@dataclass
class CFDConfig:
    """Base CFD solver configuration."""
    case_directory: Path
    solver_executable: str
    mesh_file: Optional[Path] = None
    time_step: float = 1e-6
    end_time: float = 1.0
    write_interval: int = 100
    max_iterations: int = 1000
    convergence_tolerance: Dict[str, float] = field(default_factory=dict)
    parallel_execution: bool = False
    num_processors: int = 1
    decomposition_method: str = "scotch"
    
    # Physical properties
    fluid_properties: Dict[str, float] = field(default_factory=dict)
    boundary_conditions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    initial_conditions: Dict[str, float] = field(default_factory=dict)
    
    # Numerical settings
    numerical_schemes: Dict[str, str] = field(default_factory=dict)
    solution_controls: Dict[str, Any] = field(default_factory=dict)
    
    # Output settings
    output_fields: List[str] = field(default_factory=list)
    monitor_surfaces: List[str] = field(default_factory=list)
    force_patches: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default values."""
        if not self.convergence_tolerance:
            self.convergence_tolerance = {
                'p': 1e-6,
                'U': 1e-6,
                'k': 1e-6,
                'epsilon': 1e-6,
                'omega': 1e-6,
                'T': 1e-6
            }
        
        if not self.fluid_properties:
            self.fluid_properties = {
                'nu': 1.5e-5,  # kinematic viscosity (m²/s) for air at 20°C
                'rho': 1.225,  # density (kg/m³) for air at 20°C
                'Cp': 1005.0,  # specific heat capacity (J/kg·K)
                'Pr': 0.7      # Prandtl number
            }
        
        if not self.output_fields:
            self.output_fields = ['p', 'U', 'k', 'epsilon', 'nut']

class SolverInterface(ABC):
    """Abstract interface for CFD solvers."""
    
    @abstractmethod
    def setup_case(self, config: CFDConfig) -> bool:
        """Setup CFD case directory and files."""
        pass
    
    @abstractmethod
    def run_simulation(self, config: CFDConfig) -> CFDResults:
        """Execute CFD simulation."""
        pass
    
    @abstractmethod
    def extract_results(self, case_dir: Path) -> CFDResults:
        """Extract results from completed simulation."""
        pass
    
    @abstractmethod
    def compute_sensitivities(self, config: CFDConfig, 
                            objective: ObjectiveFunction) -> Dict[str, np.ndarray]:
        """Compute sensitivity derivatives."""
        pass
    
    @abstractmethod
    def cleanup(self, case_dir: Path) -> None:
        """Clean up temporary files and directories."""
        pass

class CFDSolver(SolverInterface):
    """Base CFD solver implementation with common functionality."""
    
    def __init__(self, name: str, version: str = None):
        """Initialize CFD solver.
        
        Args:
            name: Solver name
            version: Solver version
        """
        self.name = name
        self.version = version
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._status = SolverStatus.INITIALIZED
        self._current_case = None
        self._monitoring_data = []
        
    @property
    def status(self) -> SolverStatus:
        """Get current solver status."""
        return self._status
    
    @status.setter
    def status(self, value: SolverStatus):
        """Set solver status."""
        self._status = value
        self.logger.info(f"Solver status changed to: {value.value}")
    
    def validate_config(self, config: CFDConfig) -> List[str]:
        """Validate CFD configuration.
        
        Args:
            config: CFD configuration
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check required paths
        if not config.case_directory:
            errors.append("Case directory not specified")
        
        if not config.solver_executable:
            errors.append("Solver executable not specified")
        
        # Check numerical parameters
        if config.time_step <= 0:
            errors.append("Time step must be positive")
        
        if config.end_time <= 0:
            errors.append("End time must be positive")
        
        if config.max_iterations <= 0:
            errors.append("Maximum iterations must be positive")
        
        # Check parallel settings
        if config.parallel_execution and config.num_processors < 2:
            errors.append("Number of processors must be >= 2 for parallel execution")
        
        return errors
    
    def check_convergence(self, residuals: Dict[str, float], 
                         criteria: Dict[str, float]) -> bool:
        """Check convergence based on residuals.
        
        Args:
            residuals: Current residual values
            criteria: Convergence criteria
            
        Returns:
            True if converged
        """
        for field, criterion in criteria.items():
            if field in residuals:
                if residuals[field] > criterion:
                    return False
        return True
    
    def monitor_convergence(self, iteration: int, residuals: Dict[str, float],
                          objectives: Dict[str, float] = None) -> ConvergenceData:
        """Monitor convergence progress.
        
        Args:
            iteration: Current iteration
            residuals: Residual values
            objectives: Objective function values
            
        Returns:
            Convergence data
        """
        if objectives is None:
            objectives = {}
        
        convergence_data = ConvergenceData(
            iteration=iteration,
            time_step=0.0,  # To be filled by specific solver
            residuals=residuals.copy(),
            objective_values=objectives.copy(),
            wall_time=time.time(),
            cpu_time=0.0,   # To be filled by specific solver
            memory_usage=0.0,  # To be filled by specific solver
            is_converged=self.check_convergence(residuals, {}),  # To be implemented
            convergence_criteria={}  # To be filled by specific solver
        )
        
        self._monitoring_data.append(convergence_data)
        return convergence_data
    
    def get_convergence_history(self) -> List[ConvergenceData]:
        """Get convergence monitoring history."""
        return self._monitoring_data.copy()
    
    def reset_monitoring(self):
        """Reset convergence monitoring data."""
        self._monitoring_data.clear()
    
    def estimate_runtime(self, config: CFDConfig) -> float:
        """Estimate simulation runtime.
        
        Args:
            config: CFD configuration
            
        Returns:
            Estimated runtime in seconds
        """
        # Basic estimation based on mesh size and iterations
        # This should be overridden by specific solver implementations
        base_time = config.max_iterations * 0.1  # 0.1 seconds per iteration
        
        if config.parallel_execution:
            base_time /= min(config.num_processors, 8)  # Assume max 8x speedup
        
        return base_time
    
    def get_memory_estimate(self, config: CFDConfig) -> float:
        """Estimate memory requirements.
        
        Args:
            config: CFD configuration
            
        Returns:
            Estimated memory in MB
        """
        # Basic estimation - should be overridden by specific implementations
        base_memory = 100.0  # Base memory in MB
        
        if config.parallel_execution:
            base_memory *= config.num_processors
        
        return base_memory
    
    def prepare_case_directory(self, config: CFDConfig) -> bool:
        """Prepare case directory structure.
        
        Args:
            config: CFD configuration
            
        Returns:
            True if successful
        """
        try:
            config.case_directory.mkdir(parents=True, exist_ok=True)
            
            # Create standard OpenFOAM directory structure
            (config.case_directory / "0").mkdir(exist_ok=True)
            (config.case_directory / "constant").mkdir(exist_ok=True)
            (config.case_directory / "system").mkdir(exist_ok=True)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to prepare case directory: {e}")
            return False
    
    def save_results(self, results: CFDResults, output_path: Path):
        """Save CFD results to file.
        
        Args:
            results: CFD results
            output_path: Output file path
        """
        try:
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def load_results(self, input_path: Path) -> Optional[CFDResults]:
        """Load CFD results from file.
        
        Args:
            input_path: Input file path
            
        Returns:
            CFD results or None if failed
        """
        try:
            import pickle
            with open(input_path, 'rb') as f:
                results = pickle.load(f)
            self.logger.info(f"Results loaded from {input_path}")
            return results
        except Exception as e:
            self.logger.error(f"Failed to load results: {e}")
            return None
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get solver information.
        
        Returns:
            Solver information dictionary
        """
        return {
            'name': self.name,
            'version': self.version,
            'status': self.status.value,
            'current_case': str(self._current_case) if self._current_case else None,
            'monitoring_points': len(self._monitoring_data)
        }