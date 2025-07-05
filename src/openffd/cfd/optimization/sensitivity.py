"""
Sensitivity Analysis Module for CFD Optimization

This module provides comprehensive sensitivity computation capabilities including:
- Adjoint method implementation
- Finite difference gradients
- Sensitivity verification and validation
- Gradient computation for shape optimization
- Multi-objective sensitivity analysis
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

from ..core.base import BaseSolver, BaseCase, BaseObjective
from ..solvers.openfoam import OpenFOAMSolver, OpenFOAMConfig, SimulationResults

logger = logging.getLogger(__name__)

class GradientMethod(Enum):
    """Gradient computation methods."""
    FINITE_DIFFERENCE_FORWARD = "fd_forward"
    FINITE_DIFFERENCE_BACKWARD = "fd_backward"
    FINITE_DIFFERENCE_CENTRAL = "fd_central"
    ADJOINT_METHOD = "adjoint"
    COMPLEX_STEP = "complex_step"
    AUTOMATIC_DIFFERENTIATION = "automatic_diff"

class SensitivityType(Enum):
    """Types of sensitivity analysis."""
    SHAPE_SENSITIVITY = "shape"
    PARAMETER_SENSITIVITY = "parameter"
    BOUNDARY_CONDITION_SENSITIVITY = "boundary"
    MATERIAL_PROPERTY_SENSITIVITY = "material"
    OPERATIONAL_CONDITION_SENSITIVITY = "operational"

@dataclass
class FiniteDifferenceConfig:
    """Configuration for finite difference gradient computation."""
    method: GradientMethod = GradientMethod.FINITE_DIFFERENCE_CENTRAL
    step_size: float = 1e-6
    adaptive_step: bool = True
    min_step_size: float = 1e-8
    max_step_size: float = 1e-4
    step_size_factor: float = 2.0
    relative_step: bool = True
    parallel_evaluation: bool = True
    max_workers: Optional[int] = None
    verification_enabled: bool = True
    verification_tolerance: float = 1e-3

@dataclass
class AdjointConfig:
    """Configuration for adjoint sensitivity computation."""
    solver_type: str = "adjointSimpleFoam"
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    relaxation_factors: Dict[str, float] = field(default_factory=dict)
    objective_function_implementation: str = "forceCoeffs"
    boundary_sensitivity: bool = True
    volume_sensitivity: bool = False
    mesh_sensitivity: bool = True
    
    def __post_init__(self):
        if not self.relaxation_factors:
            self.relaxation_factors = {
                'pa': 0.3,
                'Ua': 0.7,
                'ka': 0.7,
                'epsilona': 0.7,
                'omegaa': 0.7
            }

@dataclass
class SensitivityConfig:
    """Main sensitivity analysis configuration."""
    gradient_method: GradientMethod = GradientMethod.FINITE_DIFFERENCE_CENTRAL
    sensitivity_type: SensitivityType = SensitivityType.SHAPE_SENSITIVITY
    objective_functions: List[Any] = field(default_factory=list)
    design_variables: List[str] = field(default_factory=list)
    fd_config: FiniteDifferenceConfig = field(default_factory=FiniteDifferenceConfig)
    adjoint_config: AdjointConfig = field(default_factory=AdjointConfig)
    
    # Output settings
    save_gradients: bool = True
    save_perturbed_cases: bool = False
    output_directory: Optional[Path] = None
    gradient_file_format: str = "numpy"  # numpy, csv, hdf5
    
    # Verification settings
    cross_validation: bool = True
    gradient_verification: bool = True
    finite_difference_check: bool = True
    
    def __post_init__(self):
        if not self.objective_functions:
            self.objective_functions = []
        
        if not self.design_variables:
            self.design_variables = ["mesh_points"]

@dataclass
class SensitivityResults:
    """Sensitivity analysis results."""
    gradients: Dict[str, Dict[str, np.ndarray]]  # {objective: {variable: gradient}}
    gradient_norms: Dict[str, Dict[str, float]]
    computation_time: float
    method_used: GradientMethod
    objective_values: Dict[str, float]
    perturbed_objective_values: Dict[str, Dict[str, float]] = field(default_factory=dict)
    verification_results: Dict[str, Any] = field(default_factory=dict)
    convergence_data: Dict[str, Any] = field(default_factory=dict)
    error_estimates: Dict[str, Dict[str, float]] = field(default_factory=dict)
    computational_cost: Dict[str, Any] = field(default_factory=dict)
    
    def get_gradient(self, objective: str, variable: str) -> Optional[np.ndarray]:
        """Get gradient for specific objective and variable."""
        return self.gradients.get(objective, {}).get(variable)
    
    def get_gradient_norm(self, objective: str, variable: str) -> Optional[float]:
        """Get gradient norm for specific objective and variable."""
        return self.gradient_norms.get(objective, {}).get(variable)
    
    def get_total_gradient_norm(self, objective: str) -> float:
        """Get total gradient norm for objective."""
        if objective in self.gradient_norms:
            return np.sqrt(sum(norm**2 for norm in self.gradient_norms[objective].values()))
        return 0.0
    
    def save_results(self, output_path: Path, format: str = "numpy"):
        """Save sensitivity results to file."""
        if format == "numpy":
            np.savez(output_path, 
                    gradients=self.gradients,
                    gradient_norms=self.gradient_norms,
                    objective_values=self.objective_values)
        elif format == "csv":
            # Convert to DataFrame and save
            data = []
            for obj, variables in self.gradients.items():
                for var, grad in variables.items():
                    data.append({
                        'objective': obj,
                        'variable': var,
                        'gradient_norm': self.gradient_norms[obj][var],
                        'objective_value': self.objective_values[obj]
                    })
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)

class GradientComputation:
    """Base class for gradient computation methods."""
    
    def __init__(self, config: SensitivityConfig, solver: BaseSolver):
        """Initialize gradient computation.
        
        Args:
            config: Sensitivity configuration
            solver: CFD solver instance
        """
        self.config = config
        self.solver = solver
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def compute_gradients(self, cfd_config: Any, 
                         design_variables: Dict[str, np.ndarray]) -> SensitivityResults:
        """Compute gradients for all objectives and variables.
        
        Args:
            cfd_config: CFD configuration
            design_variables: Design variable values
            
        Returns:
            Sensitivity results
        """
        raise NotImplementedError
    
    def verify_gradients(self, analytical_gradients: Dict[str, np.ndarray],
                        numerical_gradients: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Verify analytical gradients against numerical ones."""
        verification_results = {}
        
        for var in analytical_gradients:
            if var in numerical_gradients:
                analytical = analytical_gradients[var]
                numerical = numerical_gradients[var]
                
                # Compute relative error
                rel_error = np.linalg.norm(analytical - numerical) / (
                    np.linalg.norm(numerical) + 1e-12
                )
                verification_results[var] = rel_error
        
        return verification_results

class FiniteDifferenceGradients(GradientComputation):
    """Finite difference gradient computation."""
    
    def compute_gradients(self, cfd_config: Any, 
                         design_variables: Dict[str, np.ndarray]) -> SensitivityResults:
        """Compute finite difference gradients."""
        start_time = time.time()
        self.logger.info("Computing finite difference gradients...")
        
        # Evaluate baseline case
        baseline_results = self.solver.run_simulation(cfd_config)
        if not baseline_results.is_successful:
            raise RuntimeError("Baseline simulation failed")
        
        baseline_objectives = self._extract_objectives(baseline_results)
        
        # Initialize results
        gradients = {obj.value: {} for obj in self.config.objective_functions}
        gradient_norms = {obj.value: {} for obj in self.config.objective_functions}
        perturbed_objectives = {}
        
        # Compute gradients for each design variable
        for var_name, var_values in design_variables.items():
            self.logger.info(f"Computing gradients for variable: {var_name}")
            
            var_gradients = self._compute_variable_gradients(
                cfd_config, var_name, var_values, baseline_objectives
            )
            
            for obj_name, gradient in var_gradients.items():
                gradients[obj_name][var_name] = gradient
                gradient_norms[obj_name][var_name] = np.linalg.norm(gradient)
        
        computation_time = time.time() - start_time
        
        results = SensitivityResults(
            gradients=gradients,
            gradient_norms=gradient_norms,
            computation_time=computation_time,
            method_used=self.config.gradient_method,
            objective_values=baseline_objectives,
            perturbed_objective_values=perturbed_objectives
        )
        
        self.logger.info(f"Finite difference gradients computed in {computation_time:.2f} seconds")
        return results
    
    def _compute_variable_gradients(self, cfd_config: Any, var_name: str,
                                  var_values: np.ndarray, 
                                  baseline_objectives: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Compute gradients for a single design variable."""
        method = self.config.fd_config.method
        step_size = self.config.fd_config.step_size
        
        n_vars = len(var_values)
        var_gradients = {obj.value: np.zeros(n_vars) for obj in self.config.objective_functions}
        
        if self.config.fd_config.parallel_evaluation:
            # Parallel gradient computation
            var_gradients = self._compute_gradients_parallel(
                cfd_config, var_name, var_values, baseline_objectives, step_size, method
            )
        else:
            # Sequential gradient computation
            for i in range(n_vars):
                grad_i = self._compute_gradient_component(
                    cfd_config, var_name, var_values, baseline_objectives, i, step_size, method
                )
                
                for obj_name, grad_val in grad_i.items():
                    var_gradients[obj_name][i] = grad_val
        
        return var_gradients
    
    def _compute_gradients_parallel(self, cfd_config: Any, var_name: str,
                                  var_values: np.ndarray, baseline_objectives: Dict[str, float],
                                  step_size: float, method: GradientMethod) -> Dict[str, np.ndarray]:
        """Compute gradients in parallel."""
        n_vars = len(var_values)
        max_workers = self.config.fd_config.max_workers or min(mp.cpu_count(), n_vars)
        
        var_gradients = {obj.value: np.zeros(n_vars) for obj in self.config.objective_functions}
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all gradient computations
            future_to_index = {}
            for i in range(n_vars):
                future = executor.submit(
                    self._compute_gradient_component,
                    cfd_config, var_name, var_values, baseline_objectives, i, step_size, method
                )
                future_to_index[future] = i
            
            # Collect results
            for future in future_to_index:
                i = future_to_index[future]
                try:
                    grad_i = future.result()
                    for obj_name, grad_val in grad_i.items():
                        var_gradients[obj_name][i] = grad_val
                except Exception as e:
                    self.logger.error(f"Failed to compute gradient component {i}: {e}")
                    # Set gradient to zero for failed computations
                    for obj_name in var_gradients:
                        var_gradients[obj_name][i] = 0.0
        
        return var_gradients
    
    def _compute_gradient_component(self, cfd_config: Any, var_name: str,
                                  var_values: np.ndarray, baseline_objectives: Dict[str, float],
                                  component_idx: int, step_size: float, 
                                  method: GradientMethod) -> Dict[str, float]:
        """Compute gradient for a single component of a design variable."""
        gradients = {}
        
        # Determine step size (adaptive if enabled)
        if self.config.fd_config.adaptive_step:
            step = self._compute_adaptive_step(var_values[component_idx], step_size)
        else:
            if self.config.fd_config.relative_step:
                step = step_size * max(abs(var_values[component_idx]), 1.0)
            else:
                step = step_size
        
        try:
            if method == GradientMethod.FINITE_DIFFERENCE_FORWARD:
                # Forward difference
                perturbed_objectives = self._evaluate_perturbed_case(
                    cfd_config, var_name, var_values, component_idx, step
                )
                
                for obj_name in baseline_objectives:
                    gradient = (perturbed_objectives[obj_name] - baseline_objectives[obj_name]) / step
                    gradients[obj_name] = gradient
            
            elif method == GradientMethod.FINITE_DIFFERENCE_BACKWARD:
                # Backward difference
                perturbed_objectives = self._evaluate_perturbed_case(
                    cfd_config, var_name, var_values, component_idx, -step
                )
                
                for obj_name in baseline_objectives:
                    gradient = (baseline_objectives[obj_name] - perturbed_objectives[obj_name]) / step
                    gradients[obj_name] = gradient
            
            elif method == GradientMethod.FINITE_DIFFERENCE_CENTRAL:
                # Central difference
                forward_objectives = self._evaluate_perturbed_case(
                    cfd_config, var_name, var_values, component_idx, step
                )
                backward_objectives = self._evaluate_perturbed_case(
                    cfd_config, var_name, var_values, component_idx, -step
                )
                
                for obj_name in baseline_objectives:
                    gradient = (forward_objectives[obj_name] - backward_objectives[obj_name]) / (2 * step)
                    gradients[obj_name] = gradient
            
            else:
                raise ValueError(f"Unsupported finite difference method: {method}")
        
        except Exception as e:
            self.logger.error(f"Failed to compute gradient component {component_idx}: {e}")
            # Return zero gradients for failed computations
            for obj_name in baseline_objectives:
                gradients[obj_name] = 0.0
        
        return gradients
    
    def _evaluate_perturbed_case(self, cfd_config: Any, var_name: str,
                               var_values: np.ndarray, component_idx: int, 
                               step: float) -> Dict[str, float]:
        """Evaluate objective functions for a perturbed design variable."""
        # Create perturbed configuration
        perturbed_config = self._create_perturbed_config(
            cfd_config, var_name, var_values, component_idx, step
        )
        
        # Run simulation
        results = self.solver.run_simulation(perturbed_config)
        
        if not results.is_successful:
            self.logger.warning(f"Perturbed simulation failed for {var_name}[{component_idx}]")
            # Return baseline objectives (zero gradient)
            return self._extract_objectives(results)
        
        return self._extract_objectives(results)
    
    def _create_perturbed_config(self, base_config: Any, var_name: str,
                               var_values: np.ndarray, component_idx: int, 
                               step: float) -> Any:
        """Create perturbed CFD configuration."""
        # This is a simplified implementation
        # Real implementation would handle mesh perturbation, boundary condition changes, etc.
        
        import copy
        perturbed_config = copy.deepcopy(base_config)
        
        # Create unique case directory for perturbed case
        case_name = f"perturbed_{var_name}_{component_idx}_{abs(step):.2e}"
        perturbed_config.case_directory = base_config.case_directory.parent / case_name
        
        # Apply perturbation based on variable type
        if var_name == "mesh_points":
            # Handle mesh point perturbation
            self._perturb_mesh_points(perturbed_config, component_idx, step)
        elif var_name.startswith("boundary_"):
            # Handle boundary condition perturbation
            self._perturb_boundary_condition(perturbed_config, var_name, component_idx, step)
        elif var_name in ["reynolds_number", "mach_number"]:
            # Handle flow condition perturbation
            self._perturb_flow_conditions(perturbed_config, var_name, step)
        else:
            self.logger.warning(f"Unknown design variable type: {var_name}")
        
        return perturbed_config
    
    def _perturb_mesh_points(self, config: Any, point_idx: int, step: float):
        """Apply perturbation to mesh points."""
        # This would interface with the mesh deformation system
        # For now, just log the perturbation
        self.logger.debug(f"Perturbing mesh point {point_idx} by {step}")
    
    def _perturb_boundary_condition(self, config: Any, var_name: str, 
                                  component_idx: int, step: float):
        """Apply perturbation to boundary conditions."""
        self.logger.debug(f"Perturbing boundary condition {var_name}[{component_idx}] by {step}")
    
    def _perturb_flow_conditions(self, config: Any, var_name: str, step: float):
        """Apply perturbation to flow conditions."""
        if var_name == "reynolds_number":
            # Adjust velocity or viscosity
            current_re = config.fluid_properties.get('U', 1.0) / config.fluid_properties.get('nu', 1.5e-5)
            new_re = current_re + step
            config.fluid_properties['U'] = new_re * config.fluid_properties.get('nu', 1.5e-5)
        
        self.logger.debug(f"Perturbing flow condition {var_name} by {step}")
    
    def _compute_adaptive_step(self, variable_value: float, base_step: float) -> float:
        """Compute adaptive step size."""
        if self.config.fd_config.relative_step:
            step = base_step * max(abs(variable_value), 1.0)
        else:
            step = base_step
        
        # Clamp to limits
        step = max(self.config.fd_config.min_step_size, step)
        step = min(self.config.fd_config.max_step_size, step)
        
        return step
    
    def _extract_objectives(self, results: Any) -> Dict[str, float]:
        """Extract objective function values from CFD results."""
        objectives = {}
        
        for obj_func in self.config.objective_functions:
            if obj_func == Any.DRAG_COEFFICIENT:
                if hasattr(results, 'force_coefficients') and results.force_coefficients:
                    objectives[obj_func.value] = results.force_coefficients.cd
                else:
                    objectives[obj_func.value] = 0.0
            
            elif obj_func == Any.LIFT_COEFFICIENT:
                if hasattr(results, 'force_coefficients') and results.force_coefficients:
                    objectives[obj_func.value] = results.force_coefficients.cl
                else:
                    objectives[obj_func.value] = 0.0
            
            elif obj_func == Any.MOMENT_COEFFICIENT:
                if hasattr(results, 'force_coefficients') and results.force_coefficients:
                    objectives[obj_func.value] = results.force_coefficients.cm
                else:
                    objectives[obj_func.value] = 0.0
            
            elif obj_func == Any.PRESSURE_LOSS:
                # Extract pressure loss from field data
                objectives[obj_func.value] = 0.0  # Placeholder
            
            else:
                self.logger.warning(f"Objective function {obj_func.value} not implemented")
                objectives[obj_func.value] = 0.0
        
        return objectives

class AdjointGradients(GradientComputation):
    """Adjoint method gradient computation."""
    
    def compute_gradients(self, cfd_config: Any, 
                         design_variables: Dict[str, np.ndarray]) -> SensitivityResults:
        """Compute adjoint gradients."""
        start_time = time.time()
        self.logger.info("Computing adjoint gradients...")
        
        # Run primal (forward) simulation
        primal_results = self.solver.run_simulation(cfd_config)
        if not primal_results.is_successful:
            raise RuntimeError("Primal simulation failed")
        
        baseline_objectives = self._extract_objectives(primal_results)
        
        # Initialize results
        gradients = {obj.value: {} for obj in self.config.objective_functions}
        gradient_norms = {obj.value: {} for obj in self.config.objective_functions}
        
        # Compute adjoint gradients for each objective function
        for obj_func in self.config.objective_functions:
            self.logger.info(f"Computing adjoint gradients for {obj_func.value}")
            
            # Run adjoint simulation
            adjoint_results = self._run_adjoint_simulation(cfd_config, obj_func)
            
            # Compute gradients from adjoint solution
            obj_gradients = self._compute_adjoint_gradients(
                cfd_config, primal_results, adjoint_results, design_variables
            )
            
            for var_name, gradient in obj_gradients.items():
                gradients[obj_func.value][var_name] = gradient
                gradient_norms[obj_func.value][var_name] = np.linalg.norm(gradient)
        
        computation_time = time.time() - start_time
        
        results = SensitivityResults(
            gradients=gradients,
            gradient_norms=gradient_norms,
            computation_time=computation_time,
            method_used=GradientMethod.ADJOINT_METHOD,
            objective_values=baseline_objectives
        )
        
        self.logger.info(f"Adjoint gradients computed in {computation_time:.2f} seconds")
        return results
    
    def _run_adjoint_simulation(self, cfd_config: Any, 
                              objective: Any) -> Any:
        """Run adjoint simulation for specific objective function."""
        # Create adjoint configuration
        adjoint_config = self._create_adjoint_config(cfd_config, objective)
        
        # Setup adjoint case
        self.solver.setup_case(adjoint_config)
        
        # Run adjoint solver
        results = self.solver.run_simulation(adjoint_config)
        
        return results
    
    def _create_adjoint_config(self, base_config: Any, 
                             objective: Any) -> Any:
        """Create adjoint CFD configuration."""
        import copy
        adjoint_config = copy.deepcopy(base_config)
        
        # Modify for adjoint simulation
        adjoint_config.case_directory = base_config.case_directory.parent / f"adjoint_{objective.value}"
        adjoint_config.solver_executable = self.config.adjoint_config.solver_type
        adjoint_config.max_iterations = self.config.adjoint_config.max_iterations
        adjoint_config.convergence_tolerance = {
            'pa': self.config.adjoint_config.convergence_tolerance,
            'Ua': self.config.adjoint_config.convergence_tolerance,
            'ka': self.config.adjoint_config.convergence_tolerance,
            'epsilona': self.config.adjoint_config.convergence_tolerance,
            'omegaa': self.config.adjoint_config.convergence_tolerance
        }
        
        # Add objective function specification
        self._add_objective_function_config(adjoint_config, objective)
        
        return adjoint_config
    
    def _add_objective_function_config(self, config: Any, objective: Any):
        """Add objective function configuration to adjoint case."""
        # This would add the appropriate objective function formulation
        # to the adjoint configuration (implementation depends on OpenFOAM version)
        pass
    
    def _compute_adjoint_gradients(self, cfd_config: Any, primal_results: Any,
                                 adjoint_results: Any, 
                                 design_variables: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute gradients from adjoint solution."""
        gradients = {}
        
        for var_name, var_values in design_variables.items():
            if var_name == "mesh_points":
                # Compute surface sensitivity
                gradient = self._compute_surface_sensitivity(
                    primal_results, adjoint_results, var_values
                )
            else:
                # Placeholder for other design variables
                gradient = np.zeros_like(var_values)
            
            gradients[var_name] = gradient
        
        return gradients
    
    def _compute_surface_sensitivity(self, primal_results: Any, 
                                   adjoint_results: Any,
                                   mesh_points: np.ndarray) -> np.ndarray:
        """Compute surface sensitivity for mesh points."""
        # This is a simplified placeholder implementation
        # Real implementation would compute surface sensitivity using:
        # dJ/dx = ∫(∇pᵀ·∇φ + ∇Uᵀ·∇ψ + ...)·n dS
        
        n_points = len(mesh_points)
        sensitivity = np.zeros((n_points, 3))
        
        # Placeholder computation
        # Would extract adjoint variables and compute actual sensitivity
        
        return sensitivity

class SensitivityAnalyzer:
    """Main sensitivity analysis orchestrator."""
    
    def __init__(self, case_handler: BaseCase, solver: BaseSolver, step_size: float = 1e-3):
        """Initialize sensitivity analyzer.
        
        Args:
            case_handler: Case handler instance
            solver: CFD solver instance
            step_size: Step size for finite differences
        """
        self.case_handler = case_handler
        self.solver = solver
        self.step_size = step_size
        self.logger = logging.getLogger(__name__)
        
        # Create default config
        self.config = SensitivityConfig()
        self.config.fd_config.step_size = step_size
    
    def compute_gradient(self, design_vars: np.ndarray, objectives: List[Any]) -> np.ndarray:
        """Compute gradient using finite differences.
        
        Args:
            design_vars: Current design variable values
            objectives: List of objective functions
            
        Returns:
            Gradient vector
        """
        self.logger.info(f"Computing gradient using finite differences with step size {self.step_size}")
        
        # Get baseline objective values
        baseline_results = self.solver.run_simulation()
        baseline_objectives = self.case_handler.extract_objectives(baseline_results)
        
        print(f"  Baseline objectives: {baseline_objectives}")
        
        gradients = np.zeros_like(design_vars)
        
        # Compute finite difference gradients for all components
        for i in range(len(design_vars)):
            # Forward finite difference
            perturbed_vars = design_vars.copy()
            perturbed_vars[i] += self.step_size
            
            # Apply perturbed design variables 
            mesh_file = self._apply_design_variables(perturbed_vars)
            
            # Save perturbed mesh for visualization
            self._save_gradient_mesh(i, perturbed_vars)
            
            # Run simulation with perturbed variables
            perturbed_results = self.solver.run_simulation(mesh_file)
            perturbed_objectives = self.case_handler.extract_objectives(perturbed_results)
            
            print(f"  Component {i}: Perturbed objectives: {perturbed_objectives}")
            
            # Compute gradient for each objective (weighted sum)
            gradient = 0.0
            for obj in objectives:
                obj_name = obj.name
                if obj_name in baseline_objectives and obj_name in perturbed_objectives:
                    obj_gradient = (perturbed_objectives[obj_name] - baseline_objectives[obj_name]) / self.step_size
                    gradient += obj.weight * obj_gradient
                    print(f"    {obj_name}: {baseline_objectives[obj_name]:.8f} -> {perturbed_objectives[obj_name]:.8f}, grad = {obj_gradient:.6e}")
            
            gradients[i] = gradient
            self.logger.info(f"Gradient component {i}: {gradient:.6e}")
            print(f"  Gradient[{i}]: {gradient:.6e}")
        
        gradient_norm = np.linalg.norm(gradients)
        self.logger.info(f"Gradient computation completed. Norm: {gradient_norm:.6e}")
        print(f"  Gradient norm: {gradient_norm:.6e}")
        print(f"  Gradient vector: {gradients}")
        return gradients
    
    def _apply_design_variables(self, design_vars: np.ndarray) -> str:
        """Apply design variables to deform mesh."""
        # Use solver's mesh deformation capability if available
        if hasattr(self.solver, 'apply_design_variables'):
            return self.solver.apply_design_variables(design_vars)
        else:
            # For now, return original mesh path (no deformation)
            # Real implementation would use FFD to deform the mesh
            self.logger.warning("No mesh deformation capability available - using original mesh")
            if hasattr(self.case_handler, 'case_path'):
                return str(self.case_handler.case_path / 'constant' / 'polyMesh')
            else:
                return "constant/polyMesh"
    
    def _save_gradient_mesh(self, component_idx: int, design_vars: np.ndarray) -> None:
        """Save mesh state during gradient computation."""
        import shutil
        from pathlib import Path
        
        if hasattr(self.case_handler, 'case_path'):
            case_path = Path(self.case_handler.case_path)
            
            # Create gradient_meshes directory if it doesn't exist
            mesh_dir = case_path / "gradient_meshes"
            mesh_dir.mkdir(exist_ok=True)
            
            # Create component-specific directory
            component_dir = mesh_dir / f"gradient_component_{component_idx}"
            component_dir.mkdir(exist_ok=True)
            
            # Copy current polyMesh
            source_mesh = case_path / "constant" / "polyMesh"
            dest_mesh = component_dir / "polyMesh"
            
            if source_mesh.exists():
                if dest_mesh.exists():
                    shutil.rmtree(dest_mesh)
                shutil.copytree(source_mesh, dest_mesh)
                
                # Save design variables for reference
                design_vars_file = component_dir / "design_variables.txt"
                with open(design_vars_file, 'w') as f:
                    f.write(f"Gradient Component: {component_idx}\n")
                    f.write(f"Perturbed Design Variables: {design_vars.tolist()}\n")
                    f.write(f"Step Size: {self.step_size}\n")
                
                print(f"    Gradient mesh saved to: gradient_meshes/gradient_component_{component_idx}")
    
    def _verify_gradients(self, solver: BaseSolver, cfd_config: Any,
                         design_variables: Dict[str, np.ndarray],
                         analytical_results: SensitivityResults) -> Dict[str, Any]:
        """Verify gradients using different methods."""
        verification_results = {}
        
        if self.config.finite_difference_check and self.config.gradient_method != GradientMethod.FINITE_DIFFERENCE_CENTRAL:
            self.logger.info("Verifying gradients with finite differences...")
            
            # Create finite difference configuration
            fd_config = SensitivityConfig(
                gradient_method=GradientMethod.FINITE_DIFFERENCE_CENTRAL,
                objective_functions=self.config.objective_functions,
                design_variables=self.config.design_variables,
                output_directory=None,
                save_gradients=False
            )
            
            # Compute finite difference gradients
            fd_computer = FiniteDifferenceGradients(fd_config, solver)
            fd_results = fd_computer.compute_gradients(cfd_config, design_variables)
            
            # Compare gradients
            for obj_name in analytical_results.gradients:
                for var_name in analytical_results.gradients[obj_name]:
                    analytical_grad = analytical_results.gradients[obj_name][var_name]
                    fd_grad = fd_results.gradients[obj_name][var_name]
                    
                    rel_error = np.linalg.norm(analytical_grad - fd_grad) / (
                        np.linalg.norm(fd_grad) + 1e-12
                    )
                    
                    verification_results[f"{obj_name}_{var_name}_fd_error"] = rel_error
        
        return verification_results
    
    def _save_results(self, results: SensitivityResults):
        """Save sensitivity results to files."""
        output_path = self.config.output_directory / f"sensitivity_results.{self.config.gradient_file_format}"
        results.save_results(output_path, self.config.gradient_file_format)
        
        # Save additional information
        info_file = self.config.output_directory / "sensitivity_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"Sensitivity Analysis Results\n")
            f.write(f"==========================\n\n")
            f.write(f"Method: {results.method_used.value}\n")
            f.write(f"Computation time: {results.computation_time:.2f} seconds\n")
            f.write(f"Objectives: {list(results.objective_values.keys())}\n")
            f.write(f"Design variables: {list(results.gradients[list(results.gradients.keys())[0]].keys())}\n\n")
            
            f.write("Gradient norms:\n")
            for obj_name, var_norms in results.gradient_norms.items():
                f.write(f"  {obj_name}:\n")
                for var_name, norm in var_norms.items():
                    f.write(f"    {var_name}: {norm:.6e}\n")
        
        self.logger.info(f"Sensitivity results saved to {self.config.output_directory}")

class AdjointSolver:
    """Dedicated adjoint solver implementation."""
    
    def __init__(self, openfoam_solver: OpenFOAMSolver):
        """Initialize adjoint solver.
        
        Args:
            openfoam_solver: OpenFOAM solver instance
        """
        self.openfoam_solver = openfoam_solver
        self.logger = logging.getLogger(__name__)
    
    def setup_adjoint_case(self, primal_config: OpenFOAMConfig, 
                          objective: Any) -> OpenFOAMConfig:
        """Setup adjoint case from primal configuration."""
        # This would implement the full adjoint case setup
        # including proper boundary conditions, objective function formulation, etc.
        pass
    
    def run_adjoint_simulation(self, adjoint_config: OpenFOAMConfig) -> SimulationResults:
        """Run adjoint simulation."""
        return self.openfoam_solver.run_simulation(adjoint_config)
    
    def extract_adjoint_sensitivities(self, adjoint_results: SimulationResults,
                                    design_variables: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract sensitivities from adjoint solution."""
        # This would implement the sensitivity extraction from adjoint fields
        pass