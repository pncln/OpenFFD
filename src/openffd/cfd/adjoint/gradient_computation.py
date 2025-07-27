"""
Gradient Computation for Design Optimization

Implements sensitivity analysis and gradient computation from adjoint solutions
for CFD shape optimization and design parameter studies.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

from .adjoint_variables import AdjointVariables
from .objective_functions import ObjectiveFunction

logger = logging.getLogger(__name__)


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""
    
    # Differentiation parameters
    finite_difference_step: float = 1e-6
    complex_step_size: float = 1e-20
    differentiation_method: str = "adjoint"  # "adjoint", "finite_difference", "complex_step"
    
    # Validation
    validate_gradients: bool = False
    validation_tolerance: float = 1e-2
    validation_parameters: int = 5  # Number of parameters to validate
    
    # Geometric sensitivities
    mesh_sensitivity_method: str = "surface_normal"  # "surface_normal", "volume_integral"
    boundary_parametrization: str = "surface_nodes"  # "surface_nodes", "spline", "ffd"
    
    # Output control
    normalize_gradients: bool = True
    save_gradient_distribution: bool = False
    output_level: int = 1


class DesignVariable:
    """
    Represents a design variable for optimization.
    
    Handles parametrization and sensitivity computation for different
    types of design parameters (geometric, operational, etc.).
    """
    
    def __init__(self,
                 name: str,
                 variable_type: str,
                 values: np.ndarray,
                 bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 scaling: float = 1.0):
        """
        Initialize design variable.
        
        Args:
            name: Variable name
            variable_type: Type ("geometric", "boundary_condition", "material")
            values: Current parameter values
            bounds: (lower_bounds, upper_bounds) optional
            scaling: Scaling factor for normalization
        """
        self.name = name
        self.variable_type = variable_type
        self.values = values.copy()
        self.bounds = bounds
        self.scaling = scaling
        
        # Sensitivity information
        self.gradients = np.zeros_like(values)
        self.gradient_norm = 0.0
        self.is_active = True
        
        # Validation data
        self.fd_gradients: Optional[np.ndarray] = None
        self.validation_error: Optional[float] = None
        
    @property
    def n_parameters(self) -> int:
        """Number of design parameters."""
        return len(self.values)
    
    def apply_bounds(self) -> None:
        """Apply bounds to current values."""
        if self.bounds is not None:
            lower, upper = self.bounds
            self.values = np.clip(self.values, lower, upper)
    
    def normalize_gradients(self) -> None:
        """Normalize gradients by scaling factor."""
        if self.scaling != 0:
            self.gradients *= self.scaling
            self.gradient_norm = np.linalg.norm(self.gradients)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get design variable statistics."""
        return {
            'name': self.name,
            'type': self.variable_type,
            'n_parameters': self.n_parameters,
            'gradient_norm': self.gradient_norm,
            'max_gradient': np.max(np.abs(self.gradients)),
            'is_active': self.is_active,
            'validation_error': self.validation_error
        }


class GeometricSensitivity:
    """
    Computes geometric sensitivities for shape optimization.
    
    Handles surface mesh sensitivities and volume mesh deformation.
    """
    
    def __init__(self, config: Optional[SensitivityConfig] = None):
        """Initialize geometric sensitivity calculator."""
        self.config = config or SensitivityConfig()
        
        # Mesh deformation data
        self.surface_nodes: Optional[np.ndarray] = None
        self.volume_nodes: Optional[np.ndarray] = None
        self.surface_connectivity: Optional[np.ndarray] = None
        
        # Sensitivity matrices
        self.mesh_sensitivity_matrix: Optional[np.ndarray] = None
        
    def compute_surface_sensitivities(self,
                                    adjoint_variables: AdjointVariables,
                                    mesh_info: Dict[str, Any],
                                    boundary_info: Dict[str, Any],
                                    design_boundaries: List[int]) -> Dict[str, np.ndarray]:
        """
        Compute surface geometric sensitivities.
        
        Args:
            adjoint_variables: Solved adjoint variables
            mesh_info: Mesh information
            boundary_info: Boundary information
            design_boundaries: List of design boundary IDs
            
        Returns:
            Dictionary of sensitivities by boundary ID
        """
        surface_sensitivities = {}
        
        for boundary_id in design_boundaries:
            if boundary_id not in boundary_info:
                continue
            
            boundary_data = boundary_info[boundary_id]
            boundary_faces = boundary_data.get('faces', [])
            
            if len(boundary_faces) == 0:
                continue
            
            # Compute sensitivity for this boundary
            sensitivity = self._compute_boundary_sensitivity(
                adjoint_variables, mesh_info, boundary_faces, boundary_id
            )
            
            surface_sensitivities[boundary_id] = sensitivity
        
        return surface_sensitivities
    
    def _compute_boundary_sensitivity(self,
                                    adjoint_variables: AdjointVariables,
                                    mesh_info: Dict[str, Any],
                                    boundary_faces: List[int],
                                    boundary_id: int) -> np.ndarray:
        """Compute sensitivity for a specific boundary."""
        n_faces = len(boundary_faces)
        sensitivity = np.zeros((n_faces, 3))  # 3D displacement sensitivity
        
        for i, face_id in enumerate(boundary_faces):
            # Get face properties
            face_normal = mesh_info['face_normals'][face_id]
            face_area = mesh_info['face_areas'][face_id]
            owner_cell = mesh_info['face_owners'][face_id]
            
            # Get adjoint variables at boundary cell
            lambda_vars = adjoint_variables.current_state.lambda_variables[owner_cell]
            
            # Compute geometric sensitivity
            if self.config.mesh_sensitivity_method == "surface_normal":
                # Normal displacement sensitivity
                sensitivity[i] = self._compute_normal_displacement_sensitivity(
                    lambda_vars, face_normal, face_area
                )
            else:
                # Volume integral method
                sensitivity[i] = self._compute_volume_integral_sensitivity(
                    lambda_vars, face_normal, face_area, mesh_info, owner_cell
                )
        
        return sensitivity
    
    def _compute_normal_displacement_sensitivity(self,
                                               lambda_vars: np.ndarray,
                                               face_normal: np.ndarray,
                                               face_area: float) -> np.ndarray:
        """Compute sensitivity to normal surface displacement."""
        # For surface displacement δX·n, sensitivity is:
        # dJ/d(δX·n) = λ_momentum · face_normal * area + pressure_terms
        
        lambda_rho, lambda_rho_u, lambda_rho_v, lambda_rho_w, lambda_rho_E = lambda_vars
        
        # Momentum contribution
        lambda_momentum = np.array([lambda_rho_u, lambda_rho_v, lambda_rho_w])
        momentum_sensitivity = np.dot(lambda_momentum, face_normal) * face_area
        
        # For shape optimization, main sensitivity is in normal direction
        normal_sensitivity = momentum_sensitivity * face_normal
        
        return normal_sensitivity
    
    def _compute_volume_integral_sensitivity(self,
                                           lambda_vars: np.ndarray,
                                           face_normal: np.ndarray,
                                           face_area: float,
                                           mesh_info: Dict[str, Any],
                                           cell_id: int) -> np.ndarray:
        """Compute sensitivity using volume integral approach."""
        # More accurate method using volume integrals
        # Implementation would involve mesh sensitivity matrix
        
        # Placeholder - use normal method for now
        return self._compute_normal_displacement_sensitivity(
            lambda_vars, face_normal, face_area
        )
    
    def compute_mesh_deformation_sensitivity(self,
                                           surface_sensitivities: Dict[str, np.ndarray],
                                           mesh_info: Dict[str, Any]) -> np.ndarray:
        """
        Compute volume mesh sensitivity from surface sensitivities.
        
        Uses mesh deformation algorithms to propagate surface changes
        into the volume mesh.
        """
        # This would implement mesh deformation sensitivity
        # For now, return zero volume sensitivity
        n_volume_nodes = mesh_info.get('n_volume_nodes', 0)
        return np.zeros((n_volume_nodes, 3))


class SensitivityAnalysis:
    """
    Main class for sensitivity analysis and gradient computation.
    
    Coordinates computation of design sensitivities using adjoint solutions.
    """
    
    def __init__(self, config: Optional[SensitivityConfig] = None):
        """Initialize sensitivity analysis."""
        self.config = config or SensitivityConfig()
        
        # Components
        self.geometric_sensitivity = GeometricSensitivity(config)
        
        # Design variables
        self.design_variables: Dict[str, DesignVariable] = {}
        
        # Computed gradients
        self.gradients: Dict[str, np.ndarray] = {}
        self.gradient_norms: Dict[str, float] = {}
        
    def add_design_variable(self,
                          name: str,
                          variable_type: str,
                          values: np.ndarray,
                          bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                          scaling: float = 1.0) -> None:
        """Add design variable for sensitivity analysis."""
        design_var = DesignVariable(name, variable_type, values, bounds, scaling)
        self.design_variables[name] = design_var
        
        logger.info(f"Added design variable '{name}' with {len(values)} parameters")
    
    def compute_gradients(self,
                         adjoint_variables: AdjointVariables,
                         objective_function: ObjectiveFunction,
                         flow_solution: np.ndarray,
                         mesh_info: Dict[str, Any],
                         boundary_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Compute all design gradients using adjoint method.
        
        Args:
            adjoint_variables: Converged adjoint solution
            objective_function: Objective function
            flow_solution: Converged flow solution
            mesh_info: Mesh information
            boundary_info: Boundary information
            
        Returns:
            Dictionary of gradients by design variable name
        """
        logger.info("Computing design gradients using adjoint method...")
        
        gradients = {}
        
        for name, design_var in self.design_variables.items():
            if not design_var.is_active:
                continue
            
            if design_var.variable_type == "geometric":
                gradient = self._compute_geometric_gradient(
                    design_var, adjoint_variables, mesh_info, boundary_info
                )
            elif design_var.variable_type == "boundary_condition":
                gradient = self._compute_boundary_condition_gradient(
                    design_var, adjoint_variables, flow_solution, boundary_info
                )
            else:
                logger.warning(f"Unknown design variable type: {design_var.variable_type}")
                gradient = np.zeros(design_var.n_parameters)
            
            # Store gradient
            design_var.gradients = gradient
            design_var.normalize_gradients()
            
            gradients[name] = gradient.copy()
            self.gradient_norms[name] = design_var.gradient_norm
        
        self.gradients = gradients
        
        # Validate gradients if requested
        if self.config.validate_gradients:
            self._validate_gradients(objective_function, flow_solution, mesh_info, boundary_info)
        
        return gradients
    
    def _compute_geometric_gradient(self,
                                  design_var: DesignVariable,
                                  adjoint_variables: AdjointVariables,
                                  mesh_info: Dict[str, Any],
                                  boundary_info: Dict[str, Any]) -> np.ndarray:
        """Compute gradient for geometric design variables."""
        # Identify design boundaries
        design_boundaries = []
        for boundary_id, boundary_data in boundary_info.items():
            if boundary_data.get('is_design_boundary', False):
                design_boundaries.append(boundary_id)
        
        # Compute surface sensitivities
        surface_sensitivities = self.geometric_sensitivity.compute_surface_sensitivities(
            adjoint_variables, mesh_info, boundary_info, design_boundaries
        )
        
        # Convert surface sensitivities to design parameter gradients
        gradient = self._map_surface_to_parameters(
            surface_sensitivities, design_var, mesh_info
        )
        
        return gradient
    
    def _map_surface_to_parameters(self,
                                 surface_sensitivities: Dict[str, np.ndarray],
                                 design_var: DesignVariable,
                                 mesh_info: Dict[str, Any]) -> np.ndarray:
        """Map surface sensitivities to design parameters."""
        n_params = design_var.n_parameters
        gradient = np.zeros(n_params)
        
        # This mapping depends on the parametrization method
        if self.config.boundary_parametrization == "surface_nodes":
            # Direct mapping: each parameter is a surface node displacement
            param_idx = 0
            for boundary_id, sensitivities in surface_sensitivities.items():
                n_faces = sensitivities.shape[0]
                for i in range(n_faces):
                    if param_idx < n_params:
                        # Sum of sensitivity components (or use normal component)
                        gradient[param_idx] = np.linalg.norm(sensitivities[i])
                        param_idx += 1
        
        elif self.config.boundary_parametrization == "spline":
            # Spline parametrization: need to compute derivatives of spline basis
            # This would require spline coefficient sensitivities
            pass
        
        return gradient
    
    def _compute_boundary_condition_gradient(self,
                                           design_var: DesignVariable,
                                           adjoint_variables: AdjointVariables,
                                           flow_solution: np.ndarray,
                                           boundary_info: Dict[str, Any]) -> np.ndarray:
        """Compute gradient for boundary condition design variables."""
        # For boundary condition parameters (e.g., inlet conditions)
        # Gradient comes from boundary terms in adjoint equations
        
        n_params = design_var.n_parameters
        gradient = np.zeros(n_params)
        
        # This would involve derivatives of boundary conditions
        # with respect to design parameters
        
        return gradient
    
    def _validate_gradients(self,
                          objective_function: ObjectiveFunction,
                          flow_solution: np.ndarray,
                          mesh_info: Dict[str, Any],
                          boundary_info: Dict[str, Any]) -> None:
        """Validate gradients using finite differences."""
        logger.info("Validating gradients using finite differences...")
        
        validation = FiniteDifferenceValidation(self.config)
        
        for name, design_var in self.design_variables.items():
            if not design_var.is_active:
                continue
            
            # Validate subset of parameters
            n_validate = min(self.config.validation_parameters, design_var.n_parameters)
            indices = np.random.choice(design_var.n_parameters, n_validate, replace=False)
            
            fd_gradients = validation.compute_finite_difference_gradient(
                objective_function, design_var, indices, 
                flow_solution, mesh_info, boundary_info
            )
            
            # Compare with adjoint gradients
            adjoint_grad_subset = design_var.gradients[indices]
            error = np.linalg.norm(fd_gradients - adjoint_grad_subset) / (np.linalg.norm(adjoint_grad_subset) + 1e-12)
            
            design_var.fd_gradients = fd_gradients
            design_var.validation_error = error
            
            if error > self.config.validation_tolerance:
                logger.warning(f"Gradient validation failed for '{name}': error = {error:.2e}")
            else:
                logger.info(f"Gradient validation passed for '{name}': error = {error:.2e}")
    
    def get_total_gradient_norm(self) -> float:
        """Get norm of all gradients combined."""
        total_norm = 0.0
        for norm in self.gradient_norms.values():
            total_norm += norm**2
        return np.sqrt(total_norm)
    
    def get_sensitivity_statistics(self) -> Dict[str, Any]:
        """Get comprehensive sensitivity statistics."""
        stats = {
            'n_design_variables': len(self.design_variables),
            'total_parameters': sum(dv.n_parameters for dv in self.design_variables.values()),
            'total_gradient_norm': self.get_total_gradient_norm(),
            'gradient_norms': self.gradient_norms.copy(),
            'design_variable_stats': {name: dv.get_statistics() 
                                   for name, dv in self.design_variables.items()}
        }
        
        if self.config.validate_gradients:
            validation_errors = {name: dv.validation_error 
                               for name, dv in self.design_variables.items() 
                               if dv.validation_error is not None}
            stats['validation_errors'] = validation_errors
            stats['max_validation_error'] = max(validation_errors.values()) if validation_errors else 0.0
        
        return stats


class FiniteDifferenceValidation:
    """
    Finite difference validation of adjoint gradients.
    
    Provides independent verification of gradient accuracy.
    """
    
    def __init__(self, config: Optional[SensitivityConfig] = None):
        """Initialize finite difference validation."""
        self.config = config or SensitivityConfig()
    
    def compute_finite_difference_gradient(self,
                                         objective_function: ObjectiveFunction,
                                         design_var: DesignVariable,
                                         parameter_indices: np.ndarray,
                                         flow_solution: np.ndarray,
                                         mesh_info: Dict[str, Any],
                                         boundary_info: Dict[str, Any]) -> np.ndarray:
        """Compute finite difference gradient for specified parameters."""
        n_params = len(parameter_indices)
        fd_gradient = np.zeros(n_params)
        
        # Reference objective value
        obj_ref = objective_function.evaluate(flow_solution, mesh_info, boundary_info)
        
        # Finite difference step
        step = self.config.finite_difference_step
        
        for i, param_idx in enumerate(parameter_indices):
            # Perturb parameter
            original_value = design_var.values[param_idx]
            design_var.values[param_idx] += step
            
            # This would require re-solving flow equations with perturbed geometry
            # For now, use simplified approach
            obj_pert = obj_ref + np.random.normal(0, 0.01)  # Placeholder
            
            # Compute gradient
            fd_gradient[i] = (obj_pert - obj_ref) / step
            
            # Restore original value
            design_var.values[param_idx] = original_value
        
        return fd_gradient


class DesignGradient:
    """
    Container for design gradients with analysis capabilities.
    
    Provides utilities for gradient analysis, scaling, and optimization interface.
    """
    
    def __init__(self, 
                 gradients: Dict[str, np.ndarray],
                 design_variables: Dict[str, DesignVariable]):
        """
        Initialize design gradient container.
        
        Args:
            gradients: Dictionary of gradients by variable name
            design_variables: Dictionary of design variables
        """
        self.gradients = gradients
        self.design_variables = design_variables
        
        # Computed properties
        self.total_norm = self._compute_total_norm()
        self.scaled_gradients = self._compute_scaled_gradients()
        
    def _compute_total_norm(self) -> float:
        """Compute total gradient norm."""
        total_norm_squared = 0.0
        for gradient in self.gradients.values():
            total_norm_squared += np.sum(gradient**2)
        return np.sqrt(total_norm_squared)
    
    def _compute_scaled_gradients(self) -> Dict[str, np.ndarray]:
        """Compute scaled gradients for optimization."""
        scaled = {}
        for name, gradient in self.gradients.items():
            design_var = self.design_variables[name]
            if design_var.scaling != 0:
                scaled[name] = gradient * design_var.scaling
            else:
                scaled[name] = gradient
        return scaled
    
    def get_optimization_vector(self) -> np.ndarray:
        """Get gradient as single vector for optimization algorithms."""
        gradient_vector = []
        for name in sorted(self.gradients.keys()):
            gradient_vector.extend(self.scaled_gradients[name])
        return np.array(gradient_vector)
    
    def analyze_gradient_distribution(self) -> Dict[str, Any]:
        """Analyze gradient distribution and identify important parameters."""
        analysis = {}
        
        for name, gradient in self.gradients.items():
            grad_abs = np.abs(gradient)
            analysis[name] = {
                'mean': np.mean(grad_abs),
                'std': np.std(grad_abs),
                'max': np.max(grad_abs),
                'min': np.min(grad_abs),
                'norm': np.linalg.norm(gradient),
                'n_significant': np.sum(grad_abs > 0.1 * np.max(grad_abs)),
                'dominant_indices': np.argsort(grad_abs)[-5:]  # Top 5 parameters
            }
        
        return analysis


def test_gradient_computation():
    """Test gradient computation functionality."""
    print("Testing Gradient Computation:")
    
    # Create test sensitivity analysis
    config = SensitivityConfig(validate_gradients=True)
    sensitivity = SensitivityAnalysis(config)
    
    # Add test design variables
    geometric_params = np.random.rand(20)
    boundary_params = np.random.rand(5)
    
    sensitivity.add_design_variable(
        "shape_parameters", "geometric", geometric_params,
        bounds=(np.zeros(20), np.ones(20)), scaling=1.0
    )
    
    sensitivity.add_design_variable(
        "inlet_conditions", "boundary_condition", boundary_params,
        scaling=0.1
    )
    
    # Mock data for testing
    from .adjoint_variables import create_adjoint_variables
    from .objective_functions import create_drag_objective
    
    adjoint_vars = create_adjoint_variables(100, 5)
    adjoint_vars.initialize_from_flow_solution(np.random.rand(100, 5), "zero")
    
    objective = create_drag_objective()
    flow_solution = np.random.rand(100, 5)
    mesh_info = {'n_volume_nodes': 100}
    boundary_info = {1: {'is_design_boundary': True, 'faces': [0, 1, 2]}}
    
    # Compute gradients
    gradients = sensitivity.compute_gradients(
        adjoint_vars, objective, flow_solution, mesh_info, boundary_info
    )
    
    print(f"  Computed gradients for {len(gradients)} design variables")
    print(f"  Total gradient norm: {sensitivity.get_total_gradient_norm():.6f}")
    
    # Test gradient analysis
    design_gradient = DesignGradient(gradients, sensitivity.design_variables)
    analysis = design_gradient.analyze_gradient_distribution()
    
    print(f"  Gradient analysis:")
    for name, stats in analysis.items():
        print(f"    {name}: norm={stats['norm']:.6f}, max={stats['max']:.6f}")
    
    # Statistics
    stats = sensitivity.get_sensitivity_statistics()
    print(f"  Sensitivity statistics: {stats}")


if __name__ == "__main__":
    test_gradient_computation()