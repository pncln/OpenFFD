"""
Adjoint Variable Data Structures

Implements data structures for storing and managing adjoint variables
in the discrete adjoint method for CFD optimization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdjointConfig:
    """Configuration for adjoint computation."""
    
    # Solver settings
    linear_solver: str = "gmres"  # "gmres", "bicgstab", "direct"
    tolerance: float = 1e-8
    max_iterations: int = 1000
    restart_frequency: int = 30  # For GMRES
    
    # Preconditioning
    use_preconditioning: bool = True
    preconditioner_type: str = "ilu"  # "ilu", "jacobi", "block_jacobi"
    
    # Linearization
    frozen_turbulence: bool = True  # Freeze turbulence for simplicity
    linearization_mode: str = "automatic"  # "automatic", "manual", "complex_step"
    perturbation_size: float = 1e-8  # For finite difference jacobian
    
    # Memory management
    store_jacobian: bool = False  # Store full Jacobian matrix
    matrix_free: bool = True  # Use matrix-free methods
    
    # Convergence monitoring
    residual_frequency: int = 10
    save_convergence_history: bool = True
    
    # Output control
    output_level: int = 1  # 0=silent, 1=basic, 2=detailed, 3=debug


@dataclass
class AdjointState:
    """
    Container for adjoint state variables.
    
    Stores adjoint variables corresponding to each conservative variable
    and provides methods for manipulation and analysis.
    """
    
    # Primary adjoint variables [n_cells, n_variables]
    lambda_variables: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Adjoint boundary variables [n_boundary_faces, n_variables] 
    lambda_boundary: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Adjoint gradients [n_cells, n_variables, 3]
    lambda_gradients: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Residual of adjoint equations [n_cells, n_variables]
    adjoint_residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Convergence history
    residual_history: List[float] = field(default_factory=list)
    
    # Solver statistics
    solver_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        if self.lambda_variables.size == 0:
            logger.warning("AdjointState initialized with empty lambda_variables")
    
    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self.lambda_variables.shape[0] if self.lambda_variables.size > 0 else 0
    
    @property
    def n_variables(self) -> int:
        """Number of adjoint variables per cell."""
        return self.lambda_variables.shape[1] if self.lambda_variables.size > 0 else 0
    
    def compute_norm(self, norm_type: str = "l2") -> float:
        """Compute norm of adjoint variables."""
        if self.lambda_variables.size == 0:
            return 0.0
        
        if norm_type == "l2":
            return np.linalg.norm(self.lambda_variables)
        elif norm_type == "linf":
            return np.max(np.abs(self.lambda_variables))
        elif norm_type == "l1":
            return np.sum(np.abs(self.lambda_variables))
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
    
    def normalize(self, reference_value: float = 1.0) -> None:
        """Normalize adjoint variables."""
        current_norm = self.compute_norm()
        if current_norm > 1e-12:
            self.lambda_variables *= reference_value / current_norm
    
    def copy(self) -> 'AdjointState':
        """Create deep copy of adjoint state."""
        return AdjointState(
            lambda_variables=self.lambda_variables.copy(),
            lambda_boundary=self.lambda_boundary.copy(),
            lambda_gradients=self.lambda_gradients.copy(),
            adjoint_residuals=self.adjoint_residuals.copy(),
            residual_history=self.residual_history.copy(),
            solver_info=self.solver_info.copy()
        )


class AdjointVariables:
    """
    Management class for adjoint variables in discrete adjoint method.
    
    Handles storage, initialization, and manipulation of adjoint variables
    for CFD optimization applications.
    """
    
    def __init__(self, 
                 n_cells: int,
                 n_variables: int = 5,
                 n_boundary_faces: Optional[int] = None,
                 config: Optional[AdjointConfig] = None):
        """
        Initialize adjoint variables.
        
        Args:
            n_cells: Number of computational cells
            n_variables: Number of conservative variables (default 5 for Euler)
            n_boundary_faces: Number of boundary faces
            config: Adjoint configuration
        """
        self.n_cells = n_cells
        self.n_variables = n_variables
        self.n_boundary_faces = n_boundary_faces or 0
        self.config = config or AdjointConfig()
        
        # Initialize adjoint state
        self.current_state = AdjointState()
        self.previous_state: Optional[AdjointState] = None
        
        # Initialize arrays
        self._initialize_arrays()
        
        # Linearization storage
        self.jacobian_storage: Dict[str, Any] = {}
        self.boundary_jacobian_storage: Dict[str, Any] = {}
        
        # Design variables and sensitivities
        self.design_variables: Dict[str, np.ndarray] = {}
        self.design_sensitivities: Dict[str, np.ndarray] = {}
        
        # Convergence monitoring
        self.iteration_count = 0
        self.is_converged = False
        
        logger.info(f"Initialized adjoint variables: {n_cells} cells, {n_variables} variables")
    
    def _initialize_arrays(self) -> None:
        """Initialize adjoint variable arrays."""
        # Primary adjoint variables (λ)
        self.current_state.lambda_variables = np.zeros((self.n_cells, self.n_variables))
        
        # Boundary adjoint variables
        if self.n_boundary_faces > 0:
            self.current_state.lambda_boundary = np.zeros((self.n_boundary_faces, self.n_variables))
        
        # Adjoint gradients
        self.current_state.lambda_gradients = np.zeros((self.n_cells, self.n_variables, 3))
        
        # Residuals
        self.current_state.adjoint_residuals = np.zeros((self.n_cells, self.n_variables))
        
        logger.debug(f"Initialized adjoint arrays: shape={self.current_state.lambda_variables.shape}")
    
    def initialize_from_flow_solution(self, 
                                    flow_solution: np.ndarray,
                                    initialization_mode: str = "zero") -> None:
        """
        Initialize adjoint variables based on flow solution.
        
        Args:
            flow_solution: Converged flow solution [n_cells, n_variables]
            initialization_mode: "zero", "ones", "flow_based", "random"
        """
        if initialization_mode == "zero":
            self.current_state.lambda_variables.fill(0.0)
            
        elif initialization_mode == "ones":
            self.current_state.lambda_variables.fill(1.0)
            
        elif initialization_mode == "flow_based":
            # Initialize based on flow solution characteristics
            self._initialize_flow_based(flow_solution)
            
        elif initialization_mode == "random":
            # Small random perturbations
            self.current_state.lambda_variables = np.random.normal(
                0.0, 0.01, (self.n_cells, self.n_variables)
            )
            
        else:
            raise ValueError(f"Unknown initialization mode: {initialization_mode}")
        
        # Clear boundary variables
        if self.current_state.lambda_boundary.size > 0:
            self.current_state.lambda_boundary.fill(0.0)
        
        logger.info(f"Initialized adjoint variables using '{initialization_mode}' mode")
    
    def _initialize_flow_based(self, flow_solution: np.ndarray) -> None:
        """Initialize adjoint variables based on flow solution characteristics."""
        # Extract primitive variables
        from ..equations.primitive_conservative import VariableConverter
        converter = VariableConverter()
        
        for cell_id in range(self.n_cells):
            conservatives = flow_solution[cell_id]
            primitives = converter.conservatives_to_primitives(conservatives)
            
            if primitives is not None:
                rho, u, v, w, p = primitives[:5]
                
                # Initialize with inverse of characteristic scales
                self.current_state.lambda_variables[cell_id, 0] = 1.0 / max(rho, 1e-6)  # ∂/∂ρ
                self.current_state.lambda_variables[cell_id, 1] = 1.0 / max(abs(u), 1.0)  # ∂/∂(ρu)
                self.current_state.lambda_variables[cell_id, 2] = 1.0 / max(abs(v), 1.0)  # ∂/∂(ρv)
                self.current_state.lambda_variables[cell_id, 3] = 1.0 / max(abs(w), 1.0)  # ∂/∂(ρw)
                self.current_state.lambda_variables[cell_id, 4] = 1.0 / max(p, 1e-3)     # ∂/∂(ρE)
    
    def update_adjoint_gradients(self, connectivity_manager) -> None:
        """Compute gradients of adjoint variables using least squares."""
        self.current_state.lambda_gradients.fill(0.0)
        
        for cell_id in range(self.n_cells):
            neighbors = connectivity_manager.get_cell_neighbors(cell_id)
            
            if len(neighbors) < 3:  # Need at least 3 neighbors for gradient
                continue
            
            # Build least squares system for gradient computation
            cell_center = connectivity_manager.get_cell_center(cell_id)
            neighbor_centers = [connectivity_manager.get_cell_center(nb) for nb in neighbors]
            
            # Compute gradients for each adjoint variable
            for var in range(self.n_variables):
                gradient = self._compute_least_squares_gradient(
                    cell_id, var, neighbors, cell_center, neighbor_centers
                )
                self.current_state.lambda_gradients[cell_id, var] = gradient
    
    def _compute_least_squares_gradient(self, 
                                      cell_id: int, 
                                      variable: int,
                                      neighbors: List[int],
                                      cell_center: np.ndarray,
                                      neighbor_centers: List[np.ndarray]) -> np.ndarray:
        """Compute least squares gradient for a specific variable."""
        n_neighbors = len(neighbors)
        if n_neighbors < 3:
            return np.zeros(3)
        
        # Build geometry matrix [Δx, Δy, Δz]
        A = np.zeros((n_neighbors, 3))
        b = np.zeros(n_neighbors)
        
        for i, (neighbor_id, neighbor_center) in enumerate(zip(neighbors, neighbor_centers)):
            A[i] = neighbor_center - cell_center
            b[i] = (self.current_state.lambda_variables[neighbor_id, variable] - 
                   self.current_state.lambda_variables[cell_id, variable])
        
        # Solve least squares system
        try:
            gradient, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return gradient
        except np.linalg.LinAlgError:
            return np.zeros(3)
    
    def compute_residual_norm(self, norm_type: str = "l2") -> float:
        """Compute norm of adjoint residuals."""
        return self._compute_array_norm(self.current_state.adjoint_residuals, norm_type)
    
    def _compute_array_norm(self, array: np.ndarray, norm_type: str) -> float:
        """Compute norm of array."""
        if array.size == 0:
            return 0.0
        
        if norm_type == "l2":
            return np.linalg.norm(array)
        elif norm_type == "linf":
            return np.max(np.abs(array))
        elif norm_type == "l1":
            return np.sum(np.abs(array))
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
    
    def check_convergence(self, tolerance: Optional[float] = None) -> bool:
        """Check if adjoint solution has converged."""
        tol = tolerance or self.config.tolerance
        residual_norm = self.compute_residual_norm()
        
        # Store residual history
        self.current_state.residual_history.append(residual_norm)
        
        # Check convergence
        self.is_converged = residual_norm < tol
        
        if self.config.output_level >= 2:
            logger.info(f"Adjoint iteration {self.iteration_count}: "
                       f"residual = {residual_norm:.2e}, converged = {self.is_converged}")
        
        return self.is_converged
    
    def save_state(self) -> None:
        """Save current state for backup/restart."""
        self.previous_state = self.current_state.copy()
    
    def restore_state(self) -> None:
        """Restore previous state."""
        if self.previous_state is not None:
            self.current_state = self.previous_state.copy()
            logger.info("Restored previous adjoint state")
        else:
            logger.warning("No previous state available to restore")
    
    def add_design_variable(self, name: str, values: np.ndarray) -> None:
        """Add design variable for sensitivity computation."""
        self.design_variables[name] = values.copy()
        # Initialize corresponding sensitivity array
        self.design_sensitivities[name] = np.zeros_like(values)
        
        logger.info(f"Added design variable '{name}' with {values.size} parameters")
    
    def get_design_sensitivity(self, name: str) -> Optional[np.ndarray]:
        """Get computed sensitivity for design variable."""
        return self.design_sensitivities.get(name)
    
    def update_design_sensitivity(self, name: str, sensitivity: np.ndarray) -> None:
        """Update sensitivity for design variable."""
        if name in self.design_variables:
            self.design_sensitivities[name] = sensitivity.copy()
        else:
            logger.warning(f"Design variable '{name}' not found")
    
    def get_solver_statistics(self) -> Dict[str, Any]:
        """Get comprehensive solver statistics."""
        return {
            'iteration_count': self.iteration_count,
            'is_converged': self.is_converged,
            'final_residual': self.current_state.residual_history[-1] if self.current_state.residual_history else 0.0,
            'residual_history': self.current_state.residual_history.copy(),
            'lambda_norm': self.current_state.compute_norm(),
            'n_design_variables': len(self.design_variables),
            'solver_info': self.current_state.solver_info.copy()
        }
    
    def export_adjoint_solution(self, filename: str) -> None:
        """Export adjoint solution to file."""
        data = {
            'lambda_variables': self.current_state.lambda_variables,
            'lambda_boundary': self.current_state.lambda_boundary,
            'lambda_gradients': self.current_state.lambda_gradients,
            'residual_history': self.current_state.residual_history,
            'design_sensitivities': self.design_sensitivities,
            'config': self.config.__dict__,
            'statistics': self.get_solver_statistics()
        }
        
        np.savez_compressed(filename, **data)
        logger.info(f"Exported adjoint solution to {filename}")
    
    def import_adjoint_solution(self, filename: str) -> None:
        """Import adjoint solution from file."""
        try:
            data = np.load(filename, allow_pickle=True)
            
            self.current_state.lambda_variables = data['lambda_variables']
            self.current_state.lambda_boundary = data['lambda_boundary']
            self.current_state.lambda_gradients = data['lambda_gradients']
            self.current_state.residual_history = data['residual_history'].tolist()
            
            # Load design sensitivities
            if 'design_sensitivities' in data:
                self.design_sensitivities = data['design_sensitivities'].item()
            
            logger.info(f"Imported adjoint solution from {filename}")
            
        except Exception as e:
            logger.error(f"Failed to import adjoint solution: {e}")
            raise


def create_adjoint_variables(n_cells: int,
                           n_variables: int = 5,
                           n_boundary_faces: Optional[int] = None,
                           config: Optional[AdjointConfig] = None) -> AdjointVariables:
    """
    Factory function to create adjoint variables.
    
    Args:
        n_cells: Number of computational cells
        n_variables: Number of conservative variables
        n_boundary_faces: Number of boundary faces  
        config: Adjoint configuration
        
    Returns:
        Configured AdjointVariables instance
    """
    return AdjointVariables(n_cells, n_variables, n_boundary_faces, config)


def test_adjoint_variables():
    """Test adjoint variables functionality."""
    print("Testing Adjoint Variables:")
    
    # Create test configuration
    config = AdjointConfig(
        tolerance=1e-6,
        max_iterations=100,
        output_level=2
    )
    
    # Create adjoint variables
    n_cells = 1000
    n_variables = 5
    adjoint_vars = create_adjoint_variables(n_cells, n_variables, config=config)
    
    # Initialize with random flow solution
    flow_solution = np.random.rand(n_cells, n_variables)
    adjoint_vars.initialize_from_flow_solution(flow_solution, "flow_based")
    
    # Add design variables
    design_params = np.random.rand(10)
    adjoint_vars.add_design_variable("shape_parameters", design_params)
    
    # Test norms and statistics
    lambda_norm = adjoint_vars.current_state.compute_norm()
    residual_norm = adjoint_vars.compute_residual_norm()
    
    print(f"  Lambda norm: {lambda_norm:.6f}")
    print(f"  Residual norm: {residual_norm:.6f}")
    print(f"  Statistics: {adjoint_vars.get_solver_statistics()}")
    
    # Test convergence check
    adjoint_vars.current_state.adjoint_residuals = np.random.rand(n_cells, n_variables) * 1e-8
    converged = adjoint_vars.check_convergence()
    print(f"  Converged: {converged}")


if __name__ == "__main__":
    test_adjoint_variables()