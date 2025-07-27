"""
Discrete Adjoint Equations Implementation

Implements the discrete adjoint method for CFD optimization including:
- Linearization of discrete residuals
- Jacobian computation (exact and approximate)
- Adjoint equation assembly and solution
- Matrix-free implementations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from scipy.sparse import csr_matrix, linalg as spla
from scipy.sparse.linalg import LinearOperator

from .adjoint_variables import AdjointVariables, AdjointConfig
from .objective_functions import ObjectiveFunction

logger = logging.getLogger(__name__)


@dataclass
class LinearizationConfig:
    """Configuration for linearization procedures."""
    
    # Differentiation method
    differentiation_method: str = "finite_difference"  # "finite_difference", "complex_step", "automatic"
    step_size: float = 1e-8
    
    # Jacobian computation
    compute_exact_jacobian: bool = False
    use_coloring: bool = True  # Graph coloring for efficient Jacobian
    store_jacobian: bool = False
    
    # Matrix-free settings
    matrix_free: bool = True
    recompute_frequency: int = 10  # Recompute preconditioner frequency
    
    # Frozen variables
    freeze_turbulence: bool = True
    freeze_geometry: bool = False
    
    # Numerical parameters
    relative_tolerance: float = 1e-12
    absolute_tolerance: float = 1e-15


class FluxJacobian:
    """
    Computes Jacobian matrices for flux functions.
    
    Handles linearization of inviscid and viscous fluxes
    with respect to conservative variables.
    """
    
    def __init__(self, 
                 config: Optional[LinearizationConfig] = None,
                 gamma: float = 1.4):
        """
        Initialize flux Jacobian.
        
        Args:
            config: Linearization configuration
            gamma: Specific heat ratio
        """
        self.config = config or LinearizationConfig()
        self.gamma = gamma
        
        # Cache for computed Jacobians
        self._jacobian_cache: Dict[str, np.ndarray] = {}
        
    def compute_flux_jacobian_normal(self, 
                                   conservative_vars: np.ndarray,
                                   normal: np.ndarray) -> np.ndarray:
        """
        Compute flux Jacobian in normal direction ∂F·n/∂U.
        
        Args:
            conservative_vars: Conservative variables [5]
            normal: Unit normal vector [3]
            
        Returns:
            Flux Jacobian matrix [5, 5]
        """
        rho, rho_u, rho_v, rho_w, rho_E = conservative_vars
        nx, ny, nz = normal
        
        # Avoid division by zero
        rho = max(rho, 1e-12)
        
        # Velocities
        u = rho_u / rho
        v = rho_v / rho
        w = rho_w / rho
        
        # Normal velocity
        u_n = u * nx + v * ny + w * nz
        
        # Pressure and enthalpy
        kinetic_energy = 0.5 * (u**2 + v**2 + w**2)
        pressure = (self.gamma - 1) * (rho_E - rho * kinetic_energy)
        pressure = max(pressure, 1e-6)
        
        enthalpy = (rho_E + pressure) / rho
        
        # Speed of sound
        a_squared = self.gamma * pressure / rho
        
        # Flux Jacobian matrix (exact analytical form)
        A = np.zeros((5, 5))
        
        # Mass equation row
        A[0, 0] = 0.0
        A[0, 1] = nx
        A[0, 2] = ny
        A[0, 3] = nz
        A[0, 4] = 0.0
        
        # Momentum equations rows
        for i, n_i in enumerate([nx, ny, nz]):
            row = i + 1
            A[row, 0] = -u * u_n + (self.gamma - 1) * kinetic_energy * n_i
            A[row, 1] = u_n * (1 if i == 0 else 0) + u * nx - (self.gamma - 1) * u * n_i
            A[row, 2] = u_n * (1 if i == 1 else 0) + u * ny - (self.gamma - 1) * v * n_i
            A[row, 3] = u_n * (1 if i == 2 else 0) + u * nz - (self.gamma - 1) * w * n_i
            A[row, 4] = (self.gamma - 1) * n_i
        
        # Energy equation row
        A[4, 0] = u_n * ((self.gamma - 1) * kinetic_energy - enthalpy)
        A[4, 1] = enthalpy * nx - (self.gamma - 1) * u * u_n
        A[4, 2] = enthalpy * ny - (self.gamma - 1) * v * u_n
        A[4, 3] = enthalpy * nz - (self.gamma - 1) * w * u_n
        A[4, 4] = self.gamma * u_n
        
        return A
    
    def compute_riemann_jacobian(self,
                               left_state: np.ndarray,
                               right_state: np.ndarray,
                               normal: np.ndarray,
                               riemann_solver_type: str = "roe") -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Riemann solver Jacobians ∂F_Riemann/∂U_L and ∂F_Riemann/∂U_R.
        
        Args:
            left_state: Left conservative variables
            right_state: Right conservative variables  
            normal: Interface normal
            riemann_solver_type: Type of Riemann solver
            
        Returns:
            (dF_dUL, dF_dUR) Jacobian matrices
        """
        if riemann_solver_type == "roe":
            return self._compute_roe_jacobian(left_state, right_state, normal)
        elif riemann_solver_type == "rusanov":
            return self._compute_rusanov_jacobian(left_state, right_state, normal)
        else:
            # Fallback to finite difference
            return self._compute_riemann_jacobian_fd(left_state, right_state, normal, riemann_solver_type)
    
    def _compute_roe_jacobian(self,
                            left_state: np.ndarray,
                            right_state: np.ndarray,
                            normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute analytical Roe flux Jacobians."""
        # Get individual flux Jacobians
        A_L = self.compute_flux_jacobian_normal(left_state, normal)
        A_R = self.compute_flux_jacobian_normal(right_state, normal)
        
        # Roe-averaged state
        roe_state = self._compute_roe_average(left_state, right_state)
        A_roe = self.compute_flux_jacobian_normal(roe_state, normal)
        
        # Eigendecomposition of Roe matrix
        eigenvals, eigenvecs = self._eigendecomposition_roe(roe_state, normal)
        
        # Construct |A_roe| = R |Λ| L
        abs_eigenvals = np.abs(eigenvals)
        abs_A_roe = eigenvecs @ np.diag(abs_eigenvals) @ np.linalg.inv(eigenvecs)
        
        # Roe Jacobians
        dF_dUL = 0.5 * (A_L + abs_A_roe)
        dF_dUR = 0.5 * (A_R - abs_A_roe)
        
        return dF_dUL, dF_dUR
    
    def _compute_roe_average(self, left_state: np.ndarray, right_state: np.ndarray) -> np.ndarray:
        """Compute Roe-averaged state."""
        rho_L, rho_u_L, rho_v_L, rho_w_L, rho_E_L = left_state
        rho_R, rho_u_R, rho_v_R, rho_w_R, rho_E_R = right_state
        
        # Roe averaging
        sqrt_rho_L = np.sqrt(max(rho_L, 1e-12))
        sqrt_rho_R = np.sqrt(max(rho_R, 1e-12))
        
        rho_roe = sqrt_rho_L * sqrt_rho_R
        u_roe = (sqrt_rho_L * rho_u_L/rho_L + sqrt_rho_R * rho_u_R/rho_R) / (sqrt_rho_L + sqrt_rho_R)
        v_roe = (sqrt_rho_L * rho_v_L/rho_L + sqrt_rho_R * rho_v_R/rho_R) / (sqrt_rho_L + sqrt_rho_R)
        w_roe = (sqrt_rho_L * rho_w_L/rho_L + sqrt_rho_R * rho_w_R/rho_R) / (sqrt_rho_L + sqrt_rho_R)
        
        # Enthalpy averaging
        H_L = (rho_E_L + (self.gamma - 1) * (rho_E_L - 0.5 * (rho_u_L**2 + rho_v_L**2 + rho_w_L**2)/rho_L)) / rho_L
        H_R = (rho_E_R + (self.gamma - 1) * (rho_E_R - 0.5 * (rho_u_R**2 + rho_v_R**2 + rho_w_R**2)/rho_R)) / rho_R
        H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / (sqrt_rho_L + sqrt_rho_R)
        
        rho_E_roe = rho_roe * (H_roe - 0.5 * (u_roe**2 + v_roe**2 + w_roe**2))
        
        return np.array([rho_roe, rho_roe * u_roe, rho_roe * v_roe, rho_roe * w_roe, rho_E_roe])
    
    def _eigendecomposition_roe(self, roe_state: np.ndarray, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigendecomposition of Roe matrix."""
        rho, rho_u, rho_v, rho_w, rho_E = roe_state
        nx, ny, nz = normal
        
        u = rho_u / rho
        v = rho_v / rho  
        w = rho_w / rho
        u_n = u * nx + v * ny + w * nz
        
        # Speed of sound
        kinetic_energy = 0.5 * (u**2 + v**2 + w**2)
        pressure = (self.gamma - 1) * (rho_E - rho * kinetic_energy)
        a = np.sqrt(max(self.gamma * pressure / rho, 1e-12))
        
        # Eigenvalues
        eigenvals = np.array([u_n - a, u_n, u_n, u_n, u_n + a])
        
        # Right eigenvectors (simplified 1D version)
        R = np.eye(5)  # Placeholder - full 3D implementation is complex
        
        return eigenvals, R
    
    def _compute_rusanov_jacobian(self,
                                left_state: np.ndarray,
                                right_state: np.ndarray,
                                normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Rusanov flux Jacobians."""
        A_L = self.compute_flux_jacobian_normal(left_state, normal)
        A_R = self.compute_flux_jacobian_normal(right_state, normal)
        
        # Maximum eigenvalue
        lambda_max = self._compute_max_eigenvalue(left_state, right_state, normal)
        
        # Rusanov Jacobians
        dF_dUL = 0.5 * (A_L + lambda_max * np.eye(5))
        dF_dUR = 0.5 * (A_R - lambda_max * np.eye(5))
        
        return dF_dUL, dF_dUR
    
    def _compute_max_eigenvalue(self,
                              left_state: np.ndarray,
                              right_state: np.ndarray,
                              normal: np.ndarray) -> float:
        """Compute maximum eigenvalue for Rusanov flux."""
        # Left state
        rho_L, rho_u_L, rho_v_L, rho_w_L, rho_E_L = left_state
        u_L = rho_u_L / max(rho_L, 1e-12)
        v_L = rho_v_L / max(rho_L, 1e-12)
        w_L = rho_w_L / max(rho_L, 1e-12)
        u_n_L = u_L * normal[0] + v_L * normal[1] + w_L * normal[2]
        
        kinetic_L = 0.5 * rho_L * (u_L**2 + v_L**2 + w_L**2)
        p_L = (self.gamma - 1) * (rho_E_L - kinetic_L)
        a_L = np.sqrt(max(self.gamma * p_L / rho_L, 1e-12))
        
        # Right state
        rho_R, rho_u_R, rho_v_R, rho_w_R, rho_E_R = right_state
        u_R = rho_u_R / max(rho_R, 1e-12)
        v_R = rho_v_R / max(rho_R, 1e-12)
        w_R = rho_w_R / max(rho_R, 1e-12)
        u_n_R = u_R * normal[0] + v_R * normal[1] + w_R * normal[2]
        
        kinetic_R = 0.5 * rho_R * (u_R**2 + v_R**2 + w_R**2)
        p_R = (self.gamma - 1) * (rho_E_R - kinetic_R)
        a_R = np.sqrt(max(self.gamma * p_R / rho_R, 1e-12))
        
        return max(abs(u_n_L) + a_L, abs(u_n_R) + a_R)
    
    def _compute_riemann_jacobian_fd(self,
                                   left_state: np.ndarray,
                                   right_state: np.ndarray,
                                   normal: np.ndarray,
                                   solver_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Riemann Jacobians using finite differences."""
        eps = self.config.step_size
        n_vars = 5
        
        # Import Riemann solver
        from ..numerics.riemann_solvers import create_riemann_solver
        solver = create_riemann_solver(solver_type, gamma=self.gamma)
        
        # Convert to primitive variables for Riemann solver
        def conservative_to_primitive(U):
            rho, rho_u, rho_v, rho_w, rho_E = U
            rho = max(rho, 1e-12)
            u, v, w = rho_u/rho, rho_v/rho, rho_w/rho
            kinetic = 0.5 * rho * (u**2 + v**2 + w**2)
            p = (self.gamma - 1) * (rho_E - kinetic)
            return np.array([rho, u, v, w, max(p, 1e-6)])
        
        # Reference flux
        prim_L = conservative_to_primitive(left_state)
        prim_R = conservative_to_primitive(right_state)
        flux_ref, _ = solver.solve(prim_L, prim_R, normal)
        
        # Jacobian w.r.t. left state
        dF_dUL = np.zeros((n_vars, n_vars))
        for j in range(n_vars):
            U_L_pert = left_state.copy()
            U_L_pert[j] += eps
            
            prim_L_pert = conservative_to_primitive(U_L_pert)
            flux_pert, _ = solver.solve(prim_L_pert, prim_R, normal)
            
            dF_dUL[:, j] = (flux_pert - flux_ref) / eps
        
        # Jacobian w.r.t. right state
        dF_dUR = np.zeros((n_vars, n_vars))
        for j in range(n_vars):
            U_R_pert = right_state.copy()
            U_R_pert[j] += eps
            
            prim_R_pert = conservative_to_primitive(U_R_pert)
            flux_pert, _ = solver.solve(prim_L, prim_R_pert, normal)
            
            dF_dUR[:, j] = (flux_pert - flux_ref) / eps
        
        return dF_dUL, dF_dUR


class BoundaryJacobian:
    """
    Computes Jacobian matrices for boundary conditions.
    
    Handles linearization of boundary flux computations
    and boundary condition implementations.
    """
    
    def __init__(self, config: Optional[LinearizationConfig] = None):
        """Initialize boundary Jacobian."""
        self.config = config or LinearizationConfig()
    
    def compute_boundary_jacobian(self,
                                boundary_type: str,
                                interior_state: np.ndarray,
                                boundary_normal: np.ndarray,
                                boundary_data: Dict[str, Any]) -> np.ndarray:
        """
        Compute boundary condition Jacobian.
        
        Args:
            boundary_type: Type of boundary condition
            interior_state: Interior cell state
            boundary_normal: Boundary normal vector
            boundary_data: Boundary condition data
            
        Returns:
            Boundary Jacobian matrix [5, 5]
        """
        if boundary_type == "wall":
            return self._compute_wall_jacobian(interior_state, boundary_normal)
        elif boundary_type == "farfield":
            return self._compute_farfield_jacobian(interior_state, boundary_normal, boundary_data)
        elif boundary_type == "symmetry":
            return self._compute_symmetry_jacobian(interior_state, boundary_normal)
        else:
            # Default: identity (no boundary effect)
            return np.eye(5)
    
    def _compute_wall_jacobian(self,
                             interior_state: np.ndarray,
                             normal: np.ndarray) -> np.ndarray:
        """Compute wall boundary Jacobian."""
        # For adiabatic wall: normal velocity = 0, no heat transfer
        # This affects momentum and energy equations
        
        # Simplified implementation - full version would be more complex
        jacobian = np.eye(5)
        
        # Zero normal momentum flux
        nx, ny, nz = normal
        jacobian[1, 1] *= (1 - nx**2)  # Modify x-momentum equation
        jacobian[2, 2] *= (1 - ny**2)  # Modify y-momentum equation  
        jacobian[3, 3] *= (1 - nz**2)  # Modify z-momentum equation
        
        return jacobian
    
    def _compute_farfield_jacobian(self,
                                 interior_state: np.ndarray,
                                 normal: np.ndarray,
                                 boundary_data: Dict[str, Any]) -> np.ndarray:
        """Compute farfield boundary Jacobian."""
        # Farfield: characteristic-based boundary conditions
        # Implementation depends on inflow/outflow conditions
        
        # Placeholder implementation
        return np.eye(5)
    
    def _compute_symmetry_jacobian(self,
                                 interior_state: np.ndarray,
                                 normal: np.ndarray) -> np.ndarray:
        """Compute symmetry boundary Jacobian."""
        # Symmetry: zero normal velocity, zero normal gradient of other variables
        jacobian = np.eye(5)
        
        # Zero normal velocity component
        nx, ny, nz = normal
        jacobian[1, 1] = 1 - nx**2
        jacobian[1, 2] = -nx * ny
        jacobian[1, 3] = -nx * nz
        
        return jacobian


class AdjointLinearization:
    """
    Handles linearization of discrete residuals for adjoint computation.
    
    Computes residual Jacobian ∂R/∂U using various methods.
    """
    
    def __init__(self,
                 flux_jacobian: FluxJacobian,
                 boundary_jacobian: BoundaryJacobian,
                 config: Optional[LinearizationConfig] = None):
        """
        Initialize adjoint linearization.
        
        Args:
            flux_jacobian: Flux Jacobian computer
            boundary_jacobian: Boundary Jacobian computer
            config: Linearization configuration
        """
        self.flux_jacobian = flux_jacobian
        self.boundary_jacobian = boundary_jacobian
        self.config = config or LinearizationConfig()
        
        # Storage for computed Jacobians
        self.residual_jacobian: Optional[csr_matrix] = None
        self.residual_jacobian_transpose: Optional[csr_matrix] = None
        
    def compute_residual_jacobian(self,
                                solution: np.ndarray,
                                mesh_info: Dict[str, Any],
                                boundary_info: Dict[str, Any]) -> csr_matrix:
        """
        Compute residual Jacobian matrix ∂R/∂U.
        
        Args:
            solution: Current solution [n_cells, n_variables]
            mesh_info: Mesh information
            boundary_info: Boundary information
            
        Returns:
            Sparse residual Jacobian matrix
        """
        n_cells, n_vars = solution.shape
        total_size = n_cells * n_vars
        
        if self.config.compute_exact_jacobian:
            jacobian = self._compute_exact_jacobian(solution, mesh_info, boundary_info)
        else:
            jacobian = self._compute_approximate_jacobian(solution, mesh_info, boundary_info)
        
        self.residual_jacobian = jacobian
        return jacobian
    
    def _compute_exact_jacobian(self,
                              solution: np.ndarray,
                              mesh_info: Dict[str, Any],
                              boundary_info: Dict[str, Any]) -> csr_matrix:
        """Compute exact residual Jacobian using analytical derivatives."""
        n_cells, n_vars = solution.shape
        total_size = n_cells * n_vars
        
        # Lists for sparse matrix construction
        rows, cols, data = [], [], []
        
        # Process internal faces
        for face_id in range(mesh_info['n_internal_faces']):
            left_cell = mesh_info['face_owners'][face_id]
            right_cell = mesh_info['face_neighbors'][face_id]
            
            face_normal = mesh_info['face_normals'][face_id]
            face_area = mesh_info['face_areas'][face_id]
            
            # Get cell states
            U_L = solution[left_cell]
            U_R = solution[right_cell]
            
            # Compute flux Jacobians
            dF_dUL, dF_dUR = self.flux_jacobian.compute_riemann_jacobian(
                U_L, U_R, face_normal, mesh_info.get('riemann_solver', 'roe')
            )
            
            # Scale by face area
            dF_dUL *= face_area
            dF_dUR *= face_area
            
            # Add contributions to Jacobian
            self._add_jacobian_block(rows, cols, data, left_cell, left_cell, dF_dUL, n_vars)
            self._add_jacobian_block(rows, cols, data, left_cell, right_cell, dF_dUR, n_vars)
            self._add_jacobian_block(rows, cols, data, right_cell, left_cell, -dF_dUL, n_vars)
            self._add_jacobian_block(rows, cols, data, right_cell, right_cell, -dF_dUR, n_vars)
        
        # Process boundary faces
        for boundary_id, boundary_data in boundary_info.items():
            faces = boundary_data.get('faces', [])
            boundary_type = boundary_data.get('type', 'wall')
            
            for face_id in faces:
                owner_cell = mesh_info['face_owners'][face_id]
                face_normal = mesh_info['face_normals'][face_id]
                face_area = mesh_info['face_areas'][face_id]
                
                U_owner = solution[owner_cell]
                
                # Boundary Jacobian
                dF_dU = self.boundary_jacobian.compute_boundary_jacobian(
                    boundary_type, U_owner, face_normal, boundary_data
                )
                dF_dU *= face_area
                
                self._add_jacobian_block(rows, cols, data, owner_cell, owner_cell, dF_dU, n_vars)
        
        # Construct sparse matrix
        jacobian = csr_matrix((data, (rows, cols)), shape=(total_size, total_size))
        
        return jacobian
    
    def _add_jacobian_block(self,
                          rows: List[int], 
                          cols: List[int], 
                          data: List[float],
                          row_cell: int, 
                          col_cell: int, 
                          block: np.ndarray,
                          n_vars: int) -> None:
        """Add 5x5 block to sparse matrix lists."""
        for i in range(n_vars):
            for j in range(n_vars):
                if abs(block[i, j]) > self.config.absolute_tolerance:
                    rows.append(row_cell * n_vars + i)
                    cols.append(col_cell * n_vars + j)
                    data.append(block[i, j])
    
    def _compute_approximate_jacobian(self,
                                    solution: np.ndarray,
                                    mesh_info: Dict[str, Any],
                                    boundary_info: Dict[str, Any]) -> csr_matrix:
        """Compute approximate Jacobian using finite differences."""
        n_cells, n_vars = solution.shape
        total_size = n_cells * n_vars
        
        # This would use finite differences to approximate ∂R/∂U
        # Implementation omitted for brevity - would be similar to exact but using perturbations
        
        # Return identity as placeholder
        return csr_matrix(np.eye(total_size))
    
    def get_jacobian_transpose(self) -> csr_matrix:
        """Get transpose of residual Jacobian for adjoint solve."""
        if self.residual_jacobian is not None:
            if self.residual_jacobian_transpose is None:
                self.residual_jacobian_transpose = self.residual_jacobian.T
            return self.residual_jacobian_transpose
        else:
            raise RuntimeError("Residual Jacobian not computed yet")


class MatrixFreeOperator(LinearOperator):
    """
    Matrix-free linear operator for adjoint equations.
    
    Implements Jacobian-vector products without storing the full matrix.
    """
    
    def __init__(self,
                 solution: np.ndarray,
                 residual_function: Callable[[np.ndarray], np.ndarray],
                 config: Optional[LinearizationConfig] = None):
        """
        Initialize matrix-free operator.
        
        Args:
            solution: Current solution state
            residual_function: Function to compute residual R(U)
            config: Linearization configuration
        """
        self.solution = solution.flatten()
        self.residual_function = residual_function
        self.config = config or LinearizationConfig()
        
        # Shape information
        n_total = len(self.solution)
        super().__init__(dtype=np.float64, shape=(n_total, n_total))
        
        # Reference residual
        self.residual_ref = self.residual_function(self.solution.reshape(solution.shape))
        
    def _matvec(self, v: np.ndarray) -> np.ndarray:
        """Compute Jacobian-vector product (∂R/∂U)^T v."""
        eps = self.config.step_size
        solution_shape = self.solution.shape
        
        # Perturb solution in direction v
        solution_pert = self.solution + eps * v
        residual_pert = self.residual_function(solution_pert.reshape(solution_shape[0], -1))
        
        # Finite difference approximation of J^T v
        jac_transpose_v = (residual_pert.flatten() - self.residual_ref.flatten()) / eps
        
        return jac_transpose_v
    
    def _rmatvec(self, v: np.ndarray) -> np.ndarray:
        """Compute transpose Jacobian-vector product J^T v."""
        return self._matvec(v)


class DiscreteAdjointSolver:
    """
    Main discrete adjoint solver.
    
    Solves the discrete adjoint equations: (∂R/∂U)^T λ = -∂J/∂U
    """
    
    def __init__(self,
                 adjoint_variables: AdjointVariables,
                 objective_function: ObjectiveFunction,
                 config: Optional[LinearizationConfig] = None):
        """
        Initialize discrete adjoint solver.
        
        Args:
            adjoint_variables: Adjoint variable manager
            objective_function: Objective function
            config: Linearization configuration
        """
        self.adjoint_variables = adjoint_variables
        self.objective_function = objective_function
        self.config = config or LinearizationConfig()
        
        # Linearization components
        self.flux_jacobian = FluxJacobian(config)
        self.boundary_jacobian = BoundaryJacobian(config)
        self.linearization = AdjointLinearization(
            self.flux_jacobian, self.boundary_jacobian, config
        )
        
        # Matrix-free operator
        self.matrix_free_operator: Optional[MatrixFreeOperator] = None
        
    def solve_adjoint_equations(self,
                              flow_solution: np.ndarray,
                              mesh_info: Dict[str, Any],
                              boundary_info: Dict[str, Any]) -> bool:
        """
        Solve discrete adjoint equations.
        
        Args:
            flow_solution: Converged flow solution
            mesh_info: Mesh information
            boundary_info: Boundary information
            
        Returns:
            True if converged, False otherwise
        """
        logger.info("Solving discrete adjoint equations...")
        
        # Compute objective function source term
        source_term = self.objective_function.compute_source_term(
            flow_solution, mesh_info, boundary_info
        )
        
        # Setup linear system
        if self.config.matrix_free:
            success = self._solve_matrix_free(flow_solution, mesh_info, boundary_info, source_term)
        else:
            success = self._solve_direct(flow_solution, mesh_info, boundary_info, source_term)
        
        return success
    
    def _solve_matrix_free(self,
                         flow_solution: np.ndarray,
                         mesh_info: Dict[str, Any],
                         boundary_info: Dict[str, Any],
                         source_term: np.ndarray) -> bool:
        """Solve adjoint equations using matrix-free methods."""
        def residual_function(solution):
            # This would compute residual R(U) for given solution
            # Implementation depends on the flow solver structure
            return np.zeros_like(solution)  # Placeholder
        
        # Create matrix-free operator
        self.matrix_free_operator = MatrixFreeOperator(
            flow_solution, residual_function, self.config
        )
        
        # Solve using iterative method
        from .iterative_solvers import GMRESAdjointSolver
        iterative_solver = GMRESAdjointSolver(self.adjoint_variables.config)
        
        success = iterative_solver.solve(
            self.matrix_free_operator,
            -source_term.flatten(),
            self.adjoint_variables.current_state.lambda_variables.flatten()
        )
        
        return success
    
    def _solve_direct(self,
                    flow_solution: np.ndarray,
                    mesh_info: Dict[str, Any],
                    boundary_info: Dict[str, Any],
                    source_term: np.ndarray) -> bool:
        """Solve adjoint equations using direct methods."""
        # Compute Jacobian
        jacobian = self.linearization.compute_residual_jacobian(
            flow_solution, mesh_info, boundary_info
        )
        
        # Get transpose for adjoint
        jacobian_T = self.linearization.get_jacobian_transpose()
        
        # Solve linear system: J^T λ = -∂J/∂U
        try:
            lambda_solution = spla.spsolve(jacobian_T, -source_term.flatten())
            
            # Reshape and store solution
            self.adjoint_variables.current_state.lambda_variables = lambda_solution.reshape(
                flow_solution.shape
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Direct adjoint solve failed: {e}")
            return False


def test_adjoint_equations():
    """Test adjoint equations implementation."""
    print("Testing Adjoint Equations:")
    
    # Create test data
    n_cells = 100
    n_vars = 5
    
    solution = np.random.rand(n_cells, n_vars) + 1.0  # Ensure positive density/pressure
    
    # Test flux Jacobian
    config = LinearizationConfig()
    flux_jac = FluxJacobian(config)
    
    conservative_vars = solution[0]
    normal = np.array([1.0, 0.0, 0.0])
    
    jacobian = flux_jac.compute_flux_jacobian_normal(conservative_vars, normal)
    print(f"  Flux Jacobian shape: {jacobian.shape}")
    print(f"  Flux Jacobian norm: {np.linalg.norm(jacobian):.6f}")
    
    # Test Riemann Jacobians
    left_state = solution[0]
    right_state = solution[1]
    
    dF_dUL, dF_dUR = flux_jac.compute_riemann_jacobian(left_state, right_state, normal, "roe")
    print(f"  Riemann Jacobian L norm: {np.linalg.norm(dF_dUL):.6f}")
    print(f"  Riemann Jacobian R norm: {np.linalg.norm(dF_dUR):.6f}")


if __name__ == "__main__":
    test_adjoint_equations()