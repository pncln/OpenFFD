"""
3D Euler Equations Solver for Supersonic Flows

Implements the compressible Euler equations using finite volume method
with advanced flux computation and shock-capturing capabilities.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import time

from .primitive_conservative import VariableConverter
from .equation_state import PerfectGas, AIR_PROPERTIES
from .flux_functions import InviscidFlux, FluxCalculator
from ..mesh.unstructured_mesh import UnstructuredMesh3D
from ..mesh.connectivity import ConnectivityManager
from ..mesh.boundary import BoundaryManager
from ..numerics.riemann_solvers import RiemannSolverManager, create_riemann_solver
from ..numerics.shock_detectors import create_default_shock_detector
from ..numerics.weno_schemes import create_weno_reconstructor
from ..numerics.tvd_schemes import create_tvd_reconstructor

logger = logging.getLogger(__name__)


@dataclass
class EulerSolverConfig:
    """Configuration for Euler equations solver."""
    # Time integration
    time_stepping: str = "explicit"  # "explicit", "implicit"
    cfl_number: float = 0.5
    max_iterations: int = 10000
    convergence_tolerance: float = 1e-6
    
    # Flux computation
    riemann_solver: str = "roe"  # "roe", "hllc", "rusanov", "ausm+", "exact"
    reconstruction_scheme: str = "linear"  # "linear", "weno3", "weno5", "tvd"
    shock_detector: str = "ducros"  # "ducros", "jameson", "pressure_jump", "composite"
    adaptive_riemann: bool = True  # Enable adaptive Riemann solver selection
    
    # Gas properties
    gamma: float = 1.4
    gas_constant: float = 287.0
    
    # Initialization
    initial_conditions: Dict[str, float] = field(default_factory=lambda: {
        'density': 1.225,
        'velocity_x': 100.0,
        'velocity_y': 0.0,
        'velocity_z': 0.0,
        'pressure': 101325.0,
        'temperature': 288.15
    })
    
    # Output and monitoring
    output_frequency: int = 100
    residual_frequency: int = 10
    save_solution: bool = True


class EulerEquations3D:
    """
    Three-dimensional Euler equations solver.
    
    Solves the compressible Euler equations:
    ∂U/∂t + ∇ · F(U) = 0
    
    Where U = [ρ, ρu, ρv, ρw, ρE]ᵀ is the conservative variable vector
    and F(U) is the inviscid flux tensor.
    
    Features:
    - Finite volume method on unstructured meshes
    - Multiple flux schemes (Roe, HLLC, Rusanov)
    - High-order reconstruction with limiters
    - Explicit and implicit time integration
    - Shock-capturing capabilities
    - Parallel computation support
    """
    
    def __init__(self, 
                 mesh: UnstructuredMesh3D,
                 config: Optional[EulerSolverConfig] = None):
        """
        Initialize Euler equations solver.
        
        Args:
            mesh: Unstructured mesh
            config: Solver configuration
        """
        self.mesh = mesh
        self.config = config or EulerSolverConfig()
        
        # Initialize physical models
        self.equation_of_state = PerfectGas(AIR_PROPERTIES)
        self.variable_converter = VariableConverter(
            gamma=self.config.gamma,
            R=self.config.gas_constant
        )
        
        # Flux computation
        self.flux_calculator = FluxCalculator(
            gamma=self.config.gamma,
            R=self.config.gas_constant
        )
        self.inviscid_flux = InviscidFlux(self.config.gamma)
        
        # Initialize advanced numerical schemes
        self.riemann_solver_manager = RiemannSolverManager(
            default_solver=self.config.riemann_solver,
            gamma=self.config.gamma,
            adaptive_switching=self.config.adaptive_riemann
        )
        
        # Shock detector
        self.shock_detector = create_default_shock_detector(self.config.shock_detector)
        
        # High-order reconstruction
        if self.config.reconstruction_scheme.startswith("weno"):
            self.reconstructor = create_weno_reconstructor(self.config.reconstruction_scheme)
        elif self.config.reconstruction_scheme == "tvd":
            self.reconstructor = create_tvd_reconstructor("minmod")
        else:
            self.reconstructor = None  # Use linear reconstruction
        
        # Connectivity and boundary management
        self.connectivity = ConnectivityManager(mesh)
        self.boundary_manager = BoundaryManager(mesh)
        
        # Solution arrays
        self.n_cells = mesh.n_cells
        self.n_variables = 5  # [rho, rho*u, rho*v, rho*w, rho*E]
        
        self.conservatives = np.zeros((self.n_cells, self.n_variables))
        self.primitives = np.zeros((self.n_cells, 6))  # [rho, u, v, w, p, T]
        self.residuals = np.zeros((self.n_cells, self.n_variables))
        self.time_steps = np.zeros(self.n_cells)
        
        # Gradients for high-order schemes
        self.gradients = np.zeros((self.n_cells, self.n_variables, 3))
        
        # Shock indicators
        self.shock_indicators = np.zeros(self.n_cells)
        
        # Solution history and monitoring
        self.iteration = 0
        self.current_time = 0.0
        self.residual_history = []
        self.convergence_history = []
        
        # Performance tracking
        self.timing = {
            'total': 0.0,
            'flux_computation': 0.0,
            'time_stepping': 0.0,
            'boundary_conditions': 0.0,
            'gradient_computation': 0.0
        }
        
        # Flags
        self._initialized = False
        self._converged = False
        
        logger.info(f"Initialized Euler solver: {self.n_cells} cells, "
                   f"Riemann={self.config.riemann_solver}, reconstruction={self.config.reconstruction_scheme}, "
                   f"shock_detector={self.config.shock_detector}, adaptive={self.config.adaptive_riemann}")
    
    def initialize_solution(self) -> None:
        """Initialize flow field with specified conditions."""
        logger.info("Initializing solution...")
        
        # Get initial conditions
        ic = self.config.initial_conditions
        initial_primitives = np.array([
            ic['density'],
            ic['velocity_x'],
            ic['velocity_y'], 
            ic['velocity_z'],
            ic['pressure'],
            ic['temperature']
        ])
        
        # Set uniform initial conditions
        for cell_id in range(self.n_cells):
            self.primitives[cell_id] = initial_primitives
            self.conservatives[cell_id] = self.variable_converter.primitives_to_conservatives(
                initial_primitives
            )
        
        # Initialize connectivity if not done
        if not self.connectivity._connectivity_built:
            self.connectivity.build_connectivity()
        
        # Initialize time steps
        self._compute_time_steps()
        
        self._initialized = True
        logger.info(f"Solution initialized: ρ={ic['density']:.3f}, M={self._compute_initial_mach():.3f}")
    
    def _compute_initial_mach(self) -> float:
        """Compute initial Mach number for monitoring."""
        ic = self.config.initial_conditions
        velocity_mag = np.sqrt(ic['velocity_x']**2 + ic['velocity_y']**2 + ic['velocity_z']**2)
        speed_of_sound = np.sqrt(self.config.gamma * ic['pressure'] / ic['density'])
        return velocity_mag / speed_of_sound
    
    def solve(self) -> Dict[str, Any]:
        """
        Solve the Euler equations to convergence.
        
        Returns:
            Dictionary with solution statistics and convergence information
        """
        if not self._initialized:
            self.initialize_solution()
        
        logger.info(f"Starting Euler solver: {self.config.max_iterations} max iterations")
        start_time = time.time()
        
        # Main iteration loop
        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            
            # Single time step
            self._time_step()
            
            # Check convergence
            if iteration % self.config.residual_frequency == 0:
                residual_norm = self._compute_residual_norm()
                self.residual_history.append(residual_norm)
                
                if residual_norm < self.config.convergence_tolerance:
                    self._converged = True
                    logger.info(f"Converged at iteration {iteration}: residual={residual_norm:.2e}")
                    break
                
                if iteration % self.config.output_frequency == 0:
                    logger.info(f"Iteration {iteration}: residual={residual_norm:.2e}, "
                              f"time={self.current_time:.6f}")
        
        # Final timing
        total_time = time.time() - start_time
        self.timing['total'] = total_time
        
        # Solution statistics
        stats = self._compute_solution_statistics()
        
        logger.info(f"Solver finished: {iteration+1} iterations, {total_time:.3f}s, "
                   f"converged={self._converged}")
        
        return {
            'converged': self._converged,
            'iterations': iteration + 1,
            'final_residual': self.residual_history[-1] if self.residual_history else 0.0,
            'computation_time': total_time,
            'solution_statistics': stats,
            'timing_breakdown': self.timing.copy()
        }
    
    def _time_step(self) -> None:
        """Perform single time step."""
        step_start = time.time()
        
        # Compute gradients for high-order schemes
        if self.config.limiter != "none":
            grad_start = time.time()
            self._compute_gradients()
            self.timing['gradient_computation'] += time.time() - grad_start
        
        # Apply boundary conditions
        bc_start = time.time()
        self.boundary_manager.apply_boundary_conditions(
            self.mesh.cell_data, self.mesh.face_data, self.current_time
        )
        self.timing['boundary_conditions'] += time.time() - bc_start
        
        # Compute fluxes and residuals
        flux_start = time.time()
        self._compute_residuals()
        self.timing['flux_computation'] += time.time() - flux_start
        
        # Time integration
        ts_start = time.time()
        if self.config.time_stepping == "explicit":
            self._explicit_time_step()
        else:
            self._implicit_time_step()
        self.timing['time_stepping'] += time.time() - ts_start
        
        # Update primitive variables
        self._update_primitive_variables()
        
        # Update shock indicators
        self._update_shock_indicators()
    
    def _compute_gradients(self) -> None:
        """Compute solution gradients for high-order reconstruction."""
        if not self.connectivity._stencils_built:
            self.connectivity.build_gradient_stencils()
        
        for cell_id in range(self.n_cells):
            stencil = self.connectivity.get_gradient_stencil(cell_id)
            if not stencil:
                continue
            
            # Compute gradients using least squares
            weights = self.connectivity.compute_stencil_weights(cell_id)
            if weights is None:
                continue
            
            cell_center = self.mesh.cell_data.centroids[cell_id]
            cell_solution = self.conservatives[cell_id]
            
            # Initialize gradients
            gradient = np.zeros((self.n_variables, 3))
            
            # Least squares gradient computation
            for i, neighbor_id in enumerate(stencil):
                neighbor_center = self.mesh.cell_data.centroids[neighbor_id]
                neighbor_solution = self.conservatives[neighbor_id]
                
                # Solution difference
                delta_solution = neighbor_solution - cell_solution
                
                # Add weighted contribution
                for var in range(self.n_variables):
                    gradient[var] += delta_solution[var] * weights[i]
            
            self.gradients[cell_id] = gradient
    
    def _update_shock_indicators(self) -> None:
        """Update shock indicators using the configured detector."""
        try:
            # Prepare mesh information for shock detection
            mesh_info = {
                'connectivity_manager': self.connectivity,
                'centroids': self.mesh.cell_data.centroids,
                'n_cells': self.n_cells
            }
            
            # Detect shocks using conservative variables
            self.shock_indicators = self.shock_detector.detect_shocks(
                self.conservatives, mesh_info
            )
            
            # Smooth shock indicators to avoid sharp transitions
            from ..numerics.shock_detectors import smooth_shock_indicator
            self.shock_indicators = smooth_shock_indicator(
                self.shock_indicators, mesh_info, smoothing_passes=2
            )
            
        except Exception as e:
            logger.warning(f"Shock detection failed: {e}")
            # Set all shock indicators to zero on failure
            self.shock_indicators.fill(0.0)
    
    def _compute_residuals(self) -> None:
        """Compute flux residuals for all cells."""
        self.residuals.fill(0.0)
        
        # Loop over all faces
        for face_id in range(self.mesh.n_faces):
            owner_cell = self.mesh.face_data.owner[face_id]
            neighbor_cell = self.mesh.face_data.neighbor[face_id]
            
            face_area = self.mesh.face_data.areas[face_id]
            face_normal = self.mesh.face_data.normals[face_id]
            
            if neighbor_cell >= 0:  # Internal face
                # Compute interface flux
                flux = self._compute_interface_flux(owner_cell, neighbor_cell, face_normal)
                
                # Add flux contributions
                self.residuals[owner_cell] += face_area * flux
                self.residuals[neighbor_cell] -= face_area * flux
            
            else:  # Boundary face
                # Get boundary flux
                flux = self._compute_boundary_flux(owner_cell, face_id, face_normal)
                self.residuals[owner_cell] += face_area * flux
    
    def _compute_interface_flux(self, 
                              left_cell: int, 
                              right_cell: int,
                              normal: np.ndarray) -> np.ndarray:
        """Compute flux at internal face interface using advanced Riemann solvers."""
        # Get primitive states
        left_primitive = self.primitives[left_cell, :5]  # [rho, u, v, w, p]
        right_primitive = self.primitives[right_cell, :5]
        
        # Apply high-order reconstruction if available
        if self.reconstructor is not None:
            left_primitive, right_primitive = self._reconstruct_interface_states(
                left_cell, right_cell, normal
            )
        
        # Solve Riemann problem
        flux, max_eigenvalue = self.riemann_solver_manager.solve(
            left_primitive, right_primitive, normal
        )
        
        # Store maximum eigenvalue for time step computation
        self._update_max_eigenvalue(left_cell, right_cell, max_eigenvalue)
        
        return flux
    
    def _reconstruct_interface_states(self, 
                                    left_cell: int, 
                                    right_cell: int,
                                    normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct primitive states at interface using high-order schemes."""
        # Get cell centers and face center
        left_center = self.mesh.cell_data.centroids[left_cell]
        right_center = self.mesh.cell_data.centroids[right_cell]
        interface_point = 0.5 * (left_center + right_center)
        
        # Gather stencil points for reconstruction
        left_stencil = self._get_reconstruction_stencil(left_cell)
        right_stencil = self._get_reconstruction_stencil(right_cell)
        
        # Reconstruct left state
        if len(left_stencil['values']) > 2:
            left_reconstructed, _ = self.reconstructor.reconstruct_all_variables(
                left_stencil['values'], left_stencil['points'], interface_point
            )
        else:
            left_reconstructed = self.primitives[left_cell, :5]
        
        # Reconstruct right state
        if len(right_stencil['values']) > 2:
            _, right_reconstructed = self.reconstructor.reconstruct_all_variables(
                right_stencil['values'], right_stencil['points'], interface_point
            )
        else:
            right_reconstructed = self.primitives[right_cell, :5]
        
        return left_reconstructed, right_reconstructed
    
    def _get_reconstruction_stencil(self, cell_id: int) -> Dict[str, np.ndarray]:
        """Get reconstruction stencil for a cell."""
        neighbors = self.connectivity.get_cell_neighbors(cell_id)
        
        # Include the cell itself and its neighbors
        stencil_cells = [cell_id] + neighbors
        stencil_values = self.primitives[stencil_cells, :5]
        stencil_points = self.mesh.cell_data.centroids[stencil_cells]
        
        return {
            'values': stencil_values,
            'points': stencil_points,
            'cells': stencil_cells
        }
    
    def _update_max_eigenvalue(self, left_cell: int, right_cell: int, max_eigenvalue: float) -> None:
        """Update maximum eigenvalue for time step computation."""
        if not hasattr(self, 'max_eigenvalues'):
            self.max_eigenvalues = np.zeros(self.n_cells)
        
        self.max_eigenvalues[left_cell] = max(self.max_eigenvalues[left_cell], max_eigenvalue)
        self.max_eigenvalues[right_cell] = max(self.max_eigenvalues[right_cell], max_eigenvalue)
    
    def _reconstruct_state(self, 
                         left_cell: int, 
                         right_cell: int,
                         side: str) -> np.ndarray:
        """Reconstruct high-order state at face."""
        # Simplified linear reconstruction
        if side == "left":
            cell_id = left_cell
            neighbor_id = right_cell
        else:
            cell_id = right_cell
            neighbor_id = left_cell
        
        # Get cell centers and face center
        cell_center = self.mesh.cell_data.centroids[cell_id]
        neighbor_center = self.mesh.cell_data.centroids[neighbor_id]
        face_center = 0.5 * (cell_center + neighbor_center)
        
        # Distance from cell center to face
        dr = face_center - cell_center
        
        # Linear reconstruction
        cell_state = self.conservatives[cell_id]
        cell_gradient = self.gradients[cell_id]
        
        # Reconstruct state
        reconstructed_state = cell_state.copy()
        for var in range(self.n_variables):
            reconstructed_state[var] += np.dot(cell_gradient[var], dr)
        
        # Apply limiter
        if self.config.limiter != "none":
            neighbor_state = self.conservatives[neighbor_id]
            reconstructed_state = self._apply_limiter(
                cell_state, reconstructed_state, neighbor_state
            )
        
        return reconstructed_state
    
    def _apply_limiter(self, 
                      cell_state: np.ndarray,
                      reconstructed_state: np.ndarray,
                      neighbor_state: np.ndarray) -> np.ndarray:
        """Apply flux limiter to prevent oscillations."""
        limited_state = cell_state.copy()
        
        for var in range(self.n_variables):
            # Compute variations
            delta_plus = neighbor_state[var] - cell_state[var]
            delta_minus = reconstructed_state[var] - cell_state[var]
            
            if abs(delta_plus) < 1e-12:
                limiter = 0.0
            else:
                r = delta_minus / delta_plus
                
                # Apply limiter function
                if self.config.limiter == "minmod":
                    limiter = max(0, min(1, r))
                elif self.config.limiter == "superbee":
                    limiter = max(0, min(2*r, 1), min(r, 2))
                elif self.config.limiter == "van_leer":
                    limiter = (r + abs(r)) / (1 + abs(r))
                else:
                    limiter = 1.0
            
            # Apply limiting
            limited_state[var] = cell_state[var] + limiter * delta_minus
        
        return limited_state
    
    def _roe_flux(self, U_L: np.ndarray, U_R: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Compute Roe approximate Riemann solver flux."""
        # Convert to primitive variables
        W_L = self.variable_converter.conservatives_to_primitives(U_L)
        W_R = self.variable_converter.conservatives_to_primitives(U_R)
        
        rho_L, u_L, v_L, w_L, p_L, T_L = W_L
        rho_R, u_R, v_R, w_R, p_R, T_R = W_R
        
        # Roe averaged quantities
        sqrt_rho_L = np.sqrt(rho_L)
        sqrt_rho_R = np.sqrt(rho_R)
        rho_roe = sqrt_rho_L * sqrt_rho_R
        
        u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / (sqrt_rho_L + sqrt_rho_R)
        v_roe = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) / (sqrt_rho_L + sqrt_rho_R)
        w_roe = (sqrt_rho_L * w_L + sqrt_rho_R * w_R) / (sqrt_rho_L + sqrt_rho_R)
        
        H_L = self.equation_of_state.total_enthalpy(rho_L, p_L, u_L**2 + v_L**2 + w_L**2)
        H_R = self.equation_of_state.total_enthalpy(rho_R, p_R, u_R**2 + v_R**2 + w_R**2)
        H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / (sqrt_rho_L + sqrt_rho_R)
        
        # Roe-averaged speed of sound
        a_roe_sq = (self.config.gamma - 1) * (H_roe - 0.5 * (u_roe**2 + v_roe**2 + w_roe**2))
        a_roe = np.sqrt(max(a_roe_sq, 1e-10))
        
        # Normal velocity
        vn_roe = u_roe * normal[0] + v_roe * normal[1] + w_roe * normal[2]
        
        # Eigenvalues
        lambda1 = vn_roe - a_roe
        lambda2 = vn_roe
        lambda3 = vn_roe
        lambda4 = vn_roe
        lambda5 = vn_roe + a_roe
        
        # Entropy fix
        epsilon = 0.1 * a_roe
        if abs(lambda1) < epsilon:
            lambda1 = 0.5 * (lambda1**2 / epsilon + epsilon)
        if abs(lambda5) < epsilon:
            lambda5 = 0.5 * (lambda5**2 / epsilon + epsilon)
        
        # Flux computation (simplified Roe flux)
        F_L = self.inviscid_flux.compute_flux_normal(U_L, normal)
        F_R = self.inviscid_flux.compute_flux_normal(U_R, normal)
        
        # Average flux
        flux = 0.5 * (F_L + F_R)
        
        # Add Roe dissipation (simplified)
        dissipation = 0.5 * max(abs(lambda1), abs(lambda5)) * (U_R - U_L)
        flux -= dissipation
        
        return flux
    
    def _hllc_flux(self, U_L: np.ndarray, U_R: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Compute HLLC (Harten-Lax-van Leer-Contact) flux."""
        # Convert to primitive variables
        W_L = self.variable_converter.conservatives_to_primitives(U_L)
        W_R = self.variable_converter.conservatives_to_primitives(U_R)
        
        rho_L, u_L, v_L, w_L, p_L, T_L = W_L
        rho_R, u_R, v_R, w_R, p_R, T_R = W_R
        
        # Normal velocities
        vn_L = u_L * normal[0] + v_L * normal[1] + w_L * normal[2]
        vn_R = u_R * normal[0] + v_R * normal[1] + w_R * normal[2]
        
        # Speed of sound
        a_L = np.sqrt(self.config.gamma * p_L / rho_L)
        a_R = np.sqrt(self.config.gamma * p_R / rho_R)
        
        # Wave speeds (simplified estimates)
        S_L = min(vn_L - a_L, vn_R - a_R)
        S_R = max(vn_L + a_L, vn_R + a_R)
        
        # Contact wave speed
        S_star = (p_R - p_L + rho_L * vn_L * (S_L - vn_L) - rho_R * vn_R * (S_R - vn_R)) / \
                 (rho_L * (S_L - vn_L) - rho_R * (S_R - vn_R))
        
        # Compute fluxes
        F_L = self.inviscid_flux.compute_flux_normal(U_L, normal)
        F_R = self.inviscid_flux.compute_flux_normal(U_R, normal)
        
        # HLLC flux
        if S_L >= 0:
            return F_L
        elif S_R <= 0:
            return F_R
        elif S_L < 0 < S_star:
            # Left star state
            U_star_L = rho_L * (S_L - vn_L) / (S_L - S_star) * np.array([
                1,
                (S_star * normal[0] + (u_L - vn_L * normal[0])),
                (S_star * normal[1] + (v_L - vn_L * normal[1])),
                (S_star * normal[2] + (w_L - vn_L * normal[2])),
                (U_L[4] / rho_L + (S_star - vn_L) * 
                 (S_star + p_L / (rho_L * (S_L - vn_L))))
            ])
            return F_L + S_L * (U_star_L - U_L)
        else:  # S_star < 0 < S_R
            # Right star state  
            U_star_R = rho_R * (S_R - vn_R) / (S_R - S_star) * np.array([
                1,
                (S_star * normal[0] + (u_R - vn_R * normal[0])),
                (S_star * normal[1] + (v_R - vn_R * normal[1])),
                (S_star * normal[2] + (w_R - vn_R * normal[2])),
                (U_R[4] / rho_R + (S_star - vn_R) * 
                 (S_star + p_R / (rho_R * (S_R - vn_R))))
            ])
            return F_R + S_R * (U_star_R - U_R)
    
    def _rusanov_flux(self, U_L: np.ndarray, U_R: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Compute Rusanov (Local Lax-Friedrichs) flux."""
        # Convert to primitive variables
        W_L = self.variable_converter.conservatives_to_primitives(U_L)
        W_R = self.variable_converter.conservatives_to_primitives(U_R)
        
        rho_L, u_L, v_L, w_L, p_L, T_L = W_L
        rho_R, u_R, v_R, w_R, p_R, T_R = W_R
        
        # Normal velocities
        vn_L = u_L * normal[0] + v_L * normal[1] + w_L * normal[2]
        vn_R = u_R * normal[0] + v_R * normal[1] + w_R * normal[2]
        
        # Speed of sound
        a_L = np.sqrt(self.config.gamma * p_L / rho_L)
        a_R = np.sqrt(self.config.gamma * p_R / rho_R)
        
        # Maximum wave speed
        lambda_max = max(abs(vn_L) + a_L, abs(vn_R) + a_R)
        
        # Compute fluxes
        F_L = self.inviscid_flux.compute_flux_normal(U_L, normal)
        F_R = self.inviscid_flux.compute_flux_normal(U_R, normal)
        
        # Rusanov flux
        flux = 0.5 * (F_L + F_R) - 0.5 * lambda_max * (U_R - U_L)
        
        return flux
    
    def _central_flux(self, U_L: np.ndarray, U_R: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Compute central difference flux."""
        F_L = self.inviscid_flux.compute_flux_normal(U_L, normal)
        F_R = self.inviscid_flux.compute_flux_normal(U_R, normal)
        
        return 0.5 * (F_L + F_R)
    
    def _compute_boundary_flux(self, 
                             owner_cell: int,
                             face_id: int,
                             normal: np.ndarray) -> np.ndarray:
        """Compute flux at boundary face."""
        # Get ghost state from boundary manager
        ghost_state = self.boundary_manager.get_ghost_state(face_id)
        
        if ghost_state is not None:
            # Use ghost state
            U_ghost = ghost_state.conservatives
            U_interior = self.conservatives[owner_cell]
            
            # Compute flux using selected scheme
            return self._compute_interface_flux_direct(U_interior, U_ghost, normal)
        else:
            # Simple extrapolation
            U_interior = self.conservatives[owner_cell]
            return self.inviscid_flux.compute_flux_normal(U_interior, normal)
    
    def _compute_interface_flux_direct(self, U_L: np.ndarray, U_R: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Direct interface flux computation without reconstruction."""
        if self.config.flux_scheme == "roe":
            return self._roe_flux(U_L, U_R, normal)
        elif self.config.flux_scheme == "hllc":
            return self._hllc_flux(U_L, U_R, normal)
        elif self.config.flux_scheme == "rusanov":
            return self._rusanov_flux(U_L, U_R, normal)
        else:
            return self._central_flux(U_L, U_R, normal)
    
    def _explicit_time_step(self) -> None:
        """Explicit time integration (forward Euler)."""
        for cell_id in range(self.n_cells):
            cell_volume = self.mesh.cell_data.volumes[cell_id]
            dt = self.time_steps[cell_id]
            
            # Update conservative variables
            self.conservatives[cell_id] -= (dt / cell_volume) * self.residuals[cell_id]
            
            # Check for unphysical states
            valid, _ = self.variable_converter.validate_state(conservatives=self.conservatives[cell_id])
            if not valid:
                self.conservatives[cell_id] = self.variable_converter.fix_unphysical_state(
                    self.conservatives[cell_id]
                )
        
        # Update time
        min_dt = np.min(self.time_steps)
        self.current_time += min_dt
    
    def _implicit_time_step(self) -> None:
        """Implicit time integration (simplified)."""
        # For now, use explicit with smaller time step
        # Full implicit would require Jacobian computation and linear solver
        logger.warning("Implicit time stepping not fully implemented, using explicit")
        self._explicit_time_step()
    
    def _compute_time_steps(self) -> None:
        """Compute local time steps based on CFL condition."""
        for cell_id in range(self.n_cells):
            # Get cell properties
            primitives = self.primitives[cell_id]
            rho, u, v, w, p, T = primitives
            
            # Velocity magnitude and speed of sound
            velocity_mag = np.sqrt(u**2 + v**2 + w**2)
            speed_of_sound = np.sqrt(self.config.gamma * p / rho)
            
            # Characteristic speed
            char_speed = velocity_mag + speed_of_sound
            
            # Cell size estimate
            cell_volume = self.mesh.cell_data.volumes[cell_id]
            cell_size = cell_volume**(1/3)  # Rough estimate
            
            # CFL-based time step
            if char_speed > 1e-12:
                dt_cfl = self.config.cfl_number * cell_size / char_speed
            else:
                dt_cfl = 1e-6  # Small default
            
            self.time_steps[cell_id] = dt_cfl
    
    def _update_primitive_variables(self) -> None:
        """Update primitive variables from conservative variables."""
        for cell_id in range(self.n_cells):
            self.primitives[cell_id] = self.variable_converter.conservatives_to_primitives(
                self.conservatives[cell_id]
            )
    
    def _compute_residual_norm(self) -> float:
        """Compute L2 norm of residuals."""
        residual_norm = 0.0
        total_volume = 0.0
        
        for cell_id in range(self.n_cells):
            cell_volume = self.mesh.cell_data.volumes[cell_id]
            residual_contribution = np.linalg.norm(self.residuals[cell_id])**2
            residual_norm += cell_volume * residual_contribution
            total_volume += cell_volume
        
        return np.sqrt(residual_norm / total_volume)
    
    def _compute_solution_statistics(self) -> Dict[str, float]:
        """Compute solution statistics for monitoring."""
        # Extract primitive variables
        densities = self.primitives[:, 0]
        velocities = self.primitives[:, 1:4]
        pressures = self.primitives[:, 4]
        temperatures = self.primitives[:, 5]
        
        # Compute derived quantities
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        speeds_of_sound = np.sqrt(self.config.gamma * pressures / densities)
        mach_numbers = velocity_magnitudes / speeds_of_sound
        
        return {
            'min_density': np.min(densities),
            'max_density': np.max(densities),
            'min_pressure': np.min(pressures),
            'max_pressure': np.max(pressures),
            'min_temperature': np.min(temperatures),
            'max_temperature': np.max(temperatures),
            'max_mach_number': np.max(mach_numbers),
            'mean_mach_number': np.mean(mach_numbers),
            'min_velocity': np.min(velocity_magnitudes),
            'max_velocity': np.max(velocity_magnitudes)
        }
    
    def get_solution(self) -> Dict[str, np.ndarray]:
        """Get current solution arrays."""
        return {
            'conservatives': self.conservatives.copy(),
            'primitives': self.primitives.copy(),
            'residuals': self.residuals.copy(),
            'gradients': self.gradients.copy() if self.gradients is not None else None,
            'time_steps': self.time_steps.copy()
        }
    
    def set_boundary_conditions(self, boundary_patches: Dict) -> None:
        """Set boundary conditions."""
        for patch_name, patch_config in boundary_patches.items():
            self.boundary_manager.add_patch(patch_name, **patch_config)
        
        # Validate boundary setup
        self.boundary_manager.validate_boundary_setup()
    
    def save_solution(self, filename: str) -> None:
        """Save solution to file."""
        if self.mesh and hasattr(self.mesh, 'save_to_vtk'):
            # Add solution data to mesh
            self.mesh.cell_data.conservatives = self.conservatives
            self.mesh.cell_data.primitives = self.primitives
            
            # Save with solution data
            self.mesh.save_to_vtk(filename)
            logger.info(f"Solution saved to {filename}")
        else:
            logger.warning("Cannot save solution: mesh does not support VTK output")