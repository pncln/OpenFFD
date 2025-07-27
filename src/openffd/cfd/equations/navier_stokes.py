"""
3D Navier-Stokes Equations Solver for Viscous Supersonic Flows

Extends the Euler solver to include viscous effects with:
- Viscous stress tensor computation
- Heat conduction terms
- Turbulence modeling capability
- Wall boundary layer treatment
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import time

from .euler_equations import EulerEquations3D, EulerSolverConfig
from .primitive_conservative import VariableConverter
from .equation_state import PerfectGas, AIR_PROPERTIES
from .flux_functions import ViscousFlux, FluxCalculator
from ..mesh.unstructured_mesh import UnstructuredMesh3D
from ..mesh.connectivity import ConnectivityManager
from ..mesh.boundary import BoundaryManager, BCType

logger = logging.getLogger(__name__)


@dataclass
class NavierStokesSolverConfig(EulerSolverConfig):
    """Configuration for Navier-Stokes equations solver."""
    # Viscous terms
    reynolds_number: float = 1e6
    reference_length: float = 1.0
    reference_velocity: float = 100.0
    
    # Transport properties
    prandtl_number: float = 0.72
    sutherland_temperature: float = 110.4  # K
    reference_viscosity: float = 1.716e-5  # Pa·s
    reference_temperature: float = 288.15  # K
    
    # Turbulence modeling
    turbulence_model: str = "laminar"  # "laminar", "spalart_allmaras", "k_epsilon", "k_omega_sst"
    wall_treatment: str = "low_reynolds"  # "low_reynolds", "wall_functions"
    
    # Viscous discretization
    viscous_scheme: str = "central"  # "central", "compact"
    gradient_reconstruction: str = "least_squares"  # "least_squares", "green_gauss"
    
    # Solver options
    viscous_cfl_factor: float = 0.5  # Additional CFL reduction for viscous terms


class NavierStokesEquations3D(EulerEquations3D):
    """
    Three-dimensional Navier-Stokes equations solver.
    
    Solves the compressible Navier-Stokes equations:
    ∂U/∂t + ∇ · F_inv(U) = ∇ · F_vis(U, ∇U)
    
    Where:
    - U = [ρ, ρu, ρv, ρw, ρE]ᵀ is the conservative variable vector
    - F_inv(U) is the inviscid flux tensor (Euler terms)
    - F_vis(U, ∇U) is the viscous flux tensor
    
    Features:
    - All Euler solver capabilities
    - Viscous stress tensor computation
    - Heat conduction with Fourier's law
    - Multiple turbulence models
    - Wall boundary layer treatment
    - Temperature-dependent transport properties
    """
    
    def __init__(self, 
                 mesh: UnstructuredMesh3D,
                 config: Optional[NavierStokesSolverConfig] = None):
        """
        Initialize Navier-Stokes equations solver.
        
        Args:
            mesh: Unstructured mesh
            config: Solver configuration
        """
        # Initialize base Euler solver
        self.ns_config = config or NavierStokesSolverConfig()
        super().__init__(mesh, self.ns_config)
        
        # Override flux calculator for viscous terms
        self.flux_calculator = FluxCalculator(
            gamma=self.config.gamma,
            prandtl_number=self.ns_config.prandtl_number,
            R=self.config.gas_constant
        )
        
        # Viscous flux computation
        self.viscous_flux = ViscousFlux(
            gamma=self.config.gamma,
            prandtl_number=self.ns_config.prandtl_number,
            R=self.config.gas_constant
        )
        
        # Gradient computation for viscous terms
        self.velocity_gradients = np.zeros((self.n_cells, 9))  # [∂u/∂x, ∂u/∂y, ∂u/∂z, ∂v/∂x, ...]
        self.temperature_gradients = np.zeros((self.n_cells, 3))  # [∂T/∂x, ∂T/∂y, ∂T/∂z]
        
        # Transport properties
        self.viscosity = np.zeros(self.n_cells)
        self.thermal_conductivity = np.zeros(self.n_cells)
        
        # Turbulence variables (if using turbulence models)
        self.turbulence_variables = {}
        self._setup_turbulence_model()
        
        # Wall distance for turbulence models
        self.wall_distance = np.zeros(self.n_cells)
        
        # Performance tracking for viscous terms
        self.timing.update({
            'viscous_gradients': 0.0,
            'transport_properties': 0.0,
            'viscous_fluxes': 0.0,
            'turbulence_model': 0.0
        })
        
        logger.info(f"Initialized Navier-Stokes solver: Re={self.ns_config.reynolds_number:.1e}, "
                   f"turbulence={self.ns_config.turbulence_model}")
    
    def _setup_turbulence_model(self) -> None:
        """Setup turbulence model variables."""
        if self.ns_config.turbulence_model == "spalart_allmaras":
            # Spalart-Allmaras model: one additional equation for modified viscosity
            self.turbulence_variables['nu_tilde'] = np.zeros(self.n_cells)
            self.turbulence_variables['nu_tilde_residual'] = np.zeros(self.n_cells)
            self.n_variables += 1  # Add turbulence equation
            
        elif self.ns_config.turbulence_model == "k_epsilon":
            # k-ε model: two additional equations
            self.turbulence_variables['k'] = np.zeros(self.n_cells)  # Turbulent kinetic energy
            self.turbulence_variables['epsilon'] = np.zeros(self.n_cells)  # Dissipation rate
            self.turbulence_variables['k_residual'] = np.zeros(self.n_cells)
            self.turbulence_variables['epsilon_residual'] = np.zeros(self.n_cells)
            self.n_variables += 2
            
        elif self.ns_config.turbulence_model == "k_omega_sst":
            # k-ω SST model: two additional equations
            self.turbulence_variables['k'] = np.zeros(self.n_cells)  # Turbulent kinetic energy
            self.turbulence_variables['omega'] = np.zeros(self.n_cells)  # Specific dissipation rate
            self.turbulence_variables['k_residual'] = np.zeros(self.n_cells)
            self.turbulence_variables['omega_residual'] = np.zeros(self.n_cells)
            self.n_variables += 2
        
        # Turbulent viscosity (computed from turbulence variables)
        self.turbulent_viscosity = np.zeros(self.n_cells)
        
        # Resize arrays if turbulence variables added
        if self.n_variables > 5:
            self.conservatives = np.zeros((self.n_cells, self.n_variables))
            self.residuals = np.zeros((self.n_cells, self.n_variables))
            self.gradients = np.zeros((self.n_cells, self.n_variables, 3))
    
    def initialize_solution(self) -> None:
        """Initialize flow field including turbulence variables."""
        # Call parent initialization
        super().initialize_solution()
        
        # Initialize turbulence variables
        if self.ns_config.turbulence_model == "spalart_allmaras":
            # Initialize modified viscosity to small fraction of molecular viscosity
            for cell_id in range(self.n_cells):
                molecular_viscosity = self._compute_molecular_viscosity(cell_id)
                self.turbulence_variables['nu_tilde'][cell_id] = 0.1 * molecular_viscosity / self.primitives[cell_id, 0]
                
        elif self.ns_config.turbulence_model in ["k_epsilon", "k_omega_sst"]:
            # Initialize k and ε/ω based on freestream turbulence intensity
            turbulence_intensity = 0.01  # 1% turbulence intensity
            
            for cell_id in range(self.n_cells):
                velocity = self.primitives[cell_id, 1:4]
                velocity_mag = np.linalg.norm(velocity)
                
                # Turbulent kinetic energy
                k_init = 1.5 * (turbulence_intensity * velocity_mag)**2
                self.turbulence_variables['k'][cell_id] = k_init
                
                if self.ns_config.turbulence_model == "k_epsilon":
                    # Dissipation rate
                    length_scale = 0.1 * self.ns_config.reference_length
                    epsilon_init = 0.09 * k_init**(3/2) / length_scale
                    self.turbulence_variables['epsilon'][cell_id] = epsilon_init
                else:  # k_omega_sst
                    # Specific dissipation rate
                    epsilon_init = 0.09 * k_init / (0.1 * self.ns_config.reference_length)
                    omega_init = epsilon_init / (0.09 * k_init)
                    self.turbulence_variables['omega'][cell_id] = omega_init
        
        # Compute initial transport properties
        self._compute_transport_properties()
        
        # Compute wall distances for turbulence models
        if self.ns_config.turbulence_model != "laminar":
            self._compute_wall_distances()
        
        logger.info("Navier-Stokes solution initialized with transport properties")
    
    def _time_step(self) -> None:
        """Perform single time step including viscous terms."""
        step_start = time.time()
        
        # Compute transport properties
        tp_start = time.time()
        self._compute_transport_properties()
        self.timing['transport_properties'] += time.time() - tp_start
        
        # Compute viscous gradients
        grad_start = time.time()
        self._compute_viscous_gradients()
        if self.config.limiter != "none":
            self._compute_gradients()  # Conservative variable gradients
        self.timing['viscous_gradients'] += time.time() - grad_start
        
        # Apply boundary conditions
        bc_start = time.time()
        self.boundary_manager.apply_boundary_conditions(
            self.mesh.cell_data, self.mesh.face_data, self.current_time
        )
        self.timing['boundary_conditions'] += time.time() - bc_start
        
        # Compute fluxes and residuals (including viscous)
        flux_start = time.time()
        self._compute_navier_stokes_residuals()
        self.timing['viscous_fluxes'] += time.time() - flux_start
        
        # Turbulence model source terms
        if self.ns_config.turbulence_model != "laminar":
            turb_start = time.time()
            self._compute_turbulence_residuals()
            self.timing['turbulence_model'] += time.time() - turb_start
        
        # Time integration with viscous CFL restriction
        ts_start = time.time()
        self._compute_viscous_time_steps()
        if self.config.time_stepping == "explicit":
            self._explicit_time_step()
        else:
            self._implicit_time_step()
        self.timing['time_stepping'] += time.time() - ts_start
        
        # Update primitive variables and turbulence quantities
        self._update_primitive_variables()
        if self.ns_config.turbulence_model != "laminar":
            self._update_turbulent_viscosity()
    
    def _compute_transport_properties(self) -> None:
        """Compute molecular viscosity and thermal conductivity."""
        for cell_id in range(self.n_cells):
            # Molecular viscosity using Sutherland's law
            self.viscosity[cell_id] = self._compute_molecular_viscosity(cell_id)
            
            # Thermal conductivity from Prandtl number
            cp = self.config.gamma * self.config.gas_constant / (self.config.gamma - 1)
            self.thermal_conductivity[cell_id] = (self.viscosity[cell_id] * cp / 
                                                 self.ns_config.prandtl_number)
    
    def _compute_molecular_viscosity(self, cell_id: int) -> float:
        """Compute molecular viscosity using Sutherland's law."""
        temperature = self.primitives[cell_id, 5]
        
        T_ref = self.ns_config.reference_temperature
        mu_ref = self.ns_config.reference_viscosity
        S = self.ns_config.sutherland_temperature
        
        # Sutherland's law
        mu = mu_ref * (temperature / T_ref)**(3/2) * (T_ref + S) / (temperature + S)
        
        return mu
    
    def _compute_viscous_gradients(self) -> None:
        """Compute velocity and temperature gradients for viscous fluxes."""
        if not self.connectivity._stencils_built:
            self.connectivity.build_gradient_stencils(self.ns_config.gradient_reconstruction)
        
        # Initialize gradient arrays
        self.velocity_gradients.fill(0.0)
        self.temperature_gradients.fill(0.0)
        
        for cell_id in range(self.n_cells):
            stencil = self.connectivity.get_gradient_stencil(cell_id)
            if not stencil:
                continue
            
            weights = self.connectivity.compute_stencil_weights(cell_id, self.ns_config.gradient_reconstruction)
            if weights is None:
                continue
            
            cell_center = self.mesh.cell_data.centroids[cell_id]
            cell_primitives = self.primitives[cell_id]
            
            # Velocity components
            u_cell, v_cell, w_cell = cell_primitives[1:4]
            T_cell = cell_primitives[5]
            
            # Initialize gradients
            velocity_grad = np.zeros((3, 3))  # [du/dx, du/dy, du/dz; dv/dx, ...]
            temperature_grad = np.zeros(3)
            
            # Least squares gradient computation
            for i, neighbor_id in enumerate(stencil):
                neighbor_primitives = self.primitives[neighbor_id]
                
                # Velocity differences
                du = neighbor_primitives[1] - u_cell
                dv = neighbor_primitives[2] - v_cell
                dw = neighbor_primitives[3] - w_cell
                dT = neighbor_primitives[5] - T_cell
                
                # Add weighted contributions
                velocity_grad[0] += du * weights[i]  # du/dx, du/dy, du/dz
                velocity_grad[1] += dv * weights[i]  # dv/dx, dv/dy, dv/dz
                velocity_grad[2] += dw * weights[i]  # dw/dx, dw/dy, dw/dz
                temperature_grad += dT * weights[i]  # dT/dx, dT/dy, dT/dz
            
            # Store gradients
            self.velocity_gradients[cell_id] = velocity_grad.flatten()
            self.temperature_gradients[cell_id] = temperature_grad
    
    def _compute_navier_stokes_residuals(self) -> None:
        """Compute residuals including both inviscid and viscous fluxes."""
        self.residuals.fill(0.0)
        
        # Loop over all faces
        for face_id in range(self.mesh.n_faces):
            owner_cell = self.mesh.face_data.owner[face_id]
            neighbor_cell = self.mesh.face_data.neighbor[face_id]
            
            face_area = self.mesh.face_data.areas[face_id]
            face_normal = self.mesh.face_data.normals[face_id]
            
            if neighbor_cell >= 0:  # Internal face
                # Compute total flux (inviscid + viscous)
                flux = self._compute_interface_flux_viscous(owner_cell, neighbor_cell, face_normal)
                
                # Add flux contributions (only for base 5 equations)
                self.residuals[owner_cell, :5] += face_area * flux
                self.residuals[neighbor_cell, :5] -= face_area * flux
            
            else:  # Boundary face
                # Boundary flux with viscous terms
                flux = self._compute_boundary_flux_viscous(owner_cell, face_id, face_normal)
                self.residuals[owner_cell, :5] += face_area * flux
    
    def _compute_interface_flux_viscous(self, 
                                       left_cell: int, 
                                       right_cell: int,
                                       normal: np.ndarray) -> np.ndarray:
        """Compute flux at internal face including viscous terms."""
        # Get cell states
        U_L = self.conservatives[left_cell, :5]
        U_R = self.conservatives[right_cell, :5]
        W_L = self.primitives[left_cell]
        W_R = self.primitives[right_cell]
        
        # Apply high-order reconstruction if available
        if self.config.limiter != "none" and self.gradients is not None:
            U_L = self._reconstruct_state(left_cell, right_cell, "left")
            U_R = self._reconstruct_state(left_cell, right_cell, "right")
            W_L = self.variable_converter.conservatives_to_primitives(U_L)
            W_R = self.variable_converter.conservatives_to_primitives(U_R)
        
        # Inviscid flux
        inviscid_flux = self._compute_interface_flux_direct(U_L, U_R, normal)
        
        # Viscous flux (averaged at interface)
        viscous_flux = self._compute_interface_viscous_flux(left_cell, right_cell, normal)
        
        # Total flux
        total_flux = inviscid_flux - viscous_flux  # Note: viscous flux is subtracted
        
        return total_flux
    
    def _compute_interface_viscous_flux(self, 
                                       left_cell: int,
                                       right_cell: int,
                                       normal: np.ndarray) -> np.ndarray:
        """Compute viscous flux at interface."""
        # Average primitive variables
        W_avg = 0.5 * (self.primitives[left_cell] + self.primitives[right_cell])
        
        # Average gradients
        vel_grad_avg = 0.5 * (self.velocity_gradients[left_cell] + self.velocity_gradients[right_cell])
        temp_grad_avg = 0.5 * (self.temperature_gradients[left_cell] + self.temperature_gradients[right_cell])
        
        # Combine gradients for viscous flux calculation
        combined_gradients = np.concatenate([vel_grad_avg, temp_grad_avg])
        
        # Average transport properties
        mu_avg = 0.5 * (self.viscosity[left_cell] + self.viscosity[right_cell])
        k_avg = 0.5 * (self.thermal_conductivity[left_cell] + self.thermal_conductivity[right_cell])
        
        # Add turbulent viscosity if applicable
        if self.ns_config.turbulence_model != "laminar":
            mu_t_avg = 0.5 * (self.turbulent_viscosity[left_cell] + self.turbulent_viscosity[right_cell])
            mu_total = mu_avg + mu_t_avg
            
            # Turbulent thermal conductivity
            Pr_t = 0.9  # Turbulent Prandtl number
            k_t_avg = mu_t_avg * self.viscous_flux.cp / Pr_t
            k_total = k_avg + k_t_avg
        else:
            mu_total = mu_avg
            k_total = k_avg
        
        # Compute viscous flux
        viscous_flux = self.viscous_flux.compute_viscous_flux_normal(
            W_avg, combined_gradients, normal, mu_total, k_total
        )
        
        return viscous_flux
    
    def _compute_boundary_flux_viscous(self, 
                                      owner_cell: int,
                                      face_id: int,
                                      normal: np.ndarray) -> np.ndarray:
        """Compute boundary flux including viscous terms."""
        # Get boundary state
        ghost_state = self.boundary_manager.get_ghost_state(face_id)
        
        if ghost_state is not None:
            U_ghost = ghost_state.conservatives[:5]
            U_interior = self.conservatives[owner_cell, :5]
            W_ghost = ghost_state.primitives
            W_interior = self.primitives[owner_cell]
            
            # Inviscid flux
            inviscid_flux = self._compute_interface_flux_direct(U_interior, U_ghost, normal)
            
            # Viscous flux for boundary
            viscous_flux = self._compute_boundary_viscous_flux(owner_cell, face_id, normal, W_ghost)
            
            return inviscid_flux - viscous_flux
        else:
            # Simple extrapolation
            U_interior = self.conservatives[owner_cell, :5]
            return self.inviscid_flux.compute_flux_normal(U_interior, normal)
    
    def _compute_boundary_viscous_flux(self, 
                                      owner_cell: int,
                                      face_id: int,
                                      normal: np.ndarray,
                                      ghost_primitives: np.ndarray) -> np.ndarray:
        """Compute viscous flux at boundary."""
        # Use interior gradients and properties
        interior_gradients = np.concatenate([
            self.velocity_gradients[owner_cell],
            self.temperature_gradients[owner_cell]
        ])
        
        mu = self.viscosity[owner_cell]
        k = self.thermal_conductivity[owner_cell]
        
        # Add turbulent viscosity if applicable
        if self.ns_config.turbulence_model != "laminar":
            mu += self.turbulent_viscosity[owner_cell]
            k += self.turbulent_viscosity[owner_cell] * self.viscous_flux.cp / 0.9
        
        # Use interior primitive variables for viscous flux
        # (boundary conditions already applied to ghost state)
        W_interior = self.primitives[owner_cell]
        
        viscous_flux = self.viscous_flux.compute_viscous_flux_normal(
            W_interior, interior_gradients, normal, mu, k
        )
        
        return viscous_flux
    
    def _compute_turbulence_residuals(self) -> None:
        """Compute residuals for turbulence model equations."""
        if self.ns_config.turbulence_model == "spalart_allmaras":
            self._compute_spalart_allmaras_residuals()
        elif self.ns_config.turbulence_model == "k_epsilon":
            self._compute_k_epsilon_residuals()
        elif self.ns_config.turbulence_model == "k_omega_sst":
            self._compute_k_omega_sst_residuals()
    
    def _compute_spalart_allmaras_residuals(self) -> None:
        """Compute Spalart-Allmaras turbulence model residuals."""
        # Simplified implementation - full model requires extensive coding
        for cell_id in range(self.n_cells):
            # Get flow variables
            rho = self.primitives[cell_id, 0]
            velocity = self.primitives[cell_id, 1:4]
            nu_tilde = self.turbulence_variables['nu_tilde'][cell_id]
            
            # Molecular viscosity
            mu = self.viscosity[cell_id]
            nu = mu / rho
            
            # Strain rate magnitude (simplified)
            vel_grad = self.velocity_gradients[cell_id].reshape(3, 3)
            strain_rate = 0.5 * (vel_grad + vel_grad.T)
            S = np.sqrt(2 * np.sum(strain_rate**2))
            
            # Wall distance
            d = max(self.wall_distance[cell_id], 1e-10)
            
            # Spalart-Allmaras model constants
            cv1 = 7.1
            kappa = 0.41
            cb1 = 0.1355
            cb2 = 0.622
            sigma = 2/3
            cw1 = cb1 / kappa**2 + (1 + cb2) / sigma
            
            # Production term
            chi = nu_tilde / nu
            fv1 = chi**3 / (chi**3 + cv1**3)
            
            S_tilde = S + nu_tilde / (kappa**2 * d**2) * fv1
            P = cb1 * S_tilde * nu_tilde
            
            # Destruction term (simplified)
            r = min(nu_tilde / (S_tilde * kappa**2 * d**2), 10.0)
            g = r + cw1 * (r**6 - r)
            fw = g * ((1 + cw1**6) / (g**6 + cw1**6))**(1/6)
            D = cw1 * fw * (nu_tilde / d)**2
            
            # Source term
            source = rho * (P - D)
            
            # Store in residual (simplified - should include convection and diffusion)
            self.turbulence_variables['nu_tilde_residual'][cell_id] = source
    
    def _compute_k_epsilon_residuals(self) -> None:
        """Compute k-ε turbulence model residuals."""
        # Simplified implementation
        for cell_id in range(self.n_cells):
            rho = self.primitives[cell_id, 0]
            k = self.turbulence_variables['k'][cell_id]
            epsilon = self.turbulence_variables['epsilon'][cell_id]
            
            # Model constants
            Cmu = 0.09
            C1 = 1.44
            C2 = 1.92
            sigma_k = 1.0
            sigma_epsilon = 1.3
            
            # Production (simplified)
            vel_grad = self.velocity_gradients[cell_id].reshape(3, 3)
            strain_rate = 0.5 * (vel_grad + vel_grad.T)
            P_k = 2 * self.turbulent_viscosity[cell_id] * np.sum(strain_rate**2)
            
            # k equation source
            k_source = P_k - rho * epsilon
            
            # ε equation source  
            epsilon_source = C1 * epsilon / k * P_k - C2 * rho * epsilon**2 / k
            
            self.turbulence_variables['k_residual'][cell_id] = k_source
            self.turbulence_variables['epsilon_residual'][cell_id] = epsilon_source
    
    def _compute_k_omega_sst_residuals(self) -> None:
        """Compute k-ω SST turbulence model residuals."""
        # Simplified implementation
        for cell_id in range(self.n_cells):
            rho = self.primitives[cell_id, 0]
            k = self.turbulence_variables['k'][cell_id]
            omega = self.turbulence_variables['omega'][cell_id]
            
            # Model constants
            beta_star = 0.09
            gamma = 0.52
            beta = 0.075
            
            # Production (simplified)
            vel_grad = self.velocity_gradients[cell_id].reshape(3, 3)
            strain_rate = 0.5 * (vel_grad + vel_grad.T)
            P_k = 2 * self.turbulent_viscosity[cell_id] * np.sum(strain_rate**2)
            
            # k equation source
            k_source = P_k - beta_star * rho * k * omega
            
            # ω equation source
            omega_source = gamma * rho / self.turbulent_viscosity[cell_id] * P_k - beta * rho * omega**2
            
            self.turbulence_variables['k_residual'][cell_id] = k_source
            self.turbulence_variables['omega_residual'][cell_id] = omega_source
    
    def _update_turbulent_viscosity(self) -> None:
        """Update turbulent viscosity from turbulence variables."""
        if self.ns_config.turbulence_model == "spalart_allmaras":
            for cell_id in range(self.n_cells):
                rho = self.primitives[cell_id, 0]
                nu_tilde = self.turbulence_variables['nu_tilde'][cell_id]
                mu = self.viscosity[cell_id]
                nu = mu / rho
                
                # Viscosity function
                cv1 = 7.1
                chi = nu_tilde / nu
                fv1 = chi**3 / (chi**3 + cv1**3)
                
                self.turbulent_viscosity[cell_id] = rho * nu_tilde * fv1
                
        elif self.ns_config.turbulence_model == "k_epsilon":
            for cell_id in range(self.n_cells):
                rho = self.primitives[cell_id, 0]
                k = max(self.turbulence_variables['k'][cell_id], 1e-10)
                epsilon = max(self.turbulence_variables['epsilon'][cell_id], 1e-10)
                
                Cmu = 0.09
                self.turbulent_viscosity[cell_id] = rho * Cmu * k**2 / epsilon
                
        elif self.ns_config.turbulence_model == "k_omega_sst":
            for cell_id in range(self.n_cells):
                rho = self.primitives[cell_id, 0]
                k = max(self.turbulence_variables['k'][cell_id], 1e-10)
                omega = max(self.turbulence_variables['omega'][cell_id], 1e-10)
                
                self.turbulent_viscosity[cell_id] = rho * k / omega
    
    def _compute_wall_distances(self) -> None:
        """Compute wall distances for turbulence models."""
        # Simplified wall distance computation
        # In practice, this requires solving an Eikonal equation or using geometric algorithms
        
        # Find wall boundary faces
        wall_faces = []
        for patch_name, patch in self.boundary_manager.patches.items():
            if patch.bc_type in [BCType.VISCOUS_WALL, BCType.INVISCID_WALL]:
                wall_faces.extend(patch.face_ids)
        
        if not wall_faces:
            self.wall_distance.fill(1.0)  # Default value
            return
        
        # Compute distance to nearest wall face for each cell
        for cell_id in range(self.n_cells):
            cell_center = self.mesh.cell_data.centroids[cell_id]
            min_distance = float('inf')
            
            for face_id in wall_faces:
                face_center = self.mesh.face_data.centers[face_id]
                distance = np.linalg.norm(cell_center - face_center)
                min_distance = min(min_distance, distance)
            
            self.wall_distance[cell_id] = min_distance
    
    def _compute_viscous_time_steps(self) -> None:
        """Compute time steps with viscous stability restriction."""
        # Call parent method for inviscid CFL
        self._compute_time_steps()
        
        # Apply additional viscous CFL restriction
        for cell_id in range(self.n_cells):
            # Get cell properties
            rho = self.primitives[cell_id, 0]
            mu_total = self.viscosity[cell_id]
            
            if self.ns_config.turbulence_model != "laminar":
                mu_total += self.turbulent_viscosity[cell_id]
            
            # Cell size estimate
            cell_volume = self.mesh.cell_data.volumes[cell_id]
            cell_size = cell_volume**(1/3)
            
            # Viscous time step restriction
            if mu_total > 1e-12:
                dt_viscous = self.ns_config.viscous_cfl_factor * rho * cell_size**2 / mu_total
            else:
                dt_viscous = float('inf')
            
            # Take minimum of inviscid and viscous time steps
            self.time_steps[cell_id] = min(self.time_steps[cell_id], dt_viscous)
    
    def get_solution(self) -> Dict[str, np.ndarray]:
        """Get current solution including turbulence variables."""
        solution = super().get_solution()
        
        # Add viscous-specific variables
        solution.update({
            'velocity_gradients': self.velocity_gradients.copy(),
            'temperature_gradients': self.temperature_gradients.copy(),
            'viscosity': self.viscosity.copy(),
            'thermal_conductivity': self.thermal_conductivity.copy(),
            'turbulent_viscosity': self.turbulent_viscosity.copy(),
            'wall_distance': self.wall_distance.copy()
        })
        
        # Add turbulence variables
        for var_name, var_data in self.turbulence_variables.items():
            solution[f'turbulence_{var_name}'] = var_data.copy()
        
        return solution
    
    def _compute_solution_statistics(self) -> Dict[str, float]:
        """Compute solution statistics including viscous quantities."""
        stats = super()._compute_solution_statistics()
        
        # Add viscous statistics
        stats.update({
            'min_viscosity': np.min(self.viscosity),
            'max_viscosity': np.max(self.viscosity),
            'mean_viscosity': np.mean(self.viscosity),
            'min_turbulent_viscosity': np.min(self.turbulent_viscosity),
            'max_turbulent_viscosity': np.max(self.turbulent_viscosity),
            'max_turbulent_ratio': np.max(self.turbulent_viscosity / (self.viscosity + 1e-12))
        })
        
        # Add turbulence statistics
        if 'k' in self.turbulence_variables:
            k_values = self.turbulence_variables['k']
            stats.update({
                'min_k': np.min(k_values),
                'max_k': np.max(k_values),
                'mean_k': np.mean(k_values)
            })
        
        return stats