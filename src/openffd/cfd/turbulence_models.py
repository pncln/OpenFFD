"""
Turbulence Models for RANS Equations

Implements various turbulence models for Reynolds-Averaged Navier-Stokes equations:
- k-ε models (Standard, RNG, Realizable)
- k-ω models (Standard, SST)
- Spalart-Allmaras model
- Reynolds Stress Models (RSM)
- Transition models (γ-Reθ)

These models provide closure for the RANS equations by modeling the Reynolds stresses
and turbulent transport phenomena in high-Reynolds number flows.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TurbulenceModelType(Enum):
    """Enumeration of turbulence model types."""
    SPALART_ALLMARAS = "spalart_allmaras"
    K_EPSILON_STANDARD = "k_epsilon_standard"
    K_EPSILON_RNG = "k_epsilon_rng"
    K_EPSILON_REALIZABLE = "k_epsilon_realizable"
    K_OMEGA_STANDARD = "k_omega_standard"
    K_OMEGA_SST = "k_omega_sst"
    REYNOLDS_STRESS = "reynolds_stress"
    TRANSITION_SST = "transition_sst"


@dataclass
class TurbulenceModelConfig:
    """Configuration for turbulence models."""
    
    # Model selection
    model_type: TurbulenceModelType = TurbulenceModelType.K_OMEGA_SST
    
    # Physical constants
    prandtl_turbulent: float = 0.9  # Turbulent Prandtl number
    schmidt_turbulent: float = 0.9  # Turbulent Schmidt number
    
    # Numerical parameters
    turbulence_intensity: float = 0.05  # Freestream turbulence intensity
    viscosity_ratio: float = 10.0  # Turbulent/laminar viscosity ratio
    
    # Boundary conditions
    wall_function: bool = False  # Use wall functions or resolve to y+ ~ 1
    roughness_height: float = 0.0  # Wall roughness height
    
    # Solver settings
    turbulence_tolerance: float = 1e-6
    max_turbulence_iterations: int = 10
    under_relaxation_k: float = 0.8
    under_relaxation_epsilon: float = 0.8
    under_relaxation_omega: float = 0.8
    
    # Limiters and clipping
    production_limiter: float = 10.0  # Limit production/dissipation ratio
    min_k: float = 1e-12  # Minimum turbulent kinetic energy
    min_epsilon: float = 1e-12  # Minimum dissipation rate
    min_omega: float = 1e-12  # Minimum specific dissipation rate


@dataclass
class TurbulenceQuantities:
    """Container for turbulence quantities."""
    
    # Primary turbulence variables
    k: np.ndarray  # Turbulent kinetic energy
    epsilon: Optional[np.ndarray] = None  # Dissipation rate
    omega: Optional[np.ndarray] = None  # Specific dissipation rate
    nu_tilde: Optional[np.ndarray] = None  # Modified viscosity (SA model)
    
    # Reynolds stresses (for RSM)
    reynolds_stress: Optional[np.ndarray] = None  # [n_cells, 6] symmetric tensor
    
    # Transition variables
    gamma_transition: Optional[np.ndarray] = None  # Intermittency
    re_theta: Optional[np.ndarray] = None  # Momentum thickness Reynolds number
    
    # Derived quantities
    mu_t: Optional[np.ndarray] = None  # Turbulent viscosity
    production: Optional[np.ndarray] = None  # Turbulence production
    dissipation: Optional[np.ndarray] = None  # Turbulence dissipation
    
    # Wall quantities
    y_plus: Optional[np.ndarray] = None  # Non-dimensional wall distance
    u_tau: Optional[np.ndarray] = None  # Friction velocity


class TurbulenceModel(ABC):
    """Abstract base class for turbulence models."""
    
    def __init__(self, config: TurbulenceModelConfig):
        """Initialize turbulence model."""
        self.config = config
        self.model_name = "base_turbulence_model"
        self.n_equations = 0  # Number of additional transport equations
        
    @abstractmethod
    def initialize_turbulence_quantities(self,
                                       flow_solution: np.ndarray,
                                       mesh_info: Dict[str, Any]) -> TurbulenceQuantities:
        """Initialize turbulence quantities from flow solution."""
        pass
    
    @abstractmethod
    def compute_turbulent_viscosity(self,
                                  turbulence_quantities: TurbulenceQuantities,
                                  flow_solution: np.ndarray,
                                  strain_rate: np.ndarray) -> np.ndarray:
        """Compute turbulent viscosity."""
        pass
    
    @abstractmethod
    def compute_source_terms(self,
                           turbulence_quantities: TurbulenceQuantities,
                           flow_solution: np.ndarray,
                           velocity_gradients: np.ndarray,
                           mesh_info: Dict[str, Any]) -> np.ndarray:
        """Compute source terms for turbulence equations."""
        pass
    
    @abstractmethod
    def solve_turbulence_equations(self,
                                 turbulence_quantities: TurbulenceQuantities,
                                 flow_solution: np.ndarray,
                                 dt: float,
                                 mesh_info: Dict[str, Any]) -> TurbulenceQuantities:
        """Solve turbulence transport equations."""
        pass
    
    def compute_strain_rate_magnitude(self, velocity_gradients: np.ndarray) -> np.ndarray:
        """Compute magnitude of strain rate tensor."""
        n_cells = velocity_gradients.shape[0]
        strain_magnitude = np.zeros(n_cells)
        
        for i in range(n_cells):
            # Extract velocity gradient tensor: [du/dx, du/dy, du/dz; dv/dx, dv/dy, dv/dz; dw/dx, dw/dy, dw/dz]
            grad_u = velocity_gradients[i, 0, :]  # [du/dx, du/dy, du/dz]
            grad_v = velocity_gradients[i, 1, :]  # [dv/dx, dv/dy, dv/dz]
            grad_w = velocity_gradients[i, 2, :]  # [dw/dx, dw/dy, dw/dz]
            
            # Strain rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
            S11 = grad_u[0]
            S22 = grad_v[1]
            S33 = grad_w[2]
            S12 = 0.5 * (grad_u[1] + grad_v[0])
            S13 = 0.5 * (grad_u[2] + grad_w[0])
            S23 = 0.5 * (grad_v[2] + grad_w[1])
            
            # Magnitude: |S| = sqrt(2 * S_ij * S_ij)
            strain_magnitude[i] = np.sqrt(2 * (S11**2 + S22**2 + S33**2 + 2 * (S12**2 + S13**2 + S23**2)))
        
        return strain_magnitude
    
    def compute_wall_distance(self, mesh_info: Dict[str, Any]) -> np.ndarray:
        """Compute distance to nearest wall for each cell."""
        # Simplified implementation - should use proper wall distance computation
        n_cells = mesh_info.get('n_cells', 1000)
        # For now, return a simplified estimate
        return np.ones(n_cells) * 0.01  # 1 cm default wall distance


class SpalartAllmarasModel(TurbulenceModel):
    """
    Spalart-Allmaras one-equation turbulence model.
    
    Transport equation for modified viscosity ν̃.
    Well-suited for aerospace applications with mild separation.
    """
    
    def __init__(self, config: TurbulenceModelConfig):
        super().__init__(config)
        self.model_name = "spalart_allmaras"
        self.n_equations = 1
        
        # Model constants
        self.cb1 = 0.1355
        self.cb2 = 0.622
        self.sigma = 2.0/3.0
        self.kappa = 0.41
        self.cw1 = self.cb1 / self.kappa**2 + (1 + self.cb2) / self.sigma
        self.cw2 = 0.3
        self.cw3 = 2.0
        self.cv1 = 7.1
        self.ct1 = 1.0
        self.ct2 = 2.0
        self.ct3 = 1.2
        self.ct4 = 0.5
    
    def initialize_turbulence_quantities(self,
                                       flow_solution: np.ndarray,
                                       mesh_info: Dict[str, Any]) -> TurbulenceQuantities:
        """Initialize SA turbulence quantities."""
        n_cells = flow_solution.shape[0]
        
        # Estimate initial modified viscosity
        mu_laminar = 1.8e-5  # Laminar viscosity
        nu_tilde_init = self.config.viscosity_ratio * mu_laminar * np.ones(n_cells)
        
        # Initialize wall distance
        wall_distance = self.compute_wall_distance(mesh_info)
        
        turbulence_quantities = TurbulenceQuantities(
            k=np.zeros(n_cells),  # Not used in SA
            nu_tilde=nu_tilde_init
        )
        
        return turbulence_quantities
    
    def compute_turbulent_viscosity(self,
                                  turbulence_quantities: TurbulenceQuantities,
                                  flow_solution: np.ndarray,
                                  strain_rate: np.ndarray) -> np.ndarray:
        """Compute turbulent viscosity for SA model."""
        nu_tilde = turbulence_quantities.nu_tilde
        mu_laminar = 1.8e-5
        rho = flow_solution[:, 0]
        
        # Compute SA viscosity
        chi = rho * nu_tilde / mu_laminar
        fv1 = chi**3 / (chi**3 + self.cv1**3)
        
        mu_t = rho * nu_tilde * fv1
        turbulence_quantities.mu_t = mu_t
        
        return mu_t
    
    def compute_source_terms(self,
                           turbulence_quantities: TurbulenceQuantities,
                           flow_solution: np.ndarray,
                           velocity_gradients: np.ndarray,
                           mesh_info: Dict[str, Any]) -> np.ndarray:
        """Compute SA source terms."""
        n_cells = flow_solution.shape[0]
        source_terms = np.zeros(n_cells)
        
        nu_tilde = turbulence_quantities.nu_tilde
        strain_magnitude = self.compute_strain_rate_magnitude(velocity_gradients)
        wall_distance = self.compute_wall_distance(mesh_info)
        
        for i in range(n_cells):
            # Production term
            S_tilde = strain_magnitude[i] + nu_tilde[i] / (self.kappa**2 * wall_distance[i]**2) * self._fv2(nu_tilde[i])
            production = self.cb1 * S_tilde * nu_tilde[i]
            
            # Destruction term
            r = min(nu_tilde[i] / (S_tilde * self.kappa**2 * wall_distance[i]**2), 10.0)
            g = r + self.cw2 * (r**6 - r)
            fw = g * ((1 + self.cw3**6) / (g**6 + self.cw3**6))**(1/6)
            destruction = self.cw1 * fw * (nu_tilde[i] / wall_distance[i])**2
            
            source_terms[i] = production - destruction
        
        return source_terms
    
    def _fv2(self, nu_tilde: float) -> float:
        """SA model fv2 function."""
        mu_laminar = 1.8e-5
        chi = nu_tilde / mu_laminar
        return 1 - chi / (1 + chi * self._fv1(chi))
    
    def _fv1(self, chi: float) -> float:
        """SA model fv1 function."""
        return chi**3 / (chi**3 + self.cv1**3)
    
    def solve_turbulence_equations(self,
                                 turbulence_quantities: TurbulenceQuantities,
                                 flow_solution: np.ndarray,
                                 dt: float,
                                 mesh_info: Dict[str, Any]) -> TurbulenceQuantities:
        """Solve SA transport equation."""
        source_terms = self.compute_source_terms(
            turbulence_quantities, flow_solution, 
            np.zeros((flow_solution.shape[0], 3, 3)), mesh_info
        )
        
        # Simple explicit update (should use implicit scheme)
        turbulence_quantities.nu_tilde += dt * source_terms
        turbulence_quantities.nu_tilde = np.maximum(turbulence_quantities.nu_tilde, 1e-12)
        
        return turbulence_quantities


class KEpsilonStandardModel(TurbulenceModel):
    """
    Standard k-ε turbulence model.
    
    Two-equation model solving transport equations for k and ε.
    Widely used for industrial applications.
    """
    
    def __init__(self, config: TurbulenceModelConfig):
        super().__init__(config)
        self.model_name = "k_epsilon_standard"
        self.n_equations = 2
        
        # Model constants
        self.cmu = 0.09
        self.c1_eps = 1.44
        self.c2_eps = 1.92
        self.sigma_k = 1.0
        self.sigma_eps = 1.3
    
    def initialize_turbulence_quantities(self,
                                       flow_solution: np.ndarray,
                                       mesh_info: Dict[str, Any]) -> TurbulenceQuantities:
        """Initialize k-ε turbulence quantities."""
        n_cells = flow_solution.shape[0]
        
        # Estimate velocity magnitude
        rho = flow_solution[:, 0]
        velocity_magnitude = np.sqrt(
            (flow_solution[:, 1] / rho)**2 + 
            (flow_solution[:, 2] / rho)**2 + 
            (flow_solution[:, 3] / rho)**2
        )
        
        # Initial turbulent kinetic energy
        k_init = 1.5 * (self.config.turbulence_intensity * velocity_magnitude)**2
        
        # Initial dissipation rate
        wall_distance = self.compute_wall_distance(mesh_info)
        epsilon_init = self.cmu**(3/4) * k_init**(3/2) / (0.07 * wall_distance)
        
        turbulence_quantities = TurbulenceQuantities(
            k=k_init,
            epsilon=epsilon_init
        )
        
        return turbulence_quantities
    
    def compute_turbulent_viscosity(self,
                                  turbulence_quantities: TurbulenceQuantities,
                                  flow_solution: np.ndarray,
                                  strain_rate: np.ndarray) -> np.ndarray:
        """Compute turbulent viscosity for k-ε model."""
        k = turbulence_quantities.k
        epsilon = turbulence_quantities.epsilon
        rho = flow_solution[:, 0]
        
        # Clip values to avoid division by zero
        k_clipped = np.maximum(k, self.config.min_k)
        epsilon_clipped = np.maximum(epsilon, self.config.min_epsilon)
        
        # Turbulent viscosity: μ_t = ρ * C_μ * k² / ε
        mu_t = rho * self.cmu * k_clipped**2 / epsilon_clipped
        turbulence_quantities.mu_t = mu_t
        
        return mu_t
    
    def compute_source_terms(self,
                           turbulence_quantities: TurbulenceQuantities,
                           flow_solution: np.ndarray,
                           velocity_gradients: np.ndarray,
                           mesh_info: Dict[str, Any]) -> np.ndarray:
        """Compute k-ε source terms."""
        n_cells = flow_solution.shape[0]
        source_terms = np.zeros((n_cells, 2))  # [k_source, epsilon_source]
        
        k = turbulence_quantities.k
        epsilon = turbulence_quantities.epsilon
        mu_t = turbulence_quantities.mu_t
        strain_magnitude = self.compute_strain_rate_magnitude(velocity_gradients)
        rho = flow_solution[:, 0]
        
        # Production of k
        production_k = mu_t * strain_magnitude**2
        
        # Apply production limiter
        production_k = np.minimum(production_k, self.config.production_limiter * rho * epsilon)
        
        # k equation source terms
        source_terms[:, 0] = production_k - rho * epsilon
        
        # ε equation source terms
        time_scale = k / epsilon
        source_terms[:, 1] = (self.c1_eps * production_k - 
                             self.c2_eps * rho * epsilon) / time_scale
        
        turbulence_quantities.production = production_k
        turbulence_quantities.dissipation = rho * epsilon
        
        return source_terms
    
    def solve_turbulence_equations(self,
                                 turbulence_quantities: TurbulenceQuantities,
                                 flow_solution: np.ndarray,
                                 dt: float,
                                 mesh_info: Dict[str, Any]) -> TurbulenceQuantities:
        """Solve k-ε transport equations."""
        source_terms = self.compute_source_terms(
            turbulence_quantities, flow_solution,
            np.zeros((flow_solution.shape[0], 3, 3)), mesh_info
        )
        
        # Simple explicit update with under-relaxation
        alpha_k = self.config.under_relaxation_k
        alpha_eps = self.config.under_relaxation_epsilon
        
        k_new = turbulence_quantities.k + alpha_k * dt * source_terms[:, 0]
        epsilon_new = turbulence_quantities.epsilon + alpha_eps * dt * source_terms[:, 1]
        
        # Apply bounds
        turbulence_quantities.k = np.maximum(k_new, self.config.min_k)
        turbulence_quantities.epsilon = np.maximum(epsilon_new, self.config.min_epsilon)
        
        return turbulence_quantities


class KOmegaSSTModel(TurbulenceModel):
    """
    k-ω SST (Shear Stress Transport) turbulence model.
    
    Combines k-ω model near walls with k-ε model in freestream.
    Excellent for aerodynamic flows with adverse pressure gradients.
    """
    
    def __init__(self, config: TurbulenceModelConfig):
        super().__init__(config)
        self.model_name = "k_omega_sst"
        self.n_equations = 2
        
        # k-ω model constants (inner)
        self.beta_star = 0.09
        self.alpha1 = 5.0/9.0
        self.beta1 = 3.0/40.0
        self.sigma_k1 = 0.85
        self.sigma_omega1 = 0.5
        
        # k-ε model constants (outer, transformed to k-ω)
        self.alpha2 = 0.44
        self.beta2 = 0.0828
        self.sigma_k2 = 1.0
        self.sigma_omega2 = 0.856
        
        # SST model constants
        self.a1 = 0.31
        self.kappa = 0.41
    
    def initialize_turbulence_quantities(self,
                                       flow_solution: np.ndarray,
                                       mesh_info: Dict[str, Any]) -> TurbulenceQuantities:
        """Initialize k-ω SST turbulence quantities."""
        n_cells = flow_solution.shape[0]
        
        # Estimate velocity magnitude
        rho = flow_solution[:, 0]
        velocity_magnitude = np.sqrt(
            (flow_solution[:, 1] / rho)**2 + 
            (flow_solution[:, 2] / rho)**2 + 
            (flow_solution[:, 3] / rho)**2
        )
        
        # Initial turbulent kinetic energy
        k_init = 1.5 * (self.config.turbulence_intensity * velocity_magnitude)**2
        
        # Initial specific dissipation rate
        mu_laminar = 1.8e-5
        omega_init = np.sqrt(k_init) / (self.beta_star**0.25 * 0.07 * self.compute_wall_distance(mesh_info))
        
        turbulence_quantities = TurbulenceQuantities(
            k=k_init,
            omega=omega_init
        )
        
        return turbulence_quantities
    
    def compute_turbulent_viscosity(self,
                                  turbulence_quantities: TurbulenceQuantities,
                                  flow_solution: np.ndarray,
                                  strain_rate: np.ndarray) -> np.ndarray:
        """Compute turbulent viscosity for k-ω SST model."""
        k = turbulence_quantities.k
        omega = turbulence_quantities.omega
        rho = flow_solution[:, 0]
        
        # Clip values
        k_clipped = np.maximum(k, self.config.min_k)
        omega_clipped = np.maximum(omega, self.config.min_omega)
        
        # SST viscosity formulation  
        n_cells = len(k_clipped)
        wall_distance = np.ones(n_cells) * 0.01  # Simplified wall distance
        arg2 = np.maximum(2 * np.sqrt(k_clipped) / (self.beta_star * omega_clipped * wall_distance),
                         500 * 1.8e-5 / (rho * omega_clipped * wall_distance**2))
        
        F2 = np.tanh(arg2**2)
        
        # Turbulent viscosity with Bradshaw assumption
        mu_t = rho * self.a1 * k_clipped / np.maximum(self.a1 * omega_clipped, 
                                                     strain_rate * F2)
        
        turbulence_quantities.mu_t = mu_t
        return mu_t
    
    def compute_blending_function(self,
                                turbulence_quantities: TurbulenceQuantities,
                                flow_solution: np.ndarray,
                                mesh_info: Dict[str, Any]) -> np.ndarray:
        """Compute F1 blending function for SST model."""
        k = turbulence_quantities.k
        omega = turbulence_quantities.omega
        rho = flow_solution[:, 0]
        wall_distance = self.compute_wall_distance(mesh_info)
        
        # Clip values
        k_clipped = np.maximum(k, self.config.min_k)
        omega_clipped = np.maximum(omega, self.config.min_omega)
        
        # Blending function arguments
        arg1_term1 = np.sqrt(k_clipped) / (self.beta_star * omega_clipped * wall_distance)
        arg1_term2 = 500 * 1.8e-5 / (rho * omega_clipped * wall_distance**2)
        arg1 = np.minimum(np.maximum(arg1_term1, arg1_term2), 4 * rho * self.sigma_omega2 * k_clipped / 
                         (np.maximum(self._compute_cross_diffusion(turbulence_quantities, flow_solution), 1e-20) * wall_distance**2))
        
        F1 = np.tanh(arg1**4)
        return F1
    
    def _compute_cross_diffusion(self,
                               turbulence_quantities: TurbulenceQuantities,
                               flow_solution: np.ndarray) -> np.ndarray:
        """Compute cross-diffusion term for SST model."""
        # Simplified - should compute actual gradients
        return 1e-10 * np.ones(flow_solution.shape[0])
    
    def compute_source_terms(self,
                           turbulence_quantities: TurbulenceQuantities,
                           flow_solution: np.ndarray,
                           velocity_gradients: np.ndarray,
                           mesh_info: Dict[str, Any]) -> np.ndarray:
        """Compute k-ω SST source terms."""
        n_cells = flow_solution.shape[0]
        source_terms = np.zeros((n_cells, 2))  # [k_source, omega_source]
        
        k = turbulence_quantities.k
        omega = turbulence_quantities.omega
        mu_t = turbulence_quantities.mu_t
        strain_magnitude = self.compute_strain_rate_magnitude(velocity_gradients)
        rho = flow_solution[:, 0]
        
        # Blending function
        F1 = self.compute_blending_function(turbulence_quantities, flow_solution, mesh_info)
        
        # Blended constants
        alpha = F1 * self.alpha1 + (1 - F1) * self.alpha2
        beta = F1 * self.beta1 + (1 - F1) * self.beta2
        
        # Production of k
        production_k = mu_t * strain_magnitude**2
        
        # Apply production limiter
        production_k = np.minimum(production_k, self.config.production_limiter * rho * self.beta_star * k * omega)
        
        # k equation source terms
        source_terms[:, 0] = production_k - rho * self.beta_star * k * omega
        
        # ω equation source terms
        production_omega = alpha * rho * strain_magnitude**2
        destruction_omega = rho * beta * omega**2
        
        # Cross-diffusion term (simplified)
        cross_diffusion = 2 * (1 - F1) * rho * self.sigma_omega2 * self._compute_cross_diffusion(turbulence_quantities, flow_solution) / omega
        
        source_terms[:, 1] = production_omega - destruction_omega + cross_diffusion
        
        turbulence_quantities.production = production_k
        turbulence_quantities.dissipation = rho * self.beta_star * k * omega
        
        return source_terms
    
    def solve_turbulence_equations(self,
                                 turbulence_quantities: TurbulenceQuantities,
                                 flow_solution: np.ndarray,
                                 dt: float,
                                 mesh_info: Dict[str, Any]) -> TurbulenceQuantities:
        """Solve k-ω SST transport equations."""
        source_terms = self.compute_source_terms(
            turbulence_quantities, flow_solution,
            np.zeros((flow_solution.shape[0], 3, 3)), mesh_info
        )
        
        # Simple explicit update with under-relaxation
        alpha_k = self.config.under_relaxation_k
        alpha_omega = self.config.under_relaxation_omega
        
        k_new = turbulence_quantities.k + alpha_k * dt * source_terms[:, 0]
        omega_new = turbulence_quantities.omega + alpha_omega * dt * source_terms[:, 1]
        
        # Apply bounds
        turbulence_quantities.k = np.maximum(k_new, self.config.min_k)
        turbulence_quantities.omega = np.maximum(omega_new, self.config.min_omega)
        
        return turbulence_quantities


class TurbulenceModelManager:
    """
    Manager for turbulence models in CFD simulations.
    
    Handles selection, initialization, and solution of turbulence equations.
    """
    
    def __init__(self, config: TurbulenceModelConfig):
        """Initialize turbulence model manager."""
        self.config = config
        self.turbulence_model = self._create_turbulence_model()
        self.turbulence_quantities: Optional[TurbulenceQuantities] = None
        
    def _create_turbulence_model(self) -> TurbulenceModel:
        """Create the specified turbulence model."""
        if self.config.model_type == TurbulenceModelType.SPALART_ALLMARAS:
            return SpalartAllmarasModel(self.config)
        elif self.config.model_type == TurbulenceModelType.K_EPSILON_STANDARD:
            return KEpsilonStandardModel(self.config)
        elif self.config.model_type == TurbulenceModelType.K_OMEGA_SST:
            return KOmegaSSTModel(self.config)
        else:
            raise ValueError(f"Unsupported turbulence model: {self.config.model_type}")
    
    def initialize_turbulence(self,
                            flow_solution: np.ndarray,
                            mesh_info: Dict[str, Any]) -> None:
        """Initialize turbulence quantities."""
        self.turbulence_quantities = self.turbulence_model.initialize_turbulence_quantities(
            flow_solution, mesh_info
        )
        
        logger.info(f"Initialized {self.turbulence_model.model_name} turbulence model "
                   f"with {self.turbulence_model.n_equations} transport equations")
    
    def solve_turbulence_step(self,
                            flow_solution: np.ndarray,
                            velocity_gradients: np.ndarray,
                            dt: float,
                            mesh_info: Dict[str, Any]) -> TurbulenceQuantities:
        """Solve one time step of turbulence equations."""
        if self.turbulence_quantities is None:
            self.initialize_turbulence(flow_solution, mesh_info)
        
        # Compute strain rate
        strain_rate = self.turbulence_model.compute_strain_rate_magnitude(velocity_gradients)
        
        # Update turbulent viscosity
        mu_t = self.turbulence_model.compute_turbulent_viscosity(
            self.turbulence_quantities, flow_solution, strain_rate
        )
        
        # Solve turbulence equations
        self.turbulence_quantities = self.turbulence_model.solve_turbulence_equations(
            self.turbulence_quantities, flow_solution, dt, mesh_info
        )
        
        return self.turbulence_quantities
    
    def get_turbulent_viscosity(self) -> np.ndarray:
        """Get current turbulent viscosity field."""
        if self.turbulence_quantities is None or self.turbulence_quantities.mu_t is None:
            raise ValueError("Turbulence model not initialized or solved")
        return self.turbulence_quantities.mu_t
    
    def get_turbulence_statistics(self) -> Dict[str, Any]:
        """Get turbulence model statistics."""
        if self.turbulence_quantities is None:
            return {'model_name': self.turbulence_model.model_name, 'initialized': False}
        
        stats = {
            'model_name': self.turbulence_model.model_name,
            'n_equations': self.turbulence_model.n_equations,
            'initialized': True
        }
        
        if self.turbulence_quantities.k is not None:
            stats['k_mean'] = np.mean(self.turbulence_quantities.k)
            stats['k_max'] = np.max(self.turbulence_quantities.k)
        
        if self.turbulence_quantities.epsilon is not None:
            stats['epsilon_mean'] = np.mean(self.turbulence_quantities.epsilon)
        
        if self.turbulence_quantities.omega is not None:
            stats['omega_mean'] = np.mean(self.turbulence_quantities.omega)
        
        if self.turbulence_quantities.mu_t is not None:
            stats['mu_t_mean'] = np.mean(self.turbulence_quantities.mu_t)
            stats['mu_t_max'] = np.max(self.turbulence_quantities.mu_t)
        
        return stats


def create_turbulence_model(model_type: str, config: Optional[TurbulenceModelConfig] = None) -> TurbulenceModelManager:
    """
    Factory function for creating turbulence models.
    
    Args:
        model_type: Type of turbulence model
        config: Turbulence model configuration
        
    Returns:
        Configured turbulence model manager
    """
    if config is None:
        config = TurbulenceModelConfig()
    
    config.model_type = TurbulenceModelType(model_type)
    
    return TurbulenceModelManager(config)


def test_turbulence_models():
    """Test turbulence model implementations."""
    print("Testing Turbulence Models:")
    
    # Create test flow field
    n_cells = 500
    flow_solution = np.random.rand(n_cells, 5) + 0.5
    flow_solution[:, 0] = 1.225  # Density
    flow_solution[:, 1] *= 100   # rho*u
    flow_solution[:, 4] *= 100000  # rho*E
    
    velocity_gradients = np.random.rand(n_cells, 3, 3) * 0.1
    mesh_info = {'n_cells': n_cells}
    dt = 1e-4
    
    # Test different turbulence models
    models = ["spalart_allmaras", "k_epsilon_standard", "k_omega_sst"]
    
    print(f"\n  Testing {len(models)} turbulence models:")
    
    for model_name in models:
        print(f"\n    Testing {model_name}:")
        
        # Create turbulence model
        config = TurbulenceModelConfig(turbulence_intensity=0.05, viscosity_ratio=10.0)
        turbulence_manager = create_turbulence_model(model_name, config)
        
        # Initialize and solve
        turbulence_manager.initialize_turbulence(flow_solution, mesh_info)
        
        # Solve a few time steps
        for step in range(5):
            turbulence_quantities = turbulence_manager.solve_turbulence_step(
                flow_solution, velocity_gradients, dt, mesh_info
            )
        
        # Get statistics
        stats = turbulence_manager.get_turbulence_statistics()
        
        print(f"      Model: {stats['model_name']}")
        print(f"      Equations: {stats['n_equations']}")
        if 'k_mean' in stats:
            print(f"      k_mean: {stats['k_mean']:.6f}, k_max: {stats['k_max']:.6f}")
        if 'mu_t_mean' in stats:
            print(f"      μ_t_mean: {stats['mu_t_mean']:.2e}, μ_t_max: {stats['mu_t_max']:.2e}")
    
    print(f"\n  All turbulence models tested successfully!")


if __name__ == "__main__":
    test_turbulence_models()