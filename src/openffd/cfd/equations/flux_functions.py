"""
Flux Functions for Euler and Navier-Stokes Equations

Provides inviscid and viscous flux computation for finite volume method
with optimized implementations for supersonic flows.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FluxComponents:
    """Components of flux vector for compressible flow."""
    mass: float
    momentum_x: float
    momentum_y: float
    momentum_z: float
    energy: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.mass, self.momentum_x, self.momentum_y, 
                        self.momentum_z, self.energy])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'FluxComponents':
        """Create from numpy array."""
        return cls(array[0], array[1], array[2], array[3], array[4])


class InviscidFlux:
    """
    Inviscid flux computation for Euler equations.
    
    Computes F = [ρu, ρu² + p, ρuv, ρuw, u(ρE + p)]
    """
    
    def __init__(self, gamma: float = 1.4):
        """Initialize inviscid flux calculator."""
        self.gamma = gamma
        self.gamma_minus_1 = gamma - 1.0
    
    def compute_flux_x(self, 
                      conservatives: np.ndarray,
                      primitives: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute x-direction inviscid flux.
        
        Args:
            conservatives: [rho, rho*u, rho*v, rho*w, rho*E]
            primitives: [rho, u, v, w, p, T] (optional, computed if not provided)
            
        Returns:
            flux_x: [rho*u, rho*u² + p, rho*u*v, rho*u*w, u*(rho*E + p)]
        """
        rho, rho_u, rho_v, rho_w, rho_E = conservatives
        
        if primitives is not None:
            _, u, v, w, p, _ = primitives
        else:
            # Compute primitives
            u = rho_u / rho
            v = rho_v / rho
            w = rho_w / rho
            
            # Pressure
            kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
            p = self.gamma_minus_1 * (rho_E - kinetic_energy)
        
        # x-direction flux
        flux_x = np.array([
            rho_u,                    # Mass flux
            rho_u * u + p,           # x-momentum flux
            rho_u * v,               # y-momentum flux
            rho_u * w,               # z-momentum flux
            u * (rho_E + p)          # Energy flux
        ])
        
        return flux_x
    
    def compute_flux_y(self, 
                      conservatives: np.ndarray,
                      primitives: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute y-direction inviscid flux."""
        rho, rho_u, rho_v, rho_w, rho_E = conservatives
        
        if primitives is not None:
            _, u, v, w, p, _ = primitives
        else:
            u = rho_u / rho
            v = rho_v / rho
            w = rho_w / rho
            kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
            p = self.gamma_minus_1 * (rho_E - kinetic_energy)
        
        flux_y = np.array([
            rho_v,                    # Mass flux
            rho_v * u,               # x-momentum flux
            rho_v * v + p,           # y-momentum flux
            rho_v * w,               # z-momentum flux
            v * (rho_E + p)          # Energy flux
        ])
        
        return flux_y
    
    def compute_flux_z(self, 
                      conservatives: np.ndarray,
                      primitives: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute z-direction inviscid flux."""
        rho, rho_u, rho_v, rho_w, rho_E = conservatives
        
        if primitives is not None:
            _, u, v, w, p, _ = primitives
        else:
            u = rho_u / rho
            v = rho_v / rho
            w = rho_w / rho
            kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
            p = self.gamma_minus_1 * (rho_E - kinetic_energy)
        
        flux_z = np.array([
            rho_w,                    # Mass flux
            rho_w * u,               # x-momentum flux
            rho_w * v,               # y-momentum flux
            rho_w * w + p,           # z-momentum flux
            w * (rho_E + p)          # Energy flux
        ])
        
        return flux_z
    
    def compute_flux_normal(self, 
                          conservatives: np.ndarray,
                          normal: np.ndarray,
                          primitives: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute inviscid flux in normal direction.
        
        Args:
            conservatives: Conservative variables
            normal: Unit normal vector [nx, ny, nz]
            primitives: Primitive variables (optional)
            
        Returns:
            Normal flux vector
        """
        nx, ny, nz = normal
        
        # Compute directional fluxes
        flux_x = self.compute_flux_x(conservatives, primitives)
        flux_y = self.compute_flux_y(conservatives, primitives)
        flux_z = self.compute_flux_z(conservatives, primitives)
        
        # Project onto normal direction
        flux_normal = nx * flux_x + ny * flux_y + nz * flux_z
        
        return flux_normal
    
    def compute_flux_jacobian_x(self, primitives: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix ∂F/∂U for x-direction flux.
        
        Used for implicit time integration and linearization.
        """
        rho, u, v, w, p, T = primitives
        
        # Derived quantities
        H = (self.gamma / (self.gamma - 1)) * p / rho + 0.5 * (u**2 + v**2 + w**2)  # Total enthalpy
        a2 = self.gamma * p / rho  # Speed of sound squared
        
        # Jacobian matrix
        A = np.zeros((5, 5))
        
        # Mass equation
        A[0, 0] = 0
        A[0, 1] = 1
        A[0, 2] = 0
        A[0, 3] = 0
        A[0, 4] = 0
        
        # x-momentum equation
        A[1, 0] = 0.5 * self.gamma_minus_1 * (v**2 + w**2) - u**2
        A[1, 1] = (3 - self.gamma) * u
        A[1, 2] = -self.gamma_minus_1 * v
        A[1, 3] = -self.gamma_minus_1 * w
        A[1, 4] = self.gamma_minus_1
        
        # y-momentum equation
        A[2, 0] = -u * v
        A[2, 1] = v
        A[2, 2] = u
        A[2, 3] = 0
        A[2, 4] = 0
        
        # z-momentum equation
        A[3, 0] = -u * w
        A[3, 1] = w
        A[3, 2] = 0
        A[3, 3] = u
        A[3, 4] = 0
        
        # Energy equation
        A[4, 0] = u * (0.5 * self.gamma_minus_1 * (u**2 + v**2 + w**2) - H)
        A[4, 1] = H - self.gamma_minus_1 * u**2
        A[4, 2] = -self.gamma_minus_1 * u * v
        A[4, 3] = -self.gamma_minus_1 * u * w
        A[4, 4] = self.gamma * u
        
        return A


class ViscousFlux:
    """
    Viscous flux computation for Navier-Stokes equations.
    
    Computes viscous stress tensor and heat flux contributions.
    """
    
    def __init__(self, 
                 gamma: float = 1.4,
                 prandtl_number: float = 0.72,
                 R: float = 287.0):
        """Initialize viscous flux calculator."""
        self.gamma = gamma
        self.gamma_minus_1 = gamma - 1.0
        self.Pr = prandtl_number
        self.R = R
        
        # Specific heats
        self.cv = R / self.gamma_minus_1
        self.cp = gamma * self.cv
    
    def compute_viscous_flux_x(self, 
                             primitives: np.ndarray,
                             gradients: np.ndarray,
                             viscosity: float,
                             thermal_conductivity: float) -> np.ndarray:
        """
        Compute x-direction viscous flux.
        
        Args:
            primitives: [rho, u, v, w, p, T]
            gradients: Velocity and temperature gradients [du/dx, du/dy, du/dz, dv/dx, ...]
            viscosity: Dynamic viscosity
            thermal_conductivity: Thermal conductivity
            
        Returns:
            Viscous flux in x-direction
        """
        rho, u, v, w, p, T = primitives
        
        # Velocity gradients
        dudx, dudy, dudz = gradients[0:3]
        dvdx, dvdy, dvdz = gradients[3:6]
        dwdx, dwdy, dwdz = gradients[6:9]
        dTdx, dTdy, dTdz = gradients[9:12]
        
        # Strain rate tensor components
        tau_xx = 2 * viscosity * (dudx - (1/3) * (dudx + dvdy + dwdz))
        tau_xy = viscosity * (dudy + dvdx)
        tau_xz = viscosity * (dudz + dwdx)
        
        # Heat flux
        q_x = -thermal_conductivity * dTdx
        
        # Viscous flux
        flux_visc_x = np.array([
            0,                                          # No mass flux
            tau_xx,                                     # x-momentum flux
            tau_xy,                                     # y-momentum flux
            tau_xz,                                     # z-momentum flux
            u * tau_xx + v * tau_xy + w * tau_xz + q_x # Energy flux
        ])
        
        return flux_visc_x
    
    def compute_viscous_flux_y(self, 
                             primitives: np.ndarray,
                             gradients: np.ndarray,
                             viscosity: float,
                             thermal_conductivity: float) -> np.ndarray:
        """Compute y-direction viscous flux."""
        rho, u, v, w, p, T = primitives
        
        # Velocity gradients
        dudx, dudy, dudz = gradients[0:3]
        dvdx, dvdy, dvdz = gradients[3:6]
        dwdx, dwdy, dwdz = gradients[6:9]
        dTdx, dTdy, dTdz = gradients[9:12]
        
        # Strain rate tensor components
        tau_yx = viscosity * (dvdx + dudy)
        tau_yy = 2 * viscosity * (dvdy - (1/3) * (dudx + dvdy + dwdz))
        tau_yz = viscosity * (dvdz + dwdy)
        
        # Heat flux
        q_y = -thermal_conductivity * dTdy
        
        flux_visc_y = np.array([
            0,
            tau_yx,
            tau_yy,
            tau_yz,
            u * tau_yx + v * tau_yy + w * tau_yz + q_y
        ])
        
        return flux_visc_y
    
    def compute_viscous_flux_z(self, 
                             primitives: np.ndarray,
                             gradients: np.ndarray,
                             viscosity: float,
                             thermal_conductivity: float) -> np.ndarray:
        """Compute z-direction viscous flux."""
        rho, u, v, w, p, T = primitives
        
        # Velocity gradients
        dudx, dudy, dudz = gradients[0:3]
        dvdx, dvdy, dvdz = gradients[3:6]
        dwdx, dwdy, dwdz = gradients[6:9]
        dTdx, dTdy, dTdz = gradients[9:12]
        
        # Strain rate tensor components
        tau_zx = viscosity * (dwdx + dudz)
        tau_zy = viscosity * (dwdy + dvdz)
        tau_zz = 2 * viscosity * (dwdz - (1/3) * (dudx + dvdy + dwdz))
        
        # Heat flux
        q_z = -thermal_conductivity * dTdz
        
        flux_visc_z = np.array([
            0,
            tau_zx,
            tau_zy,
            tau_zz,
            u * tau_zx + v * tau_zy + w * tau_zz + q_z
        ])
        
        return flux_visc_z
    
    def compute_viscous_flux_normal(self, 
                                  primitives: np.ndarray,
                                  gradients: np.ndarray,
                                  normal: np.ndarray,
                                  viscosity: float,
                                  thermal_conductivity: float) -> np.ndarray:
        """Compute viscous flux in normal direction."""
        nx, ny, nz = normal
        
        # Compute directional viscous fluxes
        flux_x = self.compute_viscous_flux_x(primitives, gradients, viscosity, thermal_conductivity)
        flux_y = self.compute_viscous_flux_y(primitives, gradients, viscosity, thermal_conductivity)
        flux_z = self.compute_viscous_flux_z(primitives, gradients, viscosity, thermal_conductivity)
        
        # Project onto normal direction
        flux_normal = nx * flux_x + ny * flux_y + nz * flux_z
        
        return flux_normal
    
    def compute_stress_tensor(self, 
                            gradients: np.ndarray,
                            viscosity: float) -> np.ndarray:
        """
        Compute full viscous stress tensor.
        
        Returns:
            3x3 stress tensor
        """
        # Velocity gradients
        dudx, dudy, dudz = gradients[0:3]
        dvdx, dvdy, dvdz = gradients[3:6]
        dwdx, dwdy, dwdz = gradients[6:9]
        
        # Divergence of velocity
        div_v = dudx + dvdy + dwdz
        
        # Stress tensor components
        tau = np.zeros((3, 3))
        
        # Diagonal terms
        tau[0, 0] = 2 * viscosity * (dudx - div_v / 3)  # τxx
        tau[1, 1] = 2 * viscosity * (dvdy - div_v / 3)  # τyy
        tau[2, 2] = 2 * viscosity * (dwdz - div_v / 3)  # τzz
        
        # Off-diagonal terms
        tau[0, 1] = tau[1, 0] = viscosity * (dudy + dvdx)  # τxy = τyx
        tau[0, 2] = tau[2, 0] = viscosity * (dudz + dwdx)  # τxz = τzx
        tau[1, 2] = tau[2, 1] = viscosity * (dvdz + dwdy)  # τyz = τzy
        
        return tau
    
    def compute_heat_flux_vector(self, 
                               temperature_gradients: np.ndarray,
                               thermal_conductivity: float) -> np.ndarray:
        """Compute heat flux vector using Fourier's law."""
        dTdx, dTdy, dTdz = temperature_gradients
        
        # Fourier's law: q = -k * ∇T
        q = -thermal_conductivity * np.array([dTdx, dTdy, dTdz])
        
        return q
    
    def compute_dissipation_rate(self, 
                               primitives: np.ndarray,
                               gradients: np.ndarray,
                               viscosity: float) -> float:
        """
        Compute viscous dissipation rate.
        
        Used for energy equation and turbulence modeling.
        """
        # Velocity gradients
        dudx, dudy, dudz = gradients[0:3]
        dvdx, dvdy, dvdz = gradients[3:6]
        dwdx, dwdy, dwdz = gradients[6:9]
        
        # Strain rate tensor (symmetric part of velocity gradient)
        S = np.zeros((3, 3))
        S[0, 0] = dudx
        S[1, 1] = dvdy
        S[2, 2] = dwdz
        S[0, 1] = S[1, 0] = 0.5 * (dudy + dvdx)
        S[0, 2] = S[2, 0] = 0.5 * (dudz + dwdx)
        S[1, 2] = S[2, 1] = 0.5 * (dvdz + dwdy)
        
        # Dissipation rate: Φ = 2μ * S:S
        dissipation = 2 * viscosity * np.sum(S**2)
        
        return dissipation


class FluxCalculator:
    """
    Combined flux calculator for complete Navier-Stokes equations.
    
    Manages both inviscid and viscous flux computations.
    """
    
    def __init__(self, 
                 gamma: float = 1.4,
                 prandtl_number: float = 0.72,
                 R: float = 287.0):
        """Initialize flux calculator."""
        self.inviscid_flux = InviscidFlux(gamma)
        self.viscous_flux = ViscousFlux(gamma, prandtl_number, R)
        
        self.gamma = gamma
        self.Pr = prandtl_number
        self.R = R
    
    def compute_total_flux(self, 
                         conservatives: np.ndarray,
                         primitives: np.ndarray,
                         normal: np.ndarray,
                         gradients: Optional[np.ndarray] = None,
                         viscosity: Optional[float] = None,
                         thermal_conductivity: Optional[float] = None,
                         include_viscous: bool = True) -> np.ndarray:
        """
        Compute total flux (inviscid + viscous) in normal direction.
        
        Args:
            conservatives: Conservative variables
            primitives: Primitive variables
            normal: Unit normal vector
            gradients: Solution gradients (required for viscous)
            viscosity: Dynamic viscosity (required for viscous)
            thermal_conductivity: Thermal conductivity (required for viscous)
            include_viscous: Whether to include viscous terms
            
        Returns:
            Total flux vector
        """
        # Inviscid flux
        flux_inviscid = self.inviscid_flux.compute_flux_normal(conservatives, normal, primitives)
        
        if not include_viscous or gradients is None or viscosity is None:
            return flux_inviscid
        
        # Viscous flux
        if thermal_conductivity is None:
            # Compute from Prandtl number
            cp = self.gamma * self.R / (self.gamma - 1)
            thermal_conductivity = viscosity * cp / self.Pr
        
        flux_viscous = self.viscous_flux.compute_viscous_flux_normal(
            primitives, gradients, normal, viscosity, thermal_conductivity
        )
        
        # Total flux
        flux_total = flux_inviscid - flux_viscous  # Note: viscous flux is subtracted
        
        return flux_total
    
    def compute_flux_residual(self, 
                            cell_data: Dict,
                            face_data: Dict,
                            mesh,
                            include_viscous: bool = True) -> np.ndarray:
        """
        Compute flux residual for all cells using finite volume method.
        
        ∫∫∫ ∂U/∂t dV = -∮∮ F⃗ · n̂ dS
        
        Returns:
            Residual array [n_cells, 5]
        """
        n_cells = mesh.n_cells
        residuals = np.zeros((n_cells, 5))
        
        # Loop over all faces
        for face_id in range(mesh.n_faces):
            owner_cell = face_data['owner'][face_id]
            neighbor_cell = face_data['neighbor'][face_id]
            
            face_area = face_data['areas'][face_id]
            face_normal = face_data['normals'][face_id]
            
            if neighbor_cell >= 0:  # Internal face
                # Average states for flux computation
                owner_cons = cell_data['conservatives'][owner_cell]
                neighbor_cons = cell_data['conservatives'][neighbor_cell]
                
                owner_prim = cell_data['primitives'][owner_cell]
                neighbor_prim = cell_data['primitives'][neighbor_cell]
                
                # Simple averaging (can be improved with Riemann solvers)
                avg_cons = 0.5 * (owner_cons + neighbor_cons)
                avg_prim = 0.5 * (owner_prim + neighbor_prim)
                
                # Compute flux
                flux = self.compute_total_flux(
                    avg_cons, avg_prim, face_normal,
                    include_viscous=include_viscous
                )
                
                # Add flux to residuals
                residuals[owner_cell] += face_area * flux
                residuals[neighbor_cell] -= face_area * flux
                
            else:  # Boundary face
                # Use owner cell state
                owner_cons = cell_data['conservatives'][owner_cell]
                owner_prim = cell_data['primitives'][owner_cell]
                
                # Boundary condition handling would go here
                # For now, use simple extrapolation
                flux = self.compute_total_flux(
                    owner_cons, owner_prim, face_normal,
                    include_viscous=include_viscous
                )
                
                residuals[owner_cell] += face_area * flux
        
        return residuals