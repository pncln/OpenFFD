"""
Adjoint Boundary Conditions

Implements boundary conditions for the discrete adjoint equations
including proper treatment of different physical boundary types.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class AdjointBCType(Enum):
    """Types of adjoint boundary conditions."""
    WALL = "wall"
    FARFIELD = "farfield"
    INLET = "inlet"
    OUTLET = "outlet"
    SYMMETRY = "symmetry"
    PERIODIC = "periodic"
    CHARACTERISTIC = "characteristic"


@dataclass
class AdjointBoundaryConfig:
    """Configuration for adjoint boundary conditions."""
    
    # Physical parameters
    gamma: float = 1.4
    gas_constant: float = 287.0
    
    # Farfield conditions
    farfield_mach: float = 0.8
    farfield_angle_of_attack: float = 0.0  # degrees
    farfield_pressure: float = 101325.0
    farfield_temperature: float = 288.15
    
    # Numerical parameters
    characteristic_relaxation: float = 1.0
    entropy_fix_parameter: float = 0.1
    
    # Output control
    debug_boundary_residuals: bool = False


class AdjointBoundaryCondition(ABC):
    """Abstract base class for adjoint boundary conditions."""
    
    def __init__(self, 
                 bc_type: AdjointBCType,
                 config: Optional[AdjointBoundaryConfig] = None):
        """
        Initialize adjoint boundary condition.
        
        Args:
            bc_type: Type of boundary condition
            config: Configuration parameters
        """
        self.bc_type = bc_type
        self.config = config or AdjointBoundaryConfig()
        
    @abstractmethod
    def apply_boundary_condition(self,
                                adjoint_variables: np.ndarray,
                                flow_solution: np.ndarray,
                                boundary_faces: List[int],
                                mesh_info: Dict[str, Any],
                                objective_gradient: np.ndarray) -> np.ndarray:
        """
        Apply adjoint boundary condition.
        
        Args:
            adjoint_variables: Adjoint variables at boundary cells [n_faces, n_vars]
            flow_solution: Flow solution at boundary cells [n_faces, n_vars]
            boundary_faces: List of boundary face IDs
            mesh_info: Mesh information
            objective_gradient: Objective function gradient at boundary
            
        Returns:
            Modified adjoint variables
        """
        pass
    
    @abstractmethod
    def compute_boundary_residual_contribution(self,
                                             adjoint_variables: np.ndarray,
                                             flow_solution: np.ndarray,
                                             boundary_faces: List[int],
                                             mesh_info: Dict[str, Any]) -> np.ndarray:
        """
        Compute boundary contribution to adjoint residual.
        
        Returns:
            Boundary residual contribution [n_boundary_cells, n_vars]
        """
        pass


class WallAdjointBC(AdjointBoundaryCondition):
    """
    Wall boundary condition for adjoint equations.
    
    For solid walls:
    - Normal velocity component of adjoint variables is set based on objective gradient
    - Tangential components follow no-slip or slip conditions
    - Energy equation handling depends on thermal conditions
    """
    
    def __init__(self, 
                 wall_type: str = "adiabatic",  # "adiabatic", "isothermal", "heat_flux"
                 config: Optional[AdjointBoundaryConfig] = None):
        """
        Initialize wall adjoint boundary condition.
        
        Args:
            wall_type: Type of wall condition
            config: Configuration
        """
        super().__init__(AdjointBCType.WALL, config)
        self.wall_type = wall_type
        
    def apply_boundary_condition(self,
                                adjoint_variables: np.ndarray,
                                flow_solution: np.ndarray,
                                boundary_faces: List[int],
                                mesh_info: Dict[str, Any],
                                objective_gradient: np.ndarray) -> np.ndarray:
        """Apply wall boundary condition to adjoint variables."""
        modified_adjoint = adjoint_variables.copy()
        
        for i, face_id in enumerate(boundary_faces):
            face_normal = mesh_info['face_normals'][face_id]
            face_area = mesh_info['face_areas'][face_id]
            
            # Get current adjoint state
            lambda_vars = modified_adjoint[i]  # [rho, rho_u, rho_v, rho_w, rho_E]
            
            # Apply wall conditions based on type
            if self.wall_type == "adiabatic":
                modified_adjoint[i] = self._apply_adiabatic_wall(
                    lambda_vars, face_normal, objective_gradient[i]
                )
            elif self.wall_type == "isothermal":
                modified_adjoint[i] = self._apply_isothermal_wall(
                    lambda_vars, face_normal, objective_gradient[i]
                )
            else:
                # Default adiabatic
                modified_adjoint[i] = self._apply_adiabatic_wall(
                    lambda_vars, face_normal, objective_gradient[i]
                )
        
        return modified_adjoint
    
    def _apply_adiabatic_wall(self,
                             lambda_vars: np.ndarray,
                             normal: np.ndarray,
                             obj_gradient: np.ndarray) -> np.ndarray:
        """Apply adiabatic wall conditions."""
        lambda_modified = lambda_vars.copy()
        
        # For adiabatic wall:
        # 1. Normal velocity = 0 (no-penetration)
        # 2. No heat transfer (∂T/∂n = 0)
        # 3. Viscous stress conditions
        
        # Extract momentum components
        lambda_rho_u, lambda_rho_v, lambda_rho_w = lambda_vars[1:4]
        nx, ny, nz = normal
        
        # Normal momentum adjoint: λ_momentum · n = objective gradient contribution
        normal_lambda_momentum = lambda_rho_u * nx + lambda_rho_v * ny + lambda_rho_w * nz
        
        # Set normal component based on objective gradient (pressure contribution)
        if len(obj_gradient) >= 4:
            # Objective gradient provides the boundary condition
            target_normal_lambda = obj_gradient[1] * nx + obj_gradient[2] * ny + obj_gradient[3] * nz
        else:
            target_normal_lambda = 0.0
        
        # Modify momentum adjoints to satisfy normal condition
        lambda_correction = target_normal_lambda - normal_lambda_momentum
        lambda_modified[1] += lambda_correction * nx
        lambda_modified[2] += lambda_correction * ny  
        lambda_modified[3] += lambda_correction * nz
        
        # Energy adjoint for adiabatic condition
        # ∂T/∂n = 0 provides constraint on energy adjoint
        if len(obj_gradient) >= 5:
            lambda_modified[4] = obj_gradient[4]
        
        return lambda_modified
    
    def _apply_isothermal_wall(self,
                              lambda_vars: np.ndarray,
                              normal: np.ndarray,
                              obj_gradient: np.ndarray) -> np.ndarray:
        """Apply isothermal wall conditions."""
        lambda_modified = lambda_vars.copy()
        
        # For isothermal wall:
        # 1. Normal velocity = 0
        # 2. Temperature = constant
        # 3. Heat flux determined by temperature gradient
        
        # Similar to adiabatic but with prescribed temperature condition
        lambda_modified = self._apply_adiabatic_wall(lambda_vars, normal, obj_gradient)
        
        # Additional constraint for fixed temperature
        # This affects the energy equation adjoint
        if len(obj_gradient) >= 5:
            # Temperature constraint provides additional relationship
            lambda_modified[4] = obj_gradient[4]
        
        return lambda_modified
    
    def compute_boundary_residual_contribution(self,
                                             adjoint_variables: np.ndarray,
                                             flow_solution: np.ndarray,
                                             boundary_faces: List[int],
                                             mesh_info: Dict[str, Any]) -> np.ndarray:
        """Compute wall boundary residual contribution."""
        n_cells = adjoint_variables.shape[0]
        residual_contribution = np.zeros((n_cells, 5))
        
        for i, face_id in enumerate(boundary_faces):
            owner_cell = mesh_info['face_owners'][face_id]
            face_normal = mesh_info['face_normals'][face_id]
            face_area = mesh_info['face_areas'][face_id]
            
            # Wall contributes pressure forces to residual
            # Extract pressure from flow solution
            flow_state = flow_solution[i]
            pressure = self._extract_pressure(flow_state)
            
            # Pressure contribution to momentum equations
            pressure_contribution = pressure * face_area * face_normal
            
            # Add to residual (negative because it's moved to RHS)
            residual_contribution[owner_cell, 1] -= pressure_contribution[0]
            residual_contribution[owner_cell, 2] -= pressure_contribution[1]
            residual_contribution[owner_cell, 3] -= pressure_contribution[2]
        
        return residual_contribution
    
    def _extract_pressure(self, conservative_vars: np.ndarray) -> float:
        """Extract pressure from conservative variables."""
        rho, rho_u, rho_v, rho_w, rho_E = conservative_vars
        rho = max(rho, 1e-12)
        
        u, v, w = rho_u/rho, rho_v/rho, rho_w/rho
        kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
        pressure = (self.config.gamma - 1) * (rho_E - kinetic_energy)
        
        return max(pressure, 1e-6)


class FarfieldAdjointBC(AdjointBoundaryCondition):
    """
    Farfield boundary condition for adjoint equations.
    
    Uses characteristic-based boundary conditions that properly
    handle inflow and outflow regions.
    """
    
    def __init__(self, config: Optional[AdjointBoundaryConfig] = None):
        """Initialize farfield adjoint boundary condition."""
        super().__init__(AdjointBCType.FARFIELD, config)
        
        # Compute farfield state
        self.farfield_state = self._compute_farfield_state()
        
    def _compute_farfield_state(self) -> np.ndarray:
        """Compute farfield conservative variables."""
        # Convert farfield conditions to conservative variables
        gamma = self.config.gamma
        R = self.config.gas_constant
        
        # Farfield primitive variables
        p_inf = self.config.farfield_pressure
        T_inf = self.config.farfield_temperature
        M_inf = self.config.farfield_mach
        alpha = np.radians(self.config.farfield_angle_of_attack)
        
        # Density from ideal gas law
        rho_inf = p_inf / (R * T_inf)
        
        # Speed of sound and velocity
        a_inf = np.sqrt(gamma * R * T_inf)
        V_inf = M_inf * a_inf
        
        # Velocity components
        u_inf = V_inf * np.cos(alpha)
        v_inf = V_inf * np.sin(alpha)
        w_inf = 0.0
        
        # Total energy
        E_inf = p_inf / (gamma - 1) + 0.5 * rho_inf * (u_inf**2 + v_inf**2 + w_inf**2)
        
        return np.array([rho_inf, rho_inf * u_inf, rho_inf * v_inf, rho_inf * w_inf, E_inf])
    
    def apply_boundary_condition(self,
                                adjoint_variables: np.ndarray,
                                flow_solution: np.ndarray,
                                boundary_faces: List[int],
                                mesh_info: Dict[str, Any],
                                objective_gradient: np.ndarray) -> np.ndarray:
        """Apply farfield boundary condition using characteristics."""
        modified_adjoint = adjoint_variables.copy()
        
        for i, face_id in enumerate(boundary_faces):
            face_normal = mesh_info['face_normals'][face_id]
            
            # Get local flow state
            flow_state = flow_solution[i]
            lambda_vars = modified_adjoint[i]
            
            # Apply characteristic-based boundary condition
            modified_adjoint[i] = self._apply_characteristic_bc(
                lambda_vars, flow_state, face_normal, objective_gradient[i]
            )
        
        return modified_adjoint
    
    def _apply_characteristic_bc(self,
                               lambda_vars: np.ndarray,
                               flow_state: np.ndarray,
                               normal: np.ndarray,
                               obj_gradient: np.ndarray) -> np.ndarray:
        """Apply characteristic-based adjoint boundary condition."""
        # Extract flow variables
        rho, rho_u, rho_v, rho_w, rho_E = flow_state
        rho = max(rho, 1e-12)
        
        u, v, w = rho_u/rho, rho_v/rho, rho_w/rho
        u_n = u * normal[0] + v * normal[1] + w * normal[2]
        
        # Pressure and speed of sound
        kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
        pressure = (self.config.gamma - 1) * (rho_E - kinetic_energy)
        pressure = max(pressure, 1e-6)
        a = np.sqrt(self.config.gamma * pressure / rho)
        
        # Local Mach number
        M_n = u_n / a
        
        # Characteristic variables and their adjoints
        lambda_modified = lambda_vars.copy()
        
        if M_n < -1.0:
            # Supersonic inflow: all characteristics from outside
            # Set all adjoint variables based on farfield conditions
            lambda_modified = obj_gradient.copy() if len(obj_gradient) >= 5 else lambda_vars
            
        elif M_n < 0.0:
            # Subsonic inflow: 4 characteristics from outside, 1 from inside
            # Fix density, velocity, temperature; pressure from inside
            lambda_modified[0] = obj_gradient[0] if len(obj_gradient) > 0 else 0.0
            lambda_modified[1] = obj_gradient[1] if len(obj_gradient) > 1 else 0.0
            lambda_modified[2] = obj_gradient[2] if len(obj_gradient) > 2 else 0.0
            lambda_modified[3] = obj_gradient[3] if len(obj_gradient) > 3 else 0.0
            # Keep pressure adjoint from interior (lambda_vars[4])
            
        elif M_n < 1.0:
            # Subsonic outflow: 1 characteristic from outside, 4 from inside
            # Fix pressure; other variables from inside
            lambda_modified[4] = obj_gradient[4] if len(obj_gradient) > 4 else 0.0
            # Keep other adjoints from interior
            
        else:
            # Supersonic outflow: all characteristics from inside
            # No modification needed - all adjoints from interior
            pass
        
        return lambda_modified
    
    def compute_boundary_residual_contribution(self,
                                             adjoint_variables: np.ndarray,
                                             flow_solution: np.ndarray,
                                             boundary_faces: List[int],
                                             mesh_info: Dict[str, Any]) -> np.ndarray:
        """Compute farfield boundary residual contribution."""
        n_cells = adjoint_variables.shape[0]
        residual_contribution = np.zeros((n_cells, 5))
        
        # Farfield typically has minimal residual contribution
        # Main effect is through characteristic boundary conditions
        
        return residual_contribution


class SymmetryAdjointBC(AdjointBoundaryCondition):
    """
    Symmetry boundary condition for adjoint equations.
    
    Enforces symmetry conditions on adjoint variables.
    """
    
    def __init__(self, config: Optional[AdjointBoundaryConfig] = None):
        """Initialize symmetry adjoint boundary condition."""
        super().__init__(AdjointBCType.SYMMETRY, config)
    
    def apply_boundary_condition(self,
                                adjoint_variables: np.ndarray,
                                flow_solution: np.ndarray,
                                boundary_faces: List[int],
                                mesh_info: Dict[str, Any],
                                objective_gradient: np.ndarray) -> np.ndarray:
        """Apply symmetry boundary condition."""
        modified_adjoint = adjoint_variables.copy()
        
        for i, face_id in enumerate(boundary_faces):
            face_normal = mesh_info['face_normals'][face_id]
            lambda_vars = modified_adjoint[i]
            
            # For symmetry: normal component of momentum adjoint = 0
            # Tangential components unchanged
            lambda_rho_u, lambda_rho_v, lambda_rho_w = lambda_vars[1:4]
            nx, ny, nz = face_normal
            
            # Normal component of momentum adjoint
            normal_lambda_momentum = lambda_rho_u * nx + lambda_rho_v * ny + lambda_rho_w * nz
            
            # Remove normal component
            modified_adjoint[i, 1] -= normal_lambda_momentum * nx
            modified_adjoint[i, 2] -= normal_lambda_momentum * ny
            modified_adjoint[i, 3] -= normal_lambda_momentum * nz
        
        return modified_adjoint
    
    def compute_boundary_residual_contribution(self,
                                             adjoint_variables: np.ndarray,
                                             flow_solution: np.ndarray,
                                             boundary_faces: List[int],
                                             mesh_info: Dict[str, Any]) -> np.ndarray:
        """Compute symmetry boundary residual contribution."""
        n_cells = adjoint_variables.shape[0]
        return np.zeros((n_cells, 5))  # No residual contribution for symmetry


class AdjointBoundaryConditions:
    """
    Manager for adjoint boundary conditions.
    
    Handles application of boundary conditions to adjoint variables
    and computation of boundary contributions to adjoint residuals.
    """
    
    def __init__(self, config: Optional[AdjointBoundaryConfig] = None):
        """
        Initialize adjoint boundary conditions manager.
        
        Args:
            config: Configuration for boundary conditions
        """
        self.config = config or AdjointBoundaryConfig()
        
        # Registry of boundary conditions
        self.boundary_conditions: Dict[int, AdjointBoundaryCondition] = {}
        
        # Statistics
        self.application_count = 0
        self.residual_contribution_norm = 0.0
        
    def register_boundary_condition(self, 
                                  boundary_id: int,
                                  bc_type: AdjointBCType,
                                  **kwargs) -> None:
        """
        Register boundary condition for a boundary.
        
        Args:
            boundary_id: Boundary identifier
            bc_type: Type of boundary condition
            **kwargs: Additional parameters for boundary condition
        """
        if bc_type == AdjointBCType.WALL:
            wall_type = kwargs.get('wall_type', 'adiabatic')
            bc = WallAdjointBC(wall_type, self.config)
            
        elif bc_type == AdjointBCType.FARFIELD:
            bc = FarfieldAdjointBC(self.config)
            
        elif bc_type == AdjointBCType.SYMMETRY:
            bc = SymmetryAdjointBC(self.config)
            
        else:
            raise ValueError(f"Unsupported adjoint boundary condition type: {bc_type}")
        
        self.boundary_conditions[boundary_id] = bc
        logger.info(f"Registered {bc_type.value} adjoint BC for boundary {boundary_id}")
    
    def apply_all_boundary_conditions(self,
                                    adjoint_variables: np.ndarray,
                                    flow_solution: np.ndarray,
                                    mesh_info: Dict[str, Any],
                                    boundary_info: Dict[str, Any],
                                    objective_gradients: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Apply all registered boundary conditions.
        
        Args:
            adjoint_variables: Adjoint variables [n_cells, n_vars]
            flow_solution: Flow solution [n_cells, n_vars]
            mesh_info: Mesh information
            boundary_info: Boundary information
            objective_gradients: Objective function gradients at boundaries
            
        Returns:
            Modified adjoint variables with boundary conditions applied
        """
        modified_adjoint = adjoint_variables.copy()
        
        for boundary_id, bc in self.boundary_conditions.items():
            if boundary_id in boundary_info:
                boundary_data = boundary_info[boundary_id]
                boundary_faces = boundary_data.get('faces', [])
                
                if len(boundary_faces) == 0:
                    continue
                
                # Get boundary cell data
                boundary_cells = [mesh_info['face_owners'][face_id] for face_id in boundary_faces]
                adjoint_boundary = modified_adjoint[boundary_cells]
                flow_boundary = flow_solution[boundary_cells]
                
                # Get objective gradients for this boundary
                obj_gradient = objective_gradients.get(boundary_id, np.zeros((len(boundary_faces), 5)))
                
                # Apply boundary condition
                adjoint_boundary_modified = bc.apply_boundary_condition(
                    adjoint_boundary, flow_boundary, boundary_faces, mesh_info, obj_gradient
                )
                
                # Update adjoint variables
                modified_adjoint[boundary_cells] = adjoint_boundary_modified
                
        self.application_count += 1
        return modified_adjoint
    
    def compute_boundary_residual_contributions(self,
                                              adjoint_variables: np.ndarray,
                                              flow_solution: np.ndarray,
                                              mesh_info: Dict[str, Any],
                                              boundary_info: Dict[str, Any]) -> np.ndarray:
        """
        Compute boundary contributions to adjoint residual.
        
        Returns:
            Boundary residual contributions [n_cells, n_vars]
        """
        total_residual = np.zeros_like(adjoint_variables)
        
        for boundary_id, bc in self.boundary_conditions.items():
            if boundary_id in boundary_info:
                boundary_data = boundary_info[boundary_id]
                boundary_faces = boundary_data.get('faces', [])
                
                if len(boundary_faces) == 0:
                    continue
                
                # Get boundary data
                boundary_cells = [mesh_info['face_owners'][face_id] for face_id in boundary_faces]
                adjoint_boundary = adjoint_variables[boundary_cells]
                flow_boundary = flow_solution[boundary_cells]
                
                # Compute boundary residual contribution
                boundary_residual = bc.compute_boundary_residual_contribution(
                    adjoint_boundary, flow_boundary, boundary_faces, mesh_info
                )
                
                # Accumulate contributions
                total_residual += boundary_residual
        
        # Store statistics
        self.residual_contribution_norm = np.linalg.norm(total_residual)
        
        return total_residual
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get boundary condition statistics."""
        return {
            'n_boundary_conditions': len(self.boundary_conditions),
            'boundary_types': [bc.bc_type.value for bc in self.boundary_conditions.values()],
            'application_count': self.application_count,
            'residual_contribution_norm': self.residual_contribution_norm
        }


def create_adjoint_boundary_conditions(boundary_info: Dict[str, Any],
                                     config: Optional[AdjointBoundaryConfig] = None) -> AdjointBoundaryConditions:
    """
    Factory function to create adjoint boundary conditions.
    
    Args:
        boundary_info: Dictionary mapping boundary IDs to boundary types
        config: Configuration
        
    Returns:
        Configured AdjointBoundaryConditions manager
    """
    adjoint_bcs = AdjointBoundaryConditions(config)
    
    for boundary_id, boundary_data in boundary_info.items():
        bc_type_str = boundary_data.get('type', 'wall')
        
        # Map string to enum
        bc_type_map = {
            'wall': AdjointBCType.WALL,
            'farfield': AdjointBCType.FARFIELD,
            'inlet': AdjointBCType.INLET,
            'outlet': AdjointBCType.OUTLET,
            'symmetry': AdjointBCType.SYMMETRY
        }
        
        bc_type = bc_type_map.get(bc_type_str, AdjointBCType.WALL)
        
        # Extract additional parameters
        kwargs = {k: v for k, v in boundary_data.items() if k not in ['type', 'faces']}
        
        adjoint_bcs.register_boundary_condition(boundary_id, bc_type, **kwargs)
    
    return adjoint_bcs


def test_adjoint_boundary_conditions():
    """Test adjoint boundary conditions."""
    print("Testing Adjoint Boundary Conditions:")
    
    # Create test data
    n_cells = 50
    n_vars = 5
    
    adjoint_variables = np.random.rand(n_cells, n_vars)
    flow_solution = np.random.rand(n_cells, n_vars) + 1.0
    
    # Mock mesh info
    mesh_info = {
        'face_normals': {0: np.array([1.0, 0.0, 0.0]), 1: np.array([0.0, 1.0, 0.0])},
        'face_areas': {0: 1.0, 1: 1.0},
        'face_owners': {0: 0, 1: 1}
    }
    
    boundary_info = {
        1: {'type': 'wall', 'faces': [0]},
        2: {'type': 'farfield', 'faces': [1]}
    }
    
    # Create adjoint boundary conditions
    config = AdjointBoundaryConfig()
    adjoint_bcs = create_adjoint_boundary_conditions(boundary_info, config)
    
    # Test boundary condition application
    objective_gradients = {
        1: np.array([[0.1, 0.2, 0.0, 0.0, 0.3]]),
        2: np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    }
    
    modified_adjoint = adjoint_bcs.apply_all_boundary_conditions(
        adjoint_variables, flow_solution, mesh_info, boundary_info, objective_gradients
    )
    
    print(f"  Original adjoint norm: {np.linalg.norm(adjoint_variables):.6f}")
    print(f"  Modified adjoint norm: {np.linalg.norm(modified_adjoint):.6f}")
    
    # Test boundary residual contributions
    boundary_residual = adjoint_bcs.compute_boundary_residual_contributions(
        modified_adjoint, flow_solution, mesh_info, boundary_info
    )
    
    print(f"  Boundary residual norm: {np.linalg.norm(boundary_residual):.6f}")
    print(f"  Statistics: {adjoint_bcs.get_statistics()}")


if __name__ == "__main__":
    test_adjoint_boundary_conditions()