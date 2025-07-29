"""
3D Boundary Condition Handling for CFD Simulations

Implements comprehensive boundary condition treatment for Euler and Navier-Stokes equations:
- Farfield boundary conditions (supersonic/subsonic inflow/outflow)
- Wall boundary conditions (slip/no-slip, adiabatic/isothermal)
- Symmetry boundary conditions
- Inlet/outlet boundary conditions with prescribed conditions
- Characteristic-based boundary conditions for proper wave treatment
- Ghost cell methodology for boundary implementation

Supports both inviscid and viscous flows with proper physical modeling.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

# Import OpenFOAM boundary condition parser
from .openfoam_boundary_parser import OpenFOAMBoundaryParser, parse_openfoam_boundary_conditions

logger = logging.getLogger(__name__)


class BoundaryType(Enum):
    """Enumeration of boundary condition types."""
    FARFIELD = "farfield"
    WALL = "wall"
    SYMMETRY = "symmetry"
    INLET = "inlet"
    OUTLET = "outlet"
    PERIODIC = "periodic"
    INTERFACE = "interface"


@dataclass
class BoundaryConditionConfig:
    """Configuration for boundary conditions."""
    
    # Global flow conditions
    mach_number: float = 0.8
    reynolds_number: float = 1e6
    angle_of_attack: float = 0.0  # degrees
    angle_of_sideslip: float = 0.0  # degrees
    
    # Reference conditions
    reference_pressure: float = 101325.0  # Pa
    reference_temperature: float = 288.15  # K
    reference_density: float = 1.225  # kg/m³
    reference_velocity: float = 250.0  # m/s
    
    # Gas properties
    gamma: float = 1.4  # Specific heat ratio
    gas_constant: float = 287.0  # J/(kg·K)
    prandtl_number: float = 0.72
    
    # Wall conditions
    wall_temperature: float = 300.0  # K (for isothermal walls)
    wall_heat_flux: float = 0.0  # W/m² (for heat flux walls)
    is_adiabatic: bool = True
    is_slip_wall: bool = False
    
    # Characteristic boundary treatment
    use_characteristic_bc: bool = True
    artificial_compressibility: float = 0.0  # For incompressible flows
    
    # Numerical parameters
    ghost_cell_layers: int = 2
    boundary_gradient_method: str = "least_squares"  # "least_squares", "green_gauss"


@dataclass
class BoundaryPatch:
    """Represents a boundary patch with associated conditions."""
    
    patch_id: int
    patch_name: str
    boundary_type: BoundaryType
    face_ids: List[int]
    
    # Boundary-specific parameters
    prescribed_values: Optional[Dict[str, float]] = None
    normal_direction: Optional[np.ndarray] = None
    
    # Inlet/outlet specific
    total_pressure: Optional[float] = None
    total_temperature: Optional[float] = None
    static_pressure: Optional[float] = None
    mass_flow_rate: Optional[float] = None
    velocity_direction: Optional[np.ndarray] = None


class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions."""
    
    def __init__(self, config: BoundaryConditionConfig, patch: BoundaryPatch):
        """Initialize boundary condition."""
        self.config = config
        self.patch = patch
        self.boundary_type = patch.boundary_type
        
    @abstractmethod
    def apply_boundary_condition(self,
                                interior_state: np.ndarray,
                                face_normal: np.ndarray,
                                face_area: float,
                                **kwargs) -> np.ndarray:
        """
        Apply boundary condition to compute ghost state or boundary flux.
        
        Args:
            interior_state: Conservative variables from interior cell
            face_normal: Outward-pointing face normal vector
            face_area: Face area
            
        Returns:
            Ghost state or boundary flux
        """
        pass
    
    def _conservative_to_primitive(self, conservative: np.ndarray) -> np.ndarray:
        """Convert conservative variables to primitive variables."""
        rho, rho_u, rho_v, rho_w, rho_E = conservative
        rho = max(rho, 1e-12)
        
        u, v, w = rho_u/rho, rho_v/rho, rho_w/rho
        velocity_magnitude_sq = u**2 + v**2 + w**2
        
        # Compute pressure
        kinetic_energy = 0.5 * rho * velocity_magnitude_sq
        pressure = (self.config.gamma - 1) * (rho_E - kinetic_energy)
        pressure = max(pressure, 1e-12)
        
        # Compute temperature
        temperature = pressure / (rho * self.config.gas_constant)
        
        return np.array([rho, u, v, w, pressure, temperature])
    
    def _primitive_to_conservative(self, primitive: np.ndarray) -> np.ndarray:
        """Convert primitive variables to conservative variables."""
        rho, u, v, w, pressure, temperature = primitive
        
        rho_u = rho * u
        rho_v = rho * v
        rho_w = rho * w
        
        velocity_magnitude_sq = u**2 + v**2 + w**2
        kinetic_energy = 0.5 * rho * velocity_magnitude_sq
        
        # Total energy
        internal_energy = pressure / ((self.config.gamma - 1) * rho)
        rho_E = rho * internal_energy + kinetic_energy
        
        return np.array([rho, rho_u, rho_v, rho_w, rho_E])


class FarfieldBoundaryCondition(BoundaryCondition):
    """
    Farfield boundary condition for external flows.
    
    Uses characteristic-based treatment to handle inflow/outflow
    automatically based on local Mach number.
    """
    
    def __init__(self, config: BoundaryConditionConfig, patch: BoundaryPatch):
        super().__init__(config, patch)
        
        # Compute freestream conditions
        self.freestream_primitive = self._compute_freestream_conditions()
        self.freestream_conservative = self._primitive_to_conservative(self.freestream_primitive)
        
    def _compute_freestream_conditions(self) -> np.ndarray:
        """Compute freestream primitive variables."""
        # Freestream velocity components
        alpha_rad = np.radians(self.config.angle_of_attack)
        beta_rad = np.radians(self.config.angle_of_sideslip)
        
        u_inf = self.config.reference_velocity * np.cos(alpha_rad) * np.cos(beta_rad)
        v_inf = self.config.reference_velocity * np.sin(beta_rad)
        w_inf = self.config.reference_velocity * np.sin(alpha_rad) * np.cos(beta_rad)
        
        return np.array([
            self.config.reference_density,
            u_inf,
            v_inf, 
            w_inf,
            self.config.reference_pressure,
            self.config.reference_temperature
        ])
    
    def apply_boundary_condition(self,
                                interior_state: np.ndarray,
                                face_normal: np.ndarray,
                                face_area: float,
                                **kwargs) -> np.ndarray:
        """Apply farfield boundary condition using characteristic analysis."""
        if self.config.use_characteristic_bc:
            return self._apply_characteristic_farfield(interior_state, face_normal)
        else:
            return self._apply_simple_farfield(interior_state, face_normal)
    
    def _apply_characteristic_farfield(self,
                                     interior_state: np.ndarray,
                                     face_normal: np.ndarray) -> np.ndarray:
        """Apply characteristic-based farfield boundary condition."""
        # Convert to primitive variables
        interior_primitive = self._conservative_to_primitive(interior_state)
        rho_i, u_i, v_i, w_i, p_i, T_i = interior_primitive
        
        # Freestream primitive variables
        rho_inf, u_inf, v_inf, w_inf, p_inf, T_inf = self.freestream_primitive
        
        # Compute normal velocity components
        u_n_i = u_i * face_normal[0] + v_i * face_normal[1] + w_i * face_normal[2]
        u_n_inf = u_inf * face_normal[0] + v_inf * face_normal[1] + w_inf * face_normal[2]
        
        # Compute speed of sound
        a_i = np.sqrt(self.config.gamma * p_i / rho_i)
        a_inf = np.sqrt(self.config.gamma * p_inf / rho_inf)
        
        # Compute local Mach number
        M_n = u_n_i / a_i
        
        # Apply characteristic boundary conditions based on Mach number
        if M_n <= -1.0:
            # Supersonic inflow: all characteristics from outside
            ghost_primitive = self.freestream_primitive.copy()
            
        elif M_n < 0.0:
            # Subsonic inflow: 4 characteristics from outside, 1 from inside
            ghost_primitive = self.freestream_primitive.copy()
            
            # Pressure from interior (outgoing characteristic)
            R_plus = u_n_i + 2 * a_i / (self.config.gamma - 1)
            a_ghost = 0.5 * (self.config.gamma - 1) * (R_plus - u_n_inf)
            p_ghost = (a_ghost**2 * rho_inf) / self.config.gamma
            
            ghost_primitive[4] = p_ghost
            
        elif M_n < 1.0:
            # Subsonic outflow: 1 characteristic from outside, 4 from inside
            ghost_primitive = interior_primitive.copy()
            
            # Pressure from outside (incoming characteristic)
            ghost_primitive[4] = p_inf
            
        else:
            # Supersonic outflow: all characteristics from inside
            ghost_primitive = interior_primitive.copy()
        
        # Convert back to conservative variables
        return self._primitive_to_conservative(ghost_primitive)
    
    def _apply_simple_farfield(self,
                              interior_state: np.ndarray,
                              face_normal: np.ndarray) -> np.ndarray:
        """Apply simple farfield boundary condition."""
        # Simple approach: use freestream conditions
        return self.freestream_conservative.copy()


class WallBoundaryCondition(BoundaryCondition):
    """
    Wall boundary condition for solid walls.
    
    Supports both slip and no-slip conditions, adiabatic and isothermal walls.
    """
    
    def apply_boundary_condition(self,
                                interior_state: np.ndarray,
                                face_normal: np.ndarray,
                                face_area: float,
                                **kwargs) -> np.ndarray:
        """Apply wall boundary condition."""
        interior_primitive = self._conservative_to_primitive(interior_state)
        rho_i, u_i, v_i, w_i, p_i, T_i = interior_primitive
        
        # Ghost state primitive variables
        ghost_primitive = interior_primitive.copy()
        
        if self.config.is_slip_wall:
            # Slip wall: reflect normal velocity component
            velocity_vector = np.array([u_i, v_i, w_i])
            normal_velocity = np.dot(velocity_vector, face_normal)
            
            # Remove normal component (reflect)
            tangential_velocity = velocity_vector - normal_velocity * face_normal
            
            ghost_primitive[1:4] = -tangential_velocity  # Mirror for ghost cell
            
        else:
            # No-slip wall: zero velocity
            ghost_primitive[1:4] = np.array([0.0, 0.0, 0.0])
        
        # Pressure: zero normal gradient (dp/dn = 0)
        ghost_primitive[4] = p_i
        
        # Temperature boundary condition
        if self.config.is_adiabatic:
            # Adiabatic wall: zero temperature gradient
            ghost_primitive[5] = T_i
        else:
            # Isothermal wall: prescribed temperature
            T_wall = self.config.wall_temperature
            ghost_primitive[5] = 2 * T_wall - T_i  # Linear extrapolation
        
        # Density from ideal gas law
        ghost_primitive[0] = ghost_primitive[4] / (self.config.gas_constant * ghost_primitive[5])
        
        return self._primitive_to_conservative(ghost_primitive)


class SymmetryBoundaryCondition(BoundaryCondition):
    """
    Symmetry boundary condition.
    
    Reflects normal velocity component while preserving tangential components.
    """
    
    def apply_boundary_condition(self,
                                interior_state: np.ndarray,
                                face_normal: np.ndarray,
                                face_area: float,
                                **kwargs) -> np.ndarray:
        """Apply symmetry boundary condition."""
        interior_primitive = self._conservative_to_primitive(interior_state)
        rho_i, u_i, v_i, w_i, p_i, T_i = interior_primitive
        
        # Reflect normal velocity component
        velocity_vector = np.array([u_i, v_i, w_i])
        normal_velocity = np.dot(velocity_vector, face_normal)
        
        # Symmetry: reverse normal component, keep tangential
        reflected_velocity = velocity_vector - 2 * normal_velocity * face_normal
        
        # Ghost state
        ghost_primitive = interior_primitive.copy()
        ghost_primitive[1:4] = reflected_velocity
        
        return self._primitive_to_conservative(ghost_primitive)


class InletBoundaryCondition(BoundaryCondition):
    """
    Inlet boundary condition with prescribed total conditions.
    
    Supports total pressure/temperature or mass flow rate specification.
    """
    
    def apply_boundary_condition(self,
                                interior_state: np.ndarray,
                                face_normal: np.ndarray,
                                face_area: float,
                                **kwargs) -> np.ndarray:
        """Apply inlet boundary condition."""
        if self.patch.total_pressure is not None and self.patch.total_temperature is not None:
            return self._apply_total_conditions_inlet(interior_state, face_normal)
        elif self.patch.mass_flow_rate is not None:
            return self._apply_mass_flow_inlet(interior_state, face_normal)
        else:
            # Default: use prescribed velocity
            return self._apply_velocity_inlet(interior_state, face_normal)
    
    def _apply_total_conditions_inlet(self,
                                    interior_state: np.ndarray,
                                    face_normal: np.ndarray) -> np.ndarray:
        """Apply inlet with total pressure and temperature."""
        interior_primitive = self._conservative_to_primitive(interior_state)
        rho_i, u_i, v_i, w_i, p_i, T_i = interior_primitive
        
        # Total conditions
        p0 = self.patch.total_pressure
        T0 = self.patch.total_temperature
        
        # Estimate Mach number from interior state
        velocity_magnitude = np.sqrt(u_i**2 + v_i**2 + w_i**2)
        a_i = np.sqrt(self.config.gamma * p_i / rho_i)
        M_estimate = velocity_magnitude / a_i
        
        # Isentropic relations
        gamma = self.config.gamma
        T_static = T0 / (1 + 0.5 * (gamma - 1) * M_estimate**2)
        p_static = p0 * (T_static / T0)**(gamma / (gamma - 1))
        rho_static = p_static / (self.config.gas_constant * T_static)
        
        # Velocity magnitude from energy equation
        a_static = np.sqrt(gamma * p_static / rho_static)
        velocity_mag = M_estimate * a_static
        
        # Velocity direction (inward normal or prescribed)
        if self.patch.velocity_direction is not None:
            vel_direction = self.patch.velocity_direction / np.linalg.norm(self.patch.velocity_direction)
        else:
            vel_direction = -face_normal  # Inward flow
        
        velocity_vector = velocity_mag * vel_direction
        
        # Ghost state
        ghost_primitive = np.array([
            rho_static,
            velocity_vector[0],
            velocity_vector[1], 
            velocity_vector[2],
            p_static,
            T_static
        ])
        
        return self._primitive_to_conservative(ghost_primitive)
    
    def _apply_mass_flow_inlet(self,
                              interior_state: np.ndarray,
                              face_normal: np.ndarray) -> np.ndarray:
        """Apply inlet with prescribed mass flow rate."""
        # Simplified implementation
        target_mass_flow = self.patch.mass_flow_rate
        
        # Estimate required velocity
        rho_estimate = self.config.reference_density
        velocity_magnitude = target_mass_flow / (rho_estimate * face_area)
        
        velocity_vector = velocity_magnitude * (-face_normal)  # Inward flow
        
        ghost_primitive = np.array([
            rho_estimate,
            velocity_vector[0],
            velocity_vector[1],
            velocity_vector[2],
            self.config.reference_pressure,
            self.config.reference_temperature
        ])
        
        return self._primitive_to_conservative(ghost_primitive)
    
    def _apply_velocity_inlet(self,
                            interior_state: np.ndarray,
                            face_normal: np.ndarray) -> np.ndarray:
        """Apply inlet with prescribed velocity."""
        if self.patch.prescribed_values is None:
            # Use freestream conditions
            velocity_vector = np.array([
                self.config.reference_velocity * np.cos(np.radians(self.config.angle_of_attack)),
                0.0,
                self.config.reference_velocity * np.sin(np.radians(self.config.angle_of_attack))
            ])
        else:
            velocity_vector = np.array([
                self.patch.prescribed_values.get('u', 0.0),
                self.patch.prescribed_values.get('v', 0.0),
                self.patch.prescribed_values.get('w', 0.0)
            ])
        
        ghost_primitive = np.array([
            self.config.reference_density,
            velocity_vector[0],
            velocity_vector[1],
            velocity_vector[2],
            self.config.reference_pressure,
            self.config.reference_temperature
        ])
        
        return self._primitive_to_conservative(ghost_primitive)


class OutletBoundaryCondition(BoundaryCondition):
    """
    Outlet boundary condition.
    
    Supports prescribed pressure or extrapolation conditions.
    """
    
    def apply_boundary_condition(self,
                                interior_state: np.ndarray,
                                face_normal: np.ndarray,
                                face_area: float,
                                **kwargs) -> np.ndarray:
        """Apply outlet boundary condition."""
        interior_primitive = self._conservative_to_primitive(interior_state)
        
        if self.patch.static_pressure is not None:
            return self._apply_pressure_outlet(interior_state, face_normal)
        else:
            return self._apply_extrapolation_outlet(interior_state, face_normal)
    
    def _apply_pressure_outlet(self,
                              interior_state: np.ndarray,
                              face_normal: np.ndarray) -> np.ndarray:
        """Apply outlet with prescribed static pressure."""
        interior_primitive = self._conservative_to_primitive(interior_state)
        rho_i, u_i, v_i, w_i, p_i, T_i = interior_primitive
        
        # Prescribed pressure
        p_outlet = self.patch.static_pressure
        
        # Extrapolate other quantities
        ghost_primitive = interior_primitive.copy()
        ghost_primitive[4] = p_outlet
        
        # Adjust density for consistency
        ghost_primitive[0] = p_outlet / (self.config.gas_constant * T_i)
        
        return self._primitive_to_conservative(ghost_primitive)
    
    def _apply_extrapolation_outlet(self,
                                   interior_state: np.ndarray,
                                   face_normal: np.ndarray) -> np.ndarray:
        """Apply outlet with zero-gradient extrapolation."""
        # Simple extrapolation: ghost state = interior state
        return interior_state.copy()


class BoundaryConditionManager:
    """
    Manager for all boundary conditions in the computational domain.
    
    Handles multiple boundary patches and applies appropriate conditions.
    """
    
    def __init__(self, config: BoundaryConditionConfig):
        """Initialize boundary condition manager."""
        self.config = config
        self.boundary_patches: Dict[int, BoundaryPatch] = {}
        self.boundary_conditions: Dict[int, BoundaryCondition] = {}
        
    def add_boundary_patch(self, patch: BoundaryPatch) -> None:
        """Add a boundary patch."""
        self.boundary_patches[patch.patch_id] = patch
        
        # Create appropriate boundary condition
        if patch.boundary_type == BoundaryType.FARFIELD:
            bc = FarfieldBoundaryCondition(self.config, patch)
        elif patch.boundary_type == BoundaryType.WALL:
            bc = WallBoundaryCondition(self.config, patch)
        elif patch.boundary_type == BoundaryType.SYMMETRY:
            bc = SymmetryBoundaryCondition(self.config, patch)
        elif patch.boundary_type == BoundaryType.INLET:
            bc = InletBoundaryCondition(self.config, patch)
        elif patch.boundary_type == BoundaryType.OUTLET:
            bc = OutletBoundaryCondition(self.config, patch)
        else:
            raise ValueError(f"Unsupported boundary type: {patch.boundary_type}")
        
        self.boundary_conditions[patch.patch_id] = bc
        
        logger.info(f"Added boundary patch '{patch.patch_name}' "
                   f"({patch.boundary_type.value}) with {len(patch.face_ids)} faces")
    
    def apply_boundary_conditions(self,
                                 solution: np.ndarray,
                                 mesh_info: Dict[str, Any]) -> Dict[int, np.ndarray]:
        """
        Apply boundary conditions to compute ghost states.
        
        Args:
            solution: Current solution [n_cells, 5]
            mesh_info: Mesh connectivity and geometry information
            
        Returns:
            Dictionary of ghost states by boundary face
        """
        ghost_states = {}
        
        for patch_id, bc in self.boundary_conditions.items():
            patch = self.boundary_patches[patch_id]
            
            for face_id in patch.face_ids:
                # Get face information
                if face_id >= len(mesh_info['face_owners']):
                    continue
                    
                owner_cell = mesh_info['face_owners'][face_id]
                face_normal = mesh_info['face_normals'][face_id]
                face_area = mesh_info['face_areas'][face_id]
                
                # Interior state
                interior_state = solution[owner_cell]
                
                # Apply boundary condition
                ghost_state = bc.apply_boundary_condition(
                    interior_state, face_normal, face_area
                )
                
                ghost_states[face_id] = ghost_state
        
        return ghost_states
    
    def compute_boundary_fluxes(self,
                               solution: np.ndarray,
                               mesh_info: Dict[str, Any],
                               riemann_solver: Callable) -> Dict[int, np.ndarray]:
        """
        Compute boundary fluxes using Riemann solver.
        
        Args:
            solution: Current solution
            mesh_info: Mesh information
            riemann_solver: Riemann solver function
            
        Returns:
            Dictionary of boundary fluxes
        """
        boundary_fluxes = {}
        ghost_states = self.apply_boundary_conditions(solution, mesh_info)
        
        for face_id, ghost_state in ghost_states.items():
            if face_id >= len(mesh_info['face_owners']):
                continue
                
            owner_cell = mesh_info['face_owners'][face_id]
            face_normal = mesh_info['face_normals'][face_id]
            
            interior_state = solution[owner_cell]
            
            # Compute flux using Riemann solver
            flux = riemann_solver(interior_state, ghost_state, face_normal)
            boundary_fluxes[face_id] = flux
        
        return boundary_fluxes
    
    def get_boundary_statistics(self) -> Dict[str, Any]:
        """Get statistics about boundary conditions."""
        stats = {
            'n_patches': len(self.boundary_patches),
            'total_boundary_faces': sum(len(patch.face_ids) for patch in self.boundary_patches.values()),
            'patch_types': {}
        }
        
        for patch in self.boundary_patches.values():
            bc_type = patch.boundary_type.value
            if bc_type not in stats['patch_types']:
                stats['patch_types'][bc_type] = {'count': 0, 'faces': 0}
            stats['patch_types'][bc_type]['count'] += 1
            stats['patch_types'][bc_type]['faces'] += len(patch.face_ids)
        
        return stats


def create_boundary_patch(patch_id: int,
                         patch_name: str,
                         boundary_type: str,
                         face_ids: List[int],
                         **kwargs) -> BoundaryPatch:
    """
    Factory function for creating boundary patches.
    
    Args:
        patch_id: Unique patch identifier
        patch_name: Descriptive name
        boundary_type: Type of boundary condition
        face_ids: List of face IDs in this patch
        **kwargs: Additional boundary-specific parameters
        
    Returns:
        Configured boundary patch
    """
    bc_type = BoundaryType(boundary_type)
    
    patch = BoundaryPatch(
        patch_id=patch_id,
        patch_name=patch_name,
        boundary_type=bc_type,
        face_ids=face_ids
    )
    
    # Set boundary-specific parameters
    if 'prescribed_values' in kwargs:
        patch.prescribed_values = kwargs['prescribed_values']
    if 'total_pressure' in kwargs:
        patch.total_pressure = kwargs['total_pressure']
    if 'total_temperature' in kwargs:
        patch.total_temperature = kwargs['total_temperature']
    if 'static_pressure' in kwargs:
        patch.static_pressure = kwargs['static_pressure']
    if 'mass_flow_rate' in kwargs:
        patch.mass_flow_rate = kwargs['mass_flow_rate']
    if 'velocity_direction' in kwargs:
        patch.velocity_direction = np.array(kwargs['velocity_direction'])
    
    return patch


def create_boundary_manager_from_openfoam(case_directory: str, 
                                         flow_config: Optional[Dict[str, float]] = None) -> BoundaryConditionManager:
    """
    Create boundary condition manager from OpenFOAM case directory.
    
    Args:
        case_directory: Path to OpenFOAM case directory
        flow_config: Optional flow configuration overrides
        
    Returns:
        Configured boundary condition manager
    """
    # Parse OpenFOAM boundary conditions
    openfoam_bcs = parse_openfoam_boundary_conditions(case_directory)
    
    # Extract flow conditions from OpenFOAM data or use defaults
    if flow_config is None:
        flow_config = {}
        
    # Create boundary condition configuration
    config = BoundaryConditionConfig(
        mach_number=flow_config.get('mach_number', 0.1),
        reynolds_number=flow_config.get('reynolds_number', 100.0),
        angle_of_attack=flow_config.get('angle_of_attack', 0.0),
        reference_pressure=flow_config.get('reference_pressure', 101325.0),
        reference_temperature=flow_config.get('reference_temperature', 288.15),
        reference_density=flow_config.get('reference_density', 1.225),
        reference_velocity=flow_config.get('reference_velocity', 1.0)
    )
    
    # Create boundary condition manager
    bc_manager = BoundaryConditionManager(config)
    
    # Convert OpenFOAM patches to boundary patches
    patches = openfoam_bcs['summary']['patch_names']
    
    for patch_id, patch_name in enumerate(patches):
        # Determine boundary type from OpenFOAM conditions
        bc_type = _determine_boundary_type_from_openfoam(patch_name, openfoam_bcs['patches'][patch_name])
        
        # Extract boundary-specific parameters
        kwargs = _extract_boundary_parameters_from_openfoam(patch_name, openfoam_bcs['patches'][patch_name])
        
        # Create boundary patch (face IDs would come from mesh reader)
        patch = create_boundary_patch(
            patch_id=patch_id,
            patch_name=patch_name,
            boundary_type=bc_type,
            face_ids=[],  # Will be populated by mesh reader
            **kwargs
        )
        
        bc_manager.add_boundary_patch(patch)
        
    logger.info(f"Created boundary condition manager with {len(patches)} patches from OpenFOAM case")
    
    return bc_manager


def _determine_boundary_type_from_openfoam(patch_name: str, openfoam_patch_data: Dict[str, Any]) -> str:
    """Determine boundary type from OpenFOAM patch data."""
    
    # Check velocity boundary condition to determine type
    if 'U' in openfoam_patch_data:
        u_bc_type = openfoam_patch_data['U']['type']
        
        if u_bc_type == 'fixedValue':
            # Check if velocity is zero (wall) or non-zero (inlet)
            velocity_value = openfoam_patch_data['U']['parameters'].get('value', [0, 0, 0])
            if isinstance(velocity_value, list) and all(v == 0 for v in velocity_value):
                return 'wall'
            else:
                return 'inlet'
        elif u_bc_type in ['inletOutlet', 'outletInlet']:
            return 'farfield'
        elif u_bc_type == 'symmetry':
            return 'symmetry'
        elif u_bc_type == 'zeroGradient':
            return 'outlet'
    
    # Fallback: determine from patch name
    patch_lower = patch_name.lower()
    if 'wall' in patch_lower or 'cylinder' in patch_lower:
        return 'wall'
    elif 'symmetry' in patch_lower:
        return 'symmetry'
    elif 'inlet' in patch_lower or 'inflow' in patch_lower:
        return 'inlet'
    elif 'outlet' in patch_lower or 'outflow' in patch_lower:
        return 'outlet'
    elif 'farfield' in patch_lower or 'freestream' in patch_lower or 'inout' in patch_lower:
        return 'farfield'
    else:
        return 'wall'  # Default to wall


def _extract_boundary_parameters_from_openfoam(patch_name: str, openfoam_patch_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract boundary-specific parameters from OpenFOAM data."""
    kwargs = {}
    
    # Extract velocity values for inlet boundaries
    if 'U' in openfoam_patch_data:
        u_params = openfoam_patch_data['U']['parameters']
        if 'inletValue' in u_params:
            velocity = u_params['inletValue']
            if isinstance(velocity, list) and len(velocity) >= 3:
                kwargs['prescribed_values'] = {
                    'u': velocity[0], 'v': velocity[1], 'w': velocity[2]
                }
        elif 'value' in u_params and u_params['value'] != '$internalField':
            velocity = u_params['value']
            if isinstance(velocity, list) and len(velocity) >= 3:
                kwargs['prescribed_values'] = {
                    'u': velocity[0], 'v': velocity[1], 'w': velocity[2]
                }
    
    # Extract pressure values for outlet boundaries
    if 'p' in openfoam_patch_data:
        p_params = openfoam_patch_data['p']['parameters']
        if 'outletValue' in p_params:
            kwargs['static_pressure'] = p_params['outletValue']
        elif 'value' in p_params and isinstance(p_params['value'], (int, float)):
            kwargs['static_pressure'] = p_params['value']
    
    return kwargs


def test_boundary_conditions():
    """Test boundary condition implementations."""
    print("Testing 3D Boundary Conditions:")
    
    # Create configuration
    config = BoundaryConditionConfig(
        mach_number=0.8,
        angle_of_attack=5.0,
        reference_pressure=101325.0,
        reference_temperature=288.15,
        reference_density=1.225,
        reference_velocity=250.0
    )
    
    # Create boundary condition manager
    bc_manager = BoundaryConditionManager(config)
    
    # Test different boundary patches
    patches = [
        create_boundary_patch(1, "farfield", "farfield", list(range(0, 50))),
        create_boundary_patch(2, "wall", "wall", list(range(50, 100))),
        create_boundary_patch(3, "symmetry", "symmetry", list(range(100, 125))),
        create_boundary_patch(4, "inlet", "inlet", list(range(125, 140)),
                            total_pressure=120000.0, total_temperature=300.0),
        create_boundary_patch(5, "outlet", "outlet", list(range(140, 150)),
                            static_pressure=95000.0)
    ]
    
    # Add patches to manager
    for patch in patches:
        bc_manager.add_boundary_patch(patch)
    
    print(f"\n  Created {len(patches)} boundary patches:")
    stats = bc_manager.get_boundary_statistics()
    for bc_type, info in stats['patch_types'].items():
        print(f"    {bc_type}: {info['count']} patches, {info['faces']} faces")
    
    # Test boundary condition application
    print(f"\n  Testing boundary condition application:")
    
    # Mock solution and mesh data
    n_cells = 1000
    solution = np.random.rand(n_cells, 5) + 0.5
    
    mesh_info = {
        'face_owners': np.random.randint(0, n_cells, 200),
        'face_normals': np.random.rand(200, 3),
        'face_areas': np.random.rand(200) + 0.1
    }
    
    # Normalize face normals
    for i in range(200):
        norm = np.linalg.norm(mesh_info['face_normals'][i])
        if norm > 1e-12:
            mesh_info['face_normals'][i] /= norm
    
    # Apply boundary conditions
    ghost_states = bc_manager.apply_boundary_conditions(solution, mesh_info)
    
    print(f"    Computed ghost states for {len(ghost_states)} boundary faces")
    
    # Test individual boundary conditions
    print(f"\n  Testing individual boundary conditions:")
    
    # Test case: interior state
    interior_state = np.array([1.225, 245.0, 0.0, 21.6, 350000.0])  # [rho, rho*u, rho*v, rho*w, rho*E]
    face_normal = np.array([1.0, 0.0, 0.0])
    face_area = 1.0
    
    # Test each boundary condition type
    for patch in patches:
        bc = bc_manager.boundary_conditions[patch.patch_id]
        ghost_state = bc.apply_boundary_condition(interior_state, face_normal, face_area)
        
        # Convert to primitive for display
        primitive = bc._conservative_to_primitive(ghost_state)
        print(f"    {patch.boundary_type.value:10s}: rho={primitive[0]:.3f}, "
              f"u={primitive[1]:.1f}, p={primitive[4]:.0f}, T={primitive[5]:.1f}")


def test_openfoam_boundary_integration():
    """Test OpenFOAM boundary condition integration."""
    print("Testing OpenFOAM Boundary Condition Integration:")
    
    case_dir = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/examples/Cylinder"
    
    try:
        # Create boundary manager from OpenFOAM case
        bc_manager = create_boundary_manager_from_openfoam(
            case_dir,
            flow_config={
                'mach_number': 0.1,
                'reynolds_number': 100.0,
                'reference_velocity': 10.0,
                'reference_pressure': 0.0,  # Gauge pressure
                'reference_temperature': 288.15
            }
        )
        
        # Get statistics
        stats = bc_manager.get_boundary_statistics()
        print(f"  Created boundary manager with {stats['n_patches']} patches:")
        
        for bc_type, info in stats['patch_types'].items():
            print(f"    {bc_type}: {info['count']} patches, {info['faces']} faces")
        
        # List individual patches
        print(f"\n  Patch details:")
        for patch_id, patch in bc_manager.boundary_patches.items():
            print(f"    {patch.patch_name}: {patch.boundary_type.value}")
            
        return bc_manager
        
    except Exception as e:
        print(f"  Error: {e}")
        return None


if __name__ == "__main__":
    test_boundary_conditions()