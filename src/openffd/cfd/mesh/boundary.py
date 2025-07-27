"""
Boundary Condition Management for 3D Supersonic Flow Solver

Provides comprehensive boundary condition handling including:
- Farfield boundary conditions for supersonic flows
- Wall boundary conditions with viscous/inviscid options
- Inlet/outlet conditions with pressure/velocity specification
- Symmetry and periodic boundary conditions
- Ghost cell management for boundary implementation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .unstructured_mesh import BoundaryCondition

logger = logging.getLogger(__name__)


@dataclass
class BoundaryState:
    """Boundary state specification for supersonic flows."""
    # Conservative variables [rho, rho*u, rho*v, rho*w, rho*E]
    conservatives: np.ndarray = field(default_factory=lambda: np.zeros(5))
    
    # Primitive variables [rho, u, v, w, p, T]
    primitives: np.ndarray = field(default_factory=lambda: np.zeros(6))
    
    # Additional properties
    mach_number: float = 0.0
    total_pressure: float = 0.0
    total_temperature: float = 0.0
    flow_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    
    # Wall properties
    wall_temperature: Optional[float] = None
    heat_flux: Optional[float] = None
    roughness: float = 0.0
    
    # Time-dependent properties
    time_function: Optional[str] = None  # Function of time for unsteady conditions


class BCType(Enum):
    """Extended boundary condition types for supersonic CFD."""
    # Supersonic flow boundaries
    SUPERSONIC_INLET = "supersonic_inlet"
    SUPERSONIC_OUTLET = "supersonic_outlet"
    SUBSONIC_INLET = "subsonic_inlet"
    SUBSONIC_OUTLET = "subsonic_outlet"
    
    # Wall boundaries
    VISCOUS_WALL = "viscous_wall"
    INVISCID_WALL = "inviscid_wall"
    ADIABATIC_WALL = "adiabatic_wall"
    ISOTHERMAL_WALL = "isothermal_wall"
    
    # Symmetry and farfield
    SYMMETRY_PLANE = "symmetry_plane"
    FARFIELD = "farfield"
    
    # Pressure boundaries
    PRESSURE_INLET = "pressure_inlet"
    PRESSURE_OUTLET = "pressure_outlet"
    
    # Special boundaries
    PERIODIC = "periodic"
    RIEMANN_INVARIANT = "riemann_invariant"


@dataclass
class BoundaryPatch:
    """Complete boundary patch definition."""
    name: str
    bc_type: BCType
    face_ids: np.ndarray  # Face indices belonging to this patch
    state: BoundaryState
    
    # Geometric properties
    area: float = 0.0
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Implementation details
    ghost_cells: List[int] = field(default_factory=list)
    stencil_cells: List[int] = field(default_factory=list)
    
    # Monitoring and output
    monitor_forces: bool = False
    monitor_flow_rate: bool = False
    
    def update_geometry(self, mesh) -> None:
        """Update geometric properties from mesh data."""
        if len(self.face_ids) == 0:
            return
        
        total_area = 0.0
        weighted_centroid = np.zeros(3)
        weighted_normal = np.zeros(3)
        
        for face_id in self.face_ids:
            face_area = mesh.face_data.areas[face_id]
            face_center = mesh.face_data.centers[face_id]
            face_normal = mesh.face_data.normals[face_id]
            
            total_area += face_area
            weighted_centroid += face_area * face_center
            weighted_normal += face_area * face_normal
        
        self.area = total_area
        if total_area > 1e-12:
            self.centroid = weighted_centroid / total_area
            self.normal = weighted_normal / np.linalg.norm(weighted_normal)


class BoundaryManager:
    """
    Comprehensive boundary condition manager for supersonic CFD.
    
    Features:
    - Multiple boundary condition types
    - Ghost cell generation and management
    - Boundary state computation
    - Force and flow rate monitoring
    - Time-dependent boundary conditions
    """
    
    def __init__(self, mesh=None):
        """Initialize boundary manager."""
        self.mesh = mesh
        self.patches: Dict[str, BoundaryPatch] = {}
        
        # Physical constants
        self.gamma = 1.4  # Specific heat ratio for air
        self.R = 287.0    # Gas constant for air [J/kg/K]
        
        # Reference conditions
        self.p_ref = 101325.0    # Reference pressure [Pa]
        self.T_ref = 288.15      # Reference temperature [K]
        self.rho_ref = 1.225     # Reference density [kg/m³]
        self.a_ref = 343.0       # Reference speed of sound [m/s]
        
        # Ghost cell data
        self._ghost_cells: Dict[int, BoundaryState] = {}
        self._boundary_gradients: Dict[str, np.ndarray] = {}
        
        self._initialized = False
    
    def add_patch(self, 
                  name: str,
                  bc_type: BCType, 
                  face_ids: np.ndarray,
                  **kwargs) -> None:
        """Add a boundary patch with specified conditions."""
        # Create boundary state from kwargs
        state = BoundaryState()
        
        # Set state based on boundary type and provided parameters
        if bc_type in [BCType.SUPERSONIC_INLET, BCType.SUBSONIC_INLET]:
            self._setup_inlet_state(state, **kwargs)
        elif bc_type in [BCType.SUPERSONIC_OUTLET, BCType.SUBSONIC_OUTLET]:
            self._setup_outlet_state(state, **kwargs)
        elif bc_type == BCType.FARFIELD:
            self._setup_farfield_state(state, **kwargs)
        elif bc_type in [BCType.VISCOUS_WALL, BCType.INVISCID_WALL]:
            self._setup_wall_state(state, bc_type, **kwargs)
        
        # Create patch
        patch = BoundaryPatch(
            name=name,
            bc_type=bc_type,
            face_ids=face_ids.copy(),
            state=state,
            monitor_forces=kwargs.get('monitor_forces', False),
            monitor_flow_rate=kwargs.get('monitor_flow_rate', False)
        )
        
        # Update geometry if mesh is available
        if self.mesh is not None:
            patch.update_geometry(self.mesh)
        
        self.patches[name] = patch
        logger.info(f"Added boundary patch '{name}' with {len(face_ids)} faces")
    
    def _setup_inlet_state(self, state: BoundaryState, **kwargs) -> None:
        """Setup inlet boundary state."""
        # Get flow properties
        mach = kwargs.get('mach_number', 0.3)
        pressure = kwargs.get('total_pressure', self.p_ref)
        temperature = kwargs.get('total_temperature', self.T_ref)
        direction = np.array(kwargs.get('flow_direction', [1.0, 0.0, 0.0]))
        
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        
        # Compute static conditions from total conditions
        T_static = temperature / (1.0 + 0.5 * (self.gamma - 1) * mach**2)
        p_static = pressure / (1.0 + 0.5 * (self.gamma - 1) * mach**2)**(self.gamma / (self.gamma - 1))
        
        # Compute primitive variables
        rho = p_static / (self.R * T_static)
        a = np.sqrt(self.gamma * self.R * T_static)
        velocity_magnitude = mach * a
        velocity = velocity_magnitude * direction
        
        state.primitives = np.array([rho, velocity[0], velocity[1], velocity[2], p_static, T_static])
        state.mach_number = mach
        state.total_pressure = pressure
        state.total_temperature = temperature
        state.flow_direction = direction
        
        # Convert to conservative variables
        state.conservatives = self._primitives_to_conservatives(state.primitives)
    
    def _setup_outlet_state(self, state: BoundaryState, **kwargs) -> None:
        """Setup outlet boundary state."""
        pressure = kwargs.get('pressure', self.p_ref)
        
        # For outlet, typically only pressure is specified
        # Other quantities are extrapolated from interior
        state.primitives[4] = pressure  # Set pressure
        
        # Store outlet pressure for boundary implementation
        state.total_pressure = pressure
    
    def _setup_farfield_state(self, state: BoundaryState, **kwargs) -> None:
        """Setup farfield boundary state."""
        mach = kwargs.get('mach_number', 2.0)  # Default supersonic
        pressure = kwargs.get('pressure', self.p_ref)
        temperature = kwargs.get('temperature', self.T_ref)
        angle_of_attack = kwargs.get('angle_of_attack', 0.0)  # Degrees
        
        # Convert angle to radians
        alpha = np.radians(angle_of_attack)
        
        # Flow direction accounting for angle of attack
        direction = np.array([np.cos(alpha), np.sin(alpha), 0.0])
        
        # Compute primitive variables
        rho = pressure / (self.R * temperature)
        a = np.sqrt(self.gamma * self.R * temperature)
        velocity_magnitude = mach * a
        velocity = velocity_magnitude * direction
        
        state.primitives = np.array([rho, velocity[0], velocity[1], velocity[2], pressure, temperature])
        state.mach_number = mach
        state.flow_direction = direction
        
        # Convert to conservative variables
        state.conservatives = self._primitives_to_conservatives(state.primitives)
    
    def _setup_wall_state(self, state: BoundaryState, bc_type: BCType, **kwargs) -> None:
        """Setup wall boundary state."""
        if bc_type == BCType.VISCOUS_WALL:
            # No-slip condition: zero velocity
            state.primitives[1:4] = 0.0
        elif bc_type == BCType.INVISCID_WALL:
            # Slip condition: zero normal velocity component
            pass  # Implemented during boundary application
        
        # Wall temperature
        if 'wall_temperature' in kwargs:
            state.wall_temperature = kwargs['wall_temperature']
            state.primitives[5] = kwargs['wall_temperature']
        
        # Heat flux for non-adiabatic walls
        if 'heat_flux' in kwargs:
            state.heat_flux = kwargs['heat_flux']
        
        # Surface roughness
        state.roughness = kwargs.get('roughness', 0.0)
    
    def _primitives_to_conservatives(self, primitives: np.ndarray) -> np.ndarray:
        """Convert primitive to conservative variables."""
        rho, u, v, w, p, T = primitives
        
        # Conservative variables
        rho_u = rho * u
        rho_v = rho * v
        rho_w = rho * w
        
        # Total energy
        kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
        internal_energy = p / (self.gamma - 1)
        rho_E = internal_energy + kinetic_energy
        
        return np.array([rho, rho_u, rho_v, rho_w, rho_E])
    
    def _conservatives_to_primitives(self, conservatives: np.ndarray) -> np.ndarray:
        """Convert conservative to primitive variables."""
        rho, rho_u, rho_v, rho_w, rho_E = conservatives
        
        # Avoid division by zero
        if rho < 1e-12:
            rho = 1e-12
        
        # Velocities
        u = rho_u / rho
        v = rho_v / rho
        w = rho_w / rho
        
        # Pressure
        kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
        p = (self.gamma - 1) * (rho_E - kinetic_energy)
        
        # Temperature
        T = p / (rho * self.R)
        
        return np.array([rho, u, v, w, p, T])
    
    def apply_boundary_conditions(self, 
                                 cell_data: Any,
                                 face_data: Any,
                                 time: float = 0.0) -> None:
        """Apply boundary conditions to ghost cells."""
        for patch_name, patch in self.patches.items():
            self._apply_patch_bc(patch, cell_data, face_data, time)
    
    def _apply_patch_bc(self, 
                       patch: BoundaryPatch, 
                       cell_data: Any,
                       face_data: Any,
                       time: float) -> None:
        """Apply boundary condition for a specific patch."""
        if patch.bc_type == BCType.FARFIELD:
            self._apply_farfield_bc(patch, cell_data, face_data)
        elif patch.bc_type in [BCType.SUPERSONIC_INLET, BCType.SUBSONIC_INLET]:
            self._apply_inlet_bc(patch, cell_data, face_data)
        elif patch.bc_type in [BCType.SUPERSONIC_OUTLET, BCType.SUBSONIC_OUTLET]:
            self._apply_outlet_bc(patch, cell_data, face_data)
        elif patch.bc_type in [BCType.VISCOUS_WALL, BCType.INVISCID_WALL]:
            self._apply_wall_bc(patch, cell_data, face_data)
        elif patch.bc_type == BCType.SYMMETRY_PLANE:
            self._apply_symmetry_bc(patch, cell_data, face_data)
    
    def _apply_farfield_bc(self, patch: BoundaryPatch, cell_data: Any, face_data: Any) -> None:
        """Apply farfield boundary conditions using Riemann invariants."""
        for face_id in patch.face_ids:
            owner_cell = face_data.owner[face_id]
            
            if owner_cell >= 0:
                # Get interior state
                interior_cons = cell_data.conservatives[owner_cell]
                interior_prim = self._conservatives_to_primitives(interior_cons)
                
                # Get face normal
                face_normal = face_data.normals[face_id]
                
                # Compute Riemann invariants
                ghost_state = self._compute_riemann_farfield(
                    interior_prim, patch.state.primitives, face_normal
                )
                
                # Store ghost cell state
                self._ghost_cells[face_id] = BoundaryState()
                self._ghost_cells[face_id].conservatives = self._primitives_to_conservatives(ghost_state)
                self._ghost_cells[face_id].primitives = ghost_state
    
    def _compute_riemann_farfield(self, 
                                 interior: np.ndarray,
                                 farfield: np.ndarray,
                                 normal: np.ndarray) -> np.ndarray:
        """Compute farfield boundary state using Riemann invariants."""
        # Interior state
        rho_i, u_i, v_i, w_i, p_i, T_i = interior
        velocity_i = np.array([u_i, v_i, w_i])
        
        # Farfield state
        rho_f, u_f, v_f, w_f, p_f, T_f = farfield
        velocity_f = np.array([u_f, v_f, w_f])
        
        # Normal velocity components
        vn_i = np.dot(velocity_i, normal)
        vn_f = np.dot(velocity_f, normal)
        
        # Speed of sound
        a_i = np.sqrt(self.gamma * p_i / rho_i)
        a_f = np.sqrt(self.gamma * p_f / rho_f)
        
        # Riemann invariants
        R_plus = vn_i + 2 * a_i / (self.gamma - 1)   # Outgoing
        R_minus = vn_f - 2 * a_f / (self.gamma - 1)  # Incoming
        
        # Determine flow direction
        if vn_i >= 0:  # Outflow
            if vn_i >= a_i:  # Supersonic outflow
                return interior  # Use interior state
            else:  # Subsonic outflow
                # Mix interior and farfield
                vn_ghost = 0.5 * (R_plus + R_minus)
                a_ghost = 0.25 * (self.gamma - 1) * (R_plus - R_minus)
                
                # Use farfield density and tangential velocity
                rho_ghost = rho_f
                velocity_ghost = velocity_f + (vn_ghost - vn_f) * normal
                p_ghost = rho_ghost * a_ghost**2 / self.gamma
                T_ghost = p_ghost / (rho_ghost * self.R)
                
                return np.array([rho_ghost, velocity_ghost[0], velocity_ghost[1], 
                               velocity_ghost[2], p_ghost, T_ghost])
        else:  # Inflow
            if abs(vn_i) >= a_i:  # Supersonic inflow
                return farfield  # Use farfield state
            else:  # Subsonic inflow
                # Mix states using Riemann invariants
                vn_ghost = 0.5 * (R_plus + R_minus)
                a_ghost = 0.25 * (self.gamma - 1) * (R_plus - R_minus)
                
                # Use farfield density and tangential velocity
                rho_ghost = rho_f
                velocity_ghost = velocity_f + (vn_ghost - vn_f) * normal
                p_ghost = rho_ghost * a_ghost**2 / self.gamma
                T_ghost = p_ghost / (rho_ghost * self.R)
                
                return np.array([rho_ghost, velocity_ghost[0], velocity_ghost[1], 
                               velocity_ghost[2], p_ghost, T_ghost])
    
    def _apply_wall_bc(self, patch: BoundaryPatch, cell_data: Any, face_data: Any) -> None:
        """Apply wall boundary conditions."""
        for face_id in patch.face_ids:
            owner_cell = face_data.owner[face_id]
            
            if owner_cell >= 0:
                # Get interior state
                interior_cons = cell_data.conservatives[owner_cell]
                interior_prim = self._conservatives_to_primitives(interior_cons)
                
                # Get face normal
                face_normal = face_data.normals[face_id]
                
                # Mirror velocity for wall
                ghost_prim = interior_prim.copy()
                velocity = interior_prim[1:4]
                
                if patch.bc_type == BCType.VISCOUS_WALL:
                    # No-slip: zero velocity
                    ghost_prim[1:4] = -velocity  # Mirror all components
                elif patch.bc_type == BCType.INVISCID_WALL:
                    # Slip: zero normal velocity
                    vn = np.dot(velocity, face_normal)
                    ghost_prim[1:4] = velocity - 2 * vn * face_normal
                
                # Wall temperature
                if patch.state.wall_temperature is not None:
                    ghost_prim[5] = patch.state.wall_temperature
                    # Update pressure to maintain density
                    ghost_prim[4] = ghost_prim[0] * self.R * ghost_prim[5]
                
                # Store ghost state
                self._ghost_cells[face_id] = BoundaryState()
                self._ghost_cells[face_id].conservatives = self._primitives_to_conservatives(ghost_prim)
                self._ghost_cells[face_id].primitives = ghost_prim
    
    def _apply_symmetry_bc(self, patch: BoundaryPatch, cell_data: Any, face_data: Any) -> None:
        """Apply symmetry boundary conditions."""
        for face_id in patch.face_ids:
            owner_cell = face_data.owner[face_id]
            
            if owner_cell >= 0:
                # Get interior state
                interior_cons = cell_data.conservatives[owner_cell]
                interior_prim = self._conservatives_to_primitives(interior_cons)
                
                # Get face normal
                face_normal = face_data.normals[face_id]
                
                # Mirror normal velocity component
                ghost_prim = interior_prim.copy()
                velocity = interior_prim[1:4]
                vn = np.dot(velocity, face_normal)
                ghost_prim[1:4] = velocity - 2 * vn * face_normal
                
                # Store ghost state
                self._ghost_cells[face_id] = BoundaryState()
                self._ghost_cells[face_id].conservatives = self._primitives_to_conservatives(ghost_prim)
                self._ghost_cells[face_id].primitives = ghost_prim
    
    def _apply_inlet_bc(self, patch: BoundaryPatch, cell_data: Any, face_data: Any) -> None:
        """Apply inlet boundary conditions."""
        for face_id in patch.face_ids:
            # Use specified inlet state
            self._ghost_cells[face_id] = patch.state
    
    def _apply_outlet_bc(self, patch: BoundaryPatch, cell_data: Any, face_data: Any) -> None:
        """Apply outlet boundary conditions."""
        for face_id in patch.face_ids:
            owner_cell = face_data.owner[face_id]
            
            if owner_cell >= 0:
                # Extrapolate from interior, set pressure
                interior_cons = cell_data.conservatives[owner_cell]
                ghost_prim = self._conservatives_to_primitives(interior_cons)
                
                # Set outlet pressure
                ghost_prim[4] = patch.state.total_pressure
                
                # Store ghost state
                self._ghost_cells[face_id] = BoundaryState()
                self._ghost_cells[face_id].conservatives = self._primitives_to_conservatives(ghost_prim)
                self._ghost_cells[face_id].primitives = ghost_prim
    
    def get_ghost_state(self, face_id: int) -> Optional[BoundaryState]:
        """Get ghost cell state for a boundary face."""
        return self._ghost_cells.get(face_id)
    
    def compute_wall_forces(self, patch_name: str) -> Dict[str, np.ndarray]:
        """Compute forces on a wall patch."""
        if patch_name not in self.patches:
            return {}
        
        patch = self.patches[patch_name]
        if patch.bc_type not in [BCType.VISCOUS_WALL, BCType.INVISCID_WALL]:
            return {}
        
        pressure_force = np.zeros(3)
        viscous_force = np.zeros(3)
        
        for face_id in patch.face_ids:
            face_area = self.mesh.face_data.areas[face_id]
            face_normal = self.mesh.face_data.normals[face_id]
            
            # Get pressure from owner cell
            owner_cell = self.mesh.face_data.owner[face_id]
            if owner_cell >= 0:
                owner_prim = self._conservatives_to_primitives(
                    self.mesh.cell_data.conservatives[owner_cell]
                )
                pressure = owner_prim[4]
                
                # Pressure force (normal to surface)
                pressure_force += pressure * face_area * face_normal
        
        return {
            'pressure_force': pressure_force,
            'viscous_force': viscous_force,
            'total_force': pressure_force + viscous_force
        }
    
    def get_patch_info(self, patch_name: str) -> Dict[str, Any]:
        """Get information about a boundary patch."""
        if patch_name not in self.patches:
            return {}
        
        patch = self.patches[patch_name]
        return {
            'name': patch.name,
            'type': patch.bc_type.value,
            'n_faces': len(patch.face_ids),
            'area': patch.area,
            'centroid': patch.centroid,
            'normal': patch.normal,
            'mach_number': patch.state.mach_number,
            'monitor_forces': patch.monitor_forces,
            'monitor_flow_rate': patch.monitor_flow_rate
        }
    
    def list_patches(self) -> List[str]:
        """Get list of all boundary patch names."""
        return list(self.patches.keys())
    
    def validate_boundary_setup(self) -> bool:
        """Validate boundary condition setup."""
        logger.info("Validating boundary conditions...")
        
        if not self.patches:
            logger.error("No boundary patches defined")
            return False
        
        # Check face coverage
        all_boundary_faces = set()
        for patch in self.patches.values():
            all_boundary_faces.update(patch.face_ids)
        
        if self.mesh:
            mesh_boundary_faces = set(np.where(self.mesh.face_data.neighbor == -1)[0])
            
            if all_boundary_faces != mesh_boundary_faces:
                missing = mesh_boundary_faces - all_boundary_faces
                extra = all_boundary_faces - mesh_boundary_faces
                if missing:
                    logger.error(f"Missing boundary faces: {missing}")
                if extra:
                    logger.error(f"Extra boundary faces: {extra}")
                return False
        
        logger.info("Boundary condition validation passed")
        return True