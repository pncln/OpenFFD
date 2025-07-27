"""
Objective Functions for Adjoint-Based Optimization

Implements various objective functions and their derivatives for
CFD shape optimization and design studies.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveFunctionConfig:
    """Configuration for objective functions."""
    
    # Normalization
    reference_area: float = 1.0  # Reference area for force coefficients
    reference_length: float = 1.0  # Reference length
    reference_pressure: float = 101325.0  # Reference pressure
    reference_velocity: float = 100.0  # Reference velocity
    reference_density: float = 1.225  # Reference density
    
    # Integration settings
    integration_method: str = "trapezoidal"  # "trapezoidal", "simpson", "gauss"
    
    # Weighting
    use_cell_weights: bool = True  # Weight by cell volume/area
    
    # Output control
    save_distributions: bool = False  # Save local contributions


class ObjectiveFunction(ABC):
    """Abstract base class for objective functions."""
    
    def __init__(self, 
                 name: str,
                 config: Optional[ObjectiveFunctionConfig] = None):
        """
        Initialize objective function.
        
        Args:
            name: Name of objective function
            config: Configuration parameters
        """
        self.name = name
        self.config = config or ObjectiveFunctionConfig()
        self.value = 0.0
        self.local_contributions = np.array([])
        self.boundary_contributions = np.array([])
        
    @abstractmethod
    def evaluate(self, 
                solution: np.ndarray,
                mesh_info: Dict[str, Any],
                boundary_info: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate objective function.
        
        Args:
            solution: Flow solution [n_cells, n_variables]
            mesh_info: Mesh geometry and connectivity information
            boundary_info: Boundary condition information
            
        Returns:
            Objective function value
        """
        pass
    
    @abstractmethod
    def compute_source_term(self, 
                           solution: np.ndarray,
                           mesh_info: Dict[str, Any],
                           boundary_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Compute source term for adjoint equations (∂J/∂U).
        
        Args:
            solution: Flow solution
            mesh_info: Mesh information
            boundary_info: Boundary information
            
        Returns:
            Source term [n_cells, n_variables]
        """
        pass
    
    @abstractmethod
    def compute_boundary_source_term(self,
                                   solution: np.ndarray,
                                   mesh_info: Dict[str, Any],
                                   boundary_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Compute boundary source terms for adjoint equations.
        
        Returns:
            Dictionary of boundary source terms by boundary ID
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get objective function statistics."""
        return {
            'name': self.name,
            'value': self.value,
            'local_min': np.min(self.local_contributions) if self.local_contributions.size > 0 else 0.0,
            'local_max': np.max(self.local_contributions) if self.local_contributions.size > 0 else 0.0,
            'local_mean': np.mean(self.local_contributions) if self.local_contributions.size > 0 else 0.0
        }


class DragObjective(ObjectiveFunction):
    """
    Drag coefficient objective function.
    
    Computes drag coefficient based on pressure and viscous forces.
    """
    
    def __init__(self, 
                 flow_direction: np.ndarray = np.array([1.0, 0.0, 0.0]),
                 wall_boundary_ids: Optional[List[int]] = None,
                 config: Optional[ObjectiveFunctionConfig] = None):
        """
        Initialize drag objective.
        
        Args:
            flow_direction: Flow direction vector
            wall_boundary_ids: IDs of wall boundaries for integration
            config: Configuration
        """
        super().__init__("drag_coefficient", config)
        self.flow_direction = flow_direction / np.linalg.norm(flow_direction)
        self.wall_boundary_ids = wall_boundary_ids or []
        
    def evaluate(self, 
                solution: np.ndarray,
                mesh_info: Dict[str, Any],
                boundary_info: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate drag coefficient."""
        if boundary_info is None:
            logger.warning("No boundary information provided for drag computation")
            return 0.0
        
        total_drag = 0.0
        self.boundary_contributions = {}
        
        # Dynamic pressure
        q_inf = 0.5 * self.config.reference_density * self.config.reference_velocity**2
        
        for boundary_id in self.wall_boundary_ids:
            if boundary_id not in boundary_info:
                continue
            
            boundary_data = boundary_info[boundary_id]
            faces = boundary_data.get('faces', [])
            
            boundary_drag = 0.0
            local_contrib = []
            
            for face_id in faces:
                # Get face properties
                face_center = mesh_info['face_centers'][face_id]
                face_normal = mesh_info['face_normals'][face_id]
                face_area = mesh_info['face_areas'][face_id]
                
                # Get adjacent cell solution
                owner_cell = mesh_info['face_owners'][face_id]
                cell_solution = solution[owner_cell]
                
                # Compute pressure force
                pressure = self._extract_pressure(cell_solution)
                pressure_force = pressure * face_area * face_normal
                
                # Project to flow direction (drag component)
                drag_contribution = np.dot(pressure_force, self.flow_direction)
                boundary_drag += drag_contribution
                local_contrib.append(drag_contribution)
            
            self.boundary_contributions[boundary_id] = np.array(local_contrib)
            total_drag += boundary_drag
        
        # Normalize by dynamic pressure and reference area
        drag_coefficient = total_drag / (q_inf * self.config.reference_area)
        self.value = drag_coefficient
        
        return drag_coefficient
    
    def _extract_pressure(self, conservative_vars: np.ndarray, gamma: float = 1.4) -> float:
        """Extract pressure from conservative variables."""
        rho, rho_u, rho_v, rho_w, rho_E = conservative_vars
        
        # Avoid division by zero
        rho = max(rho, 1e-12)
        
        # Velocities
        u = rho_u / rho
        v = rho_v / rho
        w = rho_w / rho
        
        # Pressure
        kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
        pressure = (gamma - 1) * (rho_E - kinetic_energy)
        
        return max(pressure, 1e-6)
    
    def compute_source_term(self, 
                           solution: np.ndarray,
                           mesh_info: Dict[str, Any],
                           boundary_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Compute ∂J/∂U for drag objective."""
        n_cells, n_vars = solution.shape
        source_term = np.zeros((n_cells, n_vars))
        
        # For drag, source term is zero in the volume
        # All contributions come from boundary terms
        
        return source_term
    
    def compute_boundary_source_term(self,
                                   solution: np.ndarray,
                                   mesh_info: Dict[str, Any],
                                   boundary_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Compute boundary source terms for drag."""
        boundary_sources = {}
        
        # Dynamic pressure
        q_inf = 0.5 * self.config.reference_density * self.config.reference_velocity**2
        normalization = 1.0 / (q_inf * self.config.reference_area)
        
        for boundary_id in self.wall_boundary_ids:
            if boundary_id not in boundary_info:
                continue
            
            boundary_data = boundary_info[boundary_id]
            faces = boundary_data.get('faces', [])
            n_faces = len(faces)
            
            # Source term for each face [n_faces, n_variables]
            face_source = np.zeros((n_faces, 5))
            
            for i, face_id in enumerate(faces):
                face_normal = mesh_info['face_normals'][face_id]
                face_area = mesh_info['face_areas'][face_id]
                
                # ∂(pressure)/∂U for drag computation
                # Only pressure contribution (energy equation)
                dp_dU = self._compute_pressure_derivatives()
                
                # Project to flow direction
                drag_sensitivity = np.dot(face_normal, self.flow_direction) * face_area * normalization
                
                # Apply chain rule: ∂J/∂U = ∂J/∂p * ∂p/∂U
                face_source[i] = drag_sensitivity * dp_dU
            
            boundary_sources[boundary_id] = face_source
        
        return boundary_sources
    
    def _compute_pressure_derivatives(self, gamma: float = 1.4) -> np.ndarray:
        """Compute ∂p/∂U derivatives."""
        # For perfect gas: p = (γ-1)[ρE - 0.5*ρ*(u²+v²+w²)]
        # ∂p/∂ρ = (γ-1)[-0.5*(u²+v²+w²)]
        # ∂p/∂(ρu) = (γ-1)[-u]
        # ∂p/∂(ρv) = (γ-1)[-v]
        # ∂p/∂(ρw) = (γ-1)[-w]
        # ∂p/∂(ρE) = (γ-1)
        
        # This is a simplified version - full implementation would use actual velocities
        dp_dU = np.array([0.0, 0.0, 0.0, 0.0, gamma - 1.0])
        return dp_dU


class LiftObjective(ObjectiveFunction):
    """
    Lift coefficient objective function.
    
    Computes lift coefficient based on pressure and viscous forces.
    """
    
    def __init__(self, 
                 lift_direction: np.ndarray = np.array([0.0, 1.0, 0.0]),
                 wall_boundary_ids: Optional[List[int]] = None,
                 config: Optional[ObjectiveFunctionConfig] = None):
        """
        Initialize lift objective.
        
        Args:
            lift_direction: Lift direction vector (perpendicular to flow)
            wall_boundary_ids: IDs of wall boundaries
            config: Configuration
        """
        super().__init__("lift_coefficient", config)
        self.lift_direction = lift_direction / np.linalg.norm(lift_direction)
        self.wall_boundary_ids = wall_boundary_ids or []
        
    def evaluate(self, 
                solution: np.ndarray,
                mesh_info: Dict[str, Any],
                boundary_info: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate lift coefficient."""
        if boundary_info is None:
            return 0.0
        
        total_lift = 0.0
        q_inf = 0.5 * self.config.reference_density * self.config.reference_velocity**2
        
        for boundary_id in self.wall_boundary_ids:
            if boundary_id not in boundary_info:
                continue
            
            boundary_data = boundary_info[boundary_id]
            faces = boundary_data.get('faces', [])
            
            for face_id in faces:
                face_normal = mesh_info['face_normals'][face_id]
                face_area = mesh_info['face_areas'][face_id]
                
                owner_cell = mesh_info['face_owners'][face_id]
                cell_solution = solution[owner_cell]
                
                pressure = self._extract_pressure(cell_solution)
                pressure_force = pressure * face_area * face_normal
                
                # Project to lift direction
                lift_contribution = np.dot(pressure_force, self.lift_direction)
                total_lift += lift_contribution
        
        lift_coefficient = total_lift / (q_inf * self.config.reference_area)
        self.value = lift_coefficient
        return lift_coefficient
    
    def _extract_pressure(self, conservative_vars: np.ndarray, gamma: float = 1.4) -> float:
        """Extract pressure from conservative variables."""
        # Same as drag objective
        rho, rho_u, rho_v, rho_w, rho_E = conservative_vars
        rho = max(rho, 1e-12)
        
        u, v, w = rho_u/rho, rho_v/rho, rho_w/rho
        kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
        pressure = (gamma - 1) * (rho_E - kinetic_energy)
        
        return max(pressure, 1e-6)
    
    def compute_source_term(self, 
                           solution: np.ndarray,
                           mesh_info: Dict[str, Any],
                           boundary_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Compute volume source term for lift."""
        n_cells, n_vars = solution.shape
        return np.zeros((n_cells, n_vars))
    
    def compute_boundary_source_term(self,
                                   solution: np.ndarray,
                                   mesh_info: Dict[str, Any],
                                   boundary_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Compute boundary source terms for lift."""
        # Similar to drag but with lift direction
        boundary_sources = {}
        q_inf = 0.5 * self.config.reference_density * self.config.reference_velocity**2
        normalization = 1.0 / (q_inf * self.config.reference_area)
        
        for boundary_id in self.wall_boundary_ids:
            if boundary_id not in boundary_info:
                continue
            
            boundary_data = boundary_info[boundary_id]
            faces = boundary_data.get('faces', [])
            n_faces = len(faces)
            
            face_source = np.zeros((n_faces, 5))
            
            for i, face_id in enumerate(faces):
                face_normal = mesh_info['face_normals'][face_id]
                face_area = mesh_info['face_areas'][face_id]
                
                dp_dU = self._compute_pressure_derivatives()
                lift_sensitivity = np.dot(face_normal, self.lift_direction) * face_area * normalization
                
                face_source[i] = lift_sensitivity * dp_dU
            
            boundary_sources[boundary_id] = face_source
        
        return boundary_sources
    
    def _compute_pressure_derivatives(self, gamma: float = 1.4) -> np.ndarray:
        """Compute ∂p/∂U derivatives."""
        return np.array([0.0, 0.0, 0.0, 0.0, gamma - 1.0])


class PressureLossObjective(ObjectiveFunction):
    """
    Pressure loss objective function.
    
    Computes total pressure loss between inlet and outlet.
    """
    
    def __init__(self,
                 inlet_boundary_id: int,
                 outlet_boundary_id: int,
                 config: Optional[ObjectiveFunctionConfig] = None):
        """
        Initialize pressure loss objective.
        
        Args:
            inlet_boundary_id: Inlet boundary ID
            outlet_boundary_id: Outlet boundary ID
            config: Configuration
        """
        super().__init__("pressure_loss", config)
        self.inlet_boundary_id = inlet_boundary_id
        self.outlet_boundary_id = outlet_boundary_id
        
    def evaluate(self, 
                solution: np.ndarray,
                mesh_info: Dict[str, Any],
                boundary_info: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate pressure loss."""
        if boundary_info is None:
            return 0.0
        
        # Compute total pressure at inlet and outlet
        p_total_inlet = self._compute_total_pressure(
            solution, mesh_info, boundary_info, self.inlet_boundary_id
        )
        p_total_outlet = self._compute_total_pressure(
            solution, mesh_info, boundary_info, self.outlet_boundary_id
        )
        
        # Pressure loss coefficient
        pressure_loss = (p_total_inlet - p_total_outlet) / self.config.reference_pressure
        self.value = pressure_loss
        
        return pressure_loss
    
    def _compute_total_pressure(self, 
                               solution: np.ndarray,
                               mesh_info: Dict[str, Any],
                               boundary_info: Dict[str, Any],
                               boundary_id: int,
                               gamma: float = 1.4) -> float:
        """Compute area-averaged total pressure at boundary."""
        if boundary_id not in boundary_info:
            return 0.0
        
        boundary_data = boundary_info[boundary_id]
        faces = boundary_data.get('faces', [])
        
        total_pressure_weighted = 0.0
        total_area = 0.0
        
        for face_id in faces:
            face_area = mesh_info['face_areas'][face_id]
            owner_cell = mesh_info['face_owners'][face_id]
            cell_solution = solution[owner_cell]
            
            # Extract primitive variables
            rho, rho_u, rho_v, rho_w, rho_E = cell_solution
            rho = max(rho, 1e-12)
            
            u, v, w = rho_u/rho, rho_v/rho, rho_w/rho
            velocity_magnitude = np.sqrt(u**2 + v**2 + w**2)
            
            # Static pressure
            kinetic_energy = 0.5 * rho * velocity_magnitude**2
            p_static = (gamma - 1) * (rho_E - kinetic_energy)
            p_static = max(p_static, 1e-6)
            
            # Total pressure
            p_total = p_static * (1 + 0.5 * (gamma - 1) * 
                                (velocity_magnitude / np.sqrt(gamma * p_static / rho))**2)**(gamma / (gamma - 1))
            
            total_pressure_weighted += p_total * face_area
            total_area += face_area
        
        return total_pressure_weighted / max(total_area, 1e-12)
    
    def compute_source_term(self, 
                           solution: np.ndarray,
                           mesh_info: Dict[str, Any],
                           boundary_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Compute volume source term for pressure loss."""
        n_cells, n_vars = solution.shape
        return np.zeros((n_cells, n_vars))
    
    def compute_boundary_source_term(self,
                                   solution: np.ndarray,
                                   mesh_info: Dict[str, Any],
                                   boundary_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Compute boundary source terms for pressure loss."""
        # Implementation would involve derivatives of total pressure
        # This is complex and would require careful treatment
        return {}


class CompositeObjective(ObjectiveFunction):
    """
    Composite objective function combining multiple objectives.
    
    Allows for multi-objective optimization with weighted combination.
    """
    
    def __init__(self,
                 objectives: List[Tuple[ObjectiveFunction, float]],
                 config: Optional[ObjectiveFunctionConfig] = None):
        """
        Initialize composite objective.
        
        Args:
            objectives: List of (objective_function, weight) tuples
            config: Configuration
        """
        super().__init__("composite", config)
        self.objectives = objectives
        self.individual_values = {}
        
    def evaluate(self, 
                solution: np.ndarray,
                mesh_info: Dict[str, Any],
                boundary_info: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate composite objective."""
        total_value = 0.0
        
        for objective, weight in self.objectives:
            obj_value = objective.evaluate(solution, mesh_info, boundary_info)
            self.individual_values[objective.name] = obj_value
            total_value += weight * obj_value
        
        self.value = total_value
        return total_value
    
    def compute_source_term(self, 
                           solution: np.ndarray,
                           mesh_info: Dict[str, Any],
                           boundary_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Compute composite source term."""
        n_cells, n_vars = solution.shape
        total_source = np.zeros((n_cells, n_vars))
        
        for objective, weight in self.objectives:
            obj_source = objective.compute_source_term(solution, mesh_info, boundary_info)
            total_source += weight * obj_source
        
        return total_source
    
    def compute_boundary_source_term(self,
                                   solution: np.ndarray,
                                   mesh_info: Dict[str, Any],
                                   boundary_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Compute composite boundary source terms."""
        composite_boundary_sources = {}
        
        for objective, weight in self.objectives:
            obj_boundary_sources = objective.compute_boundary_source_term(
                solution, mesh_info, boundary_info
            )
            
            for boundary_id, source in obj_boundary_sources.items():
                if boundary_id not in composite_boundary_sources:
                    composite_boundary_sources[boundary_id] = np.zeros_like(source)
                composite_boundary_sources[boundary_id] += weight * source
        
        return composite_boundary_sources
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get composite objective statistics."""
        stats = super().get_statistics()
        stats['individual_values'] = self.individual_values.copy()
        stats['weights'] = {obj.name: weight for obj, weight in self.objectives}
        return stats


def create_drag_objective(flow_direction: np.ndarray = np.array([1.0, 0.0, 0.0]),
                         wall_boundary_ids: Optional[List[int]] = None,
                         config: Optional[ObjectiveFunctionConfig] = None) -> DragObjective:
    """Factory function for drag objective."""
    return DragObjective(flow_direction, wall_boundary_ids, config)


def create_lift_objective(lift_direction: np.ndarray = np.array([0.0, 1.0, 0.0]),
                         wall_boundary_ids: Optional[List[int]] = None,
                         config: Optional[ObjectiveFunctionConfig] = None) -> LiftObjective:
    """Factory function for lift objective."""
    return LiftObjective(lift_direction, wall_boundary_ids, config)


def test_objective_functions():
    """Test objective functions."""
    print("Testing Objective Functions:")
    
    # Create test data
    n_cells = 100
    n_vars = 5
    solution = np.random.rand(n_cells, n_vars)
    
    mesh_info = {
        'face_centers': np.random.rand(50, 3),
        'face_normals': np.random.rand(50, 3),
        'face_areas': np.random.rand(50),
        'face_owners': np.random.randint(0, n_cells, 50)
    }
    
    boundary_info = {
        1: {'faces': list(range(25))},  # Wall boundary
        2: {'faces': list(range(25, 50))}  # Another boundary
    }
    
    # Test drag objective
    config = ObjectiveFunctionConfig(reference_area=2.0)
    drag_obj = create_drag_objective(wall_boundary_ids=[1], config=config)
    drag_value = drag_obj.evaluate(solution, mesh_info, boundary_info)
    
    print(f"  Drag coefficient: {drag_value:.6f}")
    print(f"  Drag statistics: {drag_obj.get_statistics()}")
    
    # Test lift objective
    lift_obj = create_lift_objective(wall_boundary_ids=[1], config=config)
    lift_value = lift_obj.evaluate(solution, mesh_info, boundary_info)
    
    print(f"  Lift coefficient: {lift_value:.6f}")
    
    # Test composite objective
    composite_obj = CompositeObjective([
        (drag_obj, 1.0),  # Minimize drag
        (lift_obj, -0.1)  # Maximize lift (negative weight)
    ], config)
    
    composite_value = composite_obj.evaluate(solution, mesh_info, boundary_info)
    print(f"  Composite objective: {composite_value:.6f}")
    print(f"  Individual values: {composite_obj.individual_values}")


if __name__ == "__main__":
    test_objective_functions()