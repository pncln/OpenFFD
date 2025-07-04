"""Objective function registry for optimization."""

from typing import Dict, Type, Any
import numpy as np

from ..core.base import BaseObjective, BaseCase
from ..core.registry import register_objective


class ObjectiveRegistry:
    """Registry for optimization objectives."""
    
    _objectives: Dict[str, Type[BaseObjective]] = {}
    
    @classmethod
    def register_objective(cls, name: str, objective_class: Type[BaseObjective]) -> None:
        """Register an objective function."""
        cls._objectives[name] = objective_class
    
    @classmethod
    def get_objective(cls, name: str) -> Type[BaseObjective]:
        """Get objective class by name."""
        if name not in cls._objectives:
            raise ValueError(f"Unknown objective: {name}. "
                           f"Available objectives: {list(cls._objectives.keys())}")
        return cls._objectives[name]
    
    @classmethod
    def list_objectives(cls) -> list:
        """List all available objectives."""
        return list(cls._objectives.keys())


@register_objective('drag_coefficient')
class DragCoefficientObjective(BaseObjective):
    """Drag coefficient objective."""
    
    def evaluate(self, results: Dict[str, Any]) -> float:
        """Evaluate drag coefficient from forces."""
        if 'forces' not in results:
            return 0.0
        
        forces = results['forces']
        patches = self.config.patches if self.config.patches else ['airfoil']
        
        # Get flow direction
        flow_direction = np.array(self.config.direction if self.config.direction else [1.0, 0.0, 0.0])
        
        total_drag = 0.0
        for patch in patches:
            if patch in forces:
                force_vector = np.array(forces[patch])
                drag_force = np.dot(force_vector, flow_direction)
                total_drag += drag_force
        
        # Convert to coefficient (this would need reference values)
        return total_drag
    
    def get_gradient(self, results: Dict[str, Any], design_vars: np.ndarray) -> np.ndarray:
        """Get gradient of drag coefficient."""
        # This would be computed using adjoint or finite differences
        return np.zeros_like(design_vars)


@register_objective('lift_coefficient')
class LiftCoefficientObjective(BaseObjective):
    """Lift coefficient objective."""
    
    def evaluate(self, results: Dict[str, Any]) -> float:
        """Evaluate lift coefficient from forces."""
        if 'forces' not in results:
            return 0.0
        
        forces = results['forces']
        patches = self.config.patches if self.config.patches else ['airfoil']
        
        # Get flow direction and compute lift direction
        flow_direction = np.array(self.config.direction if self.config.direction else [1.0, 0.0, 0.0])
        lift_direction = np.array([-flow_direction[1], flow_direction[0], 0.0])
        lift_direction = lift_direction / np.linalg.norm(lift_direction)
        
        total_lift = 0.0
        for patch in patches:
            if patch in forces:
                force_vector = np.array(forces[patch])
                lift_force = np.dot(force_vector, lift_direction)
                total_lift += lift_force
        
        return total_lift
    
    def get_gradient(self, results: Dict[str, Any], design_vars: np.ndarray) -> np.ndarray:
        """Get gradient of lift coefficient."""
        return np.zeros_like(design_vars)


@register_objective('moment_coefficient')
class MomentCoefficientObjective(BaseObjective):
    """Moment coefficient objective."""
    
    def evaluate(self, results: Dict[str, Any]) -> float:
        """Evaluate moment coefficient from moments."""
        if 'moments' not in results:
            return 0.0
        
        moments = results['moments']
        patches = self.config.patches if self.config.patches else ['airfoil']
        
        total_moment = 0.0
        for patch in patches:
            if patch in moments:
                moment_vector = np.array(moments[patch])
                moment_z = moment_vector[2]  # Z-component for 2D airfoil
                total_moment += moment_z
        
        return total_moment
    
    def get_gradient(self, results: Dict[str, Any], design_vars: np.ndarray) -> np.ndarray:
        """Get gradient of moment coefficient."""
        return np.zeros_like(design_vars)


@register_objective('heat_transfer_coefficient')
class HeatTransferCoefficientObjective(BaseObjective):
    """Heat transfer coefficient objective."""
    
    def evaluate(self, results: Dict[str, Any]) -> float:
        """Evaluate heat transfer coefficient."""
        if 'heat_flux' not in results or 'wall_temperature' not in results:
            return 0.0
        
        patches = self.config.patches if self.config.patches else ['hot_wall']
        
        total_htc = 0.0
        patch_count = 0
        
        for patch in patches:
            if patch in results['heat_flux'] and patch in results['wall_temperature']:
                q_wall = results['heat_flux'][patch]
                T_wall = results['wall_temperature'][patch]
                T_ref = 300.0  # Reference temperature
                
                if abs(T_wall - T_ref) > 1e-6:
                    h = q_wall / (T_wall - T_ref)
                    total_htc += h
                    patch_count += 1
        
        return total_htc / patch_count if patch_count > 0 else 0.0
    
    def get_gradient(self, results: Dict[str, Any], design_vars: np.ndarray) -> np.ndarray:
        """Get gradient of heat transfer coefficient."""
        return np.zeros_like(design_vars)


@register_objective('nusselt_number')
class NusseltNumberObjective(BaseObjective):
    """Nusselt number objective."""
    
    def evaluate(self, results: Dict[str, Any]) -> float:
        """Evaluate Nusselt number."""
        # This would require heat transfer coefficient and thermal conductivity
        htc_obj = HeatTransferCoefficientObjective(self.config, self.case_handler)
        h = htc_obj.evaluate(results)
        
        # Characteristic length and thermal conductivity would come from case
        L_char = 1.0  # Default
        k = 0.025  # Default thermal conductivity
        
        return h * L_char / k
    
    def get_gradient(self, results: Dict[str, Any], design_vars: np.ndarray) -> np.ndarray:
        """Get gradient of Nusselt number."""
        return np.zeros_like(design_vars)


@register_objective('pressure_drop')
class PressureDropObjective(BaseObjective):
    """Pressure drop objective."""
    
    def evaluate(self, results: Dict[str, Any]) -> float:
        """Evaluate pressure drop."""
        if 'pressure' not in results:
            return 0.0
        
        pressure_data = results['pressure']
        
        # Try to find inlet and outlet pressures
        inlet_pressure = pressure_data.get('inlet', 0.0)
        outlet_pressure = pressure_data.get('outlet', 0.0)
        
        return inlet_pressure - outlet_pressure
    
    def get_gradient(self, results: Dict[str, Any], design_vars: np.ndarray) -> np.ndarray:
        """Get gradient of pressure drop."""
        return np.zeros_like(design_vars)


# Auto-register all objectives
ObjectiveRegistry.register_objective('drag_coefficient', DragCoefficientObjective)
ObjectiveRegistry.register_objective('cd', DragCoefficientObjective)
ObjectiveRegistry.register_objective('lift_coefficient', LiftCoefficientObjective)
ObjectiveRegistry.register_objective('cl', LiftCoefficientObjective)
ObjectiveRegistry.register_objective('moment_coefficient', MomentCoefficientObjective)
ObjectiveRegistry.register_objective('cm', MomentCoefficientObjective)
ObjectiveRegistry.register_objective('heat_transfer_coefficient', HeatTransferCoefficientObjective)
ObjectiveRegistry.register_objective('h', HeatTransferCoefficientObjective)
ObjectiveRegistry.register_objective('htc', HeatTransferCoefficientObjective)
ObjectiveRegistry.register_objective('nusselt_number', NusseltNumberObjective)
ObjectiveRegistry.register_objective('nu', NusseltNumberObjective)
ObjectiveRegistry.register_objective('pressure_drop', PressureDropObjective)
ObjectiveRegistry.register_objective('dp', PressureDropObjective)