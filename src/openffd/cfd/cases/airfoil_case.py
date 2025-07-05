"""Airfoil-specific case handler."""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from .base_case import GenericCase
from ..core.config import CaseConfig
from ..core.registry import register_case_handler


@register_case_handler('airfoil')
class AirfoilCase(GenericCase):
    """Specialized case handler for airfoil optimization."""
    
    def __init__(self, case_path: Path, config: CaseConfig):
        super().__init__(case_path, config)
        self.airfoil_patch = None
        self.flow_direction = None
        self.reference_values = {}
    
    def detect_case_type(self) -> str:
        """Detect if this is an airfoil case."""
        # Call parent validation first
        super().detect_case_type()
        
        # Look for airfoil-specific indicators
        airfoil_indicators = [
            'airfoil' in str(self.case_path).lower(),
            'naca' in str(self.case_path).lower(),
            'wing' in str(self.case_path).lower(),
            'aerofoil' in str(self.case_path).lower()
        ]
        
        if any(airfoil_indicators):
            return 'airfoil'
        
        # Check boundary patches for airfoil-like names
        self._parse_boundary_patches()
        airfoil_patch_names = ['airfoil', 'wing', 'aerofoil', 'blade', 'profile', 'walls']
        
        for patch in self.boundary_patches:
            if any(name in patch['name'].lower() for name in airfoil_patch_names):
                return 'airfoil'
        
        return 'generic'
    
    def validate_case(self) -> bool:
        """Validate airfoil case setup."""
        if not super().validate_case():
            return False
        
        # Find airfoil patch
        self._identify_airfoil_patch()
        
        # Parse flow conditions
        self._parse_flow_conditions()
        
        # Validate airfoil-specific setup
        if not self.airfoil_patch:
            print("Warning: Could not identify airfoil patch")
            return False
        
        return True
    
    def _identify_airfoil_patch(self) -> None:
        """Identify which patch represents the airfoil."""
        airfoil_names = ['airfoil', 'wing', 'aerofoil', 'blade', 'profile', 'wall', 'walls']
        
        for patch in self.boundary_patches:
            patch_name = patch['name'].lower()
            
            # Check for direct matches
            for name in airfoil_names:
                if name in patch_name:
                    self.airfoil_patch = patch['name']
                    return
        
        # If no direct match, look for wall-type patches
        wall_patches = [p for p in self.boundary_patches if p['type'] == 'wall']
        if len(wall_patches) == 1:
            self.airfoil_patch = wall_patches[0]['name']
        elif len(wall_patches) > 1:
            # Choose the first wall patch as default
            self.airfoil_patch = wall_patches[0]['name']
            print(f"Warning: Multiple wall patches found, using '{self.airfoil_patch}'")
    
    def _parse_flow_conditions(self) -> None:
        """Parse flow conditions from case files."""
        # Try to get flow direction from objectives
        for obj in self.config.objectives:
            if obj.direction:
                self.flow_direction = np.array(obj.direction)
                break
        
        # Default flow direction (positive x)
        if self.flow_direction is None:
            self.flow_direction = np.array([1.0, 0.0, 0.0])
        
        # Get reference values from config
        self.reference_values = {
            'magUInf': self.config.constants.get('magUInf', 30.0),
            'rho': self.config.constants.get('rho', 1.225),
            'mu': self.config.constants.get('mu', 1.8e-5),
            'chord': self.config.constants.get('chord', 1.0)
        }
    
    def setup_optimization_domain(self) -> Dict[str, Any]:
        """Set up optimization domain for airfoil case."""
        # Get base domain setup
        domain_info = super().setup_optimization_domain()
        
        # Airfoil-specific domain adjustments
        if self.config.ffd_config.domain == "auto":
            domain = self._setup_airfoil_domain(domain_info['mesh_bounds'])
        else:
            domain = self.config.ffd_config.domain
        
        # Add airfoil-specific parameters
        domain_info.update({
            'airfoil_patch': self.airfoil_patch,
            'flow_direction': self.flow_direction.tolist(),
            'reference_values': self.reference_values,
            'domain': domain
        })
        
        return domain_info
    
    def _setup_airfoil_domain(self, mesh_bounds: Dict[str, float]) -> Dict[str, float]:
        """Set up FFD domain optimized for airfoil cases."""
        # For airfoil cases, we want tighter control around the airfoil
        chord_estimate = max(
            mesh_bounds['x_max'] - mesh_bounds['x_min'],
            self.reference_values.get('chord', 1.0)
        )
        
        # Domain should encompass airfoil with some margin
        x_center = (mesh_bounds['x_max'] + mesh_bounds['x_min']) / 2
        y_center = (mesh_bounds['y_max'] + mesh_bounds['y_min']) / 2
        
        return {
            'x_min': x_center - 0.6 * chord_estimate,
            'x_max': x_center + 0.6 * chord_estimate,
            'y_min': y_center - 0.5 * chord_estimate,
            'y_max': y_center + 0.5 * chord_estimate,
            'z_min': mesh_bounds['z_min'] - 0.01,
            'z_max': mesh_bounds['z_max'] + 0.01
        }
    
    def extract_objectives(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract airfoil-specific objectives from CFD results."""
        objectives = {}
        
        for obj_config in self.config.objectives:
            obj_name = obj_config.name
            
            if obj_name in ['drag_coefficient', 'cd']:
                objectives[obj_name] = self._extract_drag_coefficient(results, obj_config)
            elif obj_name in ['lift_coefficient', 'cl']:
                objectives[obj_name] = self._extract_lift_coefficient(results, obj_config)
            elif obj_name in ['lift_to_drag_ratio', 'cl_cd']:
                cd = self._extract_drag_coefficient(results, obj_config)
                cl = self._extract_lift_coefficient(results, obj_config)
                objectives[obj_name] = cl / cd if cd != 0 else 0.0
            elif obj_name in ['moment_coefficient', 'cm']:
                objectives[obj_name] = self._extract_moment_coefficient(results, obj_config)
            else:
                # Generic objective extraction
                objectives[obj_name] = results.get(obj_name, 0.0)
        
        return objectives
    
    def _extract_drag_coefficient(self, results: Dict[str, Any], obj_config) -> float:
        """Extract drag coefficient from forces."""
        # First check if drag coefficient is directly available
        if 'drag_coefficient' in results:
            cd = results['drag_coefficient']
            print(f"DEBUG: Found direct drag_coefficient: {cd}")
            return cd
        
        # Fallback to force calculation
        if 'forces' not in results:
            print("DEBUG: No forces data found in results")
            return 0.0
        
        forces = results['forces']
        patches = obj_config.patches if obj_config.patches else [self.airfoil_patch]
        
        total_drag = 0.0
        for patch in patches:
            if patch in forces:
                force_vector = np.array(forces[patch])
                # Drag is force component in flow direction
                drag_force = np.dot(force_vector, self.flow_direction)
                total_drag += drag_force
        
        # Convert to coefficient
        rho = self.reference_values['rho']
        u_inf = self.reference_values['magUInf']
        chord = self.reference_values['chord']
        
        q_inf = 0.5 * rho * u_inf**2
        cd = total_drag / (q_inf * chord)
        
        print(f"DEBUG: Calculated drag_coefficient from forces: {cd}")
        return cd
    
    def _extract_lift_coefficient(self, results: Dict[str, Any], obj_config) -> float:
        """Extract lift coefficient from forces."""
        # First check if lift coefficient is directly available
        if 'lift_coefficient' in results:
            cl = results['lift_coefficient']
            print(f"DEBUG: Found direct lift_coefficient: {cl}")
            return cl
        
        # Fallback to force calculation
        if 'forces' not in results:
            print("DEBUG: No forces data found in results")
            return 0.0
        
        forces = results['forces']
        patches = obj_config.patches if obj_config.patches else [self.airfoil_patch]
        
        # Lift direction is perpendicular to flow direction (in xy plane)
        lift_direction = np.array([-self.flow_direction[1], self.flow_direction[0], 0.0])
        lift_direction = lift_direction / np.linalg.norm(lift_direction)
        
        total_lift = 0.0
        for patch in patches:
            if patch in forces:
                force_vector = np.array(forces[patch])
                # Lift is force component perpendicular to flow
                lift_force = np.dot(force_vector, lift_direction)
                total_lift += lift_force
        
        # Convert to coefficient
        rho = self.reference_values['rho']
        u_inf = self.reference_values['magUInf']
        chord = self.reference_values['chord']
        
        q_inf = 0.5 * rho * u_inf**2
        cl = total_lift / (q_inf * chord)
        
        print(f"DEBUG: Calculated lift_coefficient from forces: {cl}")
        return cl
    
    def _extract_moment_coefficient(self, results: Dict[str, Any], obj_config) -> float:
        """Extract moment coefficient from moments."""
        if 'moments' not in results:
            return 0.0
        
        moments = results['moments']
        patches = obj_config.patches if obj_config.patches else [self.airfoil_patch]
        
        total_moment = 0.0
        for patch in patches:
            if patch in moments:
                moment_vector = np.array(moments[patch])
                # Moment about z-axis (for 2D airfoil)
                moment_z = moment_vector[2]
                total_moment += moment_z
        
        # Convert to coefficient
        rho = self.reference_values['rho']
        u_inf = self.reference_values['magUInf']
        chord = self.reference_values['chord']
        
        q_inf = 0.5 * rho * u_inf**2
        cm = total_moment / (q_inf * chord**2)
        
        return cm
    
    def get_airfoil_info(self) -> Dict[str, Any]:
        """Get airfoil-specific information."""
        info = self.get_physics_info()
        info.update({
            'airfoil_patch': self.airfoil_patch,
            'flow_direction': self.flow_direction.tolist() if self.flow_direction is not None else None,
            'reference_values': self.reference_values,
            'case_type': 'airfoil'
        })
        
        return info