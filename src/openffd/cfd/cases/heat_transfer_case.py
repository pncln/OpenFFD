"""Heat transfer case handler."""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from .base_case import GenericCase
from ..core.config import CaseConfig
from ..core.registry import register_case_handler


@register_case_handler('heat_transfer')
class HeatTransferCase(GenericCase):
    """Specialized case handler for heat transfer optimization."""
    
    def __init__(self, case_path: Path, config: CaseConfig):
        super().__init__(case_path, config)
        self.heat_transfer_patches = []
        self.temperature_patches = []
        self.reference_values = {}
    
    def detect_case_type(self) -> str:
        """Detect if this is a heat transfer case."""
        # Call parent validation first
        super().detect_case_type()
        
        # Look for heat transfer indicators
        heat_indicators = [
            'heat' in str(self.case_path).lower(),
            'thermal' in str(self.case_path).lower(),
            'temperature' in str(self.case_path).lower(),
            'convection' in str(self.case_path).lower()
        ]
        
        if any(heat_indicators):
            return 'heat_transfer'
        
        # Check for heat transfer specific files
        thermal_files = [
            'constant/thermophysicalProperties',
            'constant/radiationProperties',
            'constant/turbulenceProperties'
        ]
        
        for file_path in thermal_files:
            if (self.case_path / file_path).exists():
                return 'heat_transfer'
        
        return 'generic'
    
    def validate_case(self) -> bool:
        """Validate heat transfer case setup."""
        if not super().validate_case():
            return False
        
        # Identify heat transfer patches
        self._identify_heat_transfer_patches()
        
        # Parse thermal properties
        self._parse_thermal_properties()
        
        return True
    
    def _identify_heat_transfer_patches(self) -> None:
        """Identify patches relevant for heat transfer."""
        # Look for patches with heat transfer boundary conditions
        heat_patch_names = ['hot', 'cold', 'wall', 'heater', 'cooler', 'surface']
        
        for patch in self.boundary_patches:
            patch_name = patch['name'].lower()
            
            # Check for heat transfer patch names
            for name in heat_patch_names:
                if name in patch_name:
                    if 'hot' in patch_name or 'heater' in patch_name:
                        self.heat_transfer_patches.append(patch['name'])
                    elif 'cold' in patch_name or 'cooler' in patch_name:
                        self.temperature_patches.append(patch['name'])
                    elif 'wall' in patch_name or 'surface' in patch_name:
                        self.heat_transfer_patches.append(patch['name'])
        
        # If no specific patches found, use all wall patches
        if not self.heat_transfer_patches:
            wall_patches = [p for p in self.boundary_patches if p['type'] == 'wall']
            self.heat_transfer_patches = [p['name'] for p in wall_patches]
    
    def _parse_thermal_properties(self) -> None:
        """Parse thermal properties from case files."""
        # Default thermal properties
        self.reference_values = {
            'Pr': 0.7,  # Prandtl number
            'rho': 1.225,  # Density
            'Cp': 1005.0,  # Specific heat
            'k': 0.025,  # Thermal conductivity
            'mu': 1.8e-5,  # Dynamic viscosity
            'T_ref': 300.0,  # Reference temperature
            'beta': 3.3e-3  # Thermal expansion coefficient
        }
        
        # Try to read from thermophysicalProperties
        thermo_props = self.case_path / 'constant' / 'thermophysicalProperties'
        if thermo_props.exists():
            try:
                with open(thermo_props, 'r') as f:
                    content = f.read()
                
                # Parse basic properties (simplified)
                prop_patterns = {
                    'Pr': r'Pr\s+(\d+\.?\d*)',
                    'rho': r'rho\s+(\d+\.?\d*)',
                    'Cp': r'Cp\s+(\d+\.?\d*)',
                    'k': r'k\s+(\d+\.?\d*)',
                    'mu': r'mu\s+(\d+\.?\d*)'
                }
                
                for prop, pattern in prop_patterns.items():
                    match = re.search(pattern, content)
                    if match:
                        self.reference_values[prop] = float(match.group(1))
            
            except Exception as e:
                print(f"Warning: Could not parse thermal properties: {e}")
        
        # Update with config constants
        self.reference_values.update(self.config.constants)
    
    def setup_optimization_domain(self) -> Dict[str, Any]:
        """Set up optimization domain for heat transfer case."""
        # Get base domain setup
        domain_info = super().setup_optimization_domain()
        
        # Add heat transfer specific parameters
        domain_info.update({
            'heat_transfer_patches': self.heat_transfer_patches,
            'temperature_patches': self.temperature_patches,
            'reference_values': self.reference_values
        })
        
        return domain_info
    
    def extract_objectives(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract heat transfer objectives from CFD results."""
        objectives = {}
        
        for obj_config in self.config.objectives:
            obj_name = obj_config.name
            
            if obj_name in ['heat_transfer_coefficient', 'h', 'htc']:
                objectives[obj_name] = self._extract_heat_transfer_coefficient(results, obj_config)
            elif obj_name in ['nusselt_number', 'nu']:
                objectives[obj_name] = self._extract_nusselt_number(results, obj_config)
            elif obj_name in ['heat_flux', 'q']:
                objectives[obj_name] = self._extract_heat_flux(results, obj_config)
            elif obj_name in ['temperature_uniformity', 'temp_uniformity']:
                objectives[obj_name] = self._extract_temperature_uniformity(results, obj_config)
            elif obj_name in ['pressure_drop', 'dp']:
                objectives[obj_name] = self._extract_pressure_drop(results, obj_config)
            else:
                # Generic objective extraction
                objectives[obj_name] = results.get(obj_name, 0.0)
        
        return objectives
    
    def _extract_heat_transfer_coefficient(self, results: Dict[str, Any], obj_config) -> float:
        """Extract heat transfer coefficient."""
        if 'heat_flux' not in results or 'wall_temperature' not in results:
            return 0.0
        
        patches = obj_config.patches if obj_config.patches else self.heat_transfer_patches
        
        total_htc = 0.0
        patch_count = 0
        
        for patch in patches:
            if patch in results['heat_flux'] and patch in results['wall_temperature']:
                q_wall = results['heat_flux'][patch]  # Heat flux at wall
                T_wall = results['wall_temperature'][patch]  # Wall temperature
                T_fluid = self.reference_values.get('T_ref', 300.0)  # Reference fluid temperature
                
                if abs(T_wall - T_fluid) > 1e-6:
                    h = q_wall / (T_wall - T_fluid)
                    total_htc += h
                    patch_count += 1
        
        return total_htc / patch_count if patch_count > 0 else 0.0
    
    def _extract_nusselt_number(self, results: Dict[str, Any], obj_config) -> float:
        """Extract Nusselt number."""
        h = self._extract_heat_transfer_coefficient(results, obj_config)
        
        # Characteristic length (assume from mesh bounds)
        mesh_bounds = self._get_mesh_bounds()
        L_char = max(
            mesh_bounds['x_max'] - mesh_bounds['x_min'],
            mesh_bounds['y_max'] - mesh_bounds['y_min']
        )
        
        k = self.reference_values.get('k', 0.025)
        
        if k > 0:
            return h * L_char / k
        else:
            return 0.0
    
    def _extract_heat_flux(self, results: Dict[str, Any], obj_config) -> float:
        """Extract total heat flux."""
        if 'heat_flux' not in results:
            return 0.0
        
        patches = obj_config.patches if obj_config.patches else self.heat_transfer_patches
        
        total_flux = 0.0
        for patch in patches:
            if patch in results['heat_flux']:
                total_flux += results['heat_flux'][patch]
        
        return total_flux
    
    def _extract_temperature_uniformity(self, results: Dict[str, Any], obj_config) -> float:
        """Extract temperature uniformity metric."""
        if 'wall_temperature' not in results:
            return 0.0
        
        patches = obj_config.patches if obj_config.patches else self.heat_transfer_patches
        
        temperatures = []
        for patch in patches:
            if patch in results['wall_temperature']:
                temperatures.append(results['wall_temperature'][patch])
        
        if len(temperatures) < 2:
            return 1.0  # Perfect uniformity for single value
        
        # Calculate coefficient of variation (lower is more uniform)
        temperatures = np.array(temperatures)
        mean_temp = np.mean(temperatures)
        std_temp = np.std(temperatures)
        
        if mean_temp > 0:
            cv = std_temp / mean_temp
            return 1.0 / (1.0 + cv)  # Convert to uniformity metric (1 = perfect, 0 = non-uniform)
        else:
            return 0.0
    
    def _extract_pressure_drop(self, results: Dict[str, Any], obj_config) -> float:
        """Extract pressure drop between inlet and outlet."""
        if 'pressure' not in results:
            return 0.0
        
        pressure_data = results['pressure']
        
        # Try to find inlet and outlet patches
        inlet_patches = [p for p in self.boundary_patches if 'inlet' in p['name'].lower()]
        outlet_patches = [p for p in self.boundary_patches if 'outlet' in p['name'].lower()]
        
        if not inlet_patches or not outlet_patches:
            return 0.0
        
        inlet_pressure = pressure_data.get(inlet_patches[0]['name'], 0.0)
        outlet_pressure = pressure_data.get(outlet_patches[0]['name'], 0.0)
        
        return inlet_pressure - outlet_pressure
    
    def get_heat_transfer_info(self) -> Dict[str, Any]:
        """Get heat transfer specific information."""
        info = self.get_physics_info()
        info.update({
            'heat_transfer_patches': self.heat_transfer_patches,
            'temperature_patches': self.temperature_patches,
            'reference_values': self.reference_values,
            'case_type': 'heat_transfer'
        })
        
        return info