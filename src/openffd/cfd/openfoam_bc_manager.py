#!/usr/bin/env python3
"""
Universal OpenFOAM Boundary Condition Manager

Automatically detects and parses boundary conditions from OpenFOAM case directories,
providing a universal interface that works with any OpenFOAM case without manual configuration.

Features:
- Automatic case directory detection
- OpenFOAM field file parsing (U, p, T, etc.)
- Universal BC type mapping
- Patch-based boundary condition application
- Support for 0, 0_orig, and timestep directories
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OpenFOAMBCType(Enum):
    """OpenFOAM boundary condition types."""
    # Velocity boundary conditions
    FIXED_VALUE = "fixedValue"
    ZERO_GRADIENT = "zeroGradient"  
    SYMMETRY = "symmetry"
    INLET_OUTLET = "inletOutlet"
    OUTLET_INLET = "outletInlet"
    FREE_STREAM = "freestream"
    NO_SLIP = "noSlip"
    SLIP = "slip"
    
    # Pressure boundary conditions
    FIXED_FLUX_PRESSURE = "fixedFluxPressure"
    TOTAL_PRESSURE = "totalPressure"
    
    # Wall functions
    WALL_FUNCTION = "nutkWallFunction"


@dataclass
class OpenFOAMFieldBC:
    """OpenFOAM boundary condition for a specific field."""
    patch_name: str
    field_name: str
    bc_type: str
    value: Optional[Union[float, List[float], np.ndarray]] = None
    gradient: Optional[Union[float, List[float], np.ndarray]] = None
    inlet_value: Optional[Union[float, List[float], np.ndarray]] = None
    outlet_value: Optional[Union[float, List[float], np.ndarray]] = None
    normal_vector: Optional[np.ndarray] = None
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class UniversalOpenFOAMBCManager:
    """Universal boundary condition manager for OpenFOAM cases."""
    
    def __init__(self, case_directory: Union[str, Path]):
        """
        Initialize with OpenFOAM case directory.
        
        Args:
            case_directory: Path to OpenFOAM case directory
        """
        self.case_directory = Path(case_directory)
        self.field_bcs = {}  # field_name -> {patch_name -> OpenFOAMFieldBC}
        self.internal_fields = {}  # field_name -> internal field value
        self.dimensions = {}  # field_name -> OpenFOAM dimensions
        
        # Detect and parse boundary conditions
        self._detect_case_structure()
        self._parse_all_field_files()
        
        logger.info(f"OpenFOAM BC Manager initialized for case: {case_directory}")
        logger.info(f"Found fields: {list(self.field_bcs.keys())}")
        
    def _detect_case_structure(self):
        """Detect OpenFOAM case directory structure."""
        self.time_directories = []
        
        # Look for time directories (0, 0_orig, timesteps)
        for item in self.case_directory.iterdir():
            if item.is_dir():
                # Check if it looks like a time directory
                if (item.name == "0" or item.name == "0_orig" or 
                    self._is_timestep_directory(item.name)):
                    self.time_directories.append(item)
        
        # Sort time directories, preferring 0_orig > 0 > timesteps
        self.time_directories.sort(key=self._time_directory_priority)
        
        if not self.time_directories:
            raise FileNotFoundError(f"No OpenFOAM time directories found in {self.case_directory}")
            
        self.primary_time_dir = self.time_directories[0]
        logger.info(f"Using time directory: {self.primary_time_dir}")
        
    def _is_timestep_directory(self, name: str) -> bool:
        """Check if directory name looks like a timestep."""
        try:
            float(name)
            return True
        except ValueError:
            return False
            
    def _time_directory_priority(self, time_dir: Path) -> int:
        """Priority for time directory selection."""
        if time_dir.name == "0_orig":
            return 0  # Highest priority
        elif time_dir.name == "0":
            return 1  # Second priority
        else:
            try:
                return 2 + float(time_dir.name)  # Timesteps
            except ValueError:
                return 999  # Unknown
                
    def _parse_all_field_files(self):
        """Parse all field files in the primary time directory."""
        for field_file in self.primary_time_dir.iterdir():
            if field_file.is_file() and not field_file.name.startswith('.'):
                try:
                    field_name = field_file.name
                    self._parse_field_file(field_file, field_name)
                except Exception as e:
                    logger.warning(f"Failed to parse field file {field_file}: {e}")
                    
    def _parse_field_file(self, field_file: Path, field_name: str):
        """Parse a single OpenFOAM field file."""
        logger.debug(f"Parsing field file: {field_file}")
        
        with open(field_file, 'r') as f:
            content = f.read()
        
        # Parse dimensions
        dimensions_match = re.search(r'dimensions\s*\[\s*([\d\s\-]+)\s*\]', content)
        if dimensions_match:
            self.dimensions[field_name] = dimensions_match.group(1).split()
        
        # Parse internal field
        internal_field = self._parse_internal_field(content)
        if internal_field is not None:
            self.internal_fields[field_name] = internal_field
            
        # Parse boundary field
        boundary_bcs = self._parse_boundary_field(content, field_name)
        if boundary_bcs:
            self.field_bcs[field_name] = boundary_bcs
            
    def _parse_internal_field(self, content: str) -> Optional[Union[float, np.ndarray]]:
        """Parse internalField from OpenFOAM field file."""
        # Match: internalField uniform (10 0 0); or internalField uniform 0;
        uniform_match = re.search(r'internalField\s+uniform\s+\(([^)]+)\)', content)
        if uniform_match:
            values = uniform_match.group(1).split()
            if len(values) == 1:
                return float(values[0])
            else:
                return np.array([float(v) for v in values])
                
        # Match scalar uniform
        scalar_match = re.search(r'internalField\s+uniform\s+([\d\.\-e]+)', content)
        if scalar_match:
            return float(scalar_match.group(1))
            
        return None
        
    def _parse_boundary_field(self, content: str, field_name: str) -> Dict[str, OpenFOAMFieldBC]:
        """Parse boundaryField section from OpenFOAM field file."""
        boundary_bcs = {}
        
        # Find boundaryField section using a more robust approach
        boundary_start = content.find('boundaryField')
        if boundary_start == -1:
            return boundary_bcs
            
        # Find opening brace
        brace_start = content.find('{', boundary_start)
        if brace_start == -1:
            return boundary_bcs
            
        # Find matching closing brace by counting braces
        brace_count = 0
        brace_end = brace_start
        for i, char in enumerate(content[brace_start:]):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    brace_end = brace_start + i
                    break
                    
        if brace_count != 0:
            return boundary_bcs
            
        boundary_content = content[brace_start+1:brace_end]
        
        # Parse patches using line-by-line approach
        boundary_bcs = self._parse_patches_line_by_line(boundary_content, field_name)
                
        return boundary_bcs
        
    def _parse_patches_line_by_line(self, boundary_content: str, field_name: str) -> Dict[str, OpenFOAMFieldBC]:
        """Parse patches using line-by-line approach for robustness."""
        boundary_bcs = {}
        lines = boundary_content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('//') or line.startswith('/*'):
                i += 1
                continue
                
            # Check if this line is a potential patch name (alphanumeric word)
            # and the next line contains an opening brace
            if (line.replace('_', '').replace('1', '').replace('2', '').isalpha() and 
                i + 1 < len(lines) and '{' in lines[i + 1]):
                
                patch_name = line
                # Find patch content starting from the brace line
                patch_content, next_i = self._extract_patch_content(lines, i + 1)
                if patch_content:
                    bc = self._parse_patch_bc(patch_name, patch_content, field_name)
                    if bc:
                        boundary_bcs[patch_name] = bc
                i = next_i
                
            else:
                i += 1
                
        return boundary_bcs
        
    def _extract_patch_content(self, lines: List[str], start_line: int) -> Tuple[str, int]:
        """Extract content of a patch definition."""
        content_lines = []
        brace_count = 0
        
        # Start from the line with the patch name
        first_line = lines[start_line].strip()
        if '{' in first_line:
            brace_count += first_line.count('{')
            brace_count -= first_line.count('}')
            
        i = start_line + 1
        while i < len(lines) and brace_count > 0:
            line = lines[i].strip()
            if line:
                content_lines.append(line)
                brace_count += line.count('{')
                brace_count -= line.count('}')
            i += 1
            
        return '\n'.join(content_lines), i
        
    def _parse_patch_bc(self, patch_name: str, patch_content: str, field_name: str) -> Optional[OpenFOAMFieldBC]:
        """Parse boundary condition for a specific patch."""
        bc = OpenFOAMFieldBC(patch_name=patch_name, field_name=field_name, bc_type="")
        
        # Parse type
        type_match = re.search(r'type\s+(\w+)', patch_content)
        if type_match:
            bc.bc_type = type_match.group(1)
        else:
            return None
            
        # Parse value
        bc.value = self._parse_field_value(patch_content, 'value')
        
        # Parse gradient
        bc.gradient = self._parse_field_value(patch_content, 'gradient')
        
        # Parse inletValue
        bc.inlet_value = self._parse_field_value(patch_content, 'inletValue')
        
        # Parse outletValue
        bc.outlet_value = self._parse_field_value(patch_content, 'outletValue')
        
        # Handle references to internalField
        if bc.value is None and '$internalField' in patch_content:
            bc.value = self.internal_fields.get(field_name)
        if bc.inlet_value is None and '$internalField' in patch_content:
            bc.inlet_value = self.internal_fields.get(field_name)
        if bc.outlet_value is None and '$internalField' in patch_content:
            bc.outlet_value = self.internal_fields.get(field_name)
            
        return bc
        
    def _parse_field_value(self, content: str, key: str) -> Optional[Union[float, np.ndarray]]:
        """Parse field value (uniform vector or scalar)."""
        # Vector uniform: value uniform (10 0 0);
        vector_pattern = rf'{key}\s+uniform\s+\(([^)]+)\)'
        vector_match = re.search(vector_pattern, content)
        if vector_match:
            values = vector_match.group(1).split()
            if len(values) == 1:
                return float(values[0])
            else:
                return np.array([float(v) for v in values])
                
        # Scalar uniform: value uniform 0;
        scalar_pattern = rf'{key}\s+uniform\s+([\d\.\-e]+)'
        scalar_match = re.search(scalar_pattern, content)
        if scalar_match:
            return float(scalar_match.group(1))
            
        return None
        
    def get_velocity_bcs(self) -> Dict[str, OpenFOAMFieldBC]:
        """Get velocity boundary conditions."""
        return self.field_bcs.get('U', {})
        
    def get_pressure_bcs(self) -> Dict[str, OpenFOAMFieldBC]:
        """Get pressure boundary conditions."""
        return self.field_bcs.get('p', {})
        
    def get_field_bcs(self, field_name: str) -> Dict[str, OpenFOAMFieldBC]:
        """Get boundary conditions for specific field."""
        return self.field_bcs.get(field_name, {})
        
    def get_patch_names(self) -> List[str]:
        """Get all patch names found in the case."""
        patches = set()
        for field_bcs in self.field_bcs.values():
            patches.update(field_bcs.keys())
        return list(patches)
        
    def map_to_cfd_boundary_conditions(self, mesh_patches: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Map OpenFOAM boundary conditions to universal CFD boundary conditions.
        
        Args:
            mesh_patches: List of patch names from the mesh
            
        Returns:
            Dictionary mapping patch names to CFD boundary condition specifications
        """
        cfd_bcs = {}
        
        velocity_bcs = self.get_velocity_bcs()
        pressure_bcs = self.get_pressure_bcs()
        
        for patch_name in mesh_patches:
            if patch_name in velocity_bcs or patch_name in pressure_bcs:
                cfd_bc = self._convert_patch_to_cfd_bc(
                    patch_name,
                    velocity_bcs.get(patch_name),
                    pressure_bcs.get(patch_name)
                )
                if cfd_bc:
                    cfd_bcs[patch_name] = cfd_bc
                    
        logger.info(f"Mapped {len(cfd_bcs)} patches to CFD boundary conditions")
        for patch_name, bc_data in cfd_bcs.items():
            logger.info(f"  {patch_name}: {bc_data['type']}")
            
        return cfd_bcs
        
    def _convert_patch_to_cfd_bc(self, patch_name: str, 
                               velocity_bc: Optional[OpenFOAMFieldBC],
                               pressure_bc: Optional[OpenFOAMFieldBC]) -> Optional[Dict[str, Any]]:
        """Convert OpenFOAM patch BCs to CFD solver format."""
        
        if not velocity_bc and not pressure_bc:
            return None
            
        cfd_bc = {
            'patch_name': patch_name,
            'type': 'unknown',
            'velocity': None,
            'pressure': None,
            'velocity_fixed': False,
            'pressure_fixed': False
        }
        
        # Determine primary BC type based on velocity BC
        if velocity_bc:
            if velocity_bc.bc_type == 'fixedValue':
                if velocity_bc.value is not None:
                    if isinstance(velocity_bc.value, np.ndarray):
                        if np.allclose(velocity_bc.value, 0):
                            cfd_bc['type'] = 'wall'
                            cfd_bc['velocity'] = velocity_bc.value
                            cfd_bc['velocity_fixed'] = True
                        else:
                            cfd_bc['type'] = 'inlet'
                            cfd_bc['velocity'] = velocity_bc.value
                            cfd_bc['velocity_fixed'] = True
                    else:
                        cfd_bc['type'] = 'wall'
                        cfd_bc['velocity'] = np.array([0.0, 0.0, 0.0])
                        cfd_bc['velocity_fixed'] = True
                        
            elif velocity_bc.bc_type == 'symmetry':
                cfd_bc['type'] = 'symmetry'
                
            elif velocity_bc.bc_type in ['inletOutlet', 'outletInlet']:
                cfd_bc['type'] = 'farfield'
                if velocity_bc.inlet_value is not None:
                    cfd_bc['velocity'] = velocity_bc.inlet_value
                    cfd_bc['velocity_fixed'] = True
                elif velocity_bc.value is not None:
                    cfd_bc['velocity'] = velocity_bc.value
                    cfd_bc['velocity_fixed'] = True
                    
            elif velocity_bc.bc_type == 'zeroGradient':
                cfd_bc['type'] = 'outlet'
                
        # Add pressure information
        if pressure_bc:
            if pressure_bc.bc_type == 'fixedValue':
                cfd_bc['pressure'] = pressure_bc.value
                cfd_bc['pressure_fixed'] = True
            elif pressure_bc.bc_type in ['outletInlet', 'inletOutlet']:
                if pressure_bc.outlet_value is not None:
                    cfd_bc['pressure'] = pressure_bc.outlet_value
                elif pressure_bc.value is not None:
                    cfd_bc['pressure'] = pressure_bc.value
                    
        return cfd_bc
        
    def apply_to_navier_stokes_solver(self, solver, mesh_patches: List[str]):
        """
        Apply boundary conditions directly to a Navier-Stokes solver.
        
        Args:
            solver: NavierStokesSolver instance
            mesh_patches: List of mesh patch names
        """
        from .navier_stokes_solver import BoundaryCondition, BoundaryType
        
        cfd_bcs = self.map_to_cfd_boundary_conditions(mesh_patches)
        
        for patch_name, bc_data in cfd_bcs.items():
            try:
                # Map CFD BC type to solver BC type
                if bc_data['type'] == 'wall':
                    bc_type = BoundaryType.WALL
                    velocity = np.array(bc_data['velocity']) if bc_data['velocity'] is not None else np.array([0.0, 0.0, 0.0])
                    bc = BoundaryCondition(boundary_type=bc_type, velocity=velocity)
                    
                elif bc_data['type'] == 'farfield':
                    bc_type = BoundaryType.FARFIELD
                    velocity = np.array(bc_data['velocity']) if bc_data['velocity'] is not None else np.array([10.0, 0.0, 0.0])
                    pressure = bc_data['pressure'] if bc_data['pressure'] is not None else 101325.0
                    bc = BoundaryCondition(boundary_type=bc_type, velocity=velocity, pressure=pressure)
                    
                elif bc_data['type'] == 'inlet':
                    bc_type = BoundaryType.INLET
                    velocity = np.array(bc_data['velocity']) if bc_data['velocity'] is not None else np.array([10.0, 0.0, 0.0])
                    bc = BoundaryCondition(boundary_type=bc_type, velocity=velocity)
                    
                elif bc_data['type'] == 'outlet':
                    bc_type = BoundaryType.OUTLET
                    bc = BoundaryCondition(boundary_type=bc_type)
                    
                elif bc_data['type'] == 'symmetry':
                    bc_type = BoundaryType.SYMMETRY
                    bc = BoundaryCondition(boundary_type=bc_type)
                    
                else:
                    logger.warning(f"Unknown BC type '{bc_data['type']}' for patch '{patch_name}'")
                    continue
                    
                # Apply to solver
                solver.set_boundary_condition(patch_name, bc)
                logger.info(f"Applied OpenFOAM BC to solver: {patch_name} -> {bc_data['type']}")
                
            except Exception as e:
                logger.error(f"Failed to apply BC for patch '{patch_name}': {e}")
                
    def __str__(self) -> str:
        """String representation of the BC manager."""
        lines = [f"OpenFOAM BC Manager: {self.case_directory}"]
        lines.append(f"Time directory: {self.primary_time_dir}")
        lines.append(f"Fields found: {list(self.field_bcs.keys())}")
        lines.append(f"Patches found: {self.get_patch_names()}")
        return "\n".join(lines)


def create_universal_bc_manager(case_directory: Union[str, Path]) -> UniversalOpenFOAMBCManager:
    """
    Factory function to create a universal OpenFOAM boundary condition manager.
    
    Args:
        case_directory: Path to OpenFOAM case directory
        
    Returns:
        UniversalOpenFOAMBCManager instance
    """
    return UniversalOpenFOAMBCManager(case_directory)