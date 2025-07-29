#!/usr/bin/env python3
"""
OpenFOAM Boundary Condition Parser

This module provides a general parser for OpenFOAM boundary condition files,
allowing automatic detection and parsing of boundary conditions from the 0/ directory
regardless of the specific case. It supports all standard OpenFOAM field types and
boundary conditions.

The parser extracts boundary conditions for:
- Velocity (U) - vector field
- Pressure (p) - scalar field  
- Turbulence variables (k, omega, epsilon, nuTilda, nut) - scalar fields
- Temperature (T) - scalar field
- Custom fields - automatically detected

Author: OpenFFD CFD Framework
"""

import os
import re
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OpenFOAMBoundaryCondition:
    """Represents a single boundary condition for a patch."""
    patch_name: str
    field_name: str
    bc_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dimensions: Optional[List[int]] = None
    internal_field: Optional[Any] = None


@dataclass
class OpenFOAMField:
    """Represents an OpenFOAM field with its boundary conditions."""
    field_name: str
    field_class: str  # volScalarField, volVectorField, etc.
    dimensions: List[int]
    internal_field: Any
    boundary_conditions: Dict[str, OpenFOAMBoundaryCondition] = field(default_factory=dict)


class OpenFOAMBoundaryParser:
    """
    Parser for OpenFOAM boundary condition files.
    
    Automatically detects and parses all field files in the 0/ directory,
    extracting boundary conditions in a standardized format that can be
    used by any CFD solver.
    """
    
    # Standard OpenFOAM field files and their expected types
    STANDARD_FIELDS = {
        'U': 'volVectorField',
        'p': 'volScalarField', 
        'T': 'volScalarField',
        'k': 'volScalarField',
        'omega': 'volScalarField', 
        'epsilon': 'volScalarField',
        'nuTilda': 'volScalarField',
        'nut': 'volScalarField',
        'alphat': 'volScalarField',
        'rho': 'volScalarField',
        'mu': 'volScalarField'
    }
    
    # Common boundary condition types and their parameters
    BC_TYPE_PARAMS = {
        'fixedValue': ['value'],
        'zeroGradient': [],
        'symmetry': [],
        'wall': [],
        'inletOutlet': ['inletValue', 'value'],
        'outletInlet': ['outletValue', 'value'],
        'pressureInletOutletVelocity': ['value'],
        'totalPressure': ['p0', 'value'],
        'fixedFluxPressure': ['value'],
        'freestreamPressure': ['freestreamValue', 'value'],
        'slip': [],
        'noSlip': [],
        'empty': [],
        'cyclic': [],
        'processor': []
    }
    
    def __init__(self, case_directory: str):
        """
        Initialize parser for a specific OpenFOAM case.
        
        Args:
            case_directory: Path to OpenFOAM case directory
        """
        self.case_directory = Path(case_directory)
        self.zero_directory = self.case_directory / "0"
        
        # Check for 0_orig directory as fallback
        if not self.zero_directory.exists():
            self.zero_directory = self.case_directory / "0_orig"
            
        if not self.zero_directory.exists():
            raise FileNotFoundError(f"Neither '0' nor '0_orig' directory found in {case_directory}")
            
        self.fields = {}
        self.patch_names = set()
        
    def parse_all_fields(self) -> Dict[str, OpenFOAMField]:
        """
        Parse all field files in the 0/ directory.
        
        Returns:
            Dictionary mapping field names to OpenFOAMField objects
        """
        logger.info(f"Parsing boundary conditions from {self.zero_directory}")
        
        # Find all field files
        field_files = []
        for file_path in self.zero_directory.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                field_files.append(file_path)
                
        logger.info(f"Found {len(field_files)} field files: {[f.name for f in field_files]}")
        
        # Parse each field file
        for field_file in field_files:
            try:
                field = self._parse_field_file(field_file)
                if field:
                    self.fields[field.field_name] = field
                    # Collect patch names
                    self.patch_names.update(field.boundary_conditions.keys())
            except Exception as e:
                logger.warning(f"Failed to parse field file {field_file.name}: {e}")
                
        logger.info(f"Successfully parsed {len(self.fields)} fields")
        logger.info(f"Detected patches: {sorted(self.patch_names)}")
        
        return self.fields
    
    def _parse_field_file(self, file_path: Path) -> Optional[OpenFOAMField]:
        """Parse a single OpenFOAM field file."""
        field_name = file_path.name
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Could not read file {file_path}: {e}")
            return None
            
        # Parse FoamFile header
        foam_file_info = self._parse_foam_file_header(content)
        if not foam_file_info:
            logger.warning(f"Could not parse FoamFile header in {field_name}")
            return None
            
        field_class = foam_file_info.get('class', 'unknown')
        
        # Parse dimensions
        dimensions = self._parse_dimensions(content)
        
        # Parse internal field
        internal_field = self._parse_internal_field(content)
        
        # Parse boundary conditions
        boundary_conditions = self._parse_boundary_field(content, field_name)
        
        field = OpenFOAMField(
            field_name=field_name,
            field_class=field_class,
            dimensions=dimensions,
            internal_field=internal_field,
            boundary_conditions=boundary_conditions
        )
        
        logger.debug(f"Parsed field {field_name}: {len(boundary_conditions)} boundary conditions")
        return field
    
    def _parse_foam_file_header(self, content: str) -> Dict[str, str]:
        """Parse the FoamFile header section."""
        foam_file_match = re.search(r'FoamFile\s*\{([^}]+)\}', content, re.DOTALL)
        if not foam_file_match:
            return {}
            
        header_content = foam_file_match.group(1)
        header_info = {}
        
        # Parse key-value pairs
        for line in header_content.split('\n'):
            line = line.strip()
            if line and not line.startswith('//'):
                match = re.match(r'(\w+)\s+(.+?);', line)
                if match:
                    key, value = match.groups()
                    header_info[key] = value.strip()
                    
        return header_info
    
    def _parse_dimensions(self, content: str) -> List[int]:
        """Parse the dimensions array."""
        dim_match = re.search(r'dimensions\s*\[(.*?)\]', content)
        if dim_match:
            dim_str = dim_match.group(1)
            try:
                return [int(x.strip()) for x in dim_str.split()]
            except ValueError:
                logger.warning("Could not parse dimensions")
                
        return [0, 0, 0, 0, 0, 0, 0]  # Default dimensions
    
    def _parse_internal_field(self, content: str) -> Any:
        """Parse the internalField value."""
        # Match various internal field formats
        patterns = [
            r'internalField\s+uniform\s+\(([^)]+)\)',  # Vector uniform
            r'internalField\s+uniform\s+([^;]+);',      # Scalar uniform
            r'internalField\s+\$(\w+)',                 # Reference to variable
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                value_str = match.group(1).strip()
                return self._parse_value(value_str)
                
        return None
    
    def _parse_boundary_field(self, content: str, field_name: str) -> Dict[str, OpenFOAMBoundaryCondition]:
        """Parse the boundaryField section."""
        boundary_conditions = {}
        
        # Find boundaryField section
        boundary_match = re.search(r'boundaryField\s*\{(.*)\}', content, re.DOTALL)
        if not boundary_match:
            logger.warning(f"No boundaryField section found in {field_name}")
            return boundary_conditions
            
        boundary_content = boundary_match.group(1)
        
        # Parse each patch
        patch_pattern = r'(\w+)\s*\{([^}]+)\}'
        for match in re.finditer(patch_pattern, boundary_content):
            patch_name = match.group(1).strip()
            patch_content = match.group(2).strip()
            
            # Skip if this looks like a nested structure
            if '{' in patch_content:
                continue
                
            bc = self._parse_patch_boundary_condition(patch_name, patch_content, field_name)
            if bc:
                boundary_conditions[patch_name] = bc
                
        return boundary_conditions
    
    def _parse_patch_boundary_condition(self, patch_name: str, content: str, field_name: str) -> Optional[OpenFOAMBoundaryCondition]:
        """Parse boundary condition for a single patch."""
        parameters = {}
        bc_type = None
        
        # Parse each line in the patch definition
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            # Parse type
            type_match = re.match(r'type\s+(\w+);', line)
            if type_match:
                bc_type = type_match.group(1)
                continue
                
            # Parse other parameters
            param_match = re.match(r'(\w+)\s+(.+?);', line)
            if param_match:
                param_name, param_value = param_match.groups()
                param_value = param_value.strip()
                
                # Handle special cases
                if param_value.startswith('$'):
                    # Reference to another variable
                    parameters[param_name] = param_value
                elif param_value.startswith('uniform'):
                    # Uniform value
                    uniform_match = re.search(r'uniform\s+(.+)', param_value)
                    if uniform_match:
                        parameters[param_name] = self._parse_value(uniform_match.group(1))
                else:
                    parameters[param_name] = self._parse_value(param_value)
        
        if not bc_type:
            logger.warning(f"No boundary condition type found for patch {patch_name} in field {field_name}")
            return None
            
        return OpenFOAMBoundaryCondition(
            patch_name=patch_name,
            field_name=field_name,
            bc_type=bc_type,
            parameters=parameters
        )
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse a value string into appropriate Python type."""
        value_str = value_str.strip()
        
        # Vector value (x y z)
        if value_str.startswith('(') and value_str.endswith(')'):
            vector_str = value_str[1:-1].strip()
            try:
                return [float(x) for x in vector_str.split()]
            except ValueError:
                return vector_str
                
        # Scalar value
        try:
            # Try integer first
            if '.' not in value_str:
                return int(value_str)
            else:
                return float(value_str)
        except ValueError:
            # Return as string if can't convert to number
            return value_str
    
    def get_boundary_condition(self, field_name: str, patch_name: str) -> Optional[OpenFOAMBoundaryCondition]:
        """Get boundary condition for specific field and patch."""
        field = self.fields.get(field_name)
        if field:
            return field.boundary_conditions.get(patch_name)
        return None
    
    def get_field_boundary_conditions(self, field_name: str) -> Dict[str, OpenFOAMBoundaryCondition]:
        """Get all boundary conditions for a specific field."""
        field = self.fields.get(field_name)
        if field:
            return field.boundary_conditions
        return {}
    
    def get_patch_boundary_conditions(self, patch_name: str) -> Dict[str, OpenFOAMBoundaryCondition]:
        """Get all boundary conditions for a specific patch across all fields."""
        patch_bcs = {}
        for field_name, field in self.fields.items():
            bc = field.boundary_conditions.get(patch_name)
            if bc:
                patch_bcs[field_name] = bc
        return patch_bcs
    
    def export_to_standard_format(self) -> Dict[str, Any]:
        """
        Export boundary conditions to a standardized format for CFD solvers.
        
        Returns:
            Dictionary with standardized boundary condition format
        """
        standard_format = {
            'patches': {},
            'fields': {},
            'summary': {
                'n_patches': len(self.patch_names),
                'n_fields': len(self.fields),
                'patch_names': sorted(self.patch_names),
                'field_names': sorted(self.fields.keys())
            }
        }
        
        # Export by patches
        for patch_name in self.patch_names:
            patch_bcs = self.get_patch_boundary_conditions(patch_name)
            standard_format['patches'][patch_name] = {}
            
            for field_name, bc in patch_bcs.items():
                standard_format['patches'][patch_name][field_name] = {
                    'type': bc.bc_type,
                    'parameters': bc.parameters
                }
        
        # Export by fields  
        for field_name, field in self.fields.items():
            standard_format['fields'][field_name] = {
                'class': field.field_class,
                'dimensions': field.dimensions,
                'internal_field': field.internal_field,
                'boundary_conditions': {}
            }
            
            for patch_name, bc in field.boundary_conditions.items():
                standard_format['fields'][field_name]['boundary_conditions'][patch_name] = {
                    'type': bc.bc_type,
                    'parameters': bc.parameters
                }
        
        return standard_format
    
    def validate_boundary_conditions(self) -> Dict[str, List[str]]:
        """
        Validate boundary conditions for consistency and completeness.
        
        Returns:
            Dictionary with validation results (warnings and errors)
        """
        warnings = []
        errors = []
        
        # Check if all patches have boundary conditions for all fields
        for field_name in self.fields:
            field = self.fields[field_name]
            for patch_name in self.patch_names:
                if patch_name not in field.boundary_conditions:
                    warnings.append(f"Patch '{patch_name}' missing boundary condition for field '{field_name}'")
        
        # Check for unknown boundary condition types
        for field_name, field in self.fields.items():
            for patch_name, bc in field.boundary_conditions.items():
                if bc.bc_type not in self.BC_TYPE_PARAMS:
                    warnings.append(f"Unknown boundary condition type '{bc.bc_type}' for patch '{patch_name}', field '{field_name}'")
        
        # Check for missing required parameters
        for field_name, field in self.fields.items():
            for patch_name, bc in field.boundary_conditions.items():
                required_params = self.BC_TYPE_PARAMS.get(bc.bc_type, [])
                for param in required_params:
                    if param not in bc.parameters:
                        errors.append(f"Missing required parameter '{param}' for BC type '{bc.bc_type}' on patch '{patch_name}', field '{field_name}'")
        
        return {
            'warnings': warnings,
            'errors': errors,
            'is_valid': len(errors) == 0
        }


def parse_openfoam_boundary_conditions(case_directory: str) -> Dict[str, Any]:
    """
    Convenience function to parse OpenFOAM boundary conditions from a case directory.
    
    Args:
        case_directory: Path to OpenFOAM case directory
        
    Returns:
        Standardized boundary condition format
    """
    parser = OpenFOAMBoundaryParser(case_directory)
    parser.parse_all_fields()
    return parser.export_to_standard_format()


def test_boundary_parser():
    """Test the boundary parser with the cylinder case."""
    import json
    
    case_dir = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/examples/Cylinder"
    
    try:
        parser = OpenFOAMBoundaryParser(case_dir)
        fields = parser.parse_all_fields()
        
        print(f"Parsed {len(fields)} fields from {case_dir}")
        print(f"Patches found: {sorted(parser.patch_names)}")
        
        # Export to standard format
        standard_format = parser.export_to_standard_format()
        
        # Validate
        validation = parser.validate_boundary_conditions()
        print(f"\nValidation results:")
        print(f"- Errors: {len(validation['errors'])}")
        print(f"- Warnings: {len(validation['warnings'])}")
        
        if validation['errors']:
            print("Errors:")
            for error in validation['errors']:
                print(f"  - {error}")
                
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings'][:5]:  # Show first 5
                print(f"  - {warning}")
            if len(validation['warnings']) > 5:
                print(f"  ... and {len(validation['warnings']) - 5} more warnings")
        
        # Show example boundary conditions
        print(f"\nExample boundary conditions for 'cylinder' patch:")
        cylinder_bcs = parser.get_patch_boundary_conditions('cylinder')
        for field_name, bc in cylinder_bcs.items():
            print(f"  {field_name}: {bc.bc_type} - {bc.parameters}")
            
        return standard_format
        
    except Exception as e:
        print(f"Error testing boundary parser: {e}")
        return None


if __name__ == "__main__":
    test_boundary_parser()