"""Generic case handler for OpenFOAM cases."""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from ..core.base import BaseCase
from ..core.config import CaseConfig
from ..core.registry import register_case_handler


@register_case_handler('generic')
class GenericCase(BaseCase):
    """Generic OpenFOAM case handler."""
    
    def __init__(self, case_path: Path, config: CaseConfig):
        super().__init__(case_path, config)
        self.boundary_patches = []
        self.solver_type = None
    
    def detect_case_type(self) -> str:
        """Detect case type from directory structure and files."""
        # Check for standard OpenFOAM structure
        required_dirs = ['constant', 'system']
        if not all((self.case_path / d).exists() for d in required_dirs):
            raise ValueError(f"Invalid OpenFOAM case structure in {self.case_path}")
        
        return 'generic'
    
    def validate_case(self) -> bool:
        """Validate OpenFOAM case setup."""
        required_files = [
            'system/controlDict',
            'system/fvSchemes',
            'system/fvSolution',
            'constant/polyMesh'
        ]
        
        for file_path in required_files:
            if not (self.case_path / file_path).exists():
                print(f"Warning: Required file missing: {file_path}")
                return False
        
        # Parse boundary patches
        self._parse_boundary_patches()
        
        # Detect solver type
        self._detect_solver_type()
        
        self._validated = True
        return True
    
    def _parse_boundary_patches(self) -> None:
        """Parse boundary patches from constant/polyMesh/boundary."""
        boundary_file = self.case_path / 'constant' / 'polyMesh' / 'boundary'
        
        if not boundary_file.exists():
            print(f"Warning: Boundary file not found: {boundary_file}")
            return
        
        with open(boundary_file, 'r') as f:
            content = f.read()
        
        # Parse boundary patches (simplified)
        patch_pattern = r'(\w+)\s*\{[^}]*type\s+(\w+)'
        matches = re.findall(patch_pattern, content)
        
        self.boundary_patches = []
        for patch_name, patch_type in matches:
            if patch_type not in ['empty', 'wedge']:
                self.boundary_patches.append({
                    'name': patch_name,
                    'type': patch_type
                })
    
    def _detect_solver_type(self) -> None:
        """Detect solver type from system/controlDict."""
        control_dict = self.case_path / 'system' / 'controlDict'
        
        if not control_dict.exists():
            return
        
        with open(control_dict, 'r') as f:
            content = f.read()
        
        # Look for application entry
        app_match = re.search(r'application\s+(\w+)', content)
        if app_match:
            self.solver_type = app_match.group(1)
        else:
            # Default based on config
            self.solver_type = self.config.solver
    
    def setup_optimization_domain(self) -> Dict[str, Any]:
        """Set up optimization domain based on mesh bounds."""
        # Get mesh bounds
        mesh_bounds = self._get_mesh_bounds()
        
        if self.config.ffd_config.domain == "auto":
            # Auto-detect domain based on mesh
            domain = self._auto_detect_domain(mesh_bounds)
        else:
            domain = self.config.ffd_config.domain
        
        return {
            'domain': domain,
            'control_points': self.config.ffd_config.control_points,
            'ffd_type': self.config.ffd_config.ffd_type,
            'mesh_bounds': mesh_bounds
        }
    
    def _get_mesh_bounds(self) -> Dict[str, float]:
        """Get mesh bounding box."""
        # Try to read points file
        points_file = self.case_path / 'constant' / 'polyMesh' / 'points'
        
        if not points_file.exists():
            print("Warning: Points file not found, using default bounds")
            return {
                'x_min': -1.0, 'x_max': 1.0,
                'y_min': -1.0, 'y_max': 1.0,
                'z_min': -0.1, 'z_max': 0.1
            }
        
        try:
            with open(points_file, 'r') as f:
                lines = f.readlines()
            
            # Find start of point data
            start_idx = None
            for i, line in enumerate(lines):
                if line.strip().isdigit():
                    start_idx = i + 2  # Skip count and opening bracket
                    break
            
            if start_idx is None:
                raise ValueError("Could not find point data start")
            
            # Parse points
            points = []
            for line in lines[start_idx:]:
                line = line.strip()
                if line == ')':
                    break
                if line.startswith('(') and line.endswith(')'):
                    coords = line[1:-1].split()
                    if len(coords) >= 3:
                        points.append([float(coords[0]), float(coords[1]), float(coords[2])])
            
            if not points:
                raise ValueError("No points found")
            
            points = np.array(points)
            
            return {
                'x_min': float(points[:, 0].min()),
                'x_max': float(points[:, 0].max()),
                'y_min': float(points[:, 1].min()),
                'y_max': float(points[:, 1].max()),
                'z_min': float(points[:, 2].min()),
                'z_max': float(points[:, 2].max())
            }
        
        except Exception as e:
            print(f"Warning: Could not parse mesh bounds: {e}")
            return {
                'x_min': -1.0, 'x_max': 1.0,
                'y_min': -1.0, 'y_max': 1.0,
                'z_min': -0.1, 'z_max': 0.1
            }
    
    def _auto_detect_domain(self, mesh_bounds: Dict[str, float]) -> Dict[str, float]:
        """Auto-detect FFD domain based on mesh bounds."""
        # Expand bounds by 20% for FFD domain
        x_range = mesh_bounds['x_max'] - mesh_bounds['x_min']
        y_range = mesh_bounds['y_max'] - mesh_bounds['y_min']
        z_range = mesh_bounds['z_max'] - mesh_bounds['z_min']
        
        return {
            'x_min': mesh_bounds['x_min'] - 0.1 * x_range,
            'x_max': mesh_bounds['x_max'] + 0.1 * x_range,
            'y_min': mesh_bounds['y_min'] - 0.1 * y_range,
            'y_max': mesh_bounds['y_max'] + 0.1 * y_range,
            'z_min': mesh_bounds['z_min'] - 0.1 * z_range,
            'z_max': mesh_bounds['z_max'] + 0.1 * z_range
        }
    
    def extract_objectives(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract objective values from CFD results."""
        objectives = {}
        
        for obj_config in self.config.objectives:
            if obj_config.name in results:
                objectives[obj_config.name] = results[obj_config.name]
            else:
                print(f"Warning: Objective '{obj_config.name}' not found in results")
                objectives[obj_config.name] = 0.0
        
        return objectives
    
    def get_boundary_patches(self) -> List[str]:
        """Get list of boundary patches."""
        if not self._validated:
            self.validate_case()
        
        return [patch['name'] for patch in self.boundary_patches]
    
    def prepare_mesh_for_optimization(self) -> bool:
        """Prepare mesh for optimization."""
        # Check if mesh needs conversion
        polyMesh_dir = self.case_path / 'constant' / 'polyMesh'
        
        if not polyMesh_dir.exists():
            print(f"Error: polyMesh directory not found: {polyMesh_dir}")
            return False
        
        # Check for required mesh files
        required_files = ['points', 'faces', 'owner', 'neighbour', 'boundary']
        for filename in required_files:
            if not (polyMesh_dir / filename).exists():
                print(f"Error: Required mesh file missing: {filename}")
                return False
        
        return True
    
    def get_physics_info(self) -> Dict[str, Any]:
        """Get physics information from case."""
        info = {
            'solver': self.solver_type,
            'physics': self.config.physics,
            'boundary_patches': self.get_boundary_patches(),
            'mesh_bounds': self._get_mesh_bounds()
        }
        
        return info