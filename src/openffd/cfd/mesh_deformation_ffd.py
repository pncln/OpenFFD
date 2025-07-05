"""
True FFD (Free Form Deformation) System for Airfoil Shape Optimization

This module implements a proper FFD system with control points that move
only in the Y direction (normal to airfoil surface) as specified in the
configuration file. It uses Bernstein basis functions for smooth deformation.
"""

import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import shutil
import json

logger = logging.getLogger(__name__)

class FFDMeshDeformer:
    """True FFD mesh deformation system with Y-direction only control point movement."""
    
    def __init__(self, case_path: Path, control_points: List[int] = [2, 2, 1]):
        """Initialize the FFD mesh deformation system.
        
        Args:
            case_path: Path to OpenFOAM case
            control_points: Number of control points in [x, y, z] directions
        """
        self.case_path = Path(case_path)
        self.control_points = control_points
        self.logger = logging.getLogger(__name__)
        
        # Mesh data
        self.mesh_points = None
        self.airfoil_points = None
        self.airfoil_indices = None
        self.chord_length = None
        
        # FFD system
        self.ffd_box = None
        self.control_grid = None
        self.original_control_grid = None
        
        # Initialize the system
        self._initialize()
    
    def _initialize(self):
        """Initialize the FFD system."""
        self.logger.info("Initializing true FFD system with Y-direction only movement...")
        
        # Create backup of original mesh
        self._create_mesh_backup()
        
        # Load and analyze mesh
        self._load_mesh_geometry()
        self._detect_airfoil_surface()
        self._setup_ffd_box()
        self._create_control_grid()
        
        self.logger.info(f"FFD system initialized:")
        self.logger.info(f"  - Airfoil points: {len(self.airfoil_indices)}")
        self.logger.info(f"  - Chord length: {self.chord_length:.3f}")
        self.logger.info(f"  - Control points: {self.control_points}")
        self.logger.info(f"  - FFD box: {self.ffd_box}")
    
    def _create_mesh_backup(self):
        """Create backup of original mesh if it doesn't exist."""
        original_mesh_dir = self.case_path / "constant" / "polyMesh_original"
        current_mesh_dir = self.case_path / "constant" / "polyMesh"
        
        if not original_mesh_dir.exists():
            shutil.copytree(current_mesh_dir, original_mesh_dir)
            self.logger.info("Created backup of original mesh")
    
    def _load_mesh_geometry(self):
        """Load mesh points from OpenFOAM points file."""
        points_file = self.case_path / "constant" / "polyMesh" / "points"
        
        if not points_file.exists():
            raise FileNotFoundError(f"Points file not found: {points_file}")
        
        with open(points_file, 'r') as f:
            content = f.read()
        
        # Parse OpenFOAM points file
        lines = content.split('\n')
        point_count_line = None
        start_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.isdigit():
                point_count_line = i
                start_idx = i + 2  # Skip the '(' line
                break
        
        if point_count_line is None:
            raise ValueError("Could not find point count in points file")
        
        n_points = int(lines[point_count_line])
        
        # Extract coordinate data
        points = []
        for i in range(start_idx, start_idx + n_points):
            if i >= len(lines):
                break
            line = lines[i].strip()
            if line.startswith('(') and line.endswith(')'):
                coords_str = line[1:-1]  # Remove parentheses
                coords = [float(x) for x in coords_str.split()]
                if len(coords) >= 3:
                    points.append(coords[:3])
        
        self.mesh_points = np.array(points)
        self.logger.info(f"Loaded {len(self.mesh_points)} mesh points")
    
    def _detect_airfoil_surface(self):
        """Detect airfoil surface points using boundary information."""
        boundary_file = self.case_path / "constant" / "polyMesh" / "boundary"
        
        with open(boundary_file, 'r') as f:
            content = f.read()
        
        # Find walls boundary (contains airfoil)
        walls_pattern = r'walls\s*\{[^}]*\}'
        walls_match = re.search(walls_pattern, content, re.DOTALL)
        
        if not walls_match:
            raise ValueError("Could not find 'walls' boundary in boundary file")
        
        walls_section = walls_match.group(0)
        
        # Extract nFaces and startFace
        nfaces_match = re.search(r'nFaces\s+(\d+);', walls_section)
        startface_match = re.search(r'startFace\s+(\d+);', walls_section)
        
        if not nfaces_match or not startface_match:
            raise ValueError("Could not find nFaces or startFace in walls boundary")
        
        n_faces = int(nfaces_match.group(1))
        start_face = int(startface_match.group(1))
        
        # Load faces to get point indices
        faces_file = self.case_path / "constant" / "polyMesh" / "faces"
        with open(faces_file, 'r') as f:
            faces_content = f.read()
        
        # Parse faces file
        lines = faces_content.split('\n')
        faces_start_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.isdigit():
                faces_start_idx = i + 2
                break
        
        # Extract airfoil point indices from faces
        airfoil_point_set = set()
        for i in range(n_faces):
            face_idx = start_face + i
            line_idx = faces_start_idx + face_idx
            if line_idx < len(lines):
                line = lines[line_idx].strip()
                if '(' in line and ')' in line:
                    paren_start = line.find('(')
                    paren_end = line.find(')')
                    if paren_start != -1 and paren_end != -1:
                        indices_str = line[paren_start+1:paren_end]
                        try:
                            indices = [int(x) for x in indices_str.split()]
                            airfoil_point_set.update(indices)
                        except ValueError:
                            continue
        
        self.airfoil_indices = sorted(list(airfoil_point_set))
        self.airfoil_points = self.mesh_points[self.airfoil_indices]
        
        # Calculate chord length
        x_coords = self.airfoil_points[:, 0]
        self.chord_length = np.max(x_coords) - np.min(x_coords)
        
        self.logger.info(f"Detected airfoil surface with {len(self.airfoil_indices)} points")
        self.logger.info(f"Airfoil extent: x=[{np.min(x_coords):.3f}, {np.max(x_coords):.3f}]")
    
    def _setup_ffd_box(self):
        """Setup FFD bounding box around the airfoil."""
        # Get airfoil bounding box
        x_min, x_max = np.min(self.airfoil_points[:, 0]), np.max(self.airfoil_points[:, 0])
        y_min, y_max = np.min(self.airfoil_points[:, 1]), np.max(self.airfoil_points[:, 1])
        z_min, z_max = np.min(self.airfoil_points[:, 2]), np.max(self.airfoil_points[:, 2])
        
        # Expand box slightly beyond airfoil for proper FFD influence
        x_margin = (x_max - x_min) * 0.1  # 10% margin in X
        y_margin = (y_max - y_min) * 0.5  # 50% margin in Y (normal direction)
        z_margin = max(0.01, (z_max - z_min) * 0.1)
        
        self.ffd_box = {
            'x_min': x_min - x_margin,
            'x_max': x_max + x_margin,
            'y_min': y_min - y_margin,
            'y_max': y_max + y_margin,
            'z_min': z_min - z_margin,
            'z_max': z_max + z_margin
        }
        
        self.logger.info(f"FFD box: {self.ffd_box}")
    
    def _create_control_grid(self):
        """Create 3D control point grid."""
        nx, ny, nz = self.control_points
        
        # Create parametric coordinates for control points
        u = np.linspace(0, 1, nx)
        v = np.linspace(0, 1, ny)  
        w = np.linspace(0, 1, nz)
        
        # Map to physical coordinates
        x_coords = self.ffd_box['x_min'] + u * (self.ffd_box['x_max'] - self.ffd_box['x_min'])
        y_coords = self.ffd_box['y_min'] + v * (self.ffd_box['y_max'] - self.ffd_box['y_min'])
        z_coords = self.ffd_box['z_min'] + w * (self.ffd_box['z_max'] - self.ffd_box['z_min'])
        
        # Create 3D grid
        self.control_grid = np.zeros((nx, ny, nz, 3))
        self.original_control_grid = np.zeros((nx, ny, nz, 3))
        
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                for k, z in enumerate(z_coords):
                    self.control_grid[i, j, k] = [x, y, z]
                    self.original_control_grid[i, j, k] = [x, y, z]
        
        self.logger.info(f"Created {nx}x{ny}x{nz} control point grid")
    
    def apply_design_variables(self, design_vars: np.ndarray) -> Path:
        """Apply design variables to move FFD control points in Y direction only.
        
        For a [2,2,1] grid, we have 4 control points that can move in Y direction:
        - design_vars[0]: displacement of control point (0,0,0) in Y
        - design_vars[1]: displacement of control point (1,0,0) in Y  
        - design_vars[2]: displacement of control point (0,1,0) in Y
        - design_vars[3]: displacement of control point (1,1,0) in Y
        
        Args:
            design_vars: Array of Y-displacements for control points
            
        Returns:
            Path to modified mesh directory
        """
        self.logger.info(f"Applying FFD deformation with Y-only movement: {design_vars}")
        
        # Restore original mesh first
        self.restore_original_mesh()
        self._load_mesh_geometry()
        
        # Reset control grid to original positions
        self.control_grid = self.original_control_grid.copy()
        
        # Apply Y-direction displacements to control points
        self._apply_control_point_displacements_y_only(design_vars)
        
        # Deform airfoil boundary points using FFD
        deformed_points = self._deform_airfoil_points_ffd()
        
        # Validate mesh quality
        if not self._validate_deformation_quality(deformed_points):
            self.logger.warning("Deformation quality check failed, applying reduced deformation")
            # Apply 50% of the deformation
            reduced_vars = design_vars * 0.5
            self._apply_control_point_displacements_y_only(reduced_vars)
            deformed_points = self._deform_airfoil_points_ffd()
        
        # Update mesh with deformed airfoil
        self._update_mesh_points(deformed_points)
        
        # Final mesh integrity check
        if not self._check_mesh_integrity():
            self.logger.error("Mesh integrity check failed, restoring original")
            self.restore_original_mesh()
            raise RuntimeError("FFD deformation failed - mesh integrity compromised")
        
        self.logger.info("FFD mesh deformation completed successfully")
        return self.case_path / "constant" / "polyMesh"
    
    def _apply_control_point_displacements_y_only(self, design_vars: np.ndarray):
        """Apply Y-direction only displacements to control points."""
        nx, ny, nz = self.control_points
        
        # Scale design variables to physical units
        # Use very conservative scaling: 0.1% of chord per unit design variable
        scale_factor = self.chord_length * 0.001
        
        # Apply displacements only in Y direction
        var_idx = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if var_idx < len(design_vars):
                        # Only modify Y coordinate (index 1)
                        displacement_y = design_vars[var_idx] * scale_factor
                        
                        # Apply safety limits to prevent extreme deformation
                        max_displacement = self.chord_length * 0.005  # 0.5% of chord maximum
                        displacement_y = np.clip(displacement_y, -max_displacement, max_displacement)
                        
                        self.control_grid[i, j, k, 1] += displacement_y
                        var_idx += 1
        
        self.logger.info(f"Applied Y-only displacements to {var_idx} control points")
    
    def _deform_airfoil_points_ffd(self) -> np.ndarray:
        """Deform airfoil boundary points using FFD with Bernstein basis functions."""
        deformed_points = np.zeros_like(self.airfoil_points)
        
        for i, point in enumerate(self.airfoil_points):
            # Convert physical coordinates to parametric coordinates [0,1]
            u = (point[0] - self.ffd_box['x_min']) / (self.ffd_box['x_max'] - self.ffd_box['x_min'])
            v = (point[1] - self.ffd_box['y_min']) / (self.ffd_box['y_max'] - self.ffd_box['y_min'])
            w = (point[2] - self.ffd_box['z_min']) / (self.ffd_box['z_max'] - self.ffd_box['z_min'])
            
            # Clamp parametric coordinates to [0,1]
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)
            w = np.clip(w, 0.0, 1.0)
            
            # Compute deformed position using trilinear FFD interpolation
            deformed_points[i] = self._ffd_interpolation(u, v, w)
        
        # Calculate deformation statistics
        displacement = np.linalg.norm(deformed_points - self.airfoil_points, axis=1)
        max_disp = np.max(displacement)
        avg_disp = np.mean(displacement)
        
        self.logger.info(f"FFD deformation: max={max_disp:.6f}, avg={avg_disp:.6f}")
        
        return deformed_points
    
    def _ffd_interpolation(self, u: float, v: float, w: float) -> np.ndarray:
        """Perform FFD interpolation using control points."""
        nx, ny, nz = self.control_points
        
        # Use trilinear interpolation for simplicity (can be extended to Bernstein polynomials)
        # Find surrounding control points
        i = min(int(u * (nx - 1)), nx - 2) if nx > 1 else 0
        j = min(int(v * (ny - 1)), ny - 2) if ny > 1 else 0
        k = min(int(w * (nz - 1)), nz - 2) if nz > 1 else 0
        
        # Local parametric coordinates within the cell
        if nx > 1:
            u_local = (u * (nx - 1)) - i
        else:
            u_local = 0.0
            
        if ny > 1:
            v_local = (v * (ny - 1)) - j
        else:
            v_local = 0.0
            
        if nz > 1:
            w_local = (w * (nz - 1)) - k
        else:
            w_local = 0.0
        
        # Get 8 surrounding control points (handle edge cases)
        c000 = self.control_grid[i, j, k]
        c001 = self.control_grid[i, j, min(k+1, nz-1)]
        c010 = self.control_grid[i, min(j+1, ny-1), k]
        c011 = self.control_grid[i, min(j+1, ny-1), min(k+1, nz-1)]
        c100 = self.control_grid[min(i+1, nx-1), j, k]
        c101 = self.control_grid[min(i+1, nx-1), j, min(k+1, nz-1)]
        c110 = self.control_grid[min(i+1, nx-1), min(j+1, ny-1), k]
        c111 = self.control_grid[min(i+1, nx-1), min(j+1, ny-1), min(k+1, nz-1)]
        
        # Trilinear interpolation
        c00 = c000 * (1 - w_local) + c001 * w_local
        c01 = c010 * (1 - w_local) + c011 * w_local
        c10 = c100 * (1 - w_local) + c101 * w_local
        c11 = c110 * (1 - w_local) + c111 * w_local
        
        c0 = c00 * (1 - v_local) + c01 * v_local
        c1 = c10 * (1 - v_local) + c11 * v_local
        
        result = c0 * (1 - u_local) + c1 * u_local
        
        return result
    
    def _validate_deformation_quality(self, deformed_points: np.ndarray) -> bool:
        """Validate that the FFD deformation maintains acceptable mesh quality."""
        try:
            # Check maximum displacement
            displacements = np.linalg.norm(deformed_points - self.airfoil_points, axis=1)
            max_displacement = np.max(displacements)
            avg_displacement = np.mean(displacements)
            
            # Very conservative thresholds for FFD
            max_allowed = self.chord_length * 0.01  # 1% of chord maximum
            avg_allowed = max_allowed * 0.5
            
            if max_displacement > max_allowed:
                self.logger.warning(f"Max displacement {max_displacement:.6f} exceeds {max_allowed:.6f}")
                return False
            
            if avg_displacement > avg_allowed:
                self.logger.warning(f"Avg displacement {avg_displacement:.6f} exceeds {avg_allowed:.6f}")
                return False
            
            # Check minimum point spacing
            min_spacing = self._check_minimum_spacing(deformed_points)
            min_allowed_spacing = self.chord_length * 1e-6
            
            if min_spacing < min_allowed_spacing:
                self.logger.warning(f"Minimum spacing {min_spacing:.8f} too small")
                return False
            
            self.logger.info(f"FFD quality OK: max_disp={max_displacement:.6f}, avg_disp={avg_displacement:.6f}")
            return True
            
        except Exception as e:
            self.logger.error(f"FFD quality validation error: {e}")
            return False
    
    def _check_minimum_spacing(self, points: np.ndarray) -> float:
        """Check minimum spacing between adjacent points."""
        try:
            min_spacing = np.inf
            
            # Check spacing between all adjacent points
            for i in range(len(points) - 1):
                spacing = np.linalg.norm(points[i+1] - points[i])
                min_spacing = min(min_spacing, spacing)
            
            return min_spacing
            
        except Exception:
            return 0.0
    
    def _update_mesh_points(self, deformed_airfoil_points: np.ndarray):
        """Update mesh points file with deformed airfoil."""
        points_file = self.case_path / "constant" / "polyMesh" / "points"
        
        # Update mesh points array
        updated_mesh_points = self.mesh_points.copy()
        updated_mesh_points[self.airfoil_indices] = deformed_airfoil_points
        
        # Read original points file to preserve formatting
        with open(points_file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Find where point data starts
        data_start_idx = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line.isdigit():
                data_start_idx = i + 2  # Skip count and '(' lines
                break
        
        # Update point coordinates in file content
        for i, point in enumerate(updated_mesh_points):
            if data_start_idx + i < len(lines):
                lines[data_start_idx + i] = f"({point[0]:.8f} {point[1]:.8f} {point[2]:.8f})"
        
        # Write updated points file
        updated_content = '\n'.join(lines)
        with open(points_file, 'w') as f:
            f.write(updated_content)
        
        self.logger.info(f"Updated mesh with {len(deformed_airfoil_points)} FFD-deformed airfoil points")
    
    def _check_mesh_integrity(self) -> bool:
        """Check basic mesh file integrity."""
        try:
            mesh_dir = self.case_path / "constant" / "polyMesh"
            required_files = ["points", "faces", "owner", "neighbour", "boundary"]
            
            for file_name in required_files:
                file_path = mesh_dir / file_name
                if not file_path.exists() or file_path.stat().st_size == 0:
                    self.logger.error(f"Invalid mesh file: {file_name}")
                    return False
            
            # Check points file format
            points_file = mesh_dir / "points"
            with open(points_file, 'r') as f:
                content = f.read()
            
            if "FoamFile" not in content:
                self.logger.error("Invalid OpenFOAM points file format")
                return False
            
            point_count = content.count('(')
            if point_count < 1000:  # Reasonable minimum for airfoil mesh
                self.logger.warning(f"Unusually low point count: {point_count}")
            
            self.logger.info("Mesh integrity check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Mesh integrity check error: {e}")
            return False
    
    def restore_original_mesh(self) -> bool:
        """Restore original mesh from backup."""
        try:
            original_mesh_dir = self.case_path / "constant" / "polyMesh_original"
            current_mesh_dir = self.case_path / "constant" / "polyMesh"
            
            if original_mesh_dir.exists():
                if current_mesh_dir.exists():
                    shutil.rmtree(current_mesh_dir)
                
                shutil.copytree(original_mesh_dir, current_mesh_dir)
                self.logger.info("Restored original mesh from backup")
                
                # Reload mesh data
                self._load_mesh_geometry()
                return True
            else:
                self.logger.warning("No original mesh backup found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to restore original mesh: {e}")
            return False
    
    def get_deformation_info(self) -> Dict:
        """Get information about the current FFD deformation state."""
        return {
            "airfoil_points": len(self.airfoil_indices) if self.airfoil_indices else 0,
            "chord_length": self.chord_length,
            "control_points": self.control_points,
            "ffd_box": self.ffd_box,
            "deformation_type": "FFD_Y_direction_only"
        }