"""
FFD-based Mesh Deformation for CFD Shape Optimization

This module implements Free Form Deformation (FFD) for OpenFOAM mesh modification.
It allows parametric control of airfoil shape through control point displacements.
"""

import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import shutil

logger = logging.getLogger(__name__)

class FFDDeformation:
    """Free Form Deformation for airfoil shape optimization."""
    
    def __init__(self, case_path: Path, control_points: List[int] = [2, 2, 1]):
        """Initialize FFD deformation.
        
        Args:
            case_path: Path to OpenFOAM case
            control_points: Number of control points in [x, y, z] directions
        """
        self.case_path = Path(case_path)
        self.control_points = control_points
        self.logger = logging.getLogger(__name__)
        
        # FFD domain bounds (will be auto-detected from airfoil)
        self.ffd_box = None
        self.airfoil_points = None
        self.airfoil_point_indices = None
        
        # Control point grid
        self.control_grid = None
        self.original_control_grid = None
        
        self._initialize_ffd()
    
    def _initialize_ffd(self):
        """Initialize FFD system by detecting airfoil and setting up control points."""
        self.logger.info("Initializing FFD system...")
        
        # Load mesh and identify airfoil boundary points
        self._load_mesh_points()
        self._identify_airfoil_boundary()
        self._setup_ffd_box()
        self._create_control_grid()
        
        self.logger.info(f"FFD initialized with {np.prod(self.control_points)} control points")
        self.logger.info(f"FFD box: {self.ffd_box}")
    
    def _load_mesh_points(self):
        """Load mesh points from OpenFOAM points file."""
        points_file = self.case_path / "constant" / "polyMesh" / "points"
        
        if not points_file.exists():
            raise FileNotFoundError(f"Points file not found: {points_file}")
        
        # Parse OpenFOAM points file
        with open(points_file, 'r') as f:
            content = f.read()
        
        # Extract points using regex
        # Skip header until we find the number of points
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
        self.logger.info(f"Loading {n_points} mesh points")
        
        # Extract coordinate data
        points = []
        for i in range(start_idx, start_idx + n_points):
            if i >= len(lines):
                break
            line = lines[i].strip()
            if line.startswith('(') and line.endswith(')'):
                # Extract coordinates
                coords_str = line[1:-1]  # Remove parentheses
                coords = [float(x) for x in coords_str.split()]
                if len(coords) >= 3:
                    points.append(coords[:3])
        
        self.mesh_points = np.array(points)
        self.logger.info(f"Loaded {len(self.mesh_points)} mesh points")
        
        if len(self.mesh_points) == 0:
            raise ValueError("No valid points found in mesh file")
    
    def _identify_airfoil_boundary(self):
        """Identify which mesh points belong to the airfoil boundary."""
        boundary_file = self.case_path / "constant" / "polyMesh" / "boundary"
        
        # Parse boundary file to find airfoil patch info
        with open(boundary_file, 'r') as f:
            content = f.read()
        
        # Find the walls (airfoil) boundary - handle multiline format
        walls_pattern = r'walls\s*\{[^}]*\}'
        walls_match = re.search(walls_pattern, content, re.DOTALL)
        
        if not walls_match:
            raise ValueError("Could not find 'walls' boundary in boundary file")
        
        walls_section = walls_match.group(0)
        
        # Extract nFaces and startFace from the walls section
        nfaces_match = re.search(r'nFaces\s+(\d+);', walls_section)
        startface_match = re.search(r'startFace\s+(\d+);', walls_section)
        
        if not nfaces_match or not startface_match:
            raise ValueError("Could not find nFaces or startFace in walls boundary")
        
        n_faces = int(nfaces_match.group(1))
        start_face = int(startface_match.group(1))
        
        self.logger.info(f"Airfoil boundary: {n_faces} faces starting at face {start_face}")
        
        # Load faces to get point indices
        faces_file = self.case_path / "constant" / "polyMesh" / "faces"
        with open(faces_file, 'r') as f:
            faces_content = f.read()
        
        # Extract face data for airfoil
        lines = faces_content.split('\n')
        face_count_line = None
        faces_start_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.isdigit():
                face_count_line = i
                faces_start_idx = i + 2
                break
        
        # Get airfoil face point indices
        airfoil_point_set = set()
        for face_idx in range(start_face, start_face + n_faces):
            line_idx = faces_start_idx + face_idx
            if line_idx < len(lines):
                line = lines[line_idx].strip()
                if line.startswith('(') and line.endswith(')'):
                    # Extract point indices
                    indices_str = line[1:-1]
                    try:
                        indices = [int(x) for x in indices_str.split()]
                        airfoil_point_set.update(indices)
                    except ValueError:
                        continue
        
        self.airfoil_point_indices = sorted(list(airfoil_point_set))
        self.airfoil_points = self.mesh_points[self.airfoil_point_indices]
        
        self.logger.info(f"Identified {len(self.airfoil_point_indices)} airfoil boundary points")
        
        if len(self.airfoil_points) == 0:
            raise ValueError("No airfoil boundary points found")
        
        # Log airfoil extent for verification
        x_min, x_max = np.min(self.airfoil_points[:, 0]), np.max(self.airfoil_points[:, 0])
        y_min, y_max = np.min(self.airfoil_points[:, 1]), np.max(self.airfoil_points[:, 1])
        self.logger.info(f"Airfoil extent: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
    
    def _setup_ffd_box(self):
        """Setup FFD bounding box around the airfoil."""
        # Get airfoil bounding box
        x_min, x_max = np.min(self.airfoil_points[:, 0]), np.max(self.airfoil_points[:, 0])
        y_min, y_max = np.min(self.airfoil_points[:, 1]), np.max(self.airfoil_points[:, 1])
        z_min, z_max = np.min(self.airfoil_points[:, 2]), np.max(self.airfoil_points[:, 2])
        
        # Expand box slightly beyond airfoil
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.5
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
        
        # Create parametric coordinates
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
        """Apply design variable displacements to control points and deform mesh.
        
        Args:
            design_vars: Array of control point displacements
            
        Returns:
            Path to modified mesh directory
        """
        self.logger.info(f"Applying FFD deformation with {len(design_vars)} design variables")
        self.logger.info(f"Design variable values: {design_vars}")
        
        # Create backup if needed
        self._backup_original_mesh()
        
        # Reset control grid to original positions
        self.control_grid = self.original_control_grid.copy()
        
        # Apply design variable displacements to control points
        self._apply_control_point_displacements(design_vars)
        
        # Deform airfoil boundary points
        deformed_airfoil_points = self._deform_airfoil_points()
        
        # Update mesh with deformed airfoil
        mesh_path = self._update_mesh_points(deformed_airfoil_points)
        
        return mesh_path
    
    def _backup_original_mesh(self):
        """Create backup of original mesh if it doesn't exist."""
        original_mesh_dir = self.case_path / "constant" / "polyMesh_original"
        current_mesh_dir = self.case_path / "constant" / "polyMesh"
        
        if not original_mesh_dir.exists():
            shutil.copytree(current_mesh_dir, original_mesh_dir)
            self.logger.info("Created backup of original mesh")
    
    def _apply_control_point_displacements(self, design_vars: np.ndarray):
        """Apply design variable displacements to control points."""
        nx, ny, nz = self.control_points
        
        # Reshape design variables to control point grid
        # For simplicity, apply Y-displacements only (normal to airfoil)
        if len(design_vars) == nx * ny * nz:
            # Full 3D displacement
            displacements = design_vars.reshape((nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        # Apply displacement in Y direction (normal to airfoil)
                        self.control_grid[i, j, k, 1] += displacements[i, j, k] * 0.1  # Scale displacement
        else:
            # Simplified: distribute design variables evenly
            base_displacement = np.mean(design_vars) if len(design_vars) > 0 else 0.0
            
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        # Apply larger displacement to middle control points (airfoil center)
                        weight = 1.0 if j > 0 else 0.5  # Reduce displacement at boundaries
                        displacement = base_displacement * weight * 0.05  # Scale factor
                        self.control_grid[i, j, k, 1] += displacement
        
        self.logger.info("Applied control point displacements")
    
    def _deform_airfoil_points(self) -> np.ndarray:
        """Deform airfoil boundary points using FFD."""
        deformed_points = np.zeros_like(self.airfoil_points)
        
        for i, point in enumerate(self.airfoil_points):
            # Convert physical coordinates to parametric coordinates
            u = (point[0] - self.ffd_box['x_min']) / (self.ffd_box['x_max'] - self.ffd_box['x_min'])
            v = (point[1] - self.ffd_box['y_min']) / (self.ffd_box['y_max'] - self.ffd_box['y_min'])
            w = (point[2] - self.ffd_box['z_min']) / (self.ffd_box['z_max'] - self.ffd_box['z_min'])
            
            # Clamp parametric coordinates
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
            w = np.clip(w, 0, 1)
            
            # Compute deformed position using trilinear interpolation
            deformed_points[i] = self._trilinear_interpolation(u, v, w)
        
        # Calculate deformation magnitude for logging
        displacement = np.linalg.norm(deformed_points - self.airfoil_points, axis=1)
        max_disp = np.max(displacement)
        avg_disp = np.mean(displacement)
        
        self.logger.info(f"Airfoil deformation: max={max_disp:.6f}, avg={avg_disp:.6f}")
        
        return deformed_points
    
    def _trilinear_interpolation(self, u: float, v: float, w: float) -> np.ndarray:
        """Perform trilinear interpolation in the control grid."""
        nx, ny, nz = self.control_points
        
        # Find surrounding control points
        i = min(int(u * (nx - 1)), nx - 2)
        j = min(int(v * (ny - 1)), ny - 2)
        k = min(int(w * (nz - 1)), nz - 2)
        
        # Local parametric coordinates within the cell
        u_local = (u * (nx - 1)) - i
        v_local = (v * (ny - 1)) - j
        w_local = (w * (nz - 1)) - k
        
        # Get 8 surrounding control points
        c000 = self.control_grid[i, j, k]
        c001 = self.control_grid[i, j, k+1] if k+1 < nz else c000
        c010 = self.control_grid[i, j+1, k] if j+1 < ny else c000
        c011 = self.control_grid[i, j+1, k+1] if j+1 < ny and k+1 < nz else c000
        c100 = self.control_grid[i+1, j, k] if i+1 < nx else c000
        c101 = self.control_grid[i+1, j, k+1] if i+1 < nx and k+1 < nz else c000
        c110 = self.control_grid[i+1, j+1, k] if i+1 < nx and j+1 < ny else c000
        c111 = self.control_grid[i+1, j+1, k+1] if i+1 < nx and j+1 < ny and k+1 < nz else c000
        
        # Trilinear interpolation
        c00 = c000 * (1 - w_local) + c001 * w_local
        c01 = c010 * (1 - w_local) + c011 * w_local
        c10 = c100 * (1 - w_local) + c101 * w_local
        c11 = c110 * (1 - w_local) + c111 * w_local
        
        c0 = c00 * (1 - v_local) + c01 * v_local
        c1 = c10 * (1 - v_local) + c11 * v_local
        
        result = c0 * (1 - u_local) + c1 * u_local
        
        return result
    
    def _update_mesh_points(self, deformed_airfoil_points: np.ndarray) -> Path:
        """Update mesh points file with deformed airfoil."""
        points_file = self.case_path / "constant" / "polyMesh" / "points"
        
        # Update mesh points array
        updated_mesh_points = self.mesh_points.copy()
        updated_mesh_points[self.airfoil_point_indices] = deformed_airfoil_points
        
        # Read original points file to preserve formatting
        with open(points_file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Find where point data starts
        point_count_line = None
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.isdigit():
                point_count_line = i
                data_start_idx = i + 2  # Skip count and '(' lines
                break
        
        if point_count_line is None:
            raise ValueError("Could not find point count in points file")
        
        # Update point coordinates in file content
        for i, point in enumerate(updated_mesh_points):
            if data_start_idx + i < len(lines):
                lines[data_start_idx + i] = f"({point[0]:.6f} {point[1]:.6f} {point[2]:.6f})"
        
        # Write updated points file
        updated_content = '\n'.join(lines)
        with open(points_file, 'w') as f:
            f.write(updated_content)
        
        self.logger.info(f"Updated mesh points file with {len(deformed_airfoil_points)} deformed airfoil points")
        
        return self.case_path / "constant" / "polyMesh"
    
    def restore_original_mesh(self) -> bool:
        """Restore original mesh from backup."""
        try:
            original_mesh_dir = self.case_path / "constant" / "polyMesh_original"
            current_mesh_dir = self.case_path / "constant" / "polyMesh"
            
            if original_mesh_dir.exists():
                # Remove current mesh
                if current_mesh_dir.exists():
                    shutil.rmtree(current_mesh_dir)
                
                # Restore original
                shutil.copytree(original_mesh_dir, current_mesh_dir)
                self.logger.info("Restored original mesh")
                
                # Reload mesh data
                self._load_mesh_points()
                return True
            else:
                self.logger.warning("No original mesh backup found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to restore original mesh: {e}")
            return False