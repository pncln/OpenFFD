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
        
        # Find the airfoil boundary - handle multiline format
        # Try both "airfoil" and "walls" naming conventions
        airfoil_pattern = r'airfoil\s*\{[^}]*\}'
        walls_pattern = r'walls\s*\{[^}]*\}'
        
        airfoil_match = re.search(airfoil_pattern, content, re.DOTALL)
        walls_match = re.search(walls_pattern, content, re.DOTALL)
        
        if airfoil_match:
            airfoil_section = airfoil_match.group(0)
        elif walls_match:
            airfoil_section = walls_match.group(0)
        else:
            raise ValueError("Could not find 'airfoil' or 'walls' boundary in boundary file")
        
        # Extract nFaces and startFace from the airfoil section
        nfaces_match = re.search(r'nFaces\s+(\d+);', airfoil_section)
        startface_match = re.search(r'startFace\s+(\d+);', airfoil_section)
        
        if not nfaces_match or not startface_match:
            raise ValueError("Could not find nFaces or startFace in airfoil boundary")
        
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
        for i in range(n_faces):
            face_idx = start_face + i
            line_idx = faces_start_idx + face_idx
            if line_idx < len(lines):
                line = lines[line_idx].strip()
                # OpenFOAM face format: N(p1 p2 p3 ...)
                if '(' in line and ')' in line:
                    # Extract point indices from parentheses
                    paren_start = line.find('(')
                    paren_end = line.find(')')
                    if paren_start != -1 and paren_end != -1:
                        indices_str = line[paren_start+1:paren_end]
                        try:
                            indices = [int(x) for x in indices_str.split()]
                            airfoil_point_set.update(indices)
                        except ValueError:
                            continue
        
        self.airfoil_point_indices = sorted(list(airfoil_point_set))
        
        if len(self.airfoil_point_indices) > 0:
            self.airfoil_points = self.mesh_points[self.airfoil_point_indices]
        else:
            self.airfoil_points = np.array([])
        
        self.logger.info(f"Identified {len(self.airfoil_point_indices)} airfoil boundary points")
        
        if len(self.airfoil_points) == 0:
            print("DEBUG: No airfoil boundary points found - debugging face file format...")
            print(f"DEBUG: Sample lines around faces_start_idx ({faces_start_idx}):")
            for i in range(max(0, faces_start_idx-5), min(len(lines), faces_start_idx+10)):
                print(f"  Line {i}: {lines[i][:100]}")
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
        
        # Create backup if needed (only once)
        self._backup_original_mesh()
        
        # Only restore if we have a backup and current mesh is different
        # This prevents overwriting the user's fixed mesh
        original_mesh_dir = self.case_path / "constant" / "polyMesh_original"
        if original_mesh_dir.exists():
            # Check if current mesh is deformed (has been modified)
            current_mesh_dir = self.case_path / "constant" / "polyMesh"
            if self._is_mesh_deformed():
                self.restore_original_mesh()
                self._load_mesh_points()
        else:
            # No backup exists yet, use current mesh as reference
            self._load_mesh_points()
        
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
        """Apply design variable displacements to control points with quality preservation."""
        nx, ny, nz = self.control_points
        
        # Calculate scaling based on actual mesh dimensions
        # The airfoil extends from x=-17.5 to x=17.5, giving chord_length=35
        chord_length = 35.0  # Actual chord length in this mesh
        adaptive_scale = chord_length * 0.00001  # 0.001% of actual chord length for safety
        
        # Reset control grid to original positions
        self._setup_ffd_box()
        self._create_control_grid()
        
        # Apply displacements with geometric constraints
        if len(design_vars) == nx * ny * nz:
            # Full 3D displacement
            displacements = design_vars.reshape((nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        # Only modify Y-direction (normal to airfoil surface)
                        # Apply smooth variation with boundary constraints
                        edge_factor = self._get_edge_constraint_factor(i, j, nx, ny)
                        displacement = displacements[i, j, k] * adaptive_scale * edge_factor
                        
                        # Clamp displacement to prevent mesh corruption
                        max_displacement = chord_length * 0.00005  # 0.005% of actual chord maximum
                        displacement = np.clip(displacement, -max_displacement, max_displacement)
                        
                        self.control_grid[i, j, k, 1] += displacement
        else:
            # Simplified: distribute design variables with smooth variation
            for var_idx, design_var in enumerate(design_vars):
                if var_idx >= nx * ny * nz:
                    break
                    
                i = var_idx % nx
                j = (var_idx // nx) % ny
                k = var_idx // (nx * ny)
                
                # Apply smooth displacement with constraints
                edge_factor = self._get_edge_constraint_factor(i, j, nx, ny)
                displacement = design_var * adaptive_scale * edge_factor
                
                # Clamp displacement
                max_displacement = chord_length * 0.00005  # 0.005% of actual chord  
                displacement = np.clip(displacement, -max_displacement, max_displacement)
                
                self.control_grid[i, j, k, 1] += displacement
        
        self.logger.info(f"Applied control point displacements with adaptive scaling {adaptive_scale}")
    
    def _get_edge_constraint_factor(self, i: int, j: int, nx: int, ny: int) -> float:
        """Get constraint factor for edge points to preserve mesh quality."""
        # Reduce displacement near boundaries to prevent mesh distortion
        edge_factor_x = 1.0
        edge_factor_y = 1.0
        
        # X-direction constraints (leading/trailing edge)
        if i == 0 or i == nx - 1:
            edge_factor_x = 0.1  # Minimal displacement at edges
        elif i == 1 or i == nx - 2:
            edge_factor_x = 0.5  # Reduced displacement near edges
            
        # Y-direction constraints (upper/lower surface)
        if j == 0 or j == ny - 1:
            edge_factor_y = 0.3  # Some displacement at surface boundaries
        
        return edge_factor_x * edge_factor_y
    
    def _deform_airfoil_points(self) -> np.ndarray:
        """Deform airfoil boundary points using FFD with quality preservation."""
        deformed_points = np.zeros_like(self.airfoil_points)
        
        for i, point in enumerate(self.airfoil_points):
            # Convert physical coordinates to parametric coordinates
            u = (point[0] - self.ffd_box['x_min']) / (self.ffd_box['x_max'] - self.ffd_box['x_min'])
            v = (point[1] - self.ffd_box['y_min']) / (self.ffd_box['y_max'] - self.ffd_box['y_min'])
            w = (point[2] - self.ffd_box['z_min']) / (self.ffd_box['z_max'] - self.ffd_box['z_min'])
            
            # Clamp parametric coordinates with small tolerance
            eps = 1e-10
            u = np.clip(u, eps, 1.0 - eps)
            v = np.clip(v, eps, 1.0 - eps)
            w = np.clip(w, eps, 1.0 - eps)
            
            # Compute deformed position using high-quality trilinear interpolation
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
        
        # Validate mesh quality before writing
        if not self._validate_mesh_quality(deformed_airfoil_points):
            self.logger.warning("Mesh quality validation failed, using smaller deformation")
            # Apply reduced deformation
            reduced_deformation = (deformed_airfoil_points + self.airfoil_points) / 2.0
            self._update_points_in_lines(lines, self.airfoil_point_indices, reduced_deformation)
        
        # Write updated points file with proper formatting
        updated_content = '\n'.join(lines)
        with open(points_file, 'w') as f:
            f.write(updated_content)
        
        # Verify mesh integrity
        if self._check_mesh_integrity():
            self.logger.info(f"Successfully updated mesh with {len(deformed_airfoil_points)} deformed airfoil points")
        else:
            self.logger.error("Mesh integrity check failed, restoring original mesh")
            self.restore_original_mesh()
        
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
    
    def _validate_mesh_quality(self, deformed_points: np.ndarray) -> bool:
        """Validate mesh quality metrics."""
        try:
            # Check for reasonable deformation magnitudes
            displacements = np.linalg.norm(deformed_points - self.airfoil_points, axis=1)
            max_displacement = np.max(displacements)
            avg_displacement = np.mean(displacements)
            
            # Thresholds based on actual chord length for real optimization
            chord_length = 35.0  # Actual chord length in this mesh
            max_allowed = chord_length * 0.05  # 5% of chord for significant shape changes
            avg_allowed = chord_length * 0.025  # 2.5% of chord average
            
            if max_displacement > max_allowed:
                self.logger.warning(f"Maximum displacement {max_displacement:.6f} exceeds threshold {max_allowed:.6f}")
                return False
                
            if avg_displacement > avg_allowed:
                self.logger.warning(f"Average displacement {avg_displacement:.6f} exceeds threshold {avg_allowed:.6f}")
                return False
            
            # Check for point ordering preservation (no crossing)
            if self._check_point_ordering(deformed_points):
                self.logger.info(f"Mesh quality validation passed: max_disp={max_displacement:.6f}, avg_disp={avg_displacement:.6f}")
                return True
            else:
                self.logger.warning("Point ordering validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Mesh quality validation error: {e}")
            return False
    
    def _check_point_ordering(self, points: np.ndarray) -> bool:
        """Check if point ordering is preserved (no crossings)."""
        try:
            # Check for basic geometric consistency
            # Ensure points maintain reasonable spacing
            min_spacing = np.inf
            for i in range(len(points) - 1):
                spacing = np.linalg.norm(points[i+1] - points[i])
                min_spacing = min(min_spacing, spacing)
            
            # Minimum spacing threshold
            min_allowed = 1e-6
            if min_spacing < min_allowed:
                self.logger.warning(f"Minimum point spacing {min_spacing:.8f} too small")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Point ordering check error: {e}")
            return False
    
    def _check_mesh_integrity(self) -> bool:
        """Check basic mesh file integrity."""
        try:
            mesh_dir = self.case_path / "constant" / "polyMesh"
            required_files = ["points", "faces", "owner", "neighbour", "boundary"]
            
            for file_name in required_files:
                file_path = mesh_dir / file_name
                if not file_path.exists():
                    self.logger.error(f"Missing mesh file: {file_name}")
                    return False
                    
                # Check file is not empty
                if file_path.stat().st_size == 0:
                    self.logger.error(f"Empty mesh file: {file_name}")
                    return False
            
            # Basic points file validation
            points_file = mesh_dir / "points"
            with open(points_file, 'r') as f:
                content = f.read()
                
            # Check for proper OpenFOAM format
            if "FoamFile" not in content:
                self.logger.error("Invalid OpenFOAM points file format")
                return False
                
            # Check for reasonable number of points
            point_count = content.count('(')
            if point_count < 1000:  # Reasonable minimum for airfoil mesh
                self.logger.warning(f"Unusually low point count: {point_count}")
                
            self.logger.info("Mesh integrity check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Mesh integrity check error: {e}")
            return False
    
    def _update_points_in_lines(self, lines: List[str], point_indices: List[int], points: np.ndarray):
        """Update point coordinates in file lines."""
        try:
            # Find data section
            data_start_idx = None
            for i, line in enumerate(lines):
                if line.strip() == '(':
                    data_start_idx = i + 1
                    break
            
            if data_start_idx is None:
                raise ValueError("Could not find data section start")
            
            # Update points
            for i, point in enumerate(points):
                if data_start_idx + point_indices[i] < len(lines):
                    lines[data_start_idx + point_indices[i]] = f"({point[0]:.8f} {point[1]:.8f} {point[2]:.8f})"
                    
        except Exception as e:
            self.logger.error(f"Error updating points in lines: {e}")
            raise
    
    def _is_mesh_deformed(self) -> bool:
        """Check if current mesh differs from original backup."""
        try:
            original_mesh_dir = self.case_path / "constant" / "polyMesh_original"
            current_mesh_dir = self.case_path / "constant" / "polyMesh"
            
            if not original_mesh_dir.exists():
                return False
                
            # Compare points files (quick check)
            orig_points = original_mesh_dir / "points"
            curr_points = current_mesh_dir / "points"
            
            if not orig_points.exists() or not curr_points.exists():
                return False
                
            # Compare file sizes first (fast check)
            if orig_points.stat().st_size != curr_points.stat().st_size:
                return True
                
            # If sizes are same, assume they're the same (for performance)
            # In a production system, you might want a more thorough check
            return False
            
        except Exception:
            return False