"""
Hybrid FFD-Style Mesh Deformation System

This system implements FFD-style control with Y-direction only movement,
but uses a robust parametric approach to avoid mesh corruption while
still respecting the FFD control point configuration from the config file.
"""

import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import shutil
import json

logger = logging.getLogger(__name__)

class HybridFFDDeformer:
    """Hybrid FFD system that moves conceptual control points in Y-direction only."""
    
    def __init__(self, case_path: Path, control_points: List[int] = [2, 2, 1]):
        """Initialize the hybrid FFD deformation system.
        
        Args:
            case_path: Path to OpenFOAM case
            control_points: Number of conceptual control points in [x, y, z] directions
        """
        self.case_path = Path(case_path)
        self.control_points = control_points
        self.logger = logging.getLogger(__name__)
        
        # Calculate number of design variables from control points
        self.n_design_vars = np.prod(control_points)
        
        # Mesh data
        self.mesh_points = None
        self.airfoil_points = None
        self.airfoil_indices = None
        self.chord_length = None
        
        # Airfoil parameterization for robust deformation
        self.upper_surface_indices = None
        self.lower_surface_indices = None
        self.leading_edge_idx = None
        self.trailing_edge_idx = None
        
        # Initialize the system
        self._initialize()
    
    def _initialize(self):
        """Initialize the hybrid FFD system."""
        self.logger.info(f"Initializing hybrid FFD system with {self.control_points} control points (Y-direction only)...")
        
        # Create backup of original mesh
        self._create_mesh_backup()
        
        # Load and analyze mesh
        self._load_mesh_geometry()
        self._detect_airfoil_surface()
        self._parameterize_airfoil()
        
        self.logger.info(f"Hybrid FFD system initialized:")
        self.logger.info(f"  - Airfoil points: {len(self.airfoil_indices)}")
        self.logger.info(f"  - Chord length: {self.chord_length:.3f}")
        self.logger.info(f"  - Design variables: {self.n_design_vars} (from {self.control_points} control points)")
        self.logger.info(f"  - Deformation: Y-direction only, FFD-style control")
    
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
    
    def _parameterize_airfoil(self):
        """Parameterize airfoil surface for robust shape modifications."""
        if len(self.airfoil_points) == 0:
            raise ValueError("No airfoil points found for parameterization")
        
        # Find leading and trailing edges
        x_coords = self.airfoil_points[:, 0]
        y_coords = self.airfoil_points[:, 1]
        
        # Leading edge is the rightmost point (maximum x)
        le_idx = np.argmax(x_coords)
        self.leading_edge_idx = le_idx
        
        # Trailing edge is the leftmost point (minimum x)
        te_idx = np.argmin(x_coords)
        self.trailing_edge_idx = te_idx
        
        # Separate upper and lower surfaces
        le_point = self.airfoil_points[le_idx]
        
        # Calculate relative positions from leading edge
        dx = self.airfoil_points[:, 0] - le_point[0]
        dy = self.airfoil_points[:, 1] - le_point[1]
        
        # Upper surface: positive y direction from LE
        # Lower surface: negative y direction from LE
        upper_mask = dy >= 0
        lower_mask = dy < 0
        
        # Get indices and sort by x-coordinate
        upper_indices = [i for i, mask in enumerate(upper_mask) if mask]
        lower_indices = [i for i, mask in enumerate(lower_mask) if mask]
        
        # Sort upper surface from LE to TE (decreasing x)
        upper_x = [self.airfoil_points[i, 0] for i in upper_indices]
        upper_sorted = sorted(zip(upper_indices, upper_x), key=lambda x: -x[1])
        self.upper_surface_indices = [idx for idx, _ in upper_sorted]
        
        # Sort lower surface from LE to TE (decreasing x)
        lower_x = [self.airfoil_points[i, 0] for i in lower_indices]
        lower_sorted = sorted(zip(lower_indices, lower_x), key=lambda x: -x[1])
        self.lower_surface_indices = [idx for idx, _ in lower_sorted]
        
        self.logger.info(f"Airfoil parameterization complete:")
        self.logger.info(f"  - Upper surface: {len(self.upper_surface_indices)} points")
        self.logger.info(f"  - Lower surface: {len(self.lower_surface_indices)} points")
    
    def apply_design_variables(self, design_vars: np.ndarray) -> Path:
        """Apply design variables as FFD-style Y-direction control.
        
        This method interprets design variables as Y-direction displacements
        for conceptual FFD control points distributed along the airfoil.
        The actual deformation uses robust parametric methods to avoid mesh corruption.
        
        For [2,2,1] control points = 4 design variables:
        - design_vars[0]: Y-displacement for front upper control point
        - design_vars[1]: Y-displacement for rear upper control point  
        - design_vars[2]: Y-displacement for front lower control point
        - design_vars[3]: Y-displacement for rear lower control point
        
        Args:
            design_vars: Array of Y-displacements for conceptual FFD control points
            
        Returns:
            Path to modified mesh directory
        """
        self.logger.info(f"Applying hybrid FFD deformation (Y-direction only): {design_vars}")
        
        # Restore original mesh first
        self.restore_original_mesh()
        self._load_mesh_geometry()
        
        # Validate design variables
        if len(design_vars) != self.n_design_vars:
            raise ValueError(f"Expected {self.n_design_vars} design variables, got {len(design_vars)}")
        
        # Apply FFD-style Y-direction deformation using robust parametric approach
        deformed_points = self._apply_ffd_style_deformation(design_vars)
        
        # Validate mesh quality
        if not self._validate_deformation_quality(deformed_points):
            self.logger.warning("Deformation quality check failed, applying reduced deformation")
            # Apply 50% of the deformation
            reduced_vars = design_vars * 0.5
            deformed_points = self._apply_ffd_style_deformation(reduced_vars)
        
        # Update mesh with deformed airfoil
        self._update_mesh_points(deformed_points)
        
        # Final mesh integrity check
        if not self._check_mesh_integrity():
            self.logger.error("Mesh integrity check failed, restoring original")
            self.restore_original_mesh()
            raise RuntimeError("Hybrid FFD deformation failed - mesh integrity compromised")
        
        self.logger.info("Hybrid FFD mesh deformation completed successfully")
        return self.case_path / "constant" / "polyMesh"
    
    def _apply_ffd_style_deformation(self, design_vars: np.ndarray) -> np.ndarray:
        """Apply FFD-style deformation using robust parametric approach."""
        # Scale design variables to physical units
        scale_factor = self.chord_length * 0.002  # 0.2% of chord per unit for safety
        
        # For [2,2,1] control points, interpret as:
        # [front_upper, rear_upper, front_lower, rear_lower] Y-displacements
        if self.n_design_vars == 4:
            front_upper_y = design_vars[0] * scale_factor
            rear_upper_y = design_vars[1] * scale_factor  
            front_lower_y = design_vars[2] * scale_factor
            rear_lower_y = design_vars[3] * scale_factor
        else:
            # Distribute variables across control points
            front_upper_y = design_vars[0] * scale_factor if len(design_vars) > 0 else 0
            rear_upper_y = design_vars[1] * scale_factor if len(design_vars) > 1 else 0
            front_lower_y = design_vars[2] * scale_factor if len(design_vars) > 2 else 0
            rear_lower_y = design_vars[3] * scale_factor if len(design_vars) > 3 else 0
        
        # Apply safety limits
        max_displacement = self.chord_length * 0.01  # 1% of chord maximum
        front_upper_y = np.clip(front_upper_y, -max_displacement, max_displacement)
        rear_upper_y = np.clip(rear_upper_y, -max_displacement, max_displacement)
        front_lower_y = np.clip(front_lower_y, -max_displacement, max_displacement)
        rear_lower_y = np.clip(rear_lower_y, -max_displacement, max_displacement)
        
        # Create deformed airfoil points
        deformed_points = self.airfoil_points.copy()
        
        # Apply Y-direction displacements to upper surface
        for i, local_idx in enumerate(self.upper_surface_indices):
            x = self.airfoil_points[local_idx, 0]
            
            # Normalize x coordinate (0 at TE, 1 at LE)
            x_norm = (x - np.min(self.airfoil_points[:, 0])) / self.chord_length
            
            # Interpolate between front (LE) and rear (TE) control points
            # Front control point affects leading edge area (x_norm > 0.5)
            # Rear control point affects trailing edge area (x_norm < 0.5)
            if x_norm > 0.5:
                # Front region - blend towards front_upper_y
                weight_front = (x_norm - 0.5) / 0.5  # 0 at mid-chord, 1 at LE
                weight_rear = 1.0 - weight_front
                dy = weight_front * front_upper_y + weight_rear * rear_upper_y
            else:
                # Rear region - blend towards rear_upper_y
                weight_rear = (0.5 - x_norm) / 0.5  # 1 at TE, 0 at mid-chord
                weight_front = 1.0 - weight_rear
                dy = weight_front * front_upper_y + weight_rear * rear_upper_y
            
            # Apply smooth falloff to preserve surface continuity
            smoothing = np.sin(np.pi * x_norm) if x_norm > 0.1 and x_norm < 0.9 else 0.1
            dy *= smoothing
            
            deformed_points[local_idx, 1] += dy
        
        # Apply Y-direction displacements to lower surface
        for i, local_idx in enumerate(self.lower_surface_indices):
            x = self.airfoil_points[local_idx, 0]
            
            x_norm = (x - np.min(self.airfoil_points[:, 0])) / self.chord_length
            
            # Interpolate between front and rear control points for lower surface
            if x_norm > 0.5:
                weight_front = (x_norm - 0.5) / 0.5
                weight_rear = 1.0 - weight_front
                dy = weight_front * front_lower_y + weight_rear * rear_lower_y
            else:
                weight_rear = (0.5 - x_norm) / 0.5
                weight_front = 1.0 - weight_rear
                dy = weight_front * front_lower_y + weight_rear * rear_lower_y
            
            # Apply smooth falloff
            smoothing = np.sin(np.pi * x_norm) if x_norm > 0.1 and x_norm < 0.9 else 0.1
            dy *= smoothing
            
            deformed_points[local_idx, 1] += dy
        
        return deformed_points
    
    def _validate_deformation_quality(self, deformed_points: np.ndarray) -> bool:
        """Validate that the deformation maintains acceptable mesh quality."""
        try:
            # Check maximum displacement
            displacements = np.linalg.norm(deformed_points - self.airfoil_points, axis=1)
            max_displacement = np.max(displacements)
            avg_displacement = np.mean(displacements)
            
            # Conservative thresholds
            max_allowed = self.chord_length * 0.02  # 2% of chord maximum
            avg_allowed = max_allowed * 0.5
            
            if max_displacement > max_allowed:
                self.logger.warning(f"Max displacement {max_displacement:.6f} exceeds {max_allowed:.6f}")
                return False
            
            if avg_displacement > avg_allowed:
                self.logger.warning(f"Avg displacement {avg_displacement:.6f} exceeds {avg_allowed:.6f}")
                return False
            
            # Check for surface self-intersection
            if self._check_surface_intersection(deformed_points):
                self.logger.warning("Surface self-intersection detected")
                return False
            
            # Check minimum point spacing
            min_spacing = self._check_minimum_spacing(deformed_points)
            min_allowed_spacing = self.chord_length * 1e-6
            
            if min_spacing < min_allowed_spacing:
                self.logger.warning(f"Minimum spacing {min_spacing:.8f} too small")
                return False
            
            self.logger.info(f"Hybrid FFD quality OK: max_disp={max_displacement:.6f}, avg_disp={avg_displacement:.6f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Deformation quality validation error: {e}")
            return False
    
    def _check_surface_intersection(self, points: np.ndarray) -> bool:
        """Check for airfoil surface self-intersection."""
        try:
            # Simple check: ensure upper surface is always above lower surface
            if len(self.upper_surface_indices) == 0 or len(self.lower_surface_indices) == 0:
                return False
                
            upper_points = points[self.upper_surface_indices]
            lower_points = points[self.lower_surface_indices]
            
            # Check at several x-locations
            x_min = np.min(points[:, 0])
            x_max = np.max(points[:, 0])
            
            for x_check in np.linspace(x_min + 0.1 * self.chord_length, 
                                     x_max - 0.1 * self.chord_length, 10):
                # Find closest points on upper and lower surfaces
                upper_dists = np.abs(upper_points[:, 0] - x_check)
                lower_dists = np.abs(lower_points[:, 0] - x_check)
                
                if len(upper_dists) > 0 and len(lower_dists) > 0:
                    upper_y = upper_points[np.argmin(upper_dists), 1]
                    lower_y = lower_points[np.argmin(lower_dists), 1]
                    
                    if upper_y <= lower_y:  # Upper surface below lower surface
                        return True
            
            return False
            
        except Exception:
            return True  # Conservative: assume intersection if check fails
    
    def _check_minimum_spacing(self, points: np.ndarray) -> float:
        """Check minimum spacing between adjacent points."""
        try:
            min_spacing = np.inf
            
            # Check upper surface spacing
            for i in range(len(self.upper_surface_indices) - 1):
                p1 = points[self.upper_surface_indices[i]]
                p2 = points[self.upper_surface_indices[i + 1]]
                spacing = np.linalg.norm(p2 - p1)
                min_spacing = min(min_spacing, spacing)
            
            # Check lower surface spacing
            for i in range(len(self.lower_surface_indices) - 1):
                p1 = points[self.lower_surface_indices[i]]
                p2 = points[self.lower_surface_indices[i + 1]]
                spacing = np.linalg.norm(p2 - p1)
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
        
        self.logger.info(f"Updated mesh with {len(deformed_airfoil_points)} hybrid FFD-deformed airfoil points")
    
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
        """Get information about the current deformation state."""
        return {
            "airfoil_points": len(self.airfoil_indices) if self.airfoil_indices else 0,
            "chord_length": self.chord_length,
            "control_points": self.control_points,
            "design_variables": self.n_design_vars,
            "upper_surface_points": len(self.upper_surface_indices) if self.upper_surface_indices else 0,
            "lower_surface_points": len(self.lower_surface_indices) if self.lower_surface_indices else 0,
            "deformation_type": "Hybrid_FFD_Y_direction_only"
        }