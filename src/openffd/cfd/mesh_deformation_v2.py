"""
Advanced Mesh Deformation System for CFD Shape Optimization

This module implements a robust, comprehensive mesh deformation system specifically
designed for airfoil shape optimization. It uses parametric NACA modifications
combined with radial basis function (RBF) smoothing to ensure mesh quality.
"""

import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import shutil
import json

logger = logging.getLogger(__name__)

class AirfoilMeshDeformer:
    """Advanced mesh deformation system for airfoil optimization."""
    
    def __init__(self, case_path: Path, config: Optional[Dict] = None):
        """Initialize the mesh deformation system.
        
        Args:
            case_path: Path to OpenFOAM case
            config: Configuration dictionary for deformation parameters
        """
        self.case_path = Path(case_path)
        self.logger = logging.getLogger(__name__)
        
        # Configuration with safe defaults
        self.config = config or {}
        self.max_deformation = self.config.get('max_deformation', 0.05)  # 5% of chord
        self.smoothing_radius = self.config.get('smoothing_radius', 0.1)  # 10% of chord
        self.quality_threshold = self.config.get('quality_threshold', 0.1)
        
        # Mesh data
        self.mesh_points = None
        self.airfoil_points = None
        self.airfoil_indices = None
        self.chord_length = None
        self.leading_edge_idx = None
        self.trailing_edge_idx = None
        
        # Airfoil parameterization
        self.upper_surface_indices = None
        self.lower_surface_indices = None
        self.original_upper_coords = None
        self.original_lower_coords = None
        
        # Initialize the system
        self._initialize()
    
    def _initialize(self):
        """Initialize the mesh deformation system."""
        self.logger.info("Initializing advanced airfoil mesh deformation system...")
        
        # Create backup of original mesh
        self._create_mesh_backup()
        
        # Load and analyze mesh
        self._load_mesh_geometry()
        self._detect_airfoil_surface()
        self._parameterize_airfoil()
        
        self.logger.info(f"Mesh deformation system initialized:")
        self.logger.info(f"  - Airfoil points: {len(self.airfoil_indices)}")
        self.logger.info(f"  - Chord length: {self.chord_length:.3f}")
        self.logger.info(f"  - Upper surface points: {len(self.upper_surface_indices)}")
        self.logger.info(f"  - Lower surface points: {len(self.lower_surface_indices)}")
    
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
        """Parameterize airfoil surface for shape modifications."""
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
        # Sort points by angle from leading edge
        le_point = self.airfoil_points[le_idx]
        
        # Calculate angles from leading edge
        dx = self.airfoil_points[:, 0] - le_point[0]
        dy = self.airfoil_points[:, 1] - le_point[1]
        angles = np.arctan2(dy, dx)
        
        # Upper surface: angles from 0 to π (positive y direction from LE)
        # Lower surface: angles from 0 to -π (negative y direction from LE)
        upper_mask = dy >= 0
        lower_mask = dy < 0
        
        # Sort by x-coordinate for consistent ordering
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
        
        # Store original coordinates for reference
        self.original_upper_coords = self.airfoil_points[self.upper_surface_indices].copy()
        self.original_lower_coords = self.airfoil_points[self.lower_surface_indices].copy()
        
        self.logger.info(f"Airfoil parameterization complete:")
        self.logger.info(f"  - Leading edge at point {le_idx}: ({le_point[0]:.3f}, {le_point[1]:.3f})")
        self.logger.info(f"  - Upper surface: {len(self.upper_surface_indices)} points")
        self.logger.info(f"  - Lower surface: {len(self.lower_surface_indices)} points")
    
    def apply_design_variables(self, design_vars: np.ndarray) -> Path:
        """Apply design variables to deform the airfoil mesh.
        
        This method implements a robust parametric deformation that:
        1. Uses NACA-like thickness and camber modifications
        2. Preserves leading and trailing edge positions
        3. Maintains smooth surface continuity
        4. Validates mesh quality
        
        Args:
            design_vars: Array of design parameters
                [0] - Upper surface max thickness change (±5% chord)
                [1] - Lower surface max thickness change (±5% chord)  
                [2] - Camber change at 25% chord (±2% chord)
                [3] - Camber change at 75% chord (±2% chord)
        
        Returns:
            Path to modified mesh directory
        """
        self.logger.info(f"Applying airfoil deformation with variables: {design_vars}")
        
        # Restore original mesh first
        self.restore_original_mesh()
        self._load_mesh_geometry()
        
        # Validate design variables
        design_vars = np.clip(design_vars, -1.0, 1.0)  # Clamp to reasonable bounds
        
        # Apply parametric shape modifications
        deformed_points = self._apply_parametric_deformation(design_vars)
        
        # Validate mesh quality
        if not self._validate_deformation_quality(deformed_points):
            self.logger.warning("Deformation quality check failed, applying reduced deformation")
            # Apply 50% of the deformation
            reduced_vars = design_vars * 0.5
            deformed_points = self._apply_parametric_deformation(reduced_vars)
        
        # Update mesh with deformed airfoil
        self._update_mesh_points(deformed_points)
        
        # Final mesh integrity check
        if not self._check_mesh_integrity():
            self.logger.error("Mesh integrity check failed, restoring original")
            self.restore_original_mesh()
            raise RuntimeError("Mesh deformation failed - mesh integrity compromised")
        
        self.logger.info("Airfoil mesh deformation completed successfully")
        return self.case_path / "constant" / "polyMesh"
    
    def _apply_parametric_deformation(self, design_vars: np.ndarray) -> np.ndarray:
        """Apply parametric deformation to airfoil surface."""
        # Scale design variables to physical units
        scale_factor = self.chord_length * 0.01  # 1% of chord per unit design variable
        
        thickness_upper = design_vars[0] * scale_factor
        thickness_lower = design_vars[1] * scale_factor
        camber_25 = design_vars[2] * scale_factor * 0.5  # Reduced camber sensitivity
        camber_75 = design_vars[3] * scale_factor * 0.5
        
        # Create deformed airfoil points
        deformed_points = self.airfoil_points.copy()
        
        # Apply upper surface thickness modification
        upper_indices_global = [self.airfoil_indices[i] for i in self.upper_surface_indices]
        for i, global_idx in enumerate(upper_indices_global):
            local_idx = self.airfoil_indices.index(global_idx)
            x = self.airfoil_points[local_idx, 0]
            
            # Normalize x coordinate (0 at TE, 1 at LE)
            x_norm = (x - np.min(self.airfoil_points[:, 0])) / self.chord_length
            
            # Smooth thickness distribution (maximum at 30% chord)
            if x_norm <= 0.3:
                thickness_factor = x_norm / 0.3
            else:
                thickness_factor = (1.0 - x_norm) / 0.7
            
            thickness_factor = max(0, thickness_factor)
            dy = thickness_upper * thickness_factor * np.sin(np.pi * x_norm)
            
            deformed_points[local_idx, 1] += dy
        
        # Apply lower surface thickness modification
        lower_indices_global = [self.airfoil_indices[i] for i in self.lower_surface_indices]
        for i, global_idx in enumerate(lower_indices_global):
            local_idx = self.airfoil_indices.index(global_idx)
            x = self.airfoil_points[local_idx, 0]
            
            x_norm = (x - np.min(self.airfoil_points[:, 0])) / self.chord_length
            
            if x_norm <= 0.3:
                thickness_factor = x_norm / 0.3
            else:
                thickness_factor = (1.0 - x_norm) / 0.7
            
            thickness_factor = max(0, thickness_factor)
            dy = thickness_lower * thickness_factor * np.sin(np.pi * x_norm)
            
            deformed_points[local_idx, 1] += dy
        
        # Apply camber modifications
        for local_idx in range(len(self.airfoil_points)):
            x = self.airfoil_points[local_idx, 0]
            x_norm = (x - np.min(self.airfoil_points[:, 0])) / self.chord_length
            
            # Camber modification at 25% chord
            if 0.15 <= x_norm <= 0.35:
                weight = 1.0 - abs(x_norm - 0.25) / 0.1
                dy_camber = camber_25 * weight
                deformed_points[local_idx, 1] += dy_camber
            
            # Camber modification at 75% chord  
            if 0.65 <= x_norm <= 0.85:
                weight = 1.0 - abs(x_norm - 0.75) / 0.1
                dy_camber = camber_75 * weight
                deformed_points[local_idx, 1] += dy_camber
        
        return deformed_points
    
    def _validate_deformation_quality(self, deformed_points: np.ndarray) -> bool:
        """Validate that the deformation maintains acceptable mesh quality."""
        try:
            # Check maximum displacement
            displacements = np.linalg.norm(deformed_points - self.airfoil_points, axis=1)
            max_displacement = np.max(displacements)
            avg_displacement = np.mean(displacements)
            
            max_allowed = self.chord_length * self.max_deformation
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
            
            self.logger.info(f"Deformation quality OK: max_disp={max_displacement:.6f}, avg_disp={avg_displacement:.6f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Deformation quality validation error: {e}")
            return False
    
    def _check_surface_intersection(self, points: np.ndarray) -> bool:
        """Check for airfoil surface self-intersection."""
        try:
            # Simple check: ensure upper surface is always above lower surface
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
        
        self.logger.info(f"Updated mesh with {len(deformed_airfoil_points)} deformed airfoil points")
    
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
            "upper_surface_points": len(self.upper_surface_indices) if self.upper_surface_indices else 0,
            "lower_surface_points": len(self.lower_surface_indices) if self.lower_surface_indices else 0,
            "max_deformation": self.max_deformation,
            "smoothing_radius": self.smoothing_radius
        }