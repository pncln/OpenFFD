"""
OpenFOAM polyMesh Reader

Implements comprehensive reading of OpenFOAM polyMesh format:
- Points (vertices) in 3D space
- Faces with point connectivity
- Cell-face connectivity (owner/neighbour)
- Boundary patch definitions
- Cell zones and face zones
- Support for compressed and uncompressed files
- Mesh quality analysis and validation
- Conversion to internal mesh data structures

Handles the standard OpenFOAM mesh format used in constant/polyMesh/
directory including points, faces, owner, neighbour, and boundary files.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import os
import gzip
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OpenFOAMBoundaryPatch:
    """Boundary patch information from OpenFOAM."""
    
    name: str
    patch_type: str  # wall, patch, symmetry, etc.
    n_faces: int
    start_face: int
    in_groups: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Compute end face index."""
        self.end_face = self.start_face + self.n_faces


@dataclass 
class OpenFOAMMeshData:
    """Complete OpenFOAM mesh data structure."""
    
    # Basic mesh data
    points: np.ndarray  # (n_points, 3) vertex coordinates
    faces: List[List[int]]  # face-point connectivity
    owner: np.ndarray  # (n_faces,) owner cell for each face
    neighbour: np.ndarray  # (n_internal_faces,) neighbour cell for internal faces
    
    # Boundary information
    boundary_patches: List[OpenFOAMBoundaryPatch]
    
    # Mesh statistics
    n_points: int
    n_faces: int
    n_cells: int
    n_internal_faces: int
    n_boundary_faces: int
    
    # Optional zone data
    cell_zones: Optional[Dict[str, List[int]]] = None
    face_zones: Optional[Dict[str, List[int]]] = None
    point_zones: Optional[Dict[str, List[int]]] = None
    
    # Mesh quality metrics
    mesh_quality: Optional[Dict[str, Any]] = None


class OpenFOAMFileParser:
    """
    Parser for OpenFOAM file format.
    
    Handles the OpenFOAM dictionary format with automatic decompression.
    """
    
    def __init__(self):
        """Initialize OpenFOAM file parser."""
        self.foam_file_header_pattern = re.compile(r'FoamFile\s*\{[^}]*\}', re.DOTALL)
        self.comment_pattern = re.compile(r'//.*$', re.MULTILINE)
        self.block_comment_pattern = re.compile(r'/\*.*?\*/', re.DOTALL)
    
    def parse_foam_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Parse OpenFOAM file and extract header information."""
        filepath = Path(filepath)
        
        # Handle compressed files
        content = self._read_file_content(filepath)
        
        # Extract FoamFile header
        header_match = self.foam_file_header_pattern.search(content)
        if not header_match:
            raise ValueError(f"No FoamFile header found in {filepath}")
        
        header_content = header_match.group(0)
        header_dict = self._parse_dictionary_content(header_content)
        
        # Remove header and comments from content
        data_content = content[header_match.end():]
        data_content = self._remove_comments(data_content)
        
        return {
            'header': header_dict.get('FoamFile', {}),
            'data_content': data_content.strip()
        }
    
    def _read_file_content(self, filepath: Path) -> str:
        """Read file content, handling compression."""
        if filepath.suffix == '.gz':
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                return f.read()
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _remove_comments(self, content: str) -> str:
        """Remove C++ style comments from content."""
        # Remove block comments
        content = self.block_comment_pattern.sub('', content)
        # Remove line comments
        content = self.comment_pattern.sub('', content)
        return content
    
    def _parse_dictionary_content(self, content: str) -> Dict[str, Any]:
        """Parse OpenFOAM dictionary format."""
        result = {}
        
        # Simple parser for key-value pairs
        lines = content.split('\n')
        current_dict = None
        dict_stack = [result]
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            # Dictionary start
            if '{' in line and not '}' in line:
                key = line.split('{')[0].strip()
                new_dict = {}
                dict_stack[-1][key] = new_dict
                dict_stack.append(new_dict)
            
            # Dictionary end
            elif '}' in line:
                if len(dict_stack) > 1:
                    dict_stack.pop()
            
            # Key-value pair
            elif ';' in line:
                parts = line.split(';')[0].strip().split(None, 1)
                if len(parts) == 2:
                    key, value = parts
                    # Try to convert to appropriate type
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        # Keep as string, remove quotes
                        value = value.strip('"\'')
                    
                    dict_stack[-1][key] = value
        
        return result


class OpenFOAMPolyMeshReader:
    """
    Comprehensive OpenFOAM polyMesh reader.
    
    Reads all polyMesh files and constructs complete mesh data structure.
    """
    
    def __init__(self):
        """Initialize polyMesh reader."""
        self.parser = OpenFOAMFileParser()
        
    def read_polymesh(self, polymesh_dir: Union[str, Path]) -> OpenFOAMMeshData:
        """
        Read complete OpenFOAM polyMesh from directory.
        
        Args:
            polymesh_dir: Path to polyMesh directory
            
        Returns:
            Complete mesh data structure
        """
        polymesh_dir = Path(polymesh_dir)
        
        if not polymesh_dir.exists():
            raise FileNotFoundError(f"polyMesh directory not found: {polymesh_dir}")
        
        logger.info(f"Reading OpenFOAM polyMesh from {polymesh_dir}")
        
        # Read required files
        points = self._read_points(polymesh_dir / "points")
        faces = self._read_faces(polymesh_dir / "faces")
        owner = self._read_owner(polymesh_dir / "owner")
        neighbour = self._read_neighbour(polymesh_dir / "neighbour")
        boundary_patches = self._read_boundary(polymesh_dir / "boundary")
        
        # Read optional zone files
        cell_zones = self._read_zones(polymesh_dir / "cellZones", "cellZones")
        face_zones = self._read_zones(polymesh_dir / "faceZones", "faceZones") 
        point_zones = self._read_zones(polymesh_dir / "pointZones", "pointZones")
        
        # Compute mesh statistics
        n_points = len(points)
        n_faces = len(faces)
        n_cells = max(np.max(owner), np.max(neighbour) if len(neighbour) > 0 else 0) + 1
        n_internal_faces = len(neighbour)
        n_boundary_faces = n_faces - n_internal_faces
        
        # Create mesh data structure
        mesh_data = OpenFOAMMeshData(
            points=points,
            faces=faces,
            owner=owner,
            neighbour=neighbour,
            boundary_patches=boundary_patches,
            n_points=n_points,
            n_faces=n_faces,
            n_cells=n_cells,
            n_internal_faces=n_internal_faces,
            n_boundary_faces=n_boundary_faces,
            cell_zones=cell_zones,
            face_zones=face_zones,
            point_zones=point_zones
        )
        
        # Validate mesh
        self._validate_mesh(mesh_data)
        
        # Compute mesh quality
        mesh_data.mesh_quality = self._compute_mesh_quality(mesh_data)
        
        logger.info(f"Successfully read mesh: {n_cells} cells, {n_faces} faces, {n_points} points")
        
        return mesh_data
    
    def _read_points(self, points_file: Path) -> np.ndarray:
        """Read points file."""
        # Handle both .gz and uncompressed files
        if points_file.with_suffix('.gz').exists():
            points_file = points_file.with_suffix('.gz')
        elif not points_file.exists():
            raise FileNotFoundError(f"Points file not found: {points_file}")
        
        parsed = self.parser.parse_foam_file(points_file)
        content = parsed['data_content']
        
        # Extract point count and data
        lines = content.strip().split('\n')
        n_points = int(lines[0])
        
        # Find opening parenthesis
        start_idx = 1
        while start_idx < len(lines) and '(' not in lines[start_idx]:
            start_idx += 1
        
        if start_idx >= len(lines):
            raise ValueError("Could not find points data")
        
        # Read points
        points = []
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            if line == ')':
                break
            
            # Parse point coordinates (x y z)
            if line.startswith('(') and line.endswith(')'):
                coords_str = line[1:-1]  # Remove parentheses
                coords = [float(x) for x in coords_str.split()]
                if len(coords) == 3:
                    points.append(coords)
        
        points_array = np.array(points)
        
        if len(points_array) != n_points:
            logger.warning(f"Expected {n_points} points, got {len(points_array)}")
        
        return points_array
    
    def _read_faces(self, faces_file: Path) -> List[List[int]]:
        """Read faces file."""
        # Handle both .gz and uncompressed files
        if faces_file.with_suffix('.gz').exists():
            faces_file = faces_file.with_suffix('.gz')
        elif not faces_file.exists():
            raise FileNotFoundError(f"Faces file not found: {faces_file}")
        
        parsed = self.parser.parse_foam_file(faces_file)
        content = parsed['data_content']
        
        # Extract face count and data
        lines = content.strip().split('\n')
        n_faces = int(lines[0])
        
        # Find opening parenthesis
        start_idx = 1
        while start_idx < len(lines) and '(' not in lines[start_idx]:
            start_idx += 1
        
        # Read faces
        faces = []
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            if line == ')':
                break
            
            # Parse face format: n(p1 p2 p3 ... pn)
            if '(' in line and ')' in line:
                # Extract point count and point indices
                parts = line.split('(')
                if len(parts) >= 2:
                    n_points_face = int(parts[0])
                    points_str = parts[1].split(')')[0]
                    point_indices = [int(x) for x in points_str.split()]
                    
                    if len(point_indices) == n_points_face:
                        faces.append(point_indices)
        
        if len(faces) != n_faces:
            logger.warning(f"Expected {n_faces} faces, got {len(faces)}")
        
        return faces
    
    def _read_owner(self, owner_file: Path) -> np.ndarray:
        """Read owner file."""
        # Handle both .gz and uncompressed files
        if owner_file.with_suffix('.gz').exists():
            owner_file = owner_file.with_suffix('.gz')
        elif not owner_file.exists():
            raise FileNotFoundError(f"Owner file not found: {owner_file}")
        
        parsed = self.parser.parse_foam_file(owner_file)
        content = parsed['data_content']
        
        # Extract owner data
        lines = content.strip().split('\n')
        n_faces = int(lines[0])
        
        # Find opening parenthesis
        start_idx = 1
        while start_idx < len(lines) and '(' not in lines[start_idx]:
            start_idx += 1
        
        # Read owner indices
        owner_indices = []
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            if line == ')':
                break
            
            if line.isdigit():
                owner_indices.append(int(line))
        
        owner_array = np.array(owner_indices, dtype=int)
        
        if len(owner_array) != n_faces:
            logger.warning(f"Expected {n_faces} owner entries, got {len(owner_array)}")
        
        return owner_array
    
    def _read_neighbour(self, neighbour_file: Path) -> np.ndarray:
        """Read neighbour file."""
        # Handle both .gz and uncompressed files
        if neighbour_file.with_suffix('.gz').exists():
            neighbour_file = neighbour_file.with_suffix('.gz')
        elif not neighbour_file.exists():
            # Neighbour file might not exist if there are no internal faces
            logger.warning(f"Neighbour file not found: {neighbour_file}")
            return np.array([], dtype=int)
        
        parsed = self.parser.parse_foam_file(neighbour_file)
        content = parsed['data_content']
        
        # Extract neighbour data
        lines = content.strip().split('\n')
        n_internal_faces = int(lines[0])
        
        if n_internal_faces == 0:
            return np.array([], dtype=int)
        
        # Find opening parenthesis
        start_idx = 1
        while start_idx < len(lines) and '(' not in lines[start_idx]:
            start_idx += 1
        
        # Read neighbour indices
        neighbour_indices = []
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            if line == ')':
                break
            
            if line.isdigit():
                neighbour_indices.append(int(line))
        
        neighbour_array = np.array(neighbour_indices, dtype=int)
        
        if len(neighbour_array) != n_internal_faces:
            logger.warning(f"Expected {n_internal_faces} neighbour entries, got {len(neighbour_array)}")
        
        return neighbour_array
    
    def _read_boundary(self, boundary_file: Path) -> List[OpenFOAMBoundaryPatch]:
        """Read boundary file."""
        if not boundary_file.exists():
            raise FileNotFoundError(f"Boundary file not found: {boundary_file}")
        
        parsed = self.parser.parse_foam_file(boundary_file)
        content = parsed['data_content']
        
        # Parse boundary patches
        patches = []
        lines = content.strip().split('\n')
        
        # Find number of patches
        n_patches = int(lines[0])
        
        # Find opening parenthesis
        start_idx = 1
        while start_idx < len(lines) and '(' not in lines[start_idx]:
            start_idx += 1
        
        # Parse each patch
        i = start_idx + 1
        while i < len(lines) and patches.__len__() < n_patches:
            line = lines[i].strip()
            
            if line and not line.startswith('{') and not line.startswith('}') and line != ')':
                # This should be a patch name
                patch_name = line
                patch_data = {}
                
                # Look for opening brace
                i += 1
                while i < len(lines) and '{' not in lines[i]:
                    i += 1
                
                # Read patch properties
                i += 1
                while i < len(lines) and '}' not in lines[i]:
                    prop_line = lines[i].strip()
                    if ';' in prop_line:
                        parts = prop_line.split(None, 1)
                        if len(parts) >= 2:
                            key = parts[0]
                            value = parts[1].rstrip(';').strip()
                            
                            # Convert to appropriate type
                            if key in ['nFaces', 'startFace']:
                                patch_data[key] = int(value)
                            elif key == 'inGroups':
                                # Parse inGroups format: 1(group1)
                                groups = []
                                if '(' in value and ')' in value:
                                    groups_str = value.split('(')[1].split(')')[0]
                                    if groups_str:
                                        groups = [groups_str]
                                patch_data[key] = groups
                            else:
                                patch_data[key] = value
                    i += 1
                
                # Create patch object
                patch = OpenFOAMBoundaryPatch(
                    name=patch_name,
                    patch_type=patch_data.get('type', 'patch'),
                    n_faces=patch_data.get('nFaces', 0),
                    start_face=patch_data.get('startFace', 0),
                    in_groups=patch_data.get('inGroups', [])
                )
                patches.append(patch)
            
            i += 1
        
        logger.info(f"Read {len(patches)} boundary patches")
        for patch in patches:
            logger.debug(f"  {patch.name}: {patch.patch_type}, {patch.n_faces} faces at {patch.start_face}")
        
        return patches
    
    def _read_zones(self, zones_file: Path, zone_type: str) -> Optional[Dict[str, List[int]]]:
        """Read zone files (cellZones, faceZones, pointZones)."""
        # Handle both .gz and uncompressed files
        if zones_file.with_suffix('.gz').exists():
            zones_file = zones_file.with_suffix('.gz')
        elif not zones_file.exists():
            # Zone files are optional
            return None
        
        try:
            parsed = self.parser.parse_foam_file(zones_file)
            content = parsed['data_content']
            
            # Simple zone parsing - this would need more sophisticated parsing
            # for complex zone definitions
            logger.info(f"Found {zone_type} file (parsing not fully implemented)")
            return {}
            
        except Exception as e:
            logger.warning(f"Could not read {zone_type}: {e}")
            return None
    
    def _validate_mesh(self, mesh_data: OpenFOAMMeshData):
        """Validate mesh data consistency."""
        logger.info("Validating mesh data...")
        
        # Check point indices in faces
        max_point_index = np.max([np.max(face) for face in mesh_data.faces if face])
        if max_point_index >= mesh_data.n_points:
            raise ValueError(f"Face references point {max_point_index} but only {mesh_data.n_points} points exist")
        
        # Check owner/neighbour consistency
        max_owner = np.max(mesh_data.owner)
        if max_owner >= mesh_data.n_cells:
            raise ValueError(f"Owner references cell {max_owner} but only {mesh_data.n_cells} cells exist")
        
        if len(mesh_data.neighbour) > 0:
            max_neighbour = np.max(mesh_data.neighbour)
            if max_neighbour >= mesh_data.n_cells:
                raise ValueError(f"Neighbour references cell {max_neighbour} but only {mesh_data.n_cells} cells exist")
        
        # Check boundary patch consistency
        total_boundary_faces = sum(patch.n_faces for patch in mesh_data.boundary_patches)
        if total_boundary_faces != mesh_data.n_boundary_faces:
            logger.warning(f"Boundary patches have {total_boundary_faces} faces but {mesh_data.n_boundary_faces} boundary faces expected")
        
        logger.info("Mesh validation completed")
    
    def _compute_mesh_quality(self, mesh_data: OpenFOAMMeshData) -> Dict[str, Any]:
        """Compute basic mesh quality metrics."""
        logger.info("Computing mesh quality metrics...")
        
        quality = {}
        
        # Face areas and cell volumes (simplified calculation)
        face_areas = []
        for face in mesh_data.faces:
            if len(face) >= 3:
                # Approximate area for triangular/quad faces
                p1, p2, p3 = mesh_data.points[face[0]], mesh_data.points[face[1]], mesh_data.points[face[2]]
                v1, v2 = p2 - p1, p3 - p1
                area = 0.5 * np.linalg.norm(np.cross(v1, v2))
                face_areas.append(area)
        
        quality['face_areas'] = {
            'min': np.min(face_areas),
            'max': np.max(face_areas),
            'mean': np.mean(face_areas),
            'std': np.std(face_areas)
        }
        
        # Point distribution
        quality['point_bounds'] = {
            'min': np.min(mesh_data.points, axis=0).tolist(),
            'max': np.max(mesh_data.points, axis=0).tolist(),
            'center': np.mean(mesh_data.points, axis=0).tolist()
        }
        
        # Mesh size statistics
        quality['mesh_size'] = {
            'n_points': mesh_data.n_points,
            'n_faces': mesh_data.n_faces,
            'n_cells': mesh_data.n_cells,
            'n_internal_faces': mesh_data.n_internal_faces,
            'n_boundary_faces': mesh_data.n_boundary_faces,
            'n_boundary_patches': len(mesh_data.boundary_patches)
        }
        
        logger.info("Mesh quality computation completed")
        
        return quality
    
    def convert_to_unstructured_mesh(self, mesh_data: OpenFOAMMeshData) -> Dict[str, Any]:
        """Convert OpenFOAM mesh to our internal unstructured mesh format."""
        logger.info("Converting to internal mesh format...")
        
        # Create internal mesh representation
        internal_mesh = {
            'vertices': mesh_data.points,
            'faces': mesh_data.faces,
            'cells': self._extract_cells_from_faces(mesh_data),
            'boundary_patches': {patch.name: {
                'type': patch.patch_type,
                'faces': list(range(patch.start_face, patch.start_face + patch.n_faces))
            } for patch in mesh_data.boundary_patches},
            'mesh_quality': mesh_data.mesh_quality,
            'openfoam_data': mesh_data
        }
        
        logger.info("Mesh conversion completed")
        
        return internal_mesh
    
    def _extract_cells_from_faces(self, mesh_data: OpenFOAMMeshData) -> List[List[int]]:
        """Extract cell definitions from face connectivity."""
        # This is a simplified approach - in reality, we'd need to properly
        # reconstruct cells from the face-cell connectivity
        
        cells = [[] for _ in range(mesh_data.n_cells)]
        
        # Add faces to cells based on owner/neighbour
        for face_idx, owner_cell in enumerate(mesh_data.owner):
            cells[owner_cell].append(face_idx)
        
        for face_idx, neighbour_cell in enumerate(mesh_data.neighbour):
            if face_idx < len(mesh_data.neighbour):
                cells[neighbour_cell].append(face_idx)
        
        return cells


def read_openfoam_mesh(polymesh_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to read OpenFOAM polyMesh.
    
    Args:
        polymesh_dir: Path to polyMesh directory
        
    Returns:
        Internal mesh format dictionary
    """
    reader = OpenFOAMPolyMeshReader()
    mesh_data = reader.read_polymesh(polymesh_dir)
    return reader.convert_to_unstructured_mesh(mesh_data)


def test_openfoam_reader():
    """Test OpenFOAM polyMesh reader with cylinder example."""
    print("Testing OpenFOAM polyMesh Reader:")
    
    # Test with cylinder mesh
    cylinder_mesh_path = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/examples/Cylinder/polyMesh"
    
    if not os.path.exists(cylinder_mesh_path):
        print(f"  Cylinder mesh not found at {cylinder_mesh_path}")
        return
    
    print(f"  Reading cylinder mesh from {cylinder_mesh_path}")
    
    try:
        # Read mesh
        mesh_data = read_openfoam_mesh(cylinder_mesh_path)
        
        print(f"\n  Mesh Statistics:")
        print(f"    Points: {mesh_data['vertices'].shape[0]}")
        print(f"    Faces: {len(mesh_data['faces'])}")
        print(f"    Cells: {len(mesh_data['cells'])}")
        print(f"    Boundary patches: {len(mesh_data['boundary_patches'])}")
        
        print(f"\n  Boundary Patches:")
        for patch_name, patch_info in mesh_data['boundary_patches'].items():
            print(f"    {patch_name}: {patch_info['type']}, {len(patch_info['faces'])} faces")
        
        print(f"\n  Mesh Bounds:")
        bounds = mesh_data['mesh_quality']['point_bounds']
        print(f"    Min: ({bounds['min'][0]:.3f}, {bounds['min'][1]:.3f}, {bounds['min'][2]:.3f})")
        print(f"    Max: ({bounds['max'][0]:.3f}, {bounds['max'][1]:.3f}, {bounds['max'][2]:.3f})")
        print(f"    Center: ({bounds['center'][0]:.3f}, {bounds['center'][1]:.3f}, {bounds['center'][2]:.3f})")
        
        print(f"\n  Face Area Statistics:")
        face_stats = mesh_data['mesh_quality']['face_areas']
        print(f"    Min area: {face_stats['min']:.6e}")
        print(f"    Max area: {face_stats['max']:.6e}")
        print(f"    Mean area: {face_stats['mean']:.6e}")
        
        # Test with NACA0012 mesh if available
        naca_mesh_path = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/examples/naca0012_case/constant/polyMesh"
        
        if os.path.exists(naca_mesh_path):
            print(f"\n  Testing with NACA0012 mesh:")
            
            naca_mesh = read_openfoam_mesh(naca_mesh_path)
            
            print(f"    NACA0012 Points: {naca_mesh['vertices'].shape[0]}")
            print(f"    NACA0012 Faces: {len(naca_mesh['faces'])}")
            print(f"    NACA0012 Cells: {len(naca_mesh['cells'])}")
            print(f"    NACA0012 Boundary patches: {len(naca_mesh['boundary_patches'])}")
            
            print(f"    NACA0012 Boundary Patches:")
            for patch_name, patch_info in naca_mesh['boundary_patches'].items():
                print(f"      {patch_name}: {patch_info['type']}, {len(patch_info['faces'])} faces")
        
        print(f"\n  OpenFOAM mesh reader test completed successfully!")
        
        return mesh_data
        
    except Exception as e:
        print(f"  Error reading mesh: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_openfoam_reader()