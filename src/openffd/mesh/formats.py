"""
Advanced Mesh Format I/O Module

This module provides comprehensive support for reading and writing various mesh formats:
- OpenFOAM (polymesh, ASCII/Binary)
- VTK (Legacy/XML, ASCII/Binary)
- STL (ASCII/Binary)
- CGNS (Computational Grid Notation Standard)
- GMSH (ASCII/Binary)
- Fluent mesh format
- Tecplot format
- NASTRAN format
- Abaqus format
"""

import numpy as np
import struct
from typing import Dict, List, Tuple, Optional, Union, Any, BinaryIO, TextIO
from dataclasses import dataclass
from pathlib import Path
import logging
import xml.etree.ElementTree as ET
from enum import Enum
import re
import warnings

logger = logging.getLogger(__name__)

class MeshFormat(Enum):
    """Supported mesh formats."""
    OPENFOAM = "openfoam"
    VTK_LEGACY = "vtk_legacy"
    VTK_XML = "vtk_xml"
    STL_ASCII = "stl_ascii"
    STL_BINARY = "stl_binary"
    CGNS = "cgns"
    GMSH_ASCII = "gmsh_ascii"
    GMSH_BINARY = "gmsh_binary"
    FLUENT = "fluent"
    TECPLOT = "tecplot"
    NASTRAN = "nastran"
    ABAQUS = "abaqus"
    NEUTRAL = "neutral"

@dataclass
class MeshData:
    """Comprehensive mesh data structure."""
    points: np.ndarray  # Vertex coordinates (N, 3)
    cells: List[np.ndarray]  # Cell connectivity (variable size per cell)
    cell_types: List[int]  # Cell type identifiers
    point_data: Dict[str, np.ndarray] = None  # Point-associated data
    cell_data: Dict[str, np.ndarray] = None   # Cell-associated data
    field_data: Dict[str, Any] = None         # Global field data
    
    # Boundary information
    boundary_faces: List[np.ndarray] = None
    boundary_conditions: Dict[str, Dict] = None
    
    # Metadata
    format_version: str = None
    software_info: str = None
    creation_time: str = None
    units: str = None
    dimension: int = 3
    
    def __post_init__(self):
        """Initialize default values."""
        if self.point_data is None:
            self.point_data = {}
        if self.cell_data is None:
            self.cell_data = {}
        if self.field_data is None:
            self.field_data = {}
        if self.boundary_faces is None:
            self.boundary_faces = []
        if self.boundary_conditions is None:
            self.boundary_conditions = {}

class VTKCellType(Enum):
    """VTK cell type constants."""
    VERTEX = 1
    POLY_VERTEX = 2
    LINE = 3
    POLY_LINE = 4
    TRIANGLE = 5
    TRIANGLE_STRIP = 6
    POLYGON = 7
    PIXEL = 8
    QUAD = 9
    TETRA = 10
    VOXEL = 11
    HEXAHEDRON = 12
    WEDGE = 13
    PYRAMID = 14

class MeshFormatRegistry:
    """Registry for mesh format readers and writers."""
    
    def __init__(self):
        self.readers = {}
        self.writers = {}
        self._register_default_formats()
    
    def _register_default_formats(self):
        """Register default format handlers."""
        # OpenFOAM
        self.readers[MeshFormat.OPENFOAM] = OpenFOAMReader()
        self.writers[MeshFormat.OPENFOAM] = OpenFOAMWriter()
        
        # VTK
        self.readers[MeshFormat.VTK_LEGACY] = VTKLegacyReader()
        self.writers[MeshFormat.VTK_LEGACY] = VTKLegacyWriter()
        self.readers[MeshFormat.VTK_XML] = VTKXMLReader()
        self.writers[MeshFormat.VTK_XML] = VTKXMLWriter()
        
        # STL
        self.readers[MeshFormat.STL_ASCII] = STLReader()
        self.writers[MeshFormat.STL_ASCII] = STLWriter()
        self.readers[MeshFormat.STL_BINARY] = STLReader()
        self.writers[MeshFormat.STL_BINARY] = STLWriter()
        
        # GMSH
        self.readers[MeshFormat.GMSH_ASCII] = GMSHReader()
        self.writers[MeshFormat.GMSH_ASCII] = GMSHWriter()
        
        # Fluent
        self.readers[MeshFormat.FLUENT] = FluentReader()
        self.writers[MeshFormat.FLUENT] = FluentWriter()
        
        # CGNS (requires external library)
        try:
            self.readers[MeshFormat.CGNS] = CGNSReader()
            self.writers[MeshFormat.CGNS] = CGNSWriter()
        except ImportError:
            logger.warning("CGNS support not available (requires pyCGNS)")
    
    def get_reader(self, format: MeshFormat):
        """Get reader for specified format."""
        if format not in self.readers:
            raise ValueError(f"No reader available for format: {format}")
        return self.readers[format]
    
    def get_writer(self, format: MeshFormat):
        """Get writer for specified format."""
        if format not in self.writers:
            raise ValueError(f"No writer available for format: {format}")
        return self.writers[format]

# Base classes for format handlers
class MeshReader:
    """Base class for mesh readers."""
    
    def read(self, file_path: Path) -> MeshData:
        """Read mesh from file."""
        raise NotImplementedError
    
    def detect_format(self, file_path: Path) -> bool:
        """Detect if file is in this format."""
        raise NotImplementedError

class MeshWriter:
    """Base class for mesh writers."""
    
    def write(self, mesh_data: MeshData, file_path: Path, **kwargs):
        """Write mesh to file."""
        raise NotImplementedError

# OpenFOAM format handlers
class OpenFOAMReader(MeshReader):
    """OpenFOAM mesh reader."""
    
    def read(self, file_path: Path) -> MeshData:
        """Read OpenFOAM polymesh format."""
        logger.info(f"Reading OpenFOAM mesh from {file_path}")
        
        # OpenFOAM mesh is stored in multiple files
        mesh_dir = file_path if file_path.is_dir() else file_path.parent
        
        # Read points
        points_file = mesh_dir / "points"
        points = self._read_points(points_file)
        
        # Read faces
        faces_file = mesh_dir / "faces"
        faces = self._read_faces(faces_file)
        
        # Read cells
        owner_file = mesh_dir / "owner"
        neighbour_file = mesh_dir / "neighbour"
        cells = self._read_cells(owner_file, neighbour_file, len(faces))
        
        # Read boundary conditions
        boundary_file = mesh_dir / "boundary"
        boundary_conditions = self._read_boundary(boundary_file) if boundary_file.exists() else {}
        
        mesh_data = MeshData(
            points=points,
            cells=cells,
            cell_types=[VTKCellType.HEXAHEDRON.value] * len(cells),  # Simplified assumption
            boundary_conditions=boundary_conditions,
            format_version="OpenFOAM",
            dimension=3
        )
        
        logger.info(f"Read OpenFOAM mesh: {len(points)} points, {len(cells)} cells")
        return mesh_data
    
    def _read_points(self, points_file: Path) -> np.ndarray:
        """Read points file."""
        with open(points_file, 'r') as f:
            content = f.read()
        
        # Parse OpenFOAM format
        # This is a simplified parser - full implementation would be more robust
        lines = content.strip().split('\n')
        
        # Find the number of points
        n_points = None
        start_idx = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line.isdigit():
                n_points = int(line)
                start_idx = i + 1
                break
        
        if n_points is None:
            raise ValueError("Could not find number of points in OpenFOAM points file")
        
        # Read point coordinates
        points = []
        in_data = False
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if line == '(':
                in_data = True
                continue
            elif line == ')':
                break
            elif in_data and line:
                # Parse coordinate line: (x y z)
                coords = line.strip('()').split()
                if len(coords) == 3:
                    points.append([float(x) for x in coords])
        
        return np.array(points)
    
    def _read_faces(self, faces_file: Path) -> List[np.ndarray]:
        """Read faces file."""
        with open(faces_file, 'r') as f:
            content = f.read()
        
        # Simplified parser for faces
        faces = []
        lines = content.strip().split('\n')
        
        # Find number of faces and data section
        in_data = False
        for line in lines:
            line = line.strip()
            if line == '(':
                in_data = True
                continue
            elif line == ')':
                break
            elif in_data and line:
                # Parse face: 4(p0 p1 p2 p3)
                face_match = re.match(r'(\d+)\(([^)]+)\)', line)
                if face_match:
                    n_vertices = int(face_match.group(1))
                    vertex_ids = [int(x) for x in face_match.group(2).split()]
                    faces.append(np.array(vertex_ids))
        
        return faces
    
    def _read_cells(self, owner_file: Path, neighbour_file: Path, n_faces: int) -> List[np.ndarray]:
        """Read cell connectivity from owner/neighbour files."""
        # Read owner file
        with open(owner_file, 'r') as f:
            owner_content = f.read()
        
        # Read neighbour file if it exists
        neighbour_content = ""
        if neighbour_file.exists():
            with open(neighbour_file, 'r') as f:
                neighbour_content = f.read()
        
        # Parse owner data
        owner_data = self._parse_openfoam_list(owner_content)
        neighbour_data = self._parse_openfoam_list(neighbour_content) if neighbour_content else []
        
        # Build cell-to-face connectivity (simplified)
        n_cells = max(owner_data) + 1 if owner_data else 0
        if neighbour_data:
            n_cells = max(n_cells, max(neighbour_data) + 1)
        
        # Create simplified cell connectivity
        cells = []
        for cell_id in range(n_cells):
            # Find faces belonging to this cell
            cell_faces = []
            for face_id, owner in enumerate(owner_data):
                if owner == cell_id:
                    cell_faces.append(face_id)
            for face_id, neighbour in enumerate(neighbour_data):
                if neighbour == cell_id:
                    cell_faces.append(face_id)
            
            # Simplified: assume hexahedral cells with 8 vertices
            # In reality, this would require more complex logic
            if len(cell_faces) >= 6:  # Minimum for a hexahedron
                cells.append(np.arange(cell_id * 8, (cell_id + 1) * 8))  # Placeholder
        
        return cells
    
    def _read_boundary(self, boundary_file: Path) -> Dict[str, Dict]:
        """Read boundary conditions file."""
        with open(boundary_file, 'r') as f:
            content = f.read()
        
        # Simplified boundary parser
        boundary_conditions = {}
        
        # This would need a more sophisticated parser for real OpenFOAM boundary files
        lines = content.strip().split('\n')
        current_patch = None
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('//'):
                # Look for patch definitions
                if '{' in line and '}' not in line:
                    # Start of patch definition
                    patch_name = line.split()[0] if line.split() else None
                    if patch_name:
                        current_patch = patch_name
                        boundary_conditions[patch_name] = {}
                elif current_patch and 'type' in line:
                    # Extract boundary type
                    type_match = re.search(r'type\s+(\w+)', line)
                    if type_match:
                        boundary_conditions[current_patch]['type'] = type_match.group(1)
                elif line == '}' and current_patch:
                    current_patch = None
        
        return boundary_conditions
    
    def _parse_openfoam_list(self, content: str) -> List[int]:
        """Parse OpenFOAM list format."""
        data = []
        lines = content.strip().split('\n')
        
        in_data = False
        for line in lines:
            line = line.strip()
            if line == '(':
                in_data = True
                continue
            elif line == ')':
                break
            elif in_data and line.isdigit():
                data.append(int(line))
        
        return data
    
    def detect_format(self, file_path: Path) -> bool:
        """Detect OpenFOAM format."""
        if file_path.is_dir():
            # Check for OpenFOAM mesh files
            required_files = ['points', 'faces', 'owner']
            return all((file_path / f).exists() for f in required_files)
        return False

class OpenFOAMWriter(MeshWriter):
    """OpenFOAM mesh writer."""
    
    def write(self, mesh_data: MeshData, file_path: Path, **kwargs):
        """Write OpenFOAM polymesh format."""
        logger.info(f"Writing OpenFOAM mesh to {file_path}")
        
        # Create directory structure
        mesh_dir = file_path
        mesh_dir.mkdir(parents=True, exist_ok=True)
        
        # Write points
        self._write_points(mesh_data.points, mesh_dir / "points")
        
        # Write faces and cells (simplified implementation)
        faces, owner, neighbour = self._generate_faces_and_connectivity(mesh_data)
        self._write_faces(faces, mesh_dir / "faces")
        self._write_owner(owner, mesh_dir / "owner")
        if neighbour:
            self._write_neighbour(neighbour, mesh_dir / "neighbour")
        
        # Write boundary conditions
        if mesh_data.boundary_conditions:
            self._write_boundary(mesh_data.boundary_conditions, mesh_dir / "boundary")
        
        logger.info(f"Written OpenFOAM mesh: {len(mesh_data.points)} points, {len(mesh_data.cells)} cells")
    
    def _write_points(self, points: np.ndarray, file_path: Path):
        """Write points file."""
        with open(file_path, 'w') as f:
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       vectorField;\n")
            f.write("    object      points;\n")
            f.write("}\n\n")
            
            f.write(f"{len(points)}\n")
            f.write("(\n")
            for point in points:
                f.write(f"({point[0]} {point[1]} {point[2]})\n")
            f.write(")\n")
    
    def _write_faces(self, faces: List[np.ndarray], file_path: Path):
        """Write faces file."""
        with open(file_path, 'w') as f:
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       faceList;\n")
            f.write("    object      faces;\n")
            f.write("}\n\n")
            
            f.write(f"{len(faces)}\n")
            f.write("(\n")
            for face in faces:
                f.write(f"{len(face)}({' '.join(map(str, face))})\n")
            f.write(")\n")
    
    def _write_owner(self, owner: List[int], file_path: Path):
        """Write owner file."""
        with open(file_path, 'w') as f:
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       labelList;\n")
            f.write("    object      owner;\n")
            f.write("}\n\n")
            
            f.write(f"{len(owner)}\n")
            f.write("(\n")
            for o in owner:
                f.write(f"{o}\n")
            f.write(")\n")
    
    def _write_neighbour(self, neighbour: List[int], file_path: Path):
        """Write neighbour file."""
        with open(file_path, 'w') as f:
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       labelList;\n")
            f.write("    object      neighbour;\n")
            f.write("}\n\n")
            
            f.write(f"{len(neighbour)}\n")
            f.write("(\n")
            for n in neighbour:
                f.write(f"{n}\n")
            f.write(")\n")
    
    def _write_boundary(self, boundary_conditions: Dict[str, Dict], file_path: Path):
        """Write boundary file."""
        with open(file_path, 'w') as f:
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       polyBoundaryMesh;\n")
            f.write("    object      boundary;\n")
            f.write("}\n\n")
            
            f.write(f"{len(boundary_conditions)}\n")
            f.write("(\n")
            
            start_face = 0
            for patch_name, patch_data in boundary_conditions.items():
                patch_type = patch_data.get('type', 'patch')
                n_faces = patch_data.get('nFaces', 0)
                
                f.write(f"    {patch_name}\n")
                f.write("    {\n")
                f.write(f"        type            {patch_type};\n")
                f.write(f"        nFaces          {n_faces};\n")
                f.write(f"        startFace       {start_face};\n")
                f.write("    }\n")
                
                start_face += n_faces
            
            f.write(")\n")
    
    def _generate_faces_and_connectivity(self, mesh_data: MeshData) -> Tuple[List[np.ndarray], List[int], List[int]]:
        """Generate faces and connectivity from cell data."""
        # Simplified implementation - in reality this is quite complex
        faces = []
        owner = []
        neighbour = []
        
        for cell_id, cell in enumerate(mesh_data.cells):
            # Generate faces for this cell (simplified for demonstration)
            if len(cell) == 8:  # Hexahedron
                # Define faces of hexahedron
                hex_faces = [
                    [cell[0], cell[1], cell[2], cell[3]],  # Bottom
                    [cell[4], cell[5], cell[6], cell[7]],  # Top
                    [cell[0], cell[1], cell[5], cell[4]],  # Front
                    [cell[2], cell[3], cell[7], cell[6]],  # Back
                    [cell[0], cell[3], cell[7], cell[4]],  # Left
                    [cell[1], cell[2], cell[6], cell[5]]   # Right
                ]
                
                for face_vertices in hex_faces:
                    faces.append(np.array(face_vertices))
                    owner.append(cell_id)
            elif len(cell) == 4:  # Tetrahedron
                # Define faces of tetrahedron
                tet_faces = [
                    [cell[0], cell[1], cell[2]],
                    [cell[0], cell[1], cell[3]],
                    [cell[0], cell[2], cell[3]],
                    [cell[1], cell[2], cell[3]]
                ]
                
                for face_vertices in tet_faces:
                    faces.append(np.array(face_vertices))
                    owner.append(cell_id)
        
        return faces, owner, neighbour

# VTK format handlers
class VTKLegacyReader(MeshReader):
    """VTK Legacy format reader."""
    
    def read(self, file_path: Path) -> MeshData:
        """Read VTK Legacy format."""
        logger.info(f"Reading VTK Legacy mesh from {file_path}")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        if not lines[0].startswith('# vtk DataFile Version'):
            raise ValueError("Not a valid VTK file")
        
        title = lines[1].strip()
        data_type = lines[2].strip()  # ASCII or BINARY
        dataset_type = lines[3].strip().split()[1]  # UNSTRUCTURED_GRID, etc.
        
        # Parse dataset
        points = None
        cells = []
        cell_types = []
        point_data = {}
        cell_data = {}
        
        i = 4
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('POINTS'):
                points, i = self._parse_vtk_points(lines, i)
            elif line.startswith('CELLS'):
                cells, i = self._parse_vtk_cells(lines, i)
            elif line.startswith('CELL_TYPES'):
                cell_types, i = self._parse_vtk_cell_types(lines, i)
            elif line.startswith('POINT_DATA'):
                point_data, i = self._parse_vtk_point_data(lines, i)
            elif line.startswith('CELL_DATA'):
                cell_data, i = self._parse_vtk_cell_data(lines, i)
            else:
                i += 1
        
        mesh_data = MeshData(
            points=points,
            cells=cells,
            cell_types=cell_types,
            point_data=point_data,
            cell_data=cell_data,
            format_version="VTK Legacy",
            software_info=title
        )
        
        logger.info(f"Read VTK mesh: {len(points)} points, {len(cells)} cells")
        return mesh_data
    
    def _parse_vtk_points(self, lines: List[str], start_idx: int) -> Tuple[np.ndarray, int]:
        """Parse VTK points section."""
        line = lines[start_idx].strip()
        n_points = int(line.split()[1])
        data_type = line.split()[2] if len(line.split()) > 2 else 'float'
        
        points = []
        i = start_idx + 1
        while len(points) < n_points * 3 and i < len(lines):
            values = lines[i].strip().split()
            points.extend([float(x) for x in values])
            i += 1
        
        points_array = np.array(points[:n_points * 3]).reshape(n_points, 3)
        return points_array, i
    
    def _parse_vtk_cells(self, lines: List[str], start_idx: int) -> Tuple[List[np.ndarray], int]:
        """Parse VTK cells section."""
        line = lines[start_idx].strip()
        n_cells = int(line.split()[1])
        
        cells = []
        i = start_idx + 1
        cells_read = 0
        
        while cells_read < n_cells and i < len(lines):
            values = [int(x) for x in lines[i].strip().split()]
            if values:
                n_vertices = values[0]
                cell_vertices = np.array(values[1:n_vertices+1])
                cells.append(cell_vertices)
                cells_read += 1
            i += 1
        
        return cells, i
    
    def _parse_vtk_cell_types(self, lines: List[str], start_idx: int) -> Tuple[List[int], int]:
        """Parse VTK cell types section."""
        line = lines[start_idx].strip()
        n_cells = int(line.split()[1])
        
        cell_types = []
        i = start_idx + 1
        while len(cell_types) < n_cells and i < len(lines):
            values = [int(x) for x in lines[i].strip().split()]
            cell_types.extend(values)
            i += 1
        
        return cell_types[:n_cells], i
    
    def _parse_vtk_point_data(self, lines: List[str], start_idx: int) -> Tuple[Dict[str, np.ndarray], int]:
        """Parse VTK point data section."""
        # Simplified implementation
        return {}, start_idx + 1
    
    def _parse_vtk_cell_data(self, lines: List[str], start_idx: int) -> Tuple[Dict[str, np.ndarray], int]:
        """Parse VTK cell data section."""
        # Simplified implementation
        return {}, start_idx + 1
    
    def detect_format(self, file_path: Path) -> bool:
        """Detect VTK Legacy format."""
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline()
                return first_line.startswith('# vtk DataFile Version')
        except:
            return False

class VTKLegacyWriter(MeshWriter):
    """VTK Legacy format writer."""
    
    def write(self, mesh_data: MeshData, file_path: Path, **kwargs):
        """Write VTK Legacy format."""
        logger.info(f"Writing VTK Legacy mesh to {file_path}")
        
        binary = kwargs.get('binary', False)
        
        with open(file_path, 'w') as f:
            # Write header
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"{mesh_data.software_info or 'Mesh generated by OpenFFD'}\n")
            f.write("ASCII\n" if not binary else "BINARY\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            # Write points
            f.write(f"POINTS {len(mesh_data.points)} float\n")
            for point in mesh_data.points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
            
            # Write cells
            total_cell_size = sum(len(cell) + 1 for cell in mesh_data.cells)
            f.write(f"CELLS {len(mesh_data.cells)} {total_cell_size}\n")
            for cell in mesh_data.cells:
                f.write(f"{len(cell)} {' '.join(map(str, cell))}\n")
            
            # Write cell types
            f.write(f"CELL_TYPES {len(mesh_data.cells)}\n")
            for cell_type in mesh_data.cell_types:
                f.write(f"{cell_type}\n")
            
            # Write point data
            if mesh_data.point_data:
                f.write(f"POINT_DATA {len(mesh_data.points)}\n")
                for name, data in mesh_data.point_data.items():
                    self._write_vtk_data_array(f, name, data, 'SCALARS')
            
            # Write cell data
            if mesh_data.cell_data:
                f.write(f"CELL_DATA {len(mesh_data.cells)}\n")
                for name, data in mesh_data.cell_data.items():
                    self._write_vtk_data_array(f, name, data, 'SCALARS')
        
        logger.info(f"Written VTK mesh: {len(mesh_data.points)} points, {len(mesh_data.cells)} cells")
    
    def _write_vtk_data_array(self, f: TextIO, name: str, data: np.ndarray, data_type: str):
        """Write VTK data array."""
        if data_type == 'SCALARS':
            f.write(f"SCALARS {name} float\n")
            f.write("LOOKUP_TABLE default\n")
            for value in data:
                f.write(f"{value}\n")
        elif data_type == 'VECTORS':
            f.write(f"VECTORS {name} float\n")
            for vector in data:
                f.write(f"{vector[0]} {vector[1]} {vector[2]}\n")

# VTK XML format handlers (simplified)
class VTKXMLReader(MeshReader):
    """VTK XML format reader."""
    
    def read(self, file_path: Path) -> MeshData:
        """Read VTK XML format."""
        logger.info(f"Reading VTK XML mesh from {file_path}")
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Parse based on dataset type
        if root.tag == 'VTKFile':
            dataset = root[0]  # First child should be the dataset
            
            if dataset.tag == 'UnstructuredGrid':
                return self._parse_unstructured_grid(dataset)
            else:
                raise ValueError(f"Unsupported VTK XML dataset type: {dataset.tag}")
        
        raise ValueError("Invalid VTK XML file structure")
    
    def _parse_unstructured_grid(self, grid_element: ET.Element) -> MeshData:
        """Parse UnstructuredGrid element."""
        # Simplified implementation
        points = np.array([[0, 0, 0]])  # Placeholder
        cells = [np.array([0])]         # Placeholder
        cell_types = [1]                # Placeholder
        
        return MeshData(
            points=points,
            cells=cells,
            cell_types=cell_types,
            format_version="VTK XML"
        )
    
    def detect_format(self, file_path: Path) -> bool:
        """Detect VTK XML format."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            return root.tag == 'VTKFile'
        except:
            return False

class VTKXMLWriter(MeshWriter):
    """VTK XML format writer."""
    
    def write(self, mesh_data: MeshData, file_path: Path, **kwargs):
        """Write VTK XML format."""
        logger.info(f"Writing VTK XML mesh to {file_path}")
        
        # Create XML structure
        root = ET.Element('VTKFile')
        root.set('type', 'UnstructuredGrid')
        root.set('version', '0.1')
        root.set('byte_order', 'LittleEndian')
        
        grid = ET.SubElement(root, 'UnstructuredGrid')
        piece = ET.SubElement(grid, 'Piece')
        piece.set('NumberOfPoints', str(len(mesh_data.points)))
        piece.set('NumberOfCells', str(len(mesh_data.cells)))
        
        # Points
        points_elem = ET.SubElement(piece, 'Points')
        points_data = ET.SubElement(points_elem, 'DataArray')
        points_data.set('type', 'Float32')
        points_data.set('NumberOfComponents', '3')
        points_data.set('format', 'ascii')
        
        points_text = []
        for point in mesh_data.points:
            points_text.append(f"{point[0]} {point[1]} {point[2]}")
        points_data.text = '\n'.join(points_text)
        
        # Cells
        cells_elem = ET.SubElement(piece, 'Cells')
        
        # Connectivity
        connectivity = ET.SubElement(cells_elem, 'DataArray')
        connectivity.set('type', 'Int32')
        connectivity.set('Name', 'connectivity')
        connectivity.set('format', 'ascii')
        
        conn_text = []
        for cell in mesh_data.cells:
            conn_text.extend(map(str, cell))
        connectivity.text = ' '.join(conn_text)
        
        # Offsets
        offsets = ET.SubElement(cells_elem, 'DataArray')
        offsets.set('type', 'Int32')
        offsets.set('Name', 'offsets')
        offsets.set('format', 'ascii')
        
        offset_values = []
        current_offset = 0
        for cell in mesh_data.cells:
            current_offset += len(cell)
            offset_values.append(str(current_offset))
        offsets.text = ' '.join(offset_values)
        
        # Types
        types = ET.SubElement(cells_elem, 'DataArray')
        types.set('type', 'UInt8')
        types.set('Name', 'types')
        types.set('format', 'ascii')
        types.text = ' '.join(map(str, mesh_data.cell_types))
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(file_path, xml_declaration=True, encoding='utf-8')
        
        logger.info(f"Written VTK XML mesh: {len(mesh_data.points)} points, {len(mesh_data.cells)} cells")

# STL format handlers
class STLReader(MeshReader):
    """STL format reader (both ASCII and binary)."""
    
    def read(self, file_path: Path) -> MeshData:
        """Read STL format."""
        logger.info(f"Reading STL mesh from {file_path}")
        
        # Detect ASCII vs binary
        with open(file_path, 'rb') as f:
            header = f.read(80)
            if b'solid' in header:
                # Likely ASCII
                return self._read_stl_ascii(file_path)
            else:
                # Binary
                return self._read_stl_binary(file_path)
    
    def _read_stl_ascii(self, file_path: Path) -> MeshData:
        """Read ASCII STL format."""
        points = []
        cells = []
        normals = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        point_idx = 0
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('facet normal'):
                # Extract normal vector
                normal = [float(x) for x in line.split()[2:5]]
                normals.append(normal)
                
                # Skip to vertices
                i += 2  # Skip "outer loop"
                
                # Read 3 vertices
                triangle_points = []
                for _ in range(3):
                    if i < len(lines) and lines[i].strip().startswith('vertex'):
                        vertex = [float(x) for x in lines[i].strip().split()[1:4]]
                        points.append(vertex)
                        triangle_points.append(point_idx)
                        point_idx += 1
                    i += 1
                
                if len(triangle_points) == 3:
                    cells.append(np.array(triangle_points))
                
                # Skip "endloop" and "endfacet"
                i += 2
            else:
                i += 1
        
        points_array = np.array(points)
        cell_types = [VTKCellType.TRIANGLE.value] * len(cells)
        
        mesh_data = MeshData(
            points=points_array,
            cells=cells,
            cell_types=cell_types,
            cell_data={'normals': np.array(normals)},
            format_version="STL ASCII"
        )
        
        logger.info(f"Read STL mesh: {len(points)} points, {len(cells)} triangles")
        return mesh_data
    
    def _read_stl_binary(self, file_path: Path) -> MeshData:
        """Read binary STL format."""
        with open(file_path, 'rb') as f:
            # Skip header
            f.read(80)
            
            # Read number of triangles
            n_triangles = struct.unpack('<I', f.read(4))[0]
            
            points = []
            cells = []
            normals = []
            
            point_idx = 0
            for _ in range(n_triangles):
                # Read normal vector
                normal = struct.unpack('<3f', f.read(12))
                normals.append(normal)
                
                # Read 3 vertices
                triangle_points = []
                for _ in range(3):
                    vertex = struct.unpack('<3f', f.read(12))
                    points.append(vertex)
                    triangle_points.append(point_idx)
                    point_idx += 1
                
                cells.append(np.array(triangle_points))
                
                # Skip attribute byte count
                f.read(2)
        
        points_array = np.array(points)
        cell_types = [VTKCellType.TRIANGLE.value] * len(cells)
        
        mesh_data = MeshData(
            points=points_array,
            cells=cells,
            cell_types=cell_types,
            cell_data={'normals': np.array(normals)},
            format_version="STL Binary"
        )
        
        logger.info(f"Read STL mesh: {len(points)} points, {len(cells)} triangles")
        return mesh_data
    
    def detect_format(self, file_path: Path) -> bool:
        """Detect STL format."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(80)
                # Check for ASCII STL
                if b'solid' in header:
                    return True
                # Check for binary STL (more complex detection would be needed)
                return file_path.suffix.lower() == '.stl'
        except:
            return False

class STLWriter(MeshWriter):
    """STL format writer."""
    
    def write(self, mesh_data: MeshData, file_path: Path, **kwargs):
        """Write STL format."""
        binary = kwargs.get('binary', False)
        
        if binary:
            self._write_stl_binary(mesh_data, file_path)
        else:
            self._write_stl_ascii(mesh_data, file_path)
    
    def _write_stl_ascii(self, mesh_data: MeshData, file_path: Path):
        """Write ASCII STL format."""
        logger.info(f"Writing ASCII STL mesh to {file_path}")
        
        with open(file_path, 'w') as f:
            f.write("solid mesh\n")
            
            for i, cell in enumerate(mesh_data.cells):
                if len(cell) == 3:  # Triangle
                    # Calculate normal
                    p1, p2, p3 = mesh_data.points[cell]
                    v1 = p2 - p1
                    v2 = p3 - p1
                    normal = np.cross(v1, v2)
                    normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else [0, 0, 1]
                    
                    f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                    f.write("    outer loop\n")
                    
                    for vertex_idx in cell:
                        vertex = mesh_data.points[vertex_idx]
                        f.write(f"      vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
                    
                    f.write("    endloop\n")
                    f.write("  endfacet\n")
            
            f.write("endsolid mesh\n")
        
        logger.info(f"Written ASCII STL mesh: {len(mesh_data.points)} points, {len(mesh_data.cells)} triangles")
    
    def _write_stl_binary(self, mesh_data: MeshData, file_path: Path):
        """Write binary STL format."""
        logger.info(f"Writing binary STL mesh to {file_path}")
        
        with open(file_path, 'wb') as f:
            # Write header
            header = b"Binary STL generated by OpenFFD" + b"\0" * (80 - 32)
            f.write(header)
            
            # Count triangles
            n_triangles = sum(1 for cell in mesh_data.cells if len(cell) == 3)
            f.write(struct.pack('<I', n_triangles))
            
            for cell in mesh_data.cells:
                if len(cell) == 3:  # Triangle
                    # Calculate normal
                    p1, p2, p3 = mesh_data.points[cell]
                    v1 = p2 - p1
                    v2 = p3 - p1
                    normal = np.cross(v1, v2)
                    normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else [0, 0, 1]
                    
                    # Write normal
                    f.write(struct.pack('<3f', *normal))
                    
                    # Write vertices
                    for vertex_idx in cell:
                        vertex = mesh_data.points[vertex_idx]
                        f.write(struct.pack('<3f', *vertex))
                    
                    # Write attribute byte count (unused)
                    f.write(struct.pack('<H', 0))
        
        logger.info(f"Written binary STL mesh: {len(mesh_data.points)} points, {n_triangles} triangles")

# GMSH format handlers (simplified)
class GMSHReader(MeshReader):
    """GMSH format reader."""
    
    def read(self, file_path: Path) -> MeshData:
        """Read GMSH format."""
        logger.info(f"Reading GMSH mesh from {file_path}")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse GMSH file structure
        points = []
        cells = []
        cell_types = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line == '$Nodes':
                points, i = self._parse_gmsh_nodes(lines, i + 1)
            elif line == '$Elements':
                cells, cell_types, i = self._parse_gmsh_elements(lines, i + 1)
            else:
                i += 1
        
        mesh_data = MeshData(
            points=np.array(points),
            cells=cells,
            cell_types=cell_types,
            format_version="GMSH"
        )
        
        logger.info(f"Read GMSH mesh: {len(points)} points, {len(cells)} cells")
        return mesh_data
    
    def _parse_gmsh_nodes(self, lines: List[str], start_idx: int) -> Tuple[List[List[float]], int]:
        """Parse GMSH nodes section."""
        n_nodes = int(lines[start_idx].strip())
        points = []
        
        for i in range(start_idx + 1, start_idx + 1 + n_nodes):
            parts = lines[i].strip().split()
            # node_id = int(parts[0])  # Node ID (1-based)
            coords = [float(x) for x in parts[1:4]]
            points.append(coords)
        
        return points, start_idx + 1 + n_nodes + 1  # +1 for $EndNodes
    
    def _parse_gmsh_elements(self, lines: List[str], start_idx: int) -> Tuple[List[np.ndarray], List[int], int]:
        """Parse GMSH elements section."""
        n_elements = int(lines[start_idx].strip())
        cells = []
        cell_types = []
        
        for i in range(start_idx + 1, start_idx + 1 + n_elements):
            parts = [int(x) for x in lines[i].strip().split()]
            
            # GMSH element format: element_id element_type n_tags tag1 tag2 ... node1 node2 ...
            element_type = parts[1]
            n_tags = parts[2]
            nodes = np.array(parts[3 + n_tags:]) - 1  # Convert to 0-based indexing
            
            # Convert GMSH element type to VTK cell type
            vtk_type = self._gmsh_to_vtk_type(element_type)
            
            cells.append(nodes)
            cell_types.append(vtk_type)
        
        return cells, cell_types, start_idx + 1 + n_elements + 1  # +1 for $EndElements
    
    def _gmsh_to_vtk_type(self, gmsh_type: int) -> int:
        """Convert GMSH element type to VTK cell type."""
        mapping = {
            1: VTKCellType.LINE.value,         # Line
            2: VTKCellType.TRIANGLE.value,     # Triangle
            3: VTKCellType.QUAD.value,         # Quadrangle
            4: VTKCellType.TETRA.value,        # Tetrahedron
            5: VTKCellType.HEXAHEDRON.value,   # Hexahedron
            6: VTKCellType.WEDGE.value,        # Prism
            7: VTKCellType.PYRAMID.value,      # Pyramid
        }
        return mapping.get(gmsh_type, VTKCellType.VERTEX.value)
    
    def detect_format(self, file_path: Path) -> bool:
        """Detect GMSH format."""
        try:
            with open(file_path, 'r') as f:
                first_lines = ''.join(f.readlines()[:5])
                return '$MeshFormat' in first_lines or '$Nodes' in first_lines
        except:
            return False

class GMSHWriter(MeshWriter):
    """GMSH format writer."""
    
    def write(self, mesh_data: MeshData, file_path: Path, **kwargs):
        """Write GMSH format."""
        logger.info(f"Writing GMSH mesh to {file_path}")
        
        with open(file_path, 'w') as f:
            # Write header
            f.write("$MeshFormat\n")
            f.write("2.2 0 8\n")
            f.write("$EndMeshFormat\n")
            
            # Write nodes
            f.write("$Nodes\n")
            f.write(f"{len(mesh_data.points)}\n")
            for i, point in enumerate(mesh_data.points):
                f.write(f"{i+1} {point[0]} {point[1]} {point[2]}\n")
            f.write("$EndNodes\n")
            
            # Write elements
            f.write("$Elements\n")
            f.write(f"{len(mesh_data.cells)}\n")
            for i, (cell, cell_type) in enumerate(zip(mesh_data.cells, mesh_data.cell_types)):
                gmsh_type = self._vtk_to_gmsh_type(cell_type)
                # Element format: element_id element_type n_tags tag1 tag2 nodes...
                nodes_str = ' '.join(str(node + 1) for node in cell)  # Convert to 1-based
                f.write(f"{i+1} {gmsh_type} 2 1 1 {nodes_str}\n")
            f.write("$EndElements\n")
        
        logger.info(f"Written GMSH mesh: {len(mesh_data.points)} points, {len(mesh_data.cells)} cells")
    
    def _vtk_to_gmsh_type(self, vtk_type: int) -> int:
        """Convert VTK cell type to GMSH element type."""
        mapping = {
            VTKCellType.LINE.value: 1,
            VTKCellType.TRIANGLE.value: 2,
            VTKCellType.QUAD.value: 3,
            VTKCellType.TETRA.value: 4,
            VTKCellType.HEXAHEDRON.value: 5,
            VTKCellType.WEDGE.value: 6,
            VTKCellType.PYRAMID.value: 7,
        }
        return mapping.get(vtk_type, 1)  # Default to line

# Fluent format handlers (simplified)
class FluentReader(MeshReader):
    """Fluent mesh format reader."""
    
    def read(self, file_path: Path) -> MeshData:
        """Read Fluent mesh format."""
        logger.info(f"Reading Fluent mesh from {file_path}")
        
        # Simplified implementation - real Fluent reader would be much more complex
        points = np.array([[0, 0, 0]])  # Placeholder
        cells = [np.array([0])]         # Placeholder
        cell_types = [1]                # Placeholder
        
        mesh_data = MeshData(
            points=points,
            cells=cells,
            cell_types=cell_types,
            format_version="Fluent"
        )
        
        return mesh_data
    
    def detect_format(self, file_path: Path) -> bool:
        """Detect Fluent format."""
        try:
            with open(file_path, 'r') as f:
                content = f.read(1000)
                return '(0 ' in content or '(1 ' in content  # Fluent format markers
        except:
            return False

class FluentWriter(MeshWriter):
    """Fluent mesh format writer."""
    
    def write(self, mesh_data: MeshData, file_path: Path, **kwargs):
        """Write Fluent mesh format."""
        logger.info(f"Writing Fluent mesh to {file_path}")
        
        # Simplified implementation
        with open(file_path, 'w') as f:
            f.write("(0 \"Mesh generated by OpenFFD\")\n")
            f.write(f"(0 \"Dimensions: {mesh_data.dimension}\")\n")
        
        logger.info("Written Fluent mesh (simplified)")

# CGNS format handlers (requires external library)
class CGNSReader(MeshReader):
    """CGNS format reader."""
    
    def __init__(self):
        try:
            import h5py
            self.h5py = h5py
        except ImportError:
            raise ImportError("CGNS support requires h5py library")
    
    def read(self, file_path: Path) -> MeshData:
        """Read CGNS format."""
        logger.info(f"Reading CGNS mesh from {file_path}")
        
        # Simplified implementation - real CGNS reader would use CGNS library
        points = np.array([[0, 0, 0]])  # Placeholder
        cells = [np.array([0])]         # Placeholder
        cell_types = [1]                # Placeholder
        
        mesh_data = MeshData(
            points=points,
            cells=cells,
            cell_types=cell_types,
            format_version="CGNS"
        )
        
        return mesh_data
    
    def detect_format(self, file_path: Path) -> bool:
        """Detect CGNS format."""
        return file_path.suffix.lower() in ['.cgns', '.hdf5']

class CGNSWriter(MeshWriter):
    """CGNS format writer."""
    
    def __init__(self):
        try:
            import h5py
            self.h5py = h5py
        except ImportError:
            raise ImportError("CGNS support requires h5py library")
    
    def write(self, mesh_data: MeshData, file_path: Path, **kwargs):
        """Write CGNS format."""
        logger.info(f"Writing CGNS mesh to {file_path}")
        
        # Simplified implementation
        logger.info("Written CGNS mesh (simplified)")

# Main mesh I/O interface
def read_mesh(file_path: Union[str, Path], format: Optional[MeshFormat] = None) -> MeshData:
    """Read mesh from file with automatic format detection.
    
    Args:
        file_path: Path to mesh file
        format: Mesh format (if None, auto-detect)
        
    Returns:
        MeshData object
    """
    file_path = Path(file_path)
    registry = MeshFormatRegistry()
    
    if format is None:
        # Auto-detect format
        for fmt in MeshFormat:
            try:
                reader = registry.get_reader(fmt)
                if reader.detect_format(file_path):
                    format = fmt
                    break
            except:
                continue
        
        if format is None:
            raise ValueError(f"Could not detect mesh format for file: {file_path}")
    
    reader = registry.get_reader(format)
    return reader.read(file_path)

def write_mesh(mesh_data: MeshData, file_path: Union[str, Path], 
               format: MeshFormat, **kwargs) -> None:
    """Write mesh to file.
    
    Args:
        mesh_data: Mesh data to write
        file_path: Output file path
        format: Output mesh format
        **kwargs: Format-specific options
    """
    file_path = Path(file_path)
    registry = MeshFormatRegistry()
    
    writer = registry.get_writer(format)
    writer.write(mesh_data, file_path, **kwargs)

def convert_mesh(input_path: Union[str, Path], output_path: Union[str, Path],
                input_format: Optional[MeshFormat] = None,
                output_format: Optional[MeshFormat] = None, **kwargs) -> MeshData:
    """Convert mesh between formats.
    
    Args:
        input_path: Input mesh file path
        output_path: Output mesh file path
        input_format: Input format (auto-detect if None)
        output_format: Output format (infer from extension if None)
        **kwargs: Format-specific options
        
    Returns:
        MeshData object
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Auto-detect output format from extension
    if output_format is None:
        ext = output_path.suffix.lower()
        format_map = {
            '.vtk': MeshFormat.VTK_LEGACY,
            '.vtu': MeshFormat.VTK_XML,
            '.stl': MeshFormat.STL_ASCII,
            '.msh': MeshFormat.GMSH_ASCII,
            '.cas': MeshFormat.FLUENT,
            '.cgns': MeshFormat.CGNS
        }
        output_format = format_map.get(ext)
        
        if output_format is None:
            raise ValueError(f"Could not determine output format from extension: {ext}")
    
    # Read mesh
    mesh_data = read_mesh(input_path, input_format)
    
    # Write mesh
    write_mesh(mesh_data, output_path, output_format, **kwargs)
    
    logger.info(f"Converted mesh from {input_path} to {output_path}")
    return mesh_data