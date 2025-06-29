"""
FluentMeshReader class to read Fluent mesh files (.cas, .msh) and extract zone information.

This module provides functionality to read and process Fluent mesh files,
extracting points, faces, and zone information.
"""

import logging
import os
import pathlib
import re
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)

class FluentZoneType(Enum):
    """Types of zones in Fluent meshes."""
    VOLUME = auto()
    BOUNDARY = auto()
    INTERFACE = auto()
    UNKNOWN = auto()

class FluentFace:
    """Fluent mesh face.
    
    Attributes:
        face_id (int): Unique identifier for the face
        face_type (str): Type of face (e.g., 'triangle', 'quad')
        point_indices (List[int]): Indices of points that make up the face
    """
    
    def __init__(self, face_id: int, face_type: str, point_indices: List[int]):
        """Initialize a FluentFace object.
        
        Args:
            face_id: Unique identifier for the face
            face_type: Type of face
            point_indices: Indices of points that make up the face
        """
        self.face_id = face_id
        self.face_type = face_type
        self.point_indices = point_indices
    
    def __str__(self) -> str:
        """String representation of the face.
        
        Returns:
            String representation
        """
        return f"Face {self.face_id}: {self.face_type} with {len(self.point_indices)} points"

class FluentCell:
    """Fluent mesh cell.
    
    Attributes:
        cell_id (int): Unique identifier for the cell
        cell_type (str): Type of cell (e.g., 'tetra', 'hexa')
        point_indices (List[int]): Indices of points that make up the cell
    """
    
    def __init__(self, cell_id: int, cell_type: str, point_indices: List[int]):
        """Initialize a FluentCell object.
        
        Args:
            cell_id: Unique identifier for the cell
            cell_type: Type of cell
            point_indices: Indices of points that make up the cell
        """
        self.cell_id = cell_id
        self.cell_type = cell_type
        self.point_indices = point_indices
    
    def __str__(self) -> str:
        """String representation of the cell.
        
        Returns:
            String representation
        """
        return f"Cell {self.cell_id}: {self.cell_type} with {len(self.point_indices)} points"

class FluentZone:
    """Fluent mesh zone.
    
    Attributes:
        zone_id (int): Numeric ID of the zone
        zone_type (str): Type of zone (e.g., 'wall', 'fluid', 'interior')
        name (str): Name of the zone
        zone_type_enum (FluentZoneType): Enumerated zone type
        faces (List[FluentFace]): List of faces in the zone
        cells (dict): Dictionary of cells in the zone, organized by cell type
        point_indices (set): Set of point indices belonging to this zone
    """
    
    def __init__(self, zone_id: int, zone_type: str, name: str, zone_type_enum: FluentZoneType):
        """Initialize a FluentZone object.
        
        Args:
            zone_id: Numeric ID of the zone
            zone_type: Type of zone (e.g., 'wall', 'fluid', 'interior')
            name: Name of the zone
            zone_type_enum: Enumerated zone type
        """
        self.zone_id = zone_id
        self.zone_type = zone_type
        self.name = name
        self.zone_type_enum = zone_type_enum
        self.faces = []
        self.cells = {}
        self.point_indices = set()
    
    def add_face(self, face: FluentFace) -> None:
        """Add a face to the zone.
        
        Args:
            face: Face to add
        """
        self.faces.append(face)
        # Add points from the face to this zone's points
        for point_idx in face.point_indices:
            self.point_indices.add(point_idx)
    
    def add_cell(self, cell: FluentCell) -> None:
        """Add a cell to the zone.
        
        Args:
            cell: Cell to add
        """
        cell_type = cell.cell_type
        if cell_type not in self.cells:
            self.cells[cell_type] = []
        self.cells[cell_type].append(cell)
        # Add points from the cell to this zone's points
        for point_idx in cell.point_indices:
            self.point_indices.add(point_idx)
    
    def add_cells(self, cell_data: np.ndarray, cell_type: str) -> None:
        """Add multiple cells to the zone at once.
        
        Args:
            cell_data: Array of cell connectivity data (N x M where N is number of cells, M is points per cell)
            cell_type: Type of cells (e.g., 'triangle', 'tetra')
        """
        if cell_type not in self.cells:
            self.cells[cell_type] = []
        
        for i, cell_points in enumerate(cell_data):
            cell = FluentCell(i, cell_type, cell_points)
            self.cells[cell_type].append(cell)
            # Add points from the cell to this zone's points
            for point_idx in cell_points:
                self.point_indices.add(point_idx)
    
    @property
    def num_faces(self) -> int:
        """Get the number of faces in the zone.
        
        Returns:
            Number of faces
        """
        return len(self.faces)
    
    @property
    def num_cells(self) -> int:
        """Get the number of cells in the zone.
        
        Returns:
            Number of cells
        """
        return sum(len(cells) for cells in self.cells.values())
    
    @property
    def num_points(self) -> int:
        """Get the number of unique points in the zone.
        
        Returns:
            Number of points
        """
        return len(self.point_indices)
    
    def __str__(self) -> str:
        """String representation of the zone.
        
        Returns:
            String representation
        """
        return f"Zone {self.zone_id}: {self.name} (Type: {self.zone_type}, Points: {self.num_points}, Cells: {self.num_cells})"

class FluentMesh:
    """Fluent mesh data structure.
    
    Attributes:
        points (numpy.ndarray): Array of point coordinates
        zones (List[FluentZone]): List of zones in the mesh
        faces (List[FluentFace]): List of faces in the mesh
        cells (List[FluentCell]): List of cells in the mesh
    """
    
    def __init__(self, points: np.ndarray, zones: List[FluentZone]):
        """Initialize a FluentMesh object.
        
        Args:
            points: Array of point coordinates
            zones: List of zones in the mesh
        """
        self.points = points
        self.zones = zones
        self.faces = []
        self.cells = []
    
    @property
    def num_points(self) -> int:
        """Get the number of points in the mesh.
        
        Returns:
            Number of points
        """
        return len(self.points)
    
    @property
    def num_zones(self) -> int:
        """Get the number of zones in the mesh.
        
        Returns:
            Number of zones
        """
        return len(self.zones)
    
    @property
    def num_faces(self) -> int:
        """Get the number of faces in the mesh.
        
        Returns:
            Number of faces
        """
        return len(self.faces)
    
    @property
    def num_cells(self) -> int:
        """Get the number of cells in the mesh.
        
        Returns:
            Number of cells
        """
        return len(self.cells)


# Try to import meshio, which is required to read Fluent mesh files
try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    logger.warning("meshio not available. Install with 'pip install meshio' for full mesh support.")


class FluentMeshReader:
    """Specialized reader for Fluent mesh files.
    
    This class reads Fluent mesh files (.msh, .cas) and extracts zone information.
    """
    
    # Class-level storage for captured zone specifications
    captured_zone_specs = []
    
    @classmethod
    def reset_captured_zones(cls):
        """Reset the captured zone specifications to an empty list."""
        cls.captured_zone_specs = []
    
    def __init__(self, filename: str, **kwargs):
        """Initialize the FluentMeshReader.
        
        Args:
            filename: Path to the Fluent mesh file
            **kwargs: Additional options (debug: bool, force_ascii: bool, force_binary: bool)
        """
        self.filename = filename
        
        # Extract options
        self.debug = kwargs.get('debug', False)
        self.force_ascii = kwargs.get('force_ascii', False)
        self.force_binary = kwargs.get('force_binary', False)
        
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def capture_zone_spec_warning(self, message, category, filename, lineno, file=None, line=None):
        """Custom warning handler that captures zone specifications from meshio warnings.
        
        Args:
            message: Warning message
            category: Warning category
            filename: Source filename
            lineno: Line number
            file: File object
            line: Line content
        """
        msg_str = str(message)
        
        # Check if this is a zone specification warning
        if "Zone specification not supported yet" in msg_str:
            # Extract zone type and name from the warning message
            match = re.search(r'\(([^,]+),\s*([^)]+)\)', msg_str)
            if match:
                zone_type = match.group(1).strip()
                zone_name = match.group(2).strip()
                FluentMeshReader.captured_zone_specs.append((zone_type, zone_name))
                logger.info(f"Captured zone specification: ({zone_type}, {zone_name})")
        
        # Call the original warning handler
        self.original_showwarning(message, category, filename, lineno, file, line)
    
    def read(self) -> FluentMesh:
        """Read the Fluent mesh file and extract points, zones, faces, and cells.
        
        Returns:
            FluentMesh object containing the mesh data
        """
        if not MESHIO_AVAILABLE:
            raise ImportError("meshio is required to read Fluent mesh files")
        
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Mesh file not found: {self.filename}")
        
        logger.info(f"Reading Fluent mesh file using meshio: {self.filename}")
        
        # Reset any previously captured zone specifications
        FluentMeshReader.reset_captured_zones()
        
        # Set our custom warning handler to capture zone specifications
        self.original_showwarning = warnings.showwarning
        warnings.showwarning = self.capture_zone_spec_warning
        
        try:
            # Read the mesh file using meshio
            mesh = meshio.read(self.filename)
            
            # Extract points and zones
            points = mesh.points
            zones = []
            
            # Process captured zone specifications
            logger.info(f"Processing {len(FluentMeshReader.captured_zone_specs)} captured zone specifications")
            
            # Map zone types to zone objects
            zone_id = 1
            for zone_type, zone_name in FluentMeshReader.captured_zone_specs:
                # Create a sanitized zone name
                sanitized_name = f"{zone_type}_{re.sub(r'[\s-]', '_', zone_name)}"
                
                # Determine zone type enum
                if zone_type.lower() in ['interior', 'fluid']:
                    zone_type_enum = FluentZoneType.VOLUME
                elif zone_type.lower() in ['wall', 'symmetry', 'pressure-outlet', 'velocity-inlet', 
                                        'pressure-inlet', 'mass-flow-inlet', 'axis']:
                    zone_type_enum = FluentZoneType.BOUNDARY
                elif zone_type.lower() in ['interface', 'periodic', 'fan', 'porous-jump']:
                    zone_type_enum = FluentZoneType.INTERFACE
                else:
                    zone_type_enum = FluentZoneType.UNKNOWN
                
                # Create a zone object
                zone = FluentZone(zone_id, zone_type.lower(), sanitized_name, zone_type_enum)
                zones.append(zone)
                zone_id += 1
                logger.info(f"Created zone: {zone}")
            
            logger.info(f"Successfully read mesh using meshio with {len(points)} points")
            
            # Create the FluentMesh object
            fluent_mesh = FluentMesh(points, zones)
            
            # Add mesh cells to zones
            if hasattr(mesh, 'cells') and mesh.cells:
                # Try to assign cells to zones based on cell type and common patterns
                for cell_block in mesh.cells:
                    cell_type = cell_block.type
                    cells = cell_block.data
                    
                    # Assign cells to appropriate zones
                    if cell_type in ['triangle', 'quad'] and zones:
                        # Surface elements typically belong to boundary zones
                        boundary_zones = [z for z in zones if z.zone_type_enum == FluentZoneType.BOUNDARY]
                        if boundary_zones:
                            # Distribute cells evenly among boundary zones for now
                            cells_per_zone = len(cells) // len(boundary_zones) if len(boundary_zones) > 0 else 0
                            for i, zone in enumerate(boundary_zones):
                                start_idx = i * cells_per_zone
                                end_idx = (i + 1) * cells_per_zone if i < len(boundary_zones) - 1 else len(cells)
                                zone.add_cells(cells[start_idx:end_idx], cell_type)
                    
                    elif cell_type in ['tetra', 'hexahedron', 'wedge', 'pyramid'] and zones:
                        # Volume elements typically belong to volume zones
                        volume_zones = [z for z in zones if z.zone_type_enum == FluentZoneType.VOLUME]
                        if volume_zones:
                            # Assign all volume cells to the first volume zone for now
                            volume_zones[0].add_cells(cells, cell_type)
            
            # Return the mesh object
            logger.info(f"Successfully loaded {len(points)} points and {len(zones)} zones")
            logger.info(f"Found {fluent_mesh.num_faces} faces across {len(fluent_mesh.zones)} zones")
            
            return fluent_mesh
        
        except Exception as e:
            logger.error(f"Error reading mesh with meshio: {e}")
            raise
        
        finally:
            # Restore the original warning handler
            warnings.showwarning = self.original_showwarning
    
    def _create_artificial_mesh(self) -> None:
        """Create an artificial mesh for testing purposes when mesh reading fails."""
        logger.warning("Creating artificial test mesh")
        
        # Create a simple 3D grid of points
        x, y, z = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
        points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
        
        # Create an empty FluentMesh object
        zones = []
        
        # Create a single zone
        zone = FluentZone(1, "wall", "artificial_wall", FluentZoneType.BOUNDARY)
        zones.append(zone)
        
        return FluentMesh(points, zones)
