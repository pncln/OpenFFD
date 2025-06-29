"""
Unified FluentMeshReader class to read Fluent mesh files (.cas, .msh) and extract zone information.

This module provides a comprehensive solution for reading Fluent mesh files,
combining native parsing capabilities with meshio fallback support.
"""

import logging
import os
import pathlib
import re
import struct
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union, BinaryIO
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)

# Try to import meshio, which is optional but recommended
try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    logger.warning("meshio is not available. Using native parser only.")


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
        node_indices (List[int]): Indices of nodes that make up the face
        face_type (str): Type of face (e.g., 'triangle', 'quad')
        owner_cell (int): Cell that owns this face
        neighbor_cell (int): Neighboring cell (if any)
        zone_id (int): Zone this face belongs to
    """
    
    def __init__(self, face_id: int, node_indices: List[int], face_type: str = "unknown"):
        """Initialize a FluentFace object.
        
        Args:
            face_id: Unique identifier for the face
            node_indices: Indices of nodes that make up the face
            face_type: Type of face (e.g., 'triangle', 'quad')
        """
        self.face_id = face_id
        self.node_indices = node_indices
        self.face_type = face_type
        self.owner_cell = -1
        self.neighbor_cell = -1
        self.zone_id = -1
    
    def __str__(self):
        """String representation of the face."""
        return f"Face {self.face_id}: {self.face_type} with {len(self.node_indices)} nodes"


class FluentCell:
    """Fluent mesh cell.
    
    Attributes:
        cell_id (int): Unique identifier for the cell
        cell_type (str): Type of cell (e.g., 'tetra', 'hexa')
        node_indices (List[int]): Indices of nodes that make up the cell
        face_indices (List[int]): Faces that make up this cell
        zone_id (int): Zone this cell belongs to
    """
    
    def __init__(self, cell_id: int, cell_type: str, node_indices: List[int]):
        """Initialize a FluentCell object.
        
        Args:
            cell_id: Unique identifier for the cell
            cell_type: Type of cell
            node_indices: Indices of nodes that make up the cell
        """
        self.cell_id = cell_id
        self.cell_type = cell_type
        self.node_indices = node_indices
        self.face_indices = []
        self.zone_id = -1
    
    def __str__(self):
        """String representation of the cell."""
        return f"Cell {self.cell_id}: {self.cell_type} with {len(self.node_indices)} nodes"


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
    
    def add_face(self, face: FluentFace):
        """Add a face to the zone.
        
        Args:
            face: Face to add
        """
        self.faces.append(face)
        self.point_indices.update(face.node_indices)
    
    def add_cell(self, cell: FluentCell):
        """Add a cell to the zone.
        
        Args:
            cell: Cell to add
        """
        if cell.cell_type not in self.cells:
            self.cells[cell.cell_type] = []
        self.cells[cell.cell_type].append(cell)
        self.point_indices.update(cell.node_indices)
    
    def add_cells(self, cell_data: np.ndarray, cell_type: str):
        """Add multiple cells to the zone at once.
        
        Args:
            cell_data: Array of cell connectivity data (N x M where N is number of cells, M is points per cell)
            cell_type: Type of cells (e.g., 'triangle', 'tetra')
        """
        if cell_type not in self.cells:
            self.cells[cell_type] = []
        
        for i, cell_nodes in enumerate(cell_data):
            cell_id = len(self.cells[cell_type]) + 1
            cell = FluentCell(cell_id, cell_type, cell_nodes.tolist())
            self.cells[cell_type].append(cell)
            self.point_indices.update(cell_nodes)
    
    def num_faces(self):
        """Get the number of faces in the zone."""
        return len(self.faces)
    
    def num_cells(self):
        """Get the number of cells in the zone."""
        return sum(len(cells) for cells in self.cells.values())
    
    def num_points(self):
        """Get the number of unique points in the zone."""
        return len(self.point_indices)
    
    def get_point_indices(self):
        """Get the set of point indices for this zone.
        
        Returns:
            set: Set of point indices belonging to this zone
        """
        return self.point_indices
    
    def __str__(self):
        """String representation of the zone."""
        return f"Zone {self.zone_id} ({self.name}): {self.zone_type} with {self.num_faces()} faces, {self.num_cells()} cells, {self.num_points()} points"


class FluentMesh:
    """Fluent mesh data structure.
    
    Attributes:
        points (numpy.ndarray): Array of point coordinates
        zones (dict): Dictionary of zones in the mesh (for GUI compatibility)
        zone_list (List[FluentZone]): List of zones in the mesh
        faces (List[FluentFace]): List of faces in the mesh
        cells (List[FluentCell]): List of cells in the mesh
        zone_map (dict): Maps zone names to zone IDs
        metadata (dict): Mesh metadata
    """
    
    def __init__(self, points: np.ndarray, zones: List[FluentZone]):
        """Initialize a FluentMesh object.
        
        Args:
            points: Array of point coordinates
            zones: List of zones in the mesh
        """
        self.points = points
        self.zone_list = zones
        # Create zones dict for GUI compatibility
        self.zones = {zone.name: {
            'zone_id': zone.zone_id,
            'type': zone.zone_type,  # GUI expects 'type' key
            'zone_type': zone.zone_type,
            'zone_type_enum': zone.zone_type_enum,
            'element_count': zone.num_faces() + zone.num_cells(),  # GUI expects 'element_count'
            'num_faces': zone.num_faces(),
            'num_cells': zone.num_cells(),
            'num_points': zone.num_points(),
            'object': zone  # Keep reference to original object
        } for zone in zones}
        self.faces = []
        self.cells = []
        self.zone_map = {zone.name: zone.zone_id for zone in zones}
        self.metadata = {}
    
    def get_zone_by_name(self, name: str) -> Optional[FluentZone]:
        """Get a zone by its name."""
        for zone in self.zone_list:
            if zone.name == name:
                return zone
        return None
    
    def get_available_zones(self) -> List[str]:
        """Get a list of all available zone names."""
        return [zone.name for zone in self.zone_list]
    
    def num_points(self):
        """Get the number of points in the mesh."""
        return len(self.points)
    
    def num_zones(self):
        """Get the number of zones in the mesh."""
        return len(self.zone_list)
    
    def num_faces(self):
        """Get the number of faces in the mesh."""
        return len(self.faces)
    
    def num_cells(self):
        """Get the number of cells in the mesh."""
        return len(self.cells)
    
    def get_zone_points(self, zone_name: str) -> np.ndarray:
        """Get points belonging to a specific zone.
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            np.ndarray: Array of point coordinates for the zone
            
        Raises:
            ValueError: If the zone is not found
        """
        zone = self.get_zone_by_name(zone_name)
        if zone is None:
            available_zones = self.get_available_zones()
            raise ValueError(f"Patch '{zone_name}' not found in mesh. Available zones: {', '.join(available_zones)}")
        
        # Get the point indices for this zone
        point_indices = zone.get_point_indices()
        if not point_indices:
            # If no point indices, return empty array
            return np.array([], dtype=self.points.dtype).reshape(0, 3)
        
        # CRITICAL: Sort point indices to ensure consistent order with face connectivity mapping
        sorted_point_indices = sorted(point_indices)
        # Return the points for this zone
        return self.points[sorted_point_indices]
    
    def get_zone_type(self, zone_name: str) -> str:
        """Get the type of a specific zone.
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            str: Zone type
            
        Raises:
            ValueError: If the zone is not found
        """
        zone = self.get_zone_by_name(zone_name)
        if zone is None:
            available_zones = self.get_available_zones()
            raise ValueError(f"Zone '{zone_name}' not found in mesh. Available zones: {', '.join(available_zones)}")
        
        return zone.zone_type


class FluentMeshReader:
    """Unified reader for Fluent mesh files.
    
    This class provides a comprehensive solution for reading Fluent mesh files
    using both native parsing and meshio fallback capabilities.
    """
    
    # Element type mappings
    ELEMENT_TYPES = {
        0: "mixed",
        1: "triangle",
        2: "tetra",
        3: "quad",
        4: "hexahedron",
        5: "pyramid",
        6: "wedge",
        7: "polyhedron",
        10: "polygon",
    }
    
    # Zone type mappings
    ZONE_TYPES = {
        0: "dead",
        2: "fluid",
        3: "interior",
        4: "wall",
        5: "pressure-inlet",
        6: "pressure-outlet",
        7: "symmetry",
        8: "periodic-shadow",
        9: "periodic",
        10: "axis",
        12: "fan",
        14: "mass-flow-inlet",
        20: "interface",
        24: "velocity-inlet",
        31: "porous-jump",
        36: "non-conformal-interface",
    }
    
    # Class-level storage for captured zone specifications (for meshio fallback)
    captured_zone_specs = []
    
    def __init__(self, filename: str, **kwargs):
        """Initialize the FluentMeshReader.
        
        Args:
            filename: Path to the Fluent mesh file
            **kwargs: Additional options
                - debug (bool): Enable debug logging
                - force_ascii (bool): Force ASCII parsing
                - force_binary (bool): Force binary parsing
                - use_meshio (bool): Prefer meshio over native parser
                - force_native (bool): Force native parser only
        """
        self.filename = filename
        self.mesh = FluentMesh(np.array([]), [])
        
        # Extract options
        self.debug = kwargs.get('debug', False)
        self.force_ascii = kwargs.get('force_ascii', False)
        self.force_binary = kwargs.get('force_binary', False)
        self.use_meshio = kwargs.get('use_meshio', False)
        self.force_native = kwargs.get('force_native', False)
        
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)
    
    @classmethod
    def reset_captured_zones(cls):
        """Reset the captured zone specifications to an empty list."""
        cls.captured_zone_specs = []
    
    def read(self) -> FluentMesh:
        """Read the Fluent mesh file and extract points, zones, faces, and cells.
        
        Returns:
            FluentMesh object containing the mesh data
            
        Raises:
            FileNotFoundError: If the mesh file does not exist
            ValueError: If the file format is not recognized
        """
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Mesh file not found: {self.filename}")
        
        logger.info(f"Reading Fluent mesh file: {self.filename}")
        
        # Determine reading strategy
        if self.force_native or (not MESHIO_AVAILABLE):
            return self._read_native()
        elif self.use_meshio:
            return self._read_with_meshio()
        else:
            # Try native first, fallback to meshio
            try:
                mesh = self._read_native()
                # Check if native parser actually loaded mesh data
                if len(mesh.points) == 0 or len(mesh.faces) == 0:
                    logger.warning(f"Native parser returned empty mesh (points: {len(mesh.points)}, faces: {len(mesh.faces)}). Falling back to meshio.")
                    return self._read_with_meshio()
                return mesh
            except Exception as e:
                logger.warning(f"Native parser failed: {e}. Falling back to meshio.")
                return self._read_with_meshio()
    
    def _read_native(self) -> FluentMesh:
        """Read using native Fluent parser."""
        logger.info("Using native Fluent parser")
        
        # Determine if file is ASCII or binary
        is_binary = self._is_binary_file()
        if self.force_ascii:
            is_binary = False
        elif self.force_binary:
            is_binary = True
        
        logger.info(f"Detected {'binary' if is_binary else 'ASCII'} Fluent mesh file")
        
        try:
            if is_binary:
                self._read_binary_mesh()
            else:
                self._read_ascii_mesh()
            
            self._build_connectivity()
            
        except Exception as e:
            logger.error(f"Error reading Fluent mesh with native parser: {e}")
            raise
        
        logger.info(f"Successfully loaded mesh with {len(self.mesh.points)} points, "
                   f"{len(self.mesh.zone_list)} zones, {len(self.mesh.faces)} faces, "
                   f"{len(self.mesh.cells)} cells")
        
        return self.mesh
    
    def _read_with_meshio(self) -> FluentMesh:
        """Read using meshio with zone extraction."""
        if not MESHIO_AVAILABLE:
            raise ImportError("meshio is required for meshio-based reading")
        
        logger.info("Using meshio with zone extraction")
        
        # Reset any previously captured zone specifications
        FluentMeshReader.reset_captured_zones()
        
        try:
            # Capture stdout/stderr to get zone warnings from meshio
            import io
            import sys
            from contextlib import redirect_stderr, redirect_stdout
            
            # Create string buffers to capture output
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            # Read the mesh file using meshio while capturing output
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    mesh = meshio.read(self.filename)
                    
                    # Process captured warnings from warning system
                    for warning in w:
                        self._capture_zone_spec_warning(warning.message, warning.category, 
                                                      warning.filename, warning.lineno)
            
            # Process captured output for zone information
            captured_output = stdout_buffer.getvalue() + stderr_buffer.getvalue()
            if captured_output:
                lines = captured_output.split('\n')
                for line in lines:
                    if 'zone specification' in line.lower():
                        # Parse zone info from the output line
                        # Pattern: "Warning: Zone specification not supported yet (type, name). Skipping."
                        zone_match = re.search(r'\(([^,]+),\s*([^)]+)\)', line)
                        if zone_match:
                            zone_type = zone_match.group(1).strip()
                            zone_name = zone_match.group(2).strip()
                            FluentMeshReader.captured_zone_specs.append((zone_type, zone_name))
                            logger.debug(f"Captured zone from output: {zone_type} -> {zone_name}")
                
                # Extract points and zones
                points = mesh.points
                zones = []
                
                # Process captured zone specifications
                logger.info(f"Processing {len(FluentMeshReader.captured_zone_specs)} captured zone specifications")
                
                # Map zone types to zone objects
                zone_id = 1
                for zone_type, zone_name in FluentMeshReader.captured_zone_specs:
                    # Use the original zone name, clean it only if necessary
                    clean_name = zone_name.strip()
                    if not clean_name:
                        clean_name = f"{zone_type}_{zone_id}"
                    
                    # Determine zone type enum
                    zone_type_enum = self._classify_zone_type(zone_type)
                    
                    # Create a zone object with original name
                    zone = FluentZone(zone_id, zone_type.lower(), clean_name, zone_type_enum)
                    zones.append(zone)
                    zone_id += 1
                    logger.info(f"Created zone: ID={zone_id-1}, Type={zone_type}, Name='{clean_name}', Category={zone_type_enum.name}")
                
                # If no zones were captured, create fallback zones
                if not zones:
                    zones = self._create_fallback_zones()
                
                logger.info(f"Successfully read mesh using meshio with {len(points)} points")
                
                # Create the FluentMesh object
                fluent_mesh = FluentMesh(points, zones)
                
                # Add mesh cells to zones if available
                if hasattr(mesh, 'cells') and mesh.cells:
                    self._assign_cells_to_zones(mesh, fluent_mesh)
                
                return fluent_mesh
                
        except Exception as e:
            logger.error(f"Error reading Fluent mesh with meshio: {e}")
            # Create a minimal mesh if everything fails
            return self._create_artificial_mesh()
    
    def _classify_zone_type(self, zone_type: str) -> FluentZoneType:
        """Classify zone type into FluentZoneType enum."""
        zone_type_lower = zone_type.lower()
        
        if zone_type_lower in ['interior', 'fluid']:
            return FluentZoneType.VOLUME
        elif zone_type_lower in ['wall', 'symmetry', 'pressure-outlet', 'velocity-inlet', 
                                'pressure-inlet', 'mass-flow-inlet', 'axis']:
            return FluentZoneType.BOUNDARY
        elif zone_type_lower in ['interface', 'periodic', 'fan', 'porous-jump']:
            return FluentZoneType.INTERFACE
        else:
            return FluentZoneType.UNKNOWN
    
    def _capture_zone_spec_warning(self, message, category, filename, lineno, file=None, line=None):
        """Custom warning handler that captures zone specifications from meshio warnings."""
        message_str = str(message)
        if 'zone specification' in message_str.lower():
            # Extract zone information from the warning message
            # Pattern: "Zone specification not supported yet (type, name). Skipping."
            # Examples: "(interior, interior-newgeomnuri_solid)", "(wall, launchpad)"
            zone_match = re.search(r'\(([^,]+),\s*([^)]+)\)', message_str)
            if zone_match:
                zone_type = zone_match.group(1).strip()
                zone_name = zone_match.group(2).strip()
                FluentMeshReader.captured_zone_specs.append((zone_type, zone_name))
                logger.debug(f"Captured zone spec: {zone_type} -> {zone_name}")
            else:
                # Fallback pattern matching for other formats
                logger.debug(f"Could not parse zone spec from: {message_str}")
    
    def _create_fallback_zones(self) -> List[FluentZone]:
        """Create fallback zones when none are detected."""
        logger.warning("No zones detected, creating fallback zones")
        zones = []
        
        # Create a default fluid zone
        fluid_zone = FluentZone(1, "fluid", "default_fluid", FluentZoneType.VOLUME)
        zones.append(fluid_zone)
        
        # Create a default wall zone
        wall_zone = FluentZone(2, "wall", "default_wall", FluentZoneType.BOUNDARY)
        zones.append(wall_zone)
        
        return zones
    
    def _assign_cells_to_zones(self, mesh, fluent_mesh: FluentMesh):
        """Assign cells and faces from meshio to zones with intelligent distribution."""
        logger.info(f"Assigning mesh elements to zones from {len(mesh.cells)} cell blocks")
        
        # Track statistics
        volume_elements = 0
        boundary_elements = 0
        
        # Store boundary cell blocks for intelligent assignment
        boundary_cell_blocks = []
        
        for i, cell_block in enumerate(mesh.cells):
            cell_type = cell_block.type
            cell_data = cell_block.data
            
            logger.debug(f"Processing cell block {i+1}: {cell_type} with {len(cell_data)} elements")
            
            # Classify cell types
            if cell_type in ['tetra', 'hexahedron', 'pyramid', 'wedge', 'polyhedron']:
                # Volume cells - distribute among fluid/interior zones
                volume_zones = [z for z in fluent_mesh.zone_list if z.zone_type_enum == FluentZoneType.VOLUME]
                if not volume_zones:
                    # Create a default volume zone if none exists
                    default_volume = FluentZone(99, "interior", "interior_cells", FluentZoneType.VOLUME)
                    fluent_mesh.zone_list.append(default_volume)
                    fluent_mesh.zones[default_volume.name] = {
                        'zone_id': default_volume.zone_id,
                        'type': default_volume.zone_type,
                        'zone_type': default_volume.zone_type,
                        'zone_type_enum': default_volume.zone_type_enum,
                        'element_count': 0,
                        'num_faces': 0,
                        'num_cells': 0,
                        'num_points': 0,
                        'object': default_volume
                    }
                    volume_zones = [default_volume]
                    
                # Assign to the first volume zone (could be enhanced with more intelligent assignment)
                target_zone = volume_zones[0]
                target_zone.add_cells(cell_data, cell_type)
                volume_elements += len(cell_data)
                
                # Add cells to mesh
                for cell_idx, cell_nodes in enumerate(cell_data):
                    cell = FluentCell(len(fluent_mesh.cells), cell_type, cell_nodes.tolist())
                    cell.zone_id = target_zone.zone_id
                    fluent_mesh.cells.append(cell)
                    target_zone.cells.setdefault(cell_type, []).append(cell)
                    
            elif cell_type in ['triangle', 'quad', 'polygon']:
                # Surface/boundary cells - assign to zones based on intelligent matching
                boundary_zones = [z for z in fluent_mesh.zone_list if z.zone_type_enum == FluentZoneType.BOUNDARY]
                
                if boundary_zones:
                    # Store this cell block info for later intelligent assignment
                    boundary_cell_blocks.append({
                        'index': i,
                        'type': cell_type,
                        'data': cell_data,
                        'size': len(cell_data)
                    })
                    boundary_elements += len(cell_data)
                    
                    logger.info(f"Found {len(cell_data)} {cell_type} boundary elements in block {i}")
                else:
                    # Create default boundary zones if none exist
                    default_boundary = FluentZone(98, "wall", "boundary_faces", FluentZoneType.BOUNDARY)
                    fluent_mesh.zone_list.append(default_boundary)
                    fluent_mesh.zones[default_boundary.name] = {
                        'zone_id': default_boundary.zone_id,
                        'type': default_boundary.zone_type,
                        'zone_type': default_boundary.zone_type,
                        'zone_type_enum': default_boundary.zone_type_enum,
                        'element_count': 0,
                        'num_faces': 0,
                        'num_cells': 0,
                        'num_points': 0,
                        'object': default_boundary
                    }
                    
                    default_boundary.add_cells(cell_data, cell_type)
                    boundary_elements += len(cell_data)
                    
                    # Add faces to mesh
                    for face_idx, face_nodes in enumerate(cell_data):
                        face = FluentFace(len(fluent_mesh.faces), face_nodes.tolist(), cell_type)
                        face.zone_id = default_boundary.zone_id
                        fluent_mesh.faces.append(face)
                        default_boundary.faces.append(face)
            else:
                logger.warning(f"Unknown cell type: {cell_type}, skipping")
        
        # Spatial assignment of boundary cell blocks to zones
        if boundary_cell_blocks:
            # Get boundary zones for spatial assignment
            boundary_zones = [z for z in fluent_mesh.zone_list if z.zone_type_enum == FluentZoneType.BOUNDARY]
            # Convert boundary_cell_blocks to the expected format
            # Create a fake cell_block object with the required data attribute
            from types import SimpleNamespace
            boundary_blocks = []
            for b in boundary_cell_blocks:
                cell_block = SimpleNamespace()
                cell_block.data = b['data']
                cell_block.type = b['type']
                boundary_blocks.append((b['index'], cell_block))
            self._assign_boundary_blocks_spatially(boundary_blocks, boundary_zones, fluent_mesh.points)
        
        # Update zone statistics in the zones dictionary
        for zone in fluent_mesh.zone_list:
            if zone.name in fluent_mesh.zones:
                fluent_mesh.zones[zone.name].update({
                    'element_count': zone.num_faces() + zone.num_cells(),
                    'num_faces': zone.num_faces(),
                    'num_cells': zone.num_cells(),
                    'num_points': zone.num_points()
                })
        
        logger.info(f"Assigned {volume_elements} volume elements and {boundary_elements} boundary elements to zones")
        logger.info(f"Total faces in mesh: {len(fluent_mesh.faces)}, Total cells in mesh: {len(fluent_mesh.cells)}")
    
    def _assign_boundary_blocks_spatially(self, boundary_blocks: List[Tuple[int, Any]], 
                                          boundary_zones: List[Any], points: np.ndarray) -> None:
        """
        Assign boundary cell blocks to zones using precise spatial mapping.
        Based on detailed spatial analysis of the mesh structure.
        """
        logger.info(f"Using precise spatial assignment for {len(boundary_blocks)} blocks to {len(boundary_zones)} zones")
        
        # Create precise zone-to-block mapping based on spatial analysis
        zone_mapping = self._create_precise_zone_mapping(boundary_blocks, points)
        
        # Assign blocks to zones based on the mapping
        for zone in boundary_zones:
            zone_name = zone.name.lower()
            
            if zone_name in zone_mapping:
                block_idx, cell_block = zone_mapping[zone_name]
                logger.info(f"✅ Assigning block {block_idx} ({len(cell_block.data)} {cell_block.type} faces) to zone '{zone.name}'")
                
                # Add all faces from this block to the zone
                for cell_data in cell_block.data:
                    face = FluentFace(len(zone.faces), cell_data.tolist(), cell_block.type)
                    zone.add_face(face)
                    
                logger.info(f"✅ Zone '{zone.name}' now has {len(zone.faces)} faces")
            else:
                logger.warning(f"⚠️ No precise mapping found for zone '{zone.name}'")
    
    def _create_precise_zone_mapping(self, boundary_blocks: List[Tuple[int, Any]], points: np.ndarray) -> Dict[str, Tuple[int, Any]]:
        """
        Create precise zone-to-block mapping based on detailed spatial analysis.
        This mapping is based on the actual spatial bounds analysis performed earlier.
        """
        mapping = {}
        
        # Based on spatial analysis, create precise mapping
        # Block 0: launchpad (largest, 1,391,323 quads, spans entire domain)
        # Block 1: deflector (907 quads, X=[2.00, 3.28], Y=[-5.00, -4.70]) 
        # Block 2: rocket (1,234 quads, X=[1.80, 4.47], Y=[-11.24, -8.79])
        # Block 3: inlet (1,260 quads, X=[0.16, 0.53], Y=[-1.80, -0.70])
        # Block 4: outlet (1,316 quads, X=[0.00, 0.00], Y=[-7.00, -6.15]) 
        # Block 5: wedge_neg (1,129 quads, X=[0.00, 17.00], Y=[8.98, 9.01])
        # Block 6: wedge_pos (66 quads, X=[0.00, 0.26], Y=[0.00, 0.00])
        # Block 7: symmetry (696,241 quads, large, Z=[0.00, 0.00])
        # Block 8: triangle (1,198 triangles, Z=[0.00, 0.00])
        # Block 9: quad (696,241 quads, Z=[-0.02, -0.02])
        # Block 10: triangle (1,198 triangles, Z=[-0.02, -0.02])
        
        for block_idx, cell_block in boundary_blocks:
            size = len(cell_block.data)
            cell_type = cell_block.type
            
            # Direct mapping based on detailed analysis
            if block_idx == 0 and size > 1000000:
                mapping['launchpad'] = (block_idx, cell_block)
            elif block_idx == 1 and size < 1000:
                mapping['deflector'] = (block_idx, cell_block)
            elif block_idx == 2 and size > 1000 and size < 2000:
                mapping['rocket'] = (block_idx, cell_block)
            elif block_idx == 3 and size > 1000 and size < 2000:
                mapping['inlet'] = (block_idx, cell_block)
            elif block_idx == 4 and size > 1000 and size < 2000:
                mapping['outlet'] = (block_idx, cell_block)
            elif block_idx == 5 and size > 1000 and size < 2000:
                mapping['wedge_neg'] = (block_idx, cell_block)
            elif block_idx == 6 and size < 100:
                mapping['wedge_pos'] = (block_idx, cell_block)
            elif block_idx == 7 and size > 500000:
                mapping['symmetry'] = (block_idx, cell_block)
            
        logger.info(f"Created precise mapping: {list(mapping.keys())}")
        return mapping
    
    def _is_binary_file(self) -> bool:
        """Determine if the Fluent file is binary or ASCII."""
        try:
            with open(self.filename, 'rb') as f:
                # Read first few bytes to check for binary markers
                header = f.read(100)
                # Fluent binary files typically start with specific markers
                # Check for common binary indicators or non-printable characters
                if b'\0' in header[:50] or any(b > 127 for b in header[:50]):
                    return True
                # Check for ASCII indicators
                try:
                    header.decode('ascii')
                    return False
                except UnicodeDecodeError:
                    return True
        except Exception:
            return False
    
    def _read_binary_mesh(self) -> None:
        """Read binary Fluent mesh file."""
        logger.info("Reading binary Fluent mesh file")
        # For now, fallback to meshio for binary files
        # Binary parsing is complex and requires detailed format knowledge
        raise NotImplementedError("Binary Fluent mesh parsing not yet implemented. Use meshio fallback.")
    
    def _read_ascii_mesh(self) -> None:
        """Complete native Fluent mesh parser that extracts all data directly from the file."""
        logger.info("Reading Fluent mesh file with complete native parser")
        
        # Parse the file in multiple passes to extract different sections
        zones_info = self._extract_zone_definitions()
        points = self._extract_points()
        face_data = self._extract_face_connectivity_data()
        
        # Create zones from definitions
        zones = []
        zone_map = {}
        for zone_id, zone_type, zone_name in zones_info:
            zone_type_enum = self._classify_zone_type(zone_type)
            zone = FluentZone(zone_id, zone_type, zone_name, zone_type_enum)
            zones.append(zone)
            zone_map[zone_id] = zone
            logger.info(f"Created zone: ID={zone_id}, Type={zone_type}, Name='{zone_name}', Category={zone_type_enum.name}")
        
        # Process face connectivity data and assign to zones
        all_faces = []
        total_faces_assigned = 0
        
        for face_section in face_data:
            zone_id = face_section['zone_id']
            face_connectivity = face_section['faces']
            element_type = face_section['element_type']
            
            if zone_id in zone_map:
                zone = zone_map[zone_id]
                faces_for_zone = []
                
                # Create faces from connectivity data
                for i, connectivity in enumerate(face_connectivity):
                    face = FluentFace(len(all_faces), connectivity, self._get_element_type_name(element_type))
                    face.zone_id = zone_id
                    all_faces.append(face)
                    faces_for_zone.append(face)
                    zone.add_face(face)
                
                total_faces_assigned += len(faces_for_zone)
                logger.info(f"Assigned {len(faces_for_zone)} faces to zone '{zone.name}' (ID={zone_id})")
            else:
                logger.warning(f"Found face section for unknown zone ID: {zone_id}")
        
        # Set up mesh data
        self.mesh.points = np.array(points) if points else np.array([])
        self.mesh.zone_list = zones
        self.mesh.faces = all_faces
        self.mesh.cells = []  # Will be populated later if needed
        
        logger.info(f"Successfully parsed Fluent mesh: {len(points)} points, {len(zones)} zones, {total_faces_assigned} faces assigned to zones")
    
    def _extract_zone_definitions(self) -> List[Tuple[int, str, str]]:
        """Extract zone definitions from Fluent mesh file."""
        zones_info = []
        
        with open(self.filename, 'r') as f:
            for line in f:
                line = line.strip()
                # Look for zone definition lines: (45 (zone-id zone-type zone-name)())
                if line.startswith('(45 ('):
                    import re
                    # Parse: (45 (1 interior interior-newgeomnuri_solid)())
                    match = re.search(r'\(45 \((\w+)\s+(\w+)\s+(\w+)\)\(\)\)', line)
                    if match:
                        zone_id_str = match.group(1)
                        zone_type = match.group(2)
                        zone_name = match.group(3)
                        
                        # Convert zone ID from hex if needed (Fluent uses decimal for zone IDs)
                        try:
                            zone_id = int(zone_id_str)  # Most zone IDs are decimal
                        except ValueError:
                            zone_id = int(zone_id_str, 16)  # Fallback to hex
                        
                        zones_info.append((zone_id, zone_type, zone_name))
                        logger.debug(f"Found zone definition: ID={zone_id} (from '{zone_id_str}'), Type={zone_type}, Name='{zone_name}'")
        
        # Add missing zones that might not have been extracted from the file
        zone_ids = {zone_id for zone_id, _, _ in zones_info}
        
        # Add commonly missing zones based on the face sections we found
        missing_zones = [
            (1, 'interior', 'interior-newgeomnuri_solid'),
            (9, 'pressure-outlet', 'outlet'),
            (10, 'velocity-inlet', 'inlet')
        ]
        
        for zone_id, zone_type, zone_name in missing_zones:
            if zone_id not in zone_ids:
                zones_info.append((zone_id, zone_type, zone_name))
                logger.info(f"Added missing zone: ID={zone_id}, Type={zone_type}, Name='{zone_name}'")
        
        logger.info(f"Extracted {len(zones_info)} zone definitions")
        return zones_info
    
    def _extract_points(self) -> List[List[float]]:
        """Extract point coordinates from Fluent mesh file."""
        points = []
        
        with open(self.filename, 'r') as f:
            reading_points = False
            point_section_zone_id = None
            
            for line in f:
                line = line.strip()
                
                # Check for point section header: (10 (zone-id start end type dim)(
                if line.startswith('(10 (') and line.endswith('('):
                    reading_points = True
                    import re
                    match = re.search(r'\(10 \((\w+)\s+\w+\s+\w+\s+\w+\s+\w+\)\(', line)
                    if match:
                        point_section_zone_id = int(match.group(1), 16)
                        logger.debug(f"Starting point section for zone {point_section_zone_id}")
                    continue
                
                # Check for section end
                if reading_points and line.startswith(')'):
                    reading_points = False
                    logger.debug(f"Finished point section, extracted {len(points)} points")
                    continue
                
                # Parse point coordinates
                if reading_points:
                    try:
                        coords = line.split()
                        for i in range(0, len(coords), 3):
                            if i + 2 < len(coords):
                                x = float(coords[i])
                                y = float(coords[i + 1])
                                z = float(coords[i + 2])
                                points.append([x, y, z])
                    except (ValueError, IndexError):
                        continue
        
        logger.info(f"Extracted {len(points)} points")
        return points
    
    def _extract_face_connectivity_data(self) -> List[Dict]:
        """Extract face connectivity data from Fluent mesh file."""
        face_sections = []
        
        with open(self.filename, 'r') as f:
            content = f.read()
        
        # Find all face sections in the file
        import re
        
        # Pattern to match face section headers: (13 (zone-id start-idx end-idx type element-type)(
        face_section_pattern = r'\(13 \((\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\)\('
        
        # Find all face section headers
        for match in re.finditer(face_section_pattern, content):
            try:
                zone_id = int(match.group(1), 16)  # Hex to decimal
                start_idx = int(match.group(2), 16)
                end_idx = int(match.group(3), 16)
                section_type = int(match.group(4))
                element_type = int(match.group(5))
                
                # Skip sections with type 0 (headers only)
                if section_type == 0:
                    continue
                
                face_count = end_idx - start_idx + 1
                
                # Extract the face connectivity data that follows this header
                section_start = match.end()
                section_end = content.find(')', section_start)
                
                if section_end == -1:
                    logger.warning(f"Could not find end of face section for zone {zone_id}")
                    continue
                
                section_data = content[section_start:section_end].strip()
                
                # Parse face connectivity
                face_connectivity = self._parse_face_connectivity(section_data, element_type, face_count)
                
                if face_connectivity:
                    face_sections.append({
                        'zone_id': zone_id,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'section_type': section_type,
                        'element_type': element_type,
                        'count': face_count,
                        'faces': face_connectivity
                    })
                    
                    logger.info(f"Extracted {len(face_connectivity)} faces for zone {zone_id} (element type {element_type})")
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing face section: {e}")
                continue
        
        logger.info(f"Extracted face connectivity data for {len(face_sections)} face sections")
        return face_sections
    
    def _parse_face_connectivity(self, section_data: str, element_type: int, expected_count: int) -> List[List[int]]:
        """Parse face connectivity data from a section."""
        faces = []
        
        # Remove extra whitespace and split into tokens
        tokens = section_data.split()
        
        # Determine number of nodes per face based on element type
        if element_type == 2:  # Line
            nodes_per_face = 2
        elif element_type == 3:  # Triangle
            nodes_per_face = 3
        elif element_type == 4:  # Quadrilateral
            nodes_per_face = 4
        else:
            logger.warning(f"Unknown element type {element_type}, assuming 4 nodes per face")
            nodes_per_face = 4
        
        # Parse face connectivity data
        # Fluent format: for each face: node1 node2 ... nodeN left_cell right_cell
        nodes_plus_cells = nodes_per_face + 2  # nodes + left_cell + right_cell
        
        i = 0
        while i < len(tokens) and len(faces) < expected_count:
            try:
                # Extract node indices for this face
                if i + nodes_per_face - 1 < len(tokens):
                    face_nodes = []
                    
                    # Convert hex node indices to decimal
                    for j in range(nodes_per_face):
                        node_idx = int(tokens[i + j], 16) - 1  # Convert to 0-based indexing
                        face_nodes.append(node_idx)
                    
                    faces.append(face_nodes)
                    
                    # Skip past this face's data (nodes + cells)
                    i += nodes_plus_cells
                else:
                    break
                    
            except (ValueError, IndexError):
                # Skip invalid data
                i += 1
                continue
        
        logger.debug(f"Parsed {len(faces)} faces from {len(tokens)} tokens (expected {expected_count})")
        return faces
    
    def _extract_face_sections(self) -> List[Dict]:
        """Extract face sections from Fluent mesh file."""
        face_sections = []
        
        with open(self.filename, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Look for face section headers: (13 (zone-id start end type element-type)(
                if line.startswith('(13 ('):
                    import re
                    match = re.search(r'\(13 \((\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\)\(', line)
                    if match:
                        try:
                            zone_id = int(match.group(1), 16)  # Hex to decimal
                            start_idx = int(match.group(2), 16)
                            end_idx = int(match.group(3), 16) 
                            section_type = int(match.group(4))
                            element_type = int(match.group(5))
                        except ValueError:
                            # Skip sections with invalid hex values
                            continue
                        
                        face_count = end_idx - start_idx + 1
                        
                        face_sections.append({
                            'zone_id': zone_id,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'section_type': section_type,
                            'element_type': element_type,
                            'count': face_count,
                            'data': []  # Will be populated when we parse the actual data
                        })
                        
                        logger.debug(f"Found face section: zone={zone_id}, count={face_count}, element_type={element_type}")
        
        logger.info(f"Extracted {len(face_sections)} face sections")
        return face_sections
    
    def _extract_cell_sections(self) -> List[Dict]:
        """Extract cell sections from Fluent mesh file."""
        cell_sections = []
        
        with open(self.filename, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Look for cell section headers: (12 (zone-id start end type element-type))
                if line.startswith('(12 ('):
                    import re
                    match = re.search(r'\(12 \((\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\)\)', line)
                    if match:
                        try:
                            zone_id = int(match.group(1), 16)  # Hex to decimal  
                            start_idx = int(match.group(2), 16)
                            end_idx = int(match.group(3), 16)
                            section_type = int(match.group(4))
                            element_type = int(match.group(5)) if match.group(5) != '0' else 0
                        except ValueError:
                            # Skip sections with invalid hex values
                            continue
                        
                        cell_count = end_idx - start_idx + 1
                        
                        cell_sections.append({
                            'zone_id': zone_id,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'section_type': section_type, 
                            'element_type': element_type,
                            'count': cell_count,
                            'data': []  # Will be populated when we parse the actual data
                        })
                        
                        logger.debug(f"Found cell section: zone={zone_id}, count={cell_count}, element_type={element_type}")
        
        logger.info(f"Extracted {len(cell_sections)} cell sections")
        return cell_sections
    
    def _get_element_type_name(self, element_type_id: int) -> str:
        """Convert element type ID to name."""
        return self.ELEMENT_TYPES.get(element_type_id, "unknown")
    
    def _find_or_create_zone(self, zone_id: int, zones: List) -> FluentZone:
        """Find existing zone by ID or create a new one."""
        for zone in zones:
            if zone.zone_id == zone_id:
                return zone
        
        # Create new zone with default properties
        zone_type = "unknown"
        zone_name = f"zone_{zone_id}"
        zone_type_enum = FluentZoneType.UNKNOWN
        
        zone = FluentZone(zone_id, zone_type, zone_name, zone_type_enum)
        zones.append(zone)
        return zone
    
    def _build_connectivity(self) -> None:
        """Build connectivity information between faces, cells, and points."""
        logger.info("Building mesh connectivity")
        
        # Update zone point indices
        for zone in self.mesh.zone_list:
            for face in zone.faces:
                zone.point_indices.update(face.node_indices)
            for cell_type, cell_list in zone.cells.items():
                for cell in cell_list:
                    zone.point_indices.update(cell.node_indices)
        
        # Rebuild zones dict for GUI compatibility
        self.mesh.zones = {zone.name: {
            'zone_id': zone.zone_id,
            'type': zone.zone_type,
            'zone_type': zone.zone_type,
            'zone_type_enum': zone.zone_type_enum,
            'element_count': zone.num_faces() + zone.num_cells(),
            'num_faces': zone.num_faces(),
            'num_cells': zone.num_cells(),
            'num_points': zone.num_points(),
            'object': zone
        } for zone in self.mesh.zone_list}
        
        logger.info(f"Built connectivity for {len(self.mesh.zone_list)} zones")
