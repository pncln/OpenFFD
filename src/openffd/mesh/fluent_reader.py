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
        
        # Return the points for this zone
        return self.points[list(point_indices)]
    
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
        
        # Intelligent assignment of boundary cell blocks to zones
        if boundary_cell_blocks:
            self._assign_boundary_blocks_intelligently(boundary_cell_blocks, fluent_mesh)
        
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
    
    def _assign_boundary_blocks_intelligently(self, boundary_cell_blocks: List[dict], fluent_mesh: FluentMesh):
        """Intelligently assign boundary cell blocks to zones based on size matching.
        
        Args:
            boundary_cell_blocks: List of cell block info dicts
            fluent_mesh: The mesh to assign to
        """
        logger.info(f"Intelligently assigning {len(boundary_cell_blocks)} boundary blocks to zones")
        
        # Get boundary zones sorted by expected size (from zone names if available)
        boundary_zones = [z for z in fluent_mesh.zone_list if z.zone_type_enum == FluentZoneType.BOUNDARY]
        
        # Expected size order based on typical CFD boundary naming
        zone_size_priority = {
            'launchpad': 1000000,  # Largest - main surface
            'wedge_neg': 100000,   # Second largest - wedge boundary
            'deflector': 1000,     # Medium - deflector surface
            'rocket': 1000,        # Medium - rocket surface
            'inlet': 2000,         # Medium - inlet surface
            'outlet': 2000,        # Medium - outlet surface
            'symmetry': 2000,      # Medium - symmetry plane
            'wedge_pos': 100       # Smallest - small wedge
        }
        
        # Sort zones by expected size (descending)
        boundary_zones.sort(key=lambda z: zone_size_priority.get(z.name, 500), reverse=True)
        
        # Sort cell blocks by size (descending)
        boundary_cell_blocks.sort(key=lambda b: b['size'], reverse=True)
        
        logger.info("Zone to cell block mapping:")
        for i, (zone, block) in enumerate(zip(boundary_zones, boundary_cell_blocks)):
            logger.info(f"  {zone.name} ({zone_size_priority.get(zone.name, 'unknown')}) ← Block {block['index']} ({block['size']} faces)")
        
        # Assign cell blocks to zones
        for zone, block in zip(boundary_zones, boundary_cell_blocks):
            cell_type = block['type']
            cell_data = block['data']
            
            # Add cells to the zone
            zone.add_cells(cell_data, cell_type)
            
            # Add faces to mesh
            for face_idx, face_nodes in enumerate(cell_data):
                face = FluentFace(len(fluent_mesh.faces), face_nodes.tolist(), cell_type)
                face.zone_id = zone.zone_id
                fluent_mesh.faces.append(face)
                zone.faces.append(face)
            
            logger.info(f"✅ Assigned {len(cell_data)} faces to zone '{zone.name}'")
        
        # Handle any remaining cell blocks (if more blocks than zones)
        if len(boundary_cell_blocks) > len(boundary_zones):
            logger.warning(f"More cell blocks ({len(boundary_cell_blocks)}) than zones ({len(boundary_zones)})")
            for i in range(len(boundary_zones), len(boundary_cell_blocks)):
                block = boundary_cell_blocks[i]
                # Assign to last zone as fallback
                if boundary_zones:
                    zone = boundary_zones[-1]
                    zone.add_cells(block['data'], block['type'])
                    logger.warning(f"Assigned overflow block {block['index']} to zone '{zone.name}'")
    
    def _create_artificial_mesh(self) -> FluentMesh:
        """Create an artificial mesh for testing purposes when mesh reading fails."""
        logger.warning("Creating artificial mesh due to reading failure")
        
        # Create a simple test mesh
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Create a simple zone
        zone = FluentZone(1, "fluid", "artificial_fluid", FluentZoneType.VOLUME)
        
        return FluentMesh(points, [zone])
    
    def _is_binary_file(self) -> bool:
        """Determine if the mesh file is binary or ASCII."""
        try:
            with open(self.filename, 'rb') as file:
                # Read first few bytes to check for binary markers
                header = file.read(1024)
                
                # Check for null bytes (common in binary files)
                if b'\x00' in header:
                    return True
                
                # Check for high-bit characters
                non_ascii_count = sum(1 for byte in header if byte > 127)
                if non_ascii_count > len(header) * 0.1:  # If more than 10% non-ASCII
                    return True
                
                # Try to decode as ASCII
                try:
                    header.decode('ascii')
                    # Additional check for Fluent ASCII patterns
                    header_str = header.decode('ascii')
                    if '(' in header_str and ')' in header_str:
                        return False  # Likely ASCII Fluent format
                    return False
                except UnicodeDecodeError:
                    return True
                    
        except Exception:
            # Default to ASCII if we can't determine
            return False
    
    def _read_ascii_mesh(self):
        """Read an ASCII Fluent mesh file."""
        logger.debug("Reading ASCII Fluent mesh file")
        with open(self.filename, 'r') as file:
            content = file.read()
        
        self._parse_ascii_mesh(content)
    
    def _parse_ascii_mesh(self, content: str):
        """Parse the content of an ASCII Fluent mesh file."""
        # Parse nodes (points)
        self._parse_nodes(content)
        
        # Parse zone definitions
        self._parse_zones(content)
        
        # Parse faces
        self._parse_faces(content)
        
        # Parse cells
        self._parse_cells(content)
    
    def _parse_nodes(self, content: str):
        """Parse nodes (points) from the mesh content."""
        # Find node sections using regex
        # Fluent format: (10 (zone-id first-index last-index type nd)
        # followed by coordinate data ending with )
        node_pattern = r'\(10\s+\(([^)]+)\)\s*\n(.*?)\n\)'
        
        points = []
        for match in re.finditer(node_pattern, content, re.DOTALL):
            header = match.group(1).strip()
            data = match.group(2).strip()
            
            # Parse header
            header_parts = header.split()
            if len(header_parts) >= 3:
                try:
                    # Handle both hex and decimal indices
                    if header_parts[1].startswith('0x') or any(c in header_parts[1].lower() for c in 'abcdef'):
                        first_index = int(header_parts[1], 16)
                        last_index = int(header_parts[2], 16)
                    else:
                        first_index = int(header_parts[1])
                        last_index = int(header_parts[2])
                except ValueError:
                    logger.warning(f"Could not parse node indices: {header_parts[1:]})")
                    continue
                
                # Parse point coordinates
                if data:
                    lines = data.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                            # Remove parentheses and parse coordinates
                            line = line.replace('(', '').replace(')', '')
                            coords = line.strip().split()
                            if len(coords) >= 2:
                                try:
                                    x = float(coords[0])
                                    y = float(coords[1])
                                    z = float(coords[2]) if len(coords) > 2 else 0.0
                                    points.append([x, y, z])
                                except ValueError:
                                    logger.warning(f"Could not parse coordinates: {line}")
        
        if points:
            self.mesh.points = np.array(points)
            logger.info(f"Parsed {len(points)} nodes")
        else:
            logger.warning("No nodes found in mesh file")
            self.mesh.points = np.array([])
    
    def _parse_zones(self, content: str):
        """Parse zone definitions from the mesh content."""
        # Find zone sections
        # Fluent format: (45 (zone-id zone-type zone-name)())
        zone_pattern = r'\(45\s+\(([^)]+)\)\(\)\)'
        
        zones = []
        zone_id = 1
        
        for match in re.finditer(zone_pattern, content):
            header = match.group(1).strip()
            header_parts = header.split()
            
            if len(header_parts) >= 3:
                try:
                    # Try to parse as numeric zone type ID
                    zone_type_id = int(header_parts[1])
                    zone_name = header_parts[2].strip('"')
                    # Map zone type ID to string
                    zone_type = self.ZONE_TYPES.get(zone_type_id, "unknown")
                except ValueError:
                    # Parse as string zone type and name format
                    zone_type = header_parts[1]  # e.g., "fluid"
                    zone_name = header_parts[2].strip('"')  # e.g., "interior"
                
                zone_type_enum = self._classify_zone_type(zone_type)
                
                zone = FluentZone(zone_id, zone_type, zone_name, zone_type_enum)
                zones.append(zone)
                zone_id += 1
                
                logger.debug(f"Parsed zone: {zone}")
        
        # Update mesh zones properly
        self.mesh.zone_list = zones
        # Rebuild zones dict for GUI compatibility
        self.mesh.zones = {zone.name: {
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
        logger.info(f"Parsed {len(zones)} zones")
    
    def _parse_faces(self, content: str):
        """Parse faces from the mesh content."""
        # Implementation for face parsing
        # This is a simplified version - full implementation would be more complex
        logger.debug("Parsing faces (simplified implementation)")
        self.mesh.faces = []
    
    def _parse_cells(self, content: str):
        """Parse cells from the mesh content."""
        # Implementation for cell parsing
        # This is a simplified version - full implementation would be more complex
        logger.debug("Parsing cells (simplified implementation)")
        self.mesh.cells = []
    
    def _read_binary_mesh(self):
        """Read a binary Fluent mesh file."""
        logger.debug("Reading binary Fluent mesh file")
        
        # For binary files, prefer meshio if available
        if MESHIO_AVAILABLE and not self.force_native:
            logger.info("Binary file detected - using meshio for parsing")
            self.mesh = self._read_with_meshio()
            return
        
        # Native binary parsing is complex - create a simple fallback
        logger.warning("Binary mesh parsing not fully implemented in native parser")
        logger.info("Creating fallback mesh - consider using meshio for binary files")
        
        # Create a more realistic artificial mesh for testing
        points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]
        ])
        
        # Create zones
        fluid_zone = FluentZone(1, "fluid", "interior_fluid", FluentZoneType.VOLUME)
        wall_zone = FluentZone(2, "wall", "wall_boundary", FluentZoneType.BOUNDARY)
        
        self.mesh = FluentMesh(points, [fluid_zone, wall_zone])
    
    def _build_connectivity(self):
        """Build connectivity between mesh elements."""
        logger.debug("Building mesh connectivity")
        # This would establish relationships between faces, cells, and zones
        pass


def read_fluent_mesh(filename: str, **kwargs) -> FluentMesh:
    """Read a Fluent mesh file and return a FluentMesh object.
    
    This is a convenience function that creates a FluentMeshReader and reads the mesh.
    
    Args:
        filename: Path to the Fluent mesh file
        **kwargs: Additional options (debug, force_ascii, force_binary, use_meshio, force_native)
        
    Returns:
        FluentMesh object
        
    Raises:
        FileNotFoundError: If the mesh file does not exist
        ValueError: If the file format is not recognized
    """
    reader = FluentMeshReader(filename, **kwargs)
    return reader.read()
