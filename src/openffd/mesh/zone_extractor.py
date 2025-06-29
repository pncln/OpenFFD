"""
Zone extraction functionality for mesh files.

This module provides advanced zone extraction capabilities for meshes, supporting
various mesh formats and handling zone specifications not natively supported by meshio.
"""

import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Generator
from enum import Enum, auto
import copy
import json

import numpy as np

from openffd.utils.parallel import (
    ParallelConfig,
    ParallelExecutor,
    chunk_array,
    get_optimal_chunk_size,
    is_parallelizable
)

# Check for meshio availability
try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "meshio not available. Install with 'pip install meshio' for non-Fluent mesh support."
    )

logger = logging.getLogger(__name__)


class ZoneType(Enum):
    """Types of zones that can be extracted from meshes."""
    VOLUME = auto()
    BOUNDARY = auto()
    INTERFACE = auto()
    UNKNOWN = auto()
    
    @classmethod
    def from_string(cls, value: str) -> 'ZoneType':
        """Convert a string to a ZoneType enum value.
        
        Args:
            value: String representation of a zone type
            
        Returns:
            ZoneType enum value
        """
        mapping = {
            'volume': cls.VOLUME,
            'vol': cls.VOLUME,
            'cell': cls.VOLUME,
            'boundary': cls.BOUNDARY,
            'bound': cls.BOUNDARY,
            'b': cls.BOUNDARY,
            'surface': cls.BOUNDARY,
            'face': cls.BOUNDARY,
            'interface': cls.INTERFACE,
            'int': cls.INTERFACE,
            'i': cls.INTERFACE,
            'interzone': cls.INTERFACE
        }
        
        value_lower = value.lower().strip()
        return mapping.get(value_lower, cls.UNKNOWN)


class ZoneInfo:
    """Information about a mesh zone."""
    
    def __init__(
        self,
        name: str,
        zone_type: ZoneType = ZoneType.UNKNOWN,
        cell_count: int = 0,
        point_count: int = 0,
        element_types: Optional[Set[str]] = None,
        dimensions: Optional[Tuple[int, int, int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a ZoneInfo object.
        
        Args:
            name: Zone name
            zone_type: Type of zone (volume, boundary, interface)
            cell_count: Number of cells in the zone
            point_count: Number of points in the zone
            element_types: Set of element types in the zone
            dimensions: Dimensions of the zone (if structured)
            metadata: Additional metadata about the zone
        """
        self.name = name
        self.zone_type = zone_type
        self.cell_count = cell_count
        self.point_count = point_count
        self.element_types = element_types or set()
        self.dimensions = dimensions
        self.metadata = metadata or {}
        
    def __str__(self) -> str:
        """String representation of the zone info.
        
        Returns:
            String description of the zone
        """
        zone_str = f"Zone '{self.name}' ({self.zone_type.name}): "
        zone_str += f"{self.cell_count} cells, {self.point_count} points"
        
        if self.element_types:
            zone_str += f", types: {', '.join(sorted(self.element_types))}"
            
        if self.dimensions:
            zone_str += f", dimensions: {self.dimensions}"
            
        return zone_str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert zone info to a dictionary.
        
        Returns:
            Dictionary representation of the zone info
        """
        return {
            "name": self.name,
            "zone_type": self.zone_type.name,
            "cell_count": self.cell_count,
            "point_count": self.point_count,
            "element_types": sorted(list(self.element_types)) if self.element_types else [],
            "dimensions": self.dimensions,
            "metadata": self.metadata
        }


class ZoneExtractor:
    """Advanced zone extraction from mesh files.
    
    This class provides functionality to extract zones from mesh files,
    handling various mesh formats and zone specifications not natively
    supported by meshio.
    """
    
    def __init__(self, mesh_file_or_obj: Union[str, Any], is_fluent: bool = False):
        """Initialize the zone extractor.
        
        Args:
            mesh_file_or_obj: Path to the mesh file or a pre-loaded mesh object
            is_fluent: Whether the mesh is in Fluent format
        """
        if isinstance(mesh_file_or_obj, str):
            self._mesh_file = mesh_file_or_obj
            self._mesh = None
        else:
            # Use a pre-loaded mesh object
            self._mesh = mesh_file_or_obj
            self._mesh_file = getattr(mesh_file_or_obj, 'filename', 'preloaded_mesh')
            
        self._is_fluent = is_fluent
        self._zones = {}  # name -> ZoneInfo
        self._loaded = False if self._mesh is None else True
        
        # Track debug info
        self._debug_info = {}
        
        logger.debug(f"Initialized ZoneExtractor for {self._mesh_file} (Fluent format: {is_fluent})")

    def _detect_fluent_mesh(self) -> bool:
        """Detect if the mesh file is in Fluent format.
        
        Returns:
            True if the mesh is in Fluent format, False otherwise
        """
        from openffd.mesh.general import is_fluent_mesh
        return is_fluent_mesh(self._mesh_file)
    
    def _load_mesh(self) -> None:
        """Load the mesh and extract zone information."""
        if self._loaded:
            return
            
        start_time = time.time()
        logger.info(f"Loading mesh from {self.mesh_file}")
        
        if self._is_fluent:
            self._load_fluent_mesh()
        else:
            self._load_general_mesh()
            
        load_time = time.time() - start_time
        logger.info(f"Mesh loaded in {load_time:.2f} seconds with {len(self._zones)} zones")
        
        self._loaded = True
    
    def _load_fluent_mesh(self) -> None:
        """Load a Fluent mesh and extract zone information using our custom parser."""
        from openffd.mesh.fluent_reader import read_fluent_mesh
        
        logger.info(f"Loading Fluent mesh using custom parser: {self.mesh_file}")
        fluent_mesh = read_fluent_mesh(self.mesh_file, debug=True)
        self._mesh = fluent_mesh
        
        # Store the points
        self._points = fluent_mesh.points
        
        # Process all zones from the mesh
        zone_count = 0
        
        for zone_name in fluent_mesh.get_available_zones():
            # Get the zone from the mesh
            fluent_zone = fluent_mesh.get_zone_by_name(zone_name)
            if not fluent_zone:
                continue
                
            # Determine zone type
            zone_enum_type = ZoneType.UNKNOWN
            zone_type = fluent_zone.zone_type.lower()
            
            if zone_type in ['interior', 'fluid']:
                zone_enum_type = ZoneType.VOLUME
            elif zone_type in ['wall', 'symmetry', 'pressure-outlet', 'velocity-inlet', 
                              'pressure-inlet', 'mass-flow-inlet', 'axis']:
                zone_enum_type = ZoneType.BOUNDARY
            elif zone_type in ['interface', 'periodic', 'fan', 'porous-jump', 'non-conformal-interface']:
                zone_enum_type = ZoneType.INTERFACE
            
            # Get statistics
            zone_stats = fluent_zone.get_stats()
            cell_count = zone_stats['cells']
            point_count = zone_stats['nodes']
            
            # Build metadata
            metadata = {
                'type': zone_type,
                'name': zone_name,
                'element_type': fluent_zone.element_type,
                'id': fluent_zone.id,
                'face_count': zone_stats['faces']
            }
            
            # Add any additional properties from the zone
            metadata.update(fluent_zone.properties)
            
            # Create zone info
            zone_info = ZoneInfo(
                name=zone_name,
                zone_type=zone_enum_type,
                cell_count=cell_count,
                point_count=point_count,
                metadata=metadata
            )
            
            # Add to our zones dictionary
            self._zones[zone_name] = zone_info
            zone_count += 1
            
            logger.info(f"Added zone {zone_name} (Type: {zone_type}) with {point_count} points and {cell_count} cells")
        
        logger.info(f"Loaded {zone_count} zones from Fluent mesh")
        
        # If no zones were found, create default zones based on element types
        if zone_count == 0:
            logger.warning("No zones found in Fluent mesh, creating default zones")
            
            # Create default volume and boundary zones
            if len(fluent_mesh.cells) > 0:
                # Create a default volume zone for all cells
                volume_zone_name = "fluid_volume"
                
                # Count cells and points
                cell_count = len(fluent_mesh.cells)
                point_indices = set()
                for cell_id, cell in fluent_mesh.cells.items():
                    for node_id in cell.node_indices:
                        point_indices.add(node_id)
                
                # Create zone info
                zone_info = ZoneInfo(
                    name=volume_zone_name,
                    zone_type=ZoneType.VOLUME,
                    cell_count=cell_count,
                    point_count=len(point_indices),
                    metadata={'type': 'fluid', 'name': 'volume', 'auto_created': True}
                )
                
                self._zones[volume_zone_name] = zone_info
                logger.info(f"Created default volume zone with {cell_count} cells")
            
            if len(fluent_mesh.faces) > 0:
                # Create a default boundary zone for all faces
                boundary_zone_name = "wall_boundary"
                
                # Count faces and points
                face_count = len(fluent_mesh.faces)
                point_indices = set()
                for face_id, face in fluent_mesh.faces.items():
                    for node_id in face.node_indices:
                        point_indices.add(node_id)
                
                # Create zone info
                zone_info = ZoneInfo(
                    name=boundary_zone_name,
                    zone_type=ZoneType.BOUNDARY,
                    cell_count=face_count,  # For boundaries, faces are the cells
                    point_count=len(point_indices),
                    metadata={'type': 'wall', 'name': 'boundary', 'auto_created': True}
                )
                
                self._zones[boundary_zone_name] = zone_info
                logger.info(f"Created default boundary zone with {face_count} faces")
    
    def _load_general_mesh(self) -> None:
        """Load a general mesh using meshio and extract zone information."""
        if not MESHIO_AVAILABLE:
            raise ImportError("meshio is required to read non-Fluent mesh formats. Install it with 'pip install meshio'")
            
        from openffd.mesh.general import read_general_mesh
        
        mesh = read_general_mesh(self.mesh_file)
        self._mesh = mesh
        
        zones_detected = False
        
        # Extract cell sets as zones
        if hasattr(mesh, 'cell_sets') and mesh.cell_sets:
            for name, cell_set in mesh.cell_sets.items():
                self._extract_cell_set_info(name, cell_set)
                zones_detected = True
                
        # Extract physical groups from Gmsh
        if hasattr(mesh, 'field_data') and mesh.field_data:
            for name, field_data in mesh.field_data.items():
                self._extract_field_data_info(name, field_data)
                zones_detected = True
                
        # Extract from cell_data
        if hasattr(mesh, 'cell_data') and mesh.cell_data:
            zones_detected |= self._extract_cell_data_info()
            
        # If no zones were detected, create default zones
        if not zones_detected and len(self._zones) == 0:
            self._create_default_zones()

    def _extract_cell_set_info(self, name: str, cell_set: Dict[str, np.ndarray]) -> None:
        """Extract zone information from a cell set.
        
        Args:
            name: Name of the cell set
            cell_set: Dictionary mapping cell types to arrays of cell indices
        """
        element_types = set(cell_set.keys())
        cell_count = sum(len(indices) for indices in cell_set.values())
        
        # Extract points for this zone
        point_indices = set()
        for cell_type, indices in cell_set.items():
            for i, block in enumerate(self._mesh.cells):
                if block.type == cell_type:
                    cells = block.data[indices]
                    for conn in cells:
                        point_indices.update(conn.tolist())
        
        point_count = len(point_indices)
        
        # Determine zone type based on cell types
        zone_type = self._determine_zone_type(element_types)
        
        zone_info = ZoneInfo(
            name=name,
            zone_type=zone_type,
            cell_count=cell_count,
            point_count=point_count,
            element_types=element_types
        )
        
        self._zones[name] = zone_info
    
    def _extract_field_data_info(self, name: str, field_data: np.ndarray) -> None:
        """Extract zone information from field data.
        
        Args:
            name: Name of the field data
            field_data: Field data array
        """
        # Skip if field_data doesn't represent a physical group
        if len(field_data) < 2 or field_data[1] not in [1, 2, 3]:
            return
            
        # Extract physical group
        tag = field_data[0]
        dim = field_data[1]  # 1=line, 2=surface, 3=volume
        
        # Determine zone type based on dimension
        zone_type = ZoneType.VOLUME if dim == 3 else ZoneType.BOUNDARY
        
        # Find cells with this physical group
        cell_count = 0
        element_types = set()
        point_indices = set()
        
        if 'gmsh:physical' in self._mesh.cell_data:
            phys_tags = self._mesh.cell_data['gmsh:physical']
            for i, (block, phys_data) in enumerate(zip(self._mesh.cells, phys_tags)):
                mask = phys_data == tag
                cell_count += np.sum(mask)
                
                if np.any(mask):
                    element_types.add(block.type)
                    cells = block.data[mask]
                    for conn in cells:
                        point_indices.update(conn.tolist())
        
        zone_info = ZoneInfo(
            name=name,
            zone_type=zone_type,
            cell_count=cell_count,
            point_count=len(point_indices),
            element_types=element_types,
            metadata={"tag": int(tag), "dimension": int(dim)}
        )
        
        self._zones[name] = zone_info
    
    def _extract_cell_data_info(self) -> bool:
        """Extract zone information from cell data.
        
        Returns:
            bool: True if any zones were detected, False otherwise
        """
        # This is more complex as cell data can have different formats
        # We'll try to detect common patterns for zones
        
        zones_detected = False
        
        # 1. Check for zone/region markers in cell_data keys
        zone_keys = [key for key in self._mesh.cell_data.keys() 
                    if any(marker in key.lower() for marker in 
                          ['zone', 'region', 'boundary', 'group', 'part'])]
        
        for key in zone_keys:
            zones_detected |= self._extract_zones_from_cell_data_field(key)
            
        # 2. Look for any field that might contain zone information
        if not self._zones:
            for key in self._mesh.cell_data.keys():
                zones_detected |= self._extract_zones_from_cell_data_field(key)
                
        return zones_detected
    
    def _extract_zones_from_cell_data_field(self, field_name: str) -> bool:
        """Extract zones from a specific cell data field.
        
        Args:
            field_name: Name of the cell data field
            
        Returns:
            bool: True if any zones were detected, False otherwise
        """
        unique_values = set()
        
        # Collect all unique values across cell blocks
        for data_array in self._mesh.cell_data[field_name]:
            if len(data_array) > 0:
                if np.issubdtype(data_array.dtype, np.number):
                    unique_values.update(np.unique(data_array).tolist())
                elif isinstance(data_array[0], (str, bytes)):
                    unique_values.update(np.unique(data_array).tolist())
        
        zones_detected = False
        
        # Create default zone with all points
        if hasattr(self._mesh, 'points') and len(self._mesh.points) > 0:
            element_types = set()
            
            # Create zone info
            zone_info = ZoneInfo(
                name="default",
                zone_type=ZoneType.VOLUME,
                cell_count=0,
                point_count=len(self._mesh.points),
                element_types=set(),
                metadata={"auto_created": True, "contains_all_points": True}
            )
            self._zones["default"] = zone_info
            logger.warning(f"Created default zone with {len(self._mesh.points)} points")
            
        # Check if mesh has cells
        if not hasattr(self._mesh, 'cells') or not self._mesh.cells or len(self._mesh.cells) == 0:
            logger.warning("Mesh has no cells. Only default point-based zone created.")
            return
        
        # Group cells by type
        cell_types = {}
        for i, block in enumerate(self._mesh.cells):
            if block.type not in cell_types:
                cell_types[block.type] = []
            cell_types[block.type].append(i)
        
        # Create zones for each cell type
        for cell_type, block_indices in cell_types.items():
            # Collect cells and points for this type
            cell_count = 0
            point_indices = set()
            
            for block_idx in block_indices:
                cells = self._mesh.cells[block_idx].data
                cell_count += len(cells)
                for conn in cells:
                    point_indices.update(conn.tolist())
            
            # Determine zone type based on cell type
            zone_name = f"{cell_type}_zone"
            zone_type = self._determine_zone_type({cell_type})
            
            zone_info = ZoneInfo(
                name=zone_name,
                zone_type=zone_type,
                cell_count=cell_count,
                point_count=len(point_indices),
                element_types={cell_type},
                metadata={"auto_created": True, "cell_type": cell_type}
            )
            
            self._zones[zone_name] = zone_info
            
        # Create boundary zones by detecting surface cells
        self._detect_and_create_boundaries()
        
        logger.info(f"Created {len(self._zones)} default zones based on mesh geometry")
        
    def _detect_and_create_boundaries(self) -> None:
        """Detect and create boundary zones by analyzing mesh topology.
        
        This method identifies boundary faces by looking for faces that are not shared
        between multiple cells, which typically represent the exterior of the mesh.
        """
        # Only attempt to detect boundaries if the mesh has volume elements
        volume_elements = False
        for block in self._mesh.cells:
            if block.type in {'tetra', 'hexahedron', 'wedge', 'pyramid'}:
                volume_elements = True
                break
                
        if not volume_elements:
            return
        
        # Detect surface elements
        try:
            # Dictionary to track faces and the cells they belong to
            face_cells = {}
            
            # Process each volume cell block
            for i, block in enumerate(self._mesh.cells):
                if block.type not in {'tetra', 'hexahedron', 'wedge', 'pyramid'}:
                    continue
                    
                # Extract faces based on cell type
                for cell_idx, cell in enumerate(block.data):
                    faces = self._get_cell_faces(cell, block.type)
                    
                    for face in faces:
                        # Sort the face to ensure consistent representation
                        face_tuple = tuple(sorted(face))
                        
                        if face_tuple not in face_cells:
                            face_cells[face_tuple] = []
                            
                        face_cells[face_tuple].append((i, cell_idx))
            
            # Faces that appear only once are boundary faces
            boundary_faces = {face: cells for face, cells in face_cells.items() if len(cells) == 1}
            
            if not boundary_faces:
                return
                
            # Group boundary faces
            boundary_face_indices = {}
            for face, (block_idx, cell_idx) in boundary_faces.items():
                if block_idx not in boundary_face_indices:
                    boundary_face_indices[block_idx] = []
                boundary_face_indices[block_idx].append(cell_idx)
            
            # Create a boundary zone
            point_indices = set()
            face_count = len(boundary_faces)
            
            for face in boundary_faces.keys():
                point_indices.update(face)
            
            zone_info = ZoneInfo(
                name="boundary",
                zone_type=ZoneType.BOUNDARY,
                cell_count=face_count,
                point_count=len(point_indices),
                element_types={"polygon"},
                metadata={"auto_created": True, "is_boundary": True}
            )
            
            self._zones["boundary"] = zone_info
            
        except Exception as e:
            logger.warning(f"Failed to detect boundaries: {e}")
            
    def _debug_print_mesh_info(self) -> None:
        """Print debug information about the mesh structure.
        
        This helps diagnose issues with zone detection.
        """
        logger.warning("Debug mesh information:")
        
        # Check if mesh has points
        if hasattr(self._mesh, 'points'):
            logger.warning(f"  - Mesh has {len(self._mesh.points)} points")
        else:
            logger.warning("  - Mesh has no points attribute")
        
        # Check if mesh has cells
        if hasattr(self._mesh, 'cells'):
            if self._mesh.cells:
                logger.warning(f"  - Mesh has {len(self._mesh.cells)} cell blocks")
                for i, block in enumerate(self._mesh.cells):
                    logger.warning(f"    - Block {i}: type={block.type}, cells={len(block.data)}")
            else:
                logger.warning("  - Mesh cells attribute is empty")
        else:
            logger.warning("  - Mesh has no cells attribute")
        
        # Check for cell_sets
        if hasattr(self._mesh, 'cell_sets'):
            if self._mesh.cell_sets:
                logger.warning(f"  - Mesh has {len(self._mesh.cell_sets)} cell sets: {list(self._mesh.cell_sets.keys())}")
            else:
                logger.warning("  - Mesh cell_sets attribute is empty")
        else:
            logger.warning("  - Mesh has no cell_sets attribute")
        
        # Check for field_data
        if hasattr(self._mesh, 'field_data'):
            if self._mesh.field_data:
                logger.warning(f"  - Mesh has {len(self._mesh.field_data)} field data items: {list(self._mesh.field_data.keys())}")
            else:
                logger.warning("  - Mesh field_data attribute is empty")
        else:
            logger.warning("  - Mesh has no field_data attribute")
        
        # Check for cell_data
        if hasattr(self._mesh, 'cell_data'):
            if self._mesh.cell_data:
                logger.warning(f"  - Mesh has {len(self._mesh.cell_data)} cell data arrays: {list(self._mesh.cell_data.keys())}")
                # Show a sample of each cell data array
                for key, data_arrays in self._mesh.cell_data.items():
                    logger.warning(f"    - {key}: {len(data_arrays)} arrays")
                    for i, array in enumerate(data_arrays):
                        if len(array) > 0:
                            sample = array[0]
                            dtype = array.dtype
                            logger.warning(f"      - Array {i}: len={len(array)}, dtype={dtype}, sample={sample}")
            else:
                logger.warning("  - Mesh cell_data attribute is empty")
        else:
            logger.warning("  - Mesh has no cell_data attribute")
        
        # Check for point_data
        if hasattr(self._mesh, 'point_data'):
            if self._mesh.point_data:
                logger.warning(f"  - Mesh has {len(self._mesh.point_data)} point data arrays: {list(self._mesh.point_data.keys())}")
            else:
                logger.warning("  - Mesh point_data attribute is empty")
        else:
            logger.warning("  - Mesh has no point_data attribute")
            
        # Print mesh file path
        logger.warning(f"  - Mesh file: {self._mesh_file}")
    
    def _get_cell_faces(self, cell: np.ndarray, cell_type: str) -> List[List[int]]:
        """Extract faces from a volume cell.
        
        Args:
            cell: Cell connectivity array
            cell_type: Type of cell
            
        Returns:
            List of faces, where each face is a list of point indices
        """
        if cell_type == 'tetra':
            # Tetrahedron has 4 triangular faces
            return [
                [cell[0], cell[1], cell[2]],  # base
                [cell[0], cell[1], cell[3]],
                [cell[1], cell[2], cell[3]],
                [cell[2], cell[0], cell[3]]
            ]
        elif cell_type == 'hexahedron':
            # Hexahedron has 6 quadrilateral faces
            return [
                [cell[0], cell[1], cell[2], cell[3]],  # bottom
                [cell[4], cell[5], cell[6], cell[7]],  # top
                [cell[0], cell[1], cell[5], cell[4]],  # front
                [cell[1], cell[2], cell[6], cell[5]],  # right
                [cell[2], cell[3], cell[7], cell[6]],  # back
                [cell[3], cell[0], cell[4], cell[7]]   # left
            ]
        elif cell_type == 'wedge':
            # Wedge has 2 triangular and 3 quadrilateral faces
            return [
                [cell[0], cell[1], cell[2]],          # bottom triangle
                [cell[3], cell[4], cell[5]],          # top triangle
                [cell[0], cell[1], cell[4], cell[3]],  # front quad
                [cell[1], cell[2], cell[5], cell[4]],  # right quad
                [cell[2], cell[0], cell[3], cell[5]]   # back quad
            ]
        elif cell_type == 'pyramid':
            # Pyramid has 1 quadrilateral and 4 triangular faces
            return [
                [cell[0], cell[1], cell[2], cell[3]],  # base
                [cell[0], cell[1], cell[4]],          # front
                [cell[1], cell[2], cell[4]],          # right
                [cell[2], cell[3], cell[4]],          # back
                [cell[3], cell[0], cell[4]]           # left
            ]
        else:
            return []  # Unsupported cell type
    
    def _determine_zone_type(self, element_types: Set[str]) -> ZoneType:
        """Determine the zone type based on element types.
        
        Args:
            element_types: Set of element types in the zone
            
        Returns:
            Zone type
        """
        # Surface elements
        surface_elements = {'triangle', 'quad', 'polygon', 'line', 'line3'}
        # Volume elements
        volume_elements = {'tetra', 'hexahedron', 'wedge', 'pyramid', 'hexahedron20'}
        
        is_surface = all(etype in surface_elements for etype in element_types)
        is_volume = any(etype in volume_elements for etype in element_types)
        
        if is_volume:
            return ZoneType.VOLUME
        elif is_surface:
            return ZoneType.BOUNDARY
        else:
            return ZoneType.UNKNOWN
    
    def _get_zone_points(self, zone_name: str) -> np.ndarray:
        """Get all points for a specific zone.
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            Numpy array of point coordinates
        """
        if self._is_fluent:
            return self._mesh.get_zone_points(zone_name)
        
        # For meshio, extract points based on zone name
        from openffd.mesh.general import extract_patch_points
        try:
            return extract_patch_points(self._mesh, zone_name)
        except ValueError:
            # Zone might be in our extended format
            if zone_name in self._zones:
                return self._extract_zone_points(zone_name)
            raise
    
    def _extract_zone_points(self, zone_name: str) -> np.ndarray:
        """Extract point coordinates for a specific zone.
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            Numpy array of point coordinates
        """
        if zone_name not in self._zones:
            raise ValueError(f"Zone '{zone_name}' not found")
            
        # Extract cells for this zone
        zone_cells = self._extract_zone_cells(zone_name)
        
        # Extract unique point indices
        point_indices = set()
        for cell_type, indices in zone_cells.items():
            for i, block in enumerate(self._mesh.cells):
                if block.type == cell_type:
                    for idx in indices:
                        conn = block.data[idx]
                        point_indices.update(conn.tolist())
        
        # Get point coordinates
        point_indices = list(point_indices)
        return self._mesh.points[point_indices]
    
    def _extract_zone_cells(self, zone_name: str) -> Dict[str, List[int]]:
        """Extract cells for a specific zone.
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            Dictionary mapping cell types to lists of cell indices
        """
        if zone_name not in self._zones:
            raise ValueError(f"Zone '{zone_name}' not found")
            
        zone_info = self._zones[zone_name]
        
        # Try to extract from metadata if available
        if 'field' in zone_info.metadata and 'value' in zone_info.metadata:
            field = zone_info.metadata['field']
            value = zone_info.metadata['value']
            
            zone_cells = {}
            for i, data_array in enumerate(self._mesh.cell_data[field]):
                if i >= len(self._mesh.cells):
                    continue
                    
                if isinstance(value, (int, float)) and np.issubdtype(data_array.dtype, np.number):
                    mask = data_array == value
                elif isinstance(value, str) and isinstance(data_array[0], (str, bytes)):
                    mask = np.array([str(v) == value for v in data_array])
                else:
                    continue
                
                if np.any(mask):
                    cell_type = self._mesh.cells[i].type
                    if cell_type not in zone_cells:
                        zone_cells[cell_type] = []
                    zone_cells[cell_type].extend(np.where(mask)[0].tolist())
                    
            return zone_cells
            
        # If metadata doesn't contain the necessary information,
        # we need to re-extract from cell_sets or field_data
        # ...
        
        # Fallback method: scan all cell blocks for the zone
        return {}
    
    def extract_zone_mesh(self, zone_name: str) -> Any:
        """Extract a submesh for the specified zone.
        
        Args:
            zone_name: Name of the zone to extract
            
        Returns:
            meshio.Mesh object for the extracted zone
        """
        if not MESHIO_AVAILABLE:
            raise ImportError("meshio is required for zone extraction")
            
        if not self._loaded:
            self._load_mesh()
            
        if zone_name not in self._zones:
            available_zones = ", ".join(self._zones.keys())
            raise ValueError(f"Zone '{zone_name}' not found. Available zones: {available_zones}")
            
        # Check if we've already extracted this zone
        if zone_name in self._extracted_zones:
            return self._extracted_zones[zone_name]
            
        # Extract zone points
        zone_points = self._get_zone_points(zone_name)
        
        # For fluent mesh, we need to create a new meshio mesh
        if self._is_fluent:
            # Create a simplified mesh with just the points
            submesh = meshio.Mesh(
                points=zone_points,
                cells=[]
            )
            self._extracted_zones[zone_name] = submesh
            return submesh
            
        # For meshio mesh, extract a proper submesh with cells
        cells = []
        cell_data = {}
        
        # Map original points to new indices
        point_indices = []
        point_map = {}
        
        # Try to extract cells belonging to this zone
        zone_cells = self._extract_zone_cells(zone_name)
        
        if not zone_cells:
            # If we couldn't extract cells directly, use the zone points
            # to create a point cloud mesh
            submesh = meshio.Mesh(
                points=zone_points,
                cells=[]
            )
            self._extracted_zones[zone_name] = submesh
            return submesh
            
        # Process cells for each type
        for cell_type, indices in zone_cells.items():
            for i, block in enumerate(self._mesh.cells):
                if block.type == cell_type:
                    # Get cells of this type
                    zone_cell_data = block.data[indices]
                    
                    # Update point mapping
                    for cell in zone_cell_data:
                        for point_idx in cell:
                            if point_idx not in point_map:
                                point_map[point_idx] = len(point_indices)
                                point_indices.append(point_idx)
                                
                    # Remap point indices
                    new_cells = np.array([[point_map[idx] for idx in cell] for cell in zone_cell_data])
                    
                    # Add to cells list
                    cells.append(meshio.CellBlock(
                        type=cell_type,
                        data=new_cells
                    ))
                    
                    # Copy cell data for these cells
                    for key, data_arrays in self._mesh.cell_data.items():
                        if i < len(data_arrays):
                            cell_values = data_arrays[i][indices]
                            if key not in cell_data:
                                cell_data[key] = []
                            cell_data[key].append(cell_values)
        
        # Create the submesh
        submesh = meshio.Mesh(
            points=self._mesh.points[point_indices],
            cells=cells,
            cell_data=cell_data
        )
        
        self._extracted_zones[zone_name] = submesh
        return submesh
        
    def _create_default_zones(self) -> None:
        """Create default zones when none are found in the mesh.
        
        This method creates zones based on geometry and cell types when
        no explicit zone information is present in the mesh.
        """
        logger.warning("No zones found in mesh, creating default zones")
        
        # Print debug info about the mesh
        self._debug_print_mesh_info()
        
        # Initialize data structures for zone tracking if they don't exist
        if not hasattr(self, '_zone_node_indices'):
            self._zone_node_indices = {}
        if not hasattr(self, '_zone_faces'):
            self._zone_faces = {}
        if not hasattr(self, '_zone_face_types'):
            self._zone_face_types = {}
        
        # Always create a default zone with all points as an absolute fallback
        if hasattr(self._mesh, 'points') and len(self._mesh.points) > 0:
            # Create zone info
            zone_info = ZoneInfo(
                name="default",
                zone_type=ZoneType.VOLUME,
                cell_count=0,
                point_count=len(self._mesh.points),
                element_types=set(),
                metadata={"auto_created": True, "contains_all_points": True}
            )
            self._zones["default"] = zone_info
            self._zone_node_indices["default"] = list(range(len(self._mesh.points)))
            self._zone_faces["default"] = []
            self._zone_face_types["default"] = []
            logger.warning(f"Created default zone with {len(self._mesh.points)} points")
            
            # If the mesh has cells, add them to the default zone
            if hasattr(self._mesh, 'cells') and self._mesh.cells:
                for i, cell_block in enumerate(self._mesh.cells):
                    for cell in cell_block.data:
                        self._zone_faces["default"].append(list(cell))
                        self._zone_face_types["default"].append(cell_block.type)
                logger.info(f"Added {len(self._zone_faces['default'])} cells to default zone")
    
    def get_zone_names(self, zone_type: Optional[ZoneType] = None) -> List[str]:
        """Get names of all available zones, optionally filtered by type.
        
        Args:
            zone_type: Optional filter for zone type
            
        Returns:
            List of zone names
        """
        if not self._loaded:
            self._load_mesh()
            
        # If no zones were detected, create default zones
        if not self._zones:
            logger.warning("No zones found in get_zone_names, creating default zones")
            self._create_default_zones()
            
        # Still no zones? Create a forced default zone with all points
        if not self._zones and hasattr(self._mesh, 'points') and len(self._mesh.points) > 0:
            logger.warning("Creating forced default zone with all points")
            zone_info = ZoneInfo(
                name="default_all_points",
                zone_type=ZoneType.VOLUME if zone_type is None else zone_type,
                cell_count=0,
                point_count=len(self._mesh.points),
                element_types=set(),
                metadata={"auto_created": True, "forced": True}
            )
            self._zones["default_all_points"] = zone_info
            
        if zone_type is None:
            return list(self._zones.keys())
            
        zone_names = [name for name, info in self._zones.items() if info.zone_type == zone_type]
        
        # If filtering by zone type results in no zones, but we have zones of other types,
        # create a default zone of the requested type
        if not zone_names and self._zones and hasattr(self._mesh, 'points') and len(self._mesh.points) > 0:
            logger.warning(f"Creating forced default zone of type {zone_type.name}")
            zone_name = f"default_{zone_type.name.lower()}"
            zone_info = ZoneInfo(
                name=zone_name,
                zone_type=zone_type,
                cell_count=0,
                point_count=len(self._mesh.points),
                element_types=set(),
                metadata={"auto_created": True, "forced": True, "zone_type_specific": True}
            )
            self._zones[zone_name] = zone_info
            zone_names = [zone_name]
            
        return zone_names
    
    def get_zone_info(self, zone_name: Optional[str] = None) -> Union[ZoneInfo, Dict[str, ZoneInfo]]:
        """Get information about zones.
        
        Args:
            zone_name: Optional zone name to get specific info
            
        Returns:
            ZoneInfo object or dictionary of zone names to ZoneInfo objects
        """
        if not self._loaded:
            self._load_mesh()
            
        if zone_name is not None:
            if zone_name not in self._zones:
                available_zones = ", ".join(self._zones.keys())
                raise ValueError(f"Zone '{zone_name}' not found. Available zones: {available_zones}")
            return self._zones[zone_name]
            
        return self._zones
    
    def save_zone_mesh(self, zone_name: str, output_file: str) -> None:
        """Save a zone mesh to a file.
        
        Args:
            zone_name: Name of the zone to save
            output_file: Path to the output file
        """
        if not MESHIO_AVAILABLE:
            raise ImportError("meshio is required for zone extraction")
            
        submesh = self.extract_zone_mesh(zone_name)
        
        # Determine output format from extension
        ext = os.path.splitext(output_file)[1].lower()
        file_format = None
        
        # Map common extensions to meshio formats
        format_map = {
            '.vtk': 'vtk',
            '.vtu': 'vtu',
            '.stl': 'stl',
            '.obj': 'obj',
            '.ply': 'ply',
            '.off': 'off',
            '.gmsh': 'gmsh',
            '.msh': 'gmsh',
            '.mesh': 'medit',
            '.xdmf': 'xdmf',
            '.h5m': 'h5m',
            '.med': 'med',
            '.dat': 'tecplot',
            '.foam': 'foam',
            '.bdf': 'nastran',
            '.inp': 'abaqus'
        }
        
        if ext in format_map:
            file_format = format_map[ext]
        
        logger.info(f"Saving zone '{zone_name}' to {output_file}")
        meshio.write(output_file, submesh, file_format)
    
    def extract_boundary_mesh(self, boundary_name: str) -> Any:
        """Extract a mesh for the specified boundary.
        
        This is a convenience method that checks if the zone is a boundary
        and extracts it.
        
        Args:
            boundary_name: Name of the boundary zone to extract
            
        Returns:
            meshio.Mesh object for the extracted boundary
        """
        if not self._loaded:
            self._load_mesh()
            
        if boundary_name not in self._zones:
            available_boundaries = ", ".join(self.get_zone_names(ZoneType.BOUNDARY))
            raise ValueError(f"Boundary '{boundary_name}' not found. Available boundaries: {available_boundaries}")
            
        zone_info = self._zones[boundary_name]
        if zone_info.zone_type != ZoneType.BOUNDARY:
            raise ValueError(f"Zone '{boundary_name}' is not a boundary")
            
        return self.extract_zone_mesh(boundary_name)
    
    def parallel_extract_zones(self, zone_names: List[str]) -> Dict[str, Any]:
        """Extract multiple zones in parallel.
        
        Args:
            zone_names: List of zone names to extract
            
        Returns:
            Dictionary mapping zone names to extracted meshes
        """
        if not self._loaded:
            self._load_mesh()
            
        # Check if zone names are valid
        invalid_zones = [name for name in zone_names if name not in self._zones]
        if invalid_zones:
            raise ValueError(f"Invalid zone names: {', '.join(invalid_zones)}")
            
        # If parallel processing is not configured or only a few zones,
        # extract them sequentially
        if (self.parallel_config is None or 
            not is_parallelizable(len(zone_names), self.parallel_config)):
            return {name: self.extract_zone_mesh(name) for name in zone_names}
        
        # Use parallel processing for many zones
        executor = ParallelExecutor(self.parallel_config)
        
        # Create a function that extracts a zone
        def extract_zone(name: str) -> Tuple[str, Any]:
            return name, self.extract_zone_mesh(name)
        
        # Process zones in parallel
        results = executor.map(extract_zone, zone_names)
        
        # Convert results to dictionary
        return {name: mesh for name, mesh in results}
    
    def export_zones_summary(self, output_file: str) -> None:
        """Export a summary of all zones to a JSON file.
        
        Args:
            output_file: Path to the output file
        """
        if not self._loaded:
            self._load_mesh()
            
        # Create a summary dictionary
        summary = {
            "mesh_file": self.mesh_file,
            "format": "Fluent" if self._is_fluent else "General",
            "zones": {name: info.to_dict() for name, info in self._zones.items()}
        }
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Exported zones summary to {output_file}")


# Parallel processing functions for zone extraction

def extract_zones_parallel(
    mesh_file: str,
    zone_names: List[str],
    config: Optional[ParallelConfig] = None
) -> Dict[str, Any]:
    """Extract multiple zones in parallel from a mesh file.
    
    Args:
        mesh_file: Path to the mesh file
        zone_names: List of zone names to extract
        config: Parallel processing configuration
        
    Returns:
        Dictionary mapping zone names to extracted meshes
    """
    extractor = ZoneExtractor(mesh_file, config)
    return extractor.parallel_extract_zones(zone_names)


def read_mesh_with_zones(
    mesh_file: str,
    parallel_config: Optional[ParallelConfig] = None
) -> Tuple[Any, Dict[str, ZoneInfo]]:
    """Read a mesh file and extract zone information.
    
    Args:
        mesh_file: Path to the mesh file
        parallel_config: Optional parallel processing configuration
        
    Returns:
        Tuple of (mesh_data, zone_info_dict)
    """
    extractor = ZoneExtractor(mesh_file, parallel_config)
    zone_info = extractor.get_zone_info()
    
    # Return the mesh data and zone info
    if not extractor._loaded:
        extractor._load_mesh()
        
    return extractor._mesh, zone_info
