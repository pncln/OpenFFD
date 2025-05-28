"""
Specialized Fluent mesh parser module.

This module provides a direct parser for Fluent mesh files (.msh, .cas) without relying
on third-party libraries like meshio. It properly handles zone specifications and 
boundary definitions for more accurate zone extraction.
"""

import os
import re
import logging
import struct
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Union, BinaryIO

# Configure logging
logger = logging.getLogger(__name__)

class FluentZone:
    """Class representing a Fluent mesh zone with its elements and properties."""
    
    def __init__(self, zone_id: int, name: str, zone_type: str, element_type: Optional[str] = None):
        """Initialize a FluentZone object.
        
        Args:
            zone_id: Zone identifier
            name: Zone name
            zone_type: Type of zone (e.g., 'wall', 'fluid', 'interior')
            element_type: Type of elements in the zone (e.g., 'quad', 'hex')
        """
        self.id = zone_id
        self.name = name
        self.zone_type = zone_type
        self.element_type = element_type
        self.node_indices = []  # List of node indices in this zone
        self.face_indices = []  # List of face indices in this zone
        self.cell_indices = []  # List of cell indices in this zone
        self.properties = {}    # Additional zone properties
        
    def __str__(self) -> str:
        """String representation of the zone."""
        return f"Zone {self.id}: {self.name} (Type: {self.zone_type})"
    
    def add_node(self, node_id: int) -> None:
        """Add a node to this zone.
        
        Args:
            node_id: Index of the node to add
        """
        if node_id not in self.node_indices:
            self.node_indices.append(node_id)
            
    def add_face(self, face_id: int) -> None:
        """Add a face to this zone.
        
        Args:
            face_id: Index of the face to add
        """
        if face_id not in self.face_indices:
            self.face_indices.append(face_id)
    
    def add_cell(self, cell_id: int) -> None:
        """Add a cell to this zone.
        
        Args:
            cell_id: Index of the cell to add
        """
        if cell_id not in self.cell_indices:
            self.cell_indices.append(cell_id)
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about this zone.
        
        Returns:
            Dictionary with counts of nodes, faces, and cells
        """
        return {
            "nodes": len(self.node_indices),
            "faces": len(self.face_indices),
            "cells": len(self.cell_indices)
        }


class FluentFace:
    """Class representing a face in a Fluent mesh."""
    
    def __init__(self, face_id: int, node_indices: List[int], face_type: str = "unknown"):
        """Initialize a FluentFace object.
        
        Args:
            face_id: Face identifier
            node_indices: List of node indices that make up this face
            face_type: Type of face (e.g., 'triangle', 'quad')
        """
        self.id = face_id
        self.node_indices = node_indices
        self.face_type = face_type
        self.owner_cell = -1      # Cell that owns this face
        self.neighbor_cell = -1   # Neighboring cell (if any)
        self.zone_id = -1         # Zone this face belongs to
        
    def __str__(self) -> str:
        """String representation of the face."""
        return f"Face {self.id}: {self.face_type} with {len(self.node_indices)} nodes"


class FluentCell:
    """Class representing a cell in a Fluent mesh."""
    
    def __init__(self, cell_id: int, node_indices: List[int], cell_type: str = "unknown"):
        """Initialize a FluentCell object.
        
        Args:
            cell_id: Cell identifier
            node_indices: List of node indices that make up this cell
            cell_type: Type of cell (e.g., 'tetra', 'hex')
        """
        self.id = cell_id
        self.node_indices = node_indices
        self.cell_type = cell_type
        self.face_indices = []   # Faces that make up this cell
        self.zone_id = -1        # Zone this cell belongs to
        
    def __str__(self) -> str:
        """String representation of the cell."""
        return f"Cell {self.id}: {self.cell_type} with {len(self.node_indices)} nodes"
    
    def add_face(self, face_id: int) -> None:
        """Add a face to this cell.
        
        Args:
            face_id: Index of the face to add
        """
        if face_id not in self.face_indices:
            self.face_indices.append(face_id)


class FluentMesh:
    """Class representing a complete Fluent mesh."""
    
    def __init__(self):
        """Initialize an empty FluentMesh object."""
        self.points = np.array([])  # Numpy array of point coordinates
        self.zones = {}           # Dictionary mapping zone IDs to FluentZone objects
        self.faces = {}           # Dictionary mapping face IDs to FluentFace objects
        self.cells = {}           # Dictionary mapping cell IDs to FluentCell objects
        self.zone_map = {}        # Maps zone names to zone IDs
        self.metadata = {}        # Mesh metadata
        
    def get_zone_by_name(self, name: str) -> Optional[FluentZone]:
        """Get a zone by its name.
        
        Args:
            name: Name of the zone to retrieve
            
        Returns:
            FluentZone object or None if not found
        """
        if name in self.zone_map:
            zone_id = self.zone_map[name]
            return self.zones.get(zone_id)
        return None
    
    def get_zone_points(self, zone_name: str) -> np.ndarray:
        """Get point coordinates for a specific zone.
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            Array of point coordinates
            
        Raises:
            ValueError: If zone not found
        """
        zone = self.get_zone_by_name(zone_name)
        if zone is None:
            raise ValueError(f"Zone not found: {zone_name}")
        
        return self.points[zone.node_indices]
    
    def get_zone_type(self, zone_name: str) -> str:
        """Get the type of a zone.
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            Zone type string
            
        Raises:
            ValueError: If zone not found
        """
        zone = self.get_zone_by_name(zone_name)
        if zone is None:
            raise ValueError(f"Zone not found: {zone_name}")
        
        return zone.zone_type
    
    def get_available_zones(self) -> List[str]:
        """Get a list of all available zone names.
        
        Returns:
            List of zone names
        """
        return list(self.zone_map.keys())
    
    def get_zone_faces(self, zone_name: str) -> List[List[int]]:
        """Get all faces for a specific zone.
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            List of faces, where each face is a list of node indices
            
        Raises:
            ValueError: If zone not found or has no faces
        """
        zone = self.get_zone_by_name(zone_name)
        if zone is None:
            raise ValueError(f"Zone not found: {zone_name}")
        
        if not zone.face_indices:
            raise ValueError(f"Zone {zone_name} has no faces")
        
        return [list(self.faces[face_id].node_indices) for face_id in zone.face_indices]
    
    def get_zone_face_types(self, zone_name: str) -> List[str]:
        """Get the types of all faces for a specific zone.
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            List of face types
            
        Raises:
            ValueError: If zone not found or has no faces
        """
        zone = self.get_zone_by_name(zone_name)
        if zone is None:
            raise ValueError(f"Zone not found: {zone_name}")
        
        if not zone.face_indices:
            raise ValueError(f"Zone {zone_name} has no faces")
        
        return [self.faces[face_id].face_type for face_id in zone.face_indices]
    
    def get_zone_cells(self, zone_name: str) -> List[List[int]]:
        """Get all cells for a specific zone.
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            List of cells, where each cell is a list of node indices
            
        Raises:
            ValueError: If zone not found or has no cells
        """
        zone = self.get_zone_by_name(zone_name)
        if zone is None:
            raise ValueError(f"Zone not found: {zone_name}")
        
        if not zone.cell_indices:
            raise ValueError(f"Zone {zone_name} has no cells")
        
        return [list(self.cells[cell_id].node_indices) for cell_id in zone.cell_indices]
    
    def get_zone_cell_types(self, zone_name: str) -> List[str]:
        """Get the types of all cells for a specific zone.
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            List of cell types
            
        Raises:
            ValueError: If zone not found or has no cells
        """
        zone = self.get_zone_by_name(zone_name)
        if zone is None:
            raise ValueError(f"Zone not found: {zone_name}")
        
        if not zone.cell_indices:
            raise ValueError(f"Zone {zone_name} has no cells")
        
        return [self.cells[cell_id].cell_type for cell_id in zone.cell_indices]


class FluentMeshReader:
    """Reader for Fluent mesh files (.msh, .cas) that directly parses the file format."""
    
    # Fluent element types mapping to standard names
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
        # Add more as needed
    }
    
    # Fluent zone types mapping
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
        # Add more as needed
    }
    
    def __init__(self, filename: str, **kwargs):
        """Initialize the FluentMeshReader.
        
        Args:
            filename: Path to the Fluent mesh file
            **kwargs: Additional options (debug: bool, force_ascii: bool, force_binary: bool)
        """
        self.filename = filename
        self.mesh = FluentMesh()
        
        # Extract options
        self.debug = kwargs.get('debug', False)
        self.force_ascii = kwargs.get('force_ascii', False)
        self.force_binary = kwargs.get('force_binary', False)
        
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def read(self) -> FluentMesh:
        """Read the Fluent mesh file and extract points, zones, faces, and cells.
        
        Returns:
            FluentMesh object
            
        Raises:
            FileNotFoundError: If the mesh file does not exist
            ValueError: If the file format is not recognized
        """
        logger.info(f"Reading Fluent mesh file: {self.filename}")
        
        # Check if file exists
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Mesh file not found: {self.filename}")
        
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
            
            # Post-process the mesh to establish connectivity
            self._build_connectivity()
            
        except Exception as e:
            logger.error(f"Error reading Fluent mesh: {e}")
            raise
        
        logger.info(f"Successfully loaded mesh with {len(self.mesh.points)} points, "
                   f"{len(self.mesh.zones)} zones, {len(self.mesh.faces)} faces, "
                   f"{len(self.mesh.cells)} cells")
        
        return self.mesh
    
    def _is_binary_file(self) -> bool:
        """Determine if the mesh file is binary or ASCII.
        
        Returns:
            True if binary, False if ASCII
        """
        try:
            with open(self.filename, 'rb') as file:
                # Read first 100 bytes to check for binary indicators
                header = file.read(100)
                
                # Binary Fluent files typically start with '(', but also contain non-ASCII bytes
                if b'(' in header:
                    # Check for non-ASCII bytes (excluding common control chars)
                    for byte in header:
                        if byte > 127 and byte not in (9, 10, 13):  # tab, LF, CR
                            return True
                    
                    # If we found opening parentheses but no binary data, likely ASCII
                    return False
                else:
                    # If no opening parenthesis found, probably not a valid Fluent file
                    raise ValueError("File does not appear to be a valid Fluent mesh file")
        except Exception as e:
            logger.warning(f"Error determining file type: {e}")
            # Default to ASCII if we can't determine
            return False
    
    def _read_ascii_mesh(self) -> None:
        """Read an ASCII Fluent mesh file."""
        with open(self.filename, 'r') as file:
            content = file.read()
            
            # Extract sections using regex
            self._parse_ascii_mesh(content)
    
    def _parse_ascii_mesh(self, content: str) -> None:
        """Parse the content of an ASCII Fluent mesh file.
        
        Args:
            content: String content of the mesh file
        """
        # Parse header information
        header_match = re.search(r'\(0\s+"([^"]+)"\s+([^)]+)\)', content)
        if header_match:
            version = header_match.group(2).strip()
            self.mesh.metadata['version'] = version
            logger.debug(f"Mesh file version: {version}")
        
        # Parse dimensions
        dimension_match = re.search(r'\(2\s+(\d+)\)', content)
        if dimension_match:
            dimension = int(dimension_match.group(1))
            self.mesh.metadata['dimension'] = dimension
            logger.debug(f"Mesh dimension: {dimension}")
        
        # Parse nodes (points)
        self._parse_nodes(content)
        
        # Parse zone definitions
        self._parse_zones(content)
        
        # Parse faces
        self._parse_faces(content)
        
        # Parse cells
        self._parse_cells(content)
    
    def _parse_nodes(self, content: str) -> None:
        """Parse nodes (points) from the mesh content.
        
        Args:
            content: String content of the mesh file
        """
        # Find node section
        nodes_match = re.search(r'\(10\s+\(0\s+1\s+(\d+)\s+1\s+3\)\s*\n(.*?)\)\s*\n', 
                               content, re.DOTALL)
        
        if not nodes_match:
            logger.warning("No nodes found in mesh file")
            return
        
        node_count = int(nodes_match.group(1))
        nodes_data = nodes_match.group(2).strip()
        
        # Parse node coordinates
        points = []
        for line in nodes_data.split('\n'):
            if line.strip():
                coords = line.strip().split()
                if len(coords) >= 3:  # Ensure we have x, y, z coordinates
                    points.append([float(coords[0]), float(coords[1]), float(coords[2])])
        
        self.mesh.points = np.array(points)
        logger.info(f"Parsed {len(points)} nodes")
    
    def _parse_zones(self, content: str) -> None:
        """Parse zone definitions from the mesh content.
        
        Args:
            content: String content of the mesh file
        """
        # Find zone sections - format: (45 (zone-id zone-type zone-name))
        zone_pattern = r'\(45\s+\((\d+)\s+([a-zA-Z-]+|[0-9]+)\s+"([^"]+)"\s+([^)]*)\)\)'
        zone_matches = re.finditer(zone_pattern, content)
        
        for match in zone_matches:
            zone_id = int(match.group(1))
            zone_type_raw = match.group(2)
            zone_name = match.group(3)
            zone_props = match.group(4).strip()
            
            # Convert numeric zone type to string if needed
            if zone_type_raw.isdigit():
                zone_type = self.ZONE_TYPES.get(int(zone_type_raw), "unknown")
            else:
                zone_type = zone_type_raw
            
            # Create zone
            zone = FluentZone(zone_id, zone_name, zone_type)
            
            # Parse additional zone properties
            if zone_props:
                props = zone_props.split()
                for i in range(0, len(props), 2):
                    if i + 1 < len(props):
                        key = props[i]
                        value = props[i + 1]
                        zone.properties[key] = value
            
            self.mesh.zones[zone_id] = zone
            self.mesh.zone_map[zone_name] = zone_id
            
            logger.debug(f"Added zone: {zone}")
        
        logger.info(f"Parsed {len(self.mesh.zones)} zones")
    
    def _parse_faces(self, content: str) -> None:
        """Parse faces from the mesh content.
        
        Args:
            content: String content of the mesh file
        """
        # Find face sections
        # Format: (13 (zone-id first-face-id last-face-id face-type))
        face_section_pattern = r'\(13\s+\((\d+)\s+(\d+)\s+(\d+)\s+(\d+)\)\s*\n(.*?)\)\s*\n'
        face_sections = re.finditer(face_section_pattern, content, re.DOTALL)
        
        for section in face_sections:
            zone_id = int(section.group(1))
            first_face_id = int(section.group(2))
            last_face_id = int(section.group(3))
            face_type_id = int(section.group(4))
            face_data = section.group(5).strip()
            
            # Determine face type
            face_type = self.ELEMENT_TYPES.get(face_type_id, "unknown")
            
            # Parse face data
            face_lines = face_data.split('\n')
            face_id = first_face_id
            
            for line in face_lines:
                if line.strip():
                    # Parse face nodes
                    node_indices = [int(idx) for idx in line.strip().split()]
                    if node_indices:
                        face = FluentFace(face_id, node_indices, face_type)
                        face.zone_id = zone_id
                        
                        self.mesh.faces[face_id] = face
                        
                        # Add face to its zone
                        if zone_id in self.mesh.zones:
                            self.mesh.zones[zone_id].add_face(face_id)
                        
                        face_id += 1
            
            logger.debug(f"Added {face_id - first_face_id} faces to zone {zone_id}")
        
        logger.info(f"Parsed {len(self.mesh.faces)} faces")
    
    def _parse_cells(self, content: str) -> None:
        """Parse cells from the mesh content.
        
        Args:
            content: String content of the mesh file
        """
        # Find cell sections
        # Format: (12 (zone-id first-cell-id last-cell-id cell-type))
        cell_section_pattern = r'\(12\s+\((\d+)\s+(\d+)\s+(\d+)\s+(\d+)\)\s*\n(.*?)\)\s*\n'
        cell_sections = re.finditer(cell_section_pattern, content, re.DOTALL)
        
        for section in cell_sections:
            zone_id = int(section.group(1))
            first_cell_id = int(section.group(2))
            last_cell_id = int(section.group(3))
            cell_type_id = int(section.group(4))
            cell_data = section.group(5).strip()
            
            # Determine cell type
            cell_type = self.ELEMENT_TYPES.get(cell_type_id, "unknown")
            
            # Parse cell data
            cell_lines = cell_data.split('\n')
            cell_id = first_cell_id
            
            for line in cell_lines:
                if line.strip():
                    # Parse cell nodes
                    node_indices = [int(idx) for idx in line.strip().split()]
                    if node_indices:
                        cell = FluentCell(cell_id, node_indices, cell_type)
                        cell.zone_id = zone_id
                        
                        self.mesh.cells[cell_id] = cell
                        
                        # Add cell to its zone
                        if zone_id in self.mesh.zones:
                            self.mesh.zones[zone_id].add_cell(cell_id)
                            
                            # Add the cell's nodes to the zone as well
                            for node_id in node_indices:
                                self.mesh.zones[zone_id].add_node(node_id)
                        
                        cell_id += 1
            
            logger.debug(f"Added {cell_id - first_cell_id} cells to zone {zone_id}")
        
        logger.info(f"Parsed {len(self.mesh.cells)} cells")
    
    def _read_binary_mesh(self) -> None:
        """Read a binary Fluent mesh file."""
        with open(self.filename, 'rb') as file:
            # Implementation of binary file parser
            logger.warning("Binary mesh parsing not fully implemented yet")
            # This would involve reading the binary format according to Fluent specifications
            # Fluent binary files use a specific encoding for sections and data
    
    def _build_connectivity(self) -> None:
        """Build connectivity between mesh elements (faces, cells, etc.)."""
        # Build connectivity between faces and cells
        for cell_id, cell in self.mesh.cells.items():
            # For each cell, identify its faces
            cell_nodes = set(cell.node_indices)
            
            for face_id, face in self.mesh.faces.items():
                face_nodes = set(face.node_indices)
                
                # If all face nodes are in this cell, the face belongs to this cell
                if face_nodes.issubset(cell_nodes):
                    cell.add_face(face_id)
                    
                    # Set owner cell if not already set
                    if face.owner_cell == -1:
                        face.owner_cell = cell_id
                    else:
                        # If owner already set, this cell is a neighbor
                        face.neighbor_cell = cell_id
        
        logger.debug("Built mesh connectivity")
        
        # For any zones without nodes, add nodes from their faces/cells
        for zone_id, zone in self.mesh.zones.items():
            if not zone.node_indices:
                # Add nodes from faces
                for face_id in zone.face_indices:
                    if face_id in self.mesh.faces:
                        face = self.mesh.faces[face_id]
                        for node_id in face.node_indices:
                            zone.add_node(node_id)
                
                # Add nodes from cells
                for cell_id in zone.cell_indices:
                    if cell_id in self.mesh.cells:
                        cell = self.mesh.cells[cell_id]
                        for node_id in cell.node_indices:
                            zone.add_node(node_id)
        
        logger.debug("Updated zone node information")


def read_fluent_mesh(filename: str, **kwargs) -> FluentMesh:
    """Read a Fluent mesh file and return a FluentMesh object.
    
    This is a convenience function that creates a FluentMeshReader and reads the mesh.
    
    Args:
        filename: Path to the Fluent mesh file
        **kwargs: Additional options (debug: bool, force_ascii: bool, force_binary: bool)
        
    Returns:
        FluentMesh object
        
    Raises:
        FileNotFoundError: If the mesh file does not exist
        ValueError: If the file format is not recognized
    """
    reader = FluentMeshReader(filename, **kwargs)
    return reader.read()
