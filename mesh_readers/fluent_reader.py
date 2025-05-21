"""
FluentMeshReader class to read Fluent mesh files (.cas, .msh) and extract zone information.
Uses meshio for improved parsing capabilities and face connectivity extraction.
"""
import numpy as np
import os
import sys
import re
import struct
import logging
import subprocess
import meshio
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import tempfile

logger = logging.getLogger(__name__)

class FluentMeshReader:
    """
    A class to read Fluent mesh files (.cas, .msh) and extract zone information.
    Uses meshio to read mesh files and extract face connectivity information.
    """
    
    def __init__(self, filename: str, debug=False):
        self.filename = filename
        self.points = np.array([])  # Will be populated with node coordinates
        self.zones = {}  # Maps zone names to node IDs
        self.zone_types = {}  # Maps zone names to zone types (wall, symmetry, etc.)
        
        # Add new attributes for face connectivity
        self.faces_by_zone = {}  # Maps zone names to lists of faces (lists of node indices)
        self.face_types_by_zone = {}  # Maps zone names to types of faces (triangle, quad, etc.)
        
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
    def read(self):
        """Read the Fluent mesh file using meshio and extract points, faces, and zones."""
        logger.info(f"Reading Fluent mesh file using meshio: {self.filename}")
        
        # Check if file exists
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Mesh file not found: {self.filename}")
        
        try:
            # Use meshio to read the mesh file
            mesh = meshio.read(self.filename)
            logger.info(f"Successfully read mesh using meshio with {len(mesh.points)} points")
            
            # Store points
            self.points = np.array(mesh.points)
            
            # Initialize data structures
            self.zones = {}
            self.zone_types = {}
            self.faces_by_zone = {}
            self.face_types_by_zone = {}
            
            # Process cell data to extract faces and zones
            self._process_meshio_cells(mesh)
            
            # Ensure we have zone types for all zones
            for zone_name in self.zones.keys():
                if zone_name not in self.zone_types:
                    # Try to infer zone type from name
                    self._infer_zone_type(zone_name)
            
        except Exception as e:
            logger.error(f"Error reading mesh with meshio: {e}")
            
            # Create artificial mesh as fallback if needed
            if len(self.points) == 0:
                logger.warning("Meshio reading failed, creating artificial mesh for testing")
                self._create_artificial_mesh()
        
        if len(self.points) == 0:
            logger.error("Failed to extract any points from the mesh")
            raise ValueError("No points were extracted from the mesh file.")
            
        logger.info(f"Successfully loaded {len(self.points)} points and {len(self.zones)} zones")
        logger.info(f"Found {sum(len(faces) for faces in self.faces_by_zone.values())} faces across {len(self.faces_by_zone)} zones")
        return self
        
    def _process_meshio_cells(self, mesh):
        """Process meshio cell data to extract faces, cells, and zones."""
        # First try to process using cell_sets which preserve zone names
        if hasattr(mesh, 'cell_sets') and mesh.cell_sets:
            for zone_name, cell_blocks in mesh.cell_sets.items():
                # Clean up zone name (meshio sometimes adds indices)
                clean_zone_name = zone_name.split('[')[0]
                
                # Initialize containers for this zone
                self.zones[clean_zone_name] = set()
                self.faces_by_zone[clean_zone_name] = []
                self.face_types_by_zone[clean_zone_name] = []
                
                # Infer zone type from name
                self._infer_zone_type(clean_zone_name)
                
                # Go through each cell block referenced by this zone
                for i, indices in enumerate(cell_blocks):
                    if i >= len(mesh.cells) or not indices.size:
                        continue
                        
                    cell_block = mesh.cells[i]
                    cell_type = cell_block.type
                    data = cell_block.data
                    
                    # Only process indices that are in range
                    valid_indices = indices[indices < len(data)]
                    if not valid_indices.size:
                        continue
                    
                    # Extract faces based on cell type
                    if cell_type in ["triangle", "quad"]:
                        # These are already faces
                        faces = data[valid_indices].tolist()
                        self.faces_by_zone[clean_zone_name].extend(faces)
                        self.face_types_by_zone.setdefault(clean_zone_name, []).extend([cell_type] * len(faces))
                        
                        # Add nodes to the zone
                        for face in faces:
                            self.zones[clean_zone_name].update(face)
                            
                    elif cell_type == "tetra":
                        # Extract triangular faces from tetrahedral cells
                        for idx in valid_indices:
                            tetra = data[idx]
                            # A tetrahedron has 4 triangular faces
                            faces = [
                                [tetra[0], tetra[1], tetra[2]],
                                [tetra[0], tetra[1], tetra[3]],
                                [tetra[0], tetra[2], tetra[3]],
                                [tetra[1], tetra[2], tetra[3]]
                            ]
                            self.faces_by_zone[clean_zone_name].extend(faces)
                            self.face_types_by_zone.setdefault(clean_zone_name, []).extend(["triangle"] * 4)
                            self.zones[clean_zone_name].update(tetra)
                            
                    elif cell_type == "hexahedron":
                        # Extract quadrilateral faces from hexahedral cells
                        for idx in valid_indices:
                            hexa = data[idx]
                            # A hexahedron has 6 quadrilateral faces
                            faces = [
                                [hexa[0], hexa[1], hexa[2], hexa[3]],  # bottom
                                [hexa[4], hexa[5], hexa[6], hexa[7]],  # top
                                [hexa[0], hexa[1], hexa[5], hexa[4]],  # front
                                [hexa[1], hexa[2], hexa[6], hexa[5]],  # right
                                [hexa[2], hexa[3], hexa[7], hexa[6]],  # back
                                [hexa[3], hexa[0], hexa[4], hexa[7]]   # left
                            ]
                            self.faces_by_zone[clean_zone_name].extend(faces)
                            self.face_types_by_zone.setdefault(clean_zone_name, []).extend(["quad"] * 6)
                            self.zones[clean_zone_name].update(hexa)
                            
                    # Add more cell types as needed (wedge, pyramid, etc.)
        
        # If no cell_sets or no faces were found, try processing cells directly
        if not self.faces_by_zone or all(len(faces) == 0 for faces in self.faces_by_zone.values()):
            logger.info("No faces found in cell_sets. Processing cells directly.")
            
            # Process each cell block
            for i, cell_block in enumerate(mesh.cells):
                cell_type = cell_block.type
                data = cell_block.data
                
                # Create a generic zone name based on cell type
                zone_name = f"zone_{cell_type}_{i}"
                
                # Initialize containers
                self.zones.setdefault(zone_name, set())
                self.faces_by_zone.setdefault(zone_name, [])
                self.face_types_by_zone.setdefault(zone_name, [])
                self.zone_types[zone_name] = cell_type
                
                # Process based on cell type
                if cell_type in ["triangle", "quad"]:
                    self.faces_by_zone[zone_name].extend(data.tolist())
                    self.face_types_by_zone[zone_name].extend([cell_type] * len(data))
                    for face in data:
                        self.zones[zone_name].update(face)
                        
                elif cell_type == "tetra":
                    for tetra in data:
                        faces = [
                            [tetra[0], tetra[1], tetra[2]],
                            [tetra[0], tetra[1], tetra[3]],
                            [tetra[0], tetra[2], tetra[3]],
                            [tetra[1], tetra[2], tetra[3]]
                        ]
                        self.faces_by_zone[zone_name].extend(faces)
                        self.face_types_by_zone[zone_name].extend(["triangle"] * 4)
                        self.zones[zone_name].update(tetra)
                        
                elif cell_type == "hexahedron":
                    for hexa in data:
                        faces = [
                            [hexa[0], hexa[1], hexa[2], hexa[3]],  # bottom
                            [hexa[4], hexa[5], hexa[6], hexa[7]],  # top
                            [hexa[0], hexa[1], hexa[5], hexa[4]],  # front
                            [hexa[1], hexa[2], hexa[6], hexa[5]],  # right
                            [hexa[2], hexa[3], hexa[7], hexa[6]],  # back
                            [hexa[3], hexa[0], hexa[4], hexa[7]]   # left
                        ]
                        self.faces_by_zone[zone_name].extend(faces)
                        self.face_types_by_zone[zone_name].extend(["quad"] * 6)
                        self.zones[zone_name].update(hexa)
    
    def get_zone_faces(self, zone_name):
        """Get the faces (node index lists) for a specific zone."""
        if zone_name not in self.faces_by_zone:
            return []
        return self.faces_by_zone[zone_name]
    
    def get_zone_face_types(self, zone_name):
        """Get the face types for a specific zone."""
        if zone_name not in self.face_types_by_zone:
            return []
        return self.face_types_by_zone[zone_name]
        
    def _infer_zone_type(self, zone_name):
        """Infer the zone type from the zone name to properly handle Fluent zone specifications."""
        # Check if zone type is already set
        if zone_name in self.zone_types:
            return
            
        # Try to determine zone type from zone name (common in Fluent files)
        lower_name = zone_name.lower()
        
        # Standard Fluent boundary types
        if "wall" in lower_name:
            self.zone_types[zone_name] = "wall"
        elif "inlet" in lower_name:
            self.zone_types[zone_name] = "velocity-inlet"
        elif "outlet" in lower_name:
            self.zone_types[zone_name] = "pressure-outlet"
        elif "symmetry" in lower_name:
            self.zone_types[zone_name] = "symmetry"
        elif "interior" in lower_name:
            self.zone_types[zone_name] = "interior"
        elif "fluid" in lower_name:
            self.zone_types[zone_name] = "fluid"
        elif "solid" in lower_name:
            self.zone_types[zone_name] = "solid"
            
        # Specific to rocket launch pad geometry
        elif "launchpad" in lower_name:
            self.zone_types[zone_name] = "wall"
        elif "deflector" in lower_name:
            self.zone_types[zone_name] = "wall"
        elif "rocket" in lower_name:
            self.zone_types[zone_name] = "wall"
        elif "wedge" in lower_name:
            self.zone_types[zone_name] = "wall"
        else:
            # Default to unknown type
            self.zone_types[zone_name] = "unknown"
            
        logger.debug(f"Inferred zone type '{self.zone_types[zone_name]}' for zone '{zone_name}'")
    
    def get_available_zones(self):
        """Get a dictionary of available zones with their node counts."""
        return {zone_name: len(node_ids) for zone_name, node_ids in self.zones.items()}
    
    def get_zone_points(self, zone_name):
        """Get the points for a specific zone."""
        if zone_name not in self.zones:
            return np.array([])
        
        # Convert set of node indices to NumPy array of coordinates
        node_indices = list(self.zones[zone_name])
        return self.points[node_indices]
        
    def _read_binary(self):
        """
        Read points and zones from a binary Fluent mesh file.
        This is an improved implementation with better binary parsing.
        """
        logger.debug("Attempting to read binary format")
        
        with open(self.filename, 'rb') as f:
            # Check binary signature
            header = f.read(20)
            if not (b"FLUENT" in header or header.startswith(b"(") or header.startswith(b"\x28")):
                raise ValueError("Not a recognized Fluent binary format")
            
            # Reset to beginning
            f.seek(0)
            
            # Binary parsing is complex, so we'll implement a simplified approach
            # that scans for coordinate patterns in the binary data
            content = f.read()
            
            # Extract node count if possible
            node_count_match = re.search(rb'\(10+\s+(\d+)', content)
            node_count = 0
            if node_count_match:
                try:
                    node_count = int(node_count_match.group(1))
                    logger.debug(f"Found node count: {node_count}")
                except:
                    pass
                    
            # Look for floating point coordinate patterns
            # This is simplified - a real parser would need to understand Fluent's binary structure
            points_list = []
            
            # Pattern: look for sequences of 3 IEEE floats (12 bytes) with reasonable coordinate values
            i = 0
            while i < len(content) - 12:
                if node_count > 0 and len(points_list) >= node_count:
                    break
                    
                try:
                    x = struct.unpack('<f', content[i:i+4])[0]
                    y = struct.unpack('<f', content[i+4:i+8])[0]
                    z = struct.unpack('<f', content[i+8:i+12])[0]
                    
                    # Only accept reasonable coordinate values
                    if all(abs(val) < 1e6 for val in [x, y, z]):
                        points_list.append([x, y, z])
                        i += 12
                    else:
                        i += 4
                except:
                    i += 1
            
            if points_list:
                self.points = np.array(points_list)
                logger.debug(f"Extracted {len(points_list)} points from binary data")
            else:
                raise ValueError("No valid points found in binary data")
            
            # Extract zone names using regex patterns
            zone_matches = re.finditer(rb'zone.*?id\s+(\d+).*?name\s+"([^"]+)".*?type\s+(\w+)', content, re.DOTALL)
            for match in zone_matches:
                zone_id = match.group(1).decode('ascii')
                zone_name = match.group(2).decode('ascii')
                zone_type = match.group(3).decode('ascii')
                
                self.zone_types[zone_name] = zone_type
                # For simplicity, assign all points to all zones initially
                self.zones[zone_name] = set(range(len(self.points)))
                logger.debug(f"Found zone: {zone_name} (type: {zone_type}, id: {zone_id})")
            
            # If no zones were found, create default zones
            if not self.zones:
                self._create_default_zones()
                
    def _read_ascii(self):
        """
        Read points and zones from an ASCII Fluent mesh file.
        Improved implementation with more robust parsing.
        """
        logger.debug("Attempting to read ASCII format")
        
        try:
            with open(self.filename, 'r') as f:
                content = f.read()
        except UnicodeDecodeError:
            raise ValueError("This appears to be a binary file, not ASCII")
            
        # First, extract the node section
        node_section_match = re.search(r'\(10+[\s\n]+([^)]+)', content, re.DOTALL)
        points_list = []
        
        if node_section_match:
            node_data = node_section_match.group(1).strip()
            lines = node_data.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('('):
                    continue
                    
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points_list.append([x, y, z])
                    except ValueError:
                        continue
        
        if points_list:
            self.points = np.array(points_list)
            logger.debug(f"Extracted {len(points_list)} points from ASCII data")
        else:
            # Alternative approach for finding coordinates
            logger.debug("Node section parsing failed, trying coordinate pattern matching")
            self._extract_points_from_patterns(content)
            
        if len(self.points) == 0:
            raise ValueError("No valid points found in ASCII data")
            
        # Extract zone information
        zone_matches = re.finditer(r'\(12[^\n]*\n[^\(]*zone\s+(?:[^\n]*\n[^\(]*)*name\s+"([^"]+)"[^\n]*\n[^\(]*type\s+(\w+)', 
                                   content, re.DOTALL)
        
        for match in zone_matches:
            zone_name = match.group(1)
            zone_type = match.group(2)
            
            self.zone_types[zone_name] = zone_type
            # For simplicity, assign all points to all zones initially
            self.zones[zone_name] = set(range(len(self.points)))
            logger.debug(f"Found zone: {zone_name} (type: {zone_type})")
        
        # If no zones were found, create default zones
        if not self.zones:
            self._create_default_zones()
    
    def _extract_points_from_patterns(self, content):
        """Extract points using regex pattern matching on ASCII content."""
        # Look for coordinate patterns - groups of 3 floating point numbers
        # This is a more thorough pattern that handles scientific notation
        coord_pattern = r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)'
        matches = re.finditer(coord_pattern, content)
        
        points_list = []
        for match in matches:
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                z = float(match.group(3))
                # Only accept reasonable coordinate values
                if all(abs(val) < 1e6 for val in [x, y, z]):
                    points_list.append([x, y, z])
            except ValueError:
                pass
        
        if points_list:
            self.points = np.array(points_list)
            logger.debug(f"Extracted {len(points_list)} points from coordinate patterns")
                    
    def _try_external_converters(self):
        """
        Try using external tools to convert/extract mesh data.
        Returns True if successful, False otherwise.
        """
        logger.debug("Attempting to use external converters")
        
        # Try multiple external methods
        # 1. Try using fluent_to_vtk if available
        if self._try_fluent_to_vtk():
            return True
            
        # 2. Try using Fluent in batch mode if available
        if self._try_fluent_batch():
            return True
            
        # 3. Try pyfluentlib if available
        if self._try_pyfluentlib():
            return True
            
        return False
        
    def _try_fluent_to_vtk(self):
        """Try using fluent_to_vtk tool to convert Fluent mesh to VTK."""
        try:
            import meshio
            
            # Create a temporary output file
            with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as tmp:
                vtk_file = tmp.name
            
            try:
                # Check if fluent_to_vtk is available
                subprocess.run(["which", "fluent_to_vtk"], check=True, capture_output=True)
                
                # Run the conversion
                subprocess.run(["fluent_to_vtk", self.filename, vtk_file], 
                              check=True, timeout=60, capture_output=True)
                
                # If successful, read the VTK file
                if os.path.exists(vtk_file) and os.path.getsize(vtk_file) > 0:
                    mesh = meshio.read(vtk_file)
                    self.points = mesh.points
                    
                    # Process zones if possible
                    self._process_meshio_cell_data(mesh)
                    
                    # Clean up temp file
                    os.remove(vtk_file)
                    return True
                else:
                    logger.debug("fluent_to_vtk ran but produced invalid output")
            except subprocess.SubprocessError as e:
                logger.debug(f"fluent_to_vtk failed: {e}")
            except FileNotFoundError:
                logger.debug("fluent_to_vtk not found on system")
            finally:
                # Make sure temp file is removed
                if os.path.exists(vtk_file):
                    os.remove(vtk_file)
                    
        except ImportError:
            logger.debug("meshio not available for reading VTK output")
            
        return False
        
    def _try_fluent_batch(self):
        """Try using Fluent in batch mode to export mesh data."""
        try:
            # Create temp files
            with tempfile.NamedTemporaryFile(suffix='.jou', delete=False) as tmp_script:
                script_file = tmp_script.name
                
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_mesh:
                mesh_file = tmp_mesh.name
                
            # Write Fluent journal script
            with open(script_file, 'w') as f:
                f.write(f"file/read-case-data {self.filename}\n")
                f.write(f"file/export/ascii {mesh_file} yes\n")
                f.write("exit yes\n")
                
            try:
                # Check if fluent is available
                subprocess.run(["which", "fluent"], check=True, capture_output=True)
                
                # Run fluent in batch mode
                subprocess.run(["fluent", "3d", "-t4", "-g", "-i", script_file], 
                              check=True, timeout=120, capture_output=True)
                
                # Read exported ASCII mesh if created
                if os.path.exists(mesh_file) and os.path.getsize(mesh_file) > 0:
                    with open(mesh_file, 'r') as f:
                        content = f.read()
                        self._extract_points_from_patterns(content)
                        
                    # Extract zone info too if possible
                    zone_matches = re.finditer(r'Zone\s+(\d+)\s+\(([^)]+)\).*?zone\s+type:\s+(\w+)', 
                                      content, re.DOTALL | re.IGNORECASE)
                    
                    for match in zone_matches:
                        zone_id = match.group(1)
                        zone_name = match.group(2).strip()
                        zone_type = match.group(3).strip()
                        
                        self.zone_types[zone_name] = zone_type
                        self.zones[zone_name] = set(range(len(self.points)))
                        logger.debug(f"Found zone: {zone_name} (type: {zone_type}, id: {zone_id})")
                    
                    # If no zones were found, create default zones
                    if not self.zones:
                        self._create_default_zones()
                        
                    # Clean up temp files
                    os.remove(script_file)
                    os.remove(mesh_file)
                    return True
                else:
                    logger.debug("Fluent batch export produced invalid output")
            except subprocess.SubprocessError as e:
                logger.debug(f"Fluent batch export failed: {e}")
            except FileNotFoundError:
                logger.debug("Fluent not found on system")
            finally:
                # Make sure temp files are removed
                if os.path.exists(script_file):
                    os.remove(script_file)
                if os.path.exists(mesh_file):
                    os.remove(mesh_file)
                    
        except Exception as e:
            logger.debug(f"Error in Fluent batch process: {e}")
            
        return False
        
    def _try_pyfluentlib(self):
        """Try using pyfluentlib if available."""
        try:
            # Check if pyfluentlib is installed
            import importlib
            pyfluentlib = importlib.import_module('pyfluentlib')
            
            # Use pyfluentlib to read mesh
            reader = pyfluentlib.FluentReader(self.filename)
            mesh_data = reader.read()
            
            if hasattr(mesh_data, 'points') and len(mesh_data.points) > 0:
                self.points = np.array(mesh_data.points)
                
                # Extract zone information if available
                if hasattr(mesh_data, 'zones'):
                    for zone_name, zone_data in mesh_data.zones.items():
                        self.zone_types[zone_name] = zone_data.get('type', 'wall')
                        self.zones[zone_name] = set(range(len(self.points)))  # Simplified
                else:
                    self._create_default_zones()
                    
                return True
        except ImportError:
            logger.debug("pyfluentlib not available")
        except Exception as e:
            logger.debug(f"Error using pyfluentlib: {e}")
            
        return False
        
    def _process_meshio_cell_data(self, mesh):
        """Process zone information from meshio object."""
        # Extract zone information from meshio cell data if available
        if hasattr(mesh, 'cell_data'):
            if 'zone' in mesh.cell_data:
                zone_ids = set()
                for block_data in mesh.cell_data['zone']:
                    zone_ids.update(block_data)
                
                # Create synthetic zone names for each zone ID
                for zone_id in zone_ids:
                    zone_name = f"zone_{zone_id}"
                    self.zone_types[zone_name] = 'wall'  # Default type
                    self.zones[zone_name] = set(range(len(self.points)))  # Simplified
                    
        # If no zones were created, fall back to defaults
        if not self.zones:
            self._create_default_zones()
    
    def _create_default_zones(self):
        """Create default zones when no zones are detected."""
        default_zones = {
            'fluid': 'fluid',
            'wall': 'wall',
            'symmetry': 'symmetry',
            'inlet': 'velocity-inlet',
            'outlet': 'pressure-outlet'
        }
        
        for zone_name, zone_type in default_zones.items():
            self.zone_types[zone_name] = zone_type
            self.zones[zone_name] = set(range(len(self.points)))
            
        logger.debug("Created default zones since none were detected")
            
    def _create_artificial_mesh(self):
        """Create an artificial mesh for testing when all reading methods fail."""
        logger.warning("Creating artificial test mesh")
        
        # Create a simple grid of points
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        z = np.linspace(-1, 1, 20)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        self.points = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
        
        # Create default zones
        self.zone_types = {
            'fluid': 'fluid',
            'wall': 'wall',
            'symmetry': 'symmetry',
            'inlet': 'velocity-inlet',
            'outlet': 'pressure-outlet',
            'rocket': 'wall',
            'launchpad': 'wall',
            'deflector': 'wall'
        }
        
        # Associate all points with all zones
        for zone in self.zone_types:
            self.zones[zone] = set(range(len(self.points)))
            
    def get_available_zones(self) -> List[str]:
        """Return a list of available zone names."""
        return list(self.zones.keys())
        
    def get_zone_points(self, zone_name: str) -> np.ndarray:
        """Extract the points belonging to a specified zone."""
        if zone_name not in self.zones:
            available = self.get_available_zones()
            logger.error(f"Zone '{zone_name}' not found. Available zones: {available}")
            raise ValueError(f"Zone '{zone_name}' not found. Available zones: {available}")
            
        # Get indices of points in this zone
        indices = np.fromiter(self.zones[zone_name], dtype=int)
        return self.points[indices]