"""
General mesh reader functions for non-Fluent mesh formats using meshio.

This module provides functionality to read various mesh formats using the meshio library.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Check for meshio availability
try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    logger.warning("meshio not available. Install with 'pip install meshio' for non-Fluent mesh support.")


def is_fluent_mesh(filename: str) -> bool:
    """Check if the given file appears to be a Fluent mesh file.
    
    Args:
        filename: Path to the mesh file
        
    Returns:
        bool: True if the file appears to be a Fluent mesh, False otherwise
    """
    ext = os.path.splitext(filename)[1].lower()
    
    # Quick check by extension
    if ext in ('.cas', '.msh'):
        # Try to detect Fluent mesh format by content
        try:
            with open(filename, 'rb') as f:  # Read in binary mode
                header = f.read(100)  # Read just the first part of the file
                if b'FLUENT' in header or b'(10' in header:
                    return True
                    
            # Also check if readable as text for ASCII format
            try:
                with open(filename, 'r') as f:
                    header = f.read(100)
                    if '(10' in header or 'FLUENT' in header:
                        return True
            except UnicodeDecodeError:
                # Already checked binary format above
                pass
        except Exception as e:
            logger.debug(f"Error checking if file is Fluent mesh: {e}")
    
    return False


def read_general_mesh(filename: str) -> Any:
    """Read a non-Fluent mesh file using meshio.
    
    Args:
        filename: Path to the mesh file
        
    Returns:
        meshio.Mesh: The loaded mesh
        
    Raises:
        ImportError: If meshio is not available
        ValueError: If the mesh format is not supported
    """
    if not MESHIO_AVAILABLE:
        raise ImportError("meshio is required to read non-Fluent mesh formats. Install it with 'pip install meshio'")
    
    logger.info(f"Reading mesh file: {filename}")
    
    # Check file existence
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Mesh file not found: {filename}")
    
    # Determine file format
    file_format = None
    ext = os.path.splitext(filename)[1].lower()
    
    # Map common extensions to meshio formats
    format_map = {
        '.vtk': 'vtk',
        '.vtu': 'vtu',
        '.stl': 'stl',
        '.obj': 'obj',
        '.ply': 'ply',
        '.off': 'off',
        '.gmsh': 'gmsh',
        '.msh': 'gmsh',  # Gmsh also uses .msh
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
    
    # Attempt to read using meshio's standard reader
    mesh = None
    try:
        # Try to read with auto format detection first
        mesh = meshio.read(filename, file_format)
        logger.info(f"Successfully read mesh with standard reader: {len(mesh.points)} points, {sum(len(c.data) for c in mesh.cells)} cells")
    except Exception as e:
        logger.warning(f"Error reading mesh with meshio standard reader: {e}")
        
        # Try alternative format if explicit format was provided
        if file_format:
            try:
                logger.info(f"Trying to read mesh with explicit format: {file_format}")
                mesh = meshio.read(filename, file_format)
                logger.info(f"Successfully read mesh with explicit format: {len(mesh.points)} points, {sum(len(c.data) for c in mesh.cells)} cells")
            except Exception as e2:
                logger.warning(f"Error reading mesh with explicit format {file_format}: {e2}")
    
    # If meshio fails to read the mesh or reads it incorrectly (0 points/cells),
    # try alternate approaches
    if mesh is None or (hasattr(mesh, 'points') and len(mesh.points) == 0):
        # Try using PyVista as a fallback if available
        try:
            import pyvista as pv
            logger.info("Attempting to read mesh with PyVista as fallback")
            pv_mesh = pv.read(filename)
            
            # Convert PyVista mesh to meshio format
            points = pv_mesh.points
            cells = []
            
            # Handle different cell types
            if hasattr(pv_mesh, 'faces') and pv_mesh.faces is not None and len(pv_mesh.faces) > 0:
                # Extract triangular cells from faces (for surface meshes)
                i = 0
                triangles = []
                faces = pv_mesh.faces
                while i < len(faces):
                    face_size = faces[i]
                    if face_size == 3:  # Triangle
                        triangles.append([faces[i+1], faces[i+2], faces[i+3]])
                    elif face_size == 4:  # Quad
                        # Convert quad to two triangles
                        triangles.append([faces[i+1], faces[i+2], faces[i+3]])
                        triangles.append([faces[i+1], faces[i+3], faces[i+4]])
                    i += face_size + 1
                
                if triangles:
                    cells.append(meshio.CellBlock("triangle", np.array(triangles)))
            
            # Handle unstructured grids
            if hasattr(pv_mesh, 'celltypes') and pv_mesh.celltypes is not None:
                # Map VTK cell types to meshio cell types
                vtk_to_meshio_type = {
                    1: "vertex",
                    3: "line",
                    5: "triangle",
                    9: "quad",
                    10: "tetra",
                    12: "hexahedron",
                    13: "wedge",
                    14: "pyramid"
                }
                
                # Group cells by type
                cell_groups = {}
                for i, cell_type in enumerate(pv_mesh.celltypes):
                    if cell_type in vtk_to_meshio_type:
                        meshio_type = vtk_to_meshio_type[cell_type]
                        if meshio_type not in cell_groups:
                            cell_groups[meshio_type] = []
                        offset = pv_mesh.offset[i]
                        if i < len(pv_mesh.offset) - 1:
                            next_offset = pv_mesh.offset[i+1]
                        else:
                            next_offset = len(pv_mesh.cells)
                        cell_conn = pv_mesh.cells[offset:next_offset]
                        cell_groups[meshio_type].append(cell_conn)
                
                # Create cell blocks
                for cell_type, cell_list in cell_groups.items():
                    cells.append(meshio.CellBlock(cell_type, np.array(cell_list)))
            
            # Create a meshio mesh
            if len(points) > 0:
                mesh = meshio.Mesh(points, cells)
                logger.info(f"Successfully read mesh with PyVista: {len(mesh.points)} points, {sum(len(c.data) for c in mesh.cells if hasattr(c, 'data'))} cells")
        except ImportError:
            logger.warning("PyVista not available for fallback mesh reading. Install with 'pip install pyvista'")
        except Exception as e:
            logger.warning(f"Error reading mesh with PyVista fallback: {e}")
    
    # If all attempts fail or read an empty mesh, try a last resort
    if mesh is None or (hasattr(mesh, 'points') and len(mesh.points) == 0):
        # Try using vtk directly if available
        try:
            import vtk
            from vtk.util.numpy_support import vtk_to_numpy
            
            logger.info("Attempting to read mesh with VTK as last resort")
            
            # Determine reader based on file extension
            reader = None
            if ext == '.vtk':
                reader = vtk.vtkPolyDataReader()
            elif ext == '.vtu':
                reader = vtk.vtkXMLUnstructuredGridReader()
            elif ext == '.stl':
                reader = vtk.vtkSTLReader()
            elif ext == '.obj':
                reader = vtk.vtkOBJReader()
            elif ext == '.ply':
                reader = vtk.vtkPLYReader()
            elif ext in ('.msh', '.gmsh'):
                # VTK doesn't have a native GMSH reader, but we can try
                # to use ParaView's glue code if it's available
                try:
                    reader = vtk.vtkGMSHReader()
                except AttributeError:
                    logger.warning("VTK does not have a GMSH reader available")
            
            if reader is not None:
                reader.SetFileName(filename)
                reader.Update()
                
                # Get the output data
                if hasattr(reader, 'GetOutput'):
                    data = reader.GetOutput()
                    
                    # Extract points
                    vtk_points = data.GetPoints()
                    if vtk_points is not None:
                        points_array = vtk_to_numpy(vtk_points.GetData())
                        
                        # Extract cells
                        cells = []
                        for i in range(data.GetNumberOfCells()):
                            vtk_cell = data.GetCell(i)
                            cell_type = vtk_cell.GetCellType()
                            cell_points = [vtk_cell.GetPointId(j) for j in range(vtk_cell.GetNumberOfPoints())]
                            
                            # Map VTK cell types to meshio cell types
                            meshio_type = "vertex"  # default
                            if cell_type == 1:  # VTK_VERTEX
                                meshio_type = "vertex"
                            elif cell_type == 3:  # VTK_LINE
                                meshio_type = "line"
                            elif cell_type == 5:  # VTK_TRIANGLE
                                meshio_type = "triangle"
                            elif cell_type == 9:  # VTK_QUAD
                                meshio_type = "quad"
                            elif cell_type == 10:  # VTK_TETRA
                                meshio_type = "tetra"
                            elif cell_type == 12:  # VTK_HEXAHEDRON
                                meshio_type = "hexahedron"
                            elif cell_type == 13:  # VTK_WEDGE
                                meshio_type = "wedge"
                            elif cell_type == 14:  # VTK_PYRAMID
                                meshio_type = "pyramid"
                            
                            # Add to the appropriate cell block
                            found = False
                            for block in cells:
                                if block.type == meshio_type:
                                    block.data.append(cell_points)
                                    found = True
                                    break
                            
                            if not found:
                                cells.append(meshio.CellBlock(meshio_type, [cell_points]))
                        
                        # Create meshio mesh
                        mesh = meshio.Mesh(points_array, cells)
                        logger.info(f"Successfully read mesh with VTK: {len(mesh.points)} points, {sum(len(c.data) for c in mesh.cells if hasattr(c, 'data'))} cells")
        except ImportError:
            logger.warning("VTK not available for last resort mesh reading. Install with 'pip install vtk'")
        except Exception as e:
            logger.warning(f"Error reading mesh with VTK last resort: {e}")
    
    # If we still don't have a valid mesh, create a minimal one from raw file data
    if mesh is None or (hasattr(mesh, 'points') and len(mesh.points) == 0):
        # For MSH format, try a simple parser
        if ext.lower() == '.msh':
            try:
                logger.info("Attempting simple MSH format parsing")
                points = []
                triangles = []
                quads = []
                tetrahedra = []
                hexahedra = []
                
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    
                    # Parse the file
                    section = None
                    i = 0
                    while i < len(lines):
                        line = lines[i].strip()
                        i += 1
                        
                        if line.startswith('$'):
                            section = line[1:].lower()
                            continue
                        
                        if section == 'nodes':
                            try:
                                num_nodes = int(line)
                                for j in range(num_nodes):
                                    if i < len(lines):
                                        parts = lines[i].strip().split()
                                        i += 1
                                        if len(parts) >= 4:
                                            # Gmsh format: node_id x y z
                                            points.append([float(parts[1]), float(parts[2]), float(parts[3])])
                            except Exception as e:
                                logger.warning(f"Error parsing MSH nodes section: {e}")
                        
                        elif section == 'elements':
                            try:
                                num_elements = int(line)
                                for j in range(num_elements):
                                    if i < len(lines):
                                        parts = lines[i].strip().split()
                                        i += 1
                                        if len(parts) >= 3:
                                            # Gmsh format: elem_id elem_type [tags] node_ids...
                                            elem_type = int(parts[1])
                                            num_tags = int(parts[2])
                                            
                                            # Skip tags to get node indices
                                            node_ids_start = 3 + num_tags
                                            node_ids = [int(parts[k]) - 1 for k in range(node_ids_start, len(parts))]
                                            
                                            # Element types in Gmsh
                                            if elem_type == 2:  # Triangle
                                                triangles.append(node_ids)
                                            elif elem_type == 3:  # Quad
                                                quads.append(node_ids)
                                            elif elem_type == 4:  # Tetrahedron
                                                tetrahedra.append(node_ids)
                                            elif elem_type == 5:  # Hexahedron
                                                hexahedra.append(node_ids)
                            except Exception as e:
                                logger.warning(f"Error parsing MSH elements section: {e}")
                
                # Create a meshio mesh
                cells = []
                if triangles:
                    cells.append(meshio.CellBlock("triangle", np.array(triangles)))
                if quads:
                    cells.append(meshio.CellBlock("quad", np.array(quads)))
                if tetrahedra:
                    cells.append(meshio.CellBlock("tetra", np.array(tetrahedra)))
                if hexahedra:
                    cells.append(meshio.CellBlock("hexahedron", np.array(hexahedra)))
                
                if points:
                    mesh = meshio.Mesh(np.array(points), cells)
                    logger.info(f"Successfully read mesh with simple MSH parser: {len(mesh.points)} points, {sum(len(c.data) for c in mesh.cells if hasattr(c, 'data'))} cells")
            except Exception as e:
                logger.warning(f"Error with simple MSH format parsing: {e}")
    
    # If we still have no valid mesh or an empty mesh, give up
    if mesh is None or (hasattr(mesh, 'points') and len(mesh.points) == 0):
        # Let's be more informative about why the mesh might not be readable
        error_msg = f"Failed to read mesh file {filename} with any available method. "
        error_msg += "The file may be corrupted, in an unsupported format, or use features not supported by the available readers. "
        error_msg += "Try converting the mesh to a more standard format like .vtk or .stl using ParaView or another mesh conversion tool."
        raise ValueError(error_msg)
    
    return mesh


def extract_patch_points(mesh_data: Any, patch_name: str) -> np.ndarray:
    """Extract all unique point coordinates belonging to the given cell set/patch.
    
    Works with both Fluent mesh objects and meshio objects.
    
    Args:
        mesh_data: Mesh object (either FluentMeshReader or meshio.Mesh)
        patch_name: Name of zone, cell set/patch or Gmsh physical group
        
    Returns:
        np.ndarray: Array of point coordinates
        
    Raises:
        ValueError: If the patch is not found or contains no nodes
    """
    # Import FluentMeshReader here to avoid circular imports
    from openffd.mesh.fluent_reader import FluentMeshReader
    
    # Handle Fluent mesh
    if isinstance(mesh_data, FluentMeshReader):
        return mesh_data.get_zone_points(patch_name)
        
    # Handle meshio mesh
    mesh = mesh_data  # For clarity
    node_ids = set()
    
    # Check if the mesh has the required attributes
    if not hasattr(mesh, 'points') or len(mesh.points) == 0:
        raise ValueError("Mesh has no points")
    
    found = False
    
    # 1) Try meshio cell_sets
    if hasattr(mesh, 'cell_sets') and patch_name in mesh.cell_sets:
        logger.debug(f"Found patch in cell_sets: {patch_name}")
        found = True
        cell_sets = mesh.cell_sets[patch_name]
        for i, block in enumerate(mesh.cells):
            ctype = block.type
            if ctype in cell_sets:
                cells = block.data[cell_sets[ctype]]
                for conn in cells:
                    node_ids.update(conn.tolist())
    
    # 2) Try Gmsh physical groups
    elif hasattr(mesh, 'field_data') and patch_name in mesh.field_data:
        logger.debug(f"Found patch in field_data: {patch_name}")
        found = True
        tag = mesh.field_data[patch_name][0]
        if 'gmsh:physical' in mesh.cell_data:
            phys = mesh.cell_data['gmsh:physical']
            for i, (block, phys_data) in enumerate(zip(mesh.cells, phys)):
                # phys_data aligns with block.data rows
                mask = phys_data == tag
                cells = block.data[mask]
                for conn in cells:
                    node_ids.update(conn.tolist())
    
    # 3) Try cell_data for direct labels
    elif hasattr(mesh, 'cell_data'):
        for key in mesh.cell_data:
            found_in_cell_data = False
            for i, data_array in enumerate(mesh.cell_data[key]):
                # Look for cells with this label
                if isinstance(data_array[0], (str, bytes)):
                    # String labels
                    mask = data_array == patch_name
                else:
                    # Numeric labels - try to convert patch_name to number
                    try:
                        patch_value = float(patch_name)
                        mask = data_array == patch_value
                    except ValueError:
                        continue
                
                if np.any(mask):
                    found = True
                    found_in_cell_data = True
                    cells = mesh.cells[i].data[mask]
                    for conn in cells:
                        node_ids.update(conn.tolist())
            
            if found_in_cell_data:
                break
                
    # 4) Try to interpret patch_name as a numeric boundary ID
    if not found:
        try:
            boundary_id = int(patch_name)
            # Try common field names for boundary IDs
            for field_name in ['gmsh:physical', 'medit:ref', 'tag', 'boundary_id']:
                if field_name in mesh.cell_data:
                    for i, tags in enumerate(mesh.cell_data[field_name]):
                        mask = tags == boundary_id
                        if np.any(mask):
                            found = True
                            cells = mesh.cells[i].data[mask]
                            for conn in cells:
                                node_ids.update(conn.tolist())
        except ValueError:
            # Not a numeric ID, continue
            pass
    
    if not found:
        # Get list of available patches for error message
        available_patches = []
        
        # Check cell_sets
        if hasattr(mesh, 'cell_sets'):
            available_patches.extend(list(mesh.cell_sets.keys()))
            
        # Check field_data
        if hasattr(mesh, 'field_data'):
            available_patches.extend(list(mesh.field_data.keys()))
        
        error_msg = f"Patch '{patch_name}' not found in mesh"
        if available_patches:
            error_msg += f". Available patches: {', '.join(available_patches)}"
            
        raise ValueError(error_msg)
    
    if not node_ids:
        raise ValueError(f"Patch '{patch_name}' contains no points")
    
    # Extract and return the point coordinates
    node_indices = list(node_ids)
    return mesh.points[node_indices]
