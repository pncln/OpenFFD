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
    
    try:
        # Try to read with auto format detection first
        mesh = meshio.read(filename, file_format)
        logger.info(f"Successfully read mesh: {len(mesh.points)} points, {sum(len(c.data) for c in mesh.cells)} cells")
        return mesh
    except Exception as e:
        logger.error(f"Error reading mesh with meshio: {e}")
        
        # Try alternative format if explicit format was provided
        if file_format:
            try:
                logger.info(f"Trying to read mesh with explicit format: {file_format}")
                mesh = meshio.read(filename, file_format)
                return mesh
            except Exception as e2:
                logger.error(f"Error reading mesh with explicit format {file_format}: {e2}")
        
        # If all attempts fail
        raise ValueError(f"Failed to read mesh file {filename}. Original error: {e}")


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
    from openffd.mesh.fluent import FluentMeshReader
    
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
