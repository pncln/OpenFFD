#!/usr/bin/env python3
"""
Debug script to check what meshio actually provides from the Fluent file.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import meshio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_meshio_data():
    """Debug what meshio actually provides from the Fluent file."""
    
    # Load the mesh
    mesh_file = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh"
    
    print("Loading mesh with meshio...")
    mesh = meshio.read(mesh_file)
    
    print(f"\nMesh Info:")
    print(f"  Points: {len(mesh.points)}")
    print(f"  Cells: {len(mesh.cells) if hasattr(mesh, 'cells') else 'None'}")
    
    # Check cell blocks
    if hasattr(mesh, 'cells'):
        print(f"\nCell Blocks:")
        for i, cell_block in enumerate(mesh.cells):
            print(f"  Block {i}: {cell_block.type} with {len(cell_block.data)} elements")
            if hasattr(cell_block, 'tags') and cell_block.tags:
                print(f"    Tags: {cell_block.tags}")
    
    # Check cell data
    if hasattr(mesh, 'cell_data') and mesh.cell_data:
        print(f"\nCell Data:")
        for key, values in mesh.cell_data.items():
            print(f"  {key}: {len(values)} entries")
            if len(values) > 0:
                print(f"    First few values: {values[0][:10] if hasattr(values[0], '__len__') and len(values[0]) > 10 else values[0]}")
    
    # Check point data
    if hasattr(mesh, 'point_data') and mesh.point_data:
        print(f"\nPoint Data:")
        for key, values in mesh.point_data.items():
            print(f"  {key}: {len(values)} entries")
            if len(values) > 10:
                print(f"    First few values: {values[:10]}")
            else:
                print(f"    Values: {values}")
    
    # Check field data
    if hasattr(mesh, 'field_data') and mesh.field_data:
        print(f"\nField Data:")
        for key, values in mesh.field_data.items():
            print(f"  {key}: {values}")
    
    # Check info
    if hasattr(mesh, 'info') and mesh.info:
        print(f"\nInfo:")
        for key, values in mesh.info.items():
            print(f"  {key}: {values}")
    
    # Look for any other attributes that might contain zone information
    print(f"\nAll mesh attributes:")
    for attr in dir(mesh):
        if not attr.startswith('_') and not callable(getattr(mesh, attr)):
            value = getattr(mesh, attr)
            if value is not None and hasattr(value, '__len__') and len(value) > 0:
                print(f"  {attr}: {type(value)} with {len(value) if hasattr(value, '__len__') else ''} items")

if __name__ == "__main__":
    debug_meshio_data()
