#!/usr/bin/env python3
"""
Test script for the improved Fluent mesh reader using meshio fallback.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
from openffd.mesh.fluent_reader import FluentMeshReader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

def test_mesh_reader_meshio():
    """Test the improved mesh reader with meshio fallback."""
    mesh_file = "14.msh"
    
    if not os.path.exists(mesh_file):
        print(f"Error: Mesh file {mesh_file} not found!")
        return
    
    print(f"Testing enhanced Fluent mesh reader (meshio mode) with: {mesh_file}")
    print("=" * 60)
    
    try:
        # Create reader and force meshio mode
        reader = FluentMeshReader(mesh_file, use_meshio=True, debug=True)
        mesh = reader.read()
        
        print(f"\n✓ Successfully loaded mesh!")
        print(f"  Points: {len(mesh.points):,}")
        print(f"  Zones: {len(mesh.zone_list)}")
        print(f"  Faces: {len(mesh.faces):,}")
        print(f"  Cells: {len(mesh.cells):,}")
        
        print(f"\nZone Details:")
        print("-" * 40)
        for i, zone in enumerate(mesh.zone_list, 1):
            print(f"{i:2d}. {zone.name}")
            print(f"     Type: {zone.zone_type} ({zone.zone_type_enum.name})")
            print(f"     ID: {zone.zone_id}")
            print(f"     Faces: {zone.num_faces()}")
            print(f"     Cells: {zone.num_cells()}")
            print(f"     Points: {zone.num_points()}")
            print()
        
        print(f"Available zones for GUI: {list(mesh.zones.keys())}")
        
        if len(mesh.points) > 0:
            print(f"\nMesh Bounds:")
            print(f"  X: {mesh.points[:, 0].min():.3f} to {mesh.points[:, 0].max():.3f}")
            print(f"  Y: {mesh.points[:, 1].min():.3f} to {mesh.points[:, 1].max():.3f}")
            print(f"  Z: {mesh.points[:, 2].min():.3f} to {mesh.points[:, 2].max():.3f}")
            
    except Exception as e:
        print(f"✗ Error reading mesh: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mesh_reader_meshio()
