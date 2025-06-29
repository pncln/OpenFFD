#!/usr/bin/env python3
"""
Debug script to check if cell blocks are correctly mapped to zones.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import meshio
import numpy as np
from openffd.mesh.fluent_reader import FluentMeshReader

def debug_cell_block_mapping():
    """Debug the mapping between meshio cell blocks and Fluent zones."""
    
    mesh_file = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh"
    
    print("=== DEBUGGING CELL BLOCK TO ZONE MAPPING ===")
    
    # Step 1: Load with meshio directly to see the raw cell blocks
    print("Loading mesh with meshio...")
    try:
        meshio_mesh = meshio.read(mesh_file)
        print(f"Points: {len(meshio_mesh.points)}")
        print(f"Cell blocks: {len(meshio_mesh.cells)}")
        
        print("\nMeshio cell blocks:")
        for i, cell_block in enumerate(meshio_mesh.cells):
            print(f"  Block {i}: type='{cell_block.type}', count={len(cell_block.data)}")
            
        # Check cell data for zone information
        print(f"\nMeshio cell data keys: {list(meshio_mesh.cell_data.keys())}")
        
        # Look for zone tags in cell data
        if 'gmsh:physical' in meshio_mesh.cell_data:
            print("Found gmsh:physical tags:")
            for i, tags in enumerate(meshio_mesh.cell_data['gmsh:physical']):
                print(f"  Block {i}: tags={tags[:10]}...")  # Show first 10 tags
        
        if 'fluent:zone-id' in meshio_mesh.cell_data:
            print("Found fluent:zone-id tags:")
            for i, tags in enumerate(meshio_mesh.cell_data['fluent:zone-id']):
                print(f"  Block {i}: tags={tags[:10]}...")
                
    except Exception as e:
        print(f"Error loading with meshio: {e}")
        return
    
    # Step 2: Load with our FluentMeshReader and see how zones are created
    print(f"\n=== FLUENT READER ZONE CREATION ===")
    reader = FluentMeshReader(mesh_file)
    mesh = reader.read()
    
    boundary_zones = [zone for zone in mesh.zone_list if zone.zone_type_enum.name == 'BOUNDARY']
    
    print(f"Created {len(boundary_zones)} boundary zones:")
    for zone in boundary_zones:
        print(f"  Zone: {zone.name} (ID: {zone.zone_id}, Type: {zone.zone_type})")
        print(f"    Faces: {len(zone.faces)}")
        print(f"    Point indices: {len(zone.point_indices)}")
        
        # Check coordinate bounds for this zone
        if zone.point_indices:
            zone_points = mesh.points[list(zone.point_indices)]
            bounds = {
                'x': (zone_points[:, 0].min(), zone_points[:, 0].max()),
                'y': (zone_points[:, 1].min(), zone_points[:, 1].max()),
                'z': (zone_points[:, 2].min(), zone_points[:, 2].max())
            }
            print(f"    Bounds: X=[{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}], "
                  f"Y=[{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}], "
                  f"Z=[{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}]")
    
    # Step 3: Check if the rocket zone has reasonable geometry
    print(f"\n=== ROCKET ZONE DETAILED ANALYSIS ===")
    rocket_zone = None
    for zone in boundary_zones:
        if zone.name == 'rocket':
            rocket_zone = zone
            break
    
    if rocket_zone:
        print(f"Rocket zone details:")
        print(f"  Zone ID: {rocket_zone.zone_id}")
        print(f"  Faces: {len(rocket_zone.faces)}")
        print(f"  Point count: {len(rocket_zone.point_indices)}")
        
        # Check if points form a reasonable rocket-like shape
        zone_points = mesh.points[list(rocket_zone.point_indices)]
        
        # Analyze point distribution
        print(f"  Point distribution:")
        print(f"    X range: {zone_points[:, 0].min():.3f} to {zone_points[:, 0].max():.3f}")
        print(f"    Y range: {zone_points[:, 1].min():.3f} to {zone_points[:, 1].max():.3f}")
        print(f"    Z range: {zone_points[:, 2].min():.3f} to {zone_points[:, 2].max():.3f}")
        
        # Check if this looks like a reasonable rocket boundary
        x_span = zone_points[:, 0].max() - zone_points[:, 0].min()
        y_span = zone_points[:, 1].max() - zone_points[:, 1].min()
        z_span = zone_points[:, 2].max() - zone_points[:, 2].min()
        
        print(f"  Geometry spans: X={x_span:.3f}, Y={y_span:.3f}, Z={z_span:.3f}")
        
        # A rocket should probably be elongated in one direction
        max_span = max(x_span, y_span, z_span)
        print(f"  Max span: {max_span:.3f}")
        
        if max_span < 1.0:
            print(f"  ⚠️  Warning: Rocket zone seems very small (max span {max_span:.3f})")
        
        # Check face connectivity - sample a few faces
        print(f"  Sample faces:")
        for i, face in enumerate(rocket_zone.faces[:3]):
            face_points = mesh.points[face.node_indices]
            centroid = face_points.mean(axis=0)
            print(f"    Face {i}: centroid at [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")

if __name__ == "__main__":
    debug_cell_block_mapping()
