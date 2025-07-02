#!/usr/bin/env python3
"""
Debug script to check face connectivity issues after the fix.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import FluentMeshReader
from openffd.mesh.general import extract_zone_mesh
import logging

# Set up logging to see debug info
logging.basicConfig(level=logging.DEBUG)

def debug_connectivity():
    """Debug the connectivity after the fix."""
    
    # You'll need to provide the path to your .cas file
    cas_file = "your_mesh_file.cas"  # Replace with actual path
    
    if not os.path.exists(cas_file):
        print(f"Error: Mesh file {cas_file} not found")
        print("Please update the cas_file path in this script")
        return
    
    try:
        # Read mesh with native parser
        reader = FluentMeshReader(cas_file, force_native=True, debug=True)
        mesh = reader.read()
        
        print(f"Mesh loaded:")
        print(f"  Points: {len(mesh.points)}")
        print(f"  Zones: {len(mesh.zone_list)}")
        
        # Check each zone's face connectivity
        for zone in mesh.zone_list:
            print(f"\nZone: {zone.name}")
            print(f"  Type: {zone.zone_type}")
            print(f"  Faces: {len(zone.faces)}")
            print(f"  Points: {zone.num_points()}")
            
            if zone.faces:
                print(f"  First 3 faces:")
                for i, face in enumerate(zone.faces[:3]):
                    print(f"    Face {i}: {face.node_indices}")
                
                # Check if face indices are within bounds
                max_point_idx = len(mesh.points) - 1
                valid_faces = 0
                invalid_faces = 0
                
                for face in zone.faces:
                    if all(0 <= idx <= max_point_idx for idx in face.node_indices):
                        valid_faces += 1
                    else:
                        invalid_faces += 1
                
                print(f"  Connectivity: {valid_faces} valid, {invalid_faces} invalid faces")
                
                # Try extracting this zone
                print(f"\n  Extracting zone '{zone.name}'...")
                try:
                    zone_data = extract_zone_mesh(mesh, zone.name)
                    if zone_data:
                        print(f"    Extracted: {len(zone_data['points'])} points, {len(zone_data['faces'])} faces")
                        print(f"    Point cloud: {zone_data.get('is_point_cloud', 'unknown')}")
                        if zone_data['faces']:
                            print(f"    First face: {zone_data['faces'][0] if zone_data['faces'] else 'none'}")
                    else:
                        print(f"    Failed to extract zone")
                except Exception as e:
                    print(f"    Error extracting: {e}")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_connectivity()