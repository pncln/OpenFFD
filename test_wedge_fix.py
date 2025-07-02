#!/usr/bin/env python3
"""
Test script to verify the wedge_pos zone connectivity fix.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import FluentMeshReader
import logging

# Set up logging to see debug info
logging.basicConfig(level=logging.INFO)

def test_wedge_connectivity():
    """Test the wedge_pos zone connectivity after the fix."""
    
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
        
        print(f"\nMesh loaded successfully:")
        print(f"  Points: {len(mesh.points)}")
        print(f"  Zones: {len(mesh.zone_list)}")
        print(f"  Total faces: {len(mesh.faces)}")
        
        # Find wedge_pos zone
        wedge_pos = None
        for zone in mesh.zone_list:
            if 'wedge_pos' in zone.name.lower():
                wedge_pos = zone
                break
        
        if wedge_pos:
            print(f"\nwedge_pos zone found:")
            print(f"  Name: {wedge_pos.name}")
            print(f"  Faces: {len(wedge_pos.faces)}")
            print(f"  Points: {wedge_pos.num_points()}")
            
            # Check first few face connectivities
            if wedge_pos.faces:
                print(f"\nFirst few face connectivities:")
                for i, face in enumerate(wedge_pos.faces[:5]):
                    print(f"  Face {i}: {face.node_indices}")
                    
                # Check if connectivity indices are reasonable (not spanning domain)
                max_point_idx = len(mesh.points) - 1
                valid_faces = 0
                invalid_faces = 0
                
                for face in wedge_pos.faces:
                    if all(0 <= idx <= max_point_idx for idx in face.node_indices):
                        valid_faces += 1
                    else:
                        invalid_faces += 1
                
                print(f"\nConnectivity validation:")
                print(f"  Valid faces: {valid_faces}")
                print(f"  Invalid faces: {invalid_faces}")
                print(f"  Max point index: {max_point_idx}")
                
                if invalid_faces == 0:
                    print("✅ All faces have valid connectivity!")
                else:
                    print(f"❌ {invalid_faces} faces still have invalid connectivity")
        else:
            print("❌ wedge_pos zone not found")
            print("Available zones:")
            for zone in mesh.zone_list:
                print(f"  - {zone.name}")
                
    except Exception as e:
        print(f"Error testing mesh: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wedge_connectivity()