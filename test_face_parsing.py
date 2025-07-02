#!/usr/bin/env python3
"""
Test script to debug face parsing with the actual 14.msh file.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import FluentMeshReader
from openffd.mesh.general import extract_zone_mesh
import logging

# Set up logging to see debug info
logging.basicConfig(level=logging.INFO)

def test_face_parsing():
    """Test face parsing with the 14.msh file."""
    
    cas_file = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh"
    
    if not os.path.exists(cas_file):
        print(f"Error: Mesh file {cas_file} not found")
        return
    
    try:
        print("Testing native parser...")
        reader_native = FluentMeshReader(cas_file, force_native=True, debug=True)
        mesh_native = reader_native.read()
        
        print(f"\nNative parser results:")
        print(f"  Points: {len(mesh_native.points)}")
        print(f"  Zones: {len(mesh_native.zone_list)}")
        
        # Check each zone's face connectivity
        for zone in mesh_native.zone_list:
            print(f"\n  Zone: {zone.name}")
            print(f"    Type: {zone.zone_type}")
            print(f"    Faces: {len(zone.faces)}")
            print(f"    Points: {zone.num_points()}")
            
            if zone.faces and len(zone.faces) > 0:
                print(f"    First 3 faces:")
                for i, face in enumerate(zone.faces[:3]):
                    node_indices = face.node_indices
                    print(f"      Face {i}: {node_indices}")
                    
                    # Check if indices are reasonable (not huge values)
                    max_idx = max(node_indices) if node_indices else 0
                    min_idx = min(node_indices) if node_indices else 0
                    span = max_idx - min_idx if node_indices else 0
                    print(f"        Index range: {min_idx} to {max_idx}, span: {span}")
                    
                    # Check against total points
                    if max_idx >= len(mesh_native.points):
                        print(f"        ❌ INVALID: max index {max_idx} >= total points {len(mesh_native.points)}")
                    else:
                        print(f"        ✅ Valid indices (max point index: {len(mesh_native.points)-1})")
            
        print(f"\n" + "="*60)
        print("Testing meshio parser...")
        reader_meshio = FluentMeshReader(cas_file, use_meshio=True, debug=True)
        mesh_meshio = reader_meshio.read()
        
        print(f"\nMeshio parser results:")
        print(f"  Points: {len(mesh_meshio.points)}")
        print(f"  Zones: {len(mesh_meshio.zone_list)}")
        
        # Compare specific zones
        for zone in mesh_meshio.zone_list:
            print(f"\n  Zone: {zone.name}")
            print(f"    Type: {zone.zone_type}")
            print(f"    Faces: {len(zone.faces)}")
            print(f"    Points: {zone.num_points()}")
            
        # Test zone extraction for wedge_pos specifically
        print(f"\n" + "="*60)
        print("Testing zone extraction for 'wedge_pos'...")
        
        # Find wedge_pos zone
        wedge_zones = [z for z in mesh_native.zone_list if 'wedge' in z.name.lower()]
        if wedge_zones:
            test_zone = wedge_zones[0]
            print(f"Found zone: {test_zone.name}")
            
            try:
                zone_data = extract_zone_mesh(mesh_native, test_zone.name)
                if zone_data:
                    print(f"  Extracted: {len(zone_data['points'])} points, {len(zone_data['faces'])} faces")
                    print(f"  Point cloud: {zone_data.get('is_point_cloud', 'unknown')}")
                    
                    if zone_data['faces']:
                        face = zone_data['faces'][0]
                        print(f"  First face: {face}")
                        
                        # Check face size by computing distance between nodes
                        if len(face) >= 3:
                            points = zone_data['points']
                            p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
                            
                            import numpy as np
                            edge1_len = np.linalg.norm(p2 - p1)
                            edge2_len = np.linalg.norm(p3 - p2)
                            edge3_len = np.linalg.norm(p1 - p3)
                            
                            print(f"  First face edge lengths: {edge1_len:.6f}, {edge2_len:.6f}, {edge3_len:.6f}")
                            
                            if max(edge1_len, edge2_len, edge3_len) < 1e-10:
                                print(f"  ❌ PROBLEM: Face edges are extremely small - likely degenerate")
                            elif max(edge1_len, edge2_len, edge3_len) > 1000:
                                print(f"  ❌ PROBLEM: Face edges are very large - likely wrong connectivity")
                            else:
                                print(f"  ✅ Face edge lengths look reasonable")
                                
                else:
                    print(f"  Failed to extract zone")
            except Exception as e:
                print(f"  Error extracting: {e}")
        else:
            print("No wedge zones found")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_face_parsing()