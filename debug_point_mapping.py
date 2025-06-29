#!/usr/bin/env python3
"""
Debug point index mapping in zone extraction.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import read_fluent_mesh

def debug_point_mapping():
    """Debug the point index mapping issue."""
    mesh_file = "14.msh"
    
    print("🔍 Debugging Point Index Mapping")
    print("=" * 50)
    
    # Load mesh
    mesh_data = read_fluent_mesh(mesh_file, use_meshio=True, debug=False)
    
    # Get launchpad zone
    zone_name = "launchpad"
    zone = mesh_data.get_zone_by_name(zone_name)
    
    print(f"🎯 Zone: {zone_name}")
    print(f"   Zone faces: {len(zone.faces)}")
    
    # Get zone points
    zone_points = mesh_data.get_zone_points(zone_name)
    print(f"   Zone points: {len(zone_points)}")
    
    # Get point indices
    point_indices = zone.get_point_indices()
    print(f"   Point indices count: {len(point_indices)}")
    
    # Check first few point indices
    sorted_indices = sorted(point_indices)
    print(f"   First 10 point indices: {sorted_indices[:10]}")
    
    # Create point index mapping
    point_index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sorted_indices)}
    print(f"   Point index map size: {len(point_index_map)}")
    
    # Check first few faces
    print(f"\n🔧 Testing face extraction:")
    faces_processed = 0
    faces_valid = 0
    
    for i, face in enumerate(zone.faces[:5]):  # Test first 5 faces
        if hasattr(face, 'node_indices') and len(face.node_indices) >= 3:
            faces_processed += 1
            original_nodes = face.node_indices
            
            # Try to remap face node indices
            face_nodes = []
            missing_nodes = []
            
            for node_idx in original_nodes:
                if node_idx in point_index_map:
                    face_nodes.append(point_index_map[node_idx])
                else:
                    missing_nodes.append(node_idx)
            
            print(f"      Face {i}:")
            print(f"         Original nodes: {original_nodes}")
            print(f"         Remapped nodes: {face_nodes}")
            print(f"         Missing nodes: {missing_nodes}")
            
            if len(face_nodes) >= 3:
                faces_valid += 1
                print(f"         ✅ Valid face with {len(face_nodes)} nodes")
            else:
                print(f"         ❌ Invalid face - only {len(face_nodes)} nodes mapped")
    
    print(f"\n📊 Summary:")
    print(f"   Faces processed: {faces_processed}")
    print(f"   Valid faces: {faces_valid}")
    
    if faces_valid == 0:
        print(f"\n🚨 PROBLEM IDENTIFIED: Point index mapping is failing!")
        print(f"   This is why zone extraction returns 'no face connectivity'")
        
        # Check if point indices are in the expected range
        max_point_idx = max(sorted_indices)
        min_point_idx = min(sorted_indices)
        mesh_point_count = len(mesh_data.points)
        
        print(f"\n🔍 Point index analysis:")
        print(f"   Mesh total points: {mesh_point_count}")
        print(f"   Zone point indices range: {min_point_idx} to {max_point_idx}")
        print(f"   Zone point indices count: {len(point_indices)}")
        
        # Check if face node indices are in reasonable range
        sample_face = zone.faces[0]
        print(f"   Sample face nodes: {sample_face.node_indices}")
        print(f"   Nodes in point_index_map: {[n in point_index_map for n in sample_face.node_indices]}")

if __name__ == "__main__":
    debug_point_mapping()
