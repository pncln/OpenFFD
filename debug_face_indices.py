#!/usr/bin/env python3
"""
Debug script to check face node indices vs point indices in zones.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import FluentMeshReader
import numpy as np

def debug_zone_face_indices():
    """Debug zone face indices to see if they match point indices."""
    
    # Load the mesh
    mesh_file = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh"
    
    print("Loading mesh to debug face indices...")
    reader = FluentMeshReader(mesh_file)
    mesh = reader.read()
    
    # Focus on a small zone for debugging
    rocket_zone = None
    for zone in mesh.zone_list:
        if zone.name == 'rocket':
            rocket_zone = zone
            break
    
    if not rocket_zone:
        print("❌ Rocket zone not found")
        return
    
    print(f"\n=== DEBUGGING ROCKET ZONE ===")
    print(f"Zone name: {rocket_zone.name}")
    print(f"Zone faces: {len(rocket_zone.faces)}")
    
    # Get zone point indices
    point_indices = rocket_zone.get_point_indices()
    print(f"Zone point indices: {len(point_indices)} points")
    print(f"Point indices range: {min(point_indices)} to {max(point_indices)}")
    
    # Check first few faces
    print(f"\n=== FACE ANALYSIS ===")
    for i, face in enumerate(rocket_zone.faces[:5]):
        print(f"Face {i}: {face.node_indices}")
        
        # Check if face node indices are in zone point indices
        valid_nodes = [node for node in face.node_indices if node in point_indices]
        invalid_nodes = [node for node in face.node_indices if node not in point_indices]
        
        print(f"  Valid nodes: {len(valid_nodes)}/{len(face.node_indices)}")
        if invalid_nodes:
            print(f"  ❌ Invalid nodes: {invalid_nodes}")
        else:
            print(f"  ✅ All nodes valid")
            
        print()
    
    # Check mesh total points
    print(f"Total mesh points: {len(mesh.points)}")
    print(f"Point indices max: {max(point_indices)}")
    print(f"Points accessible: {max(point_indices) < len(mesh.points)}")
    
    # Test point index mapping
    print(f"\n=== POINT INDEX MAPPING TEST ===")
    point_index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sorted(point_indices))}
    print(f"Mapped indices: {len(point_index_map)}")
    
    # Test mapping for first face
    if rocket_zone.faces:
        first_face = rocket_zone.faces[0]
        print(f"Original face indices: {first_face.node_indices}")
        
        mapped_indices = [point_index_map.get(node_idx) for node_idx in first_face.node_indices]
        print(f"Mapped face indices: {mapped_indices}")
        
        # Check if any mapping failed
        failed_mappings = [node_idx for node_idx in first_face.node_indices if node_idx not in point_index_map]
        if failed_mappings:
            print(f"❌ Failed to map indices: {failed_mappings}")
        else:
            print(f"✅ All indices mapped successfully")

if __name__ == "__main__":
    debug_zone_face_indices()
