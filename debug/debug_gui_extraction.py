#!/usr/bin/env python3
"""
Debug the exact GUI zone extraction call.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import read_fluent_mesh
from openffd.mesh.general import extract_zone_mesh
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

def debug_gui_extraction():
    """Debug exact GUI zone extraction."""
    mesh_file = "14.msh"
    
    print("🔍 Debugging GUI Zone Extraction Call")
    print("=" * 50)
    
    # Load mesh exactly like GUI does
    mesh_data = read_fluent_mesh(mesh_file, use_meshio=True, debug=False)
    
    # Test zone extraction exactly like GUI does
    zone_name = "launchpad" 
    print(f"🎯 Extracting zone: {zone_name}")
    
    # Call extract_zone_mesh like the GUI does
    zone_mesh_data = extract_zone_mesh(mesh_data, zone_name)
    
    if zone_mesh_data is not None:
        points = zone_mesh_data['points']
        faces = zone_mesh_data['faces']
        zone_type = zone_mesh_data['zone_type']
        is_point_cloud = zone_mesh_data['is_point_cloud']
        
        print(f"✅ Zone extraction SUCCESS:")
        print(f"   Points: {len(points):,}")
        print(f"   Faces: {len(faces):,}")
        print(f"   Zone type: {zone_type}")
        print(f"   Is point cloud: {is_point_cloud}")
        
        if faces and not is_point_cloud:
            print(f"   🎉 SURFACE DATA AVAILABLE!")
            print(f"   Sample faces: {faces[:3]}")
        else:
            print(f"   ❌ NO SURFACE DATA - POINT CLOUD FALLBACK")
            
    else:
        print(f"❌ Zone extraction FAILED - returned None")
    
    print(f"\n🧪 Manual verification:")
    
    # Let's manually do what extract_zone_mesh does
    zone = mesh_data.get_zone_by_name(zone_name)
    if zone:
        print(f"   Zone found: {zone}")
        print(f"   Zone type enum: {zone.zone_type_enum}")
        print(f"   Is volume: {zone.zone_type_enum.name == 'VOLUME'}")
        
        zone_points = mesh_data.get_zone_points(zone_name)
        print(f"   Zone points: {len(zone_points) if zone_points is not None else 'None'}")
        
        if hasattr(zone, 'faces') and zone.faces:
            print(f"   Zone faces available: {len(zone.faces)}")
            
            # Test the actual face processing
            point_indices = zone.get_point_indices()
            point_index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sorted(point_indices))}
            
            zone_faces = []
            for i, face in enumerate(zone.faces[:10]):  # Test first 10
                if hasattr(face, 'node_indices') and len(face.node_indices) >= 3:
                    face_nodes = [point_index_map.get(node_idx) for node_idx in face.node_indices 
                                if node_idx in point_index_map]
                    if len(face_nodes) >= 3:
                        zone_faces.append(face_nodes)
            
            print(f"   Manual face extraction: {len(zone_faces)} faces from first 10")
            
            if zone_faces:
                print(f"   🎉 MANUAL EXTRACTION WORKS!")
            else:
                print(f"   ❌ MANUAL EXTRACTION FAILED!")
        else:
            print(f"   ❌ No faces in zone")

if __name__ == "__main__":
    debug_gui_extraction()
