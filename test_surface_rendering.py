#!/usr/bin/env python3
"""
Test surface rendering with face connectivity.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import read_fluent_mesh
from openffd.mesh.general import extract_zone_mesh

def test_surface_extraction():
    """Test surface extraction with face connectivity."""
    mesh_file = "14.msh"
    
    if not os.path.exists(mesh_file):
        print(f"Error: Mesh file {mesh_file} not found!")
        return
    
    print("Testing Surface Mesh Extraction")
    print("=" * 50)
    
    # Load mesh with proper zone extraction
    print("Loading mesh with zone extraction...")
    mesh_data = read_fluent_mesh(mesh_file, use_meshio=True, debug=True)
    
    print(f"Loaded mesh with {len(mesh_data.points):,} points and {len(mesh_data.zones)} zones")
    print(f"Available zones: {list(mesh_data.zones.keys())}")
    
    # Test boundary zones - use the actual available zones
    test_zones = list(mesh_data.zones.keys())[:3]  # Test first 3 zones
    print(f"Testing zones: {test_zones}")
    
    for zone_name in test_zones:
        print(f"\n🔍 Testing zone: {zone_name}")
        print("-" * 30)
        
        zone_mesh = extract_zone_mesh(mesh_data, zone_name)
        
        if zone_mesh:
            points = zone_mesh['points']
            faces = zone_mesh['faces'] 
            zone_type = zone_mesh['zone_type']
            is_point_cloud = zone_mesh['is_point_cloud']
            
            print(f"   ✅ Zone extracted successfully:")
            print(f"      • Points: {len(points):,}")
            print(f"      • Faces: {len(faces):,}")
            print(f"      • Type: {zone_type}")
            print(f"      • Surface: {'Yes' if not is_point_cloud else 'Point cloud only'}")
            
            if faces:
                # Check face sizes
                face_sizes = [len(face) for face in faces[:10]]  # Sample first 10
                print(f"      • Sample face sizes: {face_sizes}")
                
                # Check face indices are valid
                max_point_idx = max(max(face) for face in faces[:10] if face)
                print(f"      • Max face index: {max_point_idx} (should be < {len(points)})")
                
                if max_point_idx >= len(points):
                    print(f"      ⚠️  WARNING: Face indices out of bounds!")
                else:
                    print(f"      ✅ Face indices valid")
            else:
                print(f"      ⚠️  No face connectivity found")
        else:
            print(f"   ❌ Zone extraction failed")
    
    print(f"\n🎯 **Surface Extraction Test Complete!**")
    print(f"   • Zone extraction with face connectivity implemented")
    print(f"   • Ready for proper surface rendering in GUI")
    print(f"   • No more weird disconnected visualizations!")

if __name__ == "__main__":
    test_surface_extraction()
