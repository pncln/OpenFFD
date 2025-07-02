#!/usr/bin/env python3
"""
Test script to check wedge surface rendering after fixes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import FluentMeshReader
from openffd.mesh.general import extract_zone_mesh
import logging

# Set up logging to see debug info
logging.basicConfig(level=logging.INFO)

def test_wedge_surfaces():
    """Test wedge surface rendering after the fixes."""
    
    cas_file = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh"
    
    if not os.path.exists(cas_file):
        print(f"Error: Mesh file {cas_file} not found")
        return
    
    try:
        print("Loading mesh with native parser...")
        reader = FluentMeshReader(cas_file, force_native=True, debug=True)
        mesh = reader.read()
        
        print(f"\nMesh loaded:")
        print(f"  Points: {len(mesh.points)}")
        print(f"  Zones: {len(mesh.zone_list)}")
        
        # Find wedge zones specifically
        wedge_zones = [z for z in mesh.zone_list if 'wedge' in z.name.lower()]
        print(f"\nFound {len(wedge_zones)} wedge zones:")
        for zone in wedge_zones:
            print(f"  - {zone.name}: {len(zone.faces)} faces, {zone.num_points()} points")
        
        # Test extraction for both wedge zones
        for zone in wedge_zones:
            print(f"\n{'='*50}")
            print(f"Testing zone extraction for '{zone.name}'...")
            
            try:
                zone_data = extract_zone_mesh(mesh, zone.name)
                if zone_data:
                    print(f"✅ Extracted: {len(zone_data['points'])} points, {len(zone_data['faces'])} faces")
                    print(f"   Point cloud: {zone_data.get('is_point_cloud', 'unknown')}")
                    print(f"   Zone type: {zone_data.get('zone_type', 'unknown')}")
                    
                    if zone_data['faces']:
                        print(f"   First 3 faces: {zone_data['faces'][:3]}")
                        
                        # Check surface continuity
                        face_count = len(zone_data['faces'])
                        point_count = len(zone_data['points'])
                        face_point_ratio = face_count / max(point_count, 1)
                        
                        print(f"   Face/Point ratio: {face_point_ratio:.3f}")
                        if face_point_ratio < 1.0:
                            print(f"   ⚠️  LOW FACE DENSITY - may have gaps")
                        elif face_point_ratio >= 1.5:
                            print(f"   ✅ Good face density - solid surface expected")
                        else:
                            print(f"   🔶 Moderate face density")
                        
                        # Check face size consistency
                        import numpy as np
                        points = zone_data['points']
                        
                        edge_lengths = []
                        for i, face in enumerate(zone_data['faces'][:5]):  # Check first 5 faces
                            if len(face) >= 3:
                                p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
                                edge1 = np.linalg.norm(p2 - p1)
                                edge2 = np.linalg.norm(p3 - p2)
                                edge3 = np.linalg.norm(p1 - p3)
                                edge_lengths.extend([edge1, edge2, edge3])
                        
                        if edge_lengths:
                            min_edge = min(edge_lengths)
                            max_edge = max(edge_lengths)
                            avg_edge = sum(edge_lengths) / len(edge_lengths)
                            
                            print(f"   Edge lengths: min={min_edge:.2e}, max={max_edge:.2e}, avg={avg_edge:.2e}")
                            
                            if max_edge / min_edge > 1000:
                                print(f"   ⚠️  HIGH EDGE VARIATION - may indicate connectivity issues")
                            else:
                                print(f"   ✅ Consistent edge lengths")
                    else:
                        print(f"   ❌ NO FACES - would create point cloud (but we fixed this)")
                        
                else:
                    print(f"   ❌ Failed to extract zone")
                    
            except Exception as e:
                print(f"   ❌ Error extracting: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wedge_surfaces()