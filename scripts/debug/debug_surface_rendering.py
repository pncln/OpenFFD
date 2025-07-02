#!/usr/bin/env python3
"""
Debug surface rendering issue.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import read_fluent_mesh
from openffd.mesh.general import extract_zone_mesh
import numpy as np
import pyvista as pv

def debug_surface_rendering():
    """Debug why surface rendering isn't working."""
    mesh_file = "14.msh"
    
    if not os.path.exists(mesh_file):
        print(f"Error: Mesh file {mesh_file} not found!")
        return
    
    print("🔍 Debugging Surface Rendering")
    print("=" * 50)
    
    # Load mesh
    mesh_data = read_fluent_mesh(mesh_file, use_meshio=True, debug=False)
    print(f"Available zones: {list(mesh_data.zones.keys())}")
    
    # Test with first boundary zone
    boundary_zones = []
    for zone_name, zone_info in mesh_data.zones.items():
        zone_type = zone_info.get('type', 'unknown')
        zone_obj = zone_info.get('object')
        
        # Skip volume zones
        if zone_obj and hasattr(zone_obj, 'zone_type_enum'):
            is_volume = zone_obj.zone_type_enum.name == 'VOLUME'
        else:
            is_volume = zone_type in ['interior', 'fluid', 'solid']
        
        if not is_volume:
            boundary_zones.append(zone_name)
    
    if not boundary_zones:
        print("❌ No boundary zones found!")
        return
    
    test_zone = boundary_zones[0]
    print(f"\n🎯 Testing zone: {test_zone}")
    
    zone_mesh = extract_zone_mesh(mesh_data, test_zone)
    
    if zone_mesh:
        points = zone_mesh['points']
        faces = zone_mesh['faces']
        zone_type = zone_mesh['zone_type']
        is_point_cloud = zone_mesh['is_point_cloud']
        
        print(f"   ✅ Zone extracted:")
        print(f"      • Points: {len(points):,}")
        print(f"      • Faces: {len(faces):,}")
        print(f"      • Type: {zone_type}")
        print(f"      • Is point cloud: {is_point_cloud}")
        
        if faces and not is_point_cloud:
            print(f"\n🔧 Testing PyVista surface creation...")
            
            # Test face creation
            sample_faces = faces[:5]
            print(f"      • Sample faces: {[list(f) for f in sample_faces]}")
            
            # Try different PyVista formats
            print(f"\n🧪 Testing PyVista formats:")
            
            # Format 1: extend method (current)
            try:
                pv_faces_1 = []
                for face in faces[:100]:  # Test with smaller subset
                    if len(face) >= 3:
                        pv_faces_1.extend([len(face)] + list(face))
                
                pv_mesh_1 = pv.PolyData(points, faces=np.array(pv_faces_1, dtype=np.int32))
                print(f"      ✅ Format 1 (extend): SUCCESS - {pv_mesh_1.n_faces} faces")
                
                # Try to save as test file
                pv_mesh_1.save("test_surface.vtk")
                print(f"      ✅ Saved test surface to test_surface.vtk")
                
            except Exception as e:
                print(f"      ❌ Format 1 (extend): FAILED - {e}")
            
            # Format 2: VTK cell array
            try:
                from pyvista import CellArray
                cell_types = []
                cell_array = []
                
                for face in faces[:100]:
                    if len(face) == 3:
                        cell_types.append(pv.CellType.TRIANGLE)
                    elif len(face) == 4:
                        cell_types.append(pv.CellType.QUAD)
                    else:
                        continue
                    cell_array.extend(face)
                
                if cell_array:
                    pv_mesh_2 = pv.UnstructuredGrid(
                        np.array(cell_array, dtype=np.int32).reshape(-1, len(faces[0])),
                        np.array(cell_types),
                        points
                    )
                    print(f"      ✅ Format 2 (UnstructuredGrid): SUCCESS - {pv_mesh_2.n_cells} cells")
                else:
                    print(f"      ❌ Format 2: No valid cells created")
                    
            except Exception as e:
                print(f"      ❌ Format 2 (UnstructuredGrid): FAILED - {e}")
            
            # Format 3: Manual triangulation
            try:
                if len(faces[0]) == 3:  # Triangular faces
                    face_array = np.array(faces[:100])
                    pv_mesh_3 = pv.PolyData(points, faces=face_array)
                    print(f"      ✅ Format 3 (direct): SUCCESS - {pv_mesh_3.n_faces} faces")
                else:
                    print(f"      ⚠️  Format 3: Skipped - faces are not triangular")
                    
            except Exception as e:
                print(f"      ❌ Format 3 (direct): FAILED - {e}")
        else:
            print(f"      ⚠️  No face data available for surface rendering")
    else:
        print(f"      ❌ Zone extraction failed")
    
    print(f"\n🎯 **Debug Complete!**")

if __name__ == "__main__":
    debug_surface_rendering()
