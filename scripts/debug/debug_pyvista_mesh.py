#!/usr/bin/env python3
"""
Debug script to test PyVista mesh creation from zone data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import FluentMeshReader
from openffd.mesh.general import extract_zone_mesh
import numpy as np

def test_pyvista_conversion():
    """Test PyVista mesh creation from zone data."""
    
    # Load the mesh
    mesh_file = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh"
    
    print("Testing PyVista mesh conversion...")
    reader = FluentMeshReader(mesh_file)
    mesh = reader.read()
    
    # Test rocket zone
    print(f"\n=== TESTING ROCKET ZONE PYVISTA CONVERSION ===")
    rocket_data = extract_zone_mesh(mesh, 'rocket')
    
    if rocket_data:
        points = rocket_data['points']
        faces = rocket_data['faces']
        
        print(f"Zone data: {len(points)} points, {len(faces)} faces")
        
        # Test PyVista conversion exactly like in visualization.py
        try:
            import pyvista as pv
            
            # Step 1: Convert faces to PyVista format
            pv_faces = []
            for face in faces:
                if len(face) >= 3:
                    pv_faces.extend([len(face)] + list(face))
            
            print(f"PyVista faces array length: {len(pv_faces)}")
            print(f"First few PyVista face entries: {pv_faces[:20]}")
            
            # Check for any invalid indices
            face_data = np.array(pv_faces[1::5])  # Skip the count values, get actual indices
            max_index = np.max(face_data) if len(face_data) > 0 else -1
            print(f"Max face index: {max_index}, Points available: {len(points)-1}")
            
            if max_index >= len(points):
                print(f"❌ Face indices out of bounds!")
                invalid_indices = face_data[face_data >= len(points)]
                print(f"Invalid indices: {invalid_indices[:10]}...")
                return
            
            # Step 2: Create PyVista mesh
            mesh_pv = pv.PolyData(points, faces=np.array(pv_faces, dtype=np.int32))
            print(f"✅ PyVista mesh created: {mesh_pv.n_points} points, {mesh_pv.n_cells} cells")
            
            # Step 3: Test cleaning
            mesh_clean = mesh_pv.clean()
            print(f"✅ Mesh cleaned: {mesh_clean.n_points} points, {mesh_clean.n_cells} cells")
            
            # Step 4: Test triangulation
            mesh_tri = mesh_clean.triangulate()
            print(f"✅ Mesh triangulated: {mesh_tri.n_points} points, {mesh_tri.n_cells} cells")
            
            # Check if mesh has valid geometry
            bounds = mesh_tri.bounds
            print(f"Mesh bounds: {bounds}")
            
            # Test rendering
            print(f"\n=== TESTING MESH RENDERING ===")
            plotter = pv.Plotter(off_screen=True)
            actor = plotter.add_mesh(mesh_tri, color='lightblue')
            plotter.screenshot('test_rocket.png', transparent_background=False)
            plotter.close()
            print(f"✅ Screenshot saved as test_rocket.png")
            
        except Exception as e:
            print(f"❌ PyVista conversion failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Also test wedge_pos (tiny zone)
    print(f"\n=== TESTING WEDGE_POS ZONE ===")
    wedge_data = extract_zone_mesh(mesh, 'wedge_pos')
    
    if wedge_data:
        points = wedge_data['points']
        faces = wedge_data['faces']
        
        print(f"Zone data: {len(points)} points, {len(faces)} faces")
        
        try:
            import pyvista as pv
            
            # Convert to PyVista format
            pv_faces = []
            for face in faces:
                if len(face) >= 3:
                    pv_faces.extend([len(face)] + list(face))
            
            mesh_pv = pv.PolyData(points, faces=np.array(pv_faces, dtype=np.int32))
            print(f"✅ PyVista mesh created: {mesh_pv.n_points} points, {mesh_pv.n_cells} cells")
            
            # Test rendering
            plotter = pv.Plotter(off_screen=True)
            actor = plotter.add_mesh(mesh_pv, color='red', show_edges=True)
            plotter.screenshot('test_wedge.png', transparent_background=False)
            plotter.close()
            print(f"✅ Screenshot saved as test_wedge.png")
            
        except Exception as e:
            print(f"❌ PyVista conversion failed: {e}")

if __name__ == "__main__":
    test_pyvista_conversion()
