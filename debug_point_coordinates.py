#!/usr/bin/env python3
"""
Debug script to check if extracted zone points and faces create valid geometry.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import FluentMeshReader
from openffd.mesh.general import extract_zone_mesh
import numpy as np

def debug_zone_geometry():
    """Debug zone geometry to see if points and faces are correct."""
    
    # Load the mesh
    mesh_file = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh"
    
    print("Loading mesh to debug geometry...")
    reader = FluentMeshReader(mesh_file)
    mesh = reader.read()
    
    # Test rocket zone (small)
    print(f"\n=== TESTING ROCKET ZONE EXTRACTION ===")
    rocket_data = extract_zone_mesh(mesh, 'rocket')
    
    if rocket_data:
        points = rocket_data['points']
        faces = rocket_data['faces']
        
        print(f"Extracted points: {len(points)}")
        print(f"Extracted faces: {len(faces)}")
        print(f"Is point cloud: {rocket_data['is_point_cloud']}")
        
        # Check point coordinate ranges
        print(f"\nPoint coordinate ranges:")
        print(f"X: {points[:, 0].min():.3f} to {points[:, 0].max():.3f}")
        print(f"Y: {points[:, 1].min():.3f} to {points[:, 1].max():.3f}")
        print(f"Z: {points[:, 2].min():.3f} to {points[:, 2].max():.3f}")
        
        # Check face indices
        print(f"\nFace index ranges:")
        if faces:
            face_indices = np.array(faces).flatten()
            print(f"Face indices: {face_indices.min()} to {face_indices.max()}")
            print(f"Points available: 0 to {len(points)-1}")
            
            # Check if any face indices are out of bounds
            invalid_indices = face_indices[face_indices >= len(points)]
            if len(invalid_indices) > 0:
                print(f"❌ {len(invalid_indices)} face indices out of bounds!")
                print(f"Invalid indices: {invalid_indices[:10]}...")
            else:
                print(f"✅ All face indices valid")
                
            # Check first few faces
            print(f"\nFirst few faces:")
            for i, face in enumerate(faces[:3]):
                print(f"Face {i}: {face}")
                # Get the actual points for this face
                face_points = points[face]
                print(f"  Point coordinates:")
                for j, pt in enumerate(face_points):
                    print(f"    Point {face[j]}: [{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}]")
                    
        # Test with very small zone
        print(f"\n=== TESTING WEDGE_POS ZONE (TINY) ===")
        wedge_data = extract_zone_mesh(mesh, 'wedge_pos')
        
        if wedge_data:
            points = wedge_data['points']
            faces = wedge_data['faces']
            
            print(f"Extracted points: {len(points)}")
            print(f"Extracted faces: {len(faces)}")
            print(f"Is point cloud: {wedge_data['is_point_cloud']}")
            
            if not wedge_data['is_point_cloud'] and faces:
                print(f"All faces for wedge_pos:")
                for i, face in enumerate(faces):
                    print(f"Face {i}: {face}")
                    if len(face) < 3:
                        print(f"  ❌ Invalid face: only {len(face)} vertices")
                        
                # Check if this would create a valid PyVista mesh
                print(f"\nTesting PyVista mesh creation...")
                try:
                    import pyvista as pv
                    
                    # Create faces in PyVista format
                    pyvista_faces = []
                    for face in faces:
                        if len(face) >= 3:
                            pyvista_faces.extend([len(face)] + face)
                    
                    mesh_pv = pv.PolyData(points, pyvista_faces)
                    print(f"✅ PyVista mesh created successfully")
                    print(f"PyVista mesh: {mesh_pv.n_points} points, {mesh_pv.n_cells} cells")
                    
                except Exception as e:
                    print(f"❌ PyVista mesh creation failed: {e}")
            else:
                print(f"Zone is point cloud or has no faces")

if __name__ == "__main__":
    debug_zone_geometry()
