#!/usr/bin/env python3
"""
Test the intelligent assignment system to see the detailed mapping.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging

# Set up logging to see the assignment details
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from openffd.mesh.fluent_reader import FluentMeshReader
from openffd.mesh.general import extract_zone_mesh

def test_intelligent_assignment():
    """Test the intelligent assignment system."""
    
    mesh_file = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh"
    
    print("=== TESTING INTELLIGENT ASSIGNMENT ===")
    
    # Load with detailed logging
    reader = FluentMeshReader(mesh_file)
    mesh = reader.read()
    
    print(f"\n=== ZONE SUMMARY ===")
    boundary_zones = [zone for zone in mesh.zone_list if zone.zone_type_enum.name == 'BOUNDARY']
    
    for zone in boundary_zones:
        print(f"Zone: {zone.name}")
        print(f"  Faces: {len(zone.faces)}")
        print(f"  Points: {len(zone.point_indices)}")
        
        # Get sample face types
        face_types = {}
        for face in zone.faces[:100]:  # Sample first 100 faces
            face_type = len(face.node_indices)
            face_types[face_type] = face_types.get(face_type, 0) + 1
        print(f"  Face types: {face_types}")
    
    # Test rocket zone extraction
    print(f"\n=== ROCKET ZONE EXTRACTION TEST ===")
    rocket_data = extract_zone_mesh(mesh, 'rocket')
    
    if rocket_data:
        points = rocket_data['points']
        faces = rocket_data['faces']
        
        print(f"Extracted rocket zone: {len(points)} points, {len(faces)} faces")
        
        # Check bounds
        bounds = {
            'x': (points[:, 0].min(), points[:, 0].max()),
            'y': (points[:, 1].min(), points[:, 1].max()),
            'z': (points[:, 2].min(), points[:, 2].max())
        }
        print(f"Bounds: X=[{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}], "
              f"Y=[{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}], "
              f"Z=[{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}]")
        
        # Check face size distribution
        face_sizes = {}
        for face in faces:
            size = len(face)
            face_sizes[size] = face_sizes.get(size, 0) + 1
        print(f"Face size distribution: {face_sizes}")
        
        # Test PyVista rendering
        try:
            import pyvista as pv
            
            # Convert to PyVista format
            pv_faces = []
            for face in faces:
                if len(face) >= 3:
                    pv_faces.extend([len(face)] + list(face))
            
            mesh_pv = pv.PolyData(points, faces=pv_faces)
            print(f"✅ PyVista mesh: {mesh_pv.n_points} points, {mesh_pv.n_cells} cells")
            
            # Save screenshot
            plotter = pv.Plotter(off_screen=True)
            actor = plotter.add_mesh(mesh_pv, color='red', show_edges=True)
            plotter.screenshot('rocket_intelligent.png', transparent_background=False)
            plotter.close()
            print(f"✅ Screenshot saved as rocket_intelligent.png")
            
        except Exception as e:
            print(f"❌ PyVista test failed: {e}")

if __name__ == "__main__":
    test_intelligent_assignment()
