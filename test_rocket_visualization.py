#!/usr/bin/env python3

import sys
sys.path.insert(0, '/Users/pncln/Documents/tubitak/verynew/ffd_gen/src')

from openffd.mesh.fluent_reader import FluentMeshReader
import numpy as np

def test_rocket_visualization():
    print("=== TESTING ROCKET ZONE VISUALIZATION ===")
    
    # Load the mesh
    reader = FluentMeshReader('/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh')
    mesh = reader.read()
    
    if not mesh:
        print("❌ Failed to load mesh")
        return
    
    # Get the rocket zone
    rocket_zone = None
    for name, zone_info in mesh.zones.items():
        if 'rocket' in name.lower():
            rocket_zone = zone_info['object']
            break
    
    if not rocket_zone:
        print("❌ Rocket zone not found")
        return
    
    print(f"✅ Found rocket zone with {len(rocket_zone.faces)} faces")
    
    # Analyze the spatial distribution
    if rocket_zone.faces:
        # Get all face points
        face_points = []
        for face in rocket_zone.faces:
            for node_id in face.node_indices:
                if node_id < len(mesh.points):
                    face_points.append(mesh.points[node_id])
                    
        if face_points:
            face_points = np.array(face_points)
            bounds = {
                'X': [face_points[:, 0].min(), face_points[:, 0].max()],
                'Y': [face_points[:, 1].min(), face_points[:, 1].max()],
                'Z': [face_points[:, 2].min(), face_points[:, 2].max()]
            }
            
            print(f"Rocket zone spatial bounds:")
            for axis, (min_val, max_val) in bounds.items():
                span = max_val - min_val
                print(f"  {axis}: [{min_val:.3f}, {max_val:.3f}] (span: {span:.3f})")
            
            # Check if this looks like a reasonable rocket zone
            x_span = bounds['X'][1] - bounds['X'][0]
            y_span = bounds['Y'][1] - bounds['Y'][0]
            z_span = bounds['Z'][1] - bounds['Z'][0]
            
            print(f"\nSpatial analysis:")
            print(f"  X span: {x_span:.3f} (should be moderate for rocket length)")
            print(f"  Y span: {y_span:.3f} (should be small for rocket width)")
            print(f"  Z span: {z_span:.3f} (should be small for rocket thickness)")
            
            # Reasonable rocket dimensions check
            if x_span > 1.0 and y_span < 10.0 and z_span < 1.0:
                print("✅ Spatial dimensions look reasonable for a rocket zone")
            else:
                print("⚠️  Spatial dimensions may not be optimal for a rocket zone")
                
            # Check if it's not spanning the whole domain (which was the original problem)
            full_domain_x = 16.5  # Approximate from previous logs
            full_domain_y = 19.5  # Approximate from previous logs
            
            if x_span < full_domain_x * 0.8 and y_span < full_domain_y * 0.5:
                print("✅ Zone does NOT span the whole domain (good!)")
            else:
                print("❌ Zone may still be spanning too much of the domain")

if __name__ == "__main__":
    test_rocket_visualization()
