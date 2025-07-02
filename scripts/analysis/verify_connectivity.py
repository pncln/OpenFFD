#!/usr/bin/env python3

"""
Verify if the parsed connectivity is actually valid by loading the mesh
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import FluentMeshReader
import logging
import numpy as np

def verify_connectivity():
    """Load the actual mesh and verify if the connectivity is valid."""
    
    print("=== VERIFYING FACE CONNECTIVITY ===")
    
    # Load the mesh using meshio (which should work)
    reader = FluentMeshReader("/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh")
    reader.use_meshio = True
    
    mesh = reader.read()
    
    print(f"Total points in mesh: {len(mesh.points)}")
    print(f"Point index range: 0 to {len(mesh.points) - 1}")
    
    # Find wedge_pos zone
    wedge_pos_zone = None
    for zone in mesh.zone_list:
        if 'wedge_pos' in zone.name.lower():
            wedge_pos_zone = zone
            break
    
    if wedge_pos_zone:
        print(f"\nFound zone: {wedge_pos_zone.name}")
        print(f"Zone faces: {len(wedge_pos_zone.faces)}")
        
        if wedge_pos_zone.faces:
            print("\nChecking first few faces:")
            for i, face in enumerate(wedge_pos_zone.faces[:5]):
                print(f"Face {i}: {face.node_indices}")
                
                # Check if all indices are valid
                max_idx = max(face.node_indices)
                min_idx = min(face.node_indices)
                
                if max_idx >= len(mesh.points):
                    print(f"  ❌ Invalid: Max index {max_idx} >= point count {len(mesh.points)}")
                elif min_idx < 0:
                    print(f"  ❌ Invalid: Min index {min_idx} < 0")
                else:
                    print(f"  ✅ Valid indices: range [{min_idx}, {max_idx}]")
                    
                    # Get actual coordinates
                    coords = []
                    for idx in face.node_indices:
                        coords.append(mesh.points[idx])
                    coords = np.array(coords)
                    
                    # Calculate face center and size
                    center = np.mean(coords, axis=0)
                    size = np.max(np.linalg.norm(coords - center, axis=1))
                    
                    print(f"  Face center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
                    print(f"  Face size: {size:.6f}")
    else:
        print("Could not find wedge_pos zone")
        print("Available zones:")
        for zone in mesh.zone_list:
            print(f"  - {zone.name}")

if __name__ == "__main__":
    verify_connectivity()