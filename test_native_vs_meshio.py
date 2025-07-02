#!/usr/bin/env python3

"""
Compare native parser vs meshio parser results
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import FluentMeshReader
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def compare_parsers():
    """Compare native parser vs meshio parser."""
    
    print("=== COMPARING NATIVE VS MESHIO PARSERS ===")
    
    # Test with meshio first (known to work)
    print("\n1. Testing with meshio parser:")
    reader_meshio = FluentMeshReader("/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh")
    reader_meshio.use_meshio = True
    
    try:
        mesh_meshio = reader_meshio.read()
        print(f"   ✅ Meshio: {len(mesh_meshio.points)} points, {len(mesh_meshio.zone_list)} zones")
        
        # Find wedge_pos zone
        wedge_pos_meshio = None
        for zone in mesh_meshio.zone_list:
            if 'wedge_pos' in zone.name.lower():
                wedge_pos_meshio = zone
                break
        
        if wedge_pos_meshio:
            print(f"   ✅ Meshio wedge_pos: {len(wedge_pos_meshio.faces)} faces")
            if wedge_pos_meshio.faces:
                sample_face = wedge_pos_meshio.faces[0]
                print(f"   Sample face: {sample_face.node_indices}")
        
    except Exception as e:
        print(f"   ❌ Meshio failed: {e}")
        return
    
    # Test with native parser
    print("\n2. Testing with native parser:")
    reader_native = FluentMeshReader("/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh")
    reader_native.force_native = True
    
    try:
        mesh_native = reader_native.read()
        print(f"   ✅ Native: {len(mesh_native.points)} points, {len(mesh_native.zone_list)} zones")
        
        # Find wedge_pos zone
        wedge_pos_native = None
        for zone in mesh_native.zone_list:
            if 'wedge_pos' in zone.name.lower():
                wedge_pos_native = zone
                break
        
        if wedge_pos_native:
            print(f"   ✅ Native wedge_pos: {len(wedge_pos_native.faces)} faces")
            if wedge_pos_native.faces:
                sample_face = wedge_pos_native.faces[0]
                print(f"   Sample face: {sample_face.node_indices}")
                
                # Check if indices are valid
                max_idx = max(sample_face.node_indices)
                if max_idx >= len(mesh_native.points):
                    print(f"   ❌ Invalid indices: max {max_idx} >= {len(mesh_native.points)}")
                else:
                    print(f"   ✅ Valid indices")
        else:
            print(f"   ❌ No wedge_pos zone found in native parser")
            print(f"   Available zones: {[z.name for z in mesh_native.zone_list]}")
        
    except Exception as e:
        print(f"   ❌ Native failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_parsers()