#!/usr/bin/env python3
"""
Comprehensive test to verify surface creation in GUI workflow.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import read_fluent_mesh
from openffd.mesh.general import extract_zone_mesh
import numpy as np
import pyvista as pv
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

def verify_surface_creation():
    """Comprehensive verification of surface creation."""
    mesh_file = "14.msh"
    
    print("🔍 COMPREHENSIVE SURFACE CREATION VERIFICATION")
    print("=" * 60)
    
    # Step 1: Load mesh
    print("\n1️⃣ Loading mesh...")
    mesh_data = read_fluent_mesh(mesh_file, use_meshio=True, debug=False)
    print(f"   ✅ Mesh loaded: {len(mesh_data.points):,} points, {len(mesh_data.zones)} zones")
    
    # Step 2: Extract zone
    print("\n2️⃣ Extracting zone...")
    zone_name = "launchpad"
    zone_mesh_data = extract_zone_mesh(mesh_data, zone_name)
    
    if not zone_mesh_data:
        print(f"   ❌ Zone extraction failed!")
        return
        
    zone_points = zone_mesh_data['points']
    zone_faces = zone_mesh_data['faces']
    is_point_cloud = zone_mesh_data['is_point_cloud']
    
    print(f"   ✅ Zone extracted:")
    print(f"      Points: {len(zone_points):,}")
    print(f"      Faces: {len(zone_faces):,}")
    print(f"      Is point cloud: {is_point_cloud}")
    
    # Step 3: Verify face data integrity
    print("\n3️⃣ Verifying face data integrity...")
    
    if not zone_faces:
        print(f"   ❌ No faces found!")
        return
        
    # Check face format
    sample_faces = zone_faces[:5]
    print(f"   Sample faces (first 5): {sample_faces}")
    
    # Verify face indices are valid
    max_point_idx = len(zone_points) - 1
    invalid_faces = 0
    for i, face in enumerate(zone_faces[:100]):  # Check first 100
        if any(idx < 0 or idx > max_point_idx for idx in face):
            invalid_faces += 1
            if invalid_faces == 1:  # Show first invalid face
                print(f"   ❌ Invalid face {i}: {face} (max valid index: {max_point_idx})")
    
    if invalid_faces > 0:
        print(f"   ❌ Found {invalid_faces} invalid faces out of first 100!")
        return
    else:
        print(f"   ✅ Face indices are valid (checked first 100)")
    
    # Step 4: Test PyVista surface creation manually
    print("\n4️⃣ Testing PyVista surface creation...")
    
    try:
        # Prepare PyVista face array
        pv_faces = []
        face_count = min(len(zone_faces), 10000)  # Test with first 10k faces
        
        for face in zone_faces[:face_count]:
            pv_faces.extend([len(face)] + list(face))
        
        pv_faces = np.array(pv_faces, dtype=np.int32)
        
        print(f"   Preparing {face_count:,} faces...")
        print(f"   PyVista face array length: {len(pv_faces):,}")
        print(f"   Sample PyVista faces: {pv_faces[:20]}")
        
        # Create PyVista mesh
        pv_mesh = pv.PolyData(zone_points, pv_faces)
        
        print(f"   ✅ PyVista mesh created successfully!")
        print(f"      Mesh points: {pv_mesh.n_points:,}")
        print(f"      Mesh faces: {pv_mesh.n_faces:,}")
        print(f"      Mesh cells: {pv_mesh.n_cells:,}")
        
        # Test if it's a valid surface mesh
        if pv_mesh.n_faces > 0:
            print(f"   ✅ Surface mesh is valid with {pv_mesh.n_faces:,} faces")
            
            # Save test surface
            test_file = "test_extracted_surface.vtk"
            pv_mesh.save(test_file)
            print(f"   ✅ Test surface saved to: {test_file}")
            
        else:
            print(f"   ❌ Surface mesh has no faces!")
            
    except Exception as e:
        print(f"   ❌ PyVista surface creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Test complete GUI workflow simulation
    print("\n5️⃣ Testing complete GUI workflow simulation...")
    
    try:
        from openffd.gui.visualization import FFDVisualizationWidget
        from PyQt6.QtWidgets import QApplication
        
        # Create minimal PyQt app
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Create visualization widget
        viz_widget = FFDVisualizationWidget()
        
        print(f"   ✅ Visualization widget created")
        
        # Call set_mesh exactly like GUI does after fix
        print(f"   🔧 Calling set_mesh with zone data...")
        viz_widget.set_mesh(zone_mesh_data, zone_points)
        
        print(f"   ✅ GUI workflow simulation completed!")
        
    except Exception as e:
        print(f"   ❌ GUI workflow simulation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Verification completed!")

if __name__ == "__main__":
    verify_surface_creation()
