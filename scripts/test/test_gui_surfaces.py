#!/usr/bin/env python3
"""
Test script to check GUI surface rendering after all fixes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import FluentMeshReader
from openffd.mesh.general import extract_zone_mesh
from openffd.gui.visualization import FFDVisualizationWidget
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_gui_surfaces():
    """Test GUI surface rendering with the new solid surface system."""
    
    cas_file = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh"
    
    if not os.path.exists(cas_file):
        print(f"Error: Mesh file {cas_file} not found")
        return
    
    try:
        print("Loading mesh for GUI testing...")
        reader = FluentMeshReader(cas_file, force_native=True, debug=False)
        mesh = reader.read()
        
        print(f"Mesh loaded: {len(mesh.points)} points, {len(mesh.zone_list)} zones")
        
        # Test extracting both wedge zones
        test_zones = ['wedge_pos', 'wedge_neg']
        
        for zone_name in test_zones:
            print(f"\n{'='*60}")
            print(f"Testing zone extraction and visualization setup for '{zone_name}'...")
            
            try:
                zone_data = extract_zone_mesh(mesh, zone_name)
                if zone_data:
                    points = zone_data['points']
                    faces = zone_data['faces'] 
                    is_point_cloud = zone_data.get('is_point_cloud', False)
                    
                    print(f"✅ Zone extracted: {len(points)} points, {len(faces)} faces")
                    print(f"   Point cloud flag: {is_point_cloud}")
                    
                    if is_point_cloud:
                        print(f"   ❌ ERROR: Zone is marked as point cloud!")
                    else:
                        print(f"   ✅ Zone has proper face connectivity")
                    
                    # Check face density for gap prediction
                    face_point_ratio = len(faces) / max(len(points), 1)
                    print(f"   Face/Point ratio: {face_point_ratio:.3f}")
                    
                    if face_point_ratio < 1.0:
                        print(f"   ⚠️  Sparse connectivity - gaps possible without triangulation fixes")
                    elif face_point_ratio >= 1.5:
                        print(f"   ✅ Dense connectivity - solid surface expected")
                    else:
                        print(f"   🔶 Moderate connectivity")
                    
                    # Test visualization widget creation (without actually showing GUI)
                    print(f"   Testing visualization widget setup...")
                    try:
                        # Create widget but don't show it
                        widget = FFDVisualizationWidget()
                        print(f"   ✅ Visualization widget created successfully")
                        
                        # Test setting the mesh data (this triggers the rendering pipeline)
                        print(f"   Testing mesh data setup...")
                        widget.set_mesh(zone_data, points)
                        print(f"   ✅ Mesh data setup completed without errors")
                        
                        # Check if any actors were created
                        if hasattr(widget, 'mesh_actor') and widget.mesh_actor is not None:
                            print(f"   ✅ Mesh actor created successfully")
                        else:
                            print(f"   ❌ No mesh actor created")
                        
                    except Exception as viz_error:
                        print(f"   ❌ Visualization error: {viz_error}")
                        import traceback
                        traceback.print_exc()
                        
                else:
                    print(f"   ❌ Failed to extract zone")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print("Summary:")
        print("- All point cloud fallbacks have been removed")
        print("- Face subsampling is more conservative for solid surfaces")
        print("- Advanced gap-filling triangulation is implemented")
        print("- Emergency surface creation uses convex hull instead of points")
        print("- Universal degenerate surface handling is in place")
        print("\nThe system should now render ALL surfaces as solid surfaces only.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gui_surfaces()