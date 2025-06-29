#!/usr/bin/env python3
"""
Test GUI zone emission and visualization reception.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import read_fluent_mesh
from openffd.mesh.general import extract_zone_mesh
from openffd.gui.visualization import FFDVisualizationWidget
import logging
from PyQt6.QtWidgets import QApplication

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

def test_gui_zone_emission():
    """Test the exact GUI zone emission and visualization reception."""
    mesh_file = "14.msh"
    
    print("🧪 Testing GUI Zone Emission")
    print("=" * 50)
    
    # Create minimal PyQt app (needed for visualization widget)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    # Load mesh like GUI does
    mesh_data = read_fluent_mesh(mesh_file, use_meshio=True, debug=False)
    
    # Extract zone like GUI does
    zone_name = "launchpad"
    zone_mesh_data = extract_zone_mesh(mesh_data, zone_name)
    
    print(f"🎯 Extracted zone: {zone_name}")
    print(f"   Zone data type: {type(zone_mesh_data)}")
    print(f"   Zone data keys: {list(zone_mesh_data.keys()) if zone_mesh_data else 'None'}")
    
    if zone_mesh_data:
        points = zone_mesh_data['points']
        faces = zone_mesh_data['faces']
        is_point_cloud = zone_mesh_data['is_point_cloud']
        
        print(f"   Points: {len(points):,}")
        print(f"   Faces: {len(faces):,}")
        print(f"   Is point cloud: {is_point_cloud}")
        
        # Now test visualization widget
        print(f"\n📺 Testing Visualization Widget Reception:")
        
        # Create visualization widget (without showing)
        try:
            viz_widget = FFDVisualizationWidget()
            
            print(f"   Created visualization widget: {type(viz_widget)}")
            
            # Call set_mesh like GUI does
            viz_widget.set_mesh(zone_mesh_data, points)
            
            print(f"   ✅ Visualization widget received the data successfully!")
            
        except Exception as e:
            print(f"   ❌ Error in visualization widget: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Test completed!")

if __name__ == "__main__":
    test_gui_zone_emission()
