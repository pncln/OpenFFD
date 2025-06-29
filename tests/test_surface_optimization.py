#!/usr/bin/env python3
"""
Test the new intelligent surface optimization for performance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import read_fluent_mesh
from openffd.mesh.general import extract_zone_mesh
from openffd.gui.visualization import FFDVisualizationWidget
from PyQt6.QtWidgets import QApplication
import numpy as np
import time

def test_surface_optimization():
    """Test the new surface optimization features."""
    mesh_file = "14.msh"
    
    print("🚀 TESTING SURFACE OPTIMIZATION")
    print("=" * 50)
    
    # Load mesh and extract zone
    print("Loading mesh and extracting zone...")
    mesh_data = read_fluent_mesh(mesh_file, use_meshio=True, debug=False)
    zone_mesh_data = extract_zone_mesh(mesh_data, "launchpad")
    
    if not zone_mesh_data:
        print("❌ Zone extraction failed!")
        return
    
    zone_points = zone_mesh_data['points']
    zone_faces = zone_mesh_data['faces']
    
    print(f"Original surface: {len(zone_points):,} points, {len(zone_faces):,} faces")
    
    # Create visualization widget to test optimization
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    viz_widget = FFDVisualizationWidget()
    
    # Test the optimization method directly
    print("\n🔧 Testing surface optimization...")
    start_time = time.time()
    
    optimized_faces = viz_widget._optimize_surface_for_performance(zone_faces, zone_points)
    
    optimization_time = time.time() - start_time
    
    print(f"✅ Optimization completed in {optimization_time:.2f} seconds")
    print(f"   Original faces: {len(zone_faces):,}")
    print(f"   Optimized faces: {len(optimized_faces):,}")
    print(f"   Reduction ratio: {len(optimized_faces)/len(zone_faces):.3f}")
    
    # Test full rendering pipeline
    print("\n🎨 Testing full rendering pipeline...")
    start_time = time.time()
    
    viz_widget.set_mesh(zone_mesh_data, zone_points)
    
    render_time = time.time() - start_time
    
    print(f"✅ Full rendering completed in {render_time:.2f} seconds")
    
    # Performance comparison
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   Surface optimization: {optimization_time:.2f}s")
    print(f"   Full rendering: {render_time:.2f}s")
    print(f"   Total time: {optimization_time + render_time:.2f}s")
    
    expected_interactive = render_time < 2.0  # Should be interactive
    print(f"   Interactive performance: {'✅ YES' if expected_interactive else '❌ NO'}")
    
    return expected_interactive

if __name__ == "__main__":
    success = test_surface_optimization()
    if success:
        print("\n🎉 Surface optimization test PASSED!")
        print("   The surface should now render smoothly with good performance.")
    else:
        print("\n⚠️  Surface optimization test FAILED!")
        print("   Performance may still be an issue.")
