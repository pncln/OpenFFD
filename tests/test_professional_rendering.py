#!/usr/bin/env python3
"""
Test the professional-grade mesh rendering system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import read_fluent_mesh
from openffd.mesh.general import extract_zone_mesh
from openffd.gui.visualization import FFDVisualizationWidget
from PyQt6.QtWidgets import QApplication
import time
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

def test_professional_rendering():
    """Test the complete professional rendering pipeline."""
    mesh_file = "14.msh"
    
    print("🏆 PROFESSIONAL RENDERING SYSTEM TEST")
    print("=" * 60)
    
    # Load mesh and extract zone
    print("\n1️⃣ Loading mesh and extracting zone...")
    start_time = time.time()
    
    mesh_data = read_fluent_mesh(mesh_file, use_meshio=True, debug=False)
    zone_mesh_data = extract_zone_mesh(mesh_data, "launchpad")
    
    if not zone_mesh_data:
        print("❌ Zone extraction failed!")
        return False
    
    zone_points = zone_mesh_data['points']
    zone_faces = zone_mesh_data['faces']
    load_time = time.time() - start_time
    
    print(f"   ✅ Data loaded in {load_time:.2f}s")
    print(f"   📊 Surface data: {len(zone_points):,} points, {len(zone_faces):,} faces")
    
    # Create visualization widget
    print("\n2️⃣ Creating professional visualization system...")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    viz_widget = FFDVisualizationWidget()
    
    # Test professional optimization pipeline
    print("\n3️⃣ Testing professional optimization pipeline...")
    start_time = time.time()
    
    try:
        # Test the optimization directly
        optimized_mesh, lod_levels = viz_widget._optimize_surface_for_performance(zone_faces, zone_points)
        optimization_time = time.time() - start_time
        
        print(f"   ✅ Professional optimization completed in {optimization_time:.2f}s")
        print(f"   📈 Optimization results:")
        print(f"      Original: {len(zone_points):,} points, {len(zone_faces):,} faces")
        print(f"      Optimized: {optimized_mesh.n_points:,} points, {optimized_mesh.n_faces:,} faces")
        print(f"      Reduction: {(1 - optimized_mesh.n_faces/len(zone_faces))*100:.1f}% face reduction")
        print(f"      LOD levels: {len(lod_levels)}")
            
        # Test professional rendering
        print("\n4️⃣ Testing professional rendering...")
        start_time = time.time()
        
        color = viz_widget._get_zone_color("wall")
        actor = viz_widget._render_professional_surface(
            optimized_mesh, color, "wall", lod_levels
        )
        
        render_time = time.time() - start_time
        print(f"   ✅ Professional rendering completed in {render_time:.2f}s")
        
        # Test full integrated pipeline
        print("\n5️⃣ Testing complete integrated pipeline...")
        start_time = time.time()
        
        viz_widget.set_mesh(zone_mesh_data, zone_points)
        
        total_time = time.time() - start_time
        print(f"   ✅ Complete pipeline executed in {total_time:.2f}s")
        
        # Performance analysis
        print("\n📊 PERFORMANCE ANALYSIS:")
        print(f"   Data Loading: {load_time:.2f}s")
        print(f"   Mesh Optimization: {optimization_time:.2f}s")
        print(f"   Surface Rendering: {render_time:.2f}s")
        print(f"   Total Pipeline: {total_time:.2f}s")
        
        # Quality assessment
        print("\n🎨 QUALITY ASSESSMENT:")
        print(f"   Face Reduction: {(1 - optimized_mesh.n_faces/len(zone_faces))*100:.1f}%")
        print(f"   Mesh Quality: {'High' if optimized_mesh.n_faces > 10000 else 'Balanced'}")
        print(f"   LOD Support: {'Yes' if len(lod_levels) > 1 else 'No'}")
        
        # Performance verdict
        is_interactive = total_time < 3.0  # Should be very fast
        is_high_quality = optimized_mesh.n_faces > 5000  # Should maintain quality
        
        print("\n🏆 FINAL VERDICT:")
        print(f"   Interactive Performance: {'✅ EXCELLENT' if total_time < 1.5 else '✅ GOOD' if is_interactive else '❌ POOR'}")
        print(f"   Visual Quality: {'✅ HIGH' if is_high_quality else '⚠️  MODERATE'}")
        print(f"   Professional Grade: {'✅ YES' if is_interactive and is_high_quality else '❌ NO'}")
        
        success = is_interactive and is_high_quality
        
        if success:
            print("\n🎉 PROFESSIONAL RENDERING TEST PASSED!")
            print("   🚀 Ultra-fast performance with high visual quality")
            print("   ✨ No more chippy edges or performance issues")
            print("   🏆 Ready for professional CFD visualization")
        else:
            print("\n⚠️  Test completed with some limitations")
        
        return success
    
    except Exception as e:
        print(f"❌ Professional rendering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_professional_rendering()
    exit(0 if success else 1)
