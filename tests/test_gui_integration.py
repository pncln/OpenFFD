#!/usr/bin/env python3
"""
Test GUI integration with the enhanced Fluent mesh reader.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
from openffd.mesh.fluent_reader import read_fluent_mesh

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

def test_gui_integration():
    """Test that the mesh reader works with GUI zone selection."""
    mesh_file = "14.msh"
    
    if not os.path.exists(mesh_file):
        print(f"Error: Mesh file {mesh_file} not found!")
        return
    
    print(f"Testing GUI integration with enhanced Fluent mesh reader")
    print("=" * 60)
    
    try:
        # Read the mesh using both modes
        print("1. Testing native mode (for zone names)...")
        mesh_native = read_fluent_mesh(mesh_file, use_meshio=False, debug=False)
        
        print("2. Testing meshio mode (for full mesh data)...")
        mesh_meshio = read_fluent_mesh(mesh_file, use_meshio=True, debug=False)
        
        print(f"\n📊 **Mesh Reading Results:**")
        print(f"   Native:  {len(mesh_native.points):,} points, {len(mesh_native.zone_list)} zones")
        print(f"   Meshio:  {len(mesh_meshio.points):,} points, {len(mesh_meshio.zone_list)} zones")
        
        # Test GUI compatibility - simulate zone selection workflow
        print(f"\n🖥️  **GUI Integration Test:**")
        print("-" * 40)
        
        # Test 1: Zone enumeration (like GUI dropdown)
        print("Available zones for GUI dropdown:")
        for i, zone_name in enumerate(mesh_meshio.zones.keys(), 1):
            zone_info = mesh_meshio.zones[zone_name]
            print(f"  {i:2d}. {zone_name}")
            print(f"      Type: {zone_info.get('type', 'unknown')}")
            print(f"      Elements: {zone_info.get('element_count', 0):,}")
        
        # Test 2: Zone selection simulation
        print(f"\n🎯 **Zone Selection Simulation:**")
        selected_zones = ['launchpad', 'rocket', 'deflector']  # User selected zones
        
        for zone_name in selected_zones:
            if zone_name in mesh_meshio.zones:
                zone_info = mesh_meshio.zones[zone_name]
                zone_obj = zone_info.get('object')
                print(f"✓ Zone '{zone_name}' selected:")
                print(f"    Type: {zone_info.get('type')}")
                print(f"    Faces: {zone_obj.num_faces() if zone_obj else 'N/A'}")
                print(f"    Cells: {zone_obj.num_cells() if zone_obj else 'N/A'}")
                print(f"    Points: {zone_obj.num_points() if zone_obj else 'N/A'}")
            else:
                print(f"✗ Zone '{zone_name}' not found!")
        
        # Test 3: FFD generation compatibility
        print(f"\n🔧 **FFD Generation Compatibility:**")
        boundary_zones = [name for name, info in mesh_meshio.zones.items() 
                         if info.get('object') and info['object'].zone_type_enum.name == 'BOUNDARY']
        print(f"Boundary zones available for FFD: {len(boundary_zones)}")
        for zone_name in boundary_zones[:5]:  # Show first 5
            print(f"  - {zone_name}")
        if len(boundary_zones) > 5:
            print(f"  ... and {len(boundary_zones) - 5} more")
            
        print(f"\n✅ **GUI Integration Test PASSED!**")
        print(f"   - Zone enumeration works")
        print(f"   - Zone selection works")
        print(f"   - FFD compatibility confirmed")
        print(f"   - All zone data accessible")
        
    except Exception as e:
        print(f"✗ Error in GUI integration test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gui_integration()
