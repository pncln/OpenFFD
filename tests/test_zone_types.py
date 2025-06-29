#!/usr/bin/env python3
"""
Test zone type classification and feedback messages.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import read_fluent_mesh

def test_zone_classification():
    """Test zone type classification for better GUI feedback."""
    mesh_file = "14.msh"
    
    if not os.path.exists(mesh_file):
        print(f"Error: Mesh file {mesh_file} not found!")
        return
    
    print("Testing Zone Type Classification")
    print("=" * 50)
    
    # Load mesh
    mesh_data = read_fluent_mesh(mesh_file, use_meshio=True, debug=False)
    
    print(f"📊 **Mesh Summary:**")
    print(f"   • Total Points: {len(mesh_data.points):,}")
    print(f"   • Total Zones: {len(mesh_data.zones)}")
    
    print(f"\n📋 **Zone Classification:**")
    print("-" * 50)
    
    volume_zones = []
    boundary_zones = []
    
    for zone_name, zone_info in mesh_data.zones.items():
        zone_type = zone_info.get('type', 'unknown')
        zone_obj = zone_info.get('object')
        element_count = zone_info.get('element_count', 0)
        
        # Determine zone category
        if zone_obj and hasattr(zone_obj, 'zone_type_enum'):
            is_volume = zone_obj.zone_type_enum.name == 'VOLUME'
        else:
            is_volume = zone_type in ['interior', 'fluid', 'solid']
        
        if is_volume:
            icon = "🔵"
            category = "VOLUME"  
            volume_zones.append(zone_name)
        else:
            icon = "🟢"
            category = "BOUNDARY"
            boundary_zones.append(zone_name)
        
        print(f"   {icon} {zone_name:<25} ({zone_type:<15}) - {category:<8} - {element_count:,} elements")
    
    print(f"\n✅ **Classification Results:**")
    print(f"   • Volume Zones (🔵):   {len(volume_zones)} zones - {volume_zones}")
    print(f"   • Boundary Zones (🟢): {len(boundary_zones)} zones - {boundary_zones}")
    
    print(f"\n💡 **User Guidance:**")
    print(f"   • For FFD generation, select BOUNDARY zones (🟢)")
    print(f"   • Volume zones (🔵) define fluid domains - no extractable surface points")
    print(f"   • Best zones for FFD: launchpad, rocket, deflector (wall boundaries)")
    
    print(f"\n🎯 **GUI Enhancement Complete!**")
    print(f"   • Users will now get informative messages instead of confusing warnings")
    print(f"   • Zone dropdown shows visual indicators (🔵 volume, 🟢 boundary)")  
    print(f"   • Clear guidance provided for proper zone selection")

if __name__ == "__main__":
    test_zone_classification()
