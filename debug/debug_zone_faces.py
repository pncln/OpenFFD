#!/usr/bin/env python3
"""
Debug why zone extraction returns no faces.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import read_fluent_mesh
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_zone_faces():
    """Debug zone face extraction."""
    mesh_file = "14.msh"
    
    if not os.path.exists(mesh_file):
        print(f"Error: Mesh file {mesh_file} not found!")
        return
    
    print("🔍 Debugging Zone Face Extraction")
    print("=" * 50)
    
    # Load mesh
    mesh_data = read_fluent_mesh(mesh_file, use_meshio=True, debug=False)
    
    # Get first boundary zone
    boundary_zones = []
    for zone_name, zone_info in mesh_data.zones.items():
        zone_type = zone_info.get('type', 'unknown')
        zone_obj = zone_info.get('object')
        
        # Skip volume zones
        if zone_obj and hasattr(zone_obj, 'zone_type_enum'):
            is_volume = zone_obj.zone_type_enum.name == 'VOLUME'
        else:
            is_volume = zone_type in ['interior', 'fluid', 'solid']
        
        if not is_volume:
            boundary_zones.append((zone_name, zone_obj))
    
    if not boundary_zones:
        print("❌ No boundary zones found!")
        return
    
    test_zone_name, test_zone = boundary_zones[0]
    print(f"\n🎯 Debugging zone: {test_zone_name}")
    print(f"   Zone object type: {type(test_zone)}")
    print(f"   Zone attributes: {dir(test_zone)}")
    
    # Check if zone has faces
    if hasattr(test_zone, 'faces'):
        faces = test_zone.faces
        print(f"   ✅ Zone has 'faces' attribute: {type(faces)}")
        print(f"   ✅ Number of faces: {len(faces) if faces else 0}")
        
        if faces:
            print(f"   🔧 First face type: {type(faces[0])}")
            print(f"   🔧 First face attributes: {dir(faces[0])}")
            
            first_face = faces[0]
            if hasattr(first_face, 'node_indices'):
                print(f"   ✅ Face has node_indices: {first_face.node_indices}")
            else:
                print(f"   ❌ Face has no node_indices")
                print(f"   🔧 Face data: {first_face}")
        else:
            print(f"   ❌ Zone.faces is empty")
    else:
        print(f"   ❌ Zone has no 'faces' attribute")
    
    # Check if zone has cells or elements
    if hasattr(test_zone, 'cells'):
        cells = test_zone.cells
        print(f"   ✅ Zone has 'cells' attribute: {type(cells)}, count: {len(cells) if cells else 0}")
    else:
        print(f"   ❌ Zone has no 'cells' attribute")
    
    if hasattr(test_zone, 'elements'):
        elements = test_zone.elements
        print(f"   ✅ Zone has 'elements' attribute: {type(elements)}, count: {len(elements) if elements else 0}")
    else:
        print(f"   ❌ Zone has no 'elements' attribute")
    
    # Check how zone is structured in mesh_data
    zone_info = mesh_data.zones[test_zone_name]
    print(f"\n🔧 Zone info structure:")
    for key, value in zone_info.items():
        print(f"   {key}: {type(value)} = {value}")
    
    # Try alternative face extraction methods
    print(f"\n🧪 Alternative extraction methods:")
    
    # Method 1: Check if mesh_data has face information
    if hasattr(mesh_data, 'cells'):
        print(f"   📋 Mesh cells available: {len(mesh_data.cells) if mesh_data.cells else 0}")
        if mesh_data.cells:
            for i, cell_block in enumerate(mesh_data.cells):
                print(f"      Cell block {i}: {cell_block.type}, {len(cell_block.data)} cells")
    
    # Method 2: Check meshio structure
    print(f"   📋 Mesh attributes: {[attr for attr in dir(mesh_data) if not attr.startswith('_')]}")

if __name__ == "__main__":
    debug_zone_faces()
