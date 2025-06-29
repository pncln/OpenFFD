#!/usr/bin/env python3

"""
Simple test for the native Fluent mesh parser
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import FluentMeshReader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_native_parser():
    """Test the native Fluent parser end-to-end."""
    
    print("=== TESTING NATIVE FLUENT PARSER ===")
    
    # Load mesh with native parser
    reader = FluentMeshReader("/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh")
    reader.force_ascii = True  # Force ASCII parsing
    
    mesh = reader.read()
    
    print(f"\n🎉 NATIVE PARSER SUCCESS! 🎉")
    print(f"✅ Points: {len(mesh.points):,}")
    print(f"✅ Zones: {len(mesh.zone_list):,}")
    print(f"✅ Faces: {len(mesh.faces):,}")
    
    print(f"\n=== BOUNDARY ZONES ===")
    boundary_zones = []
    for zone in mesh.zone_list:
        if zone.zone_type_enum.name == 'BOUNDARY' and len(zone.faces) > 0:
            boundary_zones.append(zone)
            print(f"✅ {zone.name}: {len(zone.faces):,} faces")
    
    print(f"\n🚀 NATIVE PARSER COMPLETED SUCCESSFULLY!")
    print(f"📊 Extracted {len(boundary_zones)} boundary zones with faces")
    print(f"📈 Total mesh elements: {len(mesh.points):,} points, {len(mesh.faces):,} faces")
    
    # Test key zones
    key_zones = ['rocket', 'deflector', 'launchpad']
    found_key_zones = 0
    for zone in mesh.zone_list:
        if zone.name in key_zones and len(zone.faces) > 0:
            found_key_zones += 1
            print(f"🎯 Key zone '{zone.name}': {len(zone.faces):,} faces ✅")
    
    if found_key_zones == len(key_zones):
        print(f"\n🏆 ALL KEY ZONES SUCCESSFULLY EXTRACTED!")
    else:
        print(f"\n⚠️  Found {found_key_zones}/{len(key_zones)} key zones")
    
    return mesh

if __name__ == "__main__":
    test_native_parser()
