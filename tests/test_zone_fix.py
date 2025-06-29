#!/usr/bin/env python3
"""
Test script to verify that the zone assignment fix works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import FluentMeshReader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_zone_assignment_fix():
    """Test that zones now get their correct faces."""
    
    # Load the mesh
    mesh_file = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh"
    
    print("Loading mesh with fixed zone assignment...")
    reader = FluentMeshReader(mesh_file)
    mesh = reader.read()
    
    # Check zones
    print("\n=== FIXED ZONE ANALYSIS ===")
    boundary_zones = []
    
    for zone in mesh.zone_list:
        if zone.zone_type_enum.name == 'BOUNDARY':
            boundary_zones.append(zone)
            print(f"\nZone: {zone.name}")
            print(f"  - Type: {zone.zone_type}")
            print(f"  - Zone ID: {zone.zone_id}")
            print(f"  - Num Faces: {zone.num_faces()}")
            print(f"  - Point Indices Count: {len(zone.point_indices)}")
            
            # Show a few face examples
            if zone.faces and len(zone.faces) > 0:
                print(f"  - First few face node indices:")
                for i, face in enumerate(zone.faces[:3]):
                    print(f"    Face {i}: {face.node_indices[:8]}...")  # Show first 8 nodes
                    
    print(f"\nTotal boundary zones: {len(boundary_zones)}")
    
    # Check if boundary zones now have different faces
    print("\n=== FACE DIFFERENTIATION TEST ===")
    
    if len(boundary_zones) >= 3:
        # Test launchpad vs rocket vs deflector
        zones_to_test = []
        for zone in boundary_zones:
            if zone.name in ['launchpad', 'rocket', 'deflector']:
                zones_to_test.append(zone)
        
        if len(zones_to_test) >= 2:
            zone1 = zones_to_test[0]
            zone2 = zones_to_test[1]
            
            print(f"Comparing {zone1.name} vs {zone2.name}")
            
            # Get face node sets for comparison
            zone1_faces = set()
            zone2_faces = set()
            
            sample_size = min(100, len(zone1.faces), len(zone2.faces))
            
            for i in range(sample_size):
                if i < len(zone1.faces):
                    zone1_faces.add(tuple(sorted(zone1.faces[i].node_indices)))
                if i < len(zone2.faces):
                    zone2_faces.add(tuple(sorted(zone2.faces[i].node_indices)))
                    
            overlap = zone1_faces.intersection(zone2_faces)
            print(f"Face overlap: {len(overlap)}/{min(len(zone1_faces), len(zone2_faces))} faces")
            print(f"Overlap percentage: {100 * len(overlap) / min(len(zone1_faces), len(zone2_faces)):.1f}%")
            
            if len(overlap) < 10:  # Less than 10% overlap
                print("✅ SUCCESS: Zones now have different faces!")
            else:
                print("❌ STILL ISSUE: Zones still have significant overlap")
        
        # Test a specific zone extraction
        print(f"\n=== TESTING ZONE EXTRACTION ===")
        from openffd.mesh.general import extract_zone_mesh
        
        for zone_name in ['launchpad', 'rocket', 'deflector']:
            if any(z.name == zone_name for z in boundary_zones):
                zone_data = extract_zone_mesh(mesh, zone_name)
                if zone_data:
                    print(f"Zone '{zone_name}': {len(zone_data['points'])} points, {len(zone_data['faces'])} faces")
                else:
                    print(f"Zone '{zone_name}': Failed to extract")

if __name__ == "__main__":
    test_zone_assignment_fix()
