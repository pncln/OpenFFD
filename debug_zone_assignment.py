#!/usr/bin/env python3
"""
Debug script to check zone assignment issues.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import FluentMeshReader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_zone_assignments():
    """Debug zone assignments to see if they are correct."""
    
    # Load the mesh
    mesh_file = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh"
    
    print("Loading mesh...")
    reader = FluentMeshReader(mesh_file)
    mesh = reader.read()
    
    # Check zones
    print("\n=== ZONE ANALYSIS ===")
    boundary_zones = []
    
    for zone in mesh.zone_list:
        print(f"\nZone: {zone.name}")
        print(f"  - Type: {zone.zone_type}")
        print(f"  - Zone Type Enum: {zone.zone_type_enum}")
        print(f"  - Zone ID: {zone.zone_id}")
        print(f"  - Num Faces: {zone.num_faces()}")
        print(f"  - Num Cells: {zone.num_cells()}")
        print(f"  - Point Indices Count: {len(zone.point_indices)}")
        
        if zone.zone_type_enum.name == 'BOUNDARY':
            boundary_zones.append(zone)
            
            # Show a few face examples
            if zone.faces:
                print(f"  - First few faces:")
                for i, face in enumerate(zone.faces[:3]):
                    print(f"    Face {i}: {face.node_indices[:5]}...")  # Show first 5 nodes
                    
    print(f"\nTotal boundary zones: {len(boundary_zones)}")
    
    # Check if boundary zones have different faces
    print("\n=== FACE OVERLAP ANALYSIS ===")
    
    if len(boundary_zones) >= 2:
        zone1 = boundary_zones[0]
        zone2 = boundary_zones[1]
        
        print(f"Comparing {zone1.name} vs {zone2.name}")
        
        # Get face node sets for comparison
        zone1_faces = set()
        zone2_faces = set()
        
        for face in zone1.faces[:100]:  # Check first 100 faces
            zone1_faces.add(tuple(sorted(face.node_indices)))
            
        for face in zone2.faces[:100]:  # Check first 100 faces
            zone2_faces.add(tuple(sorted(face.node_indices)))
            
        overlap = zone1_faces.intersection(zone2_faces)
        print(f"Face overlap: {len(overlap)}/{min(len(zone1_faces), len(zone2_faces))} faces")
        
        if len(overlap) > 50:  # More than 50% overlap
            print("❌ HIGH OVERLAP DETECTED - Zones are likely getting the same faces!")
        else:
            print("✅ Low overlap - Zones seem to have different faces")

if __name__ == "__main__":
    debug_zone_assignments()
