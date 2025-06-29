#!/usr/bin/env python3

"""
Test script for the native Fluent mesh parser
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
    
    print(f"\n=== MESH SUMMARY ===")
    print(f"Points: {len(mesh.points)}")
    print(f"Zones: {len(mesh.zone_list)}")
    print(f"Total faces: {len(mesh.faces)}")
    
    print(f"\n=== ZONE DETAILS ===")
    for zone in mesh.zone_list:
        print(f"Zone: {zone.name}")
        print(f"  ID: {zone.zone_id}")
        print(f"  Type: {zone.zone_type}")
        print(f"  Category: {zone.zone_type_enum.name}")
        print(f"  Faces: {len(zone.faces)}")
        print(f"  Points: {len(zone.point_indices)}")
        print()
    
    # Test specific zones
    boundary_zones = [zone for zone in mesh.zone_list if zone.zone_type_enum.name == 'BOUNDARY']
    print(f"\n=== BOUNDARY ZONE TEST ===")
    print(f"Found {len(boundary_zones)} boundary zones")
    
    for zone in boundary_zones:
        if zone.name in ['rocket', 'deflector', 'launchpad']:
            print(f"\nTesting zone: {zone.name}")
            print(f"  Faces: {len(zone.faces)}")
            print(f"  Points: {len(zone.point_indices)}")
            
            if zone.faces:
                # Check if zone has point indices
                if hasattr(zone, 'point_indices') and len(zone.point_indices) > 0:
                    # Convert to numpy array if needed
                    import numpy as np
                    point_indices = np.array(zone.point_indices)
                    zone_points = mesh.points[point_indices]
                    
                    bounds = {
                        'x': [zone_points[:, 0].min(), zone_points[:, 0].max()],
                        'y': [zone_points[:, 1].min(), zone_points[:, 1].max()],
                        'z': [zone_points[:, 2].min(), zone_points[:, 2].max()]
                    }
                    print(f"  Bounds: X=[{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}], "
                          f"Y=[{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}], "
                          f"Z=[{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}]")
                else:
                    print(f"  No point indices available")
                
                print(f"  ✅ Zone '{zone.name}' successfully extracted")
            else:
                print(f"  ❌ Zone '{zone.name}' has no faces")
    
    print(f"\n=== NATIVE PARSER SUCCESS ===")
    print(f"✅ Successfully parsed {len(mesh.points)} points")
    print(f"✅ Successfully extracted {len(mesh.zone_list)} zones") 
    print(f"✅ Successfully assigned {len(mesh.faces)} faces")
    
    return mesh

if __name__ == "__main__":
    test_native_parser()
