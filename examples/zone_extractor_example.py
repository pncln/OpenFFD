#!/usr/bin/env python
"""
Example script demonstrating the usage of the ZoneExtractor class.

This script shows how to extract zones and boundaries from mesh files
using the OpenFFD zone extractor functionality.
"""

import argparse
import os
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openffd.mesh import ZoneExtractor, ZoneType
from openffd.utils.parallel import ParallelConfig


def main():
    """Run the zone extractor example."""
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Zone Extractor Example')
    parser.add_argument('mesh_file', help='Path to mesh file')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    parser.add_argument('--extract-zone', help='Name of zone to extract')
    parser.add_argument('--extract-boundary', help='Name of boundary to extract')
    parser.add_argument('--save-to', help='Path to save extracted zone')
    parser.add_argument('--list-zones', action='store_true', help='List all zones in the mesh')
    parser.add_argument('--export-summary', help='Export zones summary to JSON file')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('zone_extractor_example')

    # Configure parallel processing if requested
    parallel_config = ParallelConfig(
        enabled=args.parallel,
        method='process',
        max_workers=None,  # Auto-detect
        threshold=100000   # Threshold for parallelization
    )

    # Create zone extractor
    extractor = ZoneExtractor(args.mesh_file, parallel_config)
    
    # List zones if requested
    if args.list_zones:
        zones = extractor.get_zone_info()
        
        # Group zones by type
        volume_zones = {name: info for name, info in zones.items() 
                       if info.zone_type == ZoneType.VOLUME}
        boundary_zones = {name: info for name, info in zones.items() 
                         if info.zone_type == ZoneType.BOUNDARY}
        interface_zones = {name: info for name, info in zones.items() 
                          if info.zone_type == ZoneType.INTERFACE}
        other_zones = {name: info for name, info in zones.items() 
                      if info.zone_type == ZoneType.UNKNOWN}
        
        print(f"\nFound {len(zones)} zones in mesh file: {args.mesh_file}")
        print(f"  - {len(volume_zones)} volume zones")
        print(f"  - {len(boundary_zones)} boundary zones")
        print(f"  - {len(interface_zones)} interface zones")
        print(f"  - {len(other_zones)} unknown zones")
        
        if volume_zones:
            print("\nVolume Zones:")
            for name, info in sorted(volume_zones.items()):
                print(f"  - {name}: {info.cell_count} cells, {info.point_count} points")
        
        if boundary_zones:
            print("\nBoundary Zones:")
            for name, info in sorted(boundary_zones.items()):
                print(f"  - {name}: {info.cell_count} cells, {info.point_count} points")
    
    # Extract a specific zone if requested
    if args.extract_zone:
        logger.info(f"Extracting zone: {args.extract_zone}")
        zone_mesh = extractor.extract_zone_mesh(args.extract_zone)
        logger.info(f"Extracted zone with {len(zone_mesh.points)} points")
        
        # Save the extracted zone if requested
        if args.save_to:
            extractor.save_zone_mesh(args.extract_zone, args.save_to)
            logger.info(f"Saved zone to: {args.save_to}")
    
    # Extract a specific boundary if requested
    if args.extract_boundary:
        logger.info(f"Extracting boundary: {args.extract_boundary}")
        try:
            boundary_mesh = extractor.extract_boundary_mesh(args.extract_boundary)
            logger.info(f"Extracted boundary with {len(boundary_mesh.points)} points")
            
            # Save the extracted boundary if requested
            if args.save_to:
                extractor.save_zone_mesh(args.extract_boundary, args.save_to)
                logger.info(f"Saved boundary to: {args.save_to}")
        except ValueError as e:
            logger.error(f"Error extracting boundary: {e}")
            
    # Export zones summary if requested
    if args.export_summary:
        extractor.export_zones_summary(args.export_summary)
        logger.info(f"Exported zones summary to: {args.export_summary}")


if __name__ == "__main__":
    main()
