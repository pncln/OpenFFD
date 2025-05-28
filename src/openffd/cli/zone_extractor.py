"""
Command-line interface for zone extraction functionality.

This module provides CLI commands for extracting and managing zones from mesh files.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from openffd.mesh.zone_extractor import (
    ZoneExtractor, ZoneType, ZoneInfo, 
    extract_zones_parallel, read_mesh_with_zones
)
from openffd.utils.parallel import ParallelConfig
from openffd.visualization.zone_viz import (
    visualize_zones_matplotlib,
    visualize_zones_pyvista,
    visualize_zone_comparison,
    visualize_zone_distribution,
    visualize_mesh_with_zones
)


logger = logging.getLogger(__name__)


def add_zone_extractor_args(parser: argparse.ArgumentParser) -> None:
    """Add zone extractor arguments to an ArgumentParser.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    # Create a group for zone extraction options
    zone_group = parser.add_argument_group('zone extraction', 'Options for zone extraction')
    
    # Zone listing and selection
    zone_group.add_argument('--list-zones', action='store_true',
                         help='List all available zones in the mesh')
    zone_group.add_argument('--zone-type', type=str, choices=['all', 'volume', 'boundary', 'interface'],
                         default='all', help='Filter zones by type (default: all)')
    zone_group.add_argument('--extract-zone', type=str, 
                         help='Extract a specific zone by name')
    zone_group.add_argument('--extract-boundary', type=str,
                         help='Extract a specific boundary zone by name')
    
    # Output options
    zone_group.add_argument('--save-zone', type=str,
                         help='Save extracted zone to specified file path')
    zone_group.add_argument('--save-boundary', type=str,
                         help='Save extracted boundary to specified file path')
    zone_group.add_argument('--export-zones-summary', type=str,
                         help='Export a summary of all zones to specified JSON file')
    
    # Filtering options
    zone_group.add_argument('--zone-name-filter', type=str, 
                         help='Filter zones by name (supports glob patterns)')
    zone_group.add_argument('--min-cells', type=int, default=0,
                         help='Minimum number of cells for a zone to be included')
    zone_group.add_argument('--max-cells', type=int, default=None,
                         help='Maximum number of cells for a zone to be included')
    
    # Visualization options
    viz_group = parser.add_argument_group('zone visualization', 'Options for zone visualization')
    viz_group.add_argument('--visualize-zones', action='store_true',
                         help='Visualize available zones')
    viz_group.add_argument('--compare-zones', nargs=2, metavar=('ZONE1', 'ZONE2'),
                         help='Compare two zones visually')
    viz_group.add_argument('--visualize-distribution', action='store_true',
                         help='Visualize distribution of zones by size')
    viz_group.add_argument('--distribution-property', type=str, choices=['point_count', 'cell_count'],
                         default='point_count', help='Property to use for distribution visualization')
    viz_group.add_argument('--use-pyvista', action='store_true', default=True,
                         help='Use PyVista for advanced visualization if available')
    viz_group.add_argument('--use-matplotlib', action='store_true',
                         help='Force using Matplotlib instead of PyVista')
    viz_group.add_argument('--zone-save-viz', type=str,
                         help='Save zone visualization to specified file path')
    viz_group.add_argument('--zone-point-size', type=float, default=5.0,
                         help='Size of points in zone visualization')
    viz_group.add_argument('--zone-opacity', type=float, default=0.7,
                         help='Opacity of points/surfaces in zone visualization (0.0-1.0)')
    viz_group.add_argument('--zone-figure-size', nargs=2, type=int, default=[10, 8], metavar=('WIDTH', 'HEIGHT'),
                         help='Figure size for matplotlib zone visualizations')
    viz_group.add_argument('--zone-view-angle', nargs=2, type=float, metavar=('ELEVATION', 'AZIMUTH'),
                         help='View angle for 3D zone visualization')
    viz_group.add_argument('--zone-show-edges', action='store_true', default=True,
                         help='Show edges in PyVista zone visualizations')
    viz_group.add_argument('--zone-hide-edges', action='store_true',
                         help='Hide edges in PyVista zone visualizations')
    viz_group.add_argument('--zones-as-surfaces', action='store_true', default=True,
                         help='Show zones as surfaces in PyVista visualizations when possible')
    viz_group.add_argument('--zones-as-points', action='store_true',
                         help='Always show zones as points in visualizations')


def process_zone_extractor_command(args: argparse.Namespace, parallel_config: ParallelConfig) -> bool:
    """Process zone extractor command line arguments.
    
    Args:
        args: Parsed command line arguments
        parallel_config: Parallel processing configuration
        
    Returns:
        True if zone extractor command was processed, False otherwise
    """
    # Check if any zone extractor commands were specified
    zone_extractor_commands = [
        args.list_zones,
        args.extract_zone is not None,
        args.extract_boundary is not None,
        args.export_zones_summary is not None,
        args.visualize_zones,
        args.compare_zones is not None,
        args.visualize_distribution
    ]
    
    if not any(zone_extractor_commands):
        return False
    
    # Create zone extractor
    extractor = ZoneExtractor(args.mesh_file, parallel_config)
    
    # Process commands
    if args.list_zones:
        _list_zones(extractor, args.zone_type, args.zone_name_filter, args.min_cells, args.max_cells)
        
    if args.export_zones_summary is not None:
        extractor.export_zones_summary(args.export_zones_summary)
        logger.info(f"Exported zones summary to {args.export_zones_summary}")
        
    if args.extract_zone is not None:
        zone_mesh = extractor.extract_zone_mesh(args.extract_zone)
        logger.info(f"Extracted zone '{args.extract_zone}' with {len(zone_mesh.points)} points")
        
        if args.save_zone is not None:
            extractor.save_zone_mesh(args.extract_zone, args.save_zone)
            logger.info(f"Saved zone '{args.extract_zone}' to {args.save_zone}")
            
    if args.extract_boundary is not None:
        boundary_mesh = extractor.extract_boundary_mesh(args.extract_boundary)
        logger.info(f"Extracted boundary '{args.extract_boundary}' with {len(boundary_mesh.points)} points")
        
        if args.save_boundary is not None:
            extractor.save_zone_mesh(args.extract_boundary, args.save_boundary)
            logger.info(f"Saved boundary '{args.extract_boundary}' to {args.save_boundary}")
    
    # Handle visualization commands
    # Determine visualization backend
    use_pyvista = args.use_pyvista and not args.use_matplotlib
    try:
        if args.visualize_zones:
            # Filter zones if requested
            zone_type_filter = None
            if args.zone_type != 'all':
                zone_type_filter = ZoneType.from_string(args.zone_type)
                
            zone_names = extractor.get_zone_names(zone_type_filter)
            
            # Apply name filter if specified
            if args.zone_name_filter:
                import fnmatch
                zone_names = [name for name in zone_names 
                             if fnmatch.fnmatch(name, args.zone_name_filter)]
            
            # Apply cell count filters
            if args.min_cells > 0 or args.max_cells is not None:
                filtered_names = []
                for name in zone_names:
                    zone_info = extractor.get_zone_info(name)
                    if zone_info.cell_count < args.min_cells:
                        continue
                    if args.max_cells is not None and zone_info.cell_count > args.max_cells:
                        continue
                    filtered_names.append(name)
                zone_names = filtered_names
            
            logger.info(f"Visualizing {len(zone_names)} zones")
            
            # Common visualization parameters
            viz_kwargs = {
                'point_size': args.zone_point_size,
                'alpha': args.zone_opacity,
                'save_path': args.zone_save_viz,
                'show': True
            }
            
            if args.zone_view_angle:
                viz_kwargs['view_angle'] = tuple(args.zone_view_angle)
                
            if use_pyvista:
                # PyVista specific options
                viz_kwargs['show_edges'] = args.zone_show_edges and not args.zone_hide_edges
                viz_kwargs['show_zones_as_surfaces'] = args.zones_as_surfaces and not args.zones_as_points
                visualize_zones_pyvista(extractor, zone_names, **viz_kwargs)
            else:
                # Matplotlib specific options
                viz_kwargs['figure_size'] = tuple(args.zone_figure_size)
                visualize_zones_matplotlib(extractor, zone_names, **viz_kwargs)
        
        if args.compare_zones is not None:
            # Extract the two zone names
            zone1, zone2 = args.compare_zones
            logger.info(f"Comparing zones: '{zone1}' and '{zone2}'")
            
            # Visualization parameters
            viz_kwargs = {
                'point_size': args.zone_point_size,
                'alpha': args.zone_opacity,
                'save_path': args.zone_save_viz,
                'show': True,
                'use_pyvista': use_pyvista
            }
            
            if args.zone_view_angle:
                viz_kwargs['view_angle'] = tuple(args.zone_view_angle)
                
            if not use_pyvista:
                viz_kwargs['figure_size'] = tuple(args.zone_figure_size)
                
            visualize_zone_comparison(extractor, zone1, zone2, **viz_kwargs)
        
        if args.visualize_distribution:
            # Determine zone type filter
            zone_type_filter = None
            if args.zone_type != 'all':
                zone_type_filter = ZoneType.from_string(args.zone_type)
            
            logger.info(f"Visualizing distribution of {args.distribution_property} across zones")
            
            # Visualization parameters
            viz_kwargs = {
                'property_name': args.distribution_property,
                'zone_type': zone_type_filter,
                'figure_size': tuple(args.zone_figure_size),
                'save_path': args.zone_save_viz,
                'show': True
            }
            
            # Apply min/max filters
            if args.min_cells > 0:
                viz_kwargs['min_value'] = args.min_cells
            if args.max_cells is not None:
                viz_kwargs['max_value'] = args.max_cells
                
            visualize_zone_distribution(extractor, **viz_kwargs)
    
    except ImportError as e:
        logger.error(f"Visualization failed due to missing dependency: {e}")
        logger.info("Install required dependencies with: pip install matplotlib pyvista")
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        if args.debug:
            import traceback
            logger.debug(traceback.format_exc())
    
    return True


def _list_zones(
    extractor: ZoneExtractor, 
    zone_type_str: str = 'all',
    name_filter: Optional[str] = None,
    min_cells: int = 0,
    max_cells: Optional[int] = None
) -> None:
    """List all zones in the mesh, optionally filtered.
    
    Args:
        extractor: ZoneExtractor object
        zone_type_str: Zone type filter ('all', 'volume', 'boundary', 'interface')
        name_filter: Optional name filter (glob pattern)
        min_cells: Minimum number of cells for a zone to be included
        max_cells: Maximum number of cells for a zone to be included
    """
    # Force loading of the mesh if not already loaded
    if not extractor._loaded:
        extractor._load_mesh()
    
    # Check if there are any zones, and if not, force default zone creation
    if len(extractor._zones) == 0:
        logger.warning("No zones detected initially, forcing default zone creation")
        extractor._create_default_zones()
    
    # Convert zone type string to ZoneType
    zone_type = None
    if zone_type_str != 'all':
        zone_type = ZoneType.from_string(zone_type_str)
    
    # Get all zones, filtered by type if specified
    zones = extractor.get_zone_info()
    
    if not zones:
        print("\nNo zones could be detected or created in the mesh file.")
        print("This could indicate an issue with the mesh format or structure.")
        print("Check the logs for detailed mesh information.")
        return
    
    # Apply filters
    filtered_zones = {}
    for name, info in zones.items():
        # Filter by type
        if zone_type is not None and info.zone_type != zone_type:
            continue
            
        # Filter by name
        if name_filter is not None:
            import fnmatch
            if not fnmatch.fnmatch(name, name_filter):
                continue
                
        # Filter by cell count
        if info.cell_count < min_cells:
            continue
            
        if max_cells is not None and info.cell_count > max_cells:
            continue
            
        filtered_zones[name] = info
    
    # Group by zone type
    volume_zones = {name: info for name, info in filtered_zones.items() 
                    if info.zone_type == ZoneType.VOLUME}
    boundary_zones = {name: info for name, info in filtered_zones.items() 
                      if info.zone_type == ZoneType.BOUNDARY}
    interface_zones = {name: info for name, info in filtered_zones.items() 
                       if info.zone_type == ZoneType.INTERFACE}
    other_zones = {name: info for name, info in filtered_zones.items() 
                   if info.zone_type == ZoneType.UNKNOWN}
    
    # Print summary
    print(f"\nFound {len(filtered_zones)} zones in the mesh:")
    print(f"  - {len(volume_zones)} volume zones")
    print(f"  - {len(boundary_zones)} boundary zones")
    print(f"  - {len(interface_zones)} interface zones")
    print(f"  - {len(other_zones)} unknown zones")
    
    # Print details for each zone type
    if volume_zones:
        print("\nVolume Zones:")
        for name, info in sorted(volume_zones.items()):
            auto_created = "(auto-created)" if info.metadata.get("auto_created", False) else ""
            print(f"  - {name}: {info.cell_count} cells, {info.point_count} points {auto_created}")
            if info.element_types:
                print(f"    Element types: {', '.join(sorted(info.element_types))}")
    
    if boundary_zones:
        print("\nBoundary Zones:")
        for name, info in sorted(boundary_zones.items()):
            auto_created = "(auto-created)" if info.metadata.get("auto_created", False) else ""
            print(f"  - {name}: {info.cell_count} cells, {info.point_count} points {auto_created}")
            if info.element_types:
                print(f"    Element types: {', '.join(sorted(info.element_types))}")
    
    if interface_zones:
        print("\nInterface Zones:")
        for name, info in sorted(interface_zones.items()):
            auto_created = "(auto-created)" if info.metadata.get("auto_created", False) else ""
            print(f"  - {name}: {info.cell_count} cells, {info.point_count} points {auto_created}")
            if info.element_types:
                print(f"    Element types: {', '.join(sorted(info.element_types))}")
    
    if other_zones:
        print("\nOther Zones:")
        for name, info in sorted(other_zones.items()):
            auto_created = "(auto-created)" if info.metadata.get("auto_created", False) else ""
            print(f"  - {name}: {info.cell_count} cells, {info.point_count} points {auto_created}")
            if info.element_types:
                print(f"    Element types: {', '.join(sorted(info.element_types))}")
                
    # Print mesh file information
    print(f"\nMesh file: {extractor.mesh_file}")
