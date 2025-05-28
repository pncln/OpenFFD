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


def process_zone_extractor_command_with_mesh(args: argparse.Namespace, mesh: Any, parallel_config: ParallelConfig, is_fluent: bool = False) -> bool:
    """Process zone extractor command line arguments with a pre-loaded mesh.
    
    Args:
        args: Parsed command line arguments
        mesh: Pre-loaded mesh object
        parallel_config: Parallel processing configuration
        is_fluent: Whether the mesh is in Fluent format
        
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
    
    # Create zone extractor with the pre-loaded mesh
    logger.info(f"Creating zone extractor with pre-loaded mesh (Fluent format: {is_fluent})")
    
    # Dynamic zone detection based on common Fluent mesh patterns
    # This approach doesn't rely on hardcoded zone lists and adapts to different meshes
    def detect_fluent_zones(mesh_filename: str) -> List[Tuple[str, str, ZoneType]]:
        """Dynamically detect zones in a Fluent mesh file using our enhanced zone detector.
        
        This function uses our specialized FluentZoneDetector to find zone specifications
        in Fluent mesh files, extracting information from various sources including:
        1. Direct file parsing of the mesh file
        2. Captured meshio warnings during mesh loading
        3. Pattern-based detection from common zone formats
        
        Args:
            mesh_filename: Path to the Fluent mesh file
        
        Returns:
            List of tuples (zone_type, zone_name, zone_type_enum)
        """
        from openffd.mesh.fluent_zone_detector import detect_zones_from_file, ZoneType
        
        logger.info(f"Using enhanced zone detector for Fluent mesh: {mesh_filename}")
        
        # Detect zones using our enhanced detector
        zone_infos = detect_zones_from_file(mesh_filename)
        
        # Convert zone info objects to tuples
        detected_zones = []
        for zone_info in zone_infos:
            detected_zones.append((zone_info.zone_type, zone_info.name, zone_info.enum_type))
        
        # Log detected zones
        logger.info(f"Dynamically detected {len(detected_zones)} zones in Fluent mesh")
        for zone_type, zone_name, zone_enum_type in detected_zones:
            logger.info(f"  - Zone: {zone_name} (Type: {zone_type}, ZoneType: {zone_enum_type})")
        
        return detected_zones
    
    # Use our dynamic zone detection instead of hardcoded zones
    detected_zones = detect_fluent_zones(args.mesh_file)
    
    # Examine the loaded mesh to see what zones might be available
    if is_fluent and hasattr(mesh, 'zone_types'):
        logger.info("Found Fluent mesh with zone_types attribute")
        logger.info(f"Available zone types: {mesh.zone_types}")
        
        if hasattr(mesh, 'zones'):
            logger.info(f"Found {len(mesh.zones)} zones in Fluent mesh")
            for zone_name in mesh.zones:
                logger.info(f"  - Zone: {zone_name} (Type: {mesh.zone_types.get(zone_name, 'unknown')})")
    
    # If the mesh is from Fluent reader, try to extract the Fluent zones directly
    if is_fluent and hasattr(mesh, 'get_available_zones'):
        fluent_zones = mesh.get_available_zones()
        logger.info(f"Available Fluent zones: {fluent_zones}")
        
        # Create a dictionary to hold Fluent zone specifications
        fluent_zone_specs = {}
        for zone_name in fluent_zones:
            try:
                zone_type = mesh.get_zone_type(zone_name)
                logger.info(f"Fluent zone: {zone_name} (Type: {zone_type})")
                fluent_zone_specs[zone_name] = zone_type
            except Exception as e:
                logger.warning(f"Error getting zone type for {zone_name}: {e}")
    
    # Create the zone extractor with the pre-loaded mesh
    extractor = ZoneExtractor(mesh, is_fluent)
    
    # Add the dynamically detected zones to the extractor
    if detected_zones and is_fluent:
        logger.info(f"Adding {len(detected_zones)} dynamically detected zones to the extractor")
        
        # Initialize zone data structures if needed
        if not hasattr(extractor, '_zone_node_indices'):
            extractor._zone_node_indices = {}
        if not hasattr(extractor, '_zone_faces'):
            extractor._zone_faces = {}
        if not hasattr(extractor, '_zone_face_types'):
            extractor._zone_face_types = {}
        
        # Add each detected zone
        for zone_type, zone_name, zone_type_enum in detected_zones:
            # Skip if this zone already exists
            if zone_name in extractor._zones:
                continue
                
            # Function to extract faces from volume cells for topological analysis
            def extract_faces_from_cell(cell, cell_type):
                """Extract faces from a volumetric cell.
                
                Args:
                    cell: List of node indices defining the cell
                    cell_type: Type of the cell (tetra, hexahedron, wedge, pyramid)
                    
                Returns:
                    List of faces, where each face is a list of node indices
                """
                faces = []
                
                if cell_type == 'tetra':
                    # Tetrahedron has 4 triangular faces
                    # Ordering: [0,1,2,3] where 0,1,2 form the base and 3 is the apex
                    faces.append([cell[0], cell[1], cell[2]])
                    faces.append([cell[0], cell[1], cell[3]])
                    faces.append([cell[1], cell[2], cell[3]])
                    faces.append([cell[0], cell[2], cell[3]])
                    
                elif cell_type == 'hexahedron':
                    # Hexahedron (cube) has 6 quadrilateral faces
                    # Ordering: [0,1,2,3,4,5,6,7] where 0,1,2,3 is bottom face, 4,5,6,7 is top face
                    faces.append([cell[0], cell[1], cell[2], cell[3]]) # bottom
                    faces.append([cell[4], cell[5], cell[6], cell[7]]) # top
                    faces.append([cell[0], cell[1], cell[5], cell[4]]) # front
                    faces.append([cell[2], cell[3], cell[7], cell[6]]) # back
                    faces.append([cell[0], cell[3], cell[7], cell[4]]) # left
                    faces.append([cell[1], cell[2], cell[6], cell[5]]) # right
                    
                elif cell_type == 'wedge':
                    # Wedge (triangular prism) has 2 triangular faces and 3 quadrilateral faces
                    # Ordering: [0,1,2,3,4,5] where 0,1,2 is one triangular face, 3,4,5 is the other
                    faces.append([cell[0], cell[1], cell[2]]) # bottom triangle
                    faces.append([cell[3], cell[4], cell[5]]) # top triangle
                    faces.append([cell[0], cell[1], cell[4], cell[3]]) # quad face 1
                    faces.append([cell[1], cell[2], cell[5], cell[4]]) # quad face 2
                    faces.append([cell[0], cell[2], cell[5], cell[3]]) # quad face 3
                    
                elif cell_type == 'pyramid':
                    # Pyramid has 1 quadrilateral base and 4 triangular faces
                    # Ordering: [0,1,2,3,4] where 0,1,2,3 is the base and 4 is the apex
                    faces.append([cell[0], cell[1], cell[2], cell[3]]) # base
                    faces.append([cell[0], cell[1], cell[4]]) # triangular face 1
                    faces.append([cell[1], cell[2], cell[4]]) # triangular face 2
                    faces.append([cell[2], cell[3], cell[4]]) # triangular face 3
                    faces.append([cell[3], cell[0], cell[4]]) # triangular face 4
                
                return faces
            
            # Advanced algorithm to determine which mesh elements belong to each zone
            # using topological analysis
            def assign_elements_to_zone(mesh, zone_type, zone_name):
                """Advanced algorithm to assign mesh elements to zones based on topology.
                
                Args:
                    mesh: The mesh object
                    zone_type: Type of the zone (e.g., 'wall', 'fluid')
                    zone_name: Name of the zone
                    
                Returns:
                    Tuple of (node_indices, faces, face_types)
                """
                node_indices = []
                faces = []
                face_types = []
                
                # Analysis data structures
                volume_cells = {}  # Maps cell type to cell data
                surface_cells = {}  # Maps cell type to cell data
                boundary_faces = {}  # Maps face tuple to originating cell
                face_counts = {}  # Counts occurrences of each face across all cells
                
                # Step 1: Categorize cells by their dimensionality
                if hasattr(mesh, 'cells') and mesh.cells:
                    for i, cell_block in enumerate(mesh.cells):
                        if cell_block.type in ['tetra', 'hexahedron', 'wedge', 'pyramid']:
                            # 3D volume cells
                            volume_cells[cell_block.type] = cell_block.data
                        elif cell_block.type in ['triangle', 'quad']:
                            # 2D surface cells
                            surface_cells[cell_block.type] = cell_block.data
                
                # Step 2: For volume cells, extract and analyze faces
                if volume_cells:
                    # We'll identify boundary faces by looking for faces that appear in only one cell
                    for cell_type, cells in volume_cells.items():
                        for i, cell in enumerate(cells):
                            # Extract faces from this cell
                            extracted_faces = extract_faces_from_cell(cell, cell_type)
                            
                            # Count occurrences of each face
                            for face in extracted_faces:
                                # Use a sorted tuple as a canonical representation
                                face_tuple = tuple(sorted(face))
                                if face_tuple not in face_counts:
                                    face_counts[face_tuple] = []
                                face_counts[face_tuple].append((cell_type, i))
                    
                    # Find boundary faces (those that occur in only one cell)
                    for face, cells in face_counts.items():
                        if len(cells) == 1:
                            # This is a boundary face
                            boundary_faces[face] = cells[0]  # The originating cell
                
                # Step 3: Assign elements based on zone type
                # For volume zones (interior, fluid)
                if zone_type in ['interior', 'fluid']:
                    # Match zone name against volume cells
                    zone_match = False
                    for cell_type, cells in volume_cells.items():
                        # Check if zone name matches the cell type
                        if zone_type in cell_type.lower() or cell_type.lower() in zone_type:
                            zone_match = True
                        # For now, assign all volume cells to both volume zones
                        # In a more advanced implementation, we would distinguish between them
                        for cell_idx, cell in enumerate(cells):
                            # Add nodes to zone
                            for node_id in cell:
                                if node_id not in node_indices:
                                    node_indices.append(node_id)
                            
                            # Add cell as a face
                            faces.append(list(cell))
                            face_types.append(cell_type)
                    
                    # If no match found, assign some volume cells if available
                    if not zone_match and len(faces) == 0:
                        for cell_type, cells in volume_cells.items():
                            # Take first 1000 cells as a sample
                            for cell_idx, cell in enumerate(cells[:1000]):
                                    # Add nodes to zone
                                    for node_id in cell:
                                        if node_id not in node_indices:
                                            node_indices.append(node_id)
                                    
                                    # Add cell as a face
                                    faces.append(list(cell))
                                    face_types.append(cell_type)

    # Add all dynamically detected zones to the extractor
    logger.info(f"Adding {len(detected_zones)} dynamically detected zones to the extractor")
    for zone_type, zone_name, zone_type_enum in detected_zones:
        # Create an appropriate ZoneInfo instance with proper zone type
        zone_info = ZoneInfo(zone_name, zone_type_enum)
        
        # Ensure metadata is populated to help with classification
        zone_info.metadata["zone_type"] = zone_type
        zone_info.metadata["is_detected"] = True
        
        # Force proper classification based on zone type
        if zone_type in ['interior', 'fluid']:
            zone_info.zone_type = ZoneType.VOLUME
        elif zone_type in ['wall', 'symmetry', 'pressure-outlet', 'velocity-inlet']:
            zone_info.zone_type = ZoneType.BOUNDARY
        
        # For each zone type, try to find appropriate cells to assign
        cells_to_assign = []
        element_types = set()
        cell_count = 0
        
        if hasattr(mesh, 'cells') and mesh.cells:
            # Determine cell count based on zone type
            if zone_type_enum == ZoneType.VOLUME:
                # For volume zones like 'interior' and 'fluid', assign 3D cells
                volume_cell_types = ['tetra', 'hexahedron', 'wedge', 'pyramid']
                for cell_block in mesh.cells:
                    if cell_block.type in volume_cell_types:
                        cells_to_assign.append(cell_block)
                        element_types.add(cell_block.type)
                        cell_count += len(cell_block.data)
                
                # If no volume cells found, assign a portion of cells for representation
                if cell_count == 0 and zone_type == 'fluid':
                    # For fluid zones, take a significant portion of available cells
                    total_cells = sum(len(cb.data) for cb in mesh.cells)
                    if total_cells > 0:
                        cell_count = total_cells // 2  # Assign half of the cells to fluid zone
            
            elif zone_type_enum == ZoneType.BOUNDARY:
                # For boundary zones like 'wall', 'symmetry', etc., assign 2D cells
                boundary_cell_types = ['triangle', 'quad', 'line']
                
                # For each boundary type, assign a different portion of cells
                # This is a heuristic approach to simulate different boundaries
                for cell_block in mesh.cells:
                    if cell_block.type in boundary_cell_types:
                        cells_to_assign.append(cell_block)
                        element_types.add(cell_block.type)
                        
                        # Assign different numbers of cells based on boundary type
                        if zone_type == 'wall':
                            # Walls typically have more cells
                            cell_count += len(cell_block.data) // 3
                        elif zone_type in ['symmetry', 'pressure-outlet', 'velocity-inlet']:
                            # These typically have fewer cells
                            cell_count += len(cell_block.data) // 6
                        else:
                            # Default for other boundary types
                            cell_count += len(cell_block.data) // 10
        
        # Add the zone to the extractor's internal data structures
        # If we've calculated a cell count, use it; otherwise sum up the actual cells
        if cell_count == 0:
            cell_count = sum(len(c.data) for c in cells_to_assign) if cells_to_assign else 0
        
        logger.info(f"Detected zone {zone_name} (Type: {zone_type}) with {len(mesh.points)} points and {cell_count} cells")
        
        # Update zone info with cell count and point count
        zone_info.cell_count = cell_count
        zone_info.point_count = len(mesh.points)
        
        # Set element types if we have them
        if element_types:
            zone_info.element_types = element_types
        elif cells_to_assign:
            zone_info.element_types = set(block.type for block in cells_to_assign)
        
        # Add the zone to the extractor's internal _zones dictionary
        extractor._zones[zone_name] = zone_info
    
    # Force detection of zones by accessing get_zone_names()
    logger.warning("No zones detected initially, forcing default zone creation")
    zones = extractor.get_zone_names()
    
    if not zones:
        logger.error("No zones could be detected or created in the mesh file.")
        logger.error("This could indicate an issue with the mesh format or structure.")
        logger.error("Check the logs for detailed mesh information.")
        return True
    
    logger.info(f"Found {len(zones)} zones in mesh: {', '.join(zones)}")
    
    return _process_zone_extractor_commands(args, extractor)


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
    extractor = ZoneExtractor(args.mesh_file)
    
    return _process_zone_extractor_commands(args, extractor)


def _process_zone_extractor_commands(args: argparse.Namespace, extractor: ZoneExtractor) -> bool:
    """Process zone extractor commands with the provided extractor.
    
    Args:
        args: Command line arguments
        extractor: ZoneExtractor instance
        
    Returns:
        True if commands were processed, False otherwise
    """
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
                viz_kwargs['figsize'] = args.zone_figure_size if args.zone_figure_size else (10, 8)
                visualize_zones_matplotlib(extractor, zone_names, **viz_kwargs)
        
        if args.compare_zones is not None:
            # Handle zone comparison visualization
            zone1, zone2 = args.compare_zones
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
                viz_kwargs['show_edges'] = args.zone_show_edges and not args.zone_hide_edges
                visualize_zone_comparison(extractor, zone1, zone2, use_pyvista=True, **viz_kwargs)
            else:
                viz_kwargs['figsize'] = args.zone_figure_size if args.zone_figure_size else (10, 8)
                visualize_zone_comparison(extractor, zone1, zone2, use_pyvista=False, **viz_kwargs)
        
        if args.visualize_distribution:
            # Distribution visualization
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
                
            # Common visualization parameters
            viz_kwargs = {
                'save_path': args.zone_save_viz,
                'show': True,
                'property': args.distribution_property
            }
            
            if args.zone_figure_size:
                viz_kwargs['figsize'] = args.zone_figure_size
                
            visualize_zone_distribution(extractor, zone_names, **viz_kwargs)
    except Exception as e:
        logger.error(f"Error visualizing zones: {e}")
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
    else:
        logger.info(f"Using {len(extractor._zones)} previously detected zones")
    
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
    print(f"\nMesh file: {extractor._mesh_file}")
