"""Command-line interface for OpenFFD.

This module provides the main entry point for the OpenFFD command-line application.
"""

import argparse
import logging
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

from openffd.core.config import FFDConfig
from openffd.core.control_box import create_ffd_box, extract_patch_points
from openffd.core.hierarchical import create_hierarchical_ffd, HierarchicalFFD
from openffd.io.export import write_ffd_3df, write_ffd_xyz
from openffd.mesh.fluent import FluentMeshReader
from openffd.mesh.general import read_general_mesh, is_fluent_mesh
from openffd.utils.parallel import ParallelConfig
from openffd.visualization.ffd_viz import visualize_ffd
from openffd.visualization.mesh_viz import (
    visualize_mesh_with_patches,
    visualize_mesh_with_patches_pyvista,
)
from openffd.visualization.hierarchical_viz import (
    visualize_hierarchical_ffd_pyvista,
    visualize_hierarchical_ffd_matplotlib,
    visualize_influence_distribution
)
from openffd.cli.hierarchical import (
    add_hierarchical_ffd_args,
    process_hierarchical_ffd_command
)
from openffd.cli.zone_extractor import (
    add_zone_extractor_args,
    process_zone_extractor_command
)


def setup_logging(debug_mode: bool = False) -> None:
    """Configure logging based on debug mode.
    
    Args:
        debug_mode: If True, set logging level to DEBUG, otherwise INFO
    """
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate an FFD control lattice from a mesh file, optionally visualize it'
    )
    parser.add_argument('mesh_file', help='Path to mesh (Fluent .cas/.msh, VTK, STL, OBJ, etc.)')
    parser.add_argument('-p', '--patch', default=None, help='Name of zone, cell set/patch or Gmsh physical group')
    parser.add_argument('-d', '--dims', nargs=3, type=int, default=[4,4,4],
                        help='Control lattice dims: Nx Ny Nz')
    
    # Bounds control group
    bounds_group = parser.add_argument_group('bounds', 'Control the boundaries of the FFD box')
    bounds_group.add_argument('--x-min', type=float, help='Custom x-axis minimum bound')
    bounds_group.add_argument('--x-max', type=float, help='Custom x-axis maximum bound')
    bounds_group.add_argument('--y-min', type=float, help='Custom y-axis minimum bound')
    bounds_group.add_argument('--y-max', type=float, help='Custom y-axis maximum bound')
    bounds_group.add_argument('--z-min', type=float, help='Custom z-axis minimum bound')
    bounds_group.add_argument('--z-max', type=float, help='Custom z-axis maximum bound')
    bounds_group.add_argument('--x-bounds', nargs=2, type=float, help='Custom x-axis bounds (min_x max_x)')
    bounds_group.add_argument('--y-bounds', nargs=2, type=float, help='Custom y-axis bounds (min_y max_y)')
    bounds_group.add_argument('--z-bounds', nargs=2, type=float, help='Custom z-axis bounds (min_z max_z)')
    bounds_group.add_argument('-m', '--margin', type=float, default=0.0, help='Margin padding')
    
    # Output control group
    output_group = parser.add_argument_group('output', 'Control the output format and files')
    output_group.add_argument('-o', '--output', default='ffd_box.3df',
                        help='Output filename: .3df or .xyz')
    output_group.add_argument('--export-xyz', action='store_true',
                        help='Also export control points in .xyz format for DAFoam')
    
    # Visualization control group
    viz_group = parser.add_argument_group('visualization', 'Control visualization options')
    viz_group.add_argument('--plot', action='store_true', help='Visualize FFD lattice (automatically shows mesh points)')
    viz_group.add_argument('--save-plot', type=str, default=None, help='Save visualization to specified file path')
    viz_group.add_argument('--show-mesh', action='store_true', help='Show mesh points in visualization')
    viz_group.add_argument('--mesh-only', action='store_true', help='Show only the mesh without FFD box')
    viz_group.add_argument('--mesh-size', type=float, default=2.0, help='Size of mesh points in visualization')
    viz_group.add_argument('--mesh-alpha', type=float, default=0.3, help='Transparency of mesh points (0-1)')
    viz_group.add_argument('--mesh-color', default='blue', help='Color of mesh points')
    viz_group.add_argument('--ffd-point-size', type=float, default=10.0, help='Size of FFD control points')
    viz_group.add_argument('--ffd-alpha', type=float, default=1.0, help='Transparency of FFD box (0.0-1.0)')
    viz_group.add_argument('--ffd-color', type=str, default='b', help='Color of FFD control points')
    viz_group.add_argument('--show-full-ffd-grid', action='store_true', help='Show the complete internal FFD grid lines')
    viz_group.add_argument('--ffd-line-width', type=float, default=3.0, help='Width of FFD grid lines')
    viz_group.add_argument('--show-surface', action='store_true', default=True, help='Show mesh as a solid surface')
    viz_group.add_argument('--show-mesh-edges', action='store_true', help='Show mesh edges')
    viz_group.add_argument('--show-cell-ids', action='store_true', help='Show cell IDs')
    viz_group.add_argument('--no-surface', action='store_true', help='Do not show mesh as a solid surface')
    viz_group.add_argument('--surface-alpha', type=float, default=1.0, help='Transparency of surface (0.0-1.0)')
    viz_group.add_argument('--surface-color', type=str, default='gray', help='Color of surface')
    viz_group.add_argument('--hide-control-points', action='store_true', help='Hide FFD control points')
    viz_group.add_argument('--control-point-size', type=float, default=30.0, help='Size of control points')
    viz_group.add_argument('--lattice-width', type=float, default=1.5, help='Width of lattice lines')
    viz_group.add_argument('--view-angle', type=float, nargs=2, help='View angle as elevation azimuth')
    viz_group.add_argument('--view-axis', type=str, choices=['x', 'y', 'z', '-x', '-y', '-z'], help='Align view with specified axis')
    viz_group.add_argument('--no-auto-scale', action='store_true', help='Disable automatic scaling of small geometries')
    viz_group.add_argument('--scale-factor', type=float, help='Manual scale factor for visualization')
    viz_group.add_argument('--zoom-region', type=float, nargs=6, metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX', 'Z_MIN', 'Z_MAX'),
                        help='Zoom to specified region (x_min x_max y_min y_max z_min z_max)')
    viz_group.add_argument('--zoom-factor', type=float, default=1.0, help='Zoom factor (>1 zooms in, <1 zooms out)')
    viz_group.add_argument('--show-original-mesh', action='store_true', help='Show original mesh with patches in a separate window')
    viz_group.add_argument('--point-size', type=float, default=2.0, help='Size of points when showing original mesh')
    viz_group.add_argument('--detail-level', type=str, default='medium', choices=['low', 'medium', 'high'], 
                          help='Detail level for mesh visualization (low, medium, high)')
    viz_group.add_argument('--max-triangles', type=int, default=10000, help='Maximum number of triangles to generate for visualization')
    viz_group.add_argument('--max-points', type=int, default=5000, help='Maximum number of points to use for visualization per patch')
    viz_group.add_argument('--show-boundary-names', action='store_true', help='Display boundary names on the mesh patches')
    viz_group.add_argument('--text-size', type=float, default=24.0, help='Size of boundary name text labels')
    viz_group.add_argument('--text-color', type=str, default='black', help='Color of boundary name text labels')
    
    # Advanced settings
    adv_group = parser.add_argument_group('advanced', 'Advanced settings')
    adv_group.add_argument('--debug', action='store_true', help='Enable debug output')
    adv_group.add_argument('--force-ascii', action='store_true', help='Force ASCII reading for Fluent mesh')
    adv_group.add_argument('--force-binary', action='store_true', help='Force binary reading for Fluent mesh')
    
    # Parallel processing options
    parallel_group = parser.add_argument_group('parallel', 'Parallel processing options')
    parallel_group.add_argument('--parallel', action='store_true', default=False, 
                           help='Enable parallel processing for large meshes and visualization (default: disabled)')
    parallel_group.add_argument('--no-parallel', action='store_true', 
                           help='Disable parallel processing completely')
    parallel_group.add_argument('--parallel-method', choices=['process', 'thread'], default='process',
                           help='Method for parallelization: process (faster but more memory) or thread (default: process)')
    parallel_group.add_argument('--parallel-workers', type=int, default=None,
                           help='Number of worker processes/threads (default: auto-detect based on CPU count)')
    parallel_group.add_argument('--parallel-chunk-size', type=int, default=None,
                           help='Size of data chunks for parallel processing (default: auto-calculate)')
    parallel_group.add_argument('--parallel-threshold', type=int, default=100000,
                           help='Minimum data size to trigger parallelization (default: 100000 points)')
    parallel_group.add_argument('--parallel-viz', action='store_true', default=None,
                           help='Enable parallel processing for visualization only')
    parallel_group.add_argument('--no-parallel-viz', action='store_true',
                           help='Disable parallel processing for visualization even if parallel is enabled')
    
    # Add hierarchical FFD arguments
    add_hierarchical_ffd_args(parser)
    
    # Add zone extractor arguments
    add_zone_extractor_args(parser)
    
    return parser.parse_args()


def process_bounds(args: argparse.Namespace) -> List[Optional[Tuple[Optional[float], Optional[float]]]]:
    """Process custom bounds from command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of custom dimension bounds, with each element being a tuple of (min, max)
        or None if no custom bounds for that dimension.
    """
    custom_dims = [None, None, None]
    
    # Handle x bounds
    if args.x_bounds:
        custom_dims[0] = (args.x_bounds[0], args.x_bounds[1])
    else:
        x_min = args.x_min if args.x_min is not None else None
        x_max = args.x_max if args.x_max is not None else None
        
        if x_min is not None or x_max is not None:
            custom_dims[0] = (x_min, x_max)
    
    # Handle y bounds
    if args.y_bounds:
        custom_dims[1] = (args.y_bounds[0], args.y_bounds[1])
    else:
        y_min = args.y_min if args.y_min is not None else None
        y_max = args.y_max if args.y_max is not None else None
        
        if y_min is not None or y_max is not None:
            custom_dims[1] = (y_min, y_max)
    
    # Handle z bounds
    if args.z_bounds:
        custom_dims[2] = (args.z_bounds[0], args.z_bounds[1])
    else:
        z_min = args.z_min if args.z_min is not None else None
        z_max = args.z_max if args.z_max is not None else None
        
        if z_min is not None or z_max is not None:
            custom_dims[2] = (z_min, z_max)
    
    return custom_dims


def create_config_from_args(args: argparse.Namespace) -> Tuple[FFDConfig, ParallelConfig]:
    """Create FFD and parallel processing configurations from command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (FFD configuration object, Parallel configuration object)
    """
    custom_dims = process_bounds(args)
    
    # Create FFD configuration
    config = FFDConfig(
        dims=tuple(args.dims),
        margin=args.margin,
        custom_bounds=custom_dims if any(dim is not None for dim in custom_dims) else None,
        output_file=args.output,
        export_xyz=args.export_xyz,
        visualize=args.plot or args.save_plot is not None,
        visualization_options={
            'save_path': args.save_plot,
            'show_mesh': args.show_mesh,
            'mesh_only': args.mesh_only,
            'mesh_size': args.mesh_size,
            'mesh_alpha': args.mesh_alpha,
            'mesh_color': args.mesh_color,
            'ffd_point_size': args.ffd_point_size,
            'ffd_alpha': args.ffd_alpha,
            'ffd_color': args.ffd_color,
            'show_full_ffd_grid': args.show_full_ffd_grid,
            'ffd_line_width': args.ffd_line_width,
            'show_surface': args.show_surface and not args.no_surface,
            'show_mesh_edges': args.show_mesh_edges,
            'show_cell_ids': args.show_cell_ids,
            'surface_alpha': args.surface_alpha,
            'surface_color': args.surface_color,
            'hide_control_points': args.hide_control_points,
            'control_point_size': args.control_point_size,
            'lattice_width': args.lattice_width,
            'view_angle': tuple(args.view_angle) if args.view_angle else None,
            'view_axis': args.view_axis,
            'auto_scale': not args.no_auto_scale,
            'scale_factor': args.scale_factor,
            'zoom_region': args.zoom_region,
            'zoom_factor': args.zoom_factor,
            'show_original_mesh': args.show_original_mesh,
            'point_size': args.point_size,
            'detail_level': args.detail_level,
            'max_triangles': args.max_triangles,
            'max_points': args.max_points,
        },
        debug=args.debug,
        force_ascii=args.force_ascii,
        force_binary=args.force_binary,
    )
    
    # Create parallel processing configuration
    parallel_enabled = args.parallel and not args.no_parallel
    parallel_config = ParallelConfig(
        enabled=parallel_enabled,
        method=args.parallel_method,
        max_workers=args.parallel_workers,
        chunk_size=args.parallel_chunk_size,
        threshold=args.parallel_threshold
    )
    
    return config, parallel_config


def main() -> int:
    """Main function to run the FFD generator.
    
    Returns:
        Exit code: 0 for success, non-zero for error
    """
    # Parse command line arguments
    args = parse_arguments()

    # Set up logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing mesh file: {args.mesh_file}")
    
    # Validate input file exists
    if not os.path.exists(args.mesh_file):
        logger.error(f"Mesh file not found: {args.mesh_file}")
        return 1
    
    try:
        # Track mesh format for later use
        mesh_format = 'unknown'
        
        # Determine mesh type and read accordingly
        if is_fluent_mesh(args.mesh_file):
            # Read Fluent mesh
            mesh_format = 'fluent'
            reader_options = {
                "debug": args.debug
            }
            try:
                mesh = FluentMeshReader(args.mesh_file, **reader_options).read()
                logger.info(f"Successfully read Fluent mesh: {args.mesh_file}")
            except Exception as e:
                logger.error(f"Failed to read Fluent mesh: {e}")
                if args.debug:
                    logger.debug(traceback.format_exc())
                return 1
            
            if args.patch:
                try:
                    pts = mesh.get_zone_points(args.patch)
                    logger.info(f"Using points from zone '{args.patch}' ({len(pts)} points)")
                except ValueError as e:
                    logger.error(f"Error extracting zone points: {e}")
                    available_zones = mesh.get_available_zones()
                    logger.info(f"Available zones: {', '.join(available_zones)}")
                    return 1
            else:
                pts = mesh.points
                logger.info(f"Using all {len(mesh.points)} mesh points to build FFD box")
        else:
            # Use general mesh reader for other formats
            mesh_format = 'general'
            try:
                mesh = read_general_mesh(args.mesh_file)
                if args.patch:
                    try:
                        pts = extract_patch_points(mesh, args.patch)
                        logger.info(f"Using {pts.shape[0]} points from patch/group '{args.patch}'")
                    except ValueError as e:
                        logger.error(f"Error extracting patch points: {e}")
                        return 1
                else:
                    pts = mesh.points
                    logger.info(f"Using all {pts.shape[0]} mesh points to build FFD box")
            except ImportError as e:
                logger.error(f"Missing dependency: {e}")
                logger.info("Install required dependencies with: pip install meshio")
                return 1
            except Exception as e:
                logger.error(f"Error reading mesh file: {e}")
                if args.debug:
                    logger.debug(traceback.format_exc())
                return 1

        # Validate we have points to work with
        if len(pts) == 0:
            logger.error("No points found in the mesh for FFD box creation")
            return 1
            
        # Create parallel processing configuration
        parallel_enabled = args.parallel and not args.no_parallel
        parallel_config = ParallelConfig(
            enabled=parallel_enabled,
            method=args.parallel_method,
            max_workers=args.parallel_workers,
            chunk_size=args.parallel_chunk_size,
            threshold=args.parallel_threshold
        )
        
        # Pass the loaded mesh to the zone extractor
        # Process zone extractor commands if any
        try:
            from openffd.cli.zone_extractor import process_zone_extractor_command_with_mesh
            
            # If zone extractor commands are specified, use the already loaded mesh
            # Determine if this is a Fluent mesh
            mesh_is_fluent = mesh_format == 'fluent'
            
            if process_zone_extractor_command_with_mesh(args, mesh, parallel_config, mesh_is_fluent):
                # If zone extractor commands were processed successfully and no further
                # processing is needed (e.g., just listing zones), return success
                if args.list_zones and not (args.plot or args.output):
                    return 0
        except Exception as e:
            logger.error(f"Error processing zone extractor commands: {e}")
            if args.debug:
                logger.debug(traceback.format_exc())
            # Continue with standard FFD box creation even if zone extraction fails

        # Create FFD box
        try:
            # Process custom bounds
            custom_dims = process_bounds(args)
            
            # Check if any custom dimensions were specified
            custom_dims = custom_dims if any(dim is not None for dim in custom_dims) else None
            
            # Parallel config is already created before zone extraction
            
            # Check if hierarchical FFD is requested
            start_time = time.time()
            
            if args.hierarchical:
                # Create hierarchical FFD
                # Use base_dims if specified, otherwise fall back to dims
                base_dims = tuple(args.base_dims if args.base_dims is not None else args.dims)
                logger.info(f"Creating hierarchical FFD with base dimensions {base_dims}")
                logger.info(f"Max depth: {args.max_depth}, Subdivision factor: {args.subdivision_factor}")
                
                hffd = create_hierarchical_ffd(
                    mesh_points=pts,
                    base_dims=base_dims,
                    max_depth=args.max_depth,
                    subdivision_factor=args.subdivision_factor,
                    parallel_config=parallel_config,
                    custom_dims=custom_dims,  # Pass custom bounds (--x-min, --y-max, etc.)
                    margin=args.margin  # Pass margin parameter
                )
                
                # Extract information from hierarchical levels
                cps = hffd.root_level.control_points
                control_dim = hffd.root_level.dims
                bbox = hffd.root_level.bbox
                
                # Print information about levels
                logger.info(f"Created hierarchical FFD with {len(hffd.levels)} levels:")
                for level_info in hffd.get_level_info():
                    logger.info(
                        f"  Level {level_info['level_id']} (depth {level_info['depth']}): "
                        f"{level_info['dims']} dims, {level_info['num_control_points']} control points, "
                        f"weight: {level_info['weight_factor']:.2f}"
                    )
            else:
                # Create standard FFD box
                logger.info(f"Creating standard FFD box with dimensions {tuple(args.dims)}")
                cps, bbox = create_ffd_box(pts, tuple(args.dims), args.margin, custom_dims, parallel_config)
                control_dim = tuple(args.dims)  # Define control_dim based on args.dims
            
            end_time = time.time()
            logger.info(f'Bounding box min: {bbox[0]}, max: {bbox[1]}')
            logger.info(f'FFD creation completed in {end_time - start_time:.2f} seconds')
            
            # Log parallel processing status
            if parallel_config.enabled and len(pts) >= parallel_config.threshold:
                logger.info(f'Used parallel processing with {parallel_config.method} method')
                if parallel_config.max_workers:
                    logger.info(f'Used {parallel_config.max_workers} workers')
                else:
                    logger.info(f'Used auto-detected number of workers')
        except Exception as e:
            logger.error(f"Error creating FFD box: {e}")
            if args.debug:
                logger.debug(traceback.format_exc())
            return 1
        
        # Output FFD control points to file
        try:
            # First export in the requested primary format (from output filename extension)
            if args.output.endswith('.xyz'):
                write_ffd_xyz(cps, args.output)
                logger.info(f"FFD control points written to {args.output} in XYZ format")
                
                # If also requested to export in 3DF format
                if args.export_xyz:
                    # Already in XYZ format, so nothing to do
                    pass
            else:
                # Default to 3DF format
                write_ffd_3df(cps, args.output)
                logger.info(f"FFD control points written to {args.output} in 3DF format")
                
                # If also requested to export in XYZ format
                if args.export_xyz:
                    # Generate XYZ filename from the 3DF filename by replacing extension
                    xyz_filename = os.path.splitext(args.output)[0] + '.xyz'
                    write_ffd_xyz(cps, xyz_filename)
                    logger.info(f"FFD control points also written to {xyz_filename} in XYZ format")
        except Exception as e:
            logger.error(f"Error writing output file: {e}")
            if args.debug:
                logger.debug(traceback.format_exc())
            return 1
        
        # Show original mesh with patches in a separate window if requested
        if args.show_original_mesh:
            try:
                # For Fluent mesh, we already have the mesh object
                if is_fluent_mesh(args.mesh_file):
                    logger.info("Showing original Fluent mesh with patches")
                    save_path = os.path.splitext(args.save_plot)[0] + '_original.png' if args.save_plot else None
                    
                    # Set detail level parameters based on user selection
                    max_points = args.max_points
                    if args.detail_level == 'low':
                        max_points = min(1000, args.max_points)
                        use_multiple_projections = False
                    elif args.detail_level == 'medium':
                        max_points = min(5000, args.max_points)
                        use_multiple_projections = True
                    elif args.detail_level == 'high':
                        max_points = min(10000, args.max_points)
                        use_multiple_projections = True
                    
                    # Determine parallel visualization settings
                    use_parallel_viz = args.parallel_viz or (args.parallel and not args.no_parallel_viz)
                    
                    # Use PyVista-based visualization for much faster rendering and better quality
                    visualize_mesh_with_patches_pyvista(
                        mesh, 
                        save_path=save_path,
                        title=f"FFD Box: {os.path.basename(args.mesh_file)}",
                        point_size=args.point_size,
                        max_points_per_zone=args.max_points,
                        max_triangles=args.max_triangles,
                        detail_level=args.detail_level,
                        show_axes=True,
                        show_edges=False,
                        color_by_zone=True,  # Color by zone for better visualization
                        ffd_control_points=cps,
                        ffd_box_dims=list(args.dims),
                        ffd_opacity=args.ffd_alpha,
                        ffd_color=args.ffd_color,
                        ffd_point_size=args.ffd_point_size,
                        show_ffd_mesh=True,
                        zoom_region=args.zoom_region,
                        view_axis=args.view_axis,
                        show_boundary_names=args.show_boundary_names,
                        text_size=args.text_size,
                        text_color=args.text_color,
                        parallel=use_parallel_viz,
                        parallel_threshold=args.parallel_threshold,
                        parallel_workers=args.parallel_workers,
                        zoom_factor=args.zoom_factor  # Zoom factor for the view
                    )
                else:
                    # For other mesh types, we need to use the original mesh object
                    logger.info("Showing original mesh with patches")
                    save_path = os.path.splitext(args.save_plot)[0] + '_original.png' if args.save_plot else None
                    visualize_mesh_with_patches(
                        mesh, 
                        save_path=save_path,
                        title=f"Original Mesh: {os.path.basename(args.mesh_file)}",
                        point_size=args.point_size,
                        auto_scale=not args.no_auto_scale,
                        scale_factor=args.scale_factor,
                        view_angle=tuple(args.view_angle) if args.view_angle else None
                    )
            except Exception as e:
                logger.error(f"Error visualizing original mesh: {e}")
                if args.debug:
                    logger.debug(traceback.format_exc())
        
        # Visualize FFD box if requested
        if args.plot or args.save_plot:
            # When --plot is specified, automatically show the mesh too
            if args.plot:
                args.show_mesh = True
            try:
                # Determine if we should use parallel processing for visualization
                use_parallel_viz = parallel_config.enabled
                if args.parallel_viz is not None:
                    use_parallel_viz = args.parallel_viz
                if args.no_parallel_viz:
                    use_parallel_viz = False
                
                logger.info(f"Parallel processing for visualization: {'enabled' if use_parallel_viz else 'disabled'}")
                
                viz_parallel_config = None
                if use_parallel_viz:
                    viz_parallel_config = parallel_config
                
                # If hierarchical FFD, use specialized visualization
                if args.hierarchical and 'hffd' in locals():
                    try:
                        logger.info("Visualizing hierarchical FFD structure")
                        
                        # Log the information about what's being visualized
                        logger.info(f"Showing {len(hffd.levels)} hierarchical levels")
                        if args.show_levels:
                            logger.info(f"Displaying specific levels: {args.show_levels}")
                        
                        logger.info(f"Show influence regions: {args.show_influence}")
                        
                        # Select a subset of mesh points for influence visualization to avoid overwhelming display
                        if args.show_influence and len(pts) > 5000:
                            logger.info(f"Subsampling mesh points for influence visualization (original: {len(pts)} points)")
                            # Use systematic sampling to get representative points
                            sample_rate = max(1, len(pts) // 5000)
                            sample_indices = np.arange(0, len(pts), sample_rate)
                            sampled_pts = pts[sample_indices]
                            logger.info(f"Using {len(sampled_pts)} sampled points for influence visualization")
                        else:
                            sampled_pts = pts
                        
                        visualize_hierarchical_ffd_pyvista(
                            hffd=hffd,
                            show_levels=args.show_levels,
                            title=f"Hierarchical FFD for {os.path.basename(args.mesh_file)}",
                            save_path=args.save_plot,
                            mesh_points=sampled_pts if (args.show_mesh or args.mesh_only) else None,
                            show_mesh=args.show_mesh or args.mesh_only,
                            mesh_size=args.mesh_size,
                            mesh_alpha=args.mesh_alpha,
                            mesh_color=args.mesh_color,
                            point_size=args.control_point_size,
                            line_width=args.lattice_width,
                            color_by_level=True,
                            show_influence=args.show_influence,
                            view_axis=args.view_axis,
                            off_screen=False  # Force interactive display
                        )
                        
                        # Show influence distribution if requested
                        if args.show_influence:
                            logger.info("Generating influence distribution plot")
                            influence_save_path = None
                            if args.save_plot:
                                influence_save_path = os.path.splitext(args.save_plot)[0] + "_influence.png"
                            
                            visualize_influence_distribution(
                                hffd=hffd,
                                mesh_points=sampled_pts,
                                save_path=influence_save_path
                            )
                    except Exception as e:
                        if args.debug:
                            logger.exception("PyVista visualization failed, falling back to Matplotlib")
                        else:
                            logger.warning(f"PyVista visualization failed ({str(e)}), falling back to Matplotlib")
                        
                        # Fall back to Matplotlib for hierarchical FFD
                        visualize_hierarchical_ffd_matplotlib(
                            hffd=hffd,
                            show_levels=args.show_levels,
                            title=f"Hierarchical FFD for {os.path.basename(args.mesh_file)}",
                            save_path=args.save_plot,
                            mesh_points=pts if (args.show_mesh or args.mesh_only) else None,
                            show_mesh=args.show_mesh or args.mesh_only,
                            mesh_size=args.mesh_size,
                            mesh_alpha=args.mesh_alpha,
                            mesh_color=args.mesh_color,
                            point_size=args.control_point_size,
                            line_width=args.lattice_width
                        )
                else:
                    # Standard FFD visualization
                    try:
                        visualize_ffd(
                            control_points=cps,
                            bbox=bbox,
                            dims=control_dim,
                            mesh_points=pts if (args.show_mesh or args.mesh_only) else None,
                            title=f"FFD Control Box for {os.path.basename(args.mesh_file)}",
                            save_path=args.save_plot,
                            use_pyvista=True,
                            ffd_point_size=args.ffd_point_size,
                            ffd_color=args.ffd_color,
                            ffd_alpha=args.ffd_alpha,
                            mesh_point_size=args.mesh_size,
                            mesh_alpha=args.mesh_alpha,
                            mesh_color=args.mesh_color,
                            show_full_grid=args.show_full_ffd_grid,
                            control_point_size=args.control_point_size,
                            hide_control_points=args.hide_control_points,
                            view_angle=tuple(args.view_angle) if args.view_angle else None,
                            view_axis=args.view_axis,
                            auto_scale=not args.no_auto_scale,
                            scale_factor=args.scale_factor,
                            zoom_region=args.zoom_region,
                            zoom_factor=args.zoom_factor
                        )
                    except Exception as e:
                        if args.debug:
                            logger.exception("PyVista visualization failed, falling back to Matplotlib")
                        else:
                            logger.warning(f"PyVista visualization failed ({str(e)}), falling back to Matplotlib")
                        
                        # Fall back to Matplotlib
                        visualize_ffd(
                            control_points=cps,
                            bbox=bbox,
                            dims=control_dim,
                            mesh_points=pts if (args.show_mesh or args.mesh_only) else None,
                            title=f"FFD Control Box for {os.path.basename(args.mesh_file)}",
                            save_path=args.save_plot,
                            use_pyvista=False,
                            ffd_point_size=args.ffd_point_size,
                            ffd_color=args.ffd_color,
                            ffd_alpha=args.ffd_alpha,
                            mesh_point_size=args.mesh_size,
                            mesh_alpha=args.mesh_alpha,
                            mesh_color=args.mesh_color,
                            show_full_grid=args.show_full_ffd_grid,
                            control_point_size=args.control_point_size,
                            hide_control_points=args.hide_control_points,
                            view_angle=tuple(args.view_angle) if args.view_angle else None,
                            view_axis=args.view_axis,
                            auto_scale=not args.no_auto_scale,
                            scale_factor=args.scale_factor,
                            zoom_region=args.zoom_region,
                            zoom_factor=args.zoom_factor
                        )
                
            except Exception as e:
                logger.error(f"Error visualizing FFD box: {e}")
                if args.debug:
                    logger.debug(traceback.format_exc())
                return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            logger.debug(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
