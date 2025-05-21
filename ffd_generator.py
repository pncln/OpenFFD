#!/usr/bin/env python3
"""
Python script to generate an FFD (Free-Form Deformation) control box for mesh files including Fluent meshes,
export the control lattice either in .3df format or in simple .xyz format for DAFoam,
and visualize the FFD control points and bounding box.

Supports Fluent mesh (.cas, .msh) files as well as common mesh formats including VTK, STL, OBJ, and Gmsh .msh files,
and can restrict the FFD box to a specified mesh cell set/patch, including Fluent zones.

Author: TÜBİTAK
Version: 1.0.0
"""
import argparse
import os
import sys
import logging
from typing import List, Tuple, Optional, Union, Dict, Any
import traceback

# Import our modules
from mesh_readers.fluent_reader import FluentMeshReader
from mesh_readers.general_reader import read_general_mesh, is_fluent_mesh
from ffd_utils.control_box import create_ffd_box, extract_patch_points
from ffd_utils.io import write_ffd_3df, write_ffd_xyz
from ffd_utils.visualization import visualize_ffd, visualize_mesh_with_patches

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
    parser.add_argument('--x-min', type=float, help='Custom x-axis minimum bound')
    parser.add_argument('--x-max', type=float, help='Custom x-axis maximum bound')
    parser.add_argument('--y-min', type=float, help='Custom y-axis minimum bound')
    parser.add_argument('--y-max', type=float, help='Custom y-axis maximum bound')
    parser.add_argument('--z-min', type=float, help='Custom z-axis minimum bound')
    parser.add_argument('--z-max', type=float, help='Custom z-axis maximum bound')
    parser.add_argument('--x-bounds', nargs=2, type=float, help='Custom x-axis bounds (min_x max_x)')
    parser.add_argument('--y-bounds', nargs=2, type=float, help='Custom y-axis bounds (min_y max_y)')
    parser.add_argument('--z-bounds', nargs=2, type=float, help='Custom z-axis bounds (min_z max_z)')
    parser.add_argument('-m', '--margin', type=float, default=0.0, help='Margin padding')
    parser.add_argument('-o', '--output', default='ffd_box.3df',
                        help='Output filename: .3df or .xyz')
    parser.add_argument('--export-xyz', action='store_true',
                        help='Also export control points in .xyz format for DAFoam')
    parser.add_argument('--plot', action='store_true', help='Visualize FFD lattice')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--force-ascii', action='store_true', help='Force ASCII reading for Fluent mesh')
    parser.add_argument('--force-binary', action='store_true', help='Force binary reading for Fluent mesh')
    parser.add_argument('--save-plot', type=str, default=None, help='Save visualization to specified file path')
    parser.add_argument('--show-mesh', action='store_true', help='Show mesh points in visualization')
    parser.add_argument('--mesh-only', action='store_true', help='Show only the mesh without FFD box')
    parser.add_argument('--mesh-size', type=float, default=2.0, help='Size of mesh points in visualization')
    parser.add_argument('--mesh-alpha', type=float, default=0.3, help='Transparency of mesh points (0-1)')
    parser.add_argument('--mesh-color', default='blue', help='Color of mesh points')
    parser.add_argument('--ffd-point-size', type=float, default=10.0, help='Size of FFD control points in PyVista visualization')
    parser.add_argument('--ffd-alpha', type=float, default=1.0, help='Transparency of FFD box (0.0-1.0)')
    parser.add_argument('--ffd-color', type=str, default='b', help='Color of FFD control points')
    parser.add_argument('--show-full-ffd-grid', action='store_true', help='Show the complete internal FFD grid lines')
    parser.add_argument('--ffd-line-width', type=float, default=3.0, help='Width of FFD grid lines')
    parser.add_argument('--show-surface', action='store_true', default=True, help='Show mesh as a solid surface (default: True)')
    parser.add_argument('--show-mesh-edges', action='store_true', help='Show mesh edges')
    parser.add_argument('--show-cell-ids', action='store_true', help='Show cell IDs')
    parser.add_argument('--no-surface', action='store_true', help='Do not show mesh as a solid surface')
    parser.add_argument('--surface-alpha', type=float, default=1.0, help='Transparency of surface (0.0-1.0)')
    parser.add_argument('--surface-color', type=str, default='gray', help='Color of surface')
    parser.add_argument('--hide-control-points', action='store_true', help='Hide FFD control points')
    parser.add_argument('--control-point-size', type=float, default=30.0, help='Size of control points')
    parser.add_argument('--lattice-width', type=float, default=1.5, help='Width of lattice lines')
    parser.add_argument('--view-angle', type=float, nargs=2, help='View angle as elevation azimuth')
    parser.add_argument('--view-axis', type=str, choices=['x', 'y', 'z', '-x', '-y', '-z'], help='Align view with specified axis')
    parser.add_argument('--no-auto-scale', action='store_true', help='Disable automatic scaling of small geometries')
    parser.add_argument('--scale-factor', type=float, help='Manual scale factor for visualization')
    parser.add_argument('--zoom-region', type=float, nargs=6, metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX', 'Z_MIN', 'Z_MAX'),
                        help='Zoom to specified region (x_min x_max y_min y_max z_min z_max)')
    parser.add_argument('--zoom-factor', type=float, default=1.0, help='Zoom factor (>1 zooms in, <1 zooms out)')
    parser.add_argument('--show-original-mesh', action='store_true', help='Show original mesh with patches in a separate window')
    parser.add_argument('--point-size', type=float, default=2.0, help='Size of points when showing original mesh')
    parser.add_argument('--detail-level', type=str, default='medium', choices=['low', 'medium', 'high'], help='Detail level for mesh visualization (low, medium, high)')
    parser.add_argument('--max-triangles', type=int, default=10000, help='Maximum number of triangles to generate for visualization')
    parser.add_argument('--max-points', type=int, default=5000, help='Maximum number of points to use for visualization per patch')
    
    return parser.parse_args()

def main() -> None:
    """Main function to run the FFD generator."""
    # Parse command line arguments
    args = parse_arguments()

    # Set up logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing mesh file: {args.mesh_file}")
    
    # Validate input file exists
    if not os.path.exists(args.mesh_file):
        logger.error(f"Mesh file not found: {args.mesh_file}")
        sys.exit(1)
    
    try:
        # Determine mesh type and read accordingly
        if is_fluent_mesh(args.mesh_file):
            # Read Fluent mesh
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
                sys.exit(1)
            
            if args.patch:
                try:
                    pts = mesh.get_zone_points(args.patch)
                    logger.info(f"Using points from zone '{args.patch}' ({len(pts)} points)")
                except ValueError as e:
                    logger.error(f"Error extracting zone points: {e}")
                    available_zones = mesh.get_available_zones()
                    logger.info(f"Available zones: {', '.join(available_zones)}")
                    sys.exit(1)
            else:
                pts = mesh.points
                logger.info(f"Using all {len(mesh.points)} mesh points to build FFD box")
        else:
            # Use general mesh reader for other formats
            try:
                mesh = read_general_mesh(args.mesh_file)
                if args.patch:
                    try:
                        pts = extract_patch_points(mesh, args.patch)
                        logger.info(f"Using {pts.shape[0]} points from patch/group '{args.patch}'")
                    except ValueError as e:
                        logger.error(f"Error extracting patch points: {e}")
                        sys.exit(1)
                else:
                    pts = mesh.points
                    logger.info(f"Using all {pts.shape[0]} mesh points to build FFD box")
            except ImportError as e:
                logger.error(f"Missing dependency: {e}")
                logger.info("Install required dependencies with: pip install meshio")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Error reading mesh file: {e}")
                if args.debug:
                    logger.debug(traceback.format_exc())
                sys.exit(1)

        # Validate we have points to work with
        if len(pts) == 0:
            logger.error("No points found in the mesh for FFD box creation")
            sys.exit(1)

        # Create FFD box
        try:
            # Prepare custom dimensions if specified
            custom_dims = [None, None, None]
            
            # Handle x bounds (support both bounds or individual min/max)
            if args.x_bounds:
                custom_dims[0] = (args.x_bounds[0], args.x_bounds[1])
                logger.info(f'Using custom X bounds: {custom_dims[0]}')
            else:
                # Check for individual min/max settings
                x_min = args.x_min if args.x_min is not None else None
                x_max = args.x_max if args.x_max is not None else None
                
                if x_min is not None or x_max is not None:
                    custom_dims[0] = (x_min, x_max)
                    logger.info(f'Using custom X bounds: min={x_min}, max={x_max}')
            
            # Handle y bounds (support both bounds or individual min/max)
            if args.y_bounds:
                custom_dims[1] = (args.y_bounds[0], args.y_bounds[1])
                logger.info(f'Using custom Y bounds: {custom_dims[1]}')
            else:
                # Check for individual min/max settings
                y_min = args.y_min if args.y_min is not None else None
                y_max = args.y_max if args.y_max is not None else None
                
                if y_min is not None or y_max is not None:
                    custom_dims[1] = (y_min, y_max)
                    logger.info(f'Using custom Y bounds: min={y_min}, max={y_max}')
            
            # Handle z bounds (support both bounds or individual min/max)
            if args.z_bounds:
                custom_dims[2] = (args.z_bounds[0], args.z_bounds[1])
                logger.info(f'Using custom Z bounds: {custom_dims[2]}')
            else:
                # Check for individual min/max settings
                z_min = args.z_min if args.z_min is not None else None
                z_max = args.z_max if args.z_max is not None else None
                
                if z_min is not None or z_max is not None:
                    custom_dims[2] = (z_min, z_max)
                    logger.info(f'Using custom Z bounds: min={z_min}, max={z_max}')
                
            # Check if any custom dimensions were specified
            custom_dims = custom_dims if any(dim is not None for dim in custom_dims) else None
            
            cps, bbox = create_ffd_box(pts, tuple(args.dims), args.margin, custom_dims)
            logger.info(f'Bounding box min: {bbox[0]}, max: {bbox[1]}')
            logger.info(f'Control box dimensions: {args.dims} ({cps.shape[0]} control points)')
        except Exception as e:
            logger.error(f"Error creating FFD box: {e}")
            if args.debug:
                logger.debug(traceback.format_exc())
            sys.exit(1)
        
        # Output FFD control points to file
        try:
            # First export in the requested primary format (from output filename extension)
            if args.output.endswith('.xyz'):
                from ffd_utils.io import write_ffd_xyz
                write_ffd_xyz(cps, args.output)
                logger.info(f"FFD control points written to {args.output} in XYZ format")
                
                # If also requested to export in 3DF format
                if args.export_xyz:
                    # Already in XYZ format, so nothing to do
                    pass
            else:
                # Default to 3DF format
                from ffd_utils.io import write_ffd_3df
                write_ffd_3df(cps, args.output)
                logger.info(f"FFD control points written to {args.output} in 3DF format")
                
                # If also requested to export in XYZ format
                if args.export_xyz:
                    from ffd_utils.io import write_ffd_xyz
                    # Generate XYZ filename from the 3DF filename by replacing extension
                    xyz_filename = os.path.splitext(args.output)[0] + '.xyz'
                    write_ffd_xyz(cps, xyz_filename)
                    logger.info(f"FFD control points also written to {xyz_filename} in XYZ format")
        except Exception as e:
            logger.error(f"Error writing output file: {e}")
            if args.debug:
                logger.debug(traceback.format_exc())
            sys.exit(1)
        
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
                    
                    # Use PyVista-based visualization for much faster rendering and better quality
                    from ffd_utils.visualization import visualize_mesh_with_patches_pyvista
                    
                    # Direct call to PyVista visualization with optimized performance settings
                    # Pass FFD control points to visualize them together with the mesh
                    visualize_mesh_with_patches_pyvista(mesh, 
                                      save_path=save_path,
                                      title=f"Mesh with FFD Control Box: {os.path.basename(args.mesh_file)}",
                                      point_size=args.point_size * 1.5,  # Larger points for better visibility
                                      auto_scale=not args.no_auto_scale,
                                      scale_factor=args.scale_factor,
                                      show_solid=not args.no_surface,
                                      opacity=args.surface_alpha,
                                      show_edges=args.show_mesh_edges,
                                      color_by_zone=True,  # Color by zone for better visualization
                                      bgcolor='white',  # White background
                                      max_points_per_zone=1000000 if args.detail_level == 'high' else 10000,  # Much higher point count for better surface quality
                                      skip_internal_zones=True,  # Focus only on boundary zones for quality
                                      use_original_faces=True,  # Try to use original face connectivity for better edges
                                      ffd_control_points=cps,  # Add the FFD control points
                                      ffd_color=args.ffd_color,  # Show FFD in selected color
                                      ffd_opacity=args.ffd_alpha,  # Use specified transparency
                                      ffd_point_size=args.ffd_point_size,  # Use customized size for FFD control points
                                      show_ffd_mesh=True,  # Show the wireframe grid
                                      zoom_region=args.zoom_region,  # Custom region to zoom into if specified
                                      zoom_factor=args.zoom_factor,  # Zoom factor for the view
                                      ffd_box_dims=args.dims,  # Pass the FFD box dimensions explicitly
                                      view_axis=args.view_axis)  # Align view with specified axis if requested
                else:
                    # For other mesh types, we need to use the original mesh object
                    logger.info("Showing original mesh with patches")
                    save_path = os.path.splitext(args.save_plot)[0] + '_original.png' if args.save_plot else None
                    visualize_mesh_with_patches(mesh, 
                                             save_path=save_path,
                                             title=f"Original Mesh: {os.path.basename(args.mesh_file)}",
                                             point_size=args.point_size,
                                             auto_scale=not args.no_auto_scale,
                                             scale_factor=args.scale_factor,
                                             view_angle=tuple(args.view_angle) if args.view_angle else None)
            except Exception as e:
                logger.error(f"Error visualizing original mesh: {e}")
                if args.debug:
                    logger.debug(traceback.format_exc())
        
        # Visualize FFD box if requested
        if args.plot or args.save_plot:
            try:
                # If mesh-only is specified, show only the mesh
                if args.mesh_only:
                    # Create a simple figure with just the mesh points
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D
                    
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # Limit number of mesh points to avoid overloading the plot
                    max_points = 2000
                    if len(pts) > max_points:
                        logger.info(f"Subsetting mesh points from {len(pts)} to {max_points} for visualization")
                        indices = np.linspace(0, len(pts)-1, max_points, dtype=int)
                        mesh_subset = pts[indices]
                    else:
                        mesh_subset = pts
                    
                    ax.scatter(mesh_subset[:,0], mesh_subset[:,1], mesh_subset[:,2], 
                              c=args.mesh_color, marker='.', alpha=args.mesh_alpha, s=args.mesh_size)
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title('Mesh Points')
                    
                    if args.save_plot:
                        plt.savefig(args.save_plot, dpi=300, bbox_inches='tight')
                        logger.info(f"Figure saved to {args.save_plot}")
                    
                    plt.show()
                else:
                    # Show the mesh with the FFD box
                    visualize_ffd(cps, bbox, 
                                 mesh_points=pts,  # Always pass mesh points
                                 show_mesh=args.show_mesh,
                                 save_path=args.save_plot,
                                 mesh_alpha=args.mesh_alpha,
                                 mesh_point_size=args.mesh_size,
                                 mesh_color=args.mesh_color,
                                 ffd_alpha=args.ffd_alpha,
                                 ffd_color=args.ffd_color,
                                 show_surface=not args.no_surface,  # Show surface by default
                                 surface_alpha=args.surface_alpha,
                                 surface_color=args.surface_color,
                                 show_control_points=not args.hide_control_points,
                                 control_point_size=args.control_point_size,
                                 lattice_width=args.lattice_width,
                                 view_angle=tuple(args.view_angle) if args.view_angle else None,
                                 auto_scale=not args.no_auto_scale,
                                 scale_factor=args.scale_factor)
            except Exception as e:
                logger.error(f"Error during visualization: {e}")
                if args.debug:
                    logger.debug(traceback.format_exc())
                # Don't exit - this is not a critical error
                
        logger.info("FFD generation completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()