"""
Command-line interface for Hierarchical FFD operations.

This module provides CLI components for creating and manipulating
hierarchical FFD control boxes with multiple resolution levels of influence.
"""

import argparse
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

from openffd.core.hierarchical import HierarchicalFFD, create_hierarchical_ffd
from openffd.utils.parallel import ParallelConfig
from openffd.mesh.general import read_general_mesh
from openffd.visualization.hierarchical_viz import (
    visualize_hierarchical_ffd_matplotlib,
    visualize_hierarchical_ffd_pyvista,
    visualize_influence_distribution
)


def add_hierarchical_ffd_args(parser: argparse.ArgumentParser) -> None:
    """Add hierarchical FFD command-line arguments to an argument parser.
    
    Args:
        parser: ArgumentParser object
    """
    hffd_group = parser.add_argument_group("Hierarchical FFD Options")
    
    hffd_group.add_argument(
        "--hierarchical",
        action="store_true",
        help="Use hierarchical FFD instead of standard FFD"
    )
    
    hffd_group.add_argument(
        "--base-dims",
        type=int,
        nargs=3,
        default=None,  # No default - use dims if not specified
        metavar=("NX", "NY", "NZ"),
        help="Dimensions of the base (root) control lattice (uses --dims if not specified)"
    )
    
    hffd_group.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth of the hierarchy (default: 3)"
    )
    
    hffd_group.add_argument(
        "--subdivision-factor",
        type=int,
        default=2,
        help="Factor for subdividing control lattices (default: 2)"
    )
    
    hffd_group.add_argument(
        "--show-levels",
        type=int,
        nargs="+",
        help="List of level IDs to show in visualization (default: all)"
    )
    
    hffd_group.add_argument(
        "--show-influence",
        action="store_true",
        help="Show influence regions in visualization"
    )
    
    hffd_group.add_argument(
        "--export-hierarchical",
        type=str,
        help="Export hierarchical FFD control points to a directory"
    )


def process_hierarchical_ffd_command(args: argparse.Namespace) -> Optional[HierarchicalFFD]:
    """Process hierarchical FFD command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        HierarchicalFFD object if hierarchical FFD is enabled, None otherwise
    """
    if not args.hierarchical:
        return None
    
    logger = logging.getLogger(__name__)
    logger.info("Using hierarchical FFD")
    
    # Check if mesh file exists
    if not os.path.exists(args.mesh_file):
        logger.error(f"Mesh file {args.mesh_file} does not exist")
        return None
    
    # Create parallel configuration
    parallel_config = ParallelConfig(
        enabled=args.parallel,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size
    )
    
    # Read mesh file
    logger.info(f"Reading mesh file {args.mesh_file}")
    mesh_data = read_general_mesh(args.mesh_file)
    
    # Extract mesh points
    if hasattr(mesh_data, "points") and mesh_data.points is not None:
        mesh_points = mesh_data.points
    else:
        logger.error("Could not extract points from mesh data")
        return None
    
    # Create hierarchical FFD
    logger.info(f"Creating hierarchical FFD with {args.max_depth} levels")
    start_time = time.time()
    
    hffd = create_hierarchical_ffd(
        mesh_points=mesh_points,
        base_dims=tuple(args.base_dims),
        max_depth=args.max_depth,
        subdivision_factor=args.subdivision_factor,
        parallel_config=parallel_config
    )
    
    end_time = time.time()
    logger.info(f"Created hierarchical FFD with {len(hffd.levels)} levels in {end_time - start_time:.2f} seconds")
    
    # Print level information
    for level_info in hffd.get_level_info():
        logger.info(
            f"Level {level_info['level_id']} (depth {level_info['depth']}): "
            f"{level_info['dims']} dims, {level_info['num_control_points']} control points, "
            f"weight: {level_info['weight_factor']:.2f}"
        )
    
    # Export hierarchical FFD if requested
    if args.export_hierarchical:
        export_hierarchical_ffd(hffd, args.export_hierarchical)
    
    # Visualize hierarchical FFD if requested
    if args.visualize:
        visualize_hierarchical_ffd(
            hffd=hffd,
            mesh_points=mesh_points,
            show_levels=args.show_levels,
            show_influence=args.show_influence,
            save_path=args.output if args.output else None,
            parallel_config=parallel_config
        )
    
    return hffd


def visualize_hierarchical_ffd(
    hffd: HierarchicalFFD,
    mesh_points: np.ndarray,
    show_levels: Optional[List[int]] = None,
    show_influence: bool = False,
    save_path: Optional[str] = None,
    parallel_config: Optional[ParallelConfig] = None
) -> None:
    """Visualize a hierarchical FFD.
    
    Args:
        hffd: Hierarchical FFD object
        mesh_points: Mesh points
        show_levels: List of level IDs to show (None for all)
        show_influence: Whether to show influence regions
        save_path: Optional path to save the figure
        parallel_config: Configuration for parallel processing
    """
    logger = logging.getLogger(__name__)
    
    # Try using PyVista for visualization
    try:
        import pyvista
        
        logger.info("Visualizing hierarchical FFD with PyVista")
        
        # Visualize hierarchical FFD
        visualize_hierarchical_ffd_pyvista(
            hffd=hffd,
            show_levels=show_levels,
            title="Hierarchical FFD Control Lattice",
            save_path=save_path,
            mesh_points=mesh_points,
            show_mesh=True,
            mesh_size=3.0,
            mesh_alpha=0.3,
            show_influence=show_influence,
            parallel_config=parallel_config
        )
        
        # Visualize influence distribution
        if show_influence:
            influence_save_path = None
            if save_path:
                influence_save_path = os.path.splitext(save_path)[0] + "_influence.png"
            
            visualize_influence_distribution(
                hffd=hffd,
                mesh_points=mesh_points,
                save_path=influence_save_path
            )
        
    except (ImportError, Exception) as e:
        logger.warning(f"Could not use PyVista for visualization: {e}")
        logger.info("Falling back to Matplotlib for visualization")
        
        # Fall back to Matplotlib
        visualize_hierarchical_ffd_matplotlib(
            hffd=hffd,
            show_levels=show_levels,
            title="Hierarchical FFD Control Lattice",
            save_path=save_path,
            mesh_points=mesh_points,
            show_mesh=True,
            mesh_size=1.0,
            mesh_alpha=0.3
        )


def export_hierarchical_ffd(hffd: HierarchicalFFD, output_dir: str) -> None:
    """Export hierarchical FFD control points to files.
    
    Args:
        hffd: Hierarchical FFD object
        output_dir: Directory to export control points to
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export level information
    level_info = hffd.get_level_info()
    with open(os.path.join(output_dir, "level_info.txt"), "w") as f:
        f.write("level_id,depth,dims,num_control_points,weight_factor\n")
        for info in level_info:
            f.write(
                f"{info['level_id']},{info['depth']},"
                f"{info['dims'][0]}x{info['dims'][1]}x{info['dims'][2]},"
                f"{info['num_control_points']},{info['weight_factor']:.4f}\n"
            )
    
    # Export control points for each level
    for level_id, level in hffd.levels.items():
        # Create output file name
        output_file = os.path.join(output_dir, f"level_{level_id}_control_points.txt")
        
        # Save control points
        np.savetxt(
            output_file,
            level.control_points,
            delimiter=",",
            header=f"x,y,z (Level {level_id}, Depth {level.depth}, Weight {level.weight_factor:.4f})"
        )
    
    logger.info(f"Exported hierarchical FFD control points to {output_dir}")


def deform_with_hierarchical_ffd(
    hffd: HierarchicalFFD,
    deformation_file: str,
    mesh_points: np.ndarray,
    output_file: Optional[str] = None,
    visualize: bool = False,
    save_path: Optional[str] = None
) -> np.ndarray:
    """Deform mesh points using a hierarchical FFD.
    
    Args:
        hffd: Hierarchical FFD object
        deformation_file: File containing deformation parameters
        mesh_points: Mesh points to deform
        output_file: Optional file to save deformed points to
        visualize: Whether to visualize the deformation
        save_path: Optional path to save the visualization
        
    Returns:
        Deformed mesh points
    """
    logger = logging.getLogger(__name__)
    
    # Parse deformation file
    deformed_control_points = parse_deformation_file(deformation_file, hffd)
    
    # Deform mesh points
    logger.info(f"Deforming mesh with hierarchical FFD")
    start_time = time.time()
    
    deformed_points = hffd.deform_mesh(deformed_control_points, mesh_points)
    
    end_time = time.time()
    logger.info(f"Deformed mesh in {end_time - start_time:.2f} seconds")
    
    # Save deformed points if requested
    if output_file:
        np.savetxt(output_file, deformed_points, delimiter=",", header="x,y,z")
        logger.info(f"Saved deformed points to {output_file}")
    
    # Visualize deformation if requested
    if visualize:
        try:
            from openffd.visualization.hierarchical_viz import visualize_hierarchical_deformation
            
            visualize_hierarchical_deformation(
                hffd=hffd,
                deformed_control_points=deformed_control_points,
                mesh_points=mesh_points,
                save_path=save_path,
                show_original=True
            )
        except Exception as e:
            logger.warning(f"Could not visualize deformation: {e}")
    
    return deformed_points


def parse_deformation_file(deformation_file: str, hffd: HierarchicalFFD) -> Dict[int, np.ndarray]:
    """Parse a deformation file to get deformed control points.
    
    The deformation file should have the following format:
    level_id,point_id,dx,dy,dz
    
    Args:
        deformation_file: Path to the deformation file
        hffd: Hierarchical FFD object
        
    Returns:
        Dictionary mapping level_id to deformed control points
    """
    logger = logging.getLogger(__name__)
    
    # Check if deformation file exists
    if not os.path.exists(deformation_file):
        logger.error(f"Deformation file {deformation_file} does not exist")
        return {}
    
    # Load deformation parameters
    try:
        deformation_data = np.loadtxt(
            deformation_file,
            delimiter=",",
            skiprows=1,  # Skip header
            dtype=np.float64
        )
    except Exception as e:
        logger.error(f"Could not load deformation file {deformation_file}: {e}")
        return {}
    
    # Create dictionary of deformed control points
    deformed_control_points = {}
    
    # Process each level
    for level_id, level in hffd.levels.items():
        # Copy original control points
        deformed_cp = level.control_points.copy()
        
        # Filter deformation data for this level
        level_data = deformation_data[deformation_data[:, 0] == level_id]
        
        # Apply deformations
        for row in level_data:
            level_id = int(row[0])
            point_id = int(row[1])
            dx, dy, dz = row[2], row[3], row[4]
            
            # Check if point_id is valid
            if point_id >= len(deformed_cp):
                logger.warning(f"Invalid point_id {point_id} for level {level_id}")
                continue
            
            # Apply deformation
            deformed_cp[point_id] += np.array([dx, dy, dz])
        
        # Add to dictionary
        if len(level_data) > 0:
            deformed_control_points[level_id] = deformed_cp
    
    return deformed_control_points
