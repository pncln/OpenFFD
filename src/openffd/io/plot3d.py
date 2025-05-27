"""
Plot3D (XYZ) Exporter for FFD Control Grids.

This module provides a comprehensive tool for exporting FFD control point 
datasets to the Plot3D (XYZ) format with proper structured grid connectivity.
It ensures proper visualization of both outer boundary and internal grid lines
for complete visualization in visualization tools like ParaView.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# PLOT3D format constants
PLOT3D_PRECISION = 6  # Decimal places for coordinate values


def parse_dimensions(dim_str: str) -> List[int]:
    """Parse dimensions from string (e.g., '50x2x2' or '50,2,2').
    
    Args:
        dim_str: String representation of dimensions
        
    Returns:
        List of dimensions [ni, nj, nk]
    """
    if 'x' in dim_str.lower():
        return [int(d) for d in dim_str.lower().split('x')]
    elif ',' in dim_str:
        return [int(d) for d in dim_str.split(',')]
    else:
        # Attempt to parse as space-separated
        return [int(d) for d in dim_str.split()]


def read_points_file(input_file: str) -> Tuple[dict, List[Tuple[float, float, float]]]:
    """Read control points from input file with robust error handling.
    
    Args:
        input_file: Path to input file with control points
        
    Returns:
        Tuple containing:
        - Dictionary with header information (num_points, dimensions)
        - List of point coordinates as tuples (x, y, z)
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    # Get header information
    header_info = {}
    
    # Parse number of points (first line)
    try:
        header_info['num_points'] = int(lines[0].strip())
    except (ValueError, IndexError):
        header_info['num_points'] = None
    
    # Parse dimensions (second line)
    try:
        dim_parts = lines[1].strip().split()
        if len(dim_parts) >= 3:
            header_info['dimensions'] = [int(dim_parts[0]), int(dim_parts[1]), int(dim_parts[2])]
        else:
            header_info['dimensions'] = None
    except (ValueError, IndexError):
        header_info['dimensions'] = None
    
    # Extract points (starting from line 3)
    points = []
    for i in range(2, len(lines)):
        line = lines[i].strip()
        if not line:  # Skip empty lines
            continue
            
        parts = line.split()
        if len(parts) >= 3:
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                points.append((x, y, z))
            except (ValueError, IndexError):
                pass  # Skip invalid point definitions
    
    return header_info, points


def create_structured_grid(
    points: List[Tuple[float, float, float]],
    dims: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map points to a structured grid with proper ordering for Plot3D format.
    
    Args:
        points: List of point coordinates as tuples (x, y, z)
        dims: Dimensions of the FFD grid [ni, nj, nk]
        
    Returns:
        Tuple of three numpy arrays (x_array, y_array, z_array) representing the structured grid
        
    Raises:
        ValueError: If not enough points for specified dimensions
    """
    ni, nj, nk = dims
    expected_points = ni * nj * nk
    
    # Verify point count matches dimensions
    if len(points) != expected_points:
        logger.warning(f"Expected {expected_points} points for dimensions {dims}, but found {len(points)}")
        if len(points) < expected_points:
            raise ValueError(f"Not enough points ({len(points)}) for specified dimensions {dims}")
    
    # Create grid arrays for each coordinate
    x_array = np.zeros((nk, nj, ni))
    y_array = np.zeros((nk, nj, ni))
    z_array = np.zeros((nk, nj, ni))
    
    # Map flattened points to 3D grid with proper ordering for Plot3D
    # The ordering is important for correct visualization
    index = 0
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                if index < len(points):
                    x, y, z = points[index]
                    x_array[k, j, i] = x
                    y_array[k, j, i] = y
                    z_array[k, j, i] = z
                    index += 1
    
    return x_array, y_array, z_array


def write_plot3d_file(
    output_file: str,
    x_array: np.ndarray,
    y_array: np.ndarray,
    z_array: np.ndarray
) -> None:
    """Write a standard Plot3D (XYZ) file with proper formatting.
    
    Args:
        output_file: Path to output Plot3D file
        x_array: X-coordinates array
        y_array: Y-coordinates array
        z_array: Z-coordinates array
    """
    # Get dimensions from the arrays
    nk, nj, ni = x_array.shape
    
    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    # Write Plot3D multiblock file
    with open(output_file, 'w') as f:
        # Write number of grids (1 in our case)
        f.write("1\n")
        
        # Write dimensions of the grid
        f.write(f"{ni} {nj} {nk}\n")
        
        # Write X coordinates
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    f.write(f"{x_array[k, j, i]:.{PLOT3D_PRECISION}f} ")
                f.write("\n")
        
        # Write Y coordinates
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    f.write(f"{y_array[k, j, i]:.{PLOT3D_PRECISION}f} ")
                f.write("\n")
        
        # Write Z coordinates
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    f.write(f"{z_array[k, j, i]:.{PLOT3D_PRECISION}f} ")
                f.write("\n")
    
    logger.info(f"Successfully wrote Plot3D file: {output_file}")


def verify_plot3d_format(filepath: str) -> bool:
    """Verify the Plot3D file has the correct format.
    
    Args:
        filepath: Path to Plot3D file
        
    Returns:
        bool: True if file has correct format, False otherwise
    """
    try:
        with open(filepath, 'r') as f:
            # Read first line (number of grids)
            num_grids = int(f.readline().strip())
            if num_grids != 1:
                logger.warning(f"Expected 1 grid, found {num_grids}")
                return False
            
            # Read dimensions
            dims_line = f.readline().strip().split()
            if len(dims_line) != 3:
                logger.warning("Invalid dimensions line")
                return False
                
            ni, nj, nk = map(int, dims_line)
            expected_points = ni * nj * nk
            
            # Check if file has correct number of lines
            # Each coordinate set should have nk*nj lines
            line_count = sum(1 for _ in f)
            expected_lines = 3 * nk * nj  # X, Y, Z coordinate blocks
            
            if line_count != expected_lines:
                logger.warning(f"Expected {expected_lines} lines for coordinates, found {line_count}")
                return False
                
            logger.info(f"Plot3D file verified: {ni}×{nj}×{nk} grid ({expected_points} points)")
            return True
            
    except Exception as e:
        logger.error(f"Error verifying Plot3D file: {e}")
        return False


def export_plot3d(
    input_file: str,
    output_file: str,
    dims: Optional[List[int]] = None
) -> bool:
    """Export FFD control points to Plot3D (XYZ) format with complete grid structure.
    
    This function ensures proper visualization of the entire FFD grid structure,
    including internal grid lines and connections in all dimensions (x, y, z).
    
    Args:
        input_file: Path to input file with control points
        output_file: Path to output Plot3D file
        dims: Dimensions of the FFD grid [ni, nj, nk]
        
    Returns:
        bool: True if export was successful, False otherwise
    """
    try:
        # Validate input file exists
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return False
            
        # Read control points from input file
        logger.info(f"Reading control points from: {input_file}")
        header_info, points = read_points_file(input_file)
        
        if not points:
            logger.error("No valid control points found in input file")
            return False
            
        logger.info(f"Read {len(points)} control points")
        
        # Determine dimensions if not provided
        if dims is None:
            if header_info['dimensions'] is not None:
                dims = header_info['dimensions']
                logger.info(f"Using dimensions from file header: {dims}")
            elif header_info['num_points'] is not None:
                # Try to infer dimensions based on number of points
                total_points = header_info['num_points']
                
                # For 3D grids, we need to factorize the total number of points
                # into three factors ni, nj, nk
                # For simplicity, we'll use some common patterns
                if total_points == 200:  # 50×2×2
                    dims = [50, 2, 2]
                elif total_points == 60:  # 5×4×3
                    dims = [5, 4, 3]
                elif total_points == 64:  # 4×4×4
                    dims = [4, 4, 4]
                elif total_points == 125:  # 5×5×5
                    dims = [5, 5, 5]
                elif total_points == 27:  # 3×3×3
                    dims = [3, 3, 3]
                else:
                    # Default assumption for unknown point counts
                    logger.warning(f"Could not determine dimensions for {total_points} points")
                    logger.warning("Using cubic root approximation")
                    
                    # Try to approximate with a cube
                    cube_dim = round(total_points ** (1/3))
                    if cube_dim ** 3 == total_points:
                        dims = [cube_dim, cube_dim, cube_dim]
                    else:
                        logger.error("Could not determine appropriate dimensions")
                        return False
                        
                logger.info(f"Inferred dimensions: {dims}")
            else:
                logger.error("No dimensions provided and could not determine from file")
                return False
        
        # Create structured grid for Plot3D format
        logger.info(f"Creating structured grid with dimensions: {dims}")
        try:
            x_array, y_array, z_array = create_structured_grid(points, dims)
        except ValueError as e:
            logger.error(f"Error creating structured grid: {e}")
            return False
        
        # Write Plot3D file
        logger.info(f"Writing Plot3D file: {output_file}")
        write_plot3d_file(output_file, x_array, y_array, z_array)
        
        # Verify the output file
        if verify_plot3d_format(output_file):
            logger.info(f"Successfully exported to Plot3D format: {output_file}")
            return True
        else:
            logger.warning(f"Plot3D file may have incorrect format: {output_file}")
            return False
            
    except Exception as e:
        logger.error(f"Error exporting to Plot3D format: {e}")
        return False


def main(args=None):
    """Command-line interface for Plot3D export.
    
    Args:
        args: Command-line arguments (for testing)
        
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export FFD control points to Plot3D (XYZ) format"
    )
    parser.add_argument("input_file", help="Input file with control points")
    parser.add_argument("output_file", help="Output Plot3D file")
    parser.add_argument(
        "-d", "--dimensions", 
        help="Dimensions of the FFD grid (e.g., '50x2x2' or '50,2,2')"
    )
    
    args = parser.parse_args(args)
    
    # Parse dimensions if provided
    dims = None
    if args.dimensions:
        try:
            dims = parse_dimensions(args.dimensions)
            if len(dims) != 3:
                logger.error("Dimensions must have exactly 3 values (ni, nj, nk)")
                return 1
        except ValueError:
            logger.error(f"Invalid dimensions format: {args.dimensions}")
            return 1
    
    # Export to Plot3D format
    success = export_plot3d(args.input_file, args.output_file, dims)
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
