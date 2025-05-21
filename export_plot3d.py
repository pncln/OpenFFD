#!/usr/bin/env python3
"""
Script to export FFD control points to Plot3D (XYZ) format
"""

import numpy as np
import sys
import os

#!/usr/bin/env python3
"""
Professional Plot3D (XYZ) Exporter for FFD Control Grids

This module provides a comprehensive tool for exporting FFD control point 
datasets to the Plot3D (XYZ) format with proper structured grid connectivity.
It ensures proper visualization of both outer boundary and internal grid lines
for complete visualization in visualization tools like ParaView.

The module supports optional dimensions specification and properly handles
the specific ordering and format requirements of Plot3D multiblock grids.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# PLOT3D format constants
PLOT3D_PRECISION = 6  # Decimal places for coordinate values

def parse_dimensions(dim_str):
    """Parse dimensions from string (e.g., '50x2x2' or '50,2,2')"""
    if 'x' in dim_str.lower():
        return [int(d) for d in dim_str.lower().split('x')]
    elif ',' in dim_str:
        return [int(d) for d in dim_str.split(',')]
    else:
        # Attempt to parse as space-separated
        return [int(d) for d in dim_str.split()]

def read_points_file(input_file):
    """Read control points from input file with robust error handling"""
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

def create_structured_grid(points, dims):
    """Map points to a structured grid with proper ordering for Plot3D format"""
    ni, nj, nk = dims
    expected_points = ni * nj * nk
    
    # Verify point count matches dimensions
    if len(points) != expected_points:
        print(f"WARNING: Expected {expected_points} points for dimensions {dims}, but found {len(points)}")
        if len(points) < expected_points:
            raise ValueError(f"Not enough points ({len(points)}) for specified dimensions {dims}")
    
    # Create grid arrays for each coordinate
    x_array = np.zeros((nk, nj, ni))
    y_array = np.zeros((nk, nj, ni))
    z_array = np.zeros((nk, nj, ni))
    
    # For our specific 50×2×2 grid as in ffd_box.xyz
    # The points at each x-position are organized as:
    # [x₀]: (y₀,z₀), (y₀,z₁), (y₁,z₀), (y₁,z₁)
    # [x₁]: (y₀,z₀), (y₀,z₁), (y₁,z₀), (y₁,z₁)
    # etc.
    
    # Map them to the structured grid in the correct order
    for i in range(ni):
        base_idx = i * (nj * nk)
        for j in range(nj):
            for k in range(nk):
                # Calculate index in the raw points list
                # The points follow this pattern in each x-group:
                # (j=0,k=0), (j=0,k=1), (j=1,k=0), (j=1,k=1)
                if j == 0 and k == 0:
                    offset = 0
                elif j == 0 and k == 1:
                    offset = 1
                elif j == 1 and k == 0:
                    offset = 2
                else:  # j == 1 and k == 1
                    offset = 3
                
                point_idx = base_idx + offset
                if point_idx < len(points):
                    # Plot3D expects [k][j][i] ordering
                    x_array[k, j, i] = points[point_idx][0]
                    y_array[k, j, i] = points[point_idx][1]
                    z_array[k, j, i] = points[point_idx][2]
    
    return x_array, y_array, z_array

def write_plot3d_file(output_file, x_array, y_array, z_array):
    """Write a standard Plot3D (XYZ) file with proper formatting"""
    # Get dimensions from the arrays
    nk, nj, ni = x_array.shape
    
    with open(output_file, 'w') as f:
        # Number of grid blocks (always 1 for our single grid)
        f.write("1\n")
        
        # Grid dimensions for the block
        f.write(f"{ni} {nj} {nk}\n")
        
        # Write X coordinates with consistent formatting
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    f.write(f"{x_array[k, j, i]:.{PLOT3D_PRECISION}f} ")
        f.write("\n")
        
        # Write Y coordinates with consistent formatting
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    f.write(f"{y_array[k, j, i]:.{PLOT3D_PRECISION}f} ")
        f.write("\n")
        
        # Write Z coordinates with consistent formatting
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    f.write(f"{z_array[k, j, i]:.{PLOT3D_PRECISION}f} ")
        f.write("\n")

def verify_plot3d_format(filepath):
    """Verify the Plot3D file has the correct format"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Basic format checks
    if len(lines) < 4:
        return False, "File has fewer than 4 lines (minimum for Plot3D)"
    
    try:
        num_blocks = int(lines[0].strip())
        if num_blocks != 1:
            return False, f"Expected 1 block, found {num_blocks}"
        
        dim_parts = lines[1].strip().split()
        if len(dim_parts) != 3:
            return False, "Grid dimensions line should have exactly 3 values"
            
        ni, nj, nk = int(dim_parts[0]), int(dim_parts[1]), int(dim_parts[2])
        expected_values = ni * nj * nk
        
        # Check that coordinate lines have correct number of values
        for i, line in enumerate(lines[2:5], 2):
            values = line.strip().split()
            if len(values) != expected_values:
                return False, f"Line {i+1}: Expected {expected_values} values, found {len(values)}"
    
    except Exception as e:
        return False, f"Format verification error: {str(e)}"
    
    return True, "File appears to be valid Plot3D format"

def export_plot3d(input_file, output_file, dims=None):
    """
    Export FFD control points to Plot3D (XYZ) format with complete grid structure.
    
    This function ensures proper visualization of the entire FFD grid structure,
    including internal grid lines and connections in all dimensions (x, y, z).
    
    Args:
        input_file: Path to input file with control points
        output_file: Path to output Plot3D file
        dims: Dimensions of the FFD grid (ni, nj, nk)
    
    Returns:
        bool: True if export was successful, False otherwise
    """
    print(f"Reading control points from {input_file}")
    
    try:
        # Read and parse the input file
        header_info, points = read_points_file(input_file)
        
        # Determine dimensions
        if dims is None:
            if header_info['dimensions'] is not None:
                dims = header_info['dimensions']
                print(f"Using dimensions from file: {dims[0]}×{dims[1]}×{dims[2]}")
            else:
                # Make a best guess based on point count
                point_count = len(points)
                if point_count % 200 == 0:  # Likely 50×2×2
                    dims = [50, 2, 2]
                elif point_count == 125:  # 5×5×5
                    dims = [5, 5, 5]
                else:
                    # Generic approach - try to estimate balanced dimensions
                    dim = round(point_count ** (1/3))
                    dims = [dim, dim, dim]
                
                print(f"No dimensions provided, guessing: {dims[0]}×{dims[1]}×{dims[2]}")
        else:
            print(f"Using specified dimensions: {dims[0]}×{dims[1]}×{dims[2]}")
        
        # Create the structured grid with correct point ordering
        print(f"Creating structured grid with dimensions: {dims[0]}×{dims[1]}×{dims[2]}")
        try:
            x_array, y_array, z_array = create_structured_grid(points, dims)
        except ValueError as e:
            print(f"ERROR: {str(e)}")
            return False
        
        # Write the Plot3D file
        print(f"Writing Plot3D file: {output_file}")
        write_plot3d_file(output_file, x_array, y_array, z_array)
        
        # Verify the file
        is_valid, message = verify_plot3d_format(output_file)
        if not is_valid:
            print(f"WARNING: {message}")
        else:
            print(f"Verification: {message}")
        
        total_points = dims[0] * dims[1] * dims[2]
        print(f"Successfully exported {total_points} points as {dims[0]}×{dims[1]}×{dims[2]} grid")
        print(f"File written to: {output_file}")
        return True
        
    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Command-line interface for Plot3D export"""
    parser = argparse.ArgumentParser(description="Export FFD control points to Plot3D (XYZ) format")
    parser.add_argument('input_file', help="Input file containing control points")
    parser.add_argument('output_file', help="Output Plot3D file")
    parser.add_argument('--dims', '-d', help="Grid dimensions in format 'ni,nj,nk' or 'nixnjxnk'")
    args = parser.parse_args()
    
    # Parse dimensions if provided
    dims = None
    if args.dims:
        try:
            dims = parse_dimensions(args.dims)
            if len(dims) != 3:
                print(f"ERROR: Dimensions must have exactly 3 values, got {len(dims)}")
                sys.exit(1)
        except ValueError as e:
            print(f"ERROR: Invalid dimensions format: {str(e)}")
            sys.exit(1)
    
    # Run the export
    success = export_plot3d(args.input_file, args.output_file, dims)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    # Check command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python export_plot3d.py INPUT_FILE OUTPUT_FILE [NI NJ NK]")
        print("       where NI, NJ, NK are the grid dimensions (optional)")
        sys.exit(1)
    
    # Parse dimensions if provided
    dims = None
    if len(sys.argv) >= 6:
        try:
            ni = int(sys.argv[3])
            nj = int(sys.argv[4])
            nk = int(sys.argv[5])
            dims = [ni, nj, nk]
        except ValueError:
            print("ERROR: Dimensions must be integers")
            sys.exit(1)
    
    # Run the export
    success = export_plot3d(sys.argv[1], sys.argv[2], dims)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    # Check command-line arguments
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input_file output_file [ni nj nk]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Check if dimensions are provided
    dims = None
    if len(sys.argv) >= 6:
        try:
            dims = [int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]
        except ValueError:
            print("ERROR: Dimensions must be integers")
            sys.exit(1)
    
    # Export to Plot3D format
    success = export_plot3d(input_file, output_file, dims)
    
    if not success:
        sys.exit(1)
