"""
I/O utilities for FFD control box data.

This module provides functions for reading and writing FFD control points
in various file formats, including .3df (ID, x, y, z) and .xyz (DAFoam format).
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


def write_ffd_3df(control_points: np.ndarray, filename: str = 'ffd_box.3df') -> None:
    """Write FFD control points to a .3df file format.
    
    This format uses the following structure:
    # FFD control points (id, x, y, z)
    0 x0 y0 z0
    1 x1 y1 z1
    ...
    
    Args:
        control_points: Numpy array of control point coordinates with shape (n, 3)
        filename: Output filename (will be created or overwritten)
        
    Raises:
        TypeError: If control_points is not a numpy array
        ValueError: If control_points is empty or has wrong shape
        IOError: If the file cannot be written
    """
    # Validate inputs
    if not isinstance(control_points, np.ndarray):
        raise TypeError("control_points must be a numpy array")
        
    if control_points.size == 0:
        raise ValueError("No control points to write")
        
    if len(control_points.shape) != 2 or control_points.shape[1] != 3:
        raise ValueError(f"control_points must have shape (n, 3), got {control_points.shape}")
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(filename))
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            raise IOError(f"Failed to create output directory: {e}")
    
    logger.info(f"Writing FFD control points to .3df file: {filename}")
    
    try:
        with open(filename, 'w') as f:
            f.write('# FFD control points (id, x, y, z)\n')
            for idx, (x, y, z) in enumerate(control_points):
                f.write(f'{idx} {x:.6f} {y:.6f} {z:.6f}\n')
        
        logger.info(f"Successfully wrote {control_points.shape[0]} control points to {filename}")
    except PermissionError:
        logger.error(f"Permission denied when writing to {filename}")
        raise IOError(f"Permission denied when writing to {filename}")
    except Exception as e:
        logger.error(f"Error writing .3df file: {e}")
        raise


def write_ffd_xyz(
    control_points: np.ndarray,
    filename: str = 'ffd_box.xyz',
    dimensions: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Write an FFD lattice in ASCII multiblock PLOT3D format.

    This is the ``.xyz`` format consumed by DAFoam/pyGeo and ParaView's
    PLOT3D reader. Coordinates are written with the i index changing fastest.
    
    Args:
        control_points: Numpy array of control point coordinates with shape (n, 3)
        filename: Output filename (will be created or overwritten)
        dimensions: Structured lattice dimensions ``(ni, nj, nk)``
        
    Raises:
        TypeError: If control_points is not a numpy array
        ValueError: If control_points is empty or has wrong shape
        IOError: If the file cannot be written
    """
    # Validate inputs
    if not isinstance(control_points, np.ndarray):
        raise TypeError("control_points must be a numpy array")
        
    if control_points.size == 0:
        raise ValueError("No control points to write")
        
    if len(control_points.shape) != 2 or control_points.shape[1] != 3:
        raise ValueError(f"control_points must have shape (n, 3), got {control_points.shape}")

    if dimensions is None:
        cube_dim = round(control_points.shape[0] ** (1.0 / 3.0))
        if cube_dim ** 3 != control_points.shape[0]:
            raise ValueError(
                "dimensions are required for a non-cubic PLOT3D FFD lattice"
            )
        dimensions = (cube_dim, cube_dim, cube_dim)

    if len(dimensions) != 3 or any(dim < 1 for dim in dimensions):
        raise ValueError(f"dimensions must contain three positive values, got {dimensions}")

    expected_points = int(np.prod(dimensions))
    if expected_points != control_points.shape[0]:
        raise ValueError(
            f"dimensions {dimensions} require {expected_points} control points, "
            f"got {control_points.shape[0]}"
        )
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(filename))
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            raise IOError(f"Failed to create output directory: {e}")
    
    logger.info(f"Writing FFD control points to PLOT3D file: {filename}")
    
    try:
        from openffd.io.plot3d import create_structured_grid, write_plot3d_file

        arrays = create_structured_grid(control_points.tolist(), list(dimensions))
        write_plot3d_file(filename, *arrays)
        logger.info(
            f"Successfully wrote {control_points.shape[0]} control points "
            f"with dimensions {dimensions} to {filename}"
        )
    except PermissionError:
        logger.error(f"Permission denied when writing to {filename}")
        raise IOError(f"Permission denied when writing to {filename}")
    except Exception as e:
        logger.error(f"Error writing .xyz file: {e}")
        raise


def read_ffd_3df(filename: str) -> np.ndarray:
    """Read FFD control points from a .3df file.
    
    Args:
        filename: Path to .3df file
    
    Returns:
        np.ndarray: Control point coordinates with shape (n, 3)
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If no valid control points are found in the file
        IOError: If there's an error reading the file
    """
    # Validate input
    if not filename or not isinstance(filename, str):
        raise ValueError(f"filename must be a non-empty string, got {filename}")
        
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
        
    logger.info(f"Reading FFD control points from .3df file: {filename}")
    
    points = []
    line_count = 0
    invalid_lines = 0
    
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                line_count += 1
                
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue
                    
                parts = line.split()
                if len(parts) >= 4:  # id, x, y, z
                    try:
                        # Try to parse as floats
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        points.append([x, y, z])
                    except (ValueError, IndexError) as e:
                        invalid_lines += 1
                        logger.warning(f"Invalid line {line_num} in .3df file: '{line}', error: {e}")
                else:
                    invalid_lines += 1
                    logger.warning(f"Line {line_num} has insufficient values: '{line}'")
                        
        if not points:
            raise ValueError("No valid control points found in file")
            
        if invalid_lines > 0:
            logger.warning(f"Found {invalid_lines} invalid lines out of {line_count} total lines")
            
        return np.array(points)
        
    except UnicodeDecodeError:
        logger.error(f"File {filename} is not a valid text file")
        raise IOError(f"File {filename} is not a valid text file")
    except Exception as e:
        logger.error(f"Error reading .3df file: {e}")
        raise


def read_ffd_xyz(filename: str) -> np.ndarray:
    """Read FFD control points from a .xyz file used by DAFoam.
    
    Args:
        filename: Path to .xyz file
    
    Returns:
        np.ndarray: Control point coordinates with shape (n, 3)
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If no valid control points are found in the file
        IOError: If there's an error reading the file
    """
    # Validate input
    if not filename or not isinstance(filename, str):
        raise ValueError(f"filename must be a non-empty string, got {filename}")
        
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
        
    logger.info(f"Reading FFD control points from .xyz file: {filename}")
    
    points = []
    line_count = 0
    invalid_lines = 0
    header_lines = 0
    expected_points = 0
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Preferred DAFoam/pyGeo format: ASCII multiblock PLOT3D.
        if len(lines) >= 2:
            try:
                block_count = int(lines[0].strip())
                dimensions = tuple(map(int, lines[1].split()))
            except ValueError:
                block_count = 0
                dimensions = ()

            if block_count == 1 and len(dimensions) == 3:
                ni, nj, nk = dimensions
                num_points = ni * nj * nk
                values = np.fromstring(" ".join(lines[2:]), sep=" ")
                if values.size != 3 * num_points:
                    raise ValueError(
                        f"PLOT3D coordinate count mismatch: expected "
                        f"{3 * num_points}, found {values.size}"
                    )

                plot3d_points = np.column_stack(
                    (
                        values[:num_points],
                        values[num_points:2 * num_points],
                        values[2 * num_points:],
                    )
                )
                # Convert i-fastest PLOT3D ordering back to OpenFFD's
                # internal ordering (k fastest, then j, then i).
                return (
                    plot3d_points.reshape(nk, nj, ni, 3)
                    .transpose(2, 1, 0, 3)
                    .reshape(-1, 3)
                )

        # Backward compatibility for the former count/comment/XYZ layout.
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            line_count += 1

            # Skip empty lines
            if not line:
                continue

            # Handle first line with number of points
            if line_num == 1:
                try:
                    expected_points = int(line.strip())
                    header_lines += 1
                    continue
                except ValueError:
                    logger.warning(f"First line should contain number of points, got: '{line}'")
                    # Continue anyway, maybe it's a different format

            # Skip second line (description)
            if line_num == 2:
                header_lines += 1
                continue

            # Parse coordinate lines
            parts = line.split()
            if len(parts) >= 3:  # x, y, z
                try:
                    # Try to parse as floats
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
                except (ValueError, IndexError) as e:
                    invalid_lines += 1
                    logger.warning(f"Invalid line {line_num} in .xyz file: '{line}', error: {e}")
            else:
                invalid_lines += 1
                logger.warning(f"Line {line_num} has insufficient values: '{line}'")
                        
        if not points:
            raise ValueError("No valid control points found in file")
            
        if invalid_lines > 0:
            logger.warning(f"Found {invalid_lines} invalid lines out of {line_count} total lines")
            
        if expected_points > 0 and len(points) != expected_points:
            logger.warning(f"Expected {expected_points} points but found {len(points)}")
            
        return np.array(points)
        
    except UnicodeDecodeError:
        logger.error(f"File {filename} is not a valid text file")
        raise IOError(f"File {filename} is not a valid text file")
    except Exception as e:
        logger.error(f"Error reading .xyz file: {e}")
        raise
