"""Sensitivity mapping module for OpenFFD GUI.

This module provides functionality to map adjoint sensitivities from OpenFOAM
to FFD control points for shape optimization.
"""

import logging
import numpy as np
import os
from typing import Optional, Dict, List, Tuple, Any

from openffd.core.control_box import create_ffd_box
from openffd.utils.parallel import ParallelConfig

# Configure logging
logger = logging.getLogger(__name__)


class SensitivityMapper:
    """Maps mesh sensitivities to FFD control points."""
    
    def __init__(self, parallel_config: Optional[ParallelConfig] = None):
        """Initialize the sensitivity mapper.
        
        Args:
            parallel_config: Parallel processing configuration
        """
        self.parallel_config = parallel_config or ParallelConfig()
    
    def load_sensitivities(self, sensitivity_file: str) -> np.ndarray:
        """Load sensitivities from file.
        
        Args:
            sensitivity_file: Path to sensitivity file from sonicFoamAdjoint
            
        Returns:
            Numpy array of sensitivities
        """
        try:
            logger.info(f"Loading sensitivities from {sensitivity_file}")
            
            # Check file format based on extension
            if sensitivity_file.lower().endswith('.csv'):
                # CSV format (x,y,z,dx,dy,dz)
                data = np.loadtxt(sensitivity_file, delimiter=',', skiprows=1)
                
                # Extract sensitivities (dx, dy, dz)
                sensitivities = data[:, 3:6]
            elif sensitivity_file.lower().endswith('.txt'):
                # Simple text format, one sensitivity vector per line
                sensitivities = np.loadtxt(sensitivity_file)
            else:
                # Try to load as numpy binary
                sensitivities = np.load(sensitivity_file)
            
            logger.info(f"Loaded {len(sensitivities)} sensitivity vectors")
            return sensitivities
        except Exception as e:
            logger.error(f"Error loading sensitivities: {str(e)}")
            raise
    
    def map_to_control_points(
        self, 
        mesh_points: np.ndarray, 
        mesh_sensitivities: np.ndarray,
        control_points: np.ndarray,
        control_dims: Tuple[int, int, int]
    ) -> np.ndarray:
        """Map mesh surface sensitivities to FFD control points.
        
        Args:
            mesh_points: Surface mesh point coordinates
            mesh_sensitivities: Sensitivities at mesh points
            control_points: FFD control point coordinates
            control_dims: Control point dimensions (nx, ny, nz)
            
        Returns:
            Sensitivities at control points
        """
        try:
            logger.info(f"Mapping sensitivities from {len(mesh_points)} mesh points to {len(control_points)} control points")
            
            # Import the mapping functionality here to avoid circular imports
            from openffd.solvers.openfoam.sensitivity import map_sensitivities_to_control_points
            
            nx, ny, nz = control_dims
            
            # Use the sensitivity mapper from the solvers module
            control_sensitivities = map_sensitivities_to_control_points(
                mesh_points, 
                mesh_sensitivities,
                control_points,
                (nx, ny, nz),
                parallel_config=self.parallel_config
            )
            
            logger.info(f"Successfully mapped sensitivities to control points")
            return control_sensitivities
        except Exception as e:
            logger.error(f"Error mapping sensitivities: {str(e)}")
            raise
    
    def compute_deformation(
        self, 
        control_points: np.ndarray,
        control_sensitivities: np.ndarray,
        scale_factor: float = 1.0
    ) -> np.ndarray:
        """Compute deformation of control points based on sensitivities.
        
        Args:
            control_points: Original control point coordinates
            control_sensitivities: Sensitivities at control points
            scale_factor: Scale factor for deformation
            
        Returns:
            Deformed control point coordinates
        """
        try:
            logger.info(f"Computing control point deformation with scale factor {scale_factor}")
            
            # Scale the sensitivities
            scaled_sensitivities = control_sensitivities * scale_factor
            
            # Apply deformation
            deformed_points = control_points + scaled_sensitivities
            
            logger.info(f"Computed deformation for {len(control_points)} control points")
            return deformed_points
        except Exception as e:
            logger.error(f"Error computing deformation: {str(e)}")
            raise
    
    def smooth_sensitivities(
        self,
        control_sensitivities: np.ndarray,
        control_dims: Tuple[int, int, int],
        smoothing_factor: float = 0.5
    ) -> np.ndarray:
        """Apply smoothing to control point sensitivities.
        
        Args:
            control_sensitivities: Original control point sensitivities
            control_dims: Control point dimensions (nx, ny, nz)
            smoothing_factor: Smoothing factor (0.0-1.0)
            
        Returns:
            Smoothed sensitivities
        """
        try:
            logger.info(f"Smoothing sensitivities with factor {smoothing_factor}")
            
            nx, ny, nz = control_dims
            
            # Reshape sensitivities to 3D grid
            sens_3d = control_sensitivities.reshape(nx, ny, nz, 3)
            
            # Apply simple smoothing by averaging with neighbors
            smoothed = np.copy(sens_3d)
            
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        # Get neighboring indices
                        i_min = max(0, i-1)
                        i_max = min(nx-1, i+1)
                        j_min = max(0, j-1)
                        j_max = min(ny-1, j+1)
                        k_min = max(0, k-1)
                        k_max = min(nz-1, k+1)
                        
                        # Get neighboring sensitivities
                        neighbors = sens_3d[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1, :]
                        
                        # Compute average
                        avg = np.mean(neighbors, axis=(0, 1, 2))
                        
                        # Apply smoothing (blend between original and average)
                        smoothed[i, j, k] = (1.0 - smoothing_factor) * sens_3d[i, j, k] + smoothing_factor * avg
            
            # Reshape back to 2D array
            result = smoothed.reshape(-1, 3)
            
            logger.info(f"Applied smoothing to sensitivities")
            return result
        except Exception as e:
            logger.error(f"Error smoothing sensitivities: {str(e)}")
            raise
