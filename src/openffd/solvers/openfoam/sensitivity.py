"""Sensitivity mapping utilities for OpenFOAM-OpenFFD interface.

This module provides utilities for mapping sensitivities between mesh vertices
and FFD control points, which is essential for adjoint-based shape optimization.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from openffd.core.control_box import FFDBox

logger = logging.getLogger(__name__)


class SensitivityMapper:
    """Mapper for transforming sensitivities between mesh and FFD control points."""
    
    def __init__(self, ffd_box: FFDBox):
        """Initialize the sensitivity mapper.
        
        Args:
            ffd_box: FFD box for computing sensitivity mappings
        """
        self.ffd_box = ffd_box
    
    def mesh_to_control_points(
        self, 
        mesh_points: np.ndarray, 
        mesh_sensitivities: np.ndarray
    ) -> np.ndarray:
        """Map sensitivities from mesh vertices to FFD control points.
        
        Args:
            mesh_points: Mesh vertex coordinates, shape (N, 3)
            mesh_sensitivities: Sensitivities at mesh vertices, shape (N, 3)
            
        Returns:
            Sensitivities at FFD control points, shape (M, 3) where M is the number of control points
        """
        if self.ffd_box is None:
            logger.error("No FFD box set, cannot map sensitivities")
            return np.array([])
        
        # Get the parametric coordinates of the mesh points in the FFD box
        u_params, v_params, w_params = self.ffd_box.get_parametric_coordinates(mesh_points)
        
        # Get the number of control points in each dimension
        nx, ny, nz = self.ffd_box.dims
        
        # Initialize control point sensitivities
        control_point_sensitivities = np.zeros((nx * ny * nz, 3))
        
        # Compute the sensitivity of each mesh point with respect to each control point
        # This uses the chain rule: dJ/dP = sum_i (dJ/dx_i * dx_i/dP)
        # where J is the objective function, x_i are the mesh points, and P are the control points
        
        # For each mesh point
        for i in range(len(mesh_points)):
            u, v, w = u_params[i], v_params[i], w_params[i]
            
            # For each control point, compute the basis function value
            for i_cp in range(nx):
                for j_cp in range(ny):
                    for k_cp in range(nz):
                        # Compute the Bernstein basis function values
                        basis_value = self._bernstein_basis_value(u, v, w, i_cp, j_cp, k_cp, nx-1, ny-1, nz-1)
                        
                        # Accumulate the sensitivity contribution
                        control_point_idx = i_cp * ny * nz + j_cp * nz + k_cp
                        control_point_sensitivities[control_point_idx] += basis_value * mesh_sensitivities[i]
        
        return control_point_sensitivities
    
    def control_points_to_mesh(
        self, 
        mesh_points: np.ndarray, 
        control_point_sensitivities: np.ndarray
    ) -> np.ndarray:
        """Map sensitivities from FFD control points to mesh vertices.
        
        Args:
            mesh_points: Mesh vertex coordinates, shape (N, 3)
            control_point_sensitivities: Sensitivities at FFD control points, shape (M, 3)
            
        Returns:
            Sensitivities at mesh vertices, shape (N, 3)
        """
        if self.ffd_box is None:
            logger.error("No FFD box set, cannot map sensitivities")
            return np.array([])
        
        # Get the parametric coordinates of the mesh points in the FFD box
        u_params, v_params, w_params = self.ffd_box.get_parametric_coordinates(mesh_points)
        
        # Get the number of control points in each dimension
        nx, ny, nz = self.ffd_box.dims
        
        # Initialize mesh sensitivities
        mesh_sensitivities = np.zeros((len(mesh_points), 3))
        
        # For each mesh point
        for i in range(len(mesh_points)):
            u, v, w = u_params[i], v_params[i], w_params[i]
            
            # For each control point, compute the contribution to the mesh point sensitivity
            for i_cp in range(nx):
                for j_cp in range(ny):
                    for k_cp in range(nz):
                        # Compute the Bernstein basis function values
                        basis_value = self._bernstein_basis_value(u, v, w, i_cp, j_cp, k_cp, nx-1, ny-1, nz-1)
                        
                        # Get the control point sensitivity
                        control_point_idx = i_cp * ny * nz + j_cp * nz + k_cp
                        control_point_sensitivity = control_point_sensitivities[control_point_idx]
                        
                        # Accumulate the sensitivity contribution
                        mesh_sensitivities[i] += basis_value * control_point_sensitivity
        
        return mesh_sensitivities
    
    def _bernstein_basis_value(
        self, 
        u: float, 
        v: float, 
        w: float, 
        i: int, 
        j: int, 
        k: int, 
        p: int, 
        q: int, 
        r: int
    ) -> float:
        """Compute the Bernstein basis function value.
        
        Args:
            u, v, w: Parametric coordinates (0-1)
            i, j, k: Control point indices
            p, q, r: Polynomial degrees in each dimension
            
        Returns:
            Basis function value
        """
        # Compute the binomial coefficients
        def binomial(n, k):
            """Compute the binomial coefficient (n choose k)."""
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1
            return binomial(n-1, k-1) + binomial(n-1, k)
        
        # Compute the Bernstein polynomial
        def bernstein(u, i, n):
            """Compute the Bernstein polynomial."""
            return binomial(n, i) * (u ** i) * ((1 - u) ** (n - i))
        
        # Compute the tensor product of Bernstein polynomials
        return bernstein(u, i, p) * bernstein(v, j, q) * bernstein(w, k, r)
    
    def compute_finite_difference_validation(
        self, 
        mesh_points: np.ndarray, 
        objective_function: callable, 
        step_size: float = 1e-5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate sensitivities using finite differences.
        
        This function computes sensitivities of the objective function with respect to
        FFD control points using finite differences, and compares them to the ones
        computed using the adjoint approach.
        
        Args:
            mesh_points: Mesh vertex coordinates, shape (N, 3)
            objective_function: Function that takes mesh points and returns objective value
            step_size: Step size for finite differences
            
        Returns:
            Tuple of (adjoint_sensitivities, fd_sensitivities)
        """
        if self.ffd_box is None:
            logger.error("No FFD box set, cannot compute finite differences")
            return np.array([]), np.array([])
        
        # Get the number of control points
        nx, ny, nz = self.ffd_box.dims
        n_control_points = nx * ny * nz
        
        # Store the original control points
        original_control_points = self.ffd_box.control_points.copy()
        
        # Compute the baseline objective value
        baseline_mesh = self.ffd_box.evaluate_points(mesh_points)
        baseline_objective = objective_function(baseline_mesh)
        
        # Initialize finite difference sensitivities
        fd_sensitivities = np.zeros((n_control_points, 3))
        
        # For each control point and each direction
        for cp_idx in range(n_control_points):
            for dim in range(3):
                # Perturb the control point
                perturbed_control_points = original_control_points.copy()
                perturbed_control_points[cp_idx, dim] += step_size
                
                # Update the FFD box
                self.ffd_box.control_points = perturbed_control_points
                
                # Compute the new mesh and objective
                perturbed_mesh = self.ffd_box.evaluate_points(mesh_points)
                perturbed_objective = objective_function(perturbed_mesh)
                
                # Compute the finite difference sensitivity
                fd_sensitivities[cp_idx, dim] = (perturbed_objective - baseline_objective) / step_size
        
        # Restore the original control points
        self.ffd_box.control_points = original_control_points
        
        # Compute the adjoint sensitivities (this would be done using the adjoint solver)
        # For this validation function, we'd need a way to compute the actual adjoint sensitivities
        # This is just a placeholder
        adjoint_sensitivities = np.zeros_like(fd_sensitivities)
        
        return adjoint_sensitivities, fd_sensitivities
