"""Abstract base class for solver interfaces.

This module defines the abstract base class for solver interfaces that can be used
with OpenFFD. This provides a consistent API for different solver backends.
"""

import abc
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pathlib


class SolverInterface(abc.ABC):
    """Abstract base class for solver interfaces.
    
    This class defines the interface that all solver backends must implement
    to work with OpenFFD. It provides methods for mesh handling, solving,
    and sensitivity analysis.
    """
    
    def __init__(self, solver_path: str, case_dir: Optional[str] = None):
        """Initialize the solver interface.
        
        Args:
            solver_path: Path to the solver executable
            case_dir: Directory containing the case setup (optional)
        """
        self.solver_path = pathlib.Path(solver_path)
        self.case_dir = pathlib.Path(case_dir) if case_dir else None
        
    @abc.abstractmethod
    def setup_case(self, mesh_file: str, config: Dict[str, Any]) -> bool:
        """Set up the solver case with the given mesh and configuration.
        
        Args:
            mesh_file: Path to the mesh file
            config: Configuration parameters for the solver
            
        Returns:
            True if setup was successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def solve(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Run the solver with the current case setup.
        
        Args:
            config: Optional configuration overrides for this solve
            
        Returns:
            True if solve was successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def solve_adjoint(self, objective_function: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Run the adjoint solver to compute sensitivities.
        
        Args:
            objective_function: Name of the objective function to use
            config: Optional configuration overrides for this solve
            
        Returns:
            True if adjoint solve was successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def get_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current mesh vertices and connectivity.
        
        Returns:
            Tuple of (vertices, cells) where vertices is an (N, 3) array of coordinates
            and cells is a list of cell connectivities
        """
        pass
    
    @abc.abstractmethod
    def set_mesh(self, vertices: np.ndarray, cells: Optional[np.ndarray] = None) -> bool:
        """Update the mesh vertices (and optionally connectivity).
        
        Args:
            vertices: New vertex coordinates as (N, 3) array
            cells: New cell connectivity (optional, only used if topology changes)
            
        Returns:
            True if mesh update was successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def get_objective_value(self, objective_function: str) -> float:
        """Get the value of the specified objective function.
        
        Args:
            objective_function: Name of the objective function
            
        Returns:
            Value of the objective function
        """
        pass
    
    @abc.abstractmethod
    def get_surface_sensitivities(self, zone_name: Optional[str] = None) -> np.ndarray:
        """Get the sensitivities on surface mesh nodes.
        
        Args:
            zone_name: Optional name of the zone to get sensitivities for
            
        Returns:
            Array of sensitivities, shape (N, 3) where N is the number of surface nodes
        """
        pass
    
    @abc.abstractmethod
    def export_results(self, output_file: str, fields: Optional[List[str]] = None) -> bool:
        """Export the results to a file.
        
        Args:
            output_file: Path to the output file
            fields: Optional list of field names to export
            
        Returns:
            True if export was successful, False otherwise
        """
        pass
