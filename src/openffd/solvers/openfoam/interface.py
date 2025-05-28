"""OpenFOAM interface for OpenFFD.

This module provides the main interface for connecting OpenFFD with OpenFOAM solvers,
particularly focused on sonicFoam with adjoint capabilities for supersonic flows.
"""

import os
import logging
import pathlib
import subprocess
import shutil
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

from openffd.solvers.base.solver_interface import SolverInterface
from openffd.solvers.openfoam.mesh_adapter import OpenFOAMMeshAdapter
from openffd.core.control_box import FFDBox
from openffd.mesh.zone_extractor import ZoneInfo

logger = logging.getLogger(__name__)


class OpenFOAMInterface(SolverInterface):
    """Interface for OpenFOAM solvers with adjoint capabilities.
    
    This class provides methods for connecting OpenFFD with OpenFOAM,
    particularly sonicFoam with adjoint capabilities for supersonic flows.
    """
    
    def __init__(
        self, 
        solver_path: str, 
        case_dir: str,
        solver_type: str = "sonicFoam",
        adjoint_solver_type: str = "sonicFoamAdjoint"
    ):
        """Initialize the OpenFOAM interface.
        
        Args:
            solver_path: Path to the OpenFOAM installation
            case_dir: Directory containing the OpenFOAM case
            solver_type: Type of OpenFOAM solver to use (default: sonicFoam)
            adjoint_solver_type: Type of adjoint solver to use (default: sonicFoamAdjoint)
        """
        super().__init__(solver_path, case_dir)
        self.solver_type = solver_type
        self.adjoint_solver_type = adjoint_solver_type
        self.mesh_adapter = OpenFOAMMeshAdapter(case_dir)
        
        # Keep track of the design variables (FFD control points)
        self.ffd_box = None
        self.original_control_points = None
        self.sensitivity_map = {}
        
        # Objective function tracking
        self.objective_values = {}
        self.available_objectives = [
            "drag", "lift", "moment", "pressure_loss", "uniformity"
        ]
    
    def setup_case(self, mesh_file: str, config: Dict[str, Any]) -> bool:
        """Set up the OpenFOAM case with the given mesh and configuration.
        
        Args:
            mesh_file: Path to the mesh file (can be OpenFFD format or OpenFOAM case)
            config: Configuration parameters for the solver
            
        Returns:
            True if setup was successful, False otherwise
        """
        logger.info(f"Setting up OpenFOAM case in {self.case_dir}")
        
        # Check if the mesh file is an OpenFOAM case directory
        mesh_path = pathlib.Path(mesh_file)
        if mesh_path.is_dir() and (mesh_path / "constant" / "polyMesh").exists():
            # Copy the OpenFOAM case
            try:
                self._copy_openfoam_case(mesh_path, self.case_dir)
                logger.info(f"Copied OpenFOAM case from {mesh_path} to {self.case_dir}")
                return True
            except Exception as e:
                logger.error(f"Failed to copy OpenFOAM case: {e}")
                return False
        else:
            # Assume it's an OpenFFD-compatible mesh file
            # We need to convert it to OpenFOAM format
            logger.info(f"Converting mesh file {mesh_file} to OpenFOAM format")
            return False  # Placeholder for actual implementation
    
    def solve(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Run the OpenFOAM solver with the current case setup.
        
        Args:
            config: Optional configuration overrides for this solve
            
        Returns:
            True if solve was successful, False otherwise
        """
        logger.info(f"Running OpenFOAM solver ({self.solver_type}) on case {self.case_dir}")
        
        # Apply any configuration overrides
        if config:
            self._apply_config_overrides(config)
        
        # Run the solver
        try:
            result = subprocess.run(
                [self.solver_type, "-case", str(self.case_dir)],
                check=True,
                text=True,
                capture_output=True
            )
            logger.info(f"Solver completed: {self.solver_type}")
            
            # Parse the results to get objective function values
            self._parse_objective_values()
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Solver failed: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error(f"OpenFOAM solver not found: {self.solver_type}")
            return False
    
    def solve_adjoint(self, objective_function: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Run the adjoint solver to compute sensitivities.
        
        Args:
            objective_function: Name of the objective function to use
            config: Optional configuration overrides for this solve
            
        Returns:
            True if adjoint solve was successful, False otherwise
        """
        if objective_function not in self.available_objectives:
            logger.error(f"Unknown objective function: {objective_function}")
            return False
        
        logger.info(f"Running adjoint solver ({self.adjoint_solver_type}) for {objective_function}")
        
        # Apply any configuration overrides
        if config:
            self._apply_config_overrides(config)
        
        # Set up the objective function
        self._setup_objective_function(objective_function)
        
        # Run the adjoint solver
        try:
            result = subprocess.run(
                [self.adjoint_solver_type, "-case", str(self.case_dir), "-objective", objective_function],
                check=True,
                text=True,
                capture_output=True
            )
            logger.info(f"Adjoint solver completed for {objective_function}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Adjoint solver failed: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error(f"OpenFOAM adjoint solver not found: {self.adjoint_solver_type}")
            return False
    
    def get_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current mesh vertices and connectivity from the OpenFOAM case.
        
        Returns:
            Tuple of (vertices, cells) where vertices is an (N, 3) array of coordinates
            and cells is a list of cell connectivities
        """
        try:
            points, zones = self.mesh_adapter.import_mesh_to_openffd()
            return points, np.array([])  # Placeholder for cell connectivity
        except Exception as e:
            logger.error(f"Failed to get mesh: {e}")
            return np.array([]), np.array([])
    
    def set_mesh(self, vertices: np.ndarray, cells: Optional[np.ndarray] = None) -> bool:
        """Update the OpenFOAM mesh vertices (and optionally connectivity).
        
        Args:
            vertices: New vertex coordinates as (N, 3) array
            cells: New cell connectivity (optional, only used if topology changes)
            
        Returns:
            True if mesh update was successful, False otherwise
        """
        try:
            # Export the mesh to OpenFOAM format
            original_mesh_dir = self.case_dir / "constant" / "polyMesh.orig"
            if not original_mesh_dir.exists() and (self.case_dir / "constant" / "polyMesh").exists():
                # Back up original mesh if it doesn't exist yet
                shutil.copytree(
                    self.case_dir / "constant" / "polyMesh",
                    original_mesh_dir
                )
            
            success = self.mesh_adapter.export_mesh_from_openffd(
                vertices, 
                original_mesh_dir if original_mesh_dir.exists() else None
            )
            
            if success:
                # Run OpenFOAM mesh update utility
                return self.mesh_adapter.run_mesh_update_utility()
            return False
        except Exception as e:
            logger.error(f"Failed to set mesh: {e}")
            return False
    
    def get_objective_value(self, objective_function: str) -> float:
        """Get the value of the specified objective function.
        
        Args:
            objective_function: Name of the objective function
            
        Returns:
            Value of the objective function
        """
        if objective_function not in self.available_objectives:
            logger.error(f"Unknown objective function: {objective_function}")
            return float('nan')
        
        # Check if we have the value cached
        if objective_function in self.objective_values:
            return self.objective_values[objective_function]
        
        # If not, try to parse it from the results
        self._parse_objective_values()
        
        # Return the value if available, otherwise NaN
        return self.objective_values.get(objective_function, float('nan'))
    
    def get_surface_sensitivities(self, zone_name: Optional[str] = None) -> np.ndarray:
        """Get the sensitivities on surface mesh nodes from the adjoint solution.
        
        Args:
            zone_name: Optional name of the zone to get sensitivities for
            
        Returns:
            Array of sensitivities, shape (N, 3) where N is the number of surface nodes
        """
        # In a real implementation, this would parse the adjoint solution output
        # to get the sensitivities on surface mesh nodes
        
        # Placeholder for actual implementation
        return np.array([])
    
    def export_results(self, output_file: str, fields: Optional[List[str]] = None) -> bool:
        """Export the OpenFOAM results to a file.
        
        Args:
            output_file: Path to the output file
            fields: Optional list of field names to export
            
        Returns:
            True if export was successful, False otherwise
        """
        # This would use OpenFOAM's post-processing utilities to export results
        # For example, using the sample utility
        
        if fields is None:
            fields = ["p", "U", "T", "rho"]
        
        field_args = []
        for field in fields:
            field_args.extend(["-field", field])
        
        try:
            result = subprocess.run(
                ["sample", "-case", str(self.case_dir)] + field_args + ["-output", output_file],
                check=True,
                text=True,
                capture_output=True
            )
            logger.info(f"Exported results to {output_file}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to export results: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("OpenFOAM sample utility not found")
            return False
    
    def set_ffd_box(self, ffd_box: FFDBox) -> None:
        """Set the FFD box to use for mesh deformation.
        
        Args:
            ffd_box: FFD box object
        """
        self.ffd_box = ffd_box
        if ffd_box is not None:
            # Store the original control points for sensitivity mapping
            self.original_control_points = ffd_box.control_points.copy()
    
    def map_sensitivities_to_control_points(self) -> np.ndarray:
        """Map the surface sensitivities to FFD control points.
        
        Returns:
            Array of sensitivities for each FFD control point, shape (N, 3)
        """
        if self.ffd_box is None:
            logger.error("No FFD box set, cannot map sensitivities")
            return np.array([])
        
        # Get the surface sensitivities
        surface_sensitivities = self.get_surface_sensitivities()
        
        # Map them to FFD control points
        # In a real implementation, this would use the chain rule and the
        # FFD basis functions to map the sensitivities
        
        # Placeholder for actual implementation
        control_point_sensitivities = np.zeros_like(self.original_control_points)
        
        return control_point_sensitivities
    
    def _copy_openfoam_case(self, source_dir: pathlib.Path, target_dir: pathlib.Path) -> None:
        """Copy an OpenFOAM case directory.
        
        Args:
            source_dir: Source case directory
            target_dir: Target case directory
        """
        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy standard OpenFOAM directories
        for dirname in ["constant", "system"]:
            source_subdir = source_dir / dirname
            target_subdir = target_dir / dirname
            
            if source_subdir.exists():
                if target_subdir.exists():
                    shutil.rmtree(target_subdir)
                shutil.copytree(source_subdir, target_subdir)
        
        # Copy time directories (0, etc.)
        for item in source_dir.iterdir():
            if item.is_dir() and item.name.replace(".", "").isdigit():
                target_time_dir = target_dir / item.name
                if target_time_dir.exists():
                    shutil.rmtree(target_time_dir)
                shutil.copytree(item, target_time_dir)
    
    def _apply_config_overrides(self, config: Dict[str, Any]) -> None:
        """Apply configuration overrides to the OpenFOAM case.
        
        Args:
            config: Configuration parameters to override
        """
        # Placeholder for actual implementation
        pass
    
    def _setup_objective_function(self, objective_function: str) -> None:
        """Set up the specified objective function for adjoint solving.
        
        Args:
            objective_function: Name of the objective function
        """
        # Placeholder for actual implementation
        pass
    
    def _parse_objective_values(self) -> None:
        """Parse the objective function values from the solver output."""
        # Placeholder for actual implementation
        pass
