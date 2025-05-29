"""Interface module for using sonicFoam directly with OpenFFD.

This module provides a direct interface between OpenFFD and OpenFOAM's sonicFoam solver
for compressible sonic/supersonic flow simulations, with adjoint-based sensitivity analysis
for shape optimization.
"""

import os
import logging
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

from openffd.solvers.openfoam.interface import OpenFOAMInterface
from openffd.solvers.openfoam.mesh_adapter import OpenFOAMMeshAdapter
from openffd.solvers.openfoam.sensitivity import SensitivityMapper
from openffd.core.control_box import FFDBox

logger = logging.getLogger(__name__)


class SonicFoamInterface(OpenFOAMInterface):
    """Interface for the standard sonicFoam solver with OpenFFD integration.
    
    This class extends the OpenFOAMInterface to provide specific functionality
    for sonic/supersonic flow optimization using finite-difference approximations
    for sensitivity calculations.
    """
    
    def __init__(
        self,
        case_dir: str,
        mesh_adapter: OpenFOAMMeshAdapter,
        sensitivity_mapper: Optional[SensitivityMapper] = None,
        ffd_box: Optional[FFDBox] = None,
        settings: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the SonicFoamInterface.
        
        Args:
            case_dir: Path to the OpenFOAM case directory
            mesh_adapter: OpenFOAM mesh adapter for mesh manipulation
            sensitivity_mapper: Mapper for transforming sensitivities between mesh and FFD
            ffd_box: FFD control box for shape parameterization
            settings: Additional settings for the solver
        """
        solver_settings = settings or {}
        solver_path = solver_settings.get("solver_path", os.environ.get("FOAM_APPBIN", ""))
        
        super().__init__(solver_path, case_dir, solver_type="sonicFoam")
        
        self.mesh_adapter = mesh_adapter
        self.sensitivity_mapper = sensitivity_mapper
        self.ffd_box = ffd_box
        
        # Default settings for optimization
        self.default_settings = {
            "objective_type": "drag",       # Default objective function
            "surface_ids": ["wall"],        # Default surface for objective calculation
            "step_size": 0.01,              # Step size for finite differences
            "convergence_tolerance": 1e-6,  # Convergence tolerance
            "max_iterations": 1000,         # Maximum number of iterations
            "write_interval": 100           # Interval for writing results
        }
        
        # Update settings with user-provided values
        self.settings = {**self.default_settings, **(settings or {})}
        
        # Objective function value
        self.objective_value = None
        
    def run_flow_analysis(self) -> float:
        """Run the sonicFoam solver to analyze the flow.
        
        Returns:
            float: The value of the objective function
        """
        logger.info(f"Running sonicFoam analysis in {self.case_dir}")
        
        # Ensure initial fields exist
        if not os.path.exists(f"{self.case_dir}/0/p"):
            raise FileNotFoundError(f"Initial field files not found in {self.case_dir}/0")
            
        # Run the sonicFoam solver
        cmd = [
            os.path.join(self.solver_path, "sonicFoam"), 
            "-case", 
            self.case_dir
        ]
        
        try:
            logger.info("Executing: %s", " ".join(cmd))
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            logger.debug("sonicFoam output: %s", result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error("sonicFoam execution failed: %s", e.stderr)
            raise RuntimeError(f"sonicFoam failed with exit code {e.returncode}")
            
        # Calculate the objective function from the results
        self.objective_value = self._calculate_objective()
        
        return self.objective_value
        
    def _calculate_objective(self) -> float:
        """Calculate the objective function value based on the flow results.
        
        Returns:
            float: Value of the objective function
        """
        objective_type = self.settings["objective_type"]
        surface_ids = self.settings["surface_ids"]
        
        # Get the latest time directory
        time_dirs = [d for d in os.listdir(self.case_dir) 
                    if os.path.isdir(os.path.join(self.case_dir, d)) and d.replace('.', '', 1).isdigit()]
        latest_time = max(time_dirs, key=float)
        
        # Get the forces if needed
        if objective_type in ["drag", "lift", "force"]:
            force_file = os.path.join(self.case_dir, "postProcessing", "forceCoeffs", latest_time, "forceCoeffs.dat")
            
            if os.path.exists(force_file):
                # Read the force coefficients
                df = pd.read_csv(force_file, delim_whitespace=True, comment="#", header=None)
                # Assuming format: time, Cd, Cs, Cl, CmRoll, CmPitch, CmYaw
                if objective_type == "drag":
                    return df.iloc[-1, 1]  # Cd
                elif objective_type == "lift":
                    return df.iloc[-1, 3]  # Cl
                else:  # "force" - total magnitude
                    return np.sqrt(df.iloc[-1, 1]**2 + df.iloc[-1, 2]**2 + df.iloc[-1, 3]**2)
            else:
                logger.warning(f"Force coefficient file not found: {force_file}")
                return 0.0
        
        elif objective_type == "pressure_uniformity":
            # Implement pressure uniformity calculation
            # This would require extracting surface data and calculating variance
            logger.warning("Pressure uniformity calculation not implemented yet")
            return 0.0
        
        else:
            logger.warning(f"Unknown objective type: {objective_type}")
            return 0.0
    
    def calculate_sensitivities(self, step_size: float = None) -> np.ndarray:
        """Calculate sensitivities using finite difference approximations.
        
        Args:
            step_size: Step size for finite differences (defaults to settings value)
            
        Returns:
            np.ndarray: Sensitivities at FFD control points
        """
        if step_size is None:
            step_size = self.settings["step_size"]
            
        if self.ffd_box is None or self.sensitivity_mapper is None:
            logger.error("FFD box or sensitivity mapper not initialized")
            return np.array([])
            
        # Get the control points
        control_points = self.ffd_box.get_control_points()
        n_points = control_points.shape[0]
        
        # Initialize sensitivities
        sensitivities = np.zeros_like(control_points)
        
        # Get baseline objective value
        if self.objective_value is None:
            baseline_value = self.run_flow_analysis()
        else:
            baseline_value = self.objective_value
            
        logger.info(f"Baseline objective value: {baseline_value}")
        
        # Calculate sensitivities for each control point
        for i in range(n_points):
            for j in range(3):  # x, y, z directions
                # Perturb the control point
                original_value = control_points[i, j]
                control_points[i, j] += step_size
                
                # Update the FFD box and mesh
                self.ffd_box.set_control_points(control_points)
                new_points = self.ffd_box.deform_mesh(self.mesh_adapter.get_original_mesh_points())
                self.mesh_adapter.update_mesh(new_points)
                
                # Run the analysis
                perturbed_value = self.run_flow_analysis()
                
                # Calculate finite difference
                sensitivities[i, j] = (perturbed_value - baseline_value) / step_size
                
                # Restore the control point
                control_points[i, j] = original_value
                self.ffd_box.set_control_points(control_points)
                restored_points = self.ffd_box.deform_mesh(self.mesh_adapter.get_original_mesh_points())
                self.mesh_adapter.update_mesh(restored_points)
                
                logger.info(f"Sensitivity for control point {i}, direction {j}: {sensitivities[i, j]}")
                
        return sensitivities
        
    def perform_optimization_step(self, step_size: float = None) -> Tuple[float, float]:
        """Perform one step of shape optimization.
        
        Args:
            step_size: Step size for optimization update (defaults to settings value)
            
        Returns:
            Tuple[float, float]: The initial and final objective values
        """
        if step_size is None:
            step_size = self.settings["step_size"]
            
        # Run baseline analysis
        initial_obj = self.run_flow_analysis()
        
        # Calculate sensitivities
        sensitivities = self.calculate_sensitivities()
        
        # Update control points in the direction of negative gradient
        control_points = self.ffd_box.get_control_points()
        control_points -= step_size * sensitivities  # Gradient descent
        
        # Apply the updated control points
        self.ffd_box.set_control_points(control_points)
        new_points = self.ffd_box.deform_mesh(self.mesh_adapter.get_original_mesh_points())
        self.mesh_adapter.update_mesh(new_points)
        
        # Run analysis with updated shape
        final_obj = self.run_flow_analysis()
        
        logger.info(f"Optimization step: initial obj = {initial_obj}, final obj = {final_obj}")
        return initial_obj, final_obj


def create_sonic_foam_interface(
    case_dir: str,
    ffd_file: str,
    settings: Optional[Dict[str, Any]] = None
) -> SonicFoamInterface:
    """Create a complete SonicFoamInterface with all required components.
    
    Args:
        case_dir: Path to the OpenFOAM case directory
        ffd_file: Path to the FFD box file
        settings: Additional settings for the interface
        
    Returns:
        SonicFoamInterface: Fully configured interface for optimization
    """
    # Load FFD box
    ffd_box = FFDBox.from_file(ffd_file)
    
    # Create mesh adapter
    mesh_adapter = OpenFOAMMeshAdapter(case_dir)
    
    # Create sensitivity mapper
    sensitivity_mapper = SensitivityMapper(ffd_box)
    
    # Create interface
    interface = SonicFoamInterface(
        case_dir=case_dir,
        mesh_adapter=mesh_adapter,
        sensitivity_mapper=sensitivity_mapper,
        ffd_box=ffd_box,
        settings=settings
    )
    
    return interface
