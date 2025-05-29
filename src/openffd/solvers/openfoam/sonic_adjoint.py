"""Integration module for sonicAdjointFoam solver with OpenFFD.

This module provides the necessary classes and functions to interface between
the sonicAdjointFoam OpenFOAM solver and the OpenFFD framework for sensitivity-based
shape optimization of supersonic flow applications.
"""

import os
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd

from openffd.solvers.base.interface import SolverInterface
from openffd.solvers.openfoam.interface import OpenFOAMInterface
from openffd.solvers.openfoam.mesh_adapter import OpenFOAMMeshAdapter
from openffd.solvers.openfoam.sensitivity import SensitivityMapper
from openffd.core.control_box import FFDBox

logger = logging.getLogger(__name__)


class SonicAdjointInterface(OpenFOAMInterface):
    """Interface for the sonicAdjointFoam solver with OpenFFD integration.
    
    This class extends the base OpenFOAMInterface to provide specific
    functionality for supersonic flow optimization using adjoint-based
    sensitivity analysis.
    """
    
    def __init__(
        self,
        case_dir: str,
        mesh_adapter: OpenFOAMMeshAdapter,
        sensitivity_mapper: Optional[SensitivityMapper] = None,
        ffd_box: Optional[FFDBox] = None,
        settings: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the SonicAdjointInterface.
        
        Args:
            case_dir: Path to the OpenFOAM case directory
            mesh_adapter: OpenFOAM mesh adapter for mesh manipulation
            sensitivity_mapper: Mapper for transforming sensitivities between mesh and FFD
            ffd_box: FFD control box for shape parameterization
            settings: Additional settings for the solver
        """
        super().__init__(case_dir, mesh_adapter, settings)
        
        self.sensitivity_mapper = sensitivity_mapper
        self.ffd_box = ffd_box
        
        # Default settings for sonicAdjointFoam
        self._default_settings.update({
            "solver_executable": "sonicAdjointFoam",
            "objective_type": "drag",
            "sensitivity_format": "OpenFFD",
            "smooth_sensitivity": True,
            "step_size": 0.01,
            "target_patches": [],  # Patches to consider for objective and sensitivity
            "export_vtk": True,    # Whether to export VTK files for visualization
        })
        
        # Update with user settings
        if settings:
            self._settings.update(settings)
            
        # Initialize sensitivity data containers
        self.surface_sensitivities = None
        self.control_point_sensitivities = None
        self.objective_value = None
        
    def setup_optimization(
        self,
        objective_type: str = "drag",
        target_patches: Optional[List[str]] = None,
        direction: Optional[List[float]] = None,
        reference_values: Optional[Dict[str, float]] = None,
    ) -> None:
        """Set up the optimization problem.
        
        Args:
            objective_type: Type of objective function (drag, lift, pressure_uniformity, etc.)
            target_patches: List of patches to apply optimization to
            direction: Direction vector for force objectives [x, y, z]
            reference_values: Reference values for normalization
        """
        logger.info(f"Setting up optimization for objective: {objective_type}")
        
        # Update settings
        self._settings["objective_type"] = objective_type
        
        if target_patches:
            self._settings["target_patches"] = target_patches
        
        # Create system/optimizationDict if it doesn't exist
        optim_dict_path = os.path.join(self.case_dir, "system", "optimizationDict")
        
        # Construct the content of the optimizationDict
        content = [
            "/*--------------------------------*- C++ -*----------------------------------*\\",
            "| =========                 |                                                 |",
            "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |",
            "|  \\\\    /   O peration     | Version:  v2406                                 |",
            "|   \\\\  /    A nd           | Website:  www.openfoam.com                      |",
            "|    \\\\/     M anipulation  |                                                 |",
            "\\*---------------------------------------------------------------------------*/",
            "FoamFile",
            "{",
            "    version     2.0;",
            "    format      ascii;",
            "    class       dictionary;",
            "    object      optimizationDict;",
            "}",
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //",
            "",
            f"objectiveType    {objective_type};",
            f"sensitivityFormat OpenFFD;",
            f"smoothSensitivity {str(self._settings['smooth_sensitivity']).lower()};",
            f"stepSize         {self._settings['step_size']};",
            "",
            "objectiveFunction",
            "{",
        ]
        
        # Add target patches
        if target_patches:
            content.append("    patches         (")
            for patch in target_patches:
                content.append(f"        {patch}")
            content.append("    );")
        
        # Add direction for force-based objectives
        if direction and objective_type in ["drag", "lift", "force"]:
            content.append(f"    direction       ({direction[0]} {direction[1]} {direction[2]});")
        
        # Add reference values if provided
        if reference_values:
            for key, value in reference_values.items():
                content.append(f"    {key}          {value};")
        
        content.extend([
            "}",
            "",
            "// Export settings",
            "exportPatches      $objectiveFunction.patches;",
            "",
            "// ************************************************************************* //"
        ])
        
        # Write the optimizationDict
        with open(optim_dict_path, "w") as f:
            f.write("\n".join(content))
        
        logger.info(f"Created optimization dictionary: {optim_dict_path}")
    
    def run_adjoint_analysis(self) -> float:
        """Run the sonicAdjointFoam solver to perform adjoint analysis.
        
        Returns:
            float: The objective function value from the adjoint analysis
        """
        logger.info("Running sonicAdjointFoam for adjoint analysis")
        
        # Run the solver
        self.run_solver(
            solver=self._settings["solver_executable"],
            parallel=self._settings.get("parallel", False),
            processors=self._settings.get("processors", 1),
            additional_args=self._settings.get("solver_args", [])
        )
        
        # Load the sensitivity data
        self._load_sensitivity_data()
        
        return self.objective_value
    
    def _load_sensitivity_data(self) -> None:
        """Load sensitivity data from the OpenFOAM case."""
        sens_file = os.path.join(
            self.case_dir,
            "sensitivities",
            "surface_sensitivities.csv"
        )
        
        meta_file = os.path.join(
            self.case_dir,
            "sensitivities",
            "sensitivity_metadata.json"
        )
        
        if not os.path.exists(sens_file):
            raise FileNotFoundError(
                f"Sensitivity file not found: {sens_file}. "
                "Make sure the adjoint analysis completed successfully."
            )
        
        # Load sensitivity data
        logger.info(f"Loading sensitivity data from {sens_file}")
        self.surface_sensitivities = pd.read_csv(sens_file, comment="#")
        
        # Load metadata if available
        if os.path.exists(meta_file):
            import json
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            self.objective_value = metadata.get("objectiveValue", None)
            logger.info(f"Loaded objective value: {self.objective_value}")
        
        # Map sensitivities to FFD control points if sensitivity mapper is available
        if self.sensitivity_mapper and self.ffd_box:
            self._map_sensitivities_to_control_points()
    
    def _map_sensitivities_to_control_points(self) -> None:
        """Map surface sensitivities to FFD control points."""
        if self.surface_sensitivities is None:
            logger.warning("No surface sensitivities available to map")
            return
        
        logger.info("Mapping surface sensitivities to FFD control points")
        
        # Extract points and sensitivities from the dataframe
        points = self.surface_sensitivities[['x', 'y', 'z']].values
        sens_vectors = self.surface_sensitivities[
            ['sensitivity_x', 'sensitivity_y', 'sensitivity_z']
        ].values
        
        # Use the sensitivity mapper to map to control points
        self.control_point_sensitivities = self.sensitivity_mapper.map_to_control_points(
            points, sens_vectors, self.ffd_box
        )
        
        logger.info(
            f"Mapped {len(points)} surface points to "
            f"{self.control_point_sensitivities.shape} control points"
        )
    
    def get_objective_gradient(self) -> np.ndarray:
        """Get the gradient of the objective function with respect to control points.
        
        Returns:
            np.ndarray: Gradient array with shape matching the FFD control points
        """
        if self.control_point_sensitivities is None:
            raise ValueError(
                "Control point sensitivities not available. "
                "Run adjoint analysis and map sensitivities first."
            )
        
        return self.control_point_sensitivities
    
    def apply_control_point_update(
        self, 
        updates: np.ndarray,
        scale_factor: float = 1.0
    ) -> None:
        """Apply updates to FFD control points and deform the mesh.
        
        Args:
            updates: Updates to apply to control points, shape matching FFD box
            scale_factor: Scaling factor for the updates
        """
        if self.ffd_box is None:
            raise ValueError("No FFD box available for control point updates")
        
        logger.info(f"Applying control point updates with scale factor {scale_factor}")
        
        # Apply updates to FFD control points
        self.ffd_box.update_control_points(updates * scale_factor)
        
        # Deform the mesh using the updated FFD
        self.mesh_adapter.deform_mesh(self.ffd_box)
        
        logger.info("Mesh deformation completed")
    
    def run_optimization_step(
        self, 
        step_size: Optional[float] = None
    ) -> Tuple[float, np.ndarray]:
        """Run a single step of the optimization process.
        
        Args:
            step_size: Step size for the control point updates (overrides settings)
            
        Returns:
            Tuple[float, np.ndarray]: Objective value and sensitivity gradient
        """
        if step_size is None:
            step_size = self._settings["step_size"]
        
        # Run the adjoint analysis
        objective = self.run_adjoint_analysis()
        
        # Get the gradient
        gradient = self.get_objective_gradient()
        
        # Apply control point updates in the negative gradient direction
        self.apply_control_point_update(-gradient, scale_factor=step_size)
        
        return objective, gradient
    
    def export_results(self, output_dir: str) -> None:
        """Export optimization results to the specified directory.
        
        Args:
            output_dir: Directory to export results to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export surface sensitivities if available
        if self.surface_sensitivities is not None:
            sens_file = os.path.join(output_dir, "surface_sensitivities.csv")
            self.surface_sensitivities.to_csv(sens_file, index=False)
            logger.info(f"Exported surface sensitivities to {sens_file}")
        
        # Export control point sensitivities if available
        if self.control_point_sensitivities is not None:
            cp_file = os.path.join(output_dir, "control_point_sensitivities.npy")
            np.save(cp_file, self.control_point_sensitivities)
            logger.info(f"Exported control point sensitivities to {cp_file}")
        
        # Export FFD box if available
        if self.ffd_box is not None:
            ffd_file = os.path.join(output_dir, "optimized_ffd.3df")
            self.ffd_box.write(ffd_file)
            logger.info(f"Exported optimized FFD box to {ffd_file}")
        
        # Export optimization metadata
        import json
        meta_file = os.path.join(output_dir, "optimization_metadata.json")
        metadata = {
            "objective_type": self._settings["objective_type"],
            "objective_value": self.objective_value,
            "step_size": self._settings["step_size"],
            "target_patches": self._settings["target_patches"],
        }
        
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Exported optimization metadata to {meta_file}")


def create_sonic_adjoint_interface(
    case_dir: str,
    ffd_file: str,
    settings: Optional[Dict[str, Any]] = None
) -> SonicAdjointInterface:
    """Create a complete SonicAdjointInterface with all required components.
    
    Args:
        case_dir: Path to the OpenFOAM case directory
        ffd_file: Path to the FFD box file
        settings: Additional settings for the interface
        
    Returns:
        SonicAdjointInterface: Fully configured interface for optimization
    """
    # Create the FFD box
    ffd_box = FFDBox.from_file(ffd_file)
    
    # Create the mesh adapter
    mesh_adapter = OpenFOAMMeshAdapter(case_dir)
    
    # Create the sensitivity mapper
    sensitivity_mapper = SensitivityMapper(ffd_box)
    
    # Create and return the interface
    return SonicAdjointInterface(
        case_dir=case_dir,
        mesh_adapter=mesh_adapter,
        sensitivity_mapper=sensitivity_mapper,
        ffd_box=ffd_box,
        settings=settings
    )
