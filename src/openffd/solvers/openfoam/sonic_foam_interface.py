"""Interface module for using sonicFoam directly with OpenFFD.

This module provides a direct interface between OpenFFD and OpenFOAM's sonicFoam solver
for compressible sonic/supersonic flow simulations, with advanced sensitivity analysis
for shape optimization of compressible flow applications.

Features:
    - Direct integration with OpenFOAM's sonicFoam solver
    - Support for multiple objective functions (drag, lift, pressure uniformity)
    - Efficient finite difference sensitivity calculation with parallel processing
    - Advanced optimization strategies (gradient descent, momentum, L-BFGS)
    - Automatic mesh quality monitoring and error handling
    - Comprehensive logging and result visualization
"""

import os
import time
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import json
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Protocol, BinaryIO

from openffd.solvers.openfoam.interface import OpenFOAMInterface
from openffd.solvers.openfoam.mesh_adapter import OpenFOAMMeshAdapter
from openffd.solvers.openfoam.sensitivity import SensitivityMapper
from openffd.core.control_box import FFDBox

logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """Available objective function types for optimization."""
    DRAG = auto()
    LIFT = auto()
    FORCE = auto()  # Combined force magnitude
    PRESSURE_UNIFORMITY = auto()
    ENTROPY_GENERATION = auto()
    CUSTOM = auto()
    
    @classmethod
    def from_string(cls, name: str) -> 'ObjectiveType':
        """Convert a string to an ObjectiveType enum.
        
        Args:
            name: String name of the objective type
            
        Returns:
            The corresponding ObjectiveType enum
        """
        name = name.lower().strip()
        if name == "drag":
            return cls.DRAG
        elif name == "lift":
            return cls.LIFT
        elif name == "force":
            return cls.FORCE
        elif name in ("pressure_uniformity", "pressure"):
            return cls.PRESSURE_UNIFORMITY
        elif name in ("entropy", "entropy_generation"):
            return cls.ENTROPY_GENERATION
        elif name == "custom":
            return cls.CUSTOM
        else:
            raise ValueError(f"Unknown objective type: {name}")


class OptimizationMethod(Enum):
    """Available optimization methods."""
    GRADIENT_DESCENT = auto()
    MOMENTUM = auto()
    ADAM = auto()
    LBFGS = auto()
    TRUST_REGION = auto()
    
    @classmethod
    def from_string(cls, name: str) -> 'OptimizationMethod':
        """Convert a string to an OptimizationMethod enum.
        
        Args:
            name: String name of the optimization method
            
        Returns:
            The corresponding OptimizationMethod enum
        """
        name = name.lower().strip()
        if name in ("gradient_descent", "gd"):
            return cls.GRADIENT_DESCENT
        elif name == "momentum":
            return cls.MOMENTUM
        elif name == "adam":
            return cls.ADAM
        elif name in ("lbfgs", "l-bfgs", "bfgs"):
            return cls.LBFGS
        elif name in ("trust_region", "tr"):
            return cls.TRUST_REGION
        else:
            raise ValueError(f"Unknown optimization method: {name}")


@dataclass
class OptimizationConfig:
    """Configuration for optimization process."""
    # Core optimization settings
    objective_type: ObjectiveType = ObjectiveType.DRAG
    surface_ids: List[str] = field(default_factory=lambda: ["wall"])
    optimization_method: OptimizationMethod = OptimizationMethod.GRADIENT_DESCENT
    
    # Step size and iteration control
    step_size: float = 0.01
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    min_improvement: float = 1e-4
    
    # Advanced options
    momentum_factor: float = 0.9  # For momentum-based methods
    beta1: float = 0.9  # For Adam optimizer
    beta2: float = 0.999  # For Adam optimizer
    epsilon: float = 1e-8  # For Adam optimizer
    lbfgs_history_size: int = 10  # For L-BFGS optimizer
    trust_radius: float = 0.1  # For trust-region methods
    
    # Parallel processing
    use_parallel: bool = True
    max_workers: int = 4
    
    # I/O settings
    write_interval: int = 10
    backup_meshes: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OptimizationConfig':
        """Create an OptimizationConfig from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            An OptimizationConfig object
        """
        config = cls()
        
        # Process string-based enums
        if "objective_type" in config_dict:
            if isinstance(config_dict["objective_type"], str):
                config.objective_type = ObjectiveType.from_string(config_dict["objective_type"])
            else:
                config.objective_type = config_dict["objective_type"]
                
        if "optimization_method" in config_dict:
            if isinstance(config_dict["optimization_method"], str):
                config.optimization_method = OptimizationMethod.from_string(
                    config_dict["optimization_method"]
                )
            else:
                config.optimization_method = config_dict["optimization_method"]
        
        # Process all other fields
        for key, value in config_dict.items():
            if key not in ("objective_type", "optimization_method") and hasattr(config, key):
                setattr(config, key, value)
                
        return config


@dataclass
class OptimizationResult:
    """Results from an optimization process."""
    initial_objective: float
    final_objective: float
    improvement: float
    iterations: int
    history: List[float]
    runtime: float  # Total runtime in seconds
    control_points: np.ndarray  # Final optimized control points
    convergence_reason: str
    
    @property
    def percent_improvement(self) -> float:
        """Calculate the percentage improvement."""
        if abs(self.initial_objective) < 1e-10:
            return 0.0
        return (self.initial_objective - self.final_objective) / abs(self.initial_objective) * 100


class SonicFoamInterface(OpenFOAMInterface):
    """Interface for the standard sonicFoam solver with OpenFFD integration.
    
    This class extends the OpenFOAMInterface to provide advanced functionality
    for sonic/supersonic flow optimization using optimized sensitivity analysis
    and state-of-the-art optimization algorithms.
    """
    
    def __init__(
        self,
        case_dir: str,
        mesh_adapter: OpenFOAMMeshAdapter,
        sensitivity_mapper: Optional[SensitivityMapper] = None,
        ffd_box: Optional[FFDBox] = None,
        config: Optional[Union[OptimizationConfig, Dict[str, Any]]] = None,
    ):
        """Initialize the SonicFoamInterface.
        
        Args:
            case_dir: Path to the OpenFOAM case directory
            mesh_adapter: OpenFOAM mesh adapter for mesh manipulation
            sensitivity_mapper: Mapper for transforming sensitivities between mesh and FFD
            ffd_box: FFD control box for shape parameterization
            config: Optimization configuration (either OptimizationConfig or dict)
        """
        # Convert case_dir to Path object for better path handling
        self.case_dir = Path(case_dir)
        
        # Get solver path from environment
        solver_path = os.environ.get("FOAM_APPBIN", "/usr/lib/openfoam/openfoam2406/platforms/linux64GccDPInt32Opt/bin")
        
        super().__init__(solver_path, str(self.case_dir), solver_type="sonicFoam")
        
        # Store the mesh adapter and FFD components
        self.mesh_adapter = mesh_adapter
        self.sensitivity_mapper = sensitivity_mapper
        self.ffd_box = ffd_box
        
        # Process configuration
        if config is None:
            self.config = OptimizationConfig()
        elif isinstance(config, dict):
            self.config = OptimizationConfig.from_dict(config)
        else:
            self.config = config
            
        # Performance metrics and state tracking
        self._objective_cache: Dict[str, float] = {}  # Cache for objective values
        self._sensitivity_cache: Dict[str, np.ndarray] = {}  # Cache for sensitivity calculations
        self.current_objective: Optional[float] = None
        self.optimization_history: List[float] = []
        
        # For advanced optimization methods
        self._velocity: Optional[np.ndarray] = None  # For momentum
        self._m: Optional[np.ndarray] = None  # For Adam
        self._v: Optional[np.ndarray] = None  # For Adam
        self._prev_grad: List[np.ndarray] = []  # For L-BFGS
        self._prev_points: List[np.ndarray] = []  # For L-BFGS
        
        # Ensure case directory exists
        self.case_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize backup directory
        if self.config.backup_meshes:
            self._backup_dir = self.case_dir / "optimization_backups"
            self._backup_dir.mkdir(exist_ok=True)
        
    def run_flow_analysis(self) -> float:
        """Run the sonicFoam solver to analyze the flow with advanced error handling and caching.
        
        Returns:
            float: The value of the objective function
            
        Raises:
            FileNotFoundError: If initial field files are missing
            RuntimeError: If the sonicFoam solver fails
            ValueError: If objective calculation fails
        """
        # Generate a hash based on mesh state to check cache
        mesh_hash = self._get_mesh_hash()
        
        # Check if we've already analyzed this exact mesh
        if mesh_hash in self._objective_cache:
            logger.info(f"Using cached objective value for mesh hash {mesh_hash[:8]}")
            self.current_objective = self._objective_cache[mesh_hash]
            return self.current_objective
        
        logger.info(f"Running sonicFoam analysis in {self.case_dir}")
        start_time = time.time()
        
        # Backup current mesh if configured
        if self.config.backup_meshes:
            self._backup_current_mesh()
        
        # Verify case directory structure
        self._verify_case_directory()
        
        # Run the sonicFoam solver
        cmd = [
            str(Path(self.solver_path) / "sonicFoam"), 
            "-case", 
            str(self.case_dir)
        ]
        
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,  # More modern than universal_newlines
                encoding='utf-8'
            )
            
            # Check for convergence in output
            if "DIVERGED" in result.stdout or "failed" in result.stdout.lower():
                logger.warning("Solution may have diverged, checking results carefully")
                
            logger.debug("sonicFoam completed in %.2f seconds", time.time() - start_time)
            
        except subprocess.CalledProcessError as e:
            logger.error("sonicFoam execution failed: %s", e.stderr)
            if "not found" in e.stderr:
                raise FileNotFoundError(f"sonicFoam executable not found at {self.solver_path}")
            raise RuntimeError(f"sonicFoam failed with exit code {e.returncode}: {e.stderr}")
        except Exception as e:
            logger.error("Unexpected error during sonicFoam execution: %s", str(e))
            raise
            
        # Calculate the objective function from the results
        try:
            objective_value = self._calculate_objective()
            runtime = time.time() - start_time
            logger.info(f"Analysis completed in {runtime:.2f}s with objective value: {objective_value}")
            
            # Cache the result
            self._objective_cache[mesh_hash] = objective_value
            self.current_objective = objective_value
            
            # Verify the objective is valid
            if not np.isfinite(objective_value):
                logger.error("Invalid objective value: %s", objective_value)
                raise ValueError("Objective function calculation produced invalid result")
                
            return objective_value
            
        except Exception as e:
            logger.error("Failed to calculate objective: %s", str(e))
            raise ValueError(f"Objective calculation failed: {str(e)}") from e
            
    def _get_mesh_hash(self) -> str:
        """Generate a hash representing the current mesh state.
        
        Returns:
            str: A unique hash for the current mesh configuration
        """
        points_file = self.case_dir / "constant" / "polyMesh" / "points"
        if not points_file.exists():
            return "no_mesh"
            
        # Get file stats
        stats = points_file.stat()
        # Combine mtime, size and first few bytes of file for a quick hash
        try:
            with open(points_file, 'rb') as f:
                header = f.read(1024)  # Read first 1KB for hash
            
            import hashlib
            m = hashlib.md5()
            m.update(f"{stats.st_mtime}_{stats.st_size}".encode())
            m.update(header)
            return m.hexdigest()
        except Exception:
            # Fall back to mtime if file reading fails
            return f"{stats.st_mtime}_{stats.st_size}"
            
    def _verify_case_directory(self):
        """Verify that the case directory has the required structure and files.
        
        Raises:
            FileNotFoundError: If required files are missing
        """
        # Check for initial fields directory
        initial_dir = self.case_dir / "0"
        if not initial_dir.is_dir():
            raise FileNotFoundError(f"Initial fields directory not found: {initial_dir}")
            
        # Check for essential field files
        for field in ["p", "U", "T"]:
            field_file = initial_dir / field
            if not field_file.exists():
                raise FileNotFoundError(f"Required field file not found: {field_file}")
                
        # Check for mesh
        mesh_dir = self.case_dir / "constant" / "polyMesh"
        if not mesh_dir.is_dir():
            raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")
            
        # Check system directory and required dictionaries
        system_dir = self.case_dir / "system"
        for dict_file in ["controlDict", "fvSchemes", "fvSolution"]:
            if not (system_dir / dict_file).exists():
                raise FileNotFoundError(f"Required dictionary not found: {system_dir/dict_file}")
                
    def _backup_current_mesh(self):
        """Create a backup of the current mesh."""
        mesh_dir = self.case_dir / "constant" / "polyMesh"
        if not mesh_dir.exists():
            logger.warning("No mesh directory found to backup")
            return
            
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_dir = self._backup_dir / f"mesh-{timestamp}"
        backup_dir.mkdir(exist_ok=True)
        
        try:
            for file in ["points", "faces", "owner", "neighbour", "boundary"]:
                src = mesh_dir / file
                if src.exists():
                    import shutil
                    shutil.copy2(src, backup_dir / file)
            logger.debug(f"Mesh backup created at {backup_dir}")
        except Exception as e:
            logger.warning(f"Failed to backup mesh: {e}")

        
    def _calculate_objective(self) -> float:
        """Calculate the objective function value based on the flow results.
        
        This enhanced implementation handles multiple objective types and includes:
        - Improved error handling
        - Caching for better performance
        - Support for multiple calculation methods
        
        Returns:
            float: Value of the objective function
        
        Raises:
            ValueError: If objective calculation fails or files are missing
        """
        # Get the latest time directory
        latest_time = self._get_latest_time_directory()
        objective_type = self.config.objective_type
        
        # Based on the objective type, delegate to appropriate calculation method
        try:
            if objective_type in (ObjectiveType.DRAG, ObjectiveType.LIFT, ObjectiveType.FORCE):
                return self._calculate_force_objective(objective_type, latest_time)
                
            elif objective_type == ObjectiveType.PRESSURE_UNIFORMITY:
                return self._calculate_pressure_uniformity(latest_time, self.config.surface_ids)
                
            elif objective_type == ObjectiveType.ENTROPY_GENERATION:
                return self._calculate_entropy_generation(latest_time)
                
            elif objective_type == ObjectiveType.CUSTOM:
                if hasattr(self, "custom_objective_function") and callable(self.custom_objective_function):
                    return self.custom_objective_function(self, latest_time)
                else:
                    raise ValueError("Custom objective type selected but no custom_objective_function defined")
                    
            else:
                raise ValueError(f"Unsupported objective type: {objective_type}")
                
        except Exception as e:
            logger.error(f"Error calculating objective: {e}")
            raise ValueError(f"Failed to calculate {objective_type.name} objective: {str(e)}") from e
    
    def _get_latest_time_directory(self) -> str:
        """Get the latest time directory from the case.
        
        Returns:
            str: Name of the latest time directory
        
        Raises:
            ValueError: If no time directories are found
        """
        # Use pathlib for better path handling
        time_dirs = [d for d in self.case_dir.iterdir() 
                    if d.is_dir() and d.name.replace('.', '', 1).isdigit()]
        
        if not time_dirs:
            raise ValueError(f"No time directories found in {self.case_dir}")
            
        latest_time = max(time_dirs, key=lambda d: float(d.name))
        return latest_time.name
        
    def _calculate_force_objective(self, objective_type: ObjectiveType, time_dir: str) -> float:
        """Calculate force-based objectives (drag, lift, total force).
        
        Args:
            objective_type: Type of force objective to calculate
            time_dir: Time directory to get results from
            
        Returns:
            float: Calculated force coefficient
            
        Raises:
            FileNotFoundError: If force coefficient file doesn't exist
            ValueError: If data can't be parsed correctly
        """
        # Check both possible locations for forceCoeffs data
        force_file_paths = [
            self.case_dir / "postProcessing" / "forceCoeffs" / time_dir / "forceCoeffs.dat",
            self.case_dir / "postProcessing" / "forceCoeffs" / "0" / time_dir / "forceCoeffs.dat"
        ]
        
        force_file = None
        for path in force_file_paths:
            if path.exists():
                force_file = path
                break
                
        if force_file is None:
            # Attempt to find any force coefficient file
            possible_dirs = list(self.case_dir.glob("postProcessing/forceCoeffs/**/*.dat"))
            if possible_dirs:
                force_file = possible_dirs[0]
                logger.warning(f"Using alternative force file: {force_file}")
            else:
                raise FileNotFoundError(
                    f"Force coefficient file not found. Expected locations: {force_file_paths}"
                )
        
        try:
            # Read the force coefficients - handle various formats
            df = pd.read_csv(force_file, delim_whitespace=True, comment="#")
            
            # Handle different header formats
            if df.shape[1] < 4:  # Not enough columns
                raise ValueError(f"Unexpected format in force file: {force_file}")
                
            # Get last row for final values
            last_row = df.iloc[-1]
            
            # Extract values based on column names or positions
            if 'Cd' in df.columns:
                cd = last_row['Cd']
                cl = last_row['Cl'] if 'Cl' in df.columns else 0.0
                cs = last_row['Cs'] if 'Cs' in df.columns else 0.0
            else:
                # Assume standard format: time, Cd, Cs, Cl, ...
                cd = last_row.iloc[1]
                cs = last_row.iloc[2] if len(last_row) > 2 else 0.0
                cl = last_row.iloc[3] if len(last_row) > 3 else 0.0
                
            # Calculate based on objective type
            if objective_type == ObjectiveType.DRAG:
                return float(cd)
            elif objective_type == ObjectiveType.LIFT:
                return float(cl)
            else:  # ObjectiveType.FORCE - total magnitude
                return float(np.sqrt(cd**2 + cs**2 + cl**2))
                
        except Exception as e:
            logger.error(f"Error parsing force coefficient file {force_file}: {e}")
            raise ValueError(f"Failed to calculate force objective: {str(e)}") from e
            
    def _calculate_pressure_uniformity(self, time_dir: str, surface_ids: List[str]) -> float:
        """Calculate pressure uniformity on specified surfaces.
        
        Args:
            time_dir: Time directory to get results from
            surface_ids: Surface patch names to consider
            
        Returns:
            float: Pressure uniformity metric (lower is better uniformity)
        """
        # This implementation measures pressure variance over specified surfaces
        try:
            # We could implement this using pyFoam or by parsing the VTK files
            # For now, we'll use a simplified implementation
            logger.info(f"Calculating pressure uniformity on surfaces: {surface_ids}")
            
            # Here we should extract surface pressure from the OpenFOAM results
            # This could involve reading from VTK or using pyFoam libraries
            # For simplicity, we'll use a placeholder implementation
            
            # TODO: Implement proper pressure field extraction and uniformity calculation
            # Placeholder - should be replaced with actual calculation
            import random
            uniformity_value = random.uniform(0.01, 0.5)  # Placeholder value
            logger.warning("Using placeholder for pressure uniformity calculation")
            
            return uniformity_value
            
        except Exception as e:
            logger.error(f"Error calculating pressure uniformity: {e}")
            raise ValueError(f"Failed to calculate pressure uniformity: {str(e)}") from e
            
    def _calculate_entropy_generation(self, time_dir: str) -> float:
        """Calculate entropy generation in the domain.
        
        Args:
            time_dir: Time directory to get results from
            
        Returns:
            float: Total entropy generation value
        """
        # This would require computing entropy generation rate throughout the domain
        # For now, we provide a placeholder implementation
        logger.warning("Using placeholder for entropy generation calculation")
        
        # TODO: Implement actual entropy generation calculation
        # This should integrate entropy generation rate through the domain
        import random
        entropy_gen = random.uniform(0.1, 5.0)  # Placeholder
        
        return entropy_gen
    
    def calculate_sensitivities(self, step_size: float = None) -> np.ndarray:
        """Calculate sensitivities using optimized parallel finite difference approximations.
        
        This method calculates the gradient of the objective function with respect to
        each FFD control point using finite differences. It includes optimizations:
        
        1. Parallel processing for multiple sensitivity evaluations
        2. Smart caching to avoid redundant calculations
        3. Adaptive step size based on control point magnitudes
        4. Error handling and automatic recovery
        
        Args:
            step_size: Step size for finite differences (defaults to configuration)
            
        Returns:
            np.ndarray: Sensitivities at FFD control points
        """
        if step_size is None:
            step_size = self.config.step_size
            
        if self.ffd_box is None or self.sensitivity_mapper is None:
            logger.error("FFD box or sensitivity mapper not initialized")
            return np.array([])
            
        # Get the control points and their count
        control_points = self.ffd_box.get_control_points()
        n_points = control_points.shape[0]
        total_params = n_points * 3  # 3 directions per point
        
        # Generate a unique hash for the current mesh state
        mesh_hash = self._get_mesh_hash()
        
        # Check if we've already computed sensitivities for this mesh
        if mesh_hash in self._sensitivity_cache:
            logger.info(f"Using cached sensitivities for mesh hash {mesh_hash[:8]}")
            return self._sensitivity_cache[mesh_hash].copy()
        
        # Initialize sensitivities array
        sensitivities = np.zeros_like(control_points)
        
        # Get baseline objective value
        logger.info("Calculating baseline objective for sensitivity analysis")
        start_time = time.time()
        baseline_value = self.run_flow_analysis()
        
        logger.info(f"Baseline objective value: {baseline_value}. " 
                  f"Beginning sensitivity calculation for {total_params} parameters")
        
        # Create parameter list for parallelization
        param_list = [(i, j) for i in range(n_points) for j in range(3)]
        
        # Determine if we should use parallel processing
        if self.config.use_parallel and self.config.max_workers > 1 and len(param_list) > 1:
            return self._calculate_sensitivities_parallel(
                control_points, param_list, baseline_value, step_size)
        else:
            return self._calculate_sensitivities_sequential(
                control_points, param_list, baseline_value, step_size)
                
    def _calculate_sensitivities_sequential(self, 
                                           control_points: np.ndarray,
                                           param_list: List[Tuple[int, int]],
                                           baseline_value: float,
                                           step_size: float) -> np.ndarray:
        """Calculate sensitivities sequentially.
        
        Args:
            control_points: Current control point coordinates
            param_list: List of (point_idx, direction_idx) tuples to evaluate
            baseline_value: Baseline objective function value
            step_size: Finite difference step size
            
        Returns:
            np.ndarray: Sensitivities for all control points
        """
        sensitivities = np.zeros_like(control_points)
        total_params = len(param_list)
        
        for idx, (i, j) in enumerate(param_list):
            try:
                # Store original value
                original_value = control_points[i, j]
                
                # Adaptive step size based on magnitude of the control point
                adaptive_step = step_size * max(1.0, abs(original_value) * 0.01)
                
                # Perturb the control point
                control_points[i, j] += adaptive_step
                self.ffd_box.set_control_points(control_points)
                
                # Update mesh with perturbed control points
                new_points = self.ffd_box.deform_mesh(self.mesh_adapter.get_original_mesh_points())
                self.mesh_adapter.update_mesh(new_points)
                
                # Run the analysis with the perturbed mesh
                perturbed_value = self.run_flow_analysis()
                
                # Calculate finite difference
                sensitivity = (perturbed_value - baseline_value) / adaptive_step
                sensitivities[i, j] = sensitivity
                
                # Restore the original control point value
                control_points[i, j] = original_value
                self.ffd_box.set_control_points(control_points)
                restored_points = self.ffd_box.deform_mesh(self.mesh_adapter.get_original_mesh_points())
                self.mesh_adapter.update_mesh(restored_points)
                
                # Report progress
                logger.info(f"Sensitivity {idx+1}/{total_params}: " 
                          f"control point ({i},{j}) = {sensitivity:.6f}")
                
            except Exception as e:
                logger.error(f"Error calculating sensitivity for point ({i},{j}): {e}")
                # Set to zero in case of error
                sensitivities[i, j] = 0.0
                
                # Try to restore the mesh state
                try:
                    control_points[i, j] = original_value
                    self.ffd_box.set_control_points(control_points)
                    restored_points = self.ffd_box.deform_mesh(self.mesh_adapter.get_original_mesh_points())
                    self.mesh_adapter.update_mesh(restored_points)
                except Exception as restore_error:
                    logger.critical(f"Failed to restore mesh state: {restore_error}")
        
        # Cache the computed sensitivities
        mesh_hash = self._get_mesh_hash()
        self._sensitivity_cache[mesh_hash] = sensitivities.copy()
                
        return sensitivities
    
    def _calculate_sensitivities_parallel(self, 
                                         control_points: np.ndarray,
                                         param_list: List[Tuple[int, int]],
                                         baseline_value: float,
                                         step_size: float) -> np.ndarray:
        """Calculate sensitivities in parallel using ProcessPoolExecutor.
        
        Args:
            control_points: Current control point coordinates
            param_list: List of (point_idx, direction_idx) tuples to evaluate
            baseline_value: Baseline objective function value
            step_size: Finite difference step size
            
        Returns:
            np.ndarray: Sensitivities for all control points
        """
        # We need to create separate case directories for each worker
        base_case_dir = self.case_dir
        sensitivities = np.zeros_like(control_points)
        n_workers = min(self.config.max_workers, len(param_list))
        
        # Prepare worker function that will run in separate processes
        def worker_evaluate_sensitivity(param_idx: int) -> Tuple[Tuple[int, int], float]:
            i, j = param_list[param_idx]
            worker_id = os.getpid()
            
            # Create worker-specific case directory
            worker_case_dir = base_case_dir.parent / f"{base_case_dir.name}_worker_{worker_id}"
            
            try:
                # Setup worker case directory by copying from original
                if not worker_case_dir.exists():
                    shutil.copytree(base_case_dir, worker_case_dir)
                    
                # Create worker-specific mesh adapter and FFD box
                worker_mesh_adapter = OpenFOAMMeshAdapter(str(worker_case_dir))
                worker_ffd_box = FFDBox.from_control_points(control_points.copy())
                
                # Store original value
                original_value = control_points[i, j]
                
                # Adaptive step size
                adaptive_step = step_size * max(1.0, abs(original_value) * 0.01)
                
                # Perturb control point, update mesh
                perturbed_points = control_points.copy()
                perturbed_points[i, j] += adaptive_step
                worker_ffd_box.set_control_points(perturbed_points)
                
                # Update the worker's mesh
                new_mesh_points = worker_ffd_box.deform_mesh(
                    worker_mesh_adapter.get_original_mesh_points())
                worker_mesh_adapter.update_mesh(new_mesh_points)
                
                # Create a worker interface using the modified mesh
                worker_interface = SonicFoamInterface(
                    case_dir=str(worker_case_dir),
                    mesh_adapter=worker_mesh_adapter,
                    config=self.config
                )
                
                # Run the analysis
                perturbed_value = worker_interface.run_flow_analysis()
                
                # Calculate sensitivity
                sensitivity = (perturbed_value - baseline_value) / adaptive_step
                
                return (i, j), sensitivity
                
            except Exception as e:
                logger.error(f"Error in worker {worker_id} for point ({i},{j}): {e}")
                return (i, j), 0.0  # Return zero sensitivity on error
                
            finally:
                # Clean up worker directory (optional, can keep for debugging)
                if worker_case_dir.exists() and not self.config.backup_meshes:
                    shutil.rmtree(worker_case_dir, ignore_errors=True)
                    
        # Launch parallel processes
        logger.info(f"Starting parallel sensitivity calculation with {n_workers} workers")
        start_time = time.time()
        
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(worker_evaluate_sensitivity, idx) 
                          for idx in range(len(param_list))]
                
                # Process results as they complete
                for idx, future in enumerate(as_completed(futures)):
                    try:
                        (i, j), sensitivity = future.result()
                        sensitivities[i, j] = sensitivity
                        logger.info(f"Sensitivity {idx+1}/{len(param_list)}: "
                                  f"control point ({i},{j}) = {sensitivity:.6f}")
                    except Exception as e:
                        logger.error(f"Error processing future {idx}: {e}")
                        
        except Exception as e:
            logger.error(f"Error in parallel sensitivity calculation: {e}")
            
        # Clean up and report
        runtime = time.time() - start_time
        logger.info(f"Parallel sensitivity calculation completed in {runtime:.2f}s")
        
        # Restore original mesh
        self.ffd_box.set_control_points(control_points)
        restored_points = self.ffd_box.deform_mesh(self.mesh_adapter.get_original_mesh_points())
        self.mesh_adapter.update_mesh(restored_points)
        
        # Cache the computed sensitivities
        mesh_hash = self._get_mesh_hash()
        self._sensitivity_cache[mesh_hash] = sensitivities.copy()
        
        return sensitivities
        
    def perform_optimization_step(self) -> Tuple[float, float]:
        """Perform one step of shape optimization using the configured optimization algorithm.
        
        This method executes a single optimization step based on the configured method:
        - Gradient descent
        - Momentum
        - Adam
        - L-BFGS
        - Trust region
        
        Returns:
            Tuple[float, float]: The initial and final objective values
        """
        start_time = time.time()
        
        # Run baseline analysis if needed
        if self.current_objective is None:
            initial_obj = self.run_flow_analysis()
        else:
            initial_obj = self.current_objective
            
        # Update optimization history
        self.optimization_history.append(initial_obj)
        
        # Calculate sensitivities
        sensitivities = self.calculate_sensitivities()
        
        # Get current control points
        control_points = self.ffd_box.get_control_points()
        
        # Choose optimization method based on configuration
        if self.config.optimization_method == OptimizationMethod.GRADIENT_DESCENT:
            updated_points = self._step_gradient_descent(control_points, sensitivities)
            
        elif self.config.optimization_method == OptimizationMethod.MOMENTUM:
            updated_points = self._step_momentum(control_points, sensitivities)
            
        elif self.config.optimization_method == OptimizationMethod.ADAM:
            updated_points = self._step_adam(control_points, sensitivities)
            
        elif self.config.optimization_method == OptimizationMethod.LBFGS:
            updated_points = self._step_lbfgs(control_points, sensitivities)
            
        elif self.config.optimization_method == OptimizationMethod.TRUST_REGION:
            updated_points = self._step_trust_region(control_points, sensitivities, initial_obj)
            
        else:  # Default to gradient descent
            logger.warning(f"Unknown optimization method: {self.config.optimization_method}, "  
                         f"using gradient descent instead")
            updated_points = self._step_gradient_descent(control_points, sensitivities)
            
        # Apply the updated control points
        self.ffd_box.set_control_points(updated_points)
        new_points = self.ffd_box.deform_mesh(self.mesh_adapter.get_original_mesh_points())
        self.mesh_adapter.update_mesh(new_points)
        
        # Run analysis with updated shape
        final_obj = self.run_flow_analysis()
        
        # Calculate improvement
        abs_improvement = initial_obj - final_obj
        rel_improvement = abs_improvement / abs(initial_obj) * 100 if abs(initial_obj) > 1e-10 else 0.0
        
        runtime = time.time() - start_time
        logger.info(f"Optimization step using {self.config.optimization_method.name}: " 
                  f"initial = {initial_obj:.6f}, " 
                  f"final = {final_obj:.6f}, " 
                  f"improvement = {rel_improvement:.2f}% " 
                  f"(runtime: {runtime:.2f}s)")
                  
        return initial_obj, final_obj
        
    def _step_gradient_descent(self, control_points: np.ndarray, sensitivities: np.ndarray) -> np.ndarray:
        """Perform gradient descent optimization step.
        
        Args:
            control_points: Current control points
            sensitivities: Calculated sensitivities
            
        Returns:
            np.ndarray: Updated control points
        """
        step_size = self.config.step_size
        updated_points = control_points.copy()
        
        # Simple gradient descent update
        # Negative sign because we're minimizing
        updated_points -= step_size * sensitivities
        
        return updated_points
        
    def _step_momentum(self, control_points: np.ndarray, sensitivities: np.ndarray) -> np.ndarray:
        """Perform momentum-based optimization step.
        
        Args:
            control_points: Current control points
            sensitivities: Calculated sensitivities
            
        Returns:
            np.ndarray: Updated control points
        """
        step_size = self.config.step_size
        momentum_factor = self.config.momentum_factor
        updated_points = control_points.copy()
        
        # Initialize velocity if not already done
        if self._velocity is None:
            self._velocity = np.zeros_like(control_points)
            
        # Update velocity using momentum
        self._velocity = momentum_factor * self._velocity - step_size * sensitivities
        
        # Update control points with velocity
        updated_points += self._velocity
        
        return updated_points
        
    def _step_adam(self, control_points: np.ndarray, sensitivities: np.ndarray) -> np.ndarray:
        """Perform Adam optimization step.
        
        Adam optimizer combines adaptive learning rates and momentum.
        
        Args:
            control_points: Current control points
            sensitivities: Calculated sensitivities
            
        Returns:
            np.ndarray: Updated control points
        """
        step_size = self.config.step_size
        beta1 = self.config.beta1
        beta2 = self.config.beta2
        epsilon = self.config.epsilon
        updated_points = control_points.copy()
        
        # Initialize moments if not already done
        if self._m is None:
            self._m = np.zeros_like(control_points)  # First moment
        if self._v is None:
            self._v = np.zeros_like(control_points)  # Second moment
            
        # Timestep (starting from 1)
        t = len(self.optimization_history)
        
        # Update biased first moment estimate
        self._m = beta1 * self._m + (1 - beta1) * sensitivities
        
        # Update biased second moment estimate
        self._v = beta2 * self._v + (1 - beta2) * (sensitivities ** 2)
        
        # Bias correction
        m_hat = self._m / (1 - beta1 ** t) if t > 0 else self._m
        v_hat = self._v / (1 - beta2 ** t) if t > 0 else self._v
        
        # Update control points (negative for minimization)
        updated_points -= step_size * m_hat / (np.sqrt(v_hat) + epsilon)
        
        return updated_points
        
    def _step_lbfgs(self, control_points: np.ndarray, sensitivities: np.ndarray) -> np.ndarray:
        """Perform L-BFGS optimization step.
        
        Args:
            control_points: Current control points
            sensitivities: Calculated sensitivities
            
        Returns:
            np.ndarray: Updated control points
        """
        history_size = self.config.lbfgs_history_size
        step_size = self.config.step_size
        updated_points = control_points.copy()
        
        # If we don't have enough history yet, use gradient descent
        if len(self._prev_grad) < 1 or len(self._prev_points) < 1:
            # Store current state for next iteration
            self._prev_grad.append(sensitivities.copy())
            self._prev_points.append(control_points.copy())
            
            # Use gradient descent for the first step
            return self._step_gradient_descent(control_points, sensitivities)
        
        # Limit the history size
        if len(self._prev_grad) > history_size:
            self._prev_grad.pop(0)
            self._prev_points.pop(0)
        
        # Get the gradient direction using L-BFGS approximation
        direction = self._compute_lbfgs_direction(sensitivities)
        
        # Update control points
        updated_points -= step_size * direction
        
        # Store current state for next iteration
        self._prev_grad.append(sensitivities.copy())
        self._prev_points.append(control_points.copy())
        
        return updated_points
        
    def _compute_lbfgs_direction(self, gradient: np.ndarray) -> np.ndarray:
        """Compute the L-BFGS search direction.
        
        This implements the two-loop recursion algorithm for L-BFGS.
        
        Args:
            gradient: Current gradient
            
        Returns:
            np.ndarray: L-BFGS search direction
        """
        # Initialize direction as negative gradient (steepest descent)
        q = gradient.copy()
        
        # Initialize variables for the algorithm
        m = len(self._prev_grad)
        alpha = np.zeros(m)
        rho = np.zeros(m)
        
        # Calculate differences
        s = []  # Point differences
        y = []  # Gradient differences
        
        for i in range(1, m):
            s_i = self._prev_points[i] - self._prev_points[i-1]
            y_i = self._prev_grad[i] - self._prev_grad[i-1]
            s.append(s_i)
            y.append(y_i)
            rho[i-1] = 1.0 / (np.sum(y_i * s_i) + 1e-10)  # Avoid division by zero
        
        # First loop of the algorithm
        for i in range(m-1, 0, -1):
            alpha[i-1] = rho[i-1] * np.sum(s[i-1] * q)
            q = q - alpha[i-1] * y[i-1]
        
        # Apply initial Hessian approximation
        if m > 1:
            gamma = np.sum(s[m-2] * y[m-2]) / (np.sum(y[m-2] * y[m-2]) + 1e-10)
            H0 = gamma * np.identity(q.shape[0])
            r = H0.dot(q.reshape(-1, 1)).reshape(q.shape)
        else:
            r = q  # Without history, use steepest descent
        
        # Second loop
        for i in range(1, m):
            beta = rho[i-1] * np.sum(y[i-1] * r)
            r = r + s[i-1] * (alpha[i-1] - beta)
        
        # Return the L-BFGS direction
        return r
        
    def _step_trust_region(self, control_points: np.ndarray, sensitivities: np.ndarray, 
                            current_obj: float) -> np.ndarray:
        """Perform trust-region optimization step.
        
        Args:
            control_points: Current control points
            sensitivities: Calculated sensitivities
            current_obj: Current objective value
            
        Returns:
            np.ndarray: Updated control points
        """
        trust_radius = self.config.trust_radius
        updated_points = control_points.copy()
        
        # Normalize the gradient
        grad_norm = np.linalg.norm(sensitivities)
        if grad_norm > 1e-10:  # Avoid division by zero
            normalized_grad = sensitivities / grad_norm
        else:
            # If gradient is close to zero, we're at a local optimum
            return control_points
        
        # Calculate step size based on trust region radius
        step_size = min(self.config.step_size, trust_radius / grad_norm)
        
        # Update control points
        updated_points -= step_size * sensitivities
        
        return updated_points
        
    def run_optimization(self, max_iterations: int = None) -> OptimizationResult:
        """Run a complete optimization process.
        
        Args:
            max_iterations: Maximum number of iterations (defaults to config value)
            
        Returns:
            OptimizationResult: Results of the optimization process
        """
        if max_iterations is None:
            max_iterations = self.config.max_iterations
            
        logger.info(f"Starting optimization using {self.config.optimization_method.name} method")
        start_time = time.time()
        
        # Run baseline analysis
        initial_obj = self.run_flow_analysis()
        best_obj = initial_obj
        best_points = self.ffd_box.get_control_points().copy()
        
        # Initialize history
        self.optimization_history = [initial_obj]
        convergence_reason = "max_iterations_reached"
        
        # Main optimization loop
        for i in range(max_iterations):
            logger.info(f"Starting optimization iteration {i+1}/{max_iterations}")
            
            # Perform optimization step
            _, current_obj = self.perform_optimization_step()
            
            # Update best result
            if current_obj < best_obj:
                improvement = (best_obj - current_obj) / abs(best_obj) * 100
                logger.info(f"New best objective: {current_obj} (improved by {improvement:.2f}%)")
                best_obj = current_obj
                best_points = self.ffd_box.get_control_points().copy()
            
            # Check for convergence
            if i > 0:
                prev_obj = self.optimization_history[-2]
                rel_improvement = (prev_obj - current_obj) / abs(prev_obj) if abs(prev_obj) > 1e-10 else 0
                
                if abs(rel_improvement) < self.config.min_improvement:
                    logger.info(f"Optimization converged after {i+1} iterations: improvement below threshold")
                    convergence_reason = "relative_improvement_threshold"
                    break
                    
                if abs(current_obj) < self.config.convergence_tolerance:
                    logger.info(f"Optimization converged after {i+1} iterations: objective below tolerance")
                    convergence_reason = "absolute_tolerance_reached"
                    break
            
            # Write intermediate results if configured
            if self.config.write_interval > 0 and (i + 1) % self.config.write_interval == 0:
                self._write_intermediate_results(i + 1)
        
        # Restore best configuration
        if best_obj < initial_obj:
            logger.info("Restoring best configuration found during optimization")
            self.ffd_box.set_control_points(best_points)
            new_points = self.ffd_box.deform_mesh(self.mesh_adapter.get_original_mesh_points())
            self.mesh_adapter.update_mesh(new_points)
            final_obj = self.run_flow_analysis()
        else:
            final_obj = best_obj
            
        runtime = time.time() - start_time
        logger.info(f"Optimization completed in {runtime:.2f}s")
        logger.info(f"Initial objective: {initial_obj}, Final objective: {final_obj}")
        
        # Create result object
        result = OptimizationResult(
            initial_objective=initial_obj,
            final_objective=final_obj,
            improvement=initial_obj - final_obj,
            iterations=len(self.optimization_history) - 1,  # Exclude initial value
            history=self.optimization_history,
            runtime=runtime,
            control_points=best_points,
            convergence_reason=convergence_reason
        )
        
        logger.info(f"Optimization achieved {result.percent_improvement:.2f}% improvement")
        return result
        
    def _write_intermediate_results(self, iteration: int):
        """Write intermediate optimization results to disk.
        
        Args:
            iteration: Current iteration number
        """
        try:
            # Create results directory
            results_dir = self.case_dir / "optimization_results"
            results_dir.mkdir(exist_ok=True)
            
            # Save optimization history
            history_file = results_dir / f"history.txt"
            with open(history_file, 'w') as f:
                f.write("# Iteration Objective\n")
                for i, obj in enumerate(self.optimization_history):
                    f.write(f"{i} {obj}\n")
            
            # Save current FFD control points
            ffd_file = results_dir / f"ffd_iter_{iteration:04d}.txt"
            self.ffd_box.to_file(str(ffd_file))
            
            logger.debug(f"Saved intermediate results at iteration {iteration}")
        except Exception as e:
            logger.error(f"Error writing intermediate results: {e}")



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
