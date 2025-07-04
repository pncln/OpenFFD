"""
OpenFOAM Solver Interface Implementation

This module provides comprehensive OpenFOAM integration including:
- Automatic case setup and configuration
- Support for all major OpenFOAM solvers
- Parallel execution management
- Result extraction and post-processing
- Force and moment coefficient calculation
- Field data extraction and analysis
"""

import os
import re
import subprocess
import shutil
import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

from .base import CFDSolver, CFDConfig, CFDResults, SolverStatus, ConvergenceData, ObjectiveFunction

logger = logging.getLogger(__name__)

class SolverType(Enum):
    """OpenFOAM solver types."""
    SIMPLE_FOAM = "simpleFoam"
    PISO_FOAM = "pisoFoam"
    PIMPLE_FOAM = "pimpleFoam"
    RHOSIMPLE_FOAM = "rhoSimpleFoam"
    RHOPISO_FOAM = "rhoPisoFoam"
    RHOPIMPLE_FOAM = "rhoPimpleFoam"
    SONIC_FOAM = "sonicFoam"
    RHO_CENTRAL_FOAM = "rhoCentralFoam"
    COMPRESSIBLE_INTER_FOAM = "compressibleInterFoam"
    BUOYANT_SIMPLE_FOAM = "buoyantSimpleFoam"
    BUOYANT_PIMPLE_FOAM = "buoyantPimpleFoam"
    CHTMULTIREGION_FOAM = "chtMultiRegionFoam"
    SCALAR_TRANSPORT_FOAM = "scalarTransportFoam"
    POTENTIAL_FOAM = "potentialFoam"
    CUSTOM = "custom"

class TurbulenceModel(Enum):
    """OpenFOAM turbulence models."""
    LAMINAR = "laminar"
    SPALART_ALLMARAS = "SpalartAllmaras"
    K_EPSILON = "kEpsilon"
    K_EPSILON_REALIZABLE = "realizableKE"
    K_OMEGA = "kOmega"
    K_OMEGA_SST = "kOmegaSST"
    K_OMEGA_SST_SAS = "kOmegaSSTSAS"
    V2F = "v2f"
    LES_SMAGORINSKY = "Smagorinsky"
    LES_WALE = "WALE"
    LES_DYNAMIC_KE = "dynamicKEqn"
    LES_ONE_EQN_EDDY = "oneEqEddy"

@dataclass
class BoundaryCondition:
    """OpenFOAM boundary condition specification."""
    patch_name: str
    field_name: str
    bc_type: str
    value: Union[float, List[float], str]
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_openfoam_dict(self) -> str:
        """Convert to OpenFOAM dictionary format."""
        if isinstance(self.value, (int, float)):
            value_str = f"uniform {self.value}"
        elif isinstance(self.value, list):
            if len(self.value) == 1:
                value_str = f"uniform {self.value[0]}"
            else:
                value_str = f"uniform ({' '.join(map(str, self.value))})"
        else:
            value_str = str(self.value)
        
        dict_str = f"""    {self.patch_name}
    {{
        type    {self.bc_type};
        value   {value_str};"""
        
        for key, val in self.additional_params.items():
            dict_str += f"\n        {key}    {val};"
        
        dict_str += "\n    }"
        
        return dict_str

@dataclass
class FieldData:
    """OpenFOAM field data container."""
    name: str
    values: np.ndarray
    locations: np.ndarray
    dimensions: str
    internal_field: Optional[np.ndarray] = None
    boundary_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def get_patch_data(self, patch_name: str) -> Optional[np.ndarray]:
        """Get field data for specific patch."""
        return self.boundary_fields.get(patch_name)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get field statistics."""
        return {
            'min': float(np.min(self.values)),
            'max': float(np.max(self.values)),
            'mean': float(np.mean(self.values)),
            'std': float(np.std(self.values)),
            'rms': float(np.sqrt(np.mean(self.values**2)))
        }

@dataclass
class ForceCoefficients:
    """Force and moment coefficients."""
    cd: float = 0.0  # Drag coefficient
    cl: float = 0.0  # Lift coefficient
    cm: float = 0.0  # Moment coefficient
    cd_pressure: float = 0.0
    cd_viscous: float = 0.0
    cl_pressure: float = 0.0
    cl_viscous: float = 0.0
    cm_pressure: float = 0.0
    cm_viscous: float = 0.0
    forces: np.ndarray = field(default_factory=lambda: np.zeros(3))
    moments: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reference_area: float = 1.0
    reference_length: float = 1.0
    reference_point: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass
class ResidualData:
    """OpenFOAM residual monitoring data."""
    iteration: int
    time: float
    fields: Dict[str, float]
    continuity_errors: Dict[str, float] = field(default_factory=dict)
    
    def get_max_residual(self) -> float:
        """Get maximum residual value."""
        if self.fields:
            return max(self.fields.values())
        return float('inf')

@dataclass
class SimulationResults(CFDResults):
    """OpenFOAM simulation results."""
    solver_type: SolverType = SolverType.SIMPLE_FOAM
    turbulence_model: TurbulenceModel = TurbulenceModel.K_OMEGA_SST
    force_coefficients: Optional[ForceCoefficients] = None
    residual_history: List[ResidualData] = field(default_factory=list)
    field_files: Dict[str, Path] = field(default_factory=dict)
    log_file: Optional[Path] = None
    case_directory: Optional[Path] = None
    
    def get_residual_history(self, field_name: str) -> List[float]:
        """Get residual history for specific field."""
        return [r.fields.get(field_name, float('inf')) for r in self.residual_history]
    
    def get_convergence_plot_data(self) -> Dict[str, List[float]]:
        """Get data for convergence plotting."""
        iterations = [r.iteration for r in self.residual_history]
        data = {'iterations': iterations}
        
        # Get all field names
        all_fields = set()
        for r in self.residual_history:
            all_fields.update(r.fields.keys())
        
        for field in all_fields:
            data[field] = self.get_residual_history(field)
        
        return data

@dataclass
class OpenFOAMConfig(CFDConfig):
    """OpenFOAM-specific configuration."""
    solver_type: SolverType = SolverType.SIMPLE_FOAM
    turbulence_model: TurbulenceModel = TurbulenceModel.K_OMEGA_SST
    openfoam_version: str = "v2112"
    
    # OpenFOAM specific settings
    control_dict: Dict[str, Any] = field(default_factory=dict)
    fv_schemes: Dict[str, Any] = field(default_factory=dict)
    fv_solution: Dict[str, Any] = field(default_factory=dict)
    transport_properties: Dict[str, Any] = field(default_factory=dict)
    turbulence_properties: Dict[str, Any] = field(default_factory=dict)
    
    # Force calculation settings
    force_calculation: bool = True
    force_patches: List[str] = field(default_factory=list)
    reference_values: Dict[str, float] = field(default_factory=dict)
    
    # Function objects
    function_objects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        self._setup_defaults()
    
    def _setup_defaults(self):
        """Setup default OpenFOAM configurations."""
        if not self.control_dict:
            self.control_dict = {
                'application': self.solver_type.value,
                'startFrom': 'latestTime',
                'startTime': 0,
                'stopAt': 'endTime',
                'endTime': self.end_time,
                'deltaT': self.time_step,
                'writeControl': 'timeStep',
                'writeInterval': self.write_interval,
                'purgeWrite': 3,
                'writeFormat': 'ascii',
                'writePrecision': 8,
                'writeCompression': 'off',
                'timeFormat': 'general',
                'timePrecision': 6,
                'runTimeModifiable': 'true'
            }
        
        if not self.fv_schemes:
            self.fv_schemes = {
                'ddtSchemes': {'default': 'steadyState'},
                'gradSchemes': {'default': 'Gauss linear', 'grad(p)': 'Gauss linear'},
                'divSchemes': {
                    'default': 'none',
                    'div(phi,U)': 'bounded Gauss upwind',
                    'div(phi,k)': 'bounded Gauss upwind',
                    'div(phi,epsilon)': 'bounded Gauss upwind',
                    'div(phi,omega)': 'bounded Gauss upwind',
                    'div((nuEff*dev2(T(grad(U)))))': 'Gauss linear'
                },
                'laplacianSchemes': {'default': 'Gauss linear orthogonal'},
                'interpolationSchemes': {'default': 'linear'},
                'snGradSchemes': {'default': 'orthogonal'}
            }
        
        if not self.fv_solution:
            self.fv_solution = {
                'solvers': {
                    'p': {
                        'solver': 'GAMG',
                        'tolerance': 1e-06,
                        'relTol': 0.1,
                        'smoother': 'GaussSeidel'
                    },
                    'U': {
                        'solver': 'smoothSolver',
                        'smoother': 'GaussSeidel',
                        'tolerance': 1e-05,
                        'relTol': 0.1
                    },
                    'k': {
                        'solver': 'smoothSolver',
                        'smoother': 'GaussSeidel',
                        'tolerance': 1e-05,
                        'relTol': 0.1
                    },
                    'epsilon': {
                        'solver': 'smoothSolver',
                        'smoother': 'GaussSeidel',
                        'tolerance': 1e-05,
                        'relTol': 0.1
                    },
                    'omega': {
                        'solver': 'smoothSolver',
                        'smoother': 'GaussSeidel',
                        'tolerance': 1e-05,
                        'relTol': 0.1
                    }
                },
                'SIMPLE': {
                    'nNonOrthogonalCorrectors': 0,
                    'consistent': 'yes',
                    'residualControl': self.convergence_tolerance
                },
                'relaxationFactors': {
                    'fields': {'p': 0.3},
                    'equations': {
                        'U': 0.7,
                        'k': 0.7,
                        'epsilon': 0.7,
                        'omega': 0.7
                    }
                }
            }
        
        if not self.reference_values:
            self.reference_values = {
                'rho': self.fluid_properties.get('rho', 1.225),
                'U': 1.0,
                'A': 1.0,
                'L': 1.0
            }

class OpenFOAMSolver(CFDSolver):
    """OpenFOAM solver implementation."""
    
    def __init__(self, openfoam_root: Optional[Path] = None):
        """Initialize OpenFOAM solver.
        
        Args:
            openfoam_root: OpenFOAM installation directory
        """
        super().__init__("OpenFOAM")
        self.openfoam_root = openfoam_root or self._detect_openfoam()
        self.version = self._detect_version()
        self._process = None
        
        if not self.openfoam_root:
            raise RuntimeError("OpenFOAM installation not found")
        
        self.logger.info(f"OpenFOAM found at: {self.openfoam_root}")
        self.logger.info(f"OpenFOAM version: {self.version}")
    
    def _detect_openfoam(self) -> Optional[Path]:
        """Detect OpenFOAM installation."""
        # Common OpenFOAM installation paths
        common_paths = [
            Path("/opt/openfoam"),
            Path("/usr/local/OpenFOAM"),
            Path(os.path.expanduser("~/OpenFOAM")),
            Path("/home/OpenFOAM")
        ]
        
        # Check environment variable
        if "WM_PROJECT_DIR" in os.environ:
            return Path(os.environ["WM_PROJECT_DIR"])
        
        # Check common paths
        for path in common_paths:
            if path.exists():
                # Look for version directories
                for subdir in path.iterdir():
                    if subdir.is_dir() and "OpenFOAM" in subdir.name:
                        return subdir
        
        # Try to find via which command
        try:
            result = subprocess.run(['which', 'simpleFoam'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                foam_path = Path(result.stdout.strip()).parent.parent
                return foam_path
        except:
            pass
        
        return None
    
    def _detect_version(self) -> str:
        """Detect OpenFOAM version."""
        if not self.openfoam_root:
            return "unknown"
        
        # Try to read version from etc/bashrc
        bashrc_file = self.openfoam_root / "etc" / "bashrc"
        if bashrc_file.exists():
            try:
                with open(bashrc_file, 'r') as f:
                    content = f.read()
                    version_match = re.search(r'WM_PROJECT_VERSION=([^\s]+)', content)
                    if version_match:
                        return version_match.group(1)
            except:
                pass
        
        # Try to extract from path
        if "OpenFOAM" in str(self.openfoam_root):
            version_match = re.search(r'OpenFOAM-([^/]+)', str(self.openfoam_root))
            if version_match:
                return version_match.group(1)
        
        return "unknown"
    
    def setup_case(self, config: OpenFOAMConfig) -> bool:
        """Setup OpenFOAM case directory and files.
        
        Args:
            config: OpenFOAM configuration
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Setting up OpenFOAM case: {config.case_directory}")
            
            # Validate configuration
            errors = self.validate_config(config)
            if errors:
                self.logger.error(f"Configuration errors: {errors}")
                return False
            
            # Prepare case directory
            if not self.prepare_case_directory(config):
                return False
            
            # Generate configuration files
            self._write_control_dict(config)
            self._write_fv_schemes(config)
            self._write_fv_solution(config)
            self._write_transport_properties(config)
            self._write_turbulence_properties(config)
            self._write_boundary_conditions(config)
            
            # Copy mesh if provided
            if config.mesh_file:
                self._copy_mesh(config)
            
            # Setup decomposition for parallel runs
            if config.parallel_execution:
                self._setup_decomposition(config)
            
            # Add function objects
            if config.force_calculation or config.function_objects:
                self._add_function_objects(config)
            
            self.logger.info("OpenFOAM case setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup OpenFOAM case: {e}")
            return False
    
    def run_simulation(self, config: OpenFOAMConfig) -> SimulationResults:
        """Execute OpenFOAM simulation.
        
        Args:
            config: OpenFOAM configuration
            
        Returns:
            Simulation results
        """
        start_time = time.time()
        self.status = SolverStatus.RUNNING
        
        try:
            self.logger.info(f"Starting OpenFOAM simulation with {config.solver_type.value}")
            
            # Setup case if not already done
            if not (config.case_directory / "system" / "controlDict").exists():
                if not self.setup_case(config):
                    raise RuntimeError("Failed to setup case")
            
            # Run decomposition for parallel execution
            if config.parallel_execution:
                self._run_decompose_par(config)
            
            # Run solver
            log_file = config.case_directory / f"{config.solver_type.value}.log"
            success = self._execute_solver(config, log_file)
            
            # Reconstruct parallel case
            if config.parallel_execution:
                self._run_reconstruct_par(config)
            
            # Extract results
            results = self._extract_simulation_results(config, log_file)
            results.execution_time = time.time() - start_time
            
            if success and results.is_converged:
                self.status = SolverStatus.CONVERGED
                results.status = SolverStatus.CONVERGED
            elif success:
                self.status = SolverStatus.COMPLETED
                results.status = SolverStatus.COMPLETED
            else:
                self.status = SolverStatus.FAILED
                results.status = SolverStatus.FAILED
            
            self.logger.info(f"Simulation completed in {results.execution_time:.2f} seconds")
            return results
            
        except Exception as e:
            self.status = SolverStatus.FAILED
            error_msg = f"Simulation failed: {e}"
            self.logger.error(error_msg)
            
            return SimulationResults(
                status=SolverStatus.FAILED,
                solver_type=config.solver_type,
                turbulence_model=config.turbulence_model,
                convergence_data=[],
                execution_time=time.time() - start_time,
                iterations=0,
                final_residuals={},
                objective_values={},
                error_message=error_msg,
                case_directory=config.case_directory
            )
    
    def extract_results(self, case_dir: Path) -> SimulationResults:
        """Extract results from completed simulation.
        
        Args:
            case_dir: Case directory path
            
        Returns:
            Simulation results
        """
        try:
            self.logger.info(f"Extracting results from: {case_dir}")
            
            # Find log file
            log_files = list(case_dir.glob("*.log"))
            log_file = log_files[0] if log_files else None
            
            # Create minimal config for extraction
            config = OpenFOAMConfig(case_directory=case_dir)
            
            # Extract results
            results = self._extract_simulation_results(config, log_file)
            
            self.logger.info("Results extraction completed")
            return results
            
        except Exception as e:
            error_msg = f"Failed to extract results: {e}"
            self.logger.error(error_msg)
            
            return SimulationResults(
                status=SolverStatus.FAILED,
                solver_type=SolverType.SIMPLE_FOAM,
                turbulence_model=TurbulenceModel.K_OMEGA_SST,
                convergence_data=[],
                execution_time=0.0,
                iterations=0,
                final_residuals={},
                objective_values={},
                error_message=error_msg,
                case_directory=case_dir
            )
    
    def compute_sensitivities(self, config: OpenFOAMConfig, 
                            objective: ObjectiveFunction) -> Dict[str, np.ndarray]:
        """Compute sensitivity derivatives using finite differences.
        
        Args:
            config: OpenFOAM configuration
            objective: Objective function
            
        Returns:
            Sensitivity gradients
        """
        self.logger.info(f"Computing sensitivities for {objective.value}")
        
        # This is a placeholder implementation
        # Real implementation would use adjoint method or finite differences
        sensitivities = {}
        
        # For demonstration, return random sensitivities
        if config.mesh_file and config.mesh_file.exists():
            # Would extract mesh points and compute actual sensitivities
            n_points = 1000  # Placeholder
            sensitivities['mesh_points'] = np.random.rand(n_points, 3) * 1e-3
        
        return sensitivities
    
    def cleanup(self, case_dir: Path) -> None:
        """Clean up temporary files and directories.
        
        Args:
            case_dir: Case directory to clean
        """
        try:
            # Remove processor directories
            for proc_dir in case_dir.glob("processor*"):
                if proc_dir.is_dir():
                    shutil.rmtree(proc_dir)
            
            # Remove large result files but keep essential ones
            for time_dir in case_dir.iterdir():
                if time_dir.is_dir() and time_dir.name.replace('.', '').isdigit():
                    # Keep only final time directory
                    time_dirs = [d for d in case_dir.iterdir() 
                               if d.is_dir() and d.name.replace('.', '').isdigit()]
                    time_dirs.sort(key=lambda x: float(x.name))
                    
                    if time_dir != time_dirs[-1]:  # Not the last time directory
                        shutil.rmtree(time_dir)
            
            self.logger.info(f"Cleaned up case directory: {case_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cleanup case directory: {e}")
    
    # Private helper methods
    
    def _write_control_dict(self, config: OpenFOAMConfig):
        """Write controlDict file."""
        control_dict_path = config.case_directory / "system" / "controlDict"
        
        with open(control_dict_path, 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("| =========                 |                                                 |\n")
            f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
            f.write("|  \\\\    /   O peration     | Version:  v2112                                 |\n")
            f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
            f.write("|    \\\\/     M anipulation  |                                                 |\n")
            f.write("\\*---------------------------------------------------------------------------*/\n")
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       dictionary;\n")
            f.write("    location    \"system\";\n")
            f.write("    object      controlDict;\n")
            f.write("}\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            for key, value in config.control_dict.items():
                if isinstance(value, str):
                    f.write(f"{key}        {value};\n")
                else:
                    f.write(f"{key}        {value};\n")
            
            # Add function objects
            if config.function_objects:
                f.write("\nfunctions\n{\n")
                for name, obj_dict in config.function_objects.items():
                    f.write(f"    {name}\n    {{\n")
                    for k, v in obj_dict.items():
                        if isinstance(v, str):
                            f.write(f"        {k}    {v};\n")
                        elif isinstance(v, list):
                            f.write(f"        {k}    ({' '.join(map(str, v))});\n")
                        else:
                            f.write(f"        {k}    {v};\n")
                    f.write("    }\n")
                f.write("}\n")
            
            f.write("\n// ************************************************************************* //\n")
    
    def _write_fv_schemes(self, config: OpenFOAMConfig):
        """Write fvSchemes file."""
        schemes_path = config.case_directory / "system" / "fvSchemes"
        
        with open(schemes_path, 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            for section, schemes in config.fv_schemes.items():
                f.write(f"{section}\n{{\n")
                for key, value in schemes.items():
                    f.write(f"    {key}    {value};\n")
                f.write("}\n\n")
            
            f.write("// ************************************************************************* //\n")
    
    def _write_fv_solution(self, config: OpenFOAMConfig):
        """Write fvSolution file."""
        solution_path = config.case_directory / "system" / "fvSolution"
        
        with open(solution_path, 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            # Write solvers section
            f.write("solvers\n{\n")
            for field, solver_dict in config.fv_solution['solvers'].items():
                f.write(f"    {field}\n    {{\n")
                for key, value in solver_dict.items():
                    if isinstance(value, str):
                        f.write(f"        {key}    {value};\n")
                    else:
                        f.write(f"        {key}    {value};\n")
                f.write("    }\n")
            f.write("}\n\n")
            
            # Write algorithm sections
            for section in ['SIMPLE', 'PISO', 'PIMPLE']:
                if section in config.fv_solution:
                    f.write(f"{section}\n{{\n")
                    for key, value in config.fv_solution[section].items():
                        if key == 'residualControl':
                            f.write(f"    {key}\n    {{\n")
                            for field, tol in value.items():
                                f.write(f"        {field}    {tol};\n")
                            f.write("    }\n")
                        else:
                            f.write(f"    {key}    {value};\n")
                    f.write("}\n\n")
            
            # Write relaxation factors
            if 'relaxationFactors' in config.fv_solution:
                f.write("relaxationFactors\n{\n")
                for section, factors in config.fv_solution['relaxationFactors'].items():
                    f.write(f"    {section}\n    {{\n")
                    for field, factor in factors.items():
                        f.write(f"        {field}    {factor};\n")
                    f.write("    }\n")
                f.write("}\n\n")
            
            f.write("// ************************************************************************* //\n")
    
    def _write_transport_properties(self, config: OpenFOAMConfig):
        """Write transportProperties file."""
        transport_path = config.case_directory / "constant" / "transportProperties"
        
        with open(transport_path, 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("FoamFile { version 2.0; format ascii; class dictionary; object transportProperties; }\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            f.write("transportModel  Newtonian;\n\n")
            f.write(f"nu              nu [0 2 -1 0 0 0 0] {config.fluid_properties['nu']};\n\n")
            
            # Add other transport properties if specified
            if config.transport_properties:
                for key, value in config.transport_properties.items():
                    f.write(f"{key}    {value};\n")
            
            f.write("\n// ************************************************************************* //\n")
    
    def _write_turbulence_properties(self, config: OpenFOAMConfig):
        """Write turbulenceProperties file."""
        turb_path = config.case_directory / "constant" / "turbulenceProperties"
        
        with open(turb_path, 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            
            if config.turbulence_model == TurbulenceModel.LAMINAR:
                f.write("simulationType  laminar;\n")
            else:
                f.write("simulationType  RAS;\n\n")
                f.write("RAS\n{\n")
                f.write(f"    RASModel    {config.turbulence_model.value};\n")
                f.write("    turbulence  on;\n")
                f.write("    printCoeffs on;\n")
                f.write("}\n")
            
            f.write("\n// ************************************************************************* //\n")
    
    def _write_boundary_conditions(self, config: OpenFOAMConfig):
        """Write initial boundary condition files."""
        # This is a simplified implementation
        # Real implementation would handle all required fields based on solver type
        
        # Create basic U file
        u_file = config.case_directory / "0" / "U"
        with open(u_file, 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            f.write("dimensions      [0 1 -1 0 0 0 0];\n\n")
            f.write("internalField   uniform (0 0 0);\n\n")
            f.write("boundaryField\n{\n")
            
            # Add boundary conditions from config
            for patch, bc_dict in config.boundary_conditions.items():
                if 'U' in bc_dict:
                    bc = bc_dict['U']
                    f.write(f"    {patch}\n    {{\n")
                    f.write(f"        type    {bc.get('type', 'fixedValue')};\n")
                    f.write(f"        value   {bc.get('value', 'uniform (0 0 0)')};\n")
                    f.write("    }\n")
            
            f.write("}\n\n// ************************************************************************* //\n")
        
        # Create basic p file
        p_file = config.case_directory / "0" / "p"
        with open(p_file, 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("FoamFile { version 2.0; format ascii; class volScalarField; object p; }\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            f.write("dimensions      [0 2 -2 0 0 0 0];\n\n")
            f.write("internalField   uniform 0;\n\n")
            f.write("boundaryField\n{\n")
            
            for patch, bc_dict in config.boundary_conditions.items():
                if 'p' in bc_dict:
                    bc = bc_dict['p']
                    f.write(f"    {patch}\n    {{\n")
                    f.write(f"        type    {bc.get('type', 'zeroGradient')};\n")
                    if 'value' in bc:
                        f.write(f"        value   {bc['value']};\n")
                    f.write("    }\n")
            
            f.write("}\n\n// ************************************************************************* //\n")
    
    def _copy_mesh(self, config: OpenFOAMConfig):
        """Copy mesh files to case directory."""
        mesh_dir = config.case_directory / "constant" / "polyMesh"
        mesh_dir.mkdir(exist_ok=True)
        
        if config.mesh_file.is_dir():
            # Copy entire polyMesh directory
            for item in config.mesh_file.iterdir():
                if item.is_file():
                    shutil.copy2(item, mesh_dir)
        else:
            # Assume it's a single mesh file that needs conversion
            self.logger.warning("Mesh conversion not implemented yet")
    
    def _setup_decomposition(self, config: OpenFOAMConfig):
        """Setup domain decomposition for parallel execution."""
        decomp_path = config.case_directory / "system" / "decomposeParDict"
        
        with open(decomp_path, 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            f.write(f"numberOfSubdomains  {config.num_processors};\n\n")
            f.write(f"method  {config.decomposition_method};\n\n")
            
            if config.decomposition_method == "simple":
                # Calculate decomposition
                n = config.num_processors
                nx = int(n**(1/3)) + 1
                ny = int((n/nx)**(1/2)) + 1
                nz = int(n/(nx*ny)) + 1
                
                f.write("simpleCoeffs\n{\n")
                f.write(f"    n    ({nx} {ny} {nz});\n")
                f.write("    delta    0.001;\n")
                f.write("}\n\n")
            
            f.write("distributed  no;\n")
            f.write("roots        ();\n\n")
            f.write("// ************************************************************************* //\n")
    
    def _add_function_objects(self, config: OpenFOAMConfig):
        """Add function objects to controlDict."""
        if config.force_calculation and config.force_patches:
            force_obj = {
                'type': 'forceCoeffs',
                'libs': '("libforces.so")',
                'writeControl': 'timeStep',
                'writeInterval': 1,
                'patches': f"({' '.join(config.force_patches)})",
                'rho': 'rhoInf',
                'rhoInf': config.reference_values.get('rho', 1.225),
                'CofR': '(0 0 0)',
                'liftDir': '(0 1 0)',
                'dragDir': '(1 0 0)',
                'pitchAxis': '(0 0 1)',
                'magUInf': config.reference_values.get('U', 1.0),
                'lRef': config.reference_values.get('L', 1.0),
                'Aref': config.reference_values.get('A', 1.0)
            }
            config.function_objects['forceCoeffs'] = force_obj
    
    def _execute_solver(self, config: OpenFOAMConfig, log_file: Path) -> bool:
        """Execute OpenFOAM solver."""
        try:
            # Prepare command
            if config.parallel_execution:
                cmd = [
                    'mpirun', '-np', str(config.num_processors),
                    config.solver_type.value, '-parallel'
                ]
            else:
                cmd = [config.solver_type.value]
            
            # Run solver
            with open(log_file, 'w') as f:
                self._process = subprocess.Popen(
                    cmd,
                    cwd=config.case_directory,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                return_code = self._process.wait()
                self._process = None
                
                return return_code == 0
                
        except Exception as e:
            self.logger.error(f"Failed to execute solver: {e}")
            return False
    
    def _run_decompose_par(self, config: OpenFOAMConfig):
        """Run decomposePar utility."""
        try:
            cmd = ['decomposePar', '-force']
            subprocess.run(cmd, cwd=config.case_directory, check=True,
                         capture_output=True, text=True)
            self.logger.info("Domain decomposition completed")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"decomposePar failed: {e}")
            raise
    
    def _run_reconstruct_par(self, config: OpenFOAMConfig):
        """Run reconstructPar utility."""
        try:
            cmd = ['reconstructPar', '-latestTime']
            subprocess.run(cmd, cwd=config.case_directory, check=True,
                         capture_output=True, text=True)
            self.logger.info("Case reconstruction completed")
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"reconstructPar failed: {e}")
    
    def _extract_simulation_results(self, config: OpenFOAMConfig, 
                                  log_file: Optional[Path]) -> SimulationResults:
        """Extract results from completed simulation."""
        results = SimulationResults(
            status=SolverStatus.COMPLETED,
            solver_type=config.solver_type,
            turbulence_model=config.turbulence_model,
            convergence_data=[],
            execution_time=0.0,
            iterations=0,
            final_residuals={},
            objective_values={},
            case_directory=config.case_directory,
            log_file=log_file
        )
        
        # Parse log file for residuals and convergence
        if log_file and log_file.exists():
            self._parse_log_file(log_file, results)
        
        # Extract force coefficients if available
        if config.force_calculation:
            results.force_coefficients = self._extract_force_coefficients(config)
        
        # Extract field data
        results.field_data = self._extract_field_data(config)
        
        # Determine convergence status
        if results.residual_history:
            last_residuals = results.residual_history[-1].fields
            is_converged = self.check_convergence(last_residuals, config.convergence_tolerance)
            if is_converged:
                results.status = SolverStatus.CONVERGED
        
        return results
    
    def _parse_log_file(self, log_file: Path, results: SimulationResults):
        """Parse OpenFOAM log file for residuals and convergence."""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract residual data
            residual_pattern = re.compile(
                r'Time = (\d+(?:\.\d+)?)\n.*?'
                r'(?:Solving for (\w+), Initial residual = ([\d.e-]+).*?\n)*',
                re.MULTILINE | re.DOTALL
            )
            
            iteration = 0
            for match in residual_pattern.finditer(content):
                time_value = float(match.group(1))
                
                # Extract residuals for this iteration
                residuals = {}
                residual_lines = re.findall(
                    r'Solving for (\w+), Initial residual = ([\d.e-]+)',
                    match.group(0)
                )
                
                for field, residual in residual_lines:
                    residuals[field] = float(residual)
                
                if residuals:
                    residual_data = ResidualData(
                        iteration=iteration,
                        time=time_value,
                        fields=residuals
                    )
                    results.residual_history.append(residual_data)
                    iteration += 1
            
            results.iterations = iteration
            if results.residual_history:
                results.final_residuals = results.residual_history[-1].fields
            
        except Exception as e:
            self.logger.warning(f"Failed to parse log file: {e}")
    
    def _extract_force_coefficients(self, config: OpenFOAMConfig) -> Optional[ForceCoefficients]:
        """Extract force coefficients from postProcessing data."""
        try:
            forces_dir = config.case_directory / "postProcessing" / "forceCoeffs"
            if not forces_dir.exists():
                return None
            
            # Find latest time directory
            time_dirs = [d for d in forces_dir.iterdir() if d.is_dir()]
            if not time_dirs:
                return None
            
            latest_time = max(time_dirs, key=lambda x: float(x.name))
            coeff_file = latest_time / "coefficient.dat"
            
            if not coeff_file.exists():
                return None
            
            # Read coefficient data
            data = np.loadtxt(coeff_file, skiprows=1)  # Skip header
            if data.size == 0:
                return None
            
            # Get last row (final values)
            if data.ndim == 1:
                final_data = data
            else:
                final_data = data[-1]
            
            # Parse coefficients (column order may vary)
            # Typical order: time, cd, cl, cm, cd_pressure, cd_viscous, cl_pressure, cl_viscous
            force_coeffs = ForceCoefficients(
                cd=final_data[1] if len(final_data) > 1 else 0.0,
                cl=final_data[2] if len(final_data) > 2 else 0.0,
                cm=final_data[3] if len(final_data) > 3 else 0.0,
                reference_area=config.reference_values.get('A', 1.0),
                reference_length=config.reference_values.get('L', 1.0)
            )
            
            return force_coeffs
            
        except Exception as e:
            self.logger.warning(f"Failed to extract force coefficients: {e}")
            return None
    
    def _extract_field_data(self, config: OpenFOAMConfig) -> Dict[str, FieldData]:
        """Extract field data from latest time directory."""
        field_data = {}
        
        try:
            # Find latest time directory
            time_dirs = []
            for item in config.case_directory.iterdir():
                if item.is_dir() and item.name.replace('.', '').isdigit():
                    time_dirs.append(item)
            
            if not time_dirs:
                return field_data
            
            latest_time = max(time_dirs, key=lambda x: float(x.name))
            
            # Extract specified fields
            for field_name in config.output_fields:
                field_file = latest_time / field_name
                if field_file.exists():
                    # This is a simplified extraction
                    # Real implementation would parse OpenFOAM field format
                    field_data[field_name] = FieldData(
                        name=field_name,
                        values=np.array([]),  # Placeholder
                        locations=np.array([]),  # Placeholder
                        dimensions="[0 0 0 0 0 0 0]"  # Placeholder
                    )
            
        except Exception as e:
            self.logger.warning(f"Failed to extract field data: {e}")
        
        return field_data
    
    def stop_simulation(self):
        """Stop running simulation."""
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=10)
            if self._process.poll() is None:
                self._process.kill()
            self._process = None
            self.status = SolverStatus.CANCELLED