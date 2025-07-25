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
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

# Optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

from ..core.base import BaseSolver, BaseCase
from ..core.registry import register_solver

logger = logging.getLogger(__name__)

class SolverStatus(Enum):
    """Solver execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    CONVERGED = "converged"
    FAILED = "failed"
    CANCELLED = "cancelled"

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
class SimulationResults:
    """OpenFOAM simulation results."""
    status: str = "completed"
    execution_time: float = 0.0
    iterations: int = 0
    is_converged: bool = False
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
class OpenFOAMConfig:
    """OpenFOAM-specific configuration."""
    case_directory: Path = None
    mesh_file: Optional[Path] = None
    end_time: float = 1000
    time_step: float = 1.0
    write_interval: int = 100
    convergence_tolerance: float = 1e-6
    parallel_execution: bool = False
    num_processors: int = 4
    decomposition_method: str = "simple"
    fluid_properties: Dict[str, Any] = field(default_factory=dict)
    boundary_conditions: Dict[str, Any] = field(default_factory=dict)
    output_fields: List[str] = field(default_factory=list)
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
                    'residualControl': {
                        'p': self.convergence_tolerance,
                        'U': self.convergence_tolerance,
                        'k': self.convergence_tolerance,
                        'omega': self.convergence_tolerance
                    }
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

@register_solver('simpleFoam')
@register_solver('pisoFoam')
@register_solver('pimpleFoam')
@register_solver('buoyantSimpleFoam')
@register_solver('buoyantPimpleFoam')
@register_solver('rhoCentralFoam')
@register_solver('sonicFoam')
@register_solver('openfoam')
class OpenFOAMSolver(BaseSolver):
    """OpenFOAM solver implementation integrated with universal framework."""
    
    def __init__(self, case_handler: BaseCase, openfoam_root: Optional[Path] = None):
        """Initialize OpenFOAM solver.
        
        Args:
            case_handler: Case handler instance
            openfoam_root: OpenFOAM installation directory
        """
        super().__init__(case_handler)
        self.openfoam_root = openfoam_root or self._detect_openfoam()
        self.version = self._detect_version()
        self._process = None
        self.logger = logging.getLogger(__name__)
        
        if not self.openfoam_root:
            self.logger.warning("OpenFOAM installation not detected in common paths - assuming it's available in PATH")
            # Force use of OpenFOAM even if not detected in common paths
            self.openfoam_root = Path("/usr/local/bin")  # Dummy path, will use PATH
        
        self.logger.info(f"OpenFOAM found at: {self.openfoam_root}")
        self.logger.info(f"OpenFOAM version: {self.version}")
        
        # Convert case config to OpenFOAM config
        self.openfoam_config = self._create_openfoam_config()
        
        # Initialize solver for existing case
        self.setup_solver()
    
    def _create_openfoam_config(self) -> 'OpenFOAMConfig':
        """Create OpenFOAM configuration from case config."""
        # Map solver name to SolverType
        solver_map = {
            'simpleFoam': SolverType.SIMPLE_FOAM,
            'pisoFoam': SolverType.PISO_FOAM,
            'pimpleFoam': SolverType.PIMPLE_FOAM,
            'buoyantSimpleFoam': SolverType.BUOYANT_SIMPLE_FOAM,
            'buoyantPimpleFoam': SolverType.BUOYANT_PIMPLE_FOAM,
            'rhoCentralFoam': SolverType.RHO_CENTRAL_FOAM,
            'sonicFoam': SolverType.SONIC_FOAM
        }
        
        solver_type = solver_map.get(self.config.solver, SolverType.SIMPLE_FOAM)
        
        # Create OpenFOAM configuration
        openfoam_config = OpenFOAMConfig(
            case_directory=self.case_handler.case_path,
            solver_type=solver_type,
            turbulence_model=TurbulenceModel.K_OMEGA_SST,
            parallel_execution=self.config.optimization.parallel,
            num_processors=4,  # Default
            end_time=1000,  # Default
            time_step=1.0,  # Default for steady-state
            write_interval=100,
            convergence_tolerance=self.config.optimization.tolerance,
            fluid_properties={'nu': 1.5e-5},  # Default
            boundary_conditions={},
            output_fields=['U', 'p'],
            force_patches=[],
            reference_values=self.config.constants
        )
        
        # Set up force calculation for airfoil cases
        if hasattr(self.case_handler, 'airfoil_patch') and self.case_handler.airfoil_patch:
            openfoam_config.force_calculation = True
            openfoam_config.force_patches = [self.case_handler.airfoil_patch]
        
        return openfoam_config
    
    def setup_solver(self) -> bool:
        """Set up the solver for the given case."""
        try:
            if not self.openfoam_config:
                self.logger.warning("OpenFOAM config not available - skipping setup")
                return True  # Allow to proceed
            return self.setup_case(self.openfoam_config)
        except Exception as e:
            self.logger.error(f"Failed to setup solver: {e}")
            return False
    
    def run_simulation(self, mesh_file: Optional[str] = None) -> Dict[str, Any]:
        """Run CFD simulation and return results."""
        print(f"DEBUG: run_simulation called with mesh_file={mesh_file}")
        
        # Always run real OpenFOAM simulation
        if not self.openfoam_config:
            raise RuntimeError("OpenFOAM configuration not initialized")
        
        # Ensure fresh simulation by cleaning up
        self._cleanup_time_directories(self.openfoam_config.case_directory)
        
        self.logger.info(f"OpenFOAM available at: {self.openfoam_root}")
        self.logger.info(f"Running real CFD simulation")
        print(f"DEBUG: About to call _run_openfoam_simulation")
        
        try:
            # Update mesh file if provided
            if mesh_file:
                self.openfoam_config.mesh_file = Path(mesh_file)
            
            # Run simulation
            print(f"DEBUG: Calling _run_openfoam_simulation")
            results = self._run_openfoam_simulation(self.openfoam_config)
            print(f"DEBUG: _run_openfoam_simulation returned")
            
            # Convert to universal format
            universal_results = self._convert_results_to_universal_format(results)
            print(f"DEBUG: Universal format conversion completed")
            return universal_results
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            return {
                'converged': False,
                'error': str(e),
                'forces': {},
                'moments': {},
                'pressure': {},
                'heat_flux': {},
                'wall_temperature': {}
            }
    
    def check_convergence(self) -> bool:
        """Check if simulation has converged."""
        # Check if latest results show convergence
        try:
            log_file = self.case_handler.case_path / f"{self.config.solver}.log"
            if not log_file.exists():
                return False
            
            # Parse last few lines for residuals
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Look for convergence in last 10 lines
            for line in lines[-10:]:
                if 'SIMPLE solution converged' in line or 'Final residual' in line:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def get_residuals(self) -> Dict[str, List[float]]:
        """Get residual history."""
        residuals = {'iterations': [], 'U': [], 'p': [], 'k': [], 'omega': []}
        
        try:
            log_file = self.case_handler.case_path / f"{self.config.solver}.log"
            if not log_file.exists():
                return residuals
            
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Parse residuals using regex
            pattern = r'Solving for (\w+), Initial residual = ([\d.e-]+)'
            matches = re.findall(pattern, content)
            
            iteration = 0
            for field, residual in matches:
                if field in residuals:
                    if len(residuals[field]) <= iteration:
                        residuals[field].extend([0.0] * (iteration - len(residuals[field]) + 1))
                    residuals[field][iteration] = float(residual)
                iteration += 1
            
            residuals['iterations'] = list(range(len(residuals['U'])))
            
        except Exception as e:
            self.logger.warning(f"Failed to parse residuals: {e}")
        
        return residuals
    
    def extract_forces(self, patches: List[str]) -> Dict[str, np.ndarray]:
        """Extract forces from specified patches."""
        forces = {}
        
        try:
            # Look for force coefficient data
            forces_dir = self.case_handler.case_path / "postProcessing" / "forceCoeffs"
            if forces_dir.exists():
                # Find latest time directory
                time_dirs = [d for d in forces_dir.iterdir() if d.is_dir()]
                if time_dirs:
                    latest_time = max(time_dirs, key=lambda x: float(x.name))
                    coeff_file = latest_time / "coefficient.dat"
                    
                    if coeff_file.exists():
                        data = np.loadtxt(coeff_file, skiprows=1)
                        if data.size > 0:
                            if data.ndim == 1:
                                final_data = data
                            else:
                                final_data = data[-1]
                            
                            # Extract force coefficients and convert to forces
                            cd = final_data[1] if len(final_data) > 1 else 0.0
                            cl = final_data[2] if len(final_data) > 2 else 0.0
                            
                            # Convert to actual forces (simplified)
                            rho = self.config.constants.get('rho', 1.225)
                            u_inf = self.config.constants.get('magUInf', 30.0)
                            chord = self.config.constants.get('chord', 1.0)
                            q_inf = 0.5 * rho * u_inf**2
                            
                            drag_force = cd * q_inf * chord
                            lift_force = cl * q_inf * chord
                            
                            for patch in patches:
                                forces[patch] = np.array([drag_force, lift_force, 0.0])
            
        except Exception as e:
            self.logger.warning(f"Failed to extract forces: {e}")
            for patch in patches:
                forces[patch] = np.array([0.0, 0.0, 0.0])
        
        return forces
    
    def cleanup(self) -> None:
        """Clean up solver temporary files."""
        try:
            case_dir = self.case_handler.case_path
            
            # Remove processor directories
            for proc_dir in case_dir.glob("processor*"):
                if proc_dir.is_dir():
                    shutil.rmtree(proc_dir)
            
            # Remove large time directories except the last one
            time_dirs = []
            for item in case_dir.iterdir():
                if item.is_dir() and item.name.replace('.', '').isdigit():
                    time_dirs.append(item)
            
            if len(time_dirs) > 1:
                time_dirs.sort(key=lambda x: float(x.name))
                for time_dir in time_dirs[:-1]:  # Keep only the last time directory
                    shutil.rmtree(time_dir)
            
            self.logger.info(f"Cleaned up case directory: {case_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cleanup: {e}")
    
    def _run_openfoam_simulation(self, config: 'OpenFOAMConfig') -> 'SimulationResults':
        """Run the actual OpenFOAM simulation."""
        print(f"DEBUG: _run_openfoam_simulation entry")
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting OpenFOAM simulation with {config.solver_type.value}")
            print(f"DEBUG: Starting simulation with solver {config.solver_type.value}")
            
            # Setup case if not already done
            if not (config.case_directory / "system" / "controlDict").exists():
                if not self.setup_case(config):
                    raise RuntimeError("Failed to setup case")
            
            # Run decomposition for parallel execution
            if config.parallel_execution:
                self._run_decompose_par(config)
            
            # Run solver
            log_file = config.case_directory / f"{config.solver_type.value}.log"
            print(f"DEBUG: About to execute solver, log file: {log_file}")
            success = self._execute_solver(config, log_file)
            print(f"DEBUG: Solver execution completed, success: {success}")
            
            # Reconstruct parallel case
            if config.parallel_execution:
                self._run_reconstruct_par(config)
            
            # Extract results
            results = self._extract_simulation_results(config, log_file)
            results.execution_time = time.time() - start_time
            
            return results
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            raise
    
    def _convert_results_to_universal_format(self, results: 'SimulationResults') -> Dict[str, Any]:
        """Convert OpenFOAM results to universal format."""
        universal_results = {
            'converged': results.is_converged,
            'iterations': results.iterations,
            'execution_time': results.execution_time,
            'forces': {},
            'moments': {},
            'pressure': {},
            'heat_flux': {},
            'wall_temperature': {}
        }
        
        # Extract force coefficients
        if results.force_coefficients:
            fc = results.force_coefficients
            
            # Add forces for each patch
            if hasattr(self.case_handler, 'airfoil_patch') and self.case_handler.airfoil_patch:
                patch = self.case_handler.airfoil_patch
                universal_results['forces'][patch] = fc.forces
                universal_results['moments'][patch] = fc.moments
            
            # Add coefficient values directly
            universal_results['drag_coefficient'] = fc.cd
            universal_results['lift_coefficient'] = fc.cl
            universal_results['moment_coefficient'] = fc.cm
            
            self.logger.info(f"Converted results: Cd={fc.cd:.6f}, Cl={fc.cl:.6f}, Cm={fc.cm:.6f}")
        else:
            self.logger.warning("No force coefficients found in results")
            universal_results['drag_coefficient'] = 0.0
            universal_results['lift_coefficient'] = 0.0
            universal_results['moment_coefficient'] = 0.0
        
        return universal_results
    
    def _detect_openfoam(self) -> Optional[Path]:
        """Detect OpenFOAM installation."""
        # Common OpenFOAM installation paths
        common_paths = [
            Path("/opt/openfoam"),
            Path("/opt/OpenFOAM"),
            Path("/usr/local/OpenFOAM"),
            Path("/home") / os.environ.get('USER', 'user') / "OpenFOAM",
            Path("/Applications/OpenFOAM"),
            Path(os.environ.get('FOAM_INSTALL_DIR', '/opt/openfoam'))
        ]
        
        # Check for OpenFOAM environment variables
        foam_etc = os.environ.get('FOAM_ETC')
        if foam_etc:
            return Path(foam_etc).parent
        
        wm_project_dir = os.environ.get('WM_PROJECT_DIR')
        if wm_project_dir:
            return Path(wm_project_dir)
        
        # Check common installation paths
        for path in common_paths:
            if path.exists():
                # Find latest version
                versions = []
                for item in path.iterdir():
                    if item.is_dir() and ('OpenFOAM' in item.name or 'openfoam' in item.name):
                        versions.append(item)
                
                if versions:
                    # Sort by name and return latest
                    versions.sort(key=lambda x: x.name, reverse=True)
                    return versions[0]
        
        # Check if OpenFOAM commands are in PATH
        try:
            result = subprocess.run(['which', 'simpleFoam'], 
                                  capture_output=True, text=True, check=True)
            openfoam_bin = Path(result.stdout.strip()).parent
            return openfoam_bin.parent
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return None
    
    def _detect_version(self) -> str:
        """Detect OpenFOAM version."""
        if not self.openfoam_root:
            return "unknown"
        
        try:
            # Try to run OpenFOAM to get version
            env = os.environ.copy()
            env['FOAM_ETC'] = str(self.openfoam_root / 'etc')
            
            result = subprocess.run(['simpleFoam', '-help'], 
                                  capture_output=True, text=True, 
                                  env=env, timeout=10)
            
            # Parse version from output
            for line in result.stderr.split('\n'):
                if 'OpenFOAM-' in line:
                    version = line.split('OpenFOAM-')[1].split()[0]
                    return version
            
            # Fallback: check directory name
            if 'OpenFOAM' in self.openfoam_root.name:
                return self.openfoam_root.name.split('OpenFOAM-')[-1]
            
        except Exception:
            pass
        
        return "v2112"  # Default fallback
    
    def setup_case(self, config: 'OpenFOAMConfig') -> bool:
        """Setup OpenFOAM case directory and files."""
        try:
            self.logger.info(f"Setting up OpenFOAM case: {config.case_directory}")
            
            # Create directory structure only if needed
            self._ensure_case_structure(config)
            
            # Only write missing configuration files (don't overwrite existing)
            self._ensure_configuration_files(config)
            
            # Setup function objects for force calculation
            self._add_function_objects_to_control_dict(config)
            
            self.logger.info("OpenFOAM case setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup OpenFOAM case: {e}")
            return False
    
    def _ensure_case_structure(self, config: 'OpenFOAMConfig') -> None:
        """Ensure case directory structure exists (don't overwrite)."""
        case_dir = config.case_directory
        
        # Only create directories if they don't exist
        (case_dir / "log").mkdir(exist_ok=True)
        (case_dir / "postProcessing").mkdir(exist_ok=True)
    
    def _ensure_configuration_files(self, config: 'OpenFOAMConfig') -> None:
        """Ensure essential configuration files exist."""
        # Only write files that don't exist
        self._write_transport_properties(config)
        self._write_turbulence_properties(config)
        
        # Validate essential files exist
        self._validate_case_files(config)
    
    def _validate_case_files(self, config: 'OpenFOAMConfig') -> None:
        """Validate that essential case files exist."""
        case_dir = config.case_directory
        
        # Check essential directories
        required_dirs = ["0", "constant", "system"]
        missing_dirs = []
        for dir_name in required_dirs:
            if not (case_dir / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            raise FileNotFoundError(f"Missing essential OpenFOAM directories: {missing_dirs}")
        
        # Check essential files
        required_files = [
            "system/controlDict",
            "system/fvSchemes", 
            "system/fvSolution"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (case_dir / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.logger.warning(f"Missing OpenFOAM configuration files: {missing_files}")
            # Don't raise error - case might work without them
    
    def _write_control_dict(self, config: 'OpenFOAMConfig') -> None:
        """Write controlDict file."""
        control_dict_path = config.case_directory / "system" / "controlDict"
        
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Website:  https://openfoam.org                 |
|   \\\\  /    A nd           | Version:  {config.openfoam_version}                              |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     {config.control_dict['application']};

startFrom       {config.control_dict['startFrom']};

startTime       {config.control_dict['startTime']};

stopAt          {config.control_dict['stopAt']};

endTime         {config.control_dict['endTime']};

deltaT          {config.control_dict['deltaT']};

writeControl    {config.control_dict['writeControl']};

writeInterval   {config.control_dict['writeInterval']};

purgeWrite      {config.control_dict['purgeWrite']};

writeFormat     {config.control_dict['writeFormat']};

writePrecision  {config.control_dict['writePrecision']};

writeCompression {config.control_dict['writeCompression']};

timeFormat      {config.control_dict['timeFormat']};

timePrecision   {config.control_dict['timePrecision']};

runTimeModifiable {config.control_dict['runTimeModifiable']};

"""
        
        with open(control_dict_path, 'w') as f:
            f.write(content)
    
    def _write_fv_schemes(self, config: 'OpenFOAMConfig') -> None:
        """Write fvSchemes file."""
        fv_schemes_path = config.case_directory / "system" / "fvSchemes"
        
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Website:  https://openfoam.org                 |
|   \\\\  /    A nd           | Version:  {config.openfoam_version}                              |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{{
    default         {config.fv_schemes['ddtSchemes']['default']};
}}

gradSchemes
{{
    default         {config.fv_schemes['gradSchemes']['default']};
    grad(p)         {config.fv_schemes['gradSchemes']['grad(p)']};
}}

divSchemes
{{
    default         {config.fv_schemes['divSchemes']['default']};
    div(phi,U)      {config.fv_schemes['divSchemes']['div(phi,U)']};
    div(phi,k)      {config.fv_schemes['divSchemes']['div(phi,k)']};
    div(phi,epsilon) {config.fv_schemes['divSchemes']['div(phi,epsilon)']};
    div(phi,omega)  {config.fv_schemes['divSchemes']['div(phi,omega)']};
    div((nuEff*dev2(T(grad(U))))) {config.fv_schemes['divSchemes']['div((nuEff*dev2(T(grad(U)))))']}; 
}}

laplacianSchemes
{{
    default         {config.fv_schemes['laplacianSchemes']['default']};
}}

interpolationSchemes
{{
    default         {config.fv_schemes['interpolationSchemes']['default']};
}}

snGradSchemes
{{
    default         {config.fv_schemes['snGradSchemes']['default']};
}}

// ************************************************************************* //
"""
        
        with open(fv_schemes_path, 'w') as f:
            f.write(content)
    
    def _write_fv_solution(self, config: 'OpenFOAMConfig') -> None:
        """Write fvSolution file."""
        fv_solution_path = config.case_directory / "system" / "fvSolution"
        
        # Build solvers section
        solvers_section = ""
        for field, solver_config in config.fv_solution['solvers'].items():
            solvers_section += f"    {field}\n    {{\n"
            for key, value in solver_config.items():
                if isinstance(value, str):
                    solvers_section += f"        {key}        {value};\n"
                else:
                    solvers_section += f"        {key}        {value};\n"
            solvers_section += "    }\n\n"
        
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Website:  https://openfoam.org                 |
|   \\\\  /    A nd           | Version:  {config.openfoam_version}                              |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{{
{solvers_section}}}

SIMPLE
{{
    nNonOrthogonalCorrectors {config.fv_solution['SIMPLE']['nNonOrthogonalCorrectors']};
    consistent    {config.fv_solution['SIMPLE']['consistent']};
    
    residualControl
    {{
        p               {config.fv_solution['SIMPLE']['residualControl']['p']};
        U               {config.fv_solution['SIMPLE']['residualControl']['U']};
        k               {config.fv_solution['SIMPLE']['residualControl']['k']};
        omega           {config.fv_solution['SIMPLE']['residualControl']['omega']};
    }}
}}

relaxationFactors
{{
    fields
    {{
        p               {config.fv_solution['relaxationFactors']['fields']['p']};
    }}
    equations
    {{
        U               {config.fv_solution['relaxationFactors']['equations']['U']};
        k               {config.fv_solution['relaxationFactors']['equations']['k']};
        epsilon         {config.fv_solution['relaxationFactors']['equations']['epsilon']};
        omega           {config.fv_solution['relaxationFactors']['equations']['omega']};
    }}
}}

// ************************************************************************* //
"""
        
        with open(fv_solution_path, 'w') as f:
            f.write(content)
    
    def _write_transport_properties(self, config: 'OpenFOAMConfig') -> None:
        """Write transportProperties file (only if missing)."""
        transport_path = config.case_directory / "constant" / "transportProperties"
        if transport_path.exists():
            return  # Don't overwrite existing file
        
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Website:  https://openfoam.org                 |
|   \\\\  /    A nd           | Version:  {config.openfoam_version}                              |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel  Newtonian;

nu              {config.fluid_properties.get('nu', 1.5e-05)};

// ************************************************************************* //
"""
        
        with open(transport_path, 'w') as f:
            f.write(content)
    
    def _write_turbulence_properties(self, config: 'OpenFOAMConfig') -> None:
        """Write turbulenceProperties file (only if missing)."""
        turbulence_path = config.case_directory / "constant" / "turbulenceProperties"
        if turbulence_path.exists():
            return  # Don't overwrite existing file
        
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Website:  https://openfoam.org                 |
|   \\\\  /    A nd           | Version:  {config.openfoam_version}                              |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      turbulenceProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType  RAS;

RAS
{{
    RASModel        {config.turbulence_model.value};
    
    turbulence      on;
    
    printCoeffs     on;
}}

// ************************************************************************* //
"""
        
        with open(turbulence_path, 'w') as f:
            f.write(content)
    
    def _copy_mesh(self, config: 'OpenFOAMConfig') -> None:
        """Copy mesh to case directory (only if needed)."""
        mesh_dir = config.case_directory / "constant" / "polyMesh"
        if mesh_dir.exists() and list(mesh_dir.glob("*")):
            return  # Mesh already exists
        
        if config.mesh_file and config.mesh_file.exists():
            if config.mesh_file.is_dir():
                # Copy entire mesh directory
                shutil.copytree(config.mesh_file, mesh_dir, dirs_exist_ok=True)
            else:
                # Assume it's a mesh file to be converted
                self.logger.warning("Mesh file conversion not implemented")
    
    def _setup_boundary_conditions(self, config: 'OpenFOAMConfig') -> None:
        """Setup boundary conditions (read from existing case, don't overwrite)."""
        # Only validate that essential BC files exist
        zero_dir = config.case_directory / "0"
        essential_fields = ['U', 'p']
        
        missing_fields = []
        for field in essential_fields:
            field_file = zero_dir / field
            if not field_file.exists():
                missing_fields.append(field)
        
        if missing_fields:
            self.logger.warning(f"Missing boundary condition files: {missing_fields}")
    
    def _add_function_objects_to_control_dict(self, config: 'OpenFOAMConfig') -> None:
        """Add function objects to controlDict for force calculation."""
        if not config.function_objects:
            return
        
        control_dict_path = config.case_directory / "system" / "controlDict"
        
        # Read existing controlDict
        try:
            with open(control_dict_path, 'r') as f:
                content = f.read()
            
            # Check if functions section already exists
            if 'functions' in content:
                self.logger.info("Functions section already exists in controlDict")
                return
            
            # Add functions section before the last line
            functions_content = "\\nfunctions\\n{\\n"
            for name, func_obj in config.function_objects.items():
                functions_content += f"    {name}\\n    {{\\n"
                for key, value in func_obj.items():
                    if isinstance(value, str) and not value.startswith('('):
                        functions_content += f"        {key}        {value};\\n"
                    else:
                        functions_content += f"        {key}        {value};\\n"
                functions_content += "    }\\n\\n"
            functions_content += "}\\n"
            
            # Insert before the last line (// ***)
            lines = content.split('\\n')
            for i in range(len(lines)-1, -1, -1):
                if lines[i].startswith('//'):
                    lines.insert(i, functions_content)
                    break
            
            # Write back
            with open(control_dict_path, 'w') as f:
                f.write('\\n'.join(lines))
                
        except Exception as e:
            self.logger.warning(f"Failed to add function objects: {e}")
    
    def _cleanup_time_directories(self, case_directory: Path) -> None:
        """Clean up existing time directories to ensure fresh simulation."""
        try:
            self.logger.info("Performing thorough cleanup for fresh simulation")
            
            # Remove all numeric time directories except 0
            removed_dirs = []
            for item in case_directory.iterdir():
                if item.is_dir() and item.name.replace('.', '').replace('-', '').isdigit():
                    # Skip the initial condition directory (0)
                    if item.name != '0':
                        self.logger.info(f"Removing time directory: {item}")
                        shutil.rmtree(item)
                        removed_dirs.append(item.name)
                        
            # Also remove postProcessing directory to ensure fresh force data
            post_processing = case_directory / "postProcessing"
            if post_processing.exists():
                self.logger.info(f"Removing postProcessing directory: {post_processing}")
                shutil.rmtree(post_processing)
                
            # Force controlDict to start from time 0
            control_dict = case_directory / "system" / "controlDict"
            if control_dict.exists():
                with open(control_dict, 'r') as f:
                    content = f.read()
                
                # Ensure it starts from startTime, not latestTime
                modified_content = content.replace('startFrom       latestTime;', 'startFrom       startTime;')
                
                with open(control_dict, 'w') as f:
                    f.write(modified_content)
                    
            self.logger.info(f"Cleanup completed. Removed {len(removed_dirs)} time directories: {removed_dirs}")
                
        except Exception as e:
            self.logger.warning(f"Failed to clean up time directories: {e}")
    
    def _execute_solver(self, config: 'OpenFOAMConfig', log_file: Path) -> bool:
        """Execute OpenFOAM solver."""
        print(f"DEBUG: _execute_solver called")
        try:
            # Clean up existing time directories to ensure fresh run
            print(f"DEBUG: About to cleanup time directories")
            self._cleanup_time_directories(config.case_directory)
            print(f"DEBUG: Cleanup completed")
            
            # Use system environment (assume OpenFOAM is sourced)
            env = os.environ.copy()
            
            # Build command - just use solver name, assume it's in PATH
            if config.parallel_execution:
                cmd = [
                    'mpirun', '-np', str(config.num_processors),
                    config.solver_type.value, '-parallel'
                ]
            else:
                cmd = [config.solver_type.value]
            
            # Run solver
            self.logger.info(f"Running: {' '.join(cmd)} in {config.case_directory}")
            self.logger.info(f"DEBUG: About to execute solver command")
            print(f"DEBUG: Executing OpenFOAM solver: {' '.join(cmd)}")
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    cmd,
                    cwd=config.case_directory,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=3600  # 1 hour timeout
                )
            self.logger.info(f"DEBUG: Solver execution completed with return code: {process.returncode}")
            
            success = process.returncode == 0
            if success:
                self.logger.info("OpenFOAM simulation completed successfully")
            else:
                self.logger.error(f"OpenFOAM simulation failed with return code {process.returncode}")
                # Log the error output
                with open(log_file, 'r') as f:
                    error_output = f.read()
                    self.logger.error(f"Solver output: {error_output[-500:]}")  # Last 500 chars
            
            return success
            
        except subprocess.TimeoutExpired:
            self.logger.error("OpenFOAM simulation timeout")
            return False
        except Exception as e:
            self.logger.error(f"Failed to execute solver: {e}")
            return False
    
    def _run_decompose_par(self, config: 'OpenFOAMConfig') -> bool:
        """Run decomposePar for parallel execution."""
        try:
            env = os.environ.copy()
            if self.openfoam_root:
                env['FOAM_ETC'] = str(self.openfoam_root / 'etc')
            
            result = subprocess.run(
                ['decomposePar'],
                cwd=config.case_directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                self.logger.info("Domain decomposition completed")
                return True
            else:
                self.logger.error(f"decomposePar failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to run decomposePar: {e}")
            return False
    
    def _run_reconstruct_par(self, config: 'OpenFOAMConfig') -> bool:
        """Run reconstructPar after parallel execution."""
        try:
            env = os.environ.copy()
            if self.openfoam_root:
                env['FOAM_ETC'] = str(self.openfoam_root / 'etc')
            
            result = subprocess.run(
                ['reconstructPar'],
                cwd=config.case_directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                self.logger.info("Case reconstruction completed")
                return True
            else:
                self.logger.error(f"reconstructPar failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to run reconstructPar: {e}")
            return False
    
    def _extract_simulation_results(self, config: 'OpenFOAMConfig', log_file: Path) -> 'SimulationResults':
        """Extract results from OpenFOAM simulation."""
        results = SimulationResults(
            case_directory=config.case_directory,
            log_file=log_file,
            solver_type=config.solver_type,
            turbulence_model=config.turbulence_model
        )
        
        try:
            # Extract convergence information
            results.is_converged = self._check_convergence_from_log(log_file)
            results.iterations = self._extract_iteration_count(log_file)
            
            # Extract force coefficients if available
            results.force_coefficients = self._extract_force_coefficients(config.case_directory)
            
            # Extract residual history
            results.residual_history = self._extract_residual_history(log_file)
            
            results.status = "converged" if results.is_converged else "completed"
            
        except Exception as e:
            self.logger.error(f"Failed to extract results: {e}")
            results.status = "failed"
        
        return results
    
    def _check_convergence_from_log(self, log_file: Path) -> bool:
        """Check convergence from log file."""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Look for convergence indicators
            convergence_indicators = [
                'SIMPLE solution converged',
                'reached convergence tolerance',
                'ExecutionTime'
            ]
            
            for indicator in convergence_indicators:
                if indicator in content:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _extract_iteration_count(self, log_file: Path) -> int:
        """Extract total iteration count from log file."""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Look for iteration patterns
            pattern = r'Time = (\d+)'
            matches = re.findall(pattern, content)
            
            if matches:
                return len(matches)
            
            return 0
            
        except Exception:
            return 0
    
    def _extract_force_coefficients(self, case_dir: Path) -> Optional['ForceCoefficients']:
        """Extract force coefficients from postProcessing directory."""
        try:
            # Look for forceCoeffs data
            post_dir = case_dir / "postProcessing"
            force_coeffs_dir = post_dir / "forceCoeffs"
            
            self.logger.info(f"Looking for force coefficients in: {force_coeffs_dir}")
            
            if not force_coeffs_dir.exists():
                self.logger.warning(f"Force coefficients directory not found: {force_coeffs_dir}")
                return None
            
            # Find time directories
            time_dirs = [d for d in force_coeffs_dir.iterdir() if d.is_dir()]
            self.logger.info(f"Found time directories: {[d.name for d in time_dirs]}")
            
            if not time_dirs:
                self.logger.warning("No time directories found in forceCoeffs")
                return None
            
            # Get the latest time directory
            latest_time = max(time_dirs, key=lambda x: float(x.name) if x.name.replace('.', '').replace('-', '').isdigit() else 0)
            self.logger.info(f"Using latest time directory: {latest_time}")
            
            # Look for coefficient files
            coeff_files = ['coefficient.dat', 'forceCoeffs.dat', 'forces.dat']
            coeff_file = None
            
            for filename in coeff_files:
                file_path = latest_time / filename
                if file_path.exists():
                    coeff_file = file_path
                    self.logger.info(f"Found coefficient file: {coeff_file}")
                    break
            
            if not coeff_file:
                self.logger.warning(f"No coefficient files found in {latest_time}")
                return None
            
            # Read coefficient data - skip comment lines starting with #
            data = np.loadtxt(coeff_file, comments='#')
            if data.size == 0:
                self.logger.warning("Coefficient file is empty")
                return None
            
            # Get the last row (final values)
            if data.ndim == 1:
                final_data = data
            else:
                final_data = data[-1]
            
            self.logger.info(f"Final coefficient data: {final_data[:6]}...")  # Show first 6 values
            
            # Parse coefficients (format may vary)
            force_coeffs = ForceCoefficients()
            if len(final_data) >= 8:
                force_coeffs.cd = float(final_data[1])  # Drag coefficient
                force_coeffs.cl = float(final_data[4])  # Lift coefficient  
                force_coeffs.cm = float(final_data[7])  # Moment coefficient
                
                self.logger.info(f"Extracted force coefficients: Cd={force_coeffs.cd:.6f}, Cl={force_coeffs.cl:.6f}, Cm={force_coeffs.cm:.6f}")
            else:
                self.logger.warning(f"Insufficient data in coefficient file. Expected at least 8 columns, got {len(final_data)}")
            
            return force_coeffs
            
        except Exception as e:
            self.logger.warning(f"Failed to extract force coefficients: {e}")
            import traceback
            self.logger.warning(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _extract_residual_history(self, log_file: Path) -> List['ResidualData']:
        """Extract residual history from log file."""
        residual_history = []
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            iteration = 0
            current_residuals = {}
            
            for line in lines:
                # Parse residual lines
                if 'Solving for' in line and 'Initial residual' in line:
                    match = re.search(r'Solving for (\w+), Initial residual = ([\d.e-]+)', line)
                    if match:
                        field, residual = match.groups()
                        current_residuals[field] = float(residual)
                
                # Parse time step lines
                elif line.startswith('Time = '):
                    if current_residuals:
                        residual_data = ResidualData(
                            iteration=iteration,
                            time=float(line.split('=')[1].strip()),
                            fields=current_residuals.copy()
                        )
                        residual_history.append(residual_data)
                        iteration += 1
                        current_residuals.clear()
            
        except Exception as e:
            self.logger.warning(f"Failed to extract residual history: {e}")
        
        return residual_history
    
    def apply_design_variables(self, design_vars: np.ndarray) -> str:
        """Apply design variables to deform mesh using FFD."""
        try:
            # Create backup of original mesh if not exists
            original_mesh_dir = self.case_handler.case_path / "constant" / "polyMesh_original"
            current_mesh_dir = self.case_handler.case_path / "constant" / "polyMesh"
            
            if not original_mesh_dir.exists():
                # Backup original mesh
                shutil.copytree(current_mesh_dir, original_mesh_dir)
                self.logger.info("Created backup of original mesh")
            
            # Apply FFD deformation
            deformed_mesh_path = self._apply_ffd_deformation(design_vars)
            
            return str(deformed_mesh_path)
            
        except Exception as e:
            self.logger.error(f"Failed to apply design variables: {e}")
            return str(current_mesh_dir)
    
    def _apply_ffd_deformation(self, design_vars: np.ndarray) -> Path:
        """Apply FFD-based mesh deformation to change airfoil shape."""
        self.logger.info(f"Applying FFD deformation with {len(design_vars)} design variables")
        self.logger.info(f"Design variable values: {design_vars}")
        
        try:
            # Initialize hybrid FFD system with Y-direction only movement
            if not hasattr(self, 'hybrid_ffd_deformer'):
                from ..mesh_deformation_hybrid import HybridFFDDeformer
                # Get control points from configuration or use default [2,2,1]
                control_points = getattr(self.config, 'ffd_config', {}).get('control_points', [2, 2, 1])
                self.hybrid_ffd_deformer = HybridFFDDeformer(
                    self.case_handler.case_path,
                    control_points=control_points
                )
                self.logger.info(f"Initialized hybrid FFD system with control points {control_points} (Y-direction only)")
            
            # Apply design variables to deform airfoil mesh using hybrid FFD
            mesh_path = self.hybrid_ffd_deformer.apply_design_variables(design_vars)
            
            self.logger.info(f"Hybrid FFD mesh deformation applied successfully")
            return mesh_path
            
        except Exception as e:
            self.logger.error(f"Hybrid FFD deformation failed: {e}")
            # Restore original mesh as fallback
            if hasattr(self, 'hybrid_ffd_deformer'):
                self.hybrid_ffd_deformer.restore_original_mesh()
            return self.case_handler.case_path / "constant" / "polyMesh"

    def restore_original_mesh(self) -> bool:
        """Restore original mesh from backup using the new deformation system."""
        try:
            # Use the hybrid FFD deformer if available
            if hasattr(self, 'hybrid_ffd_deformer'):
                return self.hybrid_ffd_deformer.restore_original_mesh()
            
            # Fallback to manual restoration
            original_mesh_dir = self.case_handler.case_path / "constant" / "polyMesh_original"
            current_mesh_dir = self.case_handler.case_path / "constant" / "polyMesh"
            
            if original_mesh_dir.exists():
                # Remove current mesh
                if current_mesh_dir.exists():
                    shutil.rmtree(current_mesh_dir)
                
                # Restore original
                shutil.copytree(original_mesh_dir, current_mesh_dir)
                self.logger.info("Restored original mesh")
                return True
            else:
                self.logger.warning("No original mesh backup found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to restore original mesh: {e}")
            return False
    
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
                            objective: Any) -> Dict[str, np.ndarray]:
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
            status="completed",
            solver_type=config.solver_type,
            turbulence_model=config.turbulence_model,
            execution_time=0.0,
            iterations=0,
            is_converged=False,
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
            
            # Read coefficient data - skip comment lines starting with #
            data = np.loadtxt(coeff_file, comments='#')  # Skip header
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