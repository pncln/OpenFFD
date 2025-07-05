"""Universal CFD optimization engine."""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np

# Optional optimization dependencies
try:
    from scipy.optimize import minimize, differential_evolution
    HAS_SCIPY = True
except ImportError:
    minimize = None
    differential_evolution = None
    HAS_SCIPY = False

from ..core.config import CaseConfig
from ..core.registry import CaseTypeRegistry
from ..core.base import BaseCase, BaseSolver, BaseOptimizer
from ..solvers.openfoam import OpenFOAMSolver
from .objectives import ObjectiveRegistry
from .sensitivity import SensitivityAnalyzer


class UniversalOptimizer:
    """Universal CFD optimization engine that works with any case type."""
    
    def __init__(self, case_path: Union[str, Path], config_file: Optional[str] = None):
        """Initialize universal optimizer.
        
        Args:
            case_path: Path to OpenFOAM case directory
            config_file: Path to optimization configuration file (optional)
        """
        self.case_path = Path(case_path)
        self.config_file = config_file
        self.config = None
        self.case_handler = None
        self.solver = None
        self.objectives = []
        self.sensitivity_analyzer = None
        self.history = []
        self.current_iteration = 0
        
        # Initialize optimization
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize optimization components."""
        # Load or create configuration
        if self.config_file:
            self.config = CaseConfig.from_file(self.config_file)
        else:
            self.config = self._auto_detect_config()
        
        # Create case handler
        case_type = self.config.case_type
        if case_type == 'auto':
            case_type = CaseTypeRegistry.auto_detect_case_type(self.case_path)
            self.config.case_type = case_type
        
        case_handler_class = CaseTypeRegistry.get_case_handler(case_type)
        self.case_handler = case_handler_class(self.case_path, self.config)
        
        # Validate case
        if not self.case_handler.validate_case():
            raise ValueError(f"Case validation failed for {self.case_path}")
        
        # Create solver
        solver_class = CaseTypeRegistry.get_solver(self.config.solver)
        self.solver = solver_class(self.case_handler)
        
        # Set up objectives
        self._setup_objectives()
        
        # Create sensitivity analyzer
        self.sensitivity_analyzer = SensitivityAnalyzer(
            self.case_handler, 
            self.solver,
            self.config.optimization.step_size
        )
        
        print(f"Initialized universal optimizer for {case_type} case")
        print(f"Case path: {self.case_path}")
        print(f"Solver: {self.config.solver}")
        print(f"Objectives: {[obj.name for obj in self.objectives]}")
    
    def _auto_detect_config(self) -> CaseConfig:
        """Auto-detect configuration from case directory."""
        # Try to find config file in case directory
        config_files = [
            'case_config.yaml',
            'case_config.yml', 
            'optimization_config.yaml',
            'optimization_config.yml',
            'config.yaml',
            'config.yml'
        ]
        
        for config_file in config_files:
            config_path = self.case_path / config_file
            if config_path.exists():
                print(f"Found configuration file: {config_path}")
                return CaseConfig.from_file(config_path)
        
        # Auto-detect case type
        case_type = CaseTypeRegistry.auto_detect_case_type(self.case_path)
        print(f"Auto-detected case type: {case_type}")
        
        # Create default config based on case type
        if case_type == 'airfoil':
            from ..core.config import create_default_airfoil_config
            return create_default_airfoil_config()
        elif case_type == 'heat_transfer':
            from ..core.config import create_default_heat_transfer_config
            return create_default_heat_transfer_config()
        else:
            # Generic config
            return CaseConfig(
                case_type=case_type,
                solver='simpleFoam',
                physics='incompressible_flow'
            )
    
    def _setup_objectives(self) -> None:
        """Set up objective functions."""
        self.objectives = []
        
        for obj_config in self.config.objectives:
            objective_class = ObjectiveRegistry.get_objective(obj_config.name)
            objective = objective_class(obj_config, self.case_handler)
            self.objectives.append(objective)
    
    def optimize(self) -> Dict[str, Any]:
        """Run optimization."""
        print(f"Starting optimization with {len(self.objectives)} objectives")
        print(f"Max iterations: {self.config.optimization.max_iterations}")
        print(f"Tolerance: {self.config.optimization.tolerance}")
        
        # Setup optimization domain
        domain_info = self.case_handler.setup_optimization_domain()
        print(f"FFD domain: {domain_info.get('domain', 'auto')}")
        
        # Prepare mesh for optimization
        if not self.case_handler.prepare_mesh_for_optimization():
            raise RuntimeError("Failed to prepare mesh for optimization")
        
        # Initialize design variables (FFD control points)
        design_vars = self._initialize_design_variables(domain_info)
        
        # Run optimization loop
        start_time = time.time()
        
        try:
            if self.config.optimization.algorithm.lower() == 'slsqp':
                result = self._optimize_slsqp(design_vars)
            elif self.config.optimization.algorithm.lower() == 'genetic':
                result = self._optimize_genetic(design_vars)
            else:
                raise ValueError(f"Unknown optimization algorithm: {self.config.optimization.algorithm}")
            
            end_time = time.time()
            
            # Compile results
            optimization_result = {
                'success': result.get('success', False),
                'final_objective': result.get('fun', 0.0),
                'optimal_design_vars': result.get('x', design_vars).tolist(),
                'iterations': self.current_iteration,
                'total_time': end_time - start_time,
                'history': self.history,
                'case_info': self.case_handler.get_case_info(),
                'domain_info': domain_info
            }
            
            # Save results
            self._save_results(optimization_result)
            
            return optimization_result
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            raise
    
    def _initialize_design_variables(self, domain_info: Dict[str, Any]) -> np.ndarray:
        """Initialize design variables (FFD control points)."""
        control_points = domain_info['control_points']
        
        if isinstance(control_points, list) and len(control_points) == 3:
            n_vars = np.prod(control_points)
        else:
            n_vars = 24  # Default for 8x6x2 grid
        
        # Initialize to zero (no deformation)
        design_vars = np.zeros(n_vars)
        
        return design_vars
    
    def _optimize_slsqp(self, initial_vars: np.ndarray) -> Dict[str, Any]:
        """Optimize using SLSQP algorithm."""
        if not HAS_SCIPY:
            raise ImportError("SciPy is required for optimization. Install with: pip install scipy")
        
        def objective_function(x):
            return self._evaluate_objective(x)
        
        def gradient_function(x):
            return self._evaluate_gradient(x)
        
        # Set up constraints
        constraints = self._setup_constraints()
        
        # Set up bounds
        bounds = self._setup_bounds(initial_vars)
        
        # Run optimization
        result = minimize(
            objective_function,
            initial_vars,
            method='SLSQP',
            jac=gradient_function,
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.config.optimization.max_iterations,
                'ftol': self.config.optimization.tolerance * 0.1,  # Reduced function tolerance
                'gtol': 1e-8,  # Gradient tolerance
                'eps': 1e-8,   # Step size for finite difference
                'disp': True
            }
        )
        
        return result
    
    def _optimize_genetic(self, initial_vars: np.ndarray) -> Dict[str, Any]:
        """Optimize using genetic algorithm."""
        if not HAS_SCIPY:
            raise ImportError("SciPy is required for optimization. Install with: pip install scipy")
        
        def objective_function(x):
            return self._evaluate_objective(x)
        
        # Set up bounds
        bounds = self._setup_bounds(initial_vars)
        
        # Run optimization
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=self.config.optimization.max_iterations,
            tol=self.config.optimization.tolerance,
            disp=True
        )
        
        return result
    
    def _evaluate_objective(self, design_vars: np.ndarray) -> float:
        """Evaluate objective function for given design variables."""
        self.current_iteration += 1
        
        print(f"Iteration {self.current_iteration}: Evaluating objective...")
        
        # Apply design variables (deform mesh)
        mesh_file = self._apply_design_variables(design_vars)
        
        # Save mesh for this iteration
        self._save_iteration_mesh(self.current_iteration)
        
        # Run CFD simulation
        results = self.solver.run_simulation(mesh_file)
        
        # Extract objective values
        objective_values = self.case_handler.extract_objectives(results)
        
        # Compute weighted objective
        total_objective = 0.0
        for objective in self.objectives:
            if objective.name in objective_values:
                value = objective_values[objective.name]
                total_objective += objective.weight * value
        
        # Store iteration data
        iteration_data = {
            'iteration': self.current_iteration,
            'design_vars': design_vars.tolist(),
            'objective_value': total_objective,
            'objective_components': objective_values,
            'converged': results.get('converged', False)
        }
        
        self.history.append(iteration_data)
        
        print(f"  Objective value: {total_objective:.6f}")
        print(f"  Mesh saved to: optimization_meshes/iteration_{self.current_iteration:03d}")
        
        return total_objective
    
    def _evaluate_gradient(self, design_vars: np.ndarray) -> np.ndarray:
        """Evaluate gradient of objective function."""
        return self.sensitivity_analyzer.compute_gradient(design_vars, self.objectives)
    
    def _apply_design_variables(self, design_vars: np.ndarray) -> str:
        """Apply design variables to deform mesh."""
        # Use solver's mesh deformation capability
        if hasattr(self.solver, 'apply_design_variables'):
            return self.solver.apply_design_variables(design_vars)
        else:
            # Fallback: return original mesh path
            return str(self.case_path / 'constant' / 'polyMesh')
    
    def _setup_constraints(self) -> List[Dict[str, Any]]:
        """Set up optimization constraints."""
        constraints = []
        
        for objective in self.objectives:
            if objective.is_constraint():
                constraint = {
                    'type': 'ineq' if objective.config.constraint_type in ['min', 'max'] else 'eq',
                    'fun': lambda x, obj=objective: self._evaluate_constraint(x, obj)
                }
                constraints.append(constraint)
        
        return constraints
    
    def _evaluate_constraint(self, design_vars: np.ndarray, objective) -> float:
        """Evaluate constraint function."""
        # Run simulation and get objective value
        mesh_file = self._apply_design_variables(design_vars)
        results = self.solver.run_simulation(mesh_file)
        objective_values = self.case_handler.extract_objectives(results)
        
        value = objective_values.get(objective.name, 0.0)
        target = objective.config.constraint_value
        
        if objective.config.constraint_type == 'min':
            return value - target
        elif objective.config.constraint_type == 'max':
            return target - value
        else:  # equality
            return abs(value - target)
    
    def _setup_bounds(self, design_vars: np.ndarray) -> List[tuple]:
        """Set up bounds for design variables."""
        # Default bounds: allow ±20% deformation
        bounds = []
        for i in range(len(design_vars)):
            bounds.append((-0.2, 0.2))
        
        return bounds
    
    def _save_iteration_mesh(self, iteration: int) -> None:
        """Save the current mesh state for this iteration."""
        import shutil
        
        # Create optimization_meshes directory if it doesn't exist
        mesh_dir = self.case_path / "optimization_meshes"
        mesh_dir.mkdir(exist_ok=True)
        
        # Create iteration-specific directory
        iteration_dir = mesh_dir / f"iteration_{iteration:03d}"
        iteration_dir.mkdir(exist_ok=True)
        
        # Copy current polyMesh
        source_mesh = self.case_path / "constant" / "polyMesh"
        dest_mesh = iteration_dir / "polyMesh"
        
        if source_mesh.exists():
            if dest_mesh.exists():
                shutil.rmtree(dest_mesh)
            shutil.copytree(source_mesh, dest_mesh)
            
            # Also save design variables for reference
            design_vars_file = iteration_dir / "design_variables.txt"
            if hasattr(self, 'history') and self.history:
                current_vars = self.history[-1]['design_vars']
                with open(design_vars_file, 'w') as f:
                    f.write(f"Iteration: {iteration}\n")
                    f.write(f"Design Variables: {current_vars}\n")
                    f.write(f"Objective Value: {self.history[-1]['objective_value']:.8f}\n")
                    f.write(f"Drag Coefficient: {self.history[-1]['objective_components'].get('drag_coefficient', 'N/A')}\n")
                    f.write(f"Lift Coefficient: {self.history[-1]['objective_components'].get('lift_coefficient', 'N/A')}\n")
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save optimization results."""
        # Convert to JSON-serializable format
        json_results = self._make_json_serializable(results)
        
        # Save to JSON
        results_file = self.case_path / self.config.optimization.history_file
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        
        # Save configuration used
        config_file = self.case_path / 'optimization_config_used.json'  # Use JSON to avoid YAML dependency
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, '__dict__'):
            # Handle custom objects by converting to dict
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get information about the optimization setup."""
        return {
            'case_path': str(self.case_path),
            'case_type': self.config.case_type,
            'solver': self.config.solver,
            'objectives': [obj.name for obj in self.objectives],
            'algorithm': self.config.optimization.algorithm,
            'max_iterations': self.config.optimization.max_iterations,
            'tolerance': self.config.optimization.tolerance,
            'current_iteration': self.current_iteration,
            'history_length': len(self.history)
        }