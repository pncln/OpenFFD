"""Base classes for universal CFD optimization framework."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np

from .config import CaseConfig


class BaseCase(ABC):
    """Abstract base class for CFD case handling."""
    
    def __init__(self, case_path: Path, config: CaseConfig):
        self.case_path = Path(case_path)
        self.config = config
        self._validated = False
    
    @abstractmethod
    def detect_case_type(self) -> str:
        """Detect the type of CFD case from the case directory."""
        pass
    
    @abstractmethod
    def validate_case(self) -> bool:
        """Validate that the case is properly set up."""
        pass
    
    @abstractmethod
    def setup_optimization_domain(self) -> Dict[str, Any]:
        """Set up the optimization domain and FFD/HFFD parameters."""
        pass
    
    @abstractmethod
    def extract_objectives(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract objective function values from CFD results."""
        pass
    
    @abstractmethod
    def get_boundary_patches(self) -> List[str]:
        """Get list of boundary patches relevant for optimization."""
        pass
    
    @abstractmethod
    def prepare_mesh_for_optimization(self) -> bool:
        """Prepare the mesh for optimization (conversion, etc.)."""
        pass
    
    def get_case_info(self) -> Dict[str, Any]:
        """Get general information about the case."""
        return {
            'case_path': str(self.case_path),
            'case_type': self.config.case_type,
            'solver': self.config.solver,
            'physics': self.config.physics,
            'objectives': [obj.name for obj in self.config.objectives]
        }


class BaseSolver(ABC):
    """Abstract base class for CFD solvers."""
    
    def __init__(self, case_handler: BaseCase):
        self.case_handler = case_handler
        self.config = case_handler.config
    
    @abstractmethod
    def setup_solver(self) -> bool:
        """Set up the solver for the given case."""
        pass
    
    @abstractmethod
    def run_simulation(self, mesh_file: Optional[str] = None) -> Dict[str, Any]:
        """Run CFD simulation and return results."""
        pass
    
    @abstractmethod
    def check_convergence(self) -> bool:
        """Check if simulation has converged."""
        pass
    
    @abstractmethod
    def get_residuals(self) -> Dict[str, List[float]]:
        """Get residual history."""
        pass
    
    @abstractmethod
    def extract_forces(self, patches: List[str]) -> Dict[str, np.ndarray]:
        """Extract forces from specified patches."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up solver temporary files."""
        pass


class BaseObjective(ABC):
    """Abstract base class for optimization objectives."""
    
    def __init__(self, config: Any, case_handler: BaseCase):
        self.config = config
        self.case_handler = case_handler
        self.name = config.name
        self.weight = config.weight
    
    @abstractmethod
    def evaluate(self, results: Dict[str, Any]) -> float:
        """Evaluate objective function from CFD results."""
        pass
    
    @abstractmethod
    def get_gradient(self, results: Dict[str, Any], design_vars: np.ndarray) -> np.ndarray:
        """Get gradient of objective with respect to design variables."""
        pass
    
    def is_constraint(self) -> bool:
        """Check if this objective is a constraint."""
        return self.config.constraint_type is not None
    
    def check_constraint(self, value: float) -> bool:
        """Check if constraint is satisfied."""
        if not self.is_constraint():
            return True
        
        if self.config.constraint_type == 'min':
            return value >= self.config.constraint_value
        elif self.config.constraint_type == 'max':
            return value <= self.config.constraint_value
        elif self.config.constraint_type == 'equality':
            return abs(value - self.config.constraint_value) < 1e-6
        
        return True


class BaseOptimizer(ABC):
    """Abstract base class for optimization algorithms."""
    
    def __init__(self, case_handler: BaseCase, solver: BaseSolver):
        self.case_handler = case_handler
        self.solver = solver
        self.config = case_handler.config
        self.objectives = []
        self.history = []
    
    @abstractmethod
    def setup_optimization(self) -> bool:
        """Set up optimization problem."""
        pass
    
    @abstractmethod
    def optimize(self) -> Dict[str, Any]:
        """Run optimization and return results."""
        pass
    
    @abstractmethod
    def evaluate_objectives(self, design_vars: np.ndarray) -> Dict[str, float]:
        """Evaluate all objectives for given design variables."""
        pass
    
    @abstractmethod
    def get_design_variables(self) -> np.ndarray:
        """Get current design variables."""
        pass
    
    @abstractmethod
    def set_design_variables(self, design_vars: np.ndarray) -> None:
        """Set design variables."""
        pass
    
    def add_objective(self, objective: BaseObjective) -> None:
        """Add an objective to the optimization problem."""
        self.objectives.append(objective)
    
    def save_history(self, filename: str) -> None:
        """Save optimization history."""
        import json
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_history(self, filename: str) -> None:
        """Load optimization history."""
        import json
        with open(filename, 'r') as f:
            self.history = json.load(f)


class GeometryHandler(ABC):
    """Abstract base class for geometry handling."""
    
    @abstractmethod
    def load_geometry(self, geometry_file: str) -> Any:
        """Load geometry from file."""
        pass
    
    @abstractmethod
    def save_geometry(self, geometry: Any, filename: str) -> None:
        """Save geometry to file."""
        pass
    
    @abstractmethod
    def apply_deformation(self, geometry: Any, deformation: np.ndarray) -> Any:
        """Apply deformation to geometry."""
        pass
    
    @abstractmethod
    def get_surface_mesh(self, geometry: Any) -> np.ndarray:
        """Extract surface mesh from geometry."""
        pass