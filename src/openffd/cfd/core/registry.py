"""Registry for case types and components."""

from typing import Dict, Type, Any, List
from pathlib import Path
import os

from .base import BaseCase, BaseSolver, BaseObjective


class CaseTypeRegistry:
    """Registry for different CFD case types."""
    
    _case_handlers: Dict[str, Type[BaseCase]] = {}
    _solvers: Dict[str, Type[BaseSolver]] = {}
    _objectives: Dict[str, Type[BaseObjective]] = {}
    
    @classmethod
    def register_case_handler(cls, case_type: str, handler_class: Type[BaseCase]) -> None:
        """Register a case handler for a specific case type."""
        cls._case_handlers[case_type] = handler_class
    
    @classmethod
    def register_solver(cls, solver_name: str, solver_class: Type[BaseSolver]) -> None:
        """Register a solver."""
        cls._solvers[solver_name] = solver_class
    
    @classmethod
    def register_objective(cls, objective_name: str, objective_class: Type[BaseObjective]) -> None:
        """Register an objective function."""
        cls._objectives[objective_name] = objective_class
    
    @classmethod
    def get_case_handler(cls, case_type: str) -> Type[BaseCase]:
        """Get case handler for a specific case type."""
        if case_type not in cls._case_handlers:
            raise ValueError(f"Unknown case type: {case_type}. "
                           f"Available types: {list(cls._case_handlers.keys())}")
        return cls._case_handlers[case_type]
    
    @classmethod
    def get_solver(cls, solver_name: str) -> Type[BaseSolver]:
        """Get solver class by name."""
        if solver_name not in cls._solvers:
            raise ValueError(f"Unknown solver: {solver_name}. "
                           f"Available solvers: {list(cls._solvers.keys())}")
        return cls._solvers[solver_name]
    
    @classmethod
    def get_objective(cls, objective_name: str) -> Type[BaseObjective]:
        """Get objective class by name."""
        if objective_name not in cls._objectives:
            raise ValueError(f"Unknown objective: {objective_name}. "
                           f"Available objectives: {list(cls._objectives.keys())}")
        return cls._objectives[objective_name]
    
    @classmethod
    def auto_detect_case_type(cls, case_path: Path) -> str:
        """Automatically detect case type from case directory."""
        case_path = Path(case_path)
        
        # Check for specific files/directories that indicate case type
        detection_rules = {
            'airfoil': cls._detect_airfoil_case,
            'heat_transfer': cls._detect_heat_transfer_case,
            'pipe_flow': cls._detect_pipe_flow_case,
            'generic': cls._detect_generic_case
        }
        
        for case_type, detector in detection_rules.items():
            if detector(case_path):
                return case_type
        
        # Default to generic if no specific type detected
        return 'generic'
    
    @classmethod
    def _detect_airfoil_case(cls, case_path: Path) -> bool:
        """Detect if case is an airfoil optimization case."""
        # Look for airfoil-specific indicators
        constant_dir = case_path / 'constant'
        system_dir = case_path / 'system'
        
        if not (constant_dir.exists() and system_dir.exists()):
            return False
        
        # Check for airfoil-specific files
        indicators = [
            'airfoil' in str(case_path).lower(),
            'naca' in str(case_path).lower(),
            any('airfoil' in f.name.lower() for f in constant_dir.glob('*')),
            any('wing' in f.name.lower() for f in constant_dir.glob('*'))
        ]
        
        return any(indicators)
    
    @classmethod
    def _detect_heat_transfer_case(cls, case_path: Path) -> bool:
        """Detect if case is a heat transfer case."""
        constant_dir = case_path / 'constant'
        system_dir = case_path / 'system'
        
        if not (constant_dir.exists() and system_dir.exists()):
            return False
        
        # Check for heat transfer indicators
        indicators = [
            'heat' in str(case_path).lower(),
            'thermal' in str(case_path).lower(),
            'temperature' in str(case_path).lower(),
            (constant_dir / 'thermophysicalProperties').exists(),
            (constant_dir / 'radiationProperties').exists()
        ]
        
        return any(indicators)
    
    @classmethod
    def _detect_pipe_flow_case(cls, case_path: Path) -> bool:
        """Detect if case is a pipe flow case."""
        constant_dir = case_path / 'constant'
        
        if not constant_dir.exists():
            return False
        
        # Check for pipe flow indicators
        indicators = [
            'pipe' in str(case_path).lower(),
            'channel' in str(case_path).lower(),
            'duct' in str(case_path).lower(),
            'tube' in str(case_path).lower()
        ]
        
        return any(indicators)
    
    @classmethod
    def _detect_generic_case(cls, case_path: Path) -> bool:
        """Detect if case is a generic OpenFOAM case."""
        case_path = Path(case_path)
        
        # Check for standard OpenFOAM directory structure
        required_dirs = ['constant', 'system']
        return all((case_path / d).exists() for d in required_dirs)
    
    @classmethod
    def list_available_types(cls) -> Dict[str, List[str]]:
        """List all available registered types."""
        return {
            'case_handlers': list(cls._case_handlers.keys()),
            'solvers': list(cls._solvers.keys()),
            'objectives': list(cls._objectives.keys())
        }
    
    @classmethod
    def validate_case_config(cls, case_type: str, solver: str, objectives: List[str]) -> bool:
        """Validate that case configuration is supported."""
        try:
            cls.get_case_handler(case_type)
            cls.get_solver(solver)
            for obj in objectives:
                cls.get_objective(obj)
            return True
        except ValueError:
            return False


# Decorator for easy registration
def register_case_handler(case_type: str):
    """Decorator to register case handlers."""
    def decorator(cls):
        CaseTypeRegistry.register_case_handler(case_type, cls)
        return cls
    return decorator


def register_solver(solver_name: str):
    """Decorator to register solvers."""
    def decorator(cls):
        CaseTypeRegistry.register_solver(solver_name, cls)
        return cls
    return decorator


def register_objective(objective_name: str):
    """Decorator to register objectives."""
    def decorator(cls):
        CaseTypeRegistry.register_objective(objective_name, cls)
        return cls
    return decorator