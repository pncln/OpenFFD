"""Configuration management for universal CFD optimization."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

# Optional YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False


@dataclass
class ObjectiveConfig:
    """Configuration for optimization objectives."""
    name: str
    weight: float = 1.0
    target: Optional[float] = None
    constraint_type: Optional[str] = None  # 'min', 'max', 'equality'
    constraint_value: Optional[float] = None
    patches: List[str] = field(default_factory=list)
    direction: List[float] = field(default_factory=list)


@dataclass
class FFDConfig:
    """Configuration for FFD/HFFD setup."""
    control_points: List[int] = field(default_factory=lambda: [8, 6, 2])
    domain: Union[str, Dict[str, float]] = "auto"
    ffd_type: str = "ffd"  # 'ffd' or 'hffd'
    basis_functions: str = "bernstein"
    order: int = 3
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters."""
    max_iterations: int = 50
    tolerance: float = 1e-6
    algorithm: str = "slsqp"
    step_size: float = 1e-3
    parallel: bool = False
    restart: bool = False
    history_file: str = "optimization_history.json"
    initial_design_vars: Optional[List[float]] = None


@dataclass
class CaseConfig:
    """Main configuration for CFD optimization case."""
    case_type: str
    solver: str
    physics: str = "incompressible_flow"
    objectives: List[ObjectiveConfig] = field(default_factory=list)
    ffd_config: FFDConfig = field(default_factory=FFDConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    boundaries: Dict[str, Any] = field(default_factory=dict)
    constants: Dict[str, Any] = field(default_factory=dict)
    mesh_config: Dict[str, Any] = field(default_factory=dict)
    post_processing: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'CaseConfig':
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ImportError(
                        "YAML support not available. Install PyYAML with: pip install PyYAML\n"
                        "Or convert your configuration to JSON format."
                    )
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CaseConfig':
        """Create configuration from dictionary."""
        # Convert objectives
        objectives = []
        for obj_data in data.get('objectives', []):
            objectives.append(ObjectiveConfig(**obj_data))
        
        # Convert FFD config
        ffd_data = data.get('ffd_config', {})
        ffd_config = FFDConfig(**ffd_data)
        
        # Convert optimization config
        opt_data = data.get('optimization', {})
        optimization = OptimizationConfig(**opt_data)
        
        # Create main config
        config_data = data.copy()
        config_data['objectives'] = objectives
        config_data['ffd_config'] = ffd_config
        config_data['optimization'] = optimization
        
        return cls(**config_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {
            'case_type': self.case_type,
            'solver': self.solver,
            'physics': self.physics,
            'objectives': [
                {
                    'name': obj.name,
                    'weight': obj.weight,
                    'target': obj.target,
                    'constraint_type': obj.constraint_type,
                    'constraint_value': obj.constraint_value,
                    'patches': obj.patches,
                    'direction': obj.direction
                }
                for obj in self.objectives
            ],
            'ffd_config': {
                'control_points': self.ffd_config.control_points,
                'domain': self.ffd_config.domain,
                'ffd_type': self.ffd_config.ffd_type,
                'basis_functions': self.ffd_config.basis_functions,
                'order': self.ffd_config.order,
                'constraints': self.ffd_config.constraints
            },
            'optimization': {
                'max_iterations': self.optimization.max_iterations,
                'tolerance': self.optimization.tolerance,
                'algorithm': self.optimization.algorithm,
                'step_size': self.optimization.step_size,
                'parallel': self.optimization.parallel,
                'restart': self.optimization.restart,
                'history_file': self.optimization.history_file
            },
            'boundaries': self.boundaries,
            'constants': self.constants,
            'mesh_config': self.mesh_config,
            'post_processing': self.post_processing
        }
        return result
    
    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ImportError(
                        "YAML support not available. Install PyYAML with: pip install PyYAML\n"
                        "Or save as JSON format instead."
                    )
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                json.dump(self.to_dict(), f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def create_default_airfoil_config() -> CaseConfig:
    """Create default configuration for airfoil optimization."""
    return CaseConfig(
        case_type="airfoil",
        solver="simpleFoam",
        physics="incompressible_flow",
        objectives=[
            ObjectiveConfig(
                name="drag_coefficient",
                weight=1.0,
                patches=["airfoil"],
                direction=[1.0, 0.0, 0.0]
            )
        ],
        ffd_config=FFDConfig(
            control_points=[8, 6, 2],
            domain="auto",
            ffd_type="ffd"
        ),
        optimization=OptimizationConfig(
            max_iterations=50,
            tolerance=1e-6,
            algorithm="slsqp"
        ),
        constants={
            "magUInf": 30.0,
            "rho": 1.225,
            "mu": 1.8e-5
        }
    )


def create_default_heat_transfer_config() -> CaseConfig:
    """Create default configuration for heat transfer optimization."""
    return CaseConfig(
        case_type="heat_transfer",
        solver="buoyantSimpleFoam",
        physics="buoyant_flow",
        objectives=[
            ObjectiveConfig(
                name="heat_transfer_coefficient",
                weight=1.0,
                patches=["hot_wall"]
            )
        ],
        ffd_config=FFDConfig(
            control_points=[6, 6, 4],
            domain="auto",
            ffd_type="ffd"
        ),
        optimization=OptimizationConfig(
            max_iterations=30,
            tolerance=1e-5,
            algorithm="slsqp"
        )
    )