"""
Universal CFD Optimization Framework for OpenFFD.

This module provides a completely redesigned universal CFD optimization framework that works
with any OpenFOAM case type without hardcoded assumptions.

Key Features:
- Universal case handling (airfoil, heat transfer, pipe flow, etc.)
- Automatic case type detection
- Configuration-driven optimization setup
- Plugin-based architecture for extensibility
- Full OpenFOAM solver integration
- Multiple optimization algorithms
- Comprehensive objective function library
- Advanced sensitivity analysis

Architecture:
- Core framework with base classes and configuration management
- Case handlers for different physics (airfoil, heat transfer, etc.)
- Universal solver interfaces (OpenFOAM, future: SU2, etc.)
- Optimization engine with multiple algorithms
- Utility functions for mesh handling and post-processing

Example Usage - Universal Approach:
    ```python
    from openffd.cfd.optimization.optimizer import UniversalOptimizer
    
    # Works with any OpenFOAM case
    optimizer = UniversalOptimizer("/path/to/openfoam/case")
    results = optimizer.optimize()
    ```

Example Usage - Configuration-Based:
    ```python
    from openffd.cfd.optimization.optimizer import UniversalOptimizer
    
    # Use custom configuration
    optimizer = UniversalOptimizer(".", "my_config.yaml")
    results = optimizer.optimize()
    ```

Example Usage - Programmatic:
    ```python
    from openffd.cfd.core.config import CaseConfig
    from openffd.cfd.optimization.optimizer import UniversalOptimizer
    
    # Create configuration programmatically
    config = CaseConfig(
        case_type="airfoil",
        solver="simpleFoam",
        objectives=[...],
        optimization={...}
    )
    
    optimizer = UniversalOptimizer(case_path, config_file=None)
    optimizer.config = config
    results = optimizer.optimize()
    ```

For detailed examples, see examples/naca0012_case/ directory.
"""

# Universal optimization framework
from .optimization.optimizer import UniversalOptimizer
from .optimization.objectives import ObjectiveRegistry

# Core framework components
from .core.config import (
    CaseConfig,
    OptimizationConfig,
    FFDConfig,
    ObjectiveConfig,
    create_default_airfoil_config,
    create_default_heat_transfer_config
)
from .core.registry import CaseTypeRegistry
from .core.base import BaseCase, BaseSolver, BaseObjective, BaseOptimizer

# Case handlers
from .cases.base_case import GenericCase
from .cases.airfoil_case import AirfoilCase
from .cases.heat_transfer_case import HeatTransferCase

# Solvers
from .solvers.openfoam import OpenFOAMSolver

# Sensitivity analysis
from .optimization.sensitivity import SensitivityAnalyzer

# Utilities
from .utils import utilities

# Legacy compatibility imports (deprecated but maintained for backward compatibility)
try:
    # Legacy imports - these will work but are deprecated
    from .base import (
        CFDSolver,
        CFDConfig,
        CFDResults,
        SolverStatus,
        ConvergenceData,
        ObjectiveFunction
    )
except ImportError:
    pass

try:
    # Legacy optimization import
    from .optimization import OptimizationManager
except ImportError:
    pass

__all__ = [
    # Universal optimization framework
    'UniversalOptimizer',
    'ObjectiveRegistry',
    
    # Core configuration
    'CaseConfig',
    'OptimizationConfig',
    'FFDConfig', 
    'ObjectiveConfig',
    'create_default_airfoil_config',
    'create_default_heat_transfer_config',
    
    # Core framework
    'CaseTypeRegistry',
    'BaseCase',
    'BaseSolver', 
    'BaseObjective',
    'BaseOptimizer',
    
    # Case handlers
    'GenericCase',
    'AirfoilCase',
    'HeatTransferCase',
    
    # Solvers
    'OpenFOAMSolver',
    
    # Analysis tools
    'SensitivityAnalyzer',
    
    # Utilities
    'utilities',
    
    # Legacy compatibility (deprecated)
    'OptimizationManager',  # If available
]

# Version information
__version__ = "2.0.0"
__author__ = "OpenFFD Development Team"
__license__ = "MIT"

# Framework information
__framework_info__ = {
    'name': 'Universal CFD Optimization Framework',
    'version': __version__,
    'architecture': 'Plugin-based universal framework',
    'supported_solvers': ['OpenFOAM'],
    'supported_case_types': ['airfoil', 'heat_transfer', 'generic'],
    'optimization_algorithms': ['SLSQP', 'genetic'],
    'gradient_methods': ['finite_difference', 'adjoint']
}

# Configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Auto-register components on import
def _auto_register_components():
    """Auto-register all framework components."""
    # Case handlers are auto-registered via decorators
    # Solvers are auto-registered via decorators  
    # Objectives are auto-registered via decorators
    pass

_auto_register_components()

# Print framework info on first import
_framework_initialized = False
if not _framework_initialized:
    logger = logging.getLogger(__name__)
    logger.info(f"Initialized {__framework_info__['name']} v{__version__}")
    logger.info(f"Supported case types: {', '.join(__framework_info__['supported_case_types'])}")
    _framework_initialized = True