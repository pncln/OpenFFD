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

# Discrete Adjoint Framework
from .adjoint import (
    AdjointVariables, AdjointConfig, DiscreteAdjointSolver,
    DragObjective, LiftObjective, CompositeObjective,
    SensitivityAnalysis, DesignGradient, GMRESAdjointSolver
)

# CFD Equation Solvers
from .equations import EulerEquations3D, NavierStokesEquations3D
from .mesh import UnstructuredMesh3D, ConnectivityManager, BoundaryManager
from .numerics import RiemannSolverManager, WENOReconstructor, TVDReconstructor

# Slope and Flux Limiters
from .limiters import (
    SlopeLimiter, MinmodLimiter, SuperbeeLimiter, VanLeerLimiter, 
    MUSCLLimiter, VenkatakrishnanLimiter, FluxLimiter, AdaptiveLimiter,
    ShockDetector, MultiDimensionalLimiter, create_slope_limiter, LimiterConfig
)

# Time Integration Schemes
from .time_integration import (
    TimeIntegrator, ExplicitEuler, RungeKutta2, RungeKutta4, RungeKutta3TVD,
    BackwardEuler, BDF2, AdaptiveRungeKutta, LocalTimeSteppingSolver,
    create_time_integrator, TimeIntegrationConfig
)

# Boundary Conditions
from .boundary_conditions import (
    BoundaryCondition, FarfieldBoundaryCondition, WallBoundaryCondition,
    SymmetryBoundaryCondition, InletBoundaryCondition, OutletBoundaryCondition,
    BoundaryConditionManager, BoundaryPatch, BoundaryType,
    create_boundary_patch, BoundaryConditionConfig
)

# Turbulence Models
from .turbulence_models import (
    TurbulenceModel, SpalartAllmarasModel, KEpsilonStandardModel, KOmegaSSTModel,
    TurbulenceModelManager, TurbulenceQuantities, TurbulenceModelType,
    create_turbulence_model, TurbulenceModelConfig
)

# Mesh Adaptation and Refinement
from .mesh_adaptation import (
    ErrorEstimator, GradientJumpEstimator, ResidualBasedEstimator,
    ShockDetector, MeshRefiner, SolutionInterpolator, MeshAdaptationManager,
    AdaptationMetrics, RefinementType, AdaptationStrategy,
    create_mesh_adaptation_manager, MeshAdaptationConfig
)

# Parallel Computing with MPI
from .parallel_computing import (
    DomainDecomposer, GeometricDecomposer, GraphBasedDecomposer,
    MPICommunicator, LoadBalancer, ParallelSolver, DomainPartition,
    DecompositionMethod, CommunicationPattern, ParallelConfig,
    create_parallel_solver
)

# OpenMP Threading for Shared Memory Parallelization
from .openmp_threading import (
    ParallelKernels, ThreadScheduler, PerformanceProfiler, OpenMPManager,
    ThreadSafeData, ThreadingMetrics, ThreadingStrategy, WorkDistribution,
    create_openmp_manager, OpenMPConfig
)

# Memory Optimization and Cache Efficiency
from .memory_optimization import (
    CacheOptimizedArray, MemoryPool, DataStructureOptimizer, MemoryBandwidthAnalyzer,
    MemoryManager, MemoryMetrics, DataLayout, MemoryPattern, AllocatorType,
    create_memory_manager, MemoryConfig
)

# Convergence Monitoring and Residual Tracking
from .convergence_monitoring import (
    ConvergenceMonitor, ResidualComputer, ConvergenceAnalyzer, ResidualData,
    ConvergenceMetrics, ConvergenceConfig, ResidualNorm, ConvergenceStatus,
    StoppingCriterion, create_convergence_monitor
)

# Supersonic Validation Cases for CFD Testing
from .validation_cases import (
    ValidationCase, ValidationResult, ValidationSuite, FlowConditions, GeometryParameters,
    ObliqueShockCase, BowShockCase, NozzleFlowCase, ShockTubeCase,
    CaseType, FlowRegime
)

# Shape Optimization Framework Using Adjoint Gradients
from .shape_optimization import (
    ShapeOptimizer, DesignParameterization, FreeFormDeformation, BSplineParameterization,
    DesignVariable, OptimizationConstraint, OptimizationResult, OptimizationConfig,
    ParameterizationType, OptimizationAlgorithm, ConstraintType,
    create_shape_optimizer
)

# OpenFOAM polyMesh Reader
from .openfoam_mesh_reader import (
    OpenFOAMPolyMeshReader, OpenFOAMMeshData, OpenFOAMBoundaryPatch,
    read_openfoam_mesh
)

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