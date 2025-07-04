"""
OpenFFD CFD Solver Interface Module

This module provides comprehensive CFD solver integration capabilities for OpenFFD including:
- OpenFOAM solver interface and automation
- CFD simulation setup and execution
- Result extraction and post-processing
- Sensitivity computation for optimization
- Multi-physics solver support
- Parallel execution management
"""

# Import core CFD functionality
from .openfoam import (
    OpenFOAMSolver,
    OpenFOAMConfig,
    SolverType,
    TurbulenceModel,
    BoundaryCondition,
    SimulationResults,
    FieldData,
    ForceCoefficients,
    ResidualData
)

from .base import (
    CFDSolver,
    CFDConfig,
    CFDResults,
    SolverStatus,
    ConvergenceData,
    ObjectiveFunction,
    SolverInterface
)

from .sensitivity import (
    SensitivityAnalyzer,
    AdjointSolver,
    SensitivityConfig,
    GradientComputation,
    GradientMethod,
    FiniteDifferenceConfig,
    SensitivityResults
)

from .utilities import (
    MeshConverter,
    CaseGenerator,
    FieldProcessor,
    PostProcessor,
    ParallelManager,
    ResultsExtractor
)

__all__ = [
    # Core OpenFOAM functionality
    'OpenFOAMSolver',
    'OpenFOAMConfig', 
    'SolverType',
    'TurbulenceModel',
    'BoundaryCondition',
    'SimulationResults',
    'FieldData',
    'ForceCoefficients',
    'ResidualData',
    
    # Abstract base classes
    'CFDSolver',
    'CFDConfig',
    'CFDResults',
    'SolverStatus',
    'ConvergenceData',
    'ObjectiveFunction',
    'SolverInterface',
    
    # Sensitivity analysis
    'SensitivityAnalyzer',
    'AdjointSolver',
    'SensitivityConfig',
    'GradientComputation',
    'GradientMethod',
    'FiniteDifferenceConfig',
    'SensitivityResults',
    
    # Utilities
    'MeshConverter',
    'CaseGenerator',
    'FieldProcessor',
    'PostProcessor',
    'ParallelManager',
    'ResultsExtractor'
]