"""
Discrete Adjoint Framework for CFD Optimization

This module provides a comprehensive discrete adjoint implementation for:
- Gradient-based shape optimization
- Sensitivity analysis 
- Design optimization
- Parameter studies

Key Components:
- Adjoint variable data structures
- Discrete adjoint equation derivation
- Adjoint boundary conditions
- Iterative adjoint solvers
- Gradient computation
"""

from .adjoint_variables import (
    AdjointVariables, AdjointState, AdjointConfig,
    create_adjoint_variables
)

from .adjoint_equations import (
    DiscreteAdjointSolver, AdjointLinearization,
    FluxJacobian, BoundaryJacobian
)

from .adjoint_boundary_conditions import (
    AdjointBoundaryConditions, AdjointBCType,
    FarfieldAdjointBC, WallAdjointBC, SymmetryAdjointBC
)

from .objective_functions import (
    ObjectiveFunction, DragObjective, LiftObjective,
    PressureLossObjective, CompositeObjective
)

from .gradient_computation import (
    SensitivityAnalysis, DesignGradient, DesignVariable,
    FiniteDifferenceValidation, GeometricSensitivity, SensitivityConfig
)

from .iterative_solvers import (
    AdjointIterativeSolver, GMRESAdjointSolver,
    BiCGSTABAdjointSolver, PreconditionedSolver
)

__all__ = [
    # Core adjoint framework
    'AdjointVariables', 'AdjointState', 'AdjointConfig', 'create_adjoint_variables',
    
    # Adjoint equations
    'DiscreteAdjointSolver', 'AdjointLinearization', 'FluxJacobian', 'BoundaryJacobian',
    
    # Boundary conditions
    'AdjointBoundaryConditions', 'AdjointBCType', 'FarfieldAdjointBC', 'WallAdjointBC', 'SymmetryAdjointBC',
    
    # Objective functions
    'ObjectiveFunction', 'DragObjective', 'LiftObjective', 'PressureLossObjective', 
    'CompositeObjective',
    
    # Gradient computation
    'SensitivityAnalysis', 'DesignGradient', 'DesignVariable', 'FiniteDifferenceValidation', 
    'GeometricSensitivity', 'SensitivityConfig',
    
    # Iterative solvers
    'AdjointIterativeSolver', 'GMRESAdjointSolver', 'BiCGSTABAdjointSolver', 'PreconditionedSolver'
]