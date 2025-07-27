"""
3D Compressible Flow Equations for Supersonic CFD

This module provides implementation of the Euler and Navier-Stokes equations
using finite volume method for unstructured meshes.
"""

from .euler_equations import EulerEquations3D
from .navier_stokes import NavierStokesEquations3D
from .equation_state import EquationOfState, PerfectGas
from .flux_functions import InviscidFlux, ViscousFlux
from .primitive_conservative import ConservativeVariables, PrimitiveVariables

__all__ = [
    'EulerEquations3D',
    'NavierStokesEquations3D', 
    'EquationOfState',
    'PerfectGas',
    'InviscidFlux',
    'ViscousFlux',
    'ConservativeVariables',
    'PrimitiveVariables'
]