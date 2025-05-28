"""OpenFOAM solver interface for OpenFFD.

This module provides interfaces for connecting OpenFFD with OpenFOAM solvers,
particularly focused on sonicFoam with adjoint capabilities for supersonic flows.
"""

from .interface import OpenFOAMInterface
from .mesh_adapter import OpenFOAMMeshAdapter

__all__ = ["OpenFOAMInterface", "OpenFOAMMeshAdapter"]
