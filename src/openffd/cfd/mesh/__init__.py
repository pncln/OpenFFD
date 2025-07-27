"""
3D Mesh Data Structures for Supersonic Flow Solver

This module provides comprehensive 3D mesh data structures and connectivity
for unstructured finite volume CFD solvers with focus on supersonic flows.
"""

from .unstructured_mesh import UnstructuredMesh3D, CellType, BoundaryCondition
from .connectivity import ConnectivityManager, CellConnectivity
from .geometry import GeometricProperties, MetricTensor
from .boundary import BoundaryManager, BoundaryPatch
from .adaptation import MeshAdaptation

__all__ = [
    'UnstructuredMesh3D',
    'CellType', 
    'BoundaryCondition',
    'ConnectivityManager',
    'CellConnectivity',
    'GeometricProperties',
    'MetricTensor',
    'BoundaryManager',
    'BoundaryPatch',
    'MeshAdaptation'
]