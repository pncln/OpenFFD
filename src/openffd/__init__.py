"""
OpenFFD - Open-Source FFD Control Box Generator.

A robust tool for generating Free-Form Deformation (FFD) control boxes for computational mesh files.
This framework enables precise shape manipulation for aerodynamic optimization, structural analysis, 
and design workflows.
"""

__author__ = "Emil Mammadli"

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "1.0.0.dev0"

from openffd.core import create_ffd_box
from openffd.io import write_ffd_3df, write_ffd_xyz

# Import CFD solver interface
from openffd.cfd import (
    OpenFOAMSolver, OpenFOAMConfig, SolverType, TurbulenceModel,
    SensitivityAnalyzer, SensitivityConfig, GradientMethod,
    MeshConverter, PostProcessor, ParallelManager
)

__all__ = [
    "create_ffd_box", "write_ffd_3df", "write_ffd_xyz",
    "OpenFOAMSolver", "OpenFOAMConfig", "SolverType", "TurbulenceModel",
    "SensitivityAnalyzer", "SensitivityConfig", "GradientMethod",
    "MeshConverter", "PostProcessor", "ParallelManager"
]
