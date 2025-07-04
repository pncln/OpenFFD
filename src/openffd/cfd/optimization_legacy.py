#!/usr/bin/env python3
"""
DEPRECATED: Legacy CFD Optimization Module

This module has been replaced by the new Universal CFD Optimization Framework.

The hardcoded NACA-specific logic has been replaced with a universal, configuration-driven
approach that works with any OpenFOAM case type.

MIGRATION GUIDE:
================

Old approach (deprecated):
```python
from openffd.cfd.optimization import OptimizationManager

manager = OptimizationManager()
result = manager.optimize_naca0012()
```

New approach (universal):
```python
from openffd.cfd.optimization.optimizer import UniversalOptimizer

# Works with any OpenFOAM case
optimizer = UniversalOptimizer("/path/to/openfoam/case")
results = optimizer.optimize()
```

New approach (configuration-based):
```python
from openffd.cfd.optimization.optimizer import UniversalOptimizer

# Use custom configuration
optimizer = UniversalOptimizer(".", "case_config.yaml")
results = optimizer.optimize()
```

For complete examples, see examples/naca0012_case/

REMOVED HARDCODED LOGIC:
========================
- NACA 0012 specific airfoil generation
- Fixed boundary conditions for airfoil flow
- Hardcoded mesh parameters
- Fixed domain size and topology
- Specific force coefficient extraction
- Airfoil-only optimization workflows

UNIVERSAL REPLACEMENT:
======================
The new framework provides:
- Generic case handlers for any physics
- Configuration-driven setup
- Automatic case type detection
- Plugin architecture for extensibility
- Support for airfoil, heat transfer, pipe flow, and more
- User-defined objective functions
- Flexible FFD domain setup
- Runtime case type detection

This file is kept for backward compatibility but will be removed in future versions.
Please migrate to the new Universal CFD Optimization Framework.
"""

import warnings

def __getattr__(name):
    """Provide deprecation warnings for old imports."""
    warnings.warn(
        f"openffd.cfd.optimization.{name} is deprecated. "
        f"Use openffd.cfd.optimization.optimizer.UniversalOptimizer instead. "
        f"See examples/naca0012_case/ for migration examples.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Return None to avoid import errors but encourage migration
    return None

# Legacy compatibility
class OptimizationManager:
    """Deprecated: Use UniversalOptimizer instead."""
    
    def __init__(self):
        warnings.warn(
            "OptimizationManager is deprecated. "
            "Use openffd.cfd.optimization.optimizer.UniversalOptimizer instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    def optimize_naca0012(self, *args, **kwargs):
        """Deprecated: Use UniversalOptimizer with case_config.yaml instead."""
        raise NotImplementedError(
            "optimize_naca0012 has been removed. "
            "Use UniversalOptimizer with examples/naca0012_case/ as a template."
        )

# Provide migration help
def _print_migration_help():
    """Print migration help message."""
    print("=" * 80)
    print("MIGRATION TO UNIVERSAL CFD OPTIMIZATION FRAMEWORK")
    print("=" * 80)
    print("The hardcoded NACA optimization has been replaced with a universal framework.")
    print()
    print("Quick migration steps:")
    print("1. Copy examples/naca0012_case/ to your project")
    print("2. Modify case_config.yaml for your case")
    print("3. Run: python run_optimization.py")
    print()
    print("For details, see examples/naca0012_case/README.md")
    print("=" * 80)

if __name__ == "__main__":
    _print_migration_help()