"""
High-Resolution Numerical Schemes for CFD

This module provides advanced numerical methods for supersonic flow computation:
- Shock-capturing schemes (WENO, TVD, ENO)
- Riemann solvers for flux computation
- Shock detection algorithms
- Limiters for monotonicity preservation
"""

# High-order reconstruction schemes
from .weno_schemes import (
    WENO3, WENO5, WENO7, WENOReconstructor,
    create_weno_reconstructor
)

from .tvd_schemes import (
    MinModLimiter, SuperBeeLimiter, VanLeerLimiter,
    MonotonizedCentralLimiter, KorenLimiter, OspreLimiter,
    TVDReconstructor, SlopeLimiter, MonotonicityPreserver,
    create_tvd_reconstructor
)

from .eno_schemes import (
    ENO2, ENO3, ENO5, ENOReconstructor, DividedDifferences,
    create_eno_reconstructor
)

# Riemann solvers
from .riemann_solvers import (
    RoeRiemannSolver, HLLCRiemannSolver, RusanovRiemannSolver,
    AUSMPlusRiemannSolver, ExactRiemannSolver,
    RiemannSolverManager, create_riemann_solver
)

# Shock detection
from .shock_detectors import (
    DucrosShockSensor, JamesonShockSensor, PressureJumpDetector,
    MachNumberDetector, CompositeShockDetector,
    create_default_shock_detector, smooth_shock_indicator
)

# Advanced limiters
from .limiters import (
    BartkLimiter, VenkatakrishnanLimiter, MichalakGoochLimiter,
    PositivityPreserver, ShockAdaptiveLimiter, MultiDimensionalLimiter,
    FluxLimiter, MonotonicityPreserver as AdvancedMonotonicityPreserver,
    create_limiter
)

__all__ = [
    # WENO schemes
    'WENO3', 'WENO5', 'WENO7', 'WENOReconstructor', 'create_weno_reconstructor',
    
    # TVD schemes  
    'MinModLimiter', 'SuperBeeLimiter', 'VanLeerLimiter',
    'MonotonizedCentralLimiter', 'KorenLimiter', 'OspreLimiter',
    'TVDReconstructor', 'SlopeLimiter', 'MonotonicityPreserver',
    'create_tvd_reconstructor',
    
    # ENO schemes
    'ENO2', 'ENO3', 'ENO5', 'ENOReconstructor', 'DividedDifferences',
    'create_eno_reconstructor',
    
    # Riemann solvers
    'RoeRiemannSolver', 'HLLCRiemannSolver', 'RusanovRiemannSolver',
    'AUSMPlusRiemannSolver', 'ExactRiemannSolver',
    'RiemannSolverManager', 'create_riemann_solver',
    
    # Shock detection
    'DucrosShockSensor', 'JamesonShockSensor', 'PressureJumpDetector',
    'MachNumberDetector', 'CompositeShockDetector',
    'create_default_shock_detector', 'smooth_shock_indicator',
    
    # Advanced limiters
    'BartkLimiter', 'VenkatakrishnanLimiter', 'MichalakGoochLimiter',
    'PositivityPreserver', 'ShockAdaptiveLimiter', 'MultiDimensionalLimiter',
    'FluxLimiter', 'AdvancedMonotonicityPreserver', 'create_limiter'
]