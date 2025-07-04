#!/usr/bin/env python3
"""
Test all imports to ensure the CFD module integrates correctly.
"""

def test_all_imports():
    """Test that all OpenFFD CFD components can be imported."""
    
    print("Testing OpenFFD CFD imports...")
    
    # Test main package imports
    try:
        from openffd import (
            OpenFOAMSolver, OpenFOAMConfig, SolverType, TurbulenceModel,
            SensitivityAnalyzer, SensitivityConfig, GradientMethod,
            MeshConverter, PostProcessor, ParallelManager
        )
        print("✓ Main package imports successful")
    except ImportError as e:
        print(f"❌ Main package import failed: {e}")
        return False
    
    # Test CFD module imports
    try:
        from openffd.cfd import (
            OpenFOAMSolver, OpenFOAMConfig, SolverType, TurbulenceModel,
            SensitivityAnalyzer, SensitivityConfig, GradientMethod, ObjectiveFunction,
            MeshConverter, PostProcessor, ParallelManager
        )
        print("✓ CFD module imports successful")
    except ImportError as e:
        print(f"❌ CFD module import failed: {e}")
        return False
    
    # Test base classes
    try:
        from openffd.cfd.base import (
            CFDSolver, CFDConfig, CFDResults, SolverStatus,
            ConvergenceData, ObjectiveFunction, SolverInterface
        )
        print("✓ Base classes import successful")
    except ImportError as e:
        print(f"❌ Base classes import failed: {e}")
        return False
    
    # Test OpenFOAM specific
    try:
        from openffd.cfd.openfoam import (
            OpenFOAMSolver, OpenFOAMConfig, SolverType, TurbulenceModel,
            BoundaryCondition, SimulationResults, FieldData, 
            ForceCoefficients, ResidualData
        )
        print("✓ OpenFOAM imports successful")
    except ImportError as e:
        print(f"❌ OpenFOAM import failed: {e}")
        return False
    
    # Test sensitivity analysis
    try:
        from openffd.cfd.sensitivity import (
            SensitivityAnalyzer, AdjointSolver, SensitivityConfig,
            GradientComputation, GradientMethod, FiniteDifferenceConfig,
            SensitivityResults
        )
        print("✓ Sensitivity analysis imports successful")
    except ImportError as e:
        print(f"❌ Sensitivity analysis import failed: {e}")
        return False
    
    # Test utilities
    try:
        from openffd.cfd.utilities import (
            MeshConverter, CaseGenerator, FieldProcessor,
            PostProcessor, ParallelManager, ResultsExtractor
        )
        print("✓ Utilities imports successful")
    except ImportError as e:
        print(f"❌ Utilities import failed: {e}")
        return False
    
    # Test mesh integration
    try:
        from openffd.mesh import (
            MeshDeformationEngine, DeformationConfig, 
            MeshQualityAnalyzer, VTKCellType
        )
        print("✓ Mesh integration imports successful")
    except ImportError as e:
        print(f"❌ Mesh integration import failed: {e}")
        return False
    
    # Test enum values
    try:
        assert SolverType.SIMPLE_FOAM.value == "simpleFoam"
        assert TurbulenceModel.K_OMEGA_SST.value == "kOmegaSST"
        assert GradientMethod.FINITE_DIFFERENCE_CENTRAL.value == "fd_central"
        assert ObjectiveFunction.DRAG_COEFFICIENT.value == "drag_coefficient"
        print("✓ Enum values correct")
    except Exception as e:
        print(f"❌ Enum validation failed: {e}")
        return False
    
    print("\n🎉 All imports successful!")
    return True

if __name__ == "__main__":
    success = test_all_imports()
    exit(0 if success else 1)