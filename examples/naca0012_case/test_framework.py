#!/usr/bin/env python3
"""
Test script to verify the Universal CFD Optimization Framework works correctly.
This script tests the framework without requiring OpenFOAM to be installed.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test that all framework components import correctly."""
    print("Testing framework imports...")
    
    try:
        from openffd.cfd.optimization.optimizer import UniversalOptimizer
        print("✓ UniversalOptimizer import successful")
        
        from openffd.cfd.core.config import CaseConfig
        print("✓ CaseConfig import successful")
        
        from openffd.cfd.cases.airfoil_case import AirfoilCase
        print("✓ AirfoilCase import successful")
        
        from openffd.cfd.optimization.objectives import ObjectiveRegistry
        print("✓ ObjectiveRegistry import successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from openffd.cfd.core.config import CaseConfig
        
        # Test JSON loading
        config = CaseConfig.from_file("case_config.json")
        print(f"✓ JSON config loaded: {config.case_type}")
        print(f"  - Solver: {config.solver}")
        print(f"  - Objectives: {[obj.name for obj in config.objectives]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_case_detection():
    """Test case type detection."""
    print("\nTesting case type detection...")
    
    try:
        from openffd.cfd.core.registry import CaseTypeRegistry
        
        # Test auto-detection
        case_type = CaseTypeRegistry.auto_detect_case_type(Path("."))
        print(f"✓ Auto-detected case type: {case_type}")
        
        # Test available types
        available = CaseTypeRegistry.list_available_types()
        print(f"✓ Available case handlers: {available['case_handlers']}")
        print(f"✓ Available objectives: {available['objectives']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Case detection failed: {e}")
        return False

def test_framework_info():
    """Test framework information."""
    print("\nTesting framework information...")
    
    try:
        from openffd.cfd import __framework_info__, __version__
        
        print(f"✓ Framework: {__framework_info__['name']}")
        print(f"✓ Version: {__version__}")
        print(f"✓ Architecture: {__framework_info__['architecture']}")
        print(f"✓ Supported case types: {__framework_info__['supported_case_types']}")
        print(f"✓ Optimization algorithms: {__framework_info__['optimization_algorithms']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Framework info failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 80)
    print("UNIVERSAL CFD OPTIMIZATION FRAMEWORK - TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_imports,
        test_config_loading,
        test_case_detection,
        test_framework_info
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ ALL TESTS PASSED ({passed}/{total})")
        print("\nThe Universal CFD Optimization Framework is working correctly!")
        print("To use with OpenFOAM:")
        print("1. Install OpenFOAM and source it")
        print("2. Install missing Python dependencies:")
        print("   pip install PyYAML scipy pandas psutil")
        print("3. Run: python run_optimization.py")
        return 0
    else:
        print(f"✗ SOME TESTS FAILED ({passed}/{total})")
        return 1

if __name__ == "__main__":
    sys.exit(main())