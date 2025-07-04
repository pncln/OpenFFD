#!/usr/bin/env python3
"""
Universal CFD Optimization Script

This script demonstrates how to run optimization on any OpenFOAM case
using the new universal CFD optimization framework.

Usage:
    python run_optimization.py [config_file]

If no config file is specified, it will look for case_config.yaml in the current directory.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from openffd.cfd.optimization.optimizer import UniversalOptimizer


def main():
    """Main optimization function."""
    # Get configuration file
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        # Try JSON first, then YAML
        if Path("case_config.json").exists():
            config_file = "case_config.json"
        elif Path("case_config.yaml").exists():
            config_file = "case_config.yaml"
        else:
            config_file = "case_config.json"  # Default to JSON
    
    # Current case directory
    case_path = Path.cwd()
    
    print("=" * 80)
    print("Universal CFD Optimization Framework")
    print("=" * 80)
    print(f"Case directory: {case_path}")
    print(f"Configuration file: {config_file}")
    print()
    
    try:
        # Initialize universal optimizer
        print("Initializing universal optimizer...")
        optimizer = UniversalOptimizer(case_path, config_file)
        
        # Print optimization setup information
        info = optimizer.get_optimization_info()
        print("Optimization Setup:")
        print(f"  Case type: {info['case_type']}")
        print(f"  Solver: {info['solver']}")
        print(f"  Objectives: {', '.join(info['objectives'])}")
        print(f"  Algorithm: {info['algorithm']}")
        print(f"  Max iterations: {info['max_iterations']}")
        print(f"  Tolerance: {info['tolerance']}")
        print()
        
        # Run optimization
        print("Starting optimization...")
        results = optimizer.optimize()
        
        # Print results summary
        print("=" * 80)
        print("Optimization Results")
        print("=" * 80)
        print(f"Success: {results['success']}")
        print(f"Final objective: {results['final_objective']:.6f}")
        print(f"Iterations: {results['iterations']}")
        print(f"Total time: {results['total_time']:.2f} seconds")
        print(f"Results saved to: {results.get('history_file', 'optimization_history.json')}")
        
        if results['success']:
            print("\nOptimization completed successfully! ✓")
        else:
            print("\nOptimization failed or did not converge. ✗")
        
        return 0 if results['success'] else 1
        
    except ImportError as e:
        if "yaml" in str(e).lower():
            print("\nYAML dependency missing. You have two options:")
            print("1. Install PyYAML: pip install PyYAML")
            print("2. Use the provided JSON configuration: case_config.json")
            print(f"\nTo use JSON: python {sys.argv[0]} case_config.json")
        else:
            print(f"\nImport error: {e}")
        return 1
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        return 130
        
    except Exception as e:
        print(f"\nOptimization failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())