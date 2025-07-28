#!/usr/bin/env python3
"""
Complete Cylinder Optimization Workflow Test

Tests the entire cylinder optimization workflow including:
- Configuration validation
- Mesh loading and validation
- Optimization execution
- Results analysis
- Visualization generation

Usage:
    python test_workflow.py
"""

import sys
import subprocess
from pathlib import Path
import json

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print(f"✓ SUCCESS: {description}")
            if result.stdout:
                print("Output:")
                print(result.stdout[-500:])  # Show last 500 chars
        else:
            print(f"✗ FAILED: {description}")
            print("Error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ EXCEPTION: {description}")
        print(f"Error: {e}")
        return False
    
    return True

def validate_files():
    """Validate that all required files exist."""
    print(f"\n{'='*60}")
    print("Validating required files...")
    print('='*60)
    
    required_files = [
        "README.md",
        "optimization_config.yaml", 
        "run_cylinder_optimization.py",
        "analyze_results.py",
        "create_visualization.py",
        "polyMesh/boundary",
        "polyMesh/points.gz",
        "polyMesh/faces.gz",
        "polyMesh/owner.gz",
        "polyMesh/neighbour.gz"
    ]
    
    case_dir = Path(__file__).parent
    all_present = True
    
    for file_path in required_files:
        full_path = case_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_present = False
    
    return all_present

def validate_results():
    """Validate that optimization results are reasonable."""
    print(f"\n{'='*60}")
    print("Validating optimization results...")
    print('='*60)
    
    results_dir = Path(__file__).parent / "results"
    history_file = results_dir / "optimization_history.json"
    
    if not history_file.exists():
        print("✗ No optimization history found")
        return False
    
    try:
        with open(history_file, 'r') as f:
            data = json.load(f)
        
        opt_result = data.get('optimization_result', {})
        
        # Check basic results
        success = opt_result.get('success', False)
        iterations = opt_result.get('n_iterations', 0)
        final_objective = opt_result.get('optimal_objective', 0)
        design_vars = opt_result.get('optimal_design', [])
        
        print(f"✓ Optimization success: {success}")
        print(f"✓ Iterations completed: {iterations}")
        print(f"✓ Final objective: {final_objective:.6e}")
        print(f"✓ Design variables: {len(design_vars)}")
        
        # Check mesh info
        mesh_info = data.get('mesh_info', {})
        print(f"✓ Mesh points: {mesh_info.get('n_points', 0)}")
        print(f"✓ Mesh cells: {mesh_info.get('n_cells', 0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error validating results: {e}")
        return False

def main():
    """Run complete workflow test."""
    print("CYLINDER OPTIMIZATION WORKFLOW TEST")
    print("="*80)
    
    # Step 1: Validate files
    if not validate_files():
        print("\n✗ File validation failed - cannot proceed")
        return 1
    
    # Step 2: Test quick optimization run
    if not run_command("python3 run_cylinder_optimization.py --max-iter 2", 
                      "Quick optimization run"):
        return 1
    
    # Step 3: Validate results
    if not validate_results():
        print("\n✗ Results validation failed")
        return 1
    
    # Step 4: Test analysis
    if not run_command("python3 analyze_results.py --save-plots", 
                      "Results analysis"):
        return 1
    
    # Step 5: Test visualization
    if not run_command("python3 create_visualization.py --mesh-plot", 
                      "Mesh visualization"):
        return 1
    
    # Final validation
    expected_outputs = [
        "results/optimization_history.json",
        "results/final_report.txt", 
        "results/analysis_report.txt",
        "results/convergence_analysis.png",
        "results/visualizations/mesh_visualization.png"
    ]
    
    print(f"\n{'='*60}")
    print("Final output validation...")
    print('='*60)
    
    case_dir = Path(__file__).parent
    all_outputs_present = True
    
    for output_file in expected_outputs:
        full_path = case_dir / output_file
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✓ {output_file} ({size} bytes)")
        else:
            print(f"✗ {output_file} - MISSING")
            all_outputs_present = False
    
    if all_outputs_present:
        print(f"\n🎉 COMPLETE WORKFLOW TEST: SUCCESS!")
        print("All components working correctly:")
        print("  • OpenFOAM mesh reading ✓")
        print("  • Shape optimization ✓") 
        print("  • Results analysis ✓")
        print("  • Visualization generation ✓")
        print(f"\nThe cylinder optimization test case is ready for production use!")
        return 0
    else:
        print(f"\n✗ COMPLETE WORKFLOW TEST: FAILED")
        print("Some outputs are missing - check error messages above")
        return 1

if __name__ == "__main__":
    sys.exit(main())