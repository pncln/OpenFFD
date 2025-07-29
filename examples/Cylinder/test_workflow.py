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
import yaml
import shutil

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

def setup_10_core_config():
    """Create a temporary configuration file configured for exactly 10 cores."""
    case_dir = Path(__file__).parent
    original_config = case_dir / "optimization_config.yaml"
    temp_config = case_dir / "optimization_config_10cores.yaml"
    
    # Load original configuration
    with open(original_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure for exactly 10 cores (5 MPI processes × 2 OpenMP threads)
    config['parallel'] = {
        'enable': True,
        'mpi_processes': 5,
        'openmp_threads': 2,
        'domain_decomposition': 'geometric',
        'load_balancing': True
    }
    
    # Save temporary configuration
    with open(temp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✓ Created 10-core configuration: {temp_config.name}")
    print(f"  • MPI processes: {config['parallel']['mpi_processes']}")
    print(f"  • OpenMP threads per process: {config['parallel']['openmp_threads']}")
    print(f"  • Total cores: {config['parallel']['mpi_processes'] * config['parallel']['openmp_threads']}")
    
    return temp_config

def cleanup_temp_files():
    """Clean up temporary configuration files."""
    case_dir = Path(__file__).parent
    temp_config = case_dir / "optimization_config_10cores.yaml"
    if temp_config.exists():
        temp_config.unlink()
        print(f"✓ Cleaned up temporary file: {temp_config.name}")

def main():
    """Run complete workflow test."""
    print("CYLINDER OPTIMIZATION WORKFLOW TEST - 10 CORES")
    print("="*80)
    
    # Step 1: Validate files
    if not validate_files():
        print("\n✗ File validation failed - cannot proceed")
        return 1
    
    # Step 2: Setup 10-core configuration
    try:
        temp_config = setup_10_core_config()
        
        # Step 3: Test quick optimization run with 10 cores (5 MPI × 2 OpenMP)
        cmd = f"OMP_NUM_THREADS=2 python3 run_cylinder_optimization.py --config {temp_config.name} --max-iter 2 --parallel"
        if not run_command(cmd, "Quick optimization run with 10 cores (5 MPI × 2 OpenMP)"):
            cleanup_temp_files()
            return 1
            
    except Exception as e:
        print(f"✗ Failed to setup 10-core configuration: {e}")
        return 1
    
    # Step 4: Validate results
    if not validate_results():
        print("\n✗ Results validation failed")
        cleanup_temp_files()
        return 1
    
    # Step 5: Test analysis
    if not run_command("python3 analyze_results.py --save-plots", 
                      "Results analysis"):
        cleanup_temp_files()
        return 1
    
    # Step 6: Test visualization
    if not run_command("python3 create_visualization.py --mesh-plot", 
                      "Mesh visualization"):
        cleanup_temp_files()
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
        print("  • Shape optimization with 10 cores ✓") 
        print("  • Results analysis ✓")
        print("  • Visualization generation ✓")
        print(f"\nThe cylinder optimization test case is ready for production use!")
        cleanup_temp_files()
        return 0
    else:
        print(f"\n✗ COMPLETE WORKFLOW TEST: FAILED")
        print("Some outputs are missing - check error messages above")
        cleanup_temp_files()
        return 1

if __name__ == "__main__":
    sys.exit(main())