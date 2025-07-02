#!/usr/bin/env python3
"""
Example script showing how to run OpenFFD benchmarks

This script demonstrates various ways to run the benchmark suite
for different use cases.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print(f"✓ SUCCESS: {description}")
            print("Output:")
            print(result.stdout[-500:])  # Show last 500 characters
        else:
            print(f"✗ FAILED: {description}")
            print("Error output:")
            print(result.stderr[-500:])  # Show last 500 characters
            
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")

def main():
    """Run example benchmarks."""
    python_cmd = sys.executable
    
    print("OpenFFD Benchmark Suite Examples")
    print("================================")
    
    # Example 1: Quick test of single benchmark
    run_command([
        python_cmd, "benchmark_runner.py", 
        "--limited", 
        "--benchmarks", "parallelization",
        "--output-dir", "example_results_1",
        "--sequential"
    ], "Quick test of parallelization benchmark")
    
    # Example 2: Limited test of multiple benchmarks  
    run_command([
        python_cmd, "benchmark_runner.py",
        "--limited",
        "--benchmarks", "parallelization", "mesh_complexity", 
        "--output-dir", "example_results_2",
        "--sequential"
    ], "Limited test of multiple benchmarks")
    
    # Example 3: Quick comprehensive benchmark
    run_command([
        python_cmd, "benchmark_runner.py",
        "--quick",
        "--output-dir", "example_results_3"
    ], "Quick comprehensive benchmark (all modules)")
    
    # Example 4: Individual benchmark run
    run_command([
        python_cmd, "parallelization_scalability.py",
        "--quick",
        "--output-dir", "individual_parallelization_results"
    ], "Individual parallelization benchmark")
    
    print(f"\n{'='*60}")
    print("BENCHMARK EXAMPLES COMPLETED")
    print('='*60)
    print("\nResults are available in:")
    print("- example_results_1/ (single benchmark)")
    print("- example_results_2/ (multiple benchmarks)")  
    print("- example_results_3/ (comprehensive)")
    print("- individual_parallelization_results/ (individual)")
    print("\nFor your AIAA paper, run:")
    print("python3 benchmark_runner.py --output-dir aiaa_paper_results")
    print("(This will take 4-6 hours for full analysis)")

if __name__ == "__main__":
    main()