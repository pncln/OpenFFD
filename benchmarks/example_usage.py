#!/usr/bin/env python3
"""
Example usage of the OpenFFD benchmark suite.

This script demonstrates how to run individual benchmarks
and create custom analyses.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_config import QUICK_CONFIG
from benchmarks.data_generator import MeshGenerator
from benchmarks.ffd_benchmark import FFDBenchmark
from benchmarks.hffd_benchmark import HFFDBenchmark
from benchmarks.visualization import AcademicPlotter

def example_single_ffd_benchmark():
    """Example: Run a single FFD benchmark."""
    print("="*50)
    print("EXAMPLE 1: Single FFD Benchmark")
    print("="*50)
    
    # Generate test mesh
    mesh_points = MeshGenerator.generate_sphere_surface(50_000, seed=42)
    print(f"Generated sphere with {len(mesh_points):,} points")
    
    # Create FFD benchmark
    config = QUICK_CONFIG
    benchmark = FFDBenchmark(config)
    
    # Run single test
    from openffd.utils.parallel import ParallelConfig
    parallel_config = ParallelConfig(enabled=False)
    
    result = benchmark._run_single_ffd(
        mesh_points=mesh_points,
        control_dim=(6, 6, 6),
        parallel_config=parallel_config,
        geometry_type="sphere"
    )
    
    # Print results
    if result.success:
        print(f"✅ FFD benchmark successful!")
        print(f"   Execution time: {result.execution_time:.3f} seconds")
        print(f"   Memory usage: {result.memory_peak_mb:.1f} MB")
        print(f"   Control points: {result.n_control_points}")
    else:
        print(f"❌ FFD benchmark failed: {result.error_message}")

def example_mesh_size_comparison():
    """Example: Compare FFD performance across mesh sizes."""
    print("\n" + "="*50)
    print("EXAMPLE 2: Mesh Size Comparison")
    print("="*50)
    
    mesh_sizes = [1_000, 5_000, 10_000, 25_000]
    results = []
    
    config = QUICK_CONFIG
    benchmark = FFDBenchmark(config)
    
    from openffd.utils.parallel import ParallelConfig
    parallel_config = ParallelConfig(enabled=False)
    
    for size in mesh_sizes:
        print(f"Testing mesh size: {size:,} points")
        
        # Generate test mesh
        mesh_points = MeshGenerator.generate_random_points(size, seed=42)
        
        # Run benchmark
        result = benchmark._run_single_ffd(
            mesh_points=mesh_points,
            control_dim=(6, 6, 6),
            parallel_config=parallel_config,
            geometry_type="random"
        )
        
        if result.success:
            results.append({
                'mesh_size': size,
                'execution_time': result.execution_time,
                'memory_mb': result.memory_peak_mb
            })
            print(f"  ✅ {result.execution_time:.3f}s, {result.memory_peak_mb:.1f} MB")
        else:
            print(f"  ❌ Failed: {result.error_message}")
    
    # Create simple plot
    if results:
        df = pd.DataFrame(results)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Execution time plot
        ax1.loglog(df['mesh_size'], df['execution_time'], 'o-', 
                   color='blue', markersize=8)
        ax1.set_xlabel('Mesh Size (Points)')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('FFD Performance Scaling')
        ax1.grid(True, alpha=0.3)
        
        # Memory usage plot
        ax2.loglog(df['mesh_size'], df['memory_mb'], 's-', 
                   color='red', markersize=8)
        ax2.set_xlabel('Mesh Size (Points)')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('FFD Memory Scaling')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_file = Path("benchmarks/figures/example_scaling.png")
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n📊 Scaling plot saved to: {output_file}")
        
        plt.show()

def example_hffd_hierarchy_analysis():
    """Example: Analyze HFFD hierarchy performance."""
    print("\n" + "="*50)
    print("EXAMPLE 3: HFFD Hierarchy Analysis")
    print("="*50)
    
    config = QUICK_CONFIG
    benchmark = HFFDBenchmark(config)
    
    # Generate test mesh
    mesh_points = MeshGenerator.generate_wing_profile(25_000, seed=42)
    print(f"Generated wing profile with {len(mesh_points):,} points")
    
    from openffd.utils.parallel import ParallelConfig
    parallel_config = ParallelConfig(enabled=False)
    
    # Test different hierarchy depths
    depths = [2, 3, 4]
    results = []
    
    for depth in depths:
        print(f"Testing hierarchy depth: {depth}")
        
        result = benchmark._run_single_hffd(
            mesh_points=mesh_points,
            base_dims=(4, 4, 4),
            max_depth=depth,
            subdivision_factor=2,
            parallel_config=parallel_config,
            geometry_type="wing"
        )
        
        if result.success:
            results.append({
                'depth': depth,
                'execution_time': result.execution_time,
                'total_levels': result.total_levels,
                'total_control_points': result.total_control_points,
                'complexity': result.hierarchy_complexity
            })
            print(f"  ✅ Depth {depth}: {result.execution_time:.3f}s, "
                  f"{result.total_control_points} control points, "
                  f"{result.hierarchy_complexity:.1f}x complexity")
        else:
            print(f"  ❌ Failed: {result.error_message}")
    
    # Print summary
    if results:
        print(f"\n📈 HFFD Hierarchy Analysis Summary:")
        for r in results:
            print(f"   Depth {r['depth']}: {r['execution_time']:.3f}s, "
                  f"{r['total_control_points']} points, "
                  f"{r['complexity']:.1f}x complexity")

def example_custom_visualization():
    """Example: Create custom visualization."""
    print("\n" + "="*50)
    print("EXAMPLE 4: Custom Visualization")
    print("="*50)
    
    # Generate sample data
    np.random.seed(42)
    data = {
        'mesh_size': [1000, 5000, 10000, 25000, 50000],
        'ffd_time': [0.001, 0.005, 0.012, 0.035, 0.078],
        'hffd_time': [0.003, 0.015, 0.038, 0.095, 0.210],
        'ffd_memory': [12, 25, 48, 95, 180],
        'hffd_memory': [18, 35, 65, 125, 240]
    }
    
    # Create custom academic-style plots (separate figures, no titles)
    config = QUICK_CONFIG
    plotter = AcademicPlotter(config)
    
    # Figure 1: Performance comparison
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    ax1.loglog(data['mesh_size'], data['ffd_time'], 'o-', 
               color=plotter.colors['ffd'], linewidth=2, 
               markersize=8, label='FFD')
    ax1.loglog(data['mesh_size'], data['hffd_time'], 's-', 
               color=plotter.colors['hffd'], linewidth=2, 
               markersize=8, label='HFFD')
    
    ax1.set_xlabel('Mesh Size (Number of Points)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save performance figure
    output_file1 = Path("benchmarks/figures/custom_performance.png")
    output_file1.parent.mkdir(exist_ok=True)
    plotter.save_figure(fig1, "custom_performance", ["png", "pdf"])
    plt.close(fig1)
    
    # Figure 2: Memory comparison
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    
    ax2.loglog(data['mesh_size'], data['ffd_memory'], 'o-', 
               color=plotter.colors['ffd'], linewidth=2, 
               markersize=8, label='FFD')
    ax2.loglog(data['mesh_size'], data['hffd_memory'], 's-', 
               color=plotter.colors['hffd'], linewidth=2, 
               markersize=8, label='HFFD')
    
    ax2.set_xlabel('Mesh Size (Number of Points)')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save memory figure
    output_file2 = Path("benchmarks/figures/custom_memory.png")
    plotter.save_figure(fig2, "custom_memory", ["png", "pdf"])
    plt.close(fig2)
    
    print(f"📊 Custom performance figure saved to: {output_file1}")
    print(f"📊 Custom memory figure saved to: {output_file2}")
    print("💡 Note: Figures are created separately without titles for publication flexibility")

def main():
    """Run all examples."""
    print("OpenFFD Benchmark Suite - Example Usage")
    print("=" * 60)
    
    try:
        # Run examples
        example_single_ffd_benchmark()
        example_mesh_size_comparison()
        example_hffd_hierarchy_analysis()
        example_custom_visualization()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("📊 Check the benchmarks/figures/ directory for saved plots.")
        print("💡 Run 'python benchmarks/run_benchmarks.py --config quick' for full benchmark suite.")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()