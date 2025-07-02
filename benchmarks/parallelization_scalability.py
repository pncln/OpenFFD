#!/usr/bin/env python3
"""
Parallelization Scalability Benchmark for OpenFFD

This comprehensive benchmark script evaluates the scalability of OpenFFD's
parallel processing capabilities across varying mesh sizes and worker counts.
Designed for academic publication with publication-ready visualizations.

Author: OpenFFD Development Team
Purpose: AIAA Journal Performance Analysis
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogLocator, FuncFormatter
from scipy import stats

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openffd.core.control_box import create_ffd_box
from openffd.utils.parallel import ParallelConfig
from openffd.utils.benchmark import generate_random_mesh, benchmark_ffd_creation


class ParallelizationBenchmark:
    """Comprehensive parallelization scalability benchmark suite."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure academic-quality plotting
        self._setup_plotting()
        
        # Initialize logging
        self._setup_logging()
        
        # Benchmark configuration
        self.mesh_sizes = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
        self.worker_counts = [1, 2, 4, 6, 8, 12]
        self.control_dimensions = [(4, 4, 4), (6, 6, 6), (8, 8, 8), (10, 10, 10)]
        self.repetitions = 5  # For statistical significance
        
    def _setup_plotting(self):
        """Configure matplotlib for publication-quality plots."""
        plt.style.use('seaborn-v0_8')
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'axes.linewidth': 1.2,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.size': 5,
            'xtick.minor.size': 3,
            'ytick.major.size': 5,
            'ytick.minor.size': 3,
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': True,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none'
        })
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'benchmark.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def run_single_benchmark(
        self, 
        mesh_size: int, 
        workers: int, 
        control_dim: Tuple[int, int, int],
        method: str = 'process'
    ) -> Dict:
        """Run a single benchmark configuration."""
        self.logger.info(f"Running benchmark: {mesh_size} points, {workers} workers, {control_dim} control grid")
        
        # Generate mesh
        mesh_points = generate_random_mesh(mesh_size, seed=42)
        
        # Configure parallel processing
        parallel_config = ParallelConfig(
            enabled=workers > 1,
            method=method,
            max_workers=workers if workers > 1 else None,
            threshold=10_000
        )
        
        results = []
        for rep in range(self.repetitions):
            try:
                start_time = time.perf_counter()
                control_points, bbox = create_ffd_box(
                    mesh_points, control_dim, margin=0.1, 
                    custom_dims=None, parallel_config=parallel_config
                )
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                results.append({
                    'mesh_size': mesh_size,
                    'workers': workers,
                    'control_dim': control_dim,
                    'method': method,
                    'repetition': rep,
                    'execution_time': execution_time,
                    'num_control_points': len(control_points),
                    'points_per_second': mesh_size / execution_time,
                    'memory_efficiency': mesh_size * len(control_points) / execution_time
                })
                
            except Exception as e:
                self.logger.error(f"Error in benchmark: {e}")
                results.append({
                    'mesh_size': mesh_size,
                    'workers': workers,
                    'control_dim': control_dim,
                    'method': method,
                    'repetition': rep,
                    'execution_time': float('inf'),
                    'num_control_points': 0,
                    'points_per_second': 0,
                    'memory_efficiency': 0
                })
        
        return results
    
    def run_scalability_study(self) -> pd.DataFrame:
        """Run comprehensive scalability study."""
        self.logger.info("Starting comprehensive parallelization scalability study")
        
        all_results = []
        total_benchmarks = len(self.mesh_sizes) * len(self.worker_counts) * len(self.control_dimensions)
        completed = 0
        
        for mesh_size in self.mesh_sizes:
            for workers in self.worker_counts:
                for control_dim in self.control_dimensions:
                    results = self.run_single_benchmark(mesh_size, workers, control_dim)
                    all_results.extend(results)
                    
                    completed += 1
                    self.logger.info(f"Progress: {completed}/{total_benchmarks} ({completed/total_benchmarks*100:.1f}%)")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_results)
        
        # Calculate statistics
        df_stats = df.groupby(['mesh_size', 'workers', 'control_dim']).agg({
            'execution_time': ['mean', 'std', 'min', 'max'],
            'points_per_second': ['mean', 'std'],
            'memory_efficiency': ['mean', 'std']
        }).round(6)
        
        # Save raw results
        df.to_csv(self.output_dir / 'parallelization_raw_results.csv', index=False)
        df_stats.to_csv(self.output_dir / 'parallelization_statistics.csv')
        
        self.logger.info(f"Scalability study completed. Results saved to {self.output_dir}")
        return df
    
    def analyze_strong_scaling(self, df: pd.DataFrame) -> Dict:
        """Analyze strong scaling efficiency (fixed problem size, varying workers)."""
        self.logger.info("Analyzing strong scaling efficiency")
        
        strong_scaling_results = {}
        
        for mesh_size in self.mesh_sizes:
            for control_dim in self.control_dimensions:
                subset = df[(df['mesh_size'] == mesh_size) & 
                           (df['control_dim'] == control_dim)].copy()
                
                if subset.empty:
                    continue
                    
                # Calculate mean execution times
                mean_times = subset.groupby('workers')['execution_time'].mean()
                
                # Calculate efficiency relative to single worker
                if 1 in mean_times.index:
                    baseline_time = mean_times[1]
                    efficiencies = {}
                    speedups = {}
                    
                    for workers in mean_times.index:
                        speedup = baseline_time / mean_times[workers]
                        efficiency = speedup / workers
                        speedups[workers] = speedup
                        efficiencies[workers] = efficiency
                    
                    strong_scaling_results[f"{mesh_size}_{control_dim}"] = {
                        'mesh_size': mesh_size,
                        'control_dim': control_dim,
                        'execution_times': mean_times.to_dict(),
                        'speedups': speedups,
                        'efficiencies': efficiencies
                    }
        
        # Save strong scaling analysis
        with open(self.output_dir / 'strong_scaling_analysis.json', 'w') as f:
            json.dump(strong_scaling_results, f, indent=2, default=str)
            
        return strong_scaling_results
    
    def analyze_weak_scaling(self, df: pd.DataFrame) -> Dict:
        """Analyze weak scaling efficiency (proportional problem size increase)."""
        self.logger.info("Analyzing weak scaling efficiency")
        
        # For weak scaling, we need to find combinations where problem size per worker is constant
        weak_scaling_results = {}
        
        # Define base problem size per worker
        base_size_per_worker = 50_000
        
        for control_dim in self.control_dimensions:
            subset = df[df['control_dim'] == control_dim].copy()
            
            if subset.empty:
                continue
                
            weak_scaling_data = []
            
            for workers in self.worker_counts:
                target_mesh_size = workers * base_size_per_worker
                
                # Find closest mesh size
                closest_size = min(self.mesh_sizes, key=lambda x: abs(x - target_mesh_size))
                
                worker_data = subset[(subset['workers'] == workers) & 
                                   (subset['mesh_size'] == closest_size)]
                
                if not worker_data.empty:
                    mean_time = worker_data['execution_time'].mean()
                    weak_scaling_data.append({
                        'workers': workers,
                        'mesh_size': closest_size,
                        'size_per_worker': closest_size / workers,
                        'execution_time': mean_time
                    })
            
            if weak_scaling_data:
                weak_scaling_results[str(control_dim)] = weak_scaling_data
        
        # Save weak scaling analysis
        with open(self.output_dir / 'weak_scaling_analysis.json', 'w') as f:
            json.dump(weak_scaling_results, f, indent=2)
            
        return weak_scaling_results
    
    def plot_strong_scaling(self, strong_scaling_results: Dict):
        """Create publication-quality strong scaling plots."""
        self.logger.info("Creating strong scaling visualizations")
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.mesh_sizes)))
        
        # Create separate plots for each control dimension
        for idx, control_dim in enumerate(self.control_dimensions):
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot speedup curves for different mesh sizes
            for mesh_idx, mesh_size in enumerate(self.mesh_sizes):
                key = f"{mesh_size}_{control_dim}"
                if key in strong_scaling_results:
                    data = strong_scaling_results[key]
                    workers = list(data['speedups'].keys())
                    speedups = list(data['speedups'].values())
                    
                    ax.plot(workers, speedups, 'o-', color=colors[mesh_idx], 
                           label=f'{mesh_size:,} points', linewidth=2.5, markersize=8)
            
            # Add ideal scaling line
            max_workers = max(self.worker_counts)
            ideal_workers = list(range(1, max_workers + 1))
            ax.plot(ideal_workers, ideal_workers, 'k--', alpha=0.7, 
                   label='Ideal scaling', linewidth=2)
            
            ax.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
            ax.set_ylabel('Speedup', fontsize=12, fontweight='bold')
            ax.set_title(f'Strong Scaling - Control Grid: {control_dim[0]}×{control_dim[1]}×{control_dim[2]}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11, loc='upper left')
            ax.set_xlim(1, max_workers)
            
            plt.tight_layout()
            
            # Save without title for publications
            original_title = ax.get_title()
            ax.set_title('')
            plt.savefig(self.output_dir / f'strong_scaling_speedup_grid_{control_dim[0]}x{control_dim[1]}x{control_dim[2]}.png',
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(self.output_dir / f'strong_scaling_speedup_grid_{control_dim[0]}x{control_dim[1]}x{control_dim[2]}.pdf',
                       dpi=300, bbox_inches='tight', facecolor='white')
            ax.set_title(original_title)  # Restore for display
            plt.close()
        
        # Create separate efficiency plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for mesh_size in [100_000, 500_000, 1_000_000]:  # Select representative sizes
            control_dim = (8, 8, 8)  # Standard control grid
            key = f"{mesh_size}_{control_dim}"
            
            if key in strong_scaling_results:
                data = strong_scaling_results[key]
                workers = list(data['efficiencies'].keys())
                efficiencies = [e * 100 for e in data['efficiencies'].values()]  # Convert to percentage
                
                ax.plot(workers, efficiencies, 'o-', label=f'{mesh_size:,} points', 
                       linewidth=2.5, markersize=8)
        
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, 
                  label='80% Efficiency Threshold')
        ax.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
        ax.set_title('Parallel Efficiency Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        
        # Save without title for publications
        original_title = ax.get_title()
        ax.set_title('')
        plt.savefig(self.output_dir / 'parallel_efficiency.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'parallel_efficiency.pdf', dpi=300, bbox_inches='tight', facecolor='white')
        ax.set_title(original_title)  # Restore for display
        plt.close()
    
    
    def plot_throughput_analysis(self, df: pd.DataFrame):
        """Create separate throughput analysis plots."""
        self.logger.info("Creating throughput analysis")
        
        # Plot 1: Points per second analysis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mean_throughput = df.groupby(['mesh_size', 'workers'])['points_per_second'].mean().reset_index()
        
        for mesh_size in [100_000, 500_000, 1_000_000, 2_000_000]:
            subset = mean_throughput[mean_throughput['mesh_size'] == mesh_size]
            if not subset.empty:
                ax.plot(subset['workers'], subset['points_per_second'], 'o-', 
                        label=f'{mesh_size:,} points', linewidth=2.5, markersize=8)
        
        ax.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Throughput (points/second)', fontsize=12, fontweight='bold')
        ax.set_title('Processing Throughput vs Worker Count', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save without title for publications
        original_title = ax.get_title()
        ax.set_title('')
        plt.savefig(self.output_dir / 'processing_throughput.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'processing_throughput.pdf', dpi=300, bbox_inches='tight', facecolor='white')
        ax.set_title(original_title)  # Restore for display
        plt.close()
        
        # Plot 2: Memory efficiency analysis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mean_memory_eff = df.groupby(['workers', 'control_dim'])['memory_efficiency'].mean().reset_index()
        
        for control_dim in [(4,4,4), (6,6,6), (8,8,8), (10,10,10)]:
            subset = mean_memory_eff[mean_memory_eff['control_dim'] == control_dim]
            if not subset.empty:
                ax.plot(subset['workers'], subset['memory_efficiency'], 'o-', 
                        label=f'{control_dim[0]}×{control_dim[1]}×{control_dim[2]}', 
                        linewidth=2.5, markersize=8)
        
        ax.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Memory Efficiency (operations/second)', fontsize=12, fontweight='bold')
        ax.set_title('Memory Efficiency vs Control Grid Size', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save without title for publications
        original_title = ax.get_title()
        ax.set_title('')
        plt.savefig(self.output_dir / 'memory_efficiency.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'memory_efficiency.pdf', dpi=300, bbox_inches='tight', facecolor='white')
        ax.set_title(original_title)  # Restore for display
        plt.close()
    
    def generate_summary_report(self, df: pd.DataFrame, strong_scaling: Dict, weak_scaling: Dict):
        """Generate comprehensive summary report."""
        self.logger.info("Generating summary report")
        
        report = []
        report.append("# OpenFFD Parallelization Scalability Benchmark Report\n")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("## Executive Summary\n")
        
        # Calculate key metrics
        max_speedup = 0
        best_efficiency = 0
        optimal_workers = 1
        
        for key, data in strong_scaling.items():
            speedups = list(data['speedups'].values())
            efficiencies = list(data['efficiencies'].values())
            
            if speedups:
                max_speedup = max(max_speedup, max(speedups))
            if efficiencies:
                best_efficiency = max(best_efficiency, max(efficiencies))
                
        report.append(f"- Maximum observed speedup: {max_speedup:.2f}x\n")
        report.append(f"- Best parallel efficiency: {best_efficiency*100:.1f}%\n")
        
        # Performance recommendations
        report.append("\n## Performance Recommendations\n")
        
        # Analyze efficiency threshold
        efficient_configs = []
        for key, data in strong_scaling.items():
            for workers, efficiency in data['efficiencies'].items():
                if efficiency > 0.8:  # 80% efficiency threshold
                    efficient_configs.append((workers, data['mesh_size'], efficiency))
        
        if efficient_configs:
            efficient_configs.sort(key=lambda x: x[0])
            report.append("### Efficient Configurations (>80% efficiency):\n")
            for workers, mesh_size, eff in efficient_configs[:5]:  # Top 5
                report.append(f"- {workers} workers, {mesh_size:,} points: {eff*100:.1f}% efficiency\n")
        
        # Statistical analysis
        report.append("\n## Statistical Analysis\n")
        
        overall_stats = df.groupby('workers')['execution_time'].agg(['mean', 'std', 'count'])
        report.append("### Execution Time Statistics by Worker Count:\n")
        report.append("| Workers | Mean Time (s) | Std Dev (s) | Samples |\n")
        report.append("|---------|---------------|-------------|----------|\n")
        
        for workers, row in overall_stats.iterrows():
            report.append(f"| {workers} | {row['mean']:.3f} | {row['std']:.3f} | {row['count']} |\n")
        
        # Save report
        with open(self.output_dir / 'benchmark_report.md', 'w') as f:
            f.writelines(report)
        
        self.logger.info(f"Summary report saved to {self.output_dir}/benchmark_report.md")
    
    def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        self.logger.info("Starting full parallelization scalability benchmark")
        
        # Run scalability study
        df = self.run_scalability_study()
        
        # Analyze scaling performance
        strong_scaling = self.analyze_strong_scaling(df)
        weak_scaling = self.analyze_weak_scaling(df)
        
        # Generate visualizations
        self.plot_strong_scaling(strong_scaling)
        self.plot_throughput_analysis(df)
        
        # Generate summary report
        self.generate_summary_report(df, strong_scaling, weak_scaling)
        
        self.logger.info("Full benchmark completed successfully!")
        self.logger.info(f"Results available in: {self.output_dir}")


def main():
    """Main entry point for parallelization benchmark."""
    parser = argparse.ArgumentParser(
        description='Comprehensive parallelization scalability benchmark for OpenFFD'
    )
    parser.add_argument(
        '--output-dir', default='parallelization_benchmark_results',
        help='Output directory for benchmark results'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick benchmark with reduced test cases'
    )
    parser.add_argument(
        '--max-workers', type=int, default=24,
        help='Maximum number of workers to test'
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = ParallelizationBenchmark(args.output_dir)
    
    # Adjust parameters for quick run
    if args.quick:
        benchmark.mesh_sizes = [50_000, 250_000, 1_000_000]
        benchmark.worker_counts = [1, 2, 4, 8]
        benchmark.control_dimensions = [(4, 4, 4), (8, 8, 8)]
        benchmark.repetitions = 3
    
    # Limit worker counts based on argument
    benchmark.worker_counts = [w for w in benchmark.worker_counts if w <= args.max_workers]
    
    # Run benchmark
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()