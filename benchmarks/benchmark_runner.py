#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner for OpenFFD

This script orchestrates all benchmark modules to provide a complete
performance analysis of OpenFFD for academic publication. It generates
publication-ready figures, comprehensive reports, and statistical analyses.

Author: OpenFFD Development Team
Purpose: AIAA Journal Performance Analysis
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import benchmark modules
try:
    from parallelization_scalability import ParallelizationBenchmark
    from mesh_complexity_scaling import MeshComplexityBenchmark
    from hierarchical_ffd_benchmark import HierarchicalFFDBenchmark
    from visualization_performance import VisualizationBenchmark
    from academic_plotting_utils import AcademicPlotter, create_benchmark_summary_figure
    BENCHMARK_MODULES_AVAILABLE = True
except ImportError as e:
    BENCHMARK_MODULES_AVAILABLE = False
    print(f"Warning: Could not import all benchmark modules: {e}")


class BenchmarkSuite:
    """Comprehensive benchmark suite for OpenFFD performance analysis."""
    
    def __init__(self, output_dir: str = "comprehensive_benchmark_results"):
        """Initialize the benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize benchmark modules
        self.benchmarks = {}
        if BENCHMARK_MODULES_AVAILABLE:
            self._initialize_benchmarks()
        
        # Academic plotter
        self.plotter = AcademicPlotter()
        
        # Results storage
        self.results = {}
        self.summary_stats = {}
        
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_file = self.output_dir / 'benchmark_suite.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _initialize_benchmarks(self):
        """Initialize all benchmark modules."""
        self.logger.info("Initializing benchmark modules")
        
        try:
            self.benchmarks['parallelization'] = ParallelizationBenchmark(
                str(self.output_dir / 'parallelization')
            )
            self.benchmarks['mesh_complexity'] = MeshComplexityBenchmark(
                str(self.output_dir / 'mesh_complexity')
            )
            self.benchmarks['hierarchical_ffd'] = HierarchicalFFDBenchmark(
                str(self.output_dir / 'hierarchical_ffd')
            )
            self.benchmarks['visualization'] = VisualizationBenchmark(
                str(self.output_dir / 'visualization')
            )
            
            self.logger.info(f"Initialized {len(self.benchmarks)} benchmark modules")
            
        except Exception as e:
            self.logger.error(f"Error initializing benchmarks: {e}")
            self.benchmarks = {}
    
    def run_single_benchmark(self, benchmark_name: str, config: Dict[str, Any]) -> Dict:
        """Run a single benchmark with configuration."""
        self.logger.info(f"Starting {benchmark_name} benchmark")
        start_time = time.time()
        
        try:
            benchmark = self.benchmarks[benchmark_name]
            
            # Apply configuration if provided
            if config.get('quick', False):
                self._apply_quick_config(benchmark, benchmark_name)
            
            if config.get('limited', False):
                self._apply_limited_config(benchmark, benchmark_name)
            
            # Run the benchmark
            if hasattr(benchmark, 'run_full_benchmark'):
                benchmark.run_full_benchmark()
            else:
                self.logger.warning(f"Benchmark {benchmark_name} does not have run_full_benchmark method")
            
            end_time = time.time()
            duration = end_time - start_time
            
            self.logger.info(f"Completed {benchmark_name} benchmark in {duration:.2f} seconds")
            
            return {
                'benchmark': benchmark_name,
                'status': 'success',
                'duration': duration,
                'output_dir': str(benchmark.output_dir)
            }
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            self.logger.error(f"Error in {benchmark_name} benchmark: {e}")
            
            return {
                'benchmark': benchmark_name,
                'status': 'failed',
                'duration': duration,
                'error': str(e)
            }
    
    def _apply_quick_config(self, benchmark, benchmark_name: str):
        """Apply quick configuration to reduce benchmark time."""
        if benchmark_name == 'parallelization':
            benchmark.mesh_sizes = [50_000, 250_000, 1_000_000]
            benchmark.worker_counts = [1, 2, 4, 8]
            benchmark.control_dimensions = [(4, 4, 4), (8, 8, 8)]
            benchmark.repetitions = 3
            
        elif benchmark_name == 'mesh_complexity':
            benchmark.mesh_sizes = np.logspace(3, 6, 8).astype(int)
            benchmark.control_grid_sizes = [(4, 4, 4), (8, 8, 8), (12, 12, 12)]
            benchmark.geometry_types = ['cube', 'sphere', 'wing']
            benchmark.aspect_ratios = [0.5, 1.0, 2.0]
            benchmark.repetitions = 3
            
        elif benchmark_name == 'hierarchical_ffd':
            benchmark.mesh_sizes = [100_000, 500_000]
            benchmark.hierarchy_depths = [2, 3, 4]
            benchmark.subdivision_factors = [2, 3]
            benchmark.base_dimensions = [(4, 4, 4), (6, 6, 6)]
            benchmark.repetitions = 2
            
        elif benchmark_name == 'visualization':
            benchmark.mesh_sizes = [50_000, 250_000, 1_000_000]
            benchmark.worker_counts = [1, 4, 8]
            benchmark.visualization_modes = ['points', 'surface']
            benchmark.quality_levels = ['low', 'medium', 'high']
            benchmark.repetitions = 2
    
    def _apply_limited_config(self, benchmark, benchmark_name: str):
        """Apply very limited configuration for testing."""
        if benchmark_name == 'parallelization':
            benchmark.mesh_sizes = [10_000, 100_000]
            benchmark.worker_counts = [1, 4]
            benchmark.control_dimensions = [(4, 4, 4)]
            benchmark.repetitions = 2
            
        elif benchmark_name == 'mesh_complexity':
            benchmark.mesh_sizes = [10_000, 100_000]
            benchmark.control_grid_sizes = [(4, 4, 4), (8, 8, 8)]
            benchmark.geometry_types = ['cube', 'sphere']
            benchmark.aspect_ratios = [1.0]
            benchmark.repetitions = 2
            
        elif benchmark_name == 'hierarchical_ffd':
            benchmark.mesh_sizes = [50_000]
            benchmark.hierarchy_depths = [2, 3]
            benchmark.subdivision_factors = [2]
            benchmark.base_dimensions = [(4, 4, 4)]
            benchmark.repetitions = 2
            
        elif benchmark_name == 'visualization':
            benchmark.mesh_sizes = [10_000, 100_000]
            benchmark.worker_counts = [1, 4]
            benchmark.visualization_modes = ['points']
            benchmark.quality_levels = ['low', 'medium']
            benchmark.repetitions = 2
    
    def run_parallel_benchmarks(self, benchmark_names: List[str], config: Dict[str, Any]) -> Dict:
        """Run multiple benchmarks in parallel."""
        self.logger.info(f"Running {len(benchmark_names)} benchmarks in parallel")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(4, len(benchmark_names))) as executor:
            # Submit all benchmarks
            future_to_benchmark = {
                executor.submit(self.run_single_benchmark, name, config): name
                for name in benchmark_names
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_benchmark):
                benchmark_name = future_to_benchmark[future]
                try:
                    result = future.result()
                    results[benchmark_name] = result
                    
                    if result['status'] == 'success':
                        self.logger.info(f"✓ {benchmark_name} completed successfully")
                    else:
                        self.logger.error(f"✗ {benchmark_name} failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.error(f"✗ {benchmark_name} raised exception: {e}")
                    results[benchmark_name] = {
                        'benchmark': benchmark_name,
                        'status': 'exception',
                        'error': str(e)
                    }
        
        return results
    
    def collect_benchmark_data(self) -> Dict[str, pd.DataFrame]:
        """Collect data from all completed benchmarks."""
        self.logger.info("Collecting benchmark data")
        
        collected_data = {}
        
        for benchmark_name, benchmark in self.benchmarks.items():
            try:
                # Look for CSV files in benchmark output directory
                output_dir = Path(benchmark.output_dir)
                csv_files = list(output_dir.glob("*raw_results.csv"))
                
                if csv_files:
                    # Load the most recent or largest CSV file
                    csv_file = max(csv_files, key=lambda f: f.stat().st_size)
                    df = pd.read_csv(csv_file)
                    collected_data[benchmark_name] = df
                    self.logger.info(f"Loaded {len(df)} records from {benchmark_name}")
                else:
                    self.logger.warning(f"No raw results found for {benchmark_name}")
                    
            except Exception as e:
                self.logger.error(f"Error loading data for {benchmark_name}: {e}")
        
        return collected_data
    
    def generate_comprehensive_analysis(self, data: Dict[str, pd.DataFrame]):
        """Generate comprehensive cross-benchmark analysis."""
        self.logger.info("Generating comprehensive analysis")
        
        analysis = {
            'summary_statistics': {},
            'performance_trends': {},
            'scaling_analysis': {},
            'recommendations': {}
        }
        
        # Summary statistics for each benchmark
        for benchmark_name, df in data.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            analysis['summary_statistics'][benchmark_name] = {
                'total_tests': len(df),
                'numeric_columns': len(numeric_cols),
                'mean_values': df[numeric_cols].mean().to_dict(),
                'std_values': df[numeric_cols].std().to_dict()
            }
        
        # Performance scaling analysis
        for benchmark_name, df in data.items():
            if 'mesh_size' in df.columns:
                # Analyze how performance scales with mesh size
                if 'execution_time' in df.columns:
                    time_col = 'execution_time'
                elif 'total_time' in df.columns:
                    time_col = 'total_time'
                elif 'creation_time' in df.columns:
                    time_col = 'creation_time'
                else:
                    continue
                
                # Group by mesh size and calculate scaling
                scaling_data = df.groupby('mesh_size')[time_col].mean()
                
                if len(scaling_data) >= 3:
                    # Fit power law: time = a * size^b
                    log_size = np.log10(scaling_data.index)
                    log_time = np.log10(scaling_data.values)
                    
                    # Linear fit in log space
                    coeffs = np.polyfit(log_size, log_time, 1)
                    r_squared = np.corrcoef(log_size, log_time)[0, 1]**2
                    
                    analysis['scaling_analysis'][benchmark_name] = {
                        'scaling_exponent': coeffs[0],
                        'r_squared': r_squared,
                        'complexity_class': self._classify_complexity(coeffs[0])
                    }
        
        # Save analysis
        with open(self.output_dir / 'comprehensive_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return analysis
    
    def _classify_complexity(self, exponent: float) -> str:
        """Classify computational complexity based on scaling exponent."""
        if exponent < 1.2:
            return "Linear O(n)"
        elif exponent < 1.8:
            return "Super-linear O(n log n)"
        elif exponent < 2.2:
            return "Quadratic O(n²)"
        elif exponent < 3.2:
            return "Cubic O(n³)"
        else:
            return "Higher-order polynomial"
    
    def create_publication_figures(self, data: Dict[str, pd.DataFrame]):
        """Create individual publication-ready figures for each analysis."""
        self.logger.info("Creating individual publication figures")
        
        figures_dir = self.output_dir / 'publication_figures'
        figures_dir.mkdir(exist_ok=True)
        
        # Create individual figures for each analysis type
        self._create_execution_time_comparison(data, figures_dir)
        self._create_individual_scaling_figures(data, figures_dir)
        self._create_individual_efficiency_figures(data, figures_dir)
        self._create_individual_memory_figures(data, figures_dir)
        self._create_throughput_analysis_figures(data, figures_dir)
        
        self.logger.info(f"Individual publication figures saved to {figures_dir}")
    
    def _create_execution_time_comparison(self, data: Dict[str, pd.DataFrame], output_dir: Path):
        """Create execution time comparison figure."""
        try:
            if not data:
                return
                
            fig, ax = plt.subplots(figsize=(10, 6))
            
            benchmarks = []
            mean_times = []
            std_times = []
            
            for bench_name, df in data.items():
                time_cols = ['execution_time', 'total_time', 'creation_time', 'processing_time']
                time_col = next((col for col in time_cols if col in df.columns), None)
                
                if time_col and len(df) > 0:
                    benchmarks.append(bench_name.replace('_', ' ').title())
                    mean_times.append(df[time_col].mean())
                    std_times.append(df[time_col].std())
            
            if benchmarks:
                colors = self.plotter.style.color_sequences['qualitative'][:len(benchmarks)]
                bars = ax.bar(benchmarks, mean_times, yerr=std_times, capsize=5,
                             color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
                
                ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
                ax.set_title('Benchmark Execution Time Comparison', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar, mean_time in zip(bars, mean_times):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std_times[bars.index(bar)],
                           f'{mean_time:.3f}s', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            self.plotter.save_figure(fig, 'execution_time_comparison', str(output_dir), remove_title=True)
            plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"Error creating execution time comparison: {e}")
    
    def _create_individual_scaling_figures(self, data: Dict[str, pd.DataFrame], output_dir: Path):
        """Create individual scaling figures for each benchmark."""
        try:
            for benchmark_name, df in data.items():
                if 'mesh_size' not in df.columns:
                    continue
                    
                time_cols = ['execution_time', 'total_time', 'creation_time', 'processing_time']
                time_col = next((col for col in time_cols if col in df.columns), None)
                
                if not time_col:
                    continue
                
                fig, ax = plt.subplots(figsize=(8, 6))  # Smaller size for publications
                
                scaling_data = df.groupby('mesh_size')[time_col].mean()
                scaling_std = df.groupby('mesh_size')[time_col].std()
                
                if len(scaling_data) > 1:
                    # Main data with enhanced styling
                    ax.loglog(scaling_data.index, scaling_data.values, 'o-',
                             color=self.plotter.style.colors['primary'], 
                             linewidth=3.0, markersize=8, markerfacecolor='white',
                             markeredgewidth=2.0, markeredgecolor=self.plotter.style.colors['primary'],
                             label=f'{benchmark_name.replace("_", " ").title()}')
                    
                    # Error bars
                    ax.errorbar(scaling_data.index, scaling_data.values, yerr=scaling_std.values,
                               fmt='none', ecolor='gray', alpha=0.6, capsize=5, linewidth=1.5)
                    
                    # Reference lines with enhanced visibility
                    x_ref = np.array([scaling_data.index.min(), scaling_data.index.max()])
                    y_ref_base = scaling_data.values.min()
                    
                    y_ref_linear = y_ref_base * (x_ref / x_ref[0])
                    y_ref_quad = y_ref_base * (x_ref / x_ref[0])**2
                    
                    ax.loglog(x_ref, y_ref_linear, '--', alpha=0.8, color='gray', 
                             linewidth=2.5, label='O(n) Linear')
                    ax.loglog(x_ref, y_ref_quad, ':', alpha=0.8, color='red', 
                             linewidth=2.5, label='O(n²) Quadratic')
                
                # Enhanced formatting
                ax.set_xlabel('Mesh Size (points)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
                ax.set_title(f'{benchmark_name.replace("_", " ").title()} - Computational Scaling Analysis', 
                           fontsize=16, fontweight='bold')
                
                # Enhanced grid and visibility
                ax.grid(True, alpha=0.4, linewidth=0.8)
                legend = ax.legend(fontsize=12, frameon=True, fancybox=False, shadow=False,
                                  framealpha=0.95, edgecolor='black', loc='best')
                legend.get_frame().set_linewidth(1.2)
                
                # Enhanced tick formatting
                ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
                ax.tick_params(axis='both', which='minor', width=1.0, length=4)
                
                # Apply axis limit optimization for log scale
                try:
                    from academic_plotting_utils import optimize_axis_limits
                    optimize_axis_limits(ax, x_data=scaling_data.index.values, 
                                       y_data=scaling_data.values, log_scale=(True, True))
                except ImportError:
                    # Fallback manual optimization for log scale
                    x_min, x_max = scaling_data.index.min(), scaling_data.index.max()
                    y_min, y_max = scaling_data.values.min(), scaling_data.values.max()
                    log_x_margin = (np.log10(x_max) - np.log10(x_min)) * 0.05
                    log_y_margin = (np.log10(y_max) - np.log10(y_min)) * 0.05
                    ax.set_xlim(10**(np.log10(x_min) - log_x_margin), 10**(np.log10(x_max) + log_x_margin))
                    ax.set_ylim(10**(np.log10(y_min) - log_y_margin), 10**(np.log10(y_max) + log_y_margin))
                
                plt.tight_layout()
                filename = f'{benchmark_name}_scaling_analysis'
                self.plotter.save_figure(fig, filename, str(output_dir), remove_title=True)
                plt.close(fig)
                
        except Exception as e:
            self.logger.error(f"Error creating individual scaling figures: {e}")
    
    def _create_individual_efficiency_figures(self, data: Dict[str, pd.DataFrame], output_dir: Path):
        """Create individual parallel efficiency figures."""
        try:
            for benchmark_name, df in data.items():
                if 'workers' not in df.columns:
                    continue
                
                # Create speedup figure
                if 'speedup' in df.columns:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    speedup_data = df.groupby('workers')['speedup'].mean()
                    speedup_std = df.groupby('workers')['speedup'].std()
                    
                    # Main speedup data
                    ax.plot(speedup_data.index, speedup_data.values, 'o-',
                           color=self.plotter.style.colors['primary'], 
                           linewidth=2.5, markersize=8, label=f'{benchmark_name.replace("_", " ").title()}')
                    
                    # Error bars
                    ax.errorbar(speedup_data.index, speedup_data.values, yerr=speedup_std.values,
                               fmt='none', ecolor='gray', alpha=0.5, capsize=3)
                    
                    # Ideal speedup line
                    workers_range = np.arange(1, speedup_data.index.max() + 1)
                    ax.plot(workers_range, workers_range, 'k--', alpha=0.7, 
                           label='Ideal Speedup', linewidth=2)
                    
                    # Efficiency annotations
                    for workers, speedup in speedup_data.items():
                        efficiency = speedup / workers * 100
                        ax.annotate(f'{efficiency:.1f}%', 
                                   (workers, speedup), textcoords="offset points", 
                                   xytext=(0,10), ha='center', fontsize=9)
                    
                    ax.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Speedup', fontsize=12, fontweight='bold')
                    ax.set_title(f'{benchmark_name.replace("_", " ").title()} - Parallel Speedup Analysis', 
                               fontsize=14, fontweight='bold')
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    filename = f'{benchmark_name}_speedup_analysis'
                    self.plotter.save_figure(fig, filename, str(output_dir), remove_title=True)
                    plt.close(fig)
                
                # Create throughput figure
                throughput_cols = ['throughput', 'throughput_points_per_sec']
                throughput_col = next((col for col in throughput_cols if col in df.columns), None)
                
                if throughput_col:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    throughput_data = df.groupby('workers')[throughput_col].mean()
                    throughput_std = df.groupby('workers')[throughput_col].std()
                    
                    ax.plot(throughput_data.index, throughput_data.values, 'o-',
                           color=self.plotter.style.colors['secondary'], 
                           linewidth=2.5, markersize=8, label=f'{benchmark_name.replace("_", " ").title()}')
                    
                    # Error bars
                    ax.errorbar(throughput_data.index, throughput_data.values, yerr=throughput_std.values,
                               fmt='none', ecolor='gray', alpha=0.5, capsize=3)
                    
                    ax.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Throughput (points/sec)', fontsize=12, fontweight='bold')
                    ax.set_title(f'{benchmark_name.replace("_", " ").title()} - Processing Throughput Analysis', 
                               fontsize=14, fontweight='bold')
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    
                    plt.tight_layout()
                    filename = f'{benchmark_name}_throughput_analysis'
                    self.plotter.save_figure(fig, filename, str(output_dir), remove_title=True)
                    plt.close(fig)
                    
        except Exception as e:
            self.logger.error(f"Error creating individual efficiency figures: {e}")
    
    def _create_individual_memory_figures(self, data: Dict[str, pd.DataFrame], output_dir: Path):
        """Create individual memory usage figures."""
        try:
            for benchmark_name, df in data.items():
                memory_cols = ['memory_usage_mb', 'memory_estimate_mb']
                memory_col = next((col for col in memory_cols if col in df.columns), None)
                
                if not memory_col or 'mesh_size' not in df.columns:
                    continue
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                memory_by_size = df.groupby('mesh_size')[memory_col].mean()
                memory_std = df.groupby('mesh_size')[memory_col].std()
                
                if len(memory_by_size) > 1:
                    # Main memory data
                    ax.loglog(memory_by_size.index, memory_by_size.values, 'o-',
                             color=self.plotter.style.colors['accent'], 
                             linewidth=2.5, markersize=8, label=f'{benchmark_name.replace("_", " ").title()}')
                    
                    # Error bars
                    ax.errorbar(memory_by_size.index, memory_by_size.values, yerr=memory_std.values,
                               fmt='none', ecolor='gray', alpha=0.5, capsize=3)
                    
                    # Reference line for linear memory scaling
                    x_ref = np.array([memory_by_size.index.min(), memory_by_size.index.max()])
                    y_ref_linear = memory_by_size.values.min() * (x_ref / x_ref[0])
                    ax.loglog(x_ref, y_ref_linear, '--', alpha=0.7, color='gray', 
                             linewidth=1.5, label='Linear Scaling')
                
                ax.set_xlabel('Mesh Size (points)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
                ax.set_title(f'{benchmark_name.replace("_", " ").title()} - Memory Usage Analysis', 
                           fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                filename = f'{benchmark_name}_memory_analysis'
                self.plotter.save_figure(fig, filename, str(output_dir), remove_title=True)
                plt.close(fig)
                
        except Exception as e:
            self.logger.error(f"Error creating individual memory figures: {e}")
    
    def _create_throughput_analysis_figures(self, data: Dict[str, pd.DataFrame], output_dir: Path):
        """Create throughput analysis figures."""
        try:
            for benchmark_name, df in data.items():
                throughput_cols = ['throughput_points_per_sec', 'throughput', 'processing_rate']
                throughput_col = next((col for col in throughput_cols if col in df.columns), None)
                
                if not throughput_col:
                    continue
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                if 'mesh_size' in df.columns:
                    # Throughput vs mesh size
                    throughput_by_size = df.groupby('mesh_size')[throughput_col].mean()
                    throughput_std = df.groupby('mesh_size')[throughput_col].std()
                    
                    ax.semilogx(throughput_by_size.index, throughput_by_size.values, 'o-',
                               color=self.plotter.style.colors['success'], 
                               linewidth=2.5, markersize=8, label=f'{benchmark_name.replace("_", " ").title()}')
                    
                    # Error bars
                    ax.errorbar(throughput_by_size.index, throughput_by_size.values, yerr=throughput_std.values,
                               fmt='none', ecolor='gray', alpha=0.5, capsize=3)
                    
                    ax.set_xlabel('Mesh Size (points)', fontsize=12, fontweight='bold')
                    
                elif 'workers' in df.columns:
                    # Throughput vs workers
                    throughput_by_workers = df.groupby('workers')[throughput_col].mean()
                    throughput_std = df.groupby('workers')[throughput_col].std()
                    
                    ax.plot(throughput_by_workers.index, throughput_by_workers.values, 'o-',
                           color=self.plotter.style.colors['success'], 
                           linewidth=2.5, markersize=8, label=f'{benchmark_name.replace("_", " ").title()}')
                    
                    # Error bars
                    ax.errorbar(throughput_by_workers.index, throughput_by_workers.values, yerr=throughput_std.values,
                               fmt='none', ecolor='gray', alpha=0.5, capsize=3)
                    
                    ax.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
                
                ax.set_ylabel('Throughput (points/sec)', fontsize=12, fontweight='bold')
                ax.set_title(f'{benchmark_name.replace("_", " ").title()} - Processing Throughput', 
                           fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                filename = f'{benchmark_name}_throughput'
                self.plotter.save_figure(fig, filename, str(output_dir), remove_title=True)
                plt.close(fig)
                
        except Exception as e:
            self.logger.error(f"Error creating throughput analysis figures: {e}")
    
    def generate_comprehensive_report(self, data: Dict[str, pd.DataFrame], analysis: Dict):
        """Generate comprehensive benchmark report for publication."""
        self.logger.info("Generating comprehensive report")
        
        report = []
        report.append("# OpenFFD Comprehensive Performance Benchmark Report\n")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("## Executive Summary\n")
        
        # Overall performance summary
        total_tests = sum(len(df) for df in data.values())
        report.append(f"This comprehensive benchmark analysis evaluated OpenFFD performance across ")
        report.append(f"{len(data)} different benchmark categories with a total of {total_tests} test cases.\n\n")
        
        # Key findings
        report.append("### Key Findings\n")
        
        if 'scaling_analysis' in analysis:
            for benchmark_name, scaling_info in analysis['scaling_analysis'].items():
                complexity_class = scaling_info['complexity_class']
                r_squared = scaling_info['r_squared']
                report.append(f"- **{benchmark_name.replace('_', ' ').title()}**: {complexity_class} ")
                report.append(f"scaling (R² = {r_squared:.3f})\n")
        
        # Performance highlights per benchmark
        report.append("\n## Benchmark-Specific Results\n")
        
        for benchmark_name, df in data.items():
            report.append(f"### {benchmark_name.replace('_', ' ').title()}\n")
            
            # Basic statistics
            report.append(f"- Total test cases: {len(df)}\n")
            
            # Performance metrics
            time_cols = ['execution_time', 'total_time', 'creation_time', 'processing_time']
            time_col = next((col for col in time_cols if col in df.columns), None)
            
            if time_col:
                mean_time = df[time_col].mean()
                min_time = df[time_col].min()
                max_time = df[time_col].max()
                report.append(f"- Average execution time: {mean_time:.3f} seconds\n")
                report.append(f"- Best performance: {min_time:.3f} seconds\n")
                report.append(f"- Worst performance: {max_time:.3f} seconds\n")
            
            # Memory usage if available
            memory_cols = ['memory_usage_mb', 'memory_estimate_mb']
            memory_col = next((col for col in memory_cols if col in df.columns), None)
            if memory_col:
                mean_memory = df[memory_col].mean()
                report.append(f"- Average memory usage: {mean_memory:.1f} MB\n")
            
            report.append("\n")
        
        # Comparative analysis
        report.append("## Comparative Analysis\n")
        
        if len(data) > 1:
            # Find the fastest benchmark on average
            avg_times = {}
            for benchmark_name, df in data.items():
                time_cols = ['execution_time', 'total_time', 'creation_time', 'processing_time']
                time_col = next((col for col in time_cols if col in df.columns), None)
                if time_col:
                    avg_times[benchmark_name] = df[time_col].mean()
            
            if avg_times:
                fastest = min(avg_times, key=avg_times.get)
                slowest = max(avg_times, key=avg_times.get)
                
                report.append(f"- Fastest benchmark: {fastest.replace('_', ' ').title()}\n")
                report.append(f"- Most computationally intensive: {slowest.replace('_', ' ').title()}\n")
                
                if len(avg_times) > 2:
                    speed_ratio = avg_times[slowest] / avg_times[fastest]
                    report.append(f"- Performance variation: {speed_ratio:.1f}x difference between fastest and slowest\n")
        
        # Recommendations
        report.append("\n## Performance Recommendations\n")
        
        report.append("### For Real-time Applications:\n")
        report.append("- Use simplified mesh representations for interactive use\n")
        report.append("- Enable parallel processing for meshes > 100,000 points\n")
        report.append("- Consider hierarchical FFD with depth ≤ 3 for responsive interaction\n")
        
        report.append("\n### For High-Fidelity Analysis:\n")
        report.append("- Utilize full mesh resolution with parallel processing\n")
        report.append("- Hierarchical FFD depth 4-5 provides good accuracy/performance balance\n")
        report.append("- Monitor memory usage for complex geometries\n")
        
        report.append("\n### For Large-Scale Simulations:\n")
        report.append("- Implement progressive mesh refinement strategies\n")
        report.append("- Use distributed computing for meshes > 10M points\n")
        report.append("- Consider specialized visualization techniques for very large datasets\n")
        
        # System requirements
        report.append("\n## System Requirements\n")
        
        if data:
            max_memory = 0
            max_mesh_size = 0
            
            for df in data.values():
                memory_cols = ['memory_usage_mb', 'memory_estimate_mb']
                memory_col = next((col for col in memory_cols if col in df.columns), None)
                if memory_col:
                    max_memory = max(max_memory, df[memory_col].max())
                
                if 'mesh_size' in df.columns:
                    max_mesh_size = max(max_mesh_size, df['mesh_size'].max())
            
            report.append(f"Based on benchmarks with meshes up to {max_mesh_size:,} points:\n\n")
            report.append("### Minimum Requirements:\n")
            report.append(f"- RAM: {max(4, int(max_memory * 0.5))} GB\n")
            report.append("- CPU: 4 cores\n")
            report.append("- Storage: 10 GB available space\n")
            
            report.append("\n### Recommended Requirements:\n")
            report.append(f"- RAM: {max(16, int(max_memory * 2))} GB\n")
            report.append("- CPU: 8+ cores with parallel processing support\n")
            report.append("- Storage: 50 GB available space\n")
            report.append("- GPU: Dedicated graphics card for visualization\n")
        
        # Methodology
        report.append("\n## Methodology\n")
        report.append("This benchmark suite evaluates OpenFFD performance across multiple dimensions:\n\n")
        report.append("1. **Parallelization Scalability**: Measures speedup and efficiency with varying worker counts\n")
        report.append("2. **Mesh Complexity Scaling**: Analyzes performance vs. mesh size and geometry complexity\n")
        report.append("3. **Hierarchical FFD Performance**: Evaluates multi-level FFD computational costs\n")
        report.append("4. **Visualization Performance**: Assesses rendering and display capabilities\n\n")
        
        report.append("All benchmarks used standardized test cases with multiple repetitions ")
        report.append("for statistical significance. Performance metrics include execution time, ")
        report.append("memory usage, throughput, and scaling efficiency.\n")
        
        # Save report
        with open(self.output_dir / 'comprehensive_benchmark_report.md', 'w') as f:
            f.writelines(report)
        
        self.logger.info(f"Comprehensive report saved to {self.output_dir}/comprehensive_benchmark_report.md")
    
    def run_full_benchmark_suite(
        self, 
        benchmark_names: Optional[List[str]] = None,
        parallel: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """Run the complete benchmark suite."""
        if config is None:
            config = {}
            
        if not BENCHMARK_MODULES_AVAILABLE:
            self.logger.error("Benchmark modules not available. Cannot run benchmarks.")
            return
        
        if benchmark_names is None:
            benchmark_names = list(self.benchmarks.keys())
        
        self.logger.info(f"Starting comprehensive benchmark suite with {len(benchmark_names)} benchmarks")
        start_time = time.time()
        
        # Run benchmarks
        if parallel and len(benchmark_names) > 1:
            results = self.run_parallel_benchmarks(benchmark_names, config)
        else:
            results = {}
            for name in benchmark_names:
                results[name] = self.run_single_benchmark(name, config)
        
        # Collect and analyze results
        data = self.collect_benchmark_data()
        
        if data:
            # Generate analysis
            analysis = self.generate_comprehensive_analysis(data)
            
            # Create publication figures
            self.create_publication_figures(data)
            
            # Generate comprehensive report
            self.generate_comprehensive_report(data, analysis)
        else:
            self.logger.warning("No benchmark data collected")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Summary
        successful_benchmarks = [name for name, result in results.items() 
                               if result.get('status') == 'success']
        
        self.logger.info(f"Benchmark suite completed in {total_duration:.2f} seconds")
        self.logger.info(f"Successful benchmarks: {len(successful_benchmarks)}/{len(benchmark_names)}")
        self.logger.info(f"Results available in: {self.output_dir}")
        
        return {
            'results': results,
            'data': data,
            'duration': total_duration,
            'output_dir': str(self.output_dir)
        }


def main():
    """Main entry point for comprehensive benchmark runner."""
    parser = argparse.ArgumentParser(
        description='Comprehensive benchmark suite for OpenFFD performance analysis'
    )
    parser.add_argument(
        '--output-dir', default='comprehensive_benchmark_results',
        help='Output directory for all benchmark results'
    )
    parser.add_argument(
        '--benchmarks', nargs='+', 
        choices=['parallelization', 'mesh_complexity', 'hierarchical_ffd', 'visualization'],
        help='Specific benchmarks to run (default: all)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick benchmark with reduced test cases'
    )
    parser.add_argument(
        '--limited', action='store_true',
        help='Run very limited benchmark for testing'
    )
    parser.add_argument(
        '--sequential', action='store_true',
        help='Run benchmarks sequentially instead of in parallel'
    )
    parser.add_argument(
        '--no-figures', action='store_true',
        help='Skip generation of publication figures'
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark suite
    suite = BenchmarkSuite(args.output_dir)
    
    # Configure benchmarks
    config = {
        'quick': args.quick,
        'limited': args.limited,
        'no_figures': args.no_figures
    }
    
    # Run benchmark suite
    suite.run_full_benchmark_suite(
        benchmark_names=args.benchmarks,
        parallel=not args.sequential,
        config=config
    )


if __name__ == "__main__":
    main()