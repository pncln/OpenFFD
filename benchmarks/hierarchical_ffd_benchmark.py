#!/usr/bin/env python3
"""
Hierarchical FFD Performance Benchmark for OpenFFD

This benchmark evaluates the performance characteristics of OpenFFD's
hierarchical FFD capabilities across different hierarchy depths, subdivision
factors, and control grid configurations. Designed for academic analysis.

Author: OpenFFD Development Team
Purpose: AIAA Journal Performance Analysis
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogFormatter, LogLocator
from scipy import stats

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openffd.core.hierarchical import create_hierarchical_ffd, HierarchicalFFD
from openffd.utils.parallel import ParallelConfig
from openffd.utils.benchmark import generate_random_mesh


class HierarchicalFFDBenchmark:
    """Comprehensive hierarchical FFD performance benchmark suite."""
    
    def __init__(self, output_dir: str = "hierarchical_ffd_results"):
        """Initialize benchmark with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure academic-quality plotting
        self._setup_plotting()
        
        # Initialize logging
        self._setup_logging()
        
        # Benchmark parameters
        self.mesh_sizes = [50_000, 100_000, 250_000, 500_000, 1_000_000]
        self.hierarchy_depths = [1, 2, 3, 4, 5, 6]
        self.subdivision_factors = [2, 3, 4]
        self.base_dimensions = [(3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6)]
        self.repetitions = 3
        
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
                logging.FileHandler(self.output_dir / 'hierarchical_benchmark.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def run_single_hierarchical_benchmark(
        self, 
        mesh_size: int,
        base_dims: Tuple[int, int, int],
        max_depth: int,
        subdivision_factor: int
    ) -> Dict:
        """Run a single hierarchical FFD benchmark."""
        self.logger.info(f"Benchmarking H-FFD: {mesh_size} points, {base_dims} base, "
                        f"depth={max_depth}, subdivision={subdivision_factor}")
        
        results = []
        
        for rep in range(self.repetitions):
            try:
                # Generate mesh
                mesh_points = generate_random_mesh(mesh_size, seed=42+rep)
                
                # Configure parallel processing
                parallel_config = ParallelConfig(
                    enabled=mesh_size > 100_000,
                    method='process',
                    max_workers=12,
                    threshold=50_000
                )
                
                # Benchmark hierarchical FFD creation
                start_time = time.perf_counter()
                
                h_ffd = create_hierarchical_ffd(
                    mesh_points=mesh_points,
                    base_dims=base_dims,
                    max_depth=max_depth,
                    subdivision_factor=subdivision_factor,
                    parallel_config=parallel_config,
                    margin=0.05
                )
                
                creation_time = time.perf_counter() - start_time
                
                # Analyze the created hierarchy
                level_info = h_ffd.get_level_info()
                total_control_points = sum(info['num_control_points'] for info in level_info)
                total_levels = len(level_info)
                
                # Calculate theoretical operations
                theoretical_ops = mesh_size * total_control_points
                
                # Benchmark deformation (if not too expensive)
                deformation_time = 0
                if total_control_points < 10_000:  # Limit for feasible deformation testing
                    try:
                        # Create simple deformation (move control points slightly)
                        deformed_control_points = {}
                        for level_id, level in h_ffd.levels.items():
                            deformed_cp = level.control_points.copy()
                            deformed_cp[:, 2] += 0.01  # Small z-displacement
                            deformed_control_points[level_id] = deformed_cp
                        
                        # Time the deformation
                        start_def = time.perf_counter()
                        deformed_mesh = h_ffd.deform_mesh(deformed_control_points)
                        deformation_time = time.perf_counter() - start_def
                        
                    except Exception as e:
                        self.logger.warning(f"Deformation benchmark failed: {e}")
                        deformation_time = float('inf')
                
                # Memory usage estimation
                memory_estimate = 0
                for info in level_info:
                    # Control points + influence calculations
                    memory_estimate += info['num_control_points'] * 3 * 8  # 8 bytes per float
                    memory_estimate += mesh_size * 8  # Influence storage
                
                memory_estimate_mb = memory_estimate / (1024 * 1024)
                
                results.append({
                    'mesh_size': mesh_size,
                    'base_dims': base_dims,
                    'max_depth': max_depth,
                    'subdivision_factor': subdivision_factor,
                    'repetition': rep,
                    'creation_time': creation_time,
                    'deformation_time': deformation_time,
                    'total_time': creation_time + deformation_time,
                    'total_levels': total_levels,
                    'total_control_points': total_control_points,
                    'theoretical_operations': theoretical_ops,
                    'memory_estimate_mb': memory_estimate_mb,
                    'level_details': level_info,
                    'throughput': mesh_size / creation_time if creation_time > 0 else 0,
                    'control_efficiency': total_control_points / creation_time if creation_time > 0 else 0
                })
                
            except Exception as e:
                self.logger.error(f"Error in hierarchical benchmark: {e}")
                results.append({
                    'mesh_size': mesh_size,
                    'base_dims': base_dims,
                    'max_depth': max_depth,
                    'subdivision_factor': subdivision_factor,
                    'repetition': rep,
                    'creation_time': float('inf'),
                    'deformation_time': float('inf'),
                    'total_time': float('inf'),
                    'total_levels': 0,
                    'total_control_points': 0,
                    'theoretical_operations': 0,
                    'memory_estimate_mb': 0,
                    'level_details': [],
                    'throughput': 0,
                    'control_efficiency': 0
                })
        
        return results
    
    def run_hierarchy_depth_study(self) -> pd.DataFrame:
        """Run comprehensive hierarchy depth scaling study."""
        self.logger.info("Starting hierarchy depth scaling study")
        
        all_results = []
        
        # Primary study: depth scaling
        for mesh_size in self.mesh_sizes:
            for max_depth in self.hierarchy_depths:
                for base_dims in [(4, 4, 4), (6, 6, 6)]:  # Representative base dimensions
                    results = self.run_single_hierarchical_benchmark(
                        mesh_size, base_dims, max_depth, subdivision_factor=2
                    )
                    all_results.extend(results)
        
        # Secondary study: subdivision factor effects
        for subdivision_factor in self.subdivision_factors:
            for max_depth in [2, 3, 4]:  # Representative depths
                results = self.run_single_hierarchical_benchmark(
                    500_000, (4, 4, 4), max_depth, subdivision_factor
                )
                all_results.extend(results)
        
        # Tertiary study: base dimension effects
        for base_dims in self.base_dimensions:
            results = self.run_single_hierarchical_benchmark(
                500_000, base_dims, max_depth=3, subdivision_factor=2
            )
            all_results.extend(results)
        
        df = pd.DataFrame(all_results)
        
        # Save raw results
        df.to_csv(self.output_dir / 'hierarchical_ffd_raw_results.csv', index=False)
        
        # Calculate statistics
        df_stats = df.groupby(['mesh_size', 'base_dims', 'max_depth', 'subdivision_factor']).agg({
            'creation_time': ['mean', 'std', 'min', 'max'],
            'deformation_time': ['mean', 'std'],
            'total_control_points': ['mean'],
            'memory_estimate_mb': ['mean'],
            'throughput': ['mean', 'std']
        }).round(6)
        
        df_stats.to_csv(self.output_dir / 'hierarchical_ffd_statistics.csv')
        
        self.logger.info("Hierarchy depth study completed")
        return df
    
    def analyze_hierarchy_complexity(self, df: pd.DataFrame) -> Dict:
        """Analyze computational complexity of hierarchical FFD."""
        self.logger.info("Analyzing hierarchical FFD complexity")
        
        complexity_analysis = {}
        
        # Depth scaling analysis
        depth_data = df[
            (df['mesh_size'] == 500_000) &
            (df['base_dims'] == (4, 4, 4)) &
            (df['subdivision_factor'] == 2)
        ].groupby('max_depth')['creation_time'].mean().reset_index()
        
        if len(depth_data) > 3:
            # Exponential fit for depth scaling
            depths = depth_data['max_depth'].values
            times = depth_data['creation_time'].values
            
            # Try exponential fit: t = a * b^depth
            log_times = np.log(times)
            slope, intercept, r_value, p_value, std_err = stats.linregress(depths, log_times)
            
            complexity_analysis['depth_scaling'] = {
                'exponential_base': np.exp(slope),
                'r_squared': r_value**2,
                'p_value': p_value,
                'growth_type': 'exponential' if slope > 0.1 else 'sub-exponential'
            }
        
        # Control point scaling analysis
        control_scaling_data = df.groupby(['max_depth', 'subdivision_factor']).agg({
            'total_control_points': 'mean',
            'creation_time': 'mean'
        }).reset_index()
        
        if len(control_scaling_data) > 3:
            log_cp = np.log10(control_scaling_data['total_control_points'])
            log_time = np.log10(control_scaling_data['creation_time'])
            
            slope_cp, intercept_cp, r_value_cp, p_value_cp, std_err_cp = stats.linregress(
                log_cp, log_time
            )
            
            complexity_analysis['control_point_scaling'] = {
                'exponent': slope_cp,
                'r_squared': r_value_cp**2,
                'p_value': p_value_cp,
                'complexity_class': self._classify_complexity(slope_cp)
            }
        
        # Memory scaling analysis
        memory_scaling = df.groupby('max_depth').agg({
            'memory_estimate_mb': 'mean',
            'total_control_points': 'mean'
        }).reset_index()
        
        if len(memory_scaling) > 2:
            # Linear relationship between control points and memory
            slope_mem, intercept_mem, r_value_mem, _, _ = stats.linregress(
                memory_scaling['total_control_points'], memory_scaling['memory_estimate_mb']
            )
            
            complexity_analysis['memory_scaling'] = {
                'memory_per_control_point': slope_mem,
                'r_squared': r_value_mem**2,
                'base_memory_mb': intercept_mem
            }
        
        # Efficiency analysis
        efficiency_data = df.groupby(['max_depth', 'mesh_size']).agg({
            'throughput': 'mean',
            'control_efficiency': 'mean'
        }).reset_index()
        
        best_depth_efficiency = efficiency_data.groupby('max_depth')['throughput'].mean()
        optimal_depth = best_depth_efficiency.idxmax()
        
        complexity_analysis['efficiency'] = {
            'optimal_depth': int(optimal_depth),
            'best_throughput': float(best_depth_efficiency[optimal_depth]),
            'depth_efficiency_trend': best_depth_efficiency.to_dict()
        }
        
        # Save analysis
        with open(self.output_dir / 'hierarchical_complexity_analysis.json', 'w') as f:
            json.dump(complexity_analysis, f, indent=2, default=str)
        
        return complexity_analysis
    
    def _classify_complexity(self, exponent: float) -> str:
        """Classify computational complexity based on scaling exponent."""
        if exponent < 1.2:
            return "Linear"
        elif exponent < 1.8:
            return "Super-linear"
        elif exponent < 2.2:
            return "Quadratic"
        elif exponent < 3.2:
            return "Cubic"
        else:
            return "Higher-order"
    
    def plot_hierarchy_depth_analysis(self, df: pd.DataFrame):
        """Create hierarchy depth analysis plots."""
        self.logger.info("Creating hierarchy depth visualizations")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hierarchical FFD Depth Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Creation time vs hierarchy depth
        depth_data = df[
            (df['mesh_size'] == 500_000) &
            (df['base_dims'] == (4, 4, 4)) &
            (df['subdivision_factor'] == 2)
        ].groupby('max_depth').agg({
            'creation_time': ['mean', 'std'],
            'total_control_points': 'mean'
        }).round(4)
        
        depths = depth_data.index
        times_mean = depth_data['creation_time']['mean']
        times_std = depth_data['creation_time']['std']
        
        ax1.errorbar(depths, times_mean, yerr=times_std, marker='o', capsize=5,
                    linewidth=2, markersize=8, color='darkblue')
        ax1.set_xlabel('Hierarchy Depth')
        ax1.set_ylabel('Creation Time (seconds)')
        ax1.set_title('FFD Creation Time vs Hierarchy Depth')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Control points vs hierarchy depth
        control_points = depth_data['total_control_points']['mean']
        ax2.semilogy(depths, control_points, 'o-', linewidth=2, markersize=8, color='darkred')
        ax2.set_xlabel('Hierarchy Depth')
        ax2.set_ylabel('Total Control Points')
        ax2.set_title('Control Points vs Hierarchy Depth')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Subdivision factor effects
        subdivision_data = df[
            (df['mesh_size'] == 500_000) &
            (df['base_dims'] == (4, 4, 4)) &
            (df['max_depth'] == 3)
        ].groupby('subdivision_factor')['creation_time'].mean()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(subdivision_data)))
        bars = ax3.bar(subdivision_data.index, subdivision_data.values, color=colors, alpha=0.8)
        ax3.set_xlabel('Subdivision Factor')
        ax3.set_ylabel('Creation Time (seconds)')
        ax3.set_title('Impact of Subdivision Factor')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, subdivision_data.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 4: Memory usage vs depth
        memory_data = df[
            (df['mesh_size'] == 500_000) &
            (df['subdivision_factor'] == 2)
        ].groupby(['max_depth', 'base_dims'])['memory_estimate_mb'].mean().reset_index()
        
        for base_dims in [(4, 4, 4), (6, 6, 6)]:
            subset = memory_data[memory_data['base_dims'] == base_dims]
            if not subset.empty:
                ax4.plot(subset['max_depth'], subset['memory_estimate_mb'], 'o-',
                        label=f'{base_dims[0]}×{base_dims[1]}×{base_dims[2]}',
                        linewidth=2, markersize=6)
        
        ax4.set_xlabel('Hierarchy Depth')
        ax4.set_ylabel('Memory Estimate (MB)')
        ax4.set_title('Memory Usage vs Hierarchy Depth')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hierarchy_depth_analysis.png')
        plt.savefig(self.output_dir / 'hierarchy_depth_analysis.pdf')
        plt.close()
    
    def plot_hierarchy_structure_diagram(self, df: pd.DataFrame):
        """Create hierarchy structure visualization."""
        self.logger.info("Creating hierarchy structure diagram")
        
        # Get representative hierarchy information
        sample_result = df[
            (df['mesh_size'] == 500_000) &
            (df['base_dims'] == (4, 4, 4)) &
            (df['max_depth'] == 4) &
            (df['subdivision_factor'] == 2) &
            (df['repetition'] == 0)
        ]
        
        if sample_result.empty:
            self.logger.warning("No suitable data for hierarchy structure diagram")
            return
        
        level_details = sample_result.iloc[0]['level_details']
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create a tree-like visualization
        max_depth = max(info['depth'] for info in level_details)
        
        # Calculate positions for each level
        y_positions = {}
        x_positions = {}
        
        for level_info in level_details:
            depth = level_info['depth']
            level_id = level_info['level_id']
            
            # Y position based on depth (inverted so root is at top)
            y_positions[level_id] = max_depth - depth
            
            # X position: spread levels at same depth horizontally
            levels_at_depth = [info for info in level_details if info['depth'] == depth]
            if len(levels_at_depth) == 1:
                x_positions[level_id] = 0
            else:
                index_at_depth = levels_at_depth.index(level_info)
                x_positions[level_id] = index_at_depth - (len(levels_at_depth) - 1) / 2
        
        # Draw connections between parent and child levels
        for level_info in level_details:
            level_id = level_info['level_id']
            parent_id = level_info['parent_id']
            
            if parent_id is not None:
                ax.plot([x_positions[parent_id], x_positions[level_id]],
                       [y_positions[parent_id], y_positions[level_id]],
                       'k-', alpha=0.6, linewidth=1)
        
        # Draw level boxes
        for level_info in level_details:
            level_id = level_info['level_id']
            x = x_positions[level_id]
            y = y_positions[level_id]
            dims = level_info['dims']
            num_cp = level_info['num_control_points']
            weight = level_info['weight_factor']
            
            # Box size based on number of control points
            box_size = 0.3 + 0.2 * np.log10(num_cp) / 4
            
            # Color based on weight factor
            color = plt.cm.viridis(weight / max(info['weight_factor'] for info in level_details))
            
            # Draw rectangle
            rect = Rectangle((x - box_size/2, y - box_size/2), box_size, box_size,
                           facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            
            # Add text labels
            ax.text(x, y + 0.05, f'L{level_id}', ha='center', va='bottom', fontweight='bold')
            ax.text(x, y, f'{dims[0]}×{dims[1]}×{dims[2]}', ha='center', va='center', fontsize=10)
            ax.text(x, y - 0.05, f'{num_cp} CP', ha='center', va='top', fontsize=9, style='italic')
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.5, max_depth + 0.5)
        ax.set_aspect('equal')
        ax.set_title('Hierarchical FFD Structure\n(500K points, 4×4×4 base, depth=4, subdivision=2)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=plt.cm.viridis(1.0), label='Highest Weight'),
            Patch(facecolor=plt.cm.viridis(0.5), label='Medium Weight'),
            Patch(facecolor=plt.cm.viridis(0.0), label='Lowest Weight')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hierarchy_structure_diagram.png')
        plt.savefig(self.output_dir / 'hierarchy_structure_diagram.pdf')
        plt.close()
    
    def plot_performance_comparison(self, df: pd.DataFrame):
        """Create performance comparison between standard and hierarchical FFD."""
        self.logger.info("Creating performance comparison visualization")
        
        # Calculate equivalent standard FFD performance (estimated)
        # For comparison, use the finest level control points as equivalent standard FFD
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Throughput comparison
        hierarchical_perf = df[
            (df['base_dims'] == (4, 4, 4)) &
            (df['subdivision_factor'] == 2)
        ].groupby(['mesh_size', 'max_depth'])['throughput'].mean().reset_index()
        
        for depth in [2, 3, 4, 5]:
            subset = hierarchical_perf[hierarchical_perf['max_depth'] == depth]
            if not subset.empty:
                ax1.semilogx(subset['mesh_size'], subset['throughput'], 'o-',
                           label=f'H-FFD Depth {depth}', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Mesh Size (points)')
        ax1.set_ylabel('Throughput (points/second)')
        ax1.set_title('Throughput: Hierarchical FFD vs Mesh Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Control point efficiency
        control_efficiency = df[
            (df['mesh_size'] == 500_000) &
            (df['subdivision_factor'] == 2)
        ].groupby(['max_depth', 'base_dims']).agg({
            'control_efficiency': 'mean',
            'total_control_points': 'mean'
        }).reset_index()
        
        for base_dims in [(4, 4, 4), (6, 6, 6)]:
            subset = control_efficiency[control_efficiency['base_dims'] == base_dims]
            if not subset.empty:
                ax2.plot(subset['max_depth'], subset['control_efficiency'], 'o-',
                        label=f'{base_dims[0]}×{base_dims[1]}×{base_dims[2]} base',
                        linewidth=2, markersize=6)
        
        ax2.set_xlabel('Hierarchy Depth')
        ax2.set_ylabel('Control Efficiency (CP/second)')
        ax2.set_title('Control Point Generation Efficiency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png')
        plt.savefig(self.output_dir / 'performance_comparison.pdf')
        plt.close()
    
    def generate_hierarchical_report(self, df: pd.DataFrame, complexity_analysis: Dict):
        """Generate comprehensive hierarchical FFD report."""
        self.logger.info("Generating hierarchical FFD report")
        
        report = []
        report.append("# OpenFFD Hierarchical FFD Performance Analysis Report\n")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("## Executive Summary\n")
        
        # Performance highlights
        best_config = df.loc[df['throughput'].idxmax()]
        worst_config = df.loc[df['throughput'].idxmin()]
        
        report.append(f"### Performance Highlights:\n")
        report.append(f"- Best throughput: {best_config['throughput']:.1f} points/second\n")
        report.append(f"  - Configuration: {best_config['mesh_size']:,} points, "
                     f"depth={best_config['max_depth']}, base={best_config['base_dims']}\n")
        report.append(f"- Lowest throughput: {worst_config['throughput']:.1f} points/second\n")
        report.append(f"  - Configuration: {worst_config['mesh_size']:,} points, "
                     f"depth={worst_config['max_depth']}, base={worst_config['base_dims']}\n")
        
        # Complexity analysis
        if 'depth_scaling' in complexity_analysis:
            depth_analysis = complexity_analysis['depth_scaling']
            report.append(f"\n### Complexity Analysis:\n")
            report.append(f"- Depth scaling: {depth_analysis['growth_type']}\n")
            report.append(f"- Exponential base: {depth_analysis['exponential_base']:.3f}\n")
            
        if 'efficiency' in complexity_analysis:
            efficiency = complexity_analysis['efficiency']
            report.append(f"- Optimal hierarchy depth: {efficiency['optimal_depth']}\n")
            report.append(f"- Best throughput at optimal depth: {efficiency['best_throughput']:.1f} points/second\n")
        
        # Memory analysis
        if 'memory_scaling' in complexity_analysis:
            memory = complexity_analysis['memory_scaling']
            report.append(f"\n### Memory Analysis:\n")
            report.append(f"- Memory per control point: {memory['memory_per_control_point']:.2f} MB\n")
            report.append(f"- Base memory overhead: {memory['base_memory_mb']:.2f} MB\n")
        
        # Recommendations
        report.append("\n## Recommendations\n")
        
        report.append("### For Interactive Applications:\n")
        report.append("- Use hierarchy depth ≤ 3 for real-time performance\n")
        report.append("- Limit subdivision factor to 2 for balanced resolution\n")
        report.append("- Consider 4×4×4 base dimensions for general use\n")
        
        report.append("### For High-Fidelity Applications:\n")
        report.append("- Hierarchy depth 4-5 provides good detail/performance balance\n")
        report.append("- Use larger base dimensions (6×6×6) for complex geometries\n")
        report.append("- Monitor memory usage for depths > 4\n")
        
        report.append("### For Large-Scale Simulations:\n")
        report.append("- Enable parallel processing for meshes > 250K points\n")
        report.append("- Use progressive refinement strategy\n")
        report.append("- Consider memory constraints when designing hierarchy\n")
        
        # Technical insights
        report.append("\n## Technical Insights\n")
        
        avg_creation_time = df.groupby('max_depth')['creation_time'].mean()
        report.append("### Creation Time by Depth:\n")
        for depth, time_val in avg_creation_time.items():
            report.append(f"- Depth {depth}: {time_val:.3f} seconds average\n")
        
        # Control point growth
        avg_control_points = df.groupby('max_depth')['total_control_points'].mean()
        report.append("\n### Control Point Growth:\n")
        for depth, cp_count in avg_control_points.items():
            report.append(f"- Depth {depth}: {cp_count:.0f} control points average\n")
        
        # Save report
        with open(self.output_dir / 'hierarchical_ffd_report.md', 'w') as f:
            f.writelines(report)
        
        self.logger.info(f"Hierarchical FFD report saved to {self.output_dir}/hierarchical_ffd_report.md")
    
    def run_full_benchmark(self):
        """Run the complete hierarchical FFD benchmark suite."""
        self.logger.info("Starting full hierarchical FFD benchmark")
        
        # Run hierarchy depth study
        df = self.run_hierarchy_depth_study()
        
        # Analyze complexity
        complexity_analysis = self.analyze_hierarchy_complexity(df)
        
        # Generate visualizations
        self.plot_hierarchy_depth_analysis(df)
        self.plot_hierarchy_structure_diagram(df)
        self.plot_performance_comparison(df)
        
        # Generate report
        self.generate_hierarchical_report(df, complexity_analysis)
        
        self.logger.info("Full hierarchical FFD benchmark completed successfully!")
        self.logger.info(f"Results available in: {self.output_dir}")


def main():
    """Main entry point for hierarchical FFD benchmark."""
    parser = argparse.ArgumentParser(
        description='Comprehensive hierarchical FFD performance benchmark for OpenFFD'
    )
    parser.add_argument(
        '--output-dir', default='hierarchical_ffd_benchmark_results',
        help='Output directory for benchmark results'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick benchmark with reduced test cases'
    )
    parser.add_argument(
        '--max-depth', type=int, default=6,
        help='Maximum hierarchy depth to test'
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = HierarchicalFFDBenchmark(args.output_dir)
    
    # Adjust parameters for quick run
    if args.quick:
        benchmark.mesh_sizes = [100_000, 500_000]
        benchmark.hierarchy_depths = [2, 3, 4]
        benchmark.subdivision_factors = [2, 3]
        benchmark.base_dimensions = [(4, 4, 4), (6, 6, 6)]
        benchmark.repetitions = 2
    
    # Limit depth based on argument
    benchmark.hierarchy_depths = [d for d in benchmark.hierarchy_depths if d <= args.max_depth]
    
    # Run benchmark
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()