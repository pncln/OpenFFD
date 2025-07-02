#!/usr/bin/env python3
"""
Mesh Complexity Scaling Benchmark for OpenFFD

This benchmark evaluates OpenFFD performance across varying mesh complexities,
geometries, and FFD control grid resolutions. Designed for academic analysis
of computational complexity and scalability characteristics.

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
from matplotlib.ticker import LogFormatter, LogLocator
from scipy import stats
from scipy.optimize import curve_fit

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openffd.core.control_box import create_ffd_box
from openffd.utils.parallel import ParallelConfig
from openffd.utils.benchmark import generate_random_mesh


class MeshComplexityBenchmark:
    """Comprehensive mesh complexity scaling benchmark suite."""
    
    def __init__(self, output_dir: str = "mesh_complexity_results"):
        """Initialize benchmark with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure academic-quality plotting
        self._setup_plotting()
        
        # Initialize logging
        self._setup_logging()
        
        # Benchmark parameters
        self.mesh_sizes = np.logspace(3, 7, 15).astype(int)  # 10^3 to 10^7 points
        self.control_grid_sizes = [
            (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6), (8, 8, 8), 
            (10, 10, 10), (12, 12, 12), (15, 15, 15), (20, 20, 20)
        ]
        self.geometry_types = ['sphere', 'cube', 'cylinder', 'wing', 'random']
        self.aspect_ratios = [0.1, 0.5, 1.0, 2.0, 10.0]  # Width/Height ratios
        self.repetitions = 5
        
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
                logging.FileHandler(self.output_dir / 'mesh_complexity_benchmark.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def generate_geometric_mesh(self, geometry_type: str, num_points: int, 
                               aspect_ratio: float = 1.0, seed: int = 42) -> np.ndarray:
        """Generate mesh points for different geometric configurations."""
        np.random.seed(seed)
        
        if geometry_type == 'sphere':
            # Generate points on and inside a sphere
            phi = np.random.uniform(0, 2*np.pi, num_points)
            costheta = np.random.uniform(-1, 1, num_points)
            u = np.random.uniform(0, 1, num_points)
            
            theta = np.arccos(costheta)
            r = u**(1/3)  # Uniform distribution in volume
            
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi) * aspect_ratio
            z = r * np.cos(theta)
            
        elif geometry_type == 'cube':
            # Generate points in a unit cube
            x = np.random.uniform(-1, 1, num_points)
            y = np.random.uniform(-1, 1, num_points) * aspect_ratio
            z = np.random.uniform(-1, 1, num_points)
            
        elif geometry_type == 'cylinder':
            # Generate points in a cylinder
            theta = np.random.uniform(0, 2*np.pi, num_points)
            r = np.sqrt(np.random.uniform(0, 1, num_points))
            z = np.random.uniform(-1, 1, num_points) * aspect_ratio
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
        elif geometry_type == 'wing':
            # Generate airfoil-like distribution
            chord = np.random.uniform(0, 1, num_points)
            thickness = 0.12 * chord * (1 - chord)  # NACA-like thickness
            
            x = chord
            y = np.random.uniform(-thickness, thickness) * aspect_ratio
            z = np.random.uniform(-0.5, 0.5, num_points)
            
        else:  # 'random'
            # Standard random distribution
            x = np.random.uniform(-1, 1, num_points)
            y = np.random.uniform(-1, 1, num_points) * aspect_ratio
            z = np.random.uniform(-1, 1, num_points)
        
        return np.column_stack([x, y, z])
    
    def run_complexity_benchmark(self, mesh_size: int, control_dim: Tuple[int, int, int],
                                geometry_type: str, aspect_ratio: float) -> Dict:
        """Run single complexity benchmark."""
        self.logger.info(f"Benchmarking: {mesh_size} points, {control_dim} grid, "
                        f"{geometry_type} geometry, AR={aspect_ratio}")
        
        results = []
        
        for rep in range(self.repetitions):
            try:
                # Generate mesh
                start_gen = time.perf_counter()
                mesh_points = self.generate_geometric_mesh(
                    geometry_type, mesh_size, aspect_ratio, seed=42+rep
                )
                gen_time = time.perf_counter() - start_gen
                
                # Configure parallel processing for larger meshes
                parallel_config = ParallelConfig(
                    enabled=mesh_size > 100_000,
                    method='process',
                    max_workers=8,
                    threshold=50_000
                )
                
                # Run FFD creation
                start_ffd = time.perf_counter()
                control_points, bbox = create_ffd_box(
                    mesh_points, control_dim, margin=0.1,
                    custom_dims=None, parallel_config=parallel_config
                )
                ffd_time = time.perf_counter() - start_ffd
                
                # Calculate complexity metrics
                num_control_points = len(control_points)
                total_operations = mesh_size * num_control_points
                
                # Geometric metrics
                bbox_volume = np.prod(bbox[1] - bbox[0])
                point_density = mesh_size / bbox_volume if bbox_volume > 0 else 0
                
                # Aspect ratio of bounding box
                dims = bbox[1] - bbox[0]
                computed_aspect_ratio = max(dims) / min(dims) if min(dims) > 0 else 1.0
                
                results.append({
                    'mesh_size': mesh_size,
                    'control_dim': control_dim,
                    'geometry_type': geometry_type,
                    'aspect_ratio': aspect_ratio,
                    'repetition': rep,
                    'generation_time': gen_time,
                    'ffd_time': ffd_time,
                    'total_time': gen_time + ffd_time,
                    'num_control_points': num_control_points,
                    'total_operations': total_operations,
                    'operations_per_second': total_operations / ffd_time if ffd_time > 0 else 0,
                    'bbox_volume': bbox_volume,
                    'point_density': point_density,
                    'computed_aspect_ratio': computed_aspect_ratio,
                    'memory_usage_estimate': mesh_size * num_control_points * 8 / (1024**2)  # MB
                })
                
            except Exception as e:
                self.logger.error(f"Error in complexity benchmark: {e}")
                results.append({
                    'mesh_size': mesh_size,
                    'control_dim': control_dim,
                    'geometry_type': geometry_type,
                    'aspect_ratio': aspect_ratio,
                    'repetition': rep,
                    'generation_time': float('inf'),
                    'ffd_time': float('inf'),
                    'total_time': float('inf'),
                    'num_control_points': 0,
                    'total_operations': 0,
                    'operations_per_second': 0,
                    'bbox_volume': 0,
                    'point_density': 0,
                    'computed_aspect_ratio': 0,
                    'memory_usage_estimate': 0
                })
        
        return results
    
    def run_full_complexity_study(self) -> pd.DataFrame:
        """Run comprehensive complexity scaling study."""
        self.logger.info("Starting comprehensive mesh complexity study")
        
        all_results = []
        
        # Primary study: mesh size vs execution time
        self.logger.info("Phase 1: Mesh size scaling analysis")
        for mesh_size in self.mesh_sizes:
            for control_dim in [(4, 4, 4), (8, 8, 8), (12, 12, 12)]:  # Representative grids
                for geometry_type in ['cube', 'sphere', 'wing']:  # Key geometries
                    results = self.run_complexity_benchmark(
                        mesh_size, control_dim, geometry_type, 1.0
                    )
                    all_results.extend(results)
        
        # Secondary study: control grid resolution
        self.logger.info("Phase 2: Control grid resolution analysis")
        for control_dim in self.control_grid_sizes:
            for mesh_size in [100_000, 1_000_000]:  # Fixed mesh sizes
                results = self.run_complexity_benchmark(
                    mesh_size, control_dim, 'cube', 1.0
                )
                all_results.extend(results)
        
        # Tertiary study: geometry effects
        self.logger.info("Phase 3: Geometry and aspect ratio analysis")
        for geometry_type in self.geometry_types:
            for aspect_ratio in self.aspect_ratios:
                results = self.run_complexity_benchmark(
                    500_000, (8, 8, 8), geometry_type, aspect_ratio
                )
                all_results.extend(results)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Save raw results
        df.to_csv(self.output_dir / 'mesh_complexity_raw_results.csv', index=False)
        
        # Calculate statistics
        df_stats = df.groupby(['mesh_size', 'control_dim', 'geometry_type', 'aspect_ratio']).agg({
            'ffd_time': ['mean', 'std', 'min', 'max'],
            'operations_per_second': ['mean', 'std'],
            'memory_usage_estimate': ['mean'],
            'point_density': ['mean']
        }).round(6)
        
        df_stats.to_csv(self.output_dir / 'mesh_complexity_statistics.csv')
        
        self.logger.info(f"Complexity study completed. Results saved to {self.output_dir}")
        return df
    
    def analyze_computational_complexity(self, df: pd.DataFrame) -> Dict:
        """Analyze computational complexity relationships."""
        self.logger.info("Analyzing computational complexity")
        
        complexity_analysis = {}
        
        # Analyze scaling with mesh size
        mesh_scaling_data = df[
            (df['control_dim'] == (8, 8, 8)) & 
            (df['geometry_type'] == 'cube') &
            (df['aspect_ratio'] == 1.0)
        ].groupby('mesh_size')['ffd_time'].mean().reset_index()
        
        if len(mesh_scaling_data) > 3:
            # Fit polynomial models
            log_mesh_size = np.log10(mesh_scaling_data['mesh_size'])
            log_time = np.log10(mesh_scaling_data['ffd_time'])
            
            # Linear fit in log space (power law)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_mesh_size, log_time)
            
            complexity_analysis['mesh_scaling'] = {
                'exponent': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'complexity_class': self._classify_complexity(slope)
            }
        
        # Analyze scaling with control grid size
        control_scaling_data = df[
            (df['mesh_size'] == 1_000_000) & 
            (df['geometry_type'] == 'cube') &
            (df['aspect_ratio'] == 1.0)
        ].copy()
        
        control_scaling_data['grid_volume'] = control_scaling_data['control_dim'].apply(
            lambda x: x[0] * x[1] * x[2] if isinstance(x, tuple) else np.prod(eval(x))
        )
        
        control_scaling_summary = control_scaling_data.groupby('grid_volume')['ffd_time'].mean().reset_index()
        
        if len(control_scaling_summary) > 3:
            log_grid_vol = np.log10(control_scaling_summary['grid_volume'])
            log_time_ctrl = np.log10(control_scaling_summary['ffd_time'])
            
            slope_ctrl, intercept_ctrl, r_value_ctrl, p_value_ctrl, std_err_ctrl = stats.linregress(
                log_grid_vol, log_time_ctrl
            )
            
            complexity_analysis['control_scaling'] = {
                'exponent': slope_ctrl,
                'r_squared': r_value_ctrl**2,
                'p_value': p_value_ctrl,
                'complexity_class': self._classify_complexity(slope_ctrl)
            }
        
        # Geometry effect analysis
        geometry_effects = df[
            (df['mesh_size'] == 500_000) &
            (df['control_dim'] == (8, 8, 8)) &
            (df['aspect_ratio'] == 1.0)
        ].groupby('geometry_type')['ffd_time'].agg(['mean', 'std']).reset_index()
        
        complexity_analysis['geometry_effects'] = geometry_effects.to_dict('records')
        
        # Save analysis
        with open(self.output_dir / 'computational_complexity_analysis.json', 'w') as f:
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
    
    def plot_mesh_scaling_analysis(self, df: pd.DataFrame):
        """Create mesh size scaling analysis plots."""
        self.logger.info("Creating mesh scaling visualizations")
        
        # Filter for main scaling analysis
        scaling_data = df[
            (df['geometry_type'] == 'cube') &
            (df['aspect_ratio'] == 1.0)
        ].groupby(['mesh_size', 'control_dim'])['ffd_time'].mean().reset_index()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Publication-quality colors
        
        # Plot 1: Execution time vs mesh size (enhanced for publications)
        fig, ax = create_publication_figure(figsize=(8, 6))
        
        for idx, control_dim in enumerate([(4, 4, 4), (8, 8, 8), (12, 12, 12)]):
            subset = scaling_data[scaling_data['control_dim'] == control_dim]
            if not subset.empty:
                ax.loglog(subset['mesh_size'], subset['ffd_time'], 'o-', 
                          color=colors[idx], label=f'{control_dim[0]}×{control_dim[1]}×{control_dim[2]}',
                          linewidth=3.0, markersize=8, markerfacecolor='white',
                          markeredgewidth=2.0, markeredgecolor=colors[idx])
        
        # Add reference lines for different complexities
        ref_mesh_sizes = np.array([1e3, 1e7])
        ax.loglog(ref_mesh_sizes, 1e-6 * ref_mesh_sizes, 'k--', alpha=0.8, label='O(n)', linewidth=2.5)
        ax.loglog(ref_mesh_sizes, 1e-11 * ref_mesh_sizes**2, 'k:', alpha=0.8, label='O(n²)', linewidth=2.5)
        
        ax.set_xlabel('Mesh Size (points)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_title('FFD Creation Time vs Mesh Size', fontsize=16, fontweight='bold')
        
        # Enhanced legend
        legend = ax.legend(fontsize=12, frameon=True, fancybox=False, shadow=False,
                          framealpha=0.95, edgecolor='black', loc='best')
        legend.get_frame().set_linewidth(1.2)
        
        # Apply enhanced visibility and optimize axis limits
        enhance_plot_visibility(ax, grid_alpha=0.4, spine_width=1.5)
        all_x_data = np.concatenate([scaling_data[scaling_data['control_dim'] == cd]['mesh_size'].values 
                                   for cd in [(4, 4, 4), (8, 8, 8), (12, 12, 12)] 
                                   if not scaling_data[scaling_data['control_dim'] == cd].empty])
        all_y_data = np.concatenate([scaling_data[scaling_data['control_dim'] == cd]['ffd_time'].values 
                                   for cd in [(4, 4, 4), (8, 8, 8), (12, 12, 12)] 
                                   if not scaling_data[scaling_data['control_dim'] == cd].empty])
        optimize_axis_limits(ax, x_data=all_x_data, y_data=all_y_data, log_scale=(True, True))
        
        plt.tight_layout()
        
        # Save without title for publications
        original_title = ax.get_title()
        ax.set_title('')
        plt.savefig(self.output_dir / 'ffd_creation_time_scaling.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'ffd_creation_time_scaling.pdf', dpi=300, bbox_inches='tight', facecolor='white')
        ax.set_title(original_title)  # Restore for display
        plt.close()
        
        # Plot 2: Throughput analysis (enhanced for publications)
        fig, ax = create_publication_figure(figsize=(8, 6))
        
        scaling_data['throughput'] = scaling_data['mesh_size'] / scaling_data['ffd_time']
        
        for idx, control_dim in enumerate([(4, 4, 4), (8, 8, 8), (12, 12, 12)]):
            subset = scaling_data[scaling_data['control_dim'] == control_dim]
            if not subset.empty:
                ax.semilogx(subset['mesh_size'], subset['throughput'], 'o-',
                           color=colors[idx], label=f'{control_dim[0]}×{control_dim[1]}×{control_dim[2]}',
                           linewidth=3.0, markersize=8, markerfacecolor='white',
                           markeredgewidth=2.0, markeredgecolor=colors[idx])
        
        ax.set_xlabel('Mesh Size (points)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Throughput (points/second)', fontsize=14, fontweight='bold')
        ax.set_title('Processing Throughput vs Mesh Size', fontsize=16, fontweight='bold')
        
        # Enhanced legend and grid
        legend = ax.legend(fontsize=12, frameon=True, fancybox=False, shadow=False,
                          framealpha=0.95, edgecolor='black', loc='best')
        legend.get_frame().set_linewidth(1.2)
        
        # Apply enhanced visibility and optimize axis limits
        enhance_plot_visibility(ax, grid_alpha=0.4, spine_width=1.5)
        all_x_data = np.concatenate([scaling_data[scaling_data['control_dim'] == cd]['mesh_size'].values 
                                   for cd in [(4, 4, 4), (8, 8, 8), (12, 12, 12)] 
                                   if not scaling_data[scaling_data['control_dim'] == cd].empty])
        optimize_axis_limits(ax, x_data=all_x_data, y_data=scaling_data['throughput'], log_scale=(True, False))
        
        plt.tight_layout()
        
        # Save without title for publications
        original_title = ax.get_title()
        ax.set_title('')
        plt.savefig(self.output_dir / 'processing_throughput_scaling.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'processing_throughput_scaling.pdf', dpi=300, bbox_inches='tight', facecolor='white')
        ax.set_title(original_title)  # Restore for display
        plt.close()
    
    def plot_control_grid_analysis(self, df: pd.DataFrame):
        """Create control grid resolution analysis plots."""
        self.logger.info("Creating control grid analysis")
        
        # Filter for control grid analysis
        grid_data = df[
            (df['mesh_size'] == 1_000_000) &
            (df['geometry_type'] == 'cube') &
            (df['aspect_ratio'] == 1.0)
        ].copy()
        
        grid_data['grid_volume'] = grid_data['control_dim'].apply(
            lambda x: x[0] * x[1] * x[2] if isinstance(x, tuple) else np.prod(eval(x))
        )
        
        grid_summary = grid_data.groupby(['grid_volume', 'control_dim']).agg({
            'ffd_time': 'mean',
            'memory_usage_estimate': 'mean',
            'operations_per_second': 'mean'
        }).reset_index()
        
        # Plot 1: Execution time vs grid resolution (enhanced for publications)
        fig, ax = create_publication_figure(figsize=(8, 6))
        
        ax.loglog(grid_summary['grid_volume'], grid_summary['ffd_time'], 'o-',
                  color='#1f77b4', linewidth=3.0, markersize=8, markerfacecolor='white',
                  markeredgewidth=2.0, markeredgecolor='#1f77b4')
        
        # Add reference lines
        ref_volumes = np.array([10, 10000])
        ax.loglog(ref_volumes, 1e-3 * ref_volumes, 'k--', alpha=0.8, label='O(n)', linewidth=2.5)
        ax.loglog(ref_volumes, 1e-6 * ref_volumes**1.5, 'k:', alpha=0.8, label='O(n^1.5)', linewidth=2.5)
        
        ax.set_xlabel('Control Grid Volume (control points)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_title('FFD Time vs Control Grid Resolution', fontsize=16, fontweight='bold')
        
        # Enhanced legend
        legend = ax.legend(fontsize=12, frameon=True, fancybox=False, shadow=False,
                          framealpha=0.95, edgecolor='black', loc='best')
        legend.get_frame().set_linewidth(1.2)
        
        enhance_plot_visibility(ax, grid_alpha=0.4, spine_width=1.5)
        optimize_axis_limits(ax, x_data=grid_summary['grid_volume'], y_data=grid_summary['ffd_time'], log_scale=(True, True))
        
        plt.tight_layout()
        
        # Save without title for publications
        original_title = ax.get_title()
        ax.set_title('')
        plt.savefig(self.output_dir / 'control_grid_execution_time.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'control_grid_execution_time.pdf', dpi=300, bbox_inches='tight', facecolor='white')
        ax.set_title(original_title)  # Restore for display
        plt.close()
        
        # Plot 2: Memory usage vs grid resolution (enhanced for publications)
        fig, ax = create_publication_figure(figsize=(8, 6))
        
        ax.loglog(grid_summary['grid_volume'], grid_summary['memory_usage_estimate'], 'o-',
                  color='#d62728', linewidth=3.0, markersize=8, markerfacecolor='white',
                  markeredgewidth=2.0, markeredgecolor='#d62728')
        
        ax.set_xlabel('Control Grid Volume (control points)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Estimated Memory Usage (MB)', fontsize=14, fontweight='bold')
        ax.set_title('Memory Requirements vs Control Grid Resolution', fontsize=16, fontweight='bold')
        
        enhance_plot_visibility(ax, grid_alpha=0.4, spine_width=1.5)
        optimize_axis_limits(ax, x_data=grid_summary['grid_volume'], y_data=grid_summary['memory_usage_estimate'], log_scale=(True, True))
        
        plt.tight_layout()
        
        # Save without title for publications
        original_title = ax.get_title()
        ax.set_title('')
        plt.savefig(self.output_dir / 'control_grid_memory_usage.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'control_grid_memory_usage.pdf', dpi=300, bbox_inches='tight', facecolor='white')
        ax.set_title(original_title)  # Restore for display
        plt.close()
    
    def plot_geometry_effects(self, df: pd.DataFrame):
        """Create geometry and aspect ratio effect plots."""
        self.logger.info("Creating geometry effects visualization")
        
        # Plot 1: Geometry type effects (enhanced for publications)
        fig, ax = create_publication_figure(figsize=(8, 6))
        
        geometry_data = df[
            (df['mesh_size'] == 500_000) &
            (df['control_dim'] == (8, 8, 8)) &
            (df['aspect_ratio'] == 1.0)
        ].groupby('geometry_type').agg({
            'ffd_time': ['mean', 'std'],
            'point_density': 'mean'
        }).round(4)
        
        geometry_means = geometry_data['ffd_time']['mean']
        geometry_stds = geometry_data['ffd_time']['std']
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(geometry_means)]
        bars = ax.bar(range(len(geometry_means)), geometry_means, 
                      yerr=geometry_stds, capsize=8, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Geometry Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_title('FFD Performance by Geometry Type', fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(geometry_means)))
        ax.set_xticklabels(geometry_means.index, rotation=45, fontsize=12)
        
        enhance_plot_visibility(ax, grid_alpha=0.4, spine_width=1.5)
        optimize_axis_limits(ax, y_data=geometry_means.values, ensure_zero=True)
        
        # Add value labels on bars with enhanced readability
        for bar, mean_val, std_val in zip(bars, geometry_means, geometry_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.05*height,
                    f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save without title for publications
        original_title = ax.get_title()
        ax.set_title('')
        plt.savefig(self.output_dir / 'geometry_type_effects.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'geometry_type_effects.pdf', dpi=300, bbox_inches='tight', facecolor='white')
        ax.set_title(original_title)  # Restore for display
        plt.close()
        
        # Plot 2: Aspect ratio effects (enhanced for publications)
        fig, ax = create_publication_figure(figsize=(8, 6))
        
        aspect_data = df[
            (df['mesh_size'] == 500_000) &
            (df['control_dim'] == (8, 8, 8)) &
            (df['geometry_type'] == 'cube')
        ].groupby('aspect_ratio').agg({
            'ffd_time': ['mean', 'std'],
            'computed_aspect_ratio': 'mean'
        }).round(4)
        
        aspect_means = aspect_data['ffd_time']['mean']
        aspect_stds = aspect_data['ffd_time']['std']
        
        ax.errorbar(aspect_means.index, aspect_means, yerr=aspect_stds,
                    marker='o', capsize=8, linewidth=3.0, markersize=8,
                    markerfacecolor='white', markeredgewidth=2.0, 
                    markeredgecolor='#1f77b4', color='#1f77b4')
        
        ax.set_xlabel('Prescribed Aspect Ratio', fontsize=14, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_title('FFD Performance vs Geometry Aspect Ratio', fontsize=16, fontweight='bold')
        ax.set_xscale('log')
        
        enhance_plot_visibility(ax, grid_alpha=0.4, spine_width=1.5)
        optimize_axis_limits(ax, y_data=aspect_means.values, log_scale=(True, False), ensure_zero=True)
        
        plt.tight_layout()
        
        # Save without title for publications
        original_title = ax.get_title()
        ax.set_title('')
        plt.savefig(self.output_dir / 'aspect_ratio_effects.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'aspect_ratio_effects.pdf', dpi=300, bbox_inches='tight', facecolor='white')
        ax.set_title(original_title)  # Restore for display
        plt.close()
    
    def generate_complexity_report(self, df: pd.DataFrame, complexity_analysis: Dict):
        """Generate comprehensive complexity analysis report."""
        self.logger.info("Generating complexity analysis report")
        
        report = []
        report.append("# OpenFFD Mesh Complexity Scaling Analysis Report\n")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("## Executive Summary\n")
        
        # Mesh scaling summary
        if 'mesh_scaling' in complexity_analysis:
            mesh_analysis = complexity_analysis['mesh_scaling']
            report.append(f"### Mesh Size Scaling:\n")
            report.append(f"- Computational complexity: {mesh_analysis['complexity_class']}\n")
            report.append(f"- Scaling exponent: {mesh_analysis['exponent']:.3f}\n")
            report.append(f"- R² correlation: {mesh_analysis['r_squared']:.4f}\n")
            
        # Control grid scaling summary
        if 'control_scaling' in complexity_analysis:
            control_analysis = complexity_analysis['control_scaling']
            report.append(f"### Control Grid Scaling:\n")
            report.append(f"- Computational complexity: {control_analysis['complexity_class']}\n")
            report.append(f"- Scaling exponent: {control_analysis['exponent']:.3f}\n")
            report.append(f"- R² correlation: {control_analysis['r_squared']:.4f}\n")
        
        # Performance benchmarks
        report.append("\n## Performance Benchmarks\n")
        
        # Find best and worst performing configurations
        performance_summary = df.groupby(['mesh_size', 'control_dim'])['ffd_time'].mean()
        best_config = performance_summary.idxmin()
        worst_config = performance_summary.idxmax()
        
        report.append(f"### Best Performance:\n")
        report.append(f"- Configuration: {best_config[0]:,} points, {best_config[1]} grid\n")
        report.append(f"- Execution time: {performance_summary[best_config]:.4f} seconds\n")
        
        report.append(f"### Most Challenging:\n")
        report.append(f"- Configuration: {worst_config[0]:,} points, {worst_config[1]} grid\n")
        report.append(f"- Execution time: {performance_summary[worst_config]:.4f} seconds\n")
        
        # Geometry effects
        if 'geometry_effects' in complexity_analysis:
            report.append("\n## Geometry Effects\n")
            geometry_effects = complexity_analysis['geometry_effects']
            
            fastest_geom = min(geometry_effects, key=lambda x: x['mean'])
            slowest_geom = max(geometry_effects, key=lambda x: x['mean'])
            
            report.append(f"- Fastest geometry: {fastest_geom['geometry_type']} "
                         f"({fastest_geom['mean']:.4f}±{fastest_geom['std']:.4f} s)\n")
            report.append(f"- Slowest geometry: {slowest_geom['geometry_type']} "
                         f"({slowest_geom['mean']:.4f}±{slowest_geom['std']:.4f} s)\n")
        
        # Recommendations
        report.append("\n## Recommendations\n")
        
        report.append("### For Large-Scale Applications:\n")
        report.append("- Use parallel processing for meshes >100,000 points\n")
        report.append("- Consider control grid resolution vs. accuracy trade-offs\n")
        report.append("- Simple geometries (cube, sphere) show best performance\n")
        
        report.append("### For Memory-Constrained Systems:\n")
        report.append("- Limit control grid resolution to 8×8×8 or smaller\n")
        report.append("- Process large meshes in chunks\n")
        report.append("- Monitor memory usage for grids >12×12×12\n")
        
        # Save report
        with open(self.output_dir / 'mesh_complexity_report.md', 'w') as f:
            f.writelines(report)
        
        self.logger.info(f"Complexity report saved to {self.output_dir}/mesh_complexity_report.md")
    
    def run_full_benchmark(self):
        """Run the complete mesh complexity benchmark suite."""
        self.logger.info("Starting full mesh complexity benchmark")
        
        # Run complexity study
        df = self.run_full_complexity_study()
        
        # Analyze computational complexity
        complexity_analysis = self.analyze_computational_complexity(df)
        
        # Generate visualizations
        self.plot_mesh_scaling_analysis(df)
        self.plot_control_grid_analysis(df)
        self.plot_geometry_effects(df)
        
        # Generate report
        self.generate_complexity_report(df, complexity_analysis)
        
        self.logger.info("Full mesh complexity benchmark completed successfully!")
        self.logger.info(f"Results available in: {self.output_dir}")


def main():
    """Main entry point for mesh complexity benchmark."""
    parser = argparse.ArgumentParser(
        description='Comprehensive mesh complexity scaling benchmark for OpenFFD'
    )
    parser.add_argument(
        '--output-dir', default='mesh_complexity_benchmark_results',
        help='Output directory for benchmark results'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick benchmark with reduced test cases'
    )
    parser.add_argument(
        '--max-mesh-size', type=int, default=10_000_000,
        help='Maximum mesh size to test'
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = MeshComplexityBenchmark(args.output_dir)
    
    # Adjust parameters for quick run
    if args.quick:
        benchmark.mesh_sizes = np.logspace(3, 6, 8).astype(int)  # Reduced range
        benchmark.control_grid_sizes = [(4, 4, 4), (8, 8, 8), (12, 12, 12)]
        benchmark.geometry_types = ['cube', 'sphere', 'wing']
        benchmark.aspect_ratios = [0.5, 1.0, 2.0]
        benchmark.repetitions = 3
    
    # Limit mesh sizes based on argument
    benchmark.mesh_sizes = benchmark.mesh_sizes[benchmark.mesh_sizes <= args.max_mesh_size]
    
    # Run benchmark
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()