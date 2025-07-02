#!/usr/bin/env python3
"""
Visualization Performance Benchmark for OpenFFD

This benchmark evaluates the performance of OpenFFD's visualization subsystem,
including mesh rendering, parallel visualization processing, and memory usage
patterns under different visualization configurations.

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
from typing import Dict, List, Tuple, Optional, Any

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
import psutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openffd.utils.parallel import ParallelConfig
from openffd.utils.benchmark import generate_random_mesh

# Try to import visualization modules
try:
    from openffd.visualization.parallel_viz import (
        process_point_cloud_parallel,
        subsample_points_parallel,
        create_mesh_chunks_parallel,
        compute_normals_parallel,
        extract_mesh_features_parallel
    )
    from openffd.visualization.mesh_viz import create_mesh_plot
    VIZ_AVAILABLE = True
except ImportError as e:
    VIZ_AVAILABLE = False
    print(f"Warning: Visualization modules not fully available: {e}")

# Try to import PyVista for enhanced benchmarking
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not available. Some benchmarks may be limited.")


class VisualizationBenchmark:
    """Comprehensive visualization performance benchmark suite."""
    
    def __init__(self, output_dir: str = "visualization_benchmark_results"):
        """Initialize benchmark with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure academic-quality plotting
        self._setup_plotting()
        
        # Initialize logging
        self._setup_logging()
        
        # Benchmark parameters
        self.mesh_sizes = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
        self.worker_counts = [1, 2, 4, 8, 12, 16]
        self.visualization_modes = ['points', 'wireframe', 'surface', 'volume']
        self.quality_levels = ['low', 'medium', 'high', 'ultra']
        self.repetitions = 3
        
        # Performance tracking
        self.process = psutil.Process()
        
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
                logging.FileHandler(self.output_dir / 'visualization_benchmark.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def generate_test_mesh(self, mesh_size: int, with_faces: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate test mesh with optional face connectivity."""
        points = generate_random_mesh(mesh_size, seed=42)
        
        if not with_faces:
            return points, None
        
        # Generate simple triangular faces for surface visualization
        # This is a simplified approach - in practice, proper Delaunay triangulation would be used
        if mesh_size < 50_000:  # Only generate faces for smaller meshes to avoid memory issues
            from scipy.spatial import Delaunay
            
            # Project to 2D for triangulation (use x-y plane)
            points_2d = points[:, :2]
            tri = Delaunay(points_2d)
            faces = tri.simplices
            
            return points, faces
        else:
            # For large meshes, skip face generation to avoid memory issues
            return points, None
    
    def benchmark_point_processing(self, mesh_size: int, workers: int) -> Dict:
        """Benchmark point cloud processing performance."""
        self.logger.info(f"Benchmarking point processing: {mesh_size} points, {workers} workers")
        
        results = []
        
        for rep in range(self.repetitions):
            try:
                # Generate mesh
                points, _ = self.generate_test_mesh(mesh_size, with_faces=False)
                
                # Configure parallel processing
                parallel_config = ParallelConfig(
                    enabled=workers > 1,
                    method='process',
                    max_workers=workers,
                    threshold=10_000
                )
                
                # Memory before processing
                memory_before = self.process.memory_info().rss / (1024 * 1024)  # MB
                
                if VIZ_AVAILABLE:
                    # Benchmark subsampling
                    start_time = time.perf_counter()
                    subsampled = subsample_points_parallel(points, max_points=10_000, config=parallel_config)
                    subsample_time = time.perf_counter() - start_time
                    
                    # Benchmark feature extraction
                    start_time = time.perf_counter()
                    features = extract_mesh_features_parallel(
                        points, 
                        compute_bbox=True, 
                        compute_center=True, 
                        config=parallel_config
                    )
                    feature_time = time.perf_counter() - start_time
                else:
                    # Fallback to basic numpy operations
                    start_time = time.perf_counter()
                    indices = np.linspace(0, len(points) - 1, min(10_000, len(points)), dtype=int)
                    subsampled = points[indices]
                    subsample_time = time.perf_counter() - start_time
                    
                    start_time = time.perf_counter()
                    features = {
                        'bbox_min': np.min(points, axis=0),
                        'bbox_max': np.max(points, axis=0),
                        'center': np.mean(points, axis=0)
                    }
                    feature_time = time.perf_counter() - start_time
                
                # Memory after processing
                memory_after = self.process.memory_info().rss / (1024 * 1024)  # MB
                memory_usage = memory_after - memory_before
                
                results.append({
                    'mesh_size': mesh_size,
                    'workers': workers,
                    'repetition': rep,
                    'subsample_time': subsample_time,
                    'feature_time': feature_time,
                    'total_time': subsample_time + feature_time,
                    'memory_usage_mb': memory_usage,
                    'subsampled_points': len(subsampled),
                    'throughput_points_per_sec': mesh_size / (subsample_time + feature_time),
                    'processing_efficiency': mesh_size / memory_usage if memory_usage > 0 else 0
                })
                
            except Exception as e:
                self.logger.error(f"Error in point processing benchmark: {e}")
                results.append({
                    'mesh_size': mesh_size,
                    'workers': workers,
                    'repetition': rep,
                    'subsample_time': float('inf'),
                    'feature_time': float('inf'),
                    'total_time': float('inf'),
                    'memory_usage_mb': 0,
                    'subsampled_points': 0,
                    'throughput_points_per_sec': 0,
                    'processing_efficiency': 0
                })
        
        return results
    
    def benchmark_surface_rendering(self, mesh_size: int, quality_level: str) -> Dict:
        """Benchmark surface rendering performance."""
        self.logger.info(f"Benchmarking surface rendering: {mesh_size} points, {quality_level} quality")
        
        results = []
        
        # Quality parameters
        quality_params = {
            'low': {'max_triangles': 5_000, 'point_size': 2.0, 'line_width': 1.0},
            'medium': {'max_triangles': 20_000, 'point_size': 3.0, 'line_width': 1.5},
            'high': {'max_triangles': 100_000, 'point_size': 4.0, 'line_width': 2.0},
            'ultra': {'max_triangles': 500_000, 'point_size': 5.0, 'line_width': 2.5}
        }
        
        params = quality_params[quality_level]
        
        for rep in range(self.repetitions):
            try:
                # Generate mesh
                points, faces = self.generate_test_mesh(mesh_size, with_faces=True)
                
                # Memory before rendering
                memory_before = self.process.memory_info().rss / (1024 * 1024)  # MB
                
                # Benchmark mesh preparation
                start_time = time.perf_counter()
                
                if faces is not None and VIZ_AVAILABLE:
                    # Create mesh chunks for large meshes
                    if len(faces) > params['max_triangles']:
                        chunk_results = create_mesh_chunks_parallel(
                            points, faces, params['max_triangles']
                        )
                        num_chunks = len(chunk_results)
                        
                        # Compute normals for surface rendering
                        if len(faces) < 100_000:  # Limit normal computation for very large meshes
                            normals = compute_normals_parallel(points, faces)
                        else:
                            normals = None
                    else:
                        chunk_results = [(points, faces)]
                        num_chunks = 1
                        normals = compute_normals_parallel(points, faces) if len(faces) < 50_000 else None
                else:
                    # Point cloud only
                    chunk_results = [(points, None)]
                    num_chunks = 1
                    normals = None
                
                mesh_prep_time = time.perf_counter() - start_time
                
                # Benchmark visualization data structure creation
                start_time = time.perf_counter()
                
                if PYVISTA_AVAILABLE and faces is not None:
                    # Create PyVista mesh for realistic rendering benchmark
                    mesh = pv.PolyData(points, faces)
                    if normals is not None:
                        mesh['normals'] = normals
                else:
                    # Create simple point cloud
                    mesh = {'points': points, 'faces': faces, 'normals': normals}
                
                viz_creation_time = time.perf_counter() - start_time
                
                # Memory after processing
                memory_after = self.process.memory_info().rss / (1024 * 1024)  # MB
                memory_usage = memory_after - memory_before
                
                # Calculate rendering complexity metrics
                triangles_rendered = len(faces) if faces is not None else 0
                points_rendered = len(points)
                
                results.append({
                    'mesh_size': mesh_size,
                    'quality_level': quality_level,
                    'repetition': rep,
                    'mesh_prep_time': mesh_prep_time,
                    'viz_creation_time': viz_creation_time,
                    'total_time': mesh_prep_time + viz_creation_time,
                    'memory_usage_mb': memory_usage,
                    'num_chunks': num_chunks,
                    'triangles_rendered': triangles_rendered,
                    'points_rendered': points_rendered,
                    'rendering_throughput': triangles_rendered / (mesh_prep_time + viz_creation_time) if triangles_rendered > 0 else 0,
                    'memory_efficiency': points_rendered / memory_usage if memory_usage > 0 else 0
                })
                
            except Exception as e:
                self.logger.error(f"Error in surface rendering benchmark: {e}")
                results.append({
                    'mesh_size': mesh_size,
                    'quality_level': quality_level,
                    'repetition': rep,
                    'mesh_prep_time': float('inf'),
                    'viz_creation_time': float('inf'),
                    'total_time': float('inf'),
                    'memory_usage_mb': 0,
                    'num_chunks': 0,
                    'triangles_rendered': 0,
                    'points_rendered': 0,
                    'rendering_throughput': 0,
                    'memory_efficiency': 0
                })
        
        return results
    
    def benchmark_parallel_visualization(self, mesh_size: int, mode: str, workers: int) -> Dict:
        """Benchmark parallel visualization processing."""
        self.logger.info(f"Benchmarking parallel viz: {mesh_size} points, {mode} mode, {workers} workers")
        
        results = []
        
        for rep in range(self.repetitions):
            try:
                # Generate appropriate test data based on mode
                if mode in ['surface', 'wireframe']:
                    points, faces = self.generate_test_mesh(mesh_size, with_faces=True)
                else:
                    points, faces = self.generate_test_mesh(mesh_size, with_faces=False)
                
                # Configure parallel processing
                parallel_config = ParallelConfig(
                    enabled=workers > 1,
                    method='process',
                    max_workers=workers,
                    threshold=10_000
                )
                
                # Memory before processing
                memory_before = self.process.memory_info().rss / (1024 * 1024)  # MB
                
                # Benchmark different visualization modes
                start_time = time.perf_counter()
                
                if mode == 'points':
                    # Point cloud processing
                    if VIZ_AVAILABLE:
                        processed = subsample_points_parallel(points, max_points=50_000, config=parallel_config)
                        features = extract_mesh_features_parallel(points, config=parallel_config)
                    else:
                        indices = np.linspace(0, len(points) - 1, min(50_000, len(points)), dtype=int)
                        processed = points[indices]
                        features = {'center': np.mean(points, axis=0)}
                    
                elif mode == 'wireframe':
                    # Wireframe processing (edges)
                    if faces is not None:
                        # Extract edges from faces
                        edges = set()
                        for face in faces:
                            for i in range(len(face)):
                                edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
                                edges.add(edge)
                        processed = np.array(list(edges))
                    else:
                        processed = points
                    
                elif mode == 'surface':
                    # Surface mesh processing
                    if faces is not None and VIZ_AVAILABLE:
                        if len(faces) > 50_000:
                            chunk_results = create_mesh_chunks_parallel(points, faces, 50_000, parallel_config)
                            processed = chunk_results
                        else:
                            normals = compute_normals_parallel(points, faces, parallel_config)
                            processed = (points, faces, normals)
                    else:
                        processed = points
                        
                elif mode == 'volume':
                    # Volume rendering preparation
                    if VIZ_AVAILABLE:
                        features = extract_mesh_features_parallel(points, config=parallel_config)
                        bbox_min, bbox_max = features['bbox_min'], features['bbox_max']
                        
                        # Create volume grid (simplified)
                        grid_size = min(64, int(np.ceil(len(points) ** (1/3))))
                        volume_grid = np.zeros((grid_size, grid_size, grid_size))
                        processed = volume_grid
                    else:
                        processed = points
                
                processing_time = time.perf_counter() - start_time
                
                # Memory after processing
                memory_after = self.process.memory_info().rss / (1024 * 1024)  # MB
                memory_usage = memory_after - memory_before
                
                # Calculate mode-specific metrics
                if mode == 'points':
                    elements_processed = len(processed) if hasattr(processed, '__len__') else mesh_size
                elif mode == 'wireframe':
                    elements_processed = len(processed) if hasattr(processed, '__len__') else 0
                elif mode == 'surface':
                    elements_processed = len(faces) if faces is not None else 0
                elif mode == 'volume':
                    elements_processed = processed.size if hasattr(processed, 'size') else mesh_size
                else:
                    elements_processed = mesh_size
                
                results.append({
                    'mesh_size': mesh_size,
                    'mode': mode,
                    'workers': workers,
                    'repetition': rep,
                    'processing_time': processing_time,
                    'memory_usage_mb': memory_usage,
                    'elements_processed': elements_processed,
                    'throughput_elements_per_sec': elements_processed / processing_time if processing_time > 0 else 0,
                    'memory_efficiency': elements_processed / memory_usage if memory_usage > 0 else 0,
                    'scalability_factor': (mesh_size / elements_processed) if elements_processed > 0 else 1
                })
                
            except Exception as e:
                self.logger.error(f"Error in parallel visualization benchmark: {e}")
                results.append({
                    'mesh_size': mesh_size,
                    'mode': mode,
                    'workers': workers,
                    'repetition': rep,
                    'processing_time': float('inf'),
                    'memory_usage_mb': 0,
                    'elements_processed': 0,
                    'throughput_elements_per_sec': 0,
                    'memory_efficiency': 0,
                    'scalability_factor': 1
                })
        
        return results
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run comprehensive visualization performance study."""
        self.logger.info("Starting comprehensive visualization benchmark")
        
        all_results = []
        
        # Phase 1: Point processing scalability
        self.logger.info("Phase 1: Point processing scalability")
        for mesh_size in self.mesh_sizes:
            for workers in [1, 4, 8]:  # Representative worker counts
                results = self.benchmark_point_processing(mesh_size, workers)
                all_results.extend(results)
        
        # Phase 2: Surface rendering quality
        self.logger.info("Phase 2: Surface rendering quality analysis")
        for mesh_size in [50_000, 250_000, 1_000_000]:  # Representative sizes
            for quality_level in self.quality_levels:
                results = self.benchmark_surface_rendering(mesh_size, quality_level)
                all_results.extend(results)
        
        # Phase 3: Parallel visualization modes
        self.logger.info("Phase 3: Parallel visualization modes")
        for mode in self.visualization_modes:
            for workers in [1, 2, 4, 8]:
                results = self.benchmark_parallel_visualization(500_000, mode, workers)
                all_results.extend(results)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Save raw results
        df.to_csv(self.output_dir / 'visualization_raw_results.csv', index=False)
        
        # Calculate statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_stats = df.groupby(['mesh_size', 'workers', 'quality_level', 'mode']).agg({
            col: ['mean', 'std'] for col in numeric_columns if col not in ['repetition']
        }).round(6)
        
        df_stats.to_csv(self.output_dir / 'visualization_statistics.csv')
        
        self.logger.info("Comprehensive visualization benchmark completed")
        return df
    
    def plot_point_processing_analysis(self, df: pd.DataFrame):
        """Create point processing performance analysis plots."""
        self.logger.info("Creating point processing visualizations")
        
        # Filter point processing data
        point_data = df.dropna(subset=['subsample_time']).copy()
        
        if point_data.empty:
            self.logger.warning("No point processing data available for plotting")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Point Processing Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Processing time vs mesh size
        processing_times = point_data.groupby(['mesh_size', 'workers'])['total_time'].mean().reset_index()
        
        for workers in [1, 4, 8]:
            subset = processing_times[processing_times['workers'] == workers]
            if not subset.empty:
                ax1.loglog(subset['mesh_size'], subset['total_time'], 'o-',
                          label=f'{workers} workers', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Mesh Size (points)')
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.set_title('Processing Time vs Mesh Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Throughput vs workers
        throughput_data = point_data.groupby(['workers', 'mesh_size'])['throughput_points_per_sec'].mean().reset_index()
        
        for mesh_size in [100_000, 500_000, 1_000_000]:
            subset = throughput_data[throughput_data['mesh_size'] == mesh_size]
            if not subset.empty:
                ax2.plot(subset['workers'], subset['throughput_points_per_sec'], 'o-',
                        label=f'{mesh_size:,} points', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Number of Workers')
        ax2.set_ylabel('Throughput (points/second)')
        ax2.set_title('Throughput vs Worker Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Memory usage vs mesh size
        memory_data = point_data.groupby('mesh_size')['memory_usage_mb'].mean()
        
        ax3.loglog(memory_data.index, memory_data.values, 'o-',
                  color='darkred', linewidth=2, markersize=8)
        ax3.set_xlabel('Mesh Size (points)')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Usage vs Mesh Size')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Processing efficiency
        efficiency_data = point_data.groupby(['mesh_size', 'workers'])['processing_efficiency'].mean().reset_index()
        
        pivot_efficiency = efficiency_data.pivot(index='mesh_size', columns='workers', values='processing_efficiency')
        
        im = ax4.imshow(pivot_efficiency.values, aspect='auto', cmap='viridis', origin='lower')
        ax4.set_xticks(range(len(pivot_efficiency.columns)))
        ax4.set_xticklabels(pivot_efficiency.columns)
        ax4.set_yticks(range(len(pivot_efficiency.index)))
        ax4.set_yticklabels([f'{int(x):,}' for x in pivot_efficiency.index])
        ax4.set_xlabel('Number of Workers')
        ax4.set_ylabel('Mesh Size (points)')
        ax4.set_title('Processing Efficiency Heatmap')
        
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Processing Efficiency (points/MB)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'point_processing_analysis.png')
        plt.savefig(self.output_dir / 'point_processing_analysis.pdf')
        plt.close()
    
    def plot_rendering_quality_analysis(self, df: pd.DataFrame):
        """Create rendering quality analysis plots."""
        self.logger.info("Creating rendering quality visualizations")
        
        # Filter rendering data
        render_data = df.dropna(subset=['viz_creation_time']).copy()
        
        if render_data.empty:
            self.logger.warning("No rendering data available for plotting")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Rendering Quality Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Rendering time vs quality level
        quality_times = render_data.groupby(['quality_level', 'mesh_size'])['total_time'].mean().reset_index()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.quality_levels)))
        for idx, quality in enumerate(self.quality_levels):
            subset = quality_times[quality_times['quality_level'] == quality]
            if not subset.empty:
                ax1.semilogx(subset['mesh_size'], subset['total_time'], 'o-',
                           color=colors[idx], label=quality.capitalize(),
                           linewidth=2, markersize=6)
        
        ax1.set_xlabel('Mesh Size (points)')
        ax1.set_ylabel('Rendering Time (seconds)')
        ax1.set_title('Rendering Time vs Quality Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory usage vs quality
        memory_quality = render_data.groupby(['quality_level', 'mesh_size'])['memory_usage_mb'].mean().reset_index()
        
        for idx, quality in enumerate(self.quality_levels):
            subset = memory_quality[memory_quality['quality_level'] == quality]
            if not subset.empty:
                ax2.semilogx(subset['mesh_size'], subset['memory_usage_mb'], 'o-',
                           color=colors[idx], label=quality.capitalize(),
                           linewidth=2, markersize=6)
        
        ax2.set_xlabel('Mesh Size (points)')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Quality Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Rendering throughput by quality
        throughput_quality = render_data.groupby('quality_level').agg({
            'rendering_throughput': ['mean', 'std'],
            'triangles_rendered': 'mean'
        }).round(2)
        
        qualities = throughput_quality.index
        throughputs = throughput_quality['rendering_throughput']['mean']
        throughput_stds = throughput_quality['rendering_throughput']['std']
        
        bars = ax3.bar(qualities, throughputs, yerr=throughput_stds, capsize=5,
                      color=colors[:len(qualities)], alpha=0.8)
        ax3.set_xlabel('Quality Level')
        ax3.set_ylabel('Rendering Throughput (triangles/second)')
        ax3.set_title('Rendering Throughput by Quality')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, throughput, std in zip(bars, throughputs, throughput_stds):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{throughput:.0f}', ha='center', va='bottom')
        
        # Plot 4: Memory efficiency vs quality
        efficiency_quality = render_data.groupby('quality_level')['memory_efficiency'].mean()
        
        ax4.bar(efficiency_quality.index, efficiency_quality.values,
               color=colors[:len(efficiency_quality)], alpha=0.8)
        ax4.set_xlabel('Quality Level')
        ax4.set_ylabel('Memory Efficiency (points/MB)')
        ax4.set_title('Memory Efficiency by Quality Level')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'rendering_quality_analysis.png')
        plt.savefig(self.output_dir / 'rendering_quality_analysis.pdf')
        plt.close()
    
    def plot_visualization_modes_comparison(self, df: pd.DataFrame):
        """Create visualization modes comparison plots."""
        self.logger.info("Creating visualization modes comparison")
        
        # Filter parallel visualization data
        parallel_data = df.dropna(subset=['processing_time']).copy()
        
        if parallel_data.empty:
            self.logger.warning("No parallel visualization data available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Processing time by mode and workers
        mode_times = parallel_data.groupby(['mode', 'workers'])['processing_time'].mean().reset_index()
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.visualization_modes)))
        for idx, mode in enumerate(self.visualization_modes):
            subset = mode_times[mode_times['mode'] == mode]
            if not subset.empty:
                ax1.plot(subset['workers'], subset['processing_time'], 'o-',
                        color=colors[idx], label=mode.capitalize(),
                        linewidth=2, markersize=6)
        
        ax1.set_xlabel('Number of Workers')
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.set_title('Processing Time by Visualization Mode')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Throughput comparison
        throughput_modes = parallel_data.groupby('mode').agg({
            'throughput_elements_per_sec': ['mean', 'std'],
            'elements_processed': 'mean'
        }).round(2)
        
        modes = throughput_modes.index
        throughputs = throughput_modes['throughput_elements_per_sec']['mean']
        throughput_stds = throughput_modes['throughput_elements_per_sec']['std']
        
        bars = ax2.bar(modes, throughputs, yerr=throughput_stds, capsize=5,
                      color=colors[:len(modes)], alpha=0.8)
        ax2.set_xlabel('Visualization Mode')
        ax2.set_ylabel('Throughput (elements/second)')
        ax2.set_title('Throughput by Visualization Mode')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Rotate x-axis labels for better readability
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualization_modes_comparison.png')
        plt.savefig(self.output_dir / 'visualization_modes_comparison.pdf')
        plt.close()
    
    def generate_visualization_report(self, df: pd.DataFrame):
        """Generate comprehensive visualization performance report."""
        self.logger.info("Generating visualization performance report")
        
        report = []
        report.append("# OpenFFD Visualization Performance Analysis Report\n")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("## Executive Summary\n")
        
        # Performance highlights
        if not df.empty:
            # Best point processing performance
            point_data = df.dropna(subset=['throughput_points_per_sec'])
            if not point_data.empty:
                best_point_perf = point_data.loc[point_data['throughput_points_per_sec'].idxmax()]
                report.append(f"### Point Processing Performance:\n")
                report.append(f"- Best throughput: {best_point_perf['throughput_points_per_sec']:.1f} points/second\n")
                report.append(f"- Configuration: {best_point_perf['mesh_size']:,} points, {best_point_perf['workers']} workers\n")
            
            # Best rendering performance
            render_data = df.dropna(subset=['rendering_throughput'])
            if not render_data.empty:
                best_render_perf = render_data.loc[render_data['rendering_throughput'].idxmax()]
                report.append(f"\n### Rendering Performance:\n")
                report.append(f"- Best rendering throughput: {best_render_perf['rendering_throughput']:.1f} triangles/second\n")
                report.append(f"- Quality level: {best_render_perf['quality_level']}\n")
            
            # Memory efficiency
            memory_data = df.dropna(subset=['memory_efficiency'])
            if not memory_data.empty:
                best_memory_eff = memory_data.loc[memory_data['memory_efficiency'].idxmax()]
                report.append(f"\n### Memory Efficiency:\n")
                report.append(f"- Best efficiency: {best_memory_eff['memory_efficiency']:.1f} points/MB\n")
        
        # Recommendations
        report.append("\n## Performance Recommendations\n")
        
        report.append("### For Interactive Visualization:\n")
        report.append("- Use 'low' or 'medium' quality settings for real-time interaction\n")
        report.append("- Enable parallel processing for meshes > 100,000 points\n")
        report.append("- Limit triangle count to < 50,000 for smooth interaction\n")
        
        report.append("### For High-Quality Rendering:\n")
        report.append("- Use 'high' or 'ultra' quality for publication figures\n")
        report.append("- Monitor memory usage for large meshes\n")
        report.append("- Consider mesh chunking for very large datasets\n")
        
        report.append("### For Large-Scale Visualization:\n")
        report.append("- Use point cloud mode for exploratory analysis\n")
        report.append("- Implement level-of-detail (LOD) strategies\n")
        report.append("- Use parallel processing with 4-8 workers for optimal performance\n")
        
        # Technical insights
        report.append("\n## Technical Insights\n")
        
        if not df.empty:
            # Processing time analysis
            if 'total_time' in df.columns:
                avg_times = df.groupby('mesh_size')['total_time'].mean()
                report.append("### Average Processing Times by Mesh Size:\n")
                for size, time_val in avg_times.items():
                    if not np.isnan(time_val) and time_val != float('inf'):
                        report.append(f"- {size:,} points: {time_val:.3f} seconds\n")
            
            # Memory usage analysis
            if 'memory_usage_mb' in df.columns:
                avg_memory = df.groupby('mesh_size')['memory_usage_mb'].mean()
                report.append("\n### Average Memory Usage by Mesh Size:\n")
                for size, memory_val in avg_memory.items():
                    if not np.isnan(memory_val):
                        report.append(f"- {size:,} points: {memory_val:.1f} MB\n")
        
        # System requirements
        report.append("\n## System Requirements\n")
        report.append("### Minimum Requirements:\n")
        report.append("- 4 GB RAM for meshes up to 500K points\n")
        report.append("- 2 CPU cores for basic visualization\n")
        report.append("- OpenGL 3.3 support for hardware acceleration\n")
        
        report.append("### Recommended Requirements:\n")
        report.append("- 16 GB RAM for meshes up to 2M points\n")
        report.append("- 8 CPU cores for parallel processing\n")
        report.append("- Dedicated GPU for high-quality rendering\n")
        
        # Save report
        with open(self.output_dir / 'visualization_performance_report.md', 'w') as f:
            f.writelines(report)
        
        self.logger.info(f"Visualization report saved to {self.output_dir}/visualization_performance_report.md")
    
    def run_full_benchmark(self):
        """Run the complete visualization performance benchmark suite."""
        self.logger.info("Starting full visualization performance benchmark")
        
        # Check availability of required modules
        if not VIZ_AVAILABLE:
            self.logger.warning("Visualization modules not fully available. Some benchmarks may be limited.")
        
        # Run comprehensive benchmark
        df = self.run_comprehensive_benchmark()
        
        # Generate visualizations
        self.plot_point_processing_analysis(df)
        self.plot_rendering_quality_analysis(df)
        self.plot_visualization_modes_comparison(df)
        
        # Generate report
        self.generate_visualization_report(df)
        
        self.logger.info("Full visualization benchmark completed successfully!")
        self.logger.info(f"Results available in: {self.output_dir}")


def main():
    """Main entry point for visualization benchmark."""
    parser = argparse.ArgumentParser(
        description='Comprehensive visualization performance benchmark for OpenFFD'
    )
    parser.add_argument(
        '--output-dir', default='visualization_benchmark_results',
        help='Output directory for benchmark results'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick benchmark with reduced test cases'
    )
    parser.add_argument(
        '--max-mesh-size', type=int, default=2_000_000,
        help='Maximum mesh size to test'
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = VisualizationBenchmark(args.output_dir)
    
    # Adjust parameters for quick run
    if args.quick:
        benchmark.mesh_sizes = [50_000, 250_000, 1_000_000]
        benchmark.worker_counts = [1, 4, 8]
        benchmark.visualization_modes = ['points', 'surface']
        benchmark.quality_levels = ['low', 'medium', 'high']
        benchmark.repetitions = 2
    
    # Limit mesh sizes based on argument
    benchmark.mesh_sizes = [s for s in benchmark.mesh_sizes if s <= args.max_mesh_size]
    
    # Run benchmark
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()