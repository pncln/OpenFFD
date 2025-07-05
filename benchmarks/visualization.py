"""
Academic-quality visualization system for FFD and HFFD benchmark results.

This module creates publication-ready figures suitable for research papers,
with proper formatting, color schemes, and statistical analysis.
Each figure is created separately without titles for maximum flexibility.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from scipy import stats
from scipy.optimize import curve_fit
import warnings

logger = logging.getLogger(__name__)

# Suppress scipy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

class AcademicPlotter:
    """Create publication-ready plots for FFD/HFFD benchmarks."""
    
    def __init__(self, config, style: str = "seaborn-v0_8-paper"):
        """Initialize the academic plotter.
        
        Args:
            config: BenchmarkConfig instance
            style: Matplotlib style to use
        """
        self.config = config
        
        # Set up publication-quality plotting
        plt.style.use('default')  # Reset to default first
        
        # Configure matplotlib for publication quality
        self.setup_publication_style()
        
        # Academic color palette (colorblind-friendly)
        self.colors = {
            'primary': '#1f77b4',    # Blue
            'secondary': '#ff7f0e',  # Orange  
            'tertiary': '#2ca02c',   # Green
            'quaternary': '#d62728', # Red
            'quinary': '#9467bd',    # Purple
            'senary': '#8c564b',     # Brown
            'septenary': '#e377c2',  # Pink
            'octonary': '#7f7f7f',   # Gray
            'serial': '#2ca02c',     # Green for serial
            'parallel': '#d62728',   # Red for parallel
            'ffd': '#1f77b4',        # Blue for FFD
            'hffd': '#ff7f0e'        # Orange for HFFD
        }
        
        # Define geometry type colors and markers
        self.geometry_styles = {
            'random': {'color': self.colors['primary'], 'marker': 'o', 'label': 'Random Points'},
            'sphere': {'color': self.colors['secondary'], 'marker': 's', 'label': 'Sphere Surface'},
            'cylinder': {'color': self.colors['tertiary'], 'marker': '^', 'label': 'Cylinder Surface'},
            'wing': {'color': self.colors['quaternary'], 'marker': 'D', 'label': 'Wing Profile'}
        }
    
    def setup_publication_style(self):
        """Configure matplotlib for publication-quality figures."""
        plt.rcParams.update({
            # Figure and layout
            'figure.figsize': (8, 6),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            
            # Fonts
            'font.size': 22,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
            'mathtext.fontset': 'stix',
            
            # Axes
            'axes.titlesize': 36,
            'axes.labelsize': 30,
            'axes.linewidth': 1.0,
            'axes.grid': True,
            'axes.axisbelow': True,
            'axes.edgecolor': 'black',
            'axes.facecolor': 'white',
            
            # Grid
            'grid.color': 'gray',
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.7,
            
            # Ticks
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 10,
            'ytick.major.size': 10,
            'xtick.minor.size': 5,
            'ytick.minor.size': 5,
            
            # Legend
            'legend.fontsize': 16,
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': False,
            'legend.framealpha': 0.8,
            'legend.edgecolor': 'black',
            
            # Lines and markers
            'lines.linewidth': 5.0,
            'lines.markersize': 12,
            'lines.markeredgewidth': 0.5,
            
            # Error bars
            'errorbar.capsize': 3,
            
            # LaTeX
            'text.usetex': False,  # Set to True if LaTeX is available
        })
    
    def format_scientific_notation(self, x, pos):
        """Format numbers in scientific notation for axes."""
        if x == 0:
            return '0'
        elif x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.0f}K'
        else:
            return f'{x:.0f}'
    
    def power_law_fit(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """Fit power law y = a * x^b and return parameters with R²."""
        try:
            # Remove zeros and negative values
            mask = (x > 0) & (y > 0)
            x_clean, y_clean = x[mask], y[mask]
            
            if len(x_clean) < 3:
                return 0, 0, 0
            
            # Fit in log space: log(y) = log(a) + b*log(x)
            log_x, log_y = np.log(x_clean), np.log(y_clean)
            slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)
            
            a = np.exp(intercept)
            b = slope
            r_squared = r_value ** 2
            
            return a, b, r_squared
        except Exception:
            return 0, 0, 0
    
    def plot_ffd_mesh_scaling(self, ffd_df: pd.DataFrame) -> plt.Figure:
        """Plot FFD execution time vs mesh size."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Filter successful runs only
        ffd_success = ffd_df[ffd_df['success'] == True]
        
        # Plot FFD scaling by geometry type
        for geometry_type in ffd_success['geometry_type'].unique():
            if geometry_type not in self.geometry_styles:
                continue
                
            geom_data = ffd_success[ffd_success['geometry_type'] == geometry_type]
            if len(geom_data) == 0:
                continue
            
            # Group by mesh size and calculate statistics
            grouped = geom_data.groupby('mesh_size')['execution_time']
            mesh_sizes = list(grouped.groups.keys())
            means = grouped.mean()
            stds = grouped.std().fillna(0)
            
            style = self.geometry_styles[geometry_type]
            ax.errorbar(mesh_sizes, means, yerr=stds, 
                       label=style['label'], color=style['color'], 
                       marker=style['marker'], capsize=3, capthick=1)
        
        # Fit and plot power law trend
        if len(ffd_success) > 0:
            ffd_grouped = ffd_success.groupby('mesh_size')['execution_time'].mean()
            x_ffd = np.array(list(ffd_grouped.index))
            y_ffd = ffd_grouped.values
            a, b, r2 = self.power_law_fit(x_ffd, y_ffd)
            if r2 > 0.5:
                x_fit = np.logspace(np.log10(x_ffd.min()), np.log10(x_ffd.max()), 100)
                y_fit = a * x_fit ** b
                ax.plot(x_fit, y_fit, '--', color='gray', alpha=0.7, 
                       label=f'Power Law: $t \\propto n^{{{b:.2f}}}$ ($R^2={r2:.3f}$)')
        
        # Format axes (no title)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Mesh Size (Number of Points)')
        ax.set_ylabel('Execution Time (seconds)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(self.format_scientific_notation))
        
        plt.tight_layout()
        return fig
    
    def plot_hffd_mesh_scaling(self, hffd_df: pd.DataFrame) -> plt.Figure:
        """Plot HFFD execution time vs mesh size."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Filter successful runs only
        hffd_success = hffd_df[hffd_df['success'] == True]
        
        # Plot HFFD scaling by geometry type
        for geometry_type in hffd_success['geometry_type'].unique():
            if geometry_type not in self.geometry_styles:
                continue
                
            geom_data = hffd_success[hffd_success['geometry_type'] == geometry_type]
            if len(geom_data) == 0:
                continue
            
            # Group by mesh size and calculate statistics
            grouped = geom_data.groupby('mesh_size')['execution_time']
            mesh_sizes = list(grouped.groups.keys())
            means = grouped.mean()
            stds = grouped.std().fillna(0)
            
            style = self.geometry_styles[geometry_type]
            ax.errorbar(mesh_sizes, means, yerr=stds, 
                       label=style['label'], color=style['color'], 
                       marker=style['marker'], capsize=3, capthick=1)
        
        # Fit and plot power law trend
        if len(hffd_success) > 0:
            hffd_grouped = hffd_success.groupby('mesh_size')['execution_time'].mean()
            x_hffd = np.array(list(hffd_grouped.index))
            y_hffd = hffd_grouped.values
            a, b, r2 = self.power_law_fit(x_hffd, y_hffd)
            if r2 > 0.5:
                x_fit = np.logspace(np.log10(x_hffd.min()), np.log10(x_hffd.max()), 100)
                y_fit = a * x_fit ** b
                ax.plot(x_fit, y_fit, '--', color='gray', alpha=0.7,
                       label=f'Power Law: $t \\propto n^{{{b:.2f}}}$ ($R^2={r2:.3f}$)')
        
        # Format axes (no title)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Mesh Size (Number of Points)')
        ax.set_ylabel('Execution Time (seconds)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(self.format_scientific_notation))
        
        plt.tight_layout()
        return fig
    
    def plot_ffd_control_complexity(self, ffd_df: pd.DataFrame) -> plt.Figure:
        """Plot FFD execution time vs control complexity."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Filter successful runs
        ffd_success = ffd_df[ffd_df['success'] == True]
        
        # FFD: Plot by control dimensions
        if len(ffd_success) > 0 and 'control_dim' in ffd_success.columns:
            # Calculate total control points for each dimension
            ffd_success = ffd_success.copy()
            ffd_success['total_control_points'] = ffd_success['control_dim'].apply(
                lambda x: eval(x)[0] * eval(x)[1] * eval(x)[2] if isinstance(x, str) else x[0] * x[1] * x[2]
            )
            
            grouped = ffd_success.groupby('total_control_points')['execution_time']
            control_points = list(grouped.groups.keys())
            means = grouped.mean()
            stds = grouped.std().fillna(0)
            
            ax.errorbar(control_points, means, yerr=stds, 
                       color=self.colors['ffd'], marker='o', 
                       capsize=3, capthick=1, label='FFD Control Box')
        
        # Format axes (no title)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Total Control Points')
        ax.set_ylabel('Execution Time (seconds)')
        
        # Only show legend if there's data plotted
        if len(ffd_success) > 0 and 'control_dim' in ffd_success.columns:
            ax.legend()
        else:
            # Add a text annotation if no data is available
            ax.text(0.5, 0.5, 'No FFD control complexity data available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, alpha=0.7)
        
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(self.format_scientific_notation))
        
        plt.tight_layout()
        return fig
    
    def plot_hffd_hierarchy_complexity(self, hffd_df: pd.DataFrame) -> plt.Figure:
        """Plot HFFD execution time vs hierarchy complexity."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Filter successful runs
        hffd_success = hffd_df[hffd_df['success'] == True]
        
        # HFFD: Plot by hierarchy complexity
        if len(hffd_success) > 0 and 'hierarchy_complexity' in hffd_success.columns:
            # Use total control points as x-axis
            if 'total_control_points' in hffd_success.columns:
                grouped = hffd_success.groupby('total_control_points')['execution_time']
                control_points = list(grouped.groups.keys())
                means = grouped.mean()
                stds = grouped.std().fillna(0)
                
                ax.errorbar(control_points, means, yerr=stds, 
                           color=self.colors['hffd'], marker='s', 
                           capsize=3, capthick=1, label='HFFD Hierarchy')
        
        # Format axes (no title)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Total Control Points')
        ax.set_ylabel('Execution Time (seconds)')
        
        # Only show legend if there's data plotted
        has_data = (len(hffd_success) > 0 and 'hierarchy_complexity' in hffd_success.columns 
                   and 'total_control_points' in hffd_success.columns)
        if has_data:
            ax.legend()
        else:
            # Add a text annotation if no data is available
            ax.text(0.5, 0.5, 'No HFFD hierarchy complexity data available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, alpha=0.7)
        
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(self.format_scientific_notation))
        
        plt.tight_layout()
        return fig
    
    def plot_parallelization_speedup(self, ffd_df: pd.DataFrame, hffd_df: pd.DataFrame) -> plt.Figure:
        """Plot parallelization speedup."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Filter successful runs with parallelization data (handle empty DataFrames)
        # Include both serial (max_workers=NaN, parallel_enabled=False) and parallel runs
        ffd_parallel = pd.DataFrame()
        if len(ffd_df) > 0 and 'success' in ffd_df.columns and 'max_workers' in ffd_df.columns:
            ffd_parallel = ffd_df[ffd_df['success'] == True]
        
        hffd_parallel = pd.DataFrame()
        if len(hffd_df) > 0 and 'success' in hffd_df.columns and 'max_workers' in hffd_df.columns:
            hffd_parallel = hffd_df[hffd_df['success'] == True]
        
        # Calculate speedup (serial time / parallel time)
        def calculate_speedup(df):
            if len(df) == 0:
                return pd.DataFrame()
            
            # Get serial baseline (1 worker or parallel_enabled=False)
            serial_mask = (df['max_workers'] == 1) | (df['parallel_enabled'] == False)
            serial_data = df[serial_mask]
            
            if len(serial_data) == 0:
                return pd.DataFrame()
            
            serial_time = serial_data['execution_time'].mean()
            
            # Calculate speedup for each worker count
            speedup_data = []
            for workers in sorted(df['max_workers'].unique()):
                if pd.isna(workers):
                    continue
                    
                worker_data = df[df['max_workers'] == workers]
                if len(worker_data) == 0:
                    continue
                    
                parallel_time = worker_data['execution_time'].mean()
                speedup = serial_time / parallel_time if parallel_time > 0 else 0
                
                speedup_data.append({
                    'workers': workers,
                    'speedup': speedup
                })
            
            return pd.DataFrame(speedup_data)
        
        # Calculate speedup for FFD and HFFD
        ffd_speedup = calculate_speedup(ffd_parallel)
        hffd_speedup = calculate_speedup(hffd_parallel)
        
        # Plot speedup
        if len(ffd_speedup) > 0:
            ax.plot(ffd_speedup['workers'], ffd_speedup['speedup'], 
                   color=self.colors['ffd'], marker='o', label='FFD')
        
        if len(hffd_speedup) > 0:
            ax.plot(hffd_speedup['workers'], hffd_speedup['speedup'], 
                   color=self.colors['hffd'], marker='s', label='HFFD')
        
        # Plot ideal speedup line
        if len(ffd_speedup) > 0 or len(hffd_speedup) > 0:
            max_workers = max(
                ffd_speedup['workers'].max() if len(ffd_speedup) > 0 else 1,
                hffd_speedup['workers'].max() if len(hffd_speedup) > 0 else 1
            )
            ideal_workers = np.arange(1, max_workers + 1)
            ax.plot(ideal_workers, ideal_workers, '--', 
                   color='gray', alpha=0.7, label='Ideal Speedup')
        
        # Format axes (no title)
        ax.set_xlabel('Number of Workers')
        ax.set_ylabel('Speedup')
        
        # Only show legend if there are data series plotted
        if len(ffd_speedup) > 0 or len(hffd_speedup) > 0:
            ax.legend()
        else:
            # Add a text annotation if no data is available
            ax.text(0.5, 0.5, 'No parallelization data available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, alpha=0.7)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_parallelization_efficiency(self, ffd_df: pd.DataFrame, hffd_df: pd.DataFrame) -> plt.Figure:
        """Plot parallelization efficiency."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Filter successful runs with parallelization data (handle empty DataFrames)
        # Include both serial (max_workers=NaN, parallel_enabled=False) and parallel runs
        ffd_parallel = pd.DataFrame()
        if len(ffd_df) > 0 and 'success' in ffd_df.columns and 'max_workers' in ffd_df.columns:
            ffd_parallel = ffd_df[ffd_df['success'] == True]
        
        hffd_parallel = pd.DataFrame()
        if len(hffd_df) > 0 and 'success' in hffd_df.columns and 'max_workers' in hffd_df.columns:
            hffd_parallel = hffd_df[hffd_df['success'] == True]
        
        # Calculate efficiency (speedup / workers)
        def calculate_efficiency(df):
            if len(df) == 0:
                return pd.DataFrame()
            
            # Get serial baseline (1 worker or parallel_enabled=False)
            serial_mask = (df['max_workers'] == 1) | (df['parallel_enabled'] == False)
            serial_data = df[serial_mask]
            
            if len(serial_data) == 0:
                return pd.DataFrame()
            
            serial_time = serial_data['execution_time'].mean()
            
            # Calculate efficiency for each worker count
            efficiency_data = []
            for workers in sorted(df['max_workers'].unique()):
                if pd.isna(workers):
                    continue
                    
                worker_data = df[df['max_workers'] == workers]
                if len(worker_data) == 0:
                    continue
                    
                parallel_time = worker_data['execution_time'].mean()
                speedup = serial_time / parallel_time if parallel_time > 0 else 0
                efficiency = speedup / workers if workers > 0 else 0
                
                efficiency_data.append({
                    'workers': workers,
                    'efficiency': efficiency * 100  # Convert to percentage
                })
            
            return pd.DataFrame(efficiency_data)
        
        # Calculate efficiency for FFD and HFFD
        ffd_efficiency = calculate_efficiency(ffd_parallel)
        hffd_efficiency = calculate_efficiency(hffd_parallel)
        
        # Plot efficiency
        if len(ffd_efficiency) > 0:
            ax.plot(ffd_efficiency['workers'], ffd_efficiency['efficiency'], 
                   color=self.colors['ffd'], marker='o', label='FFD')
        
        if len(hffd_efficiency) > 0:
            ax.plot(hffd_efficiency['workers'], hffd_efficiency['efficiency'], 
                   color=self.colors['hffd'], marker='s', label='HFFD')
        
        # Format axes (no title)
        ax.set_xlabel('Number of Workers')
        ax.set_ylabel('Efficiency (%)')
        ax.set_ylim(0, 100)
        
        # Only show legend if there are data series plotted
        if len(ffd_efficiency) > 0 or len(hffd_efficiency) > 0:
            ax.legend()
        else:
            # Add a text annotation if no data is available
            ax.text(0.5, 0.5, 'No parallelization data available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, alpha=0.7)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_memory_usage(self, ffd_df: pd.DataFrame, hffd_df: pd.DataFrame) -> plt.Figure:
        """Plot memory usage vs mesh size."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Filter successful runs (handle empty DataFrames)
        ffd_success = pd.DataFrame()
        if len(ffd_df) > 0 and 'success' in ffd_df.columns:
            ffd_success = ffd_df[ffd_df['success'] == True]
        
        hffd_success = pd.DataFrame()
        if len(hffd_df) > 0 and 'success' in hffd_df.columns:
            hffd_success = hffd_df[hffd_df['success'] == True]
        
        # Plot memory vs mesh size
        if len(ffd_success) > 0:
            grouped = ffd_success.groupby('mesh_size')['memory_peak_mb']
            mesh_sizes = list(grouped.groups.keys())
            means = grouped.mean()
            stds = grouped.std().fillna(0)
            
            ax.errorbar(mesh_sizes, means, yerr=stds, 
                       color=self.colors['ffd'], marker='o', 
                       capsize=3, label='FFD')
        
        if len(hffd_success) > 0:
            grouped = hffd_success.groupby('mesh_size')['memory_peak_mb']
            mesh_sizes = list(grouped.groups.keys())
            means = grouped.mean()
            stds = grouped.std().fillna(0)
            
            ax.errorbar(mesh_sizes, means, yerr=stds, 
                       color=self.colors['hffd'], marker='s', 
                       capsize=3, label='HFFD')
        
        # Format axes (no title)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Mesh Size (Number of Points)')
        ax.set_ylabel('Peak Memory Usage (MB)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(self.format_scientific_notation))
        
        plt.tight_layout()
        return fig
    
    def plot_hffd_depth_scaling(self, hffd_df: pd.DataFrame) -> plt.Figure:
        """Plot HFFD performance vs hierarchy depth."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Filter successful runs
        hffd_success = hffd_df[hffd_df['success'] == True]
        
        if len(hffd_success) == 0:
            return fig
        
        # Execution time vs hierarchy depth
        if 'max_depth' in hffd_success.columns:
            grouped = hffd_success.groupby('max_depth')['execution_time']
            depths = list(grouped.groups.keys())
            means = grouped.mean()
            stds = grouped.std().fillna(0)
            
            ax.errorbar(depths, means, yerr=stds, 
                       color=self.colors['hffd'], marker='o', 
                       capsize=3)
        
        # Format axes (no title)
        ax.set_xlabel('Maximum Hierarchy Depth')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, 
                   formats: List[str] = None) -> List[Path]:
        """Save figure in multiple formats."""
        if formats is None:
            formats = ['pdf', 'png']
        
        saved_files = []
        
        for fmt in formats:
            filepath = self.config.figures_dir / f"{filename}.{fmt}"
            
            # Set format-specific DPI
            dpi = 300 if fmt in ['png', 'jpg'] else None
            
            fig.savefig(filepath, format=fmt, dpi=dpi, 
                       bbox_inches='tight', pad_inches=0.1)
            
            saved_files.append(filepath)
            logger.info(f"Saved figure: {filepath}")
        
        return saved_files

def create_publication_figures(ffd_results: pd.DataFrame, 
                             hffd_results: pd.DataFrame,
                             config) -> Dict[str, List[Path]]:
    """Create all publication-ready figures as separate plots without titles."""
    plotter = AcademicPlotter(config)
    saved_figures = {}
    
    logger.info("Creating publication-ready figures...")
    
    # 1. FFD mesh size scaling
    logger.info("Creating FFD mesh size scaling figure...")
    fig1 = plotter.plot_ffd_mesh_scaling(ffd_results)
    saved_figures['ffd_mesh_scaling'] = plotter.save_figure(
        fig1, 'ffd_mesh_scaling', ['pdf', 'png']
    )
    plt.close(fig1)
    
    # 2. HFFD mesh size scaling
    logger.info("Creating HFFD mesh size scaling figure...")
    fig2 = plotter.plot_hffd_mesh_scaling(hffd_results)
    saved_figures['hffd_mesh_scaling'] = plotter.save_figure(
        fig2, 'hffd_mesh_scaling', ['pdf', 'png']
    )
    plt.close(fig2)
    
    # 3. FFD control complexity
    logger.info("Creating FFD control complexity figure...")
    fig3 = plotter.plot_ffd_control_complexity(ffd_results)
    saved_figures['ffd_control_complexity'] = plotter.save_figure(
        fig3, 'ffd_control_complexity', ['pdf', 'png']
    )
    plt.close(fig3)
    
    # 4. HFFD hierarchy complexity
    logger.info("Creating HFFD hierarchy complexity figure...")
    fig4 = plotter.plot_hffd_hierarchy_complexity(hffd_results)
    saved_figures['hffd_hierarchy_complexity'] = plotter.save_figure(
        fig4, 'hffd_hierarchy_complexity', ['pdf', 'png']
    )
    plt.close(fig4)
    
    # 5. Parallelization speedup (if data available)
    # Check if we have actual parallelization data (more than 1 worker) 
    # Include successful runs only
    ffd_has_parallel = False
    if len(ffd_results) > 0 and 'max_workers' in ffd_results.columns and 'success' in ffd_results.columns:
        ffd_success = ffd_results[ffd_results['success'] == True]
        # Check for both serial runs (parallel_enabled=False) and parallel runs (max_workers > 1)
        has_serial = len(ffd_success[ffd_success['parallel_enabled'] == False]) > 0
        has_parallel = len(ffd_success[ffd_success['max_workers'] > 1]) > 0
        ffd_has_parallel = has_serial and has_parallel
    
    hffd_has_parallel = False
    if len(hffd_results) > 0 and 'max_workers' in hffd_results.columns and 'success' in hffd_results.columns:
        hffd_success = hffd_results[hffd_results['success'] == True]
        # Check for both serial runs (parallel_enabled=False) and parallel runs (max_workers > 1)
        has_serial = len(hffd_success[hffd_success['parallel_enabled'] == False]) > 0
        has_parallel = len(hffd_success[hffd_success['max_workers'] > 1]) > 0
        hffd_has_parallel = has_serial and has_parallel
    
    if ffd_has_parallel or hffd_has_parallel:
        logger.info("Creating parallelization speedup figure...")
        fig5 = plotter.plot_parallelization_speedup(ffd_results, hffd_results)
        saved_figures['parallelization_speedup'] = plotter.save_figure(
            fig5, 'parallelization_speedup', ['pdf', 'png']
        )
        plt.close(fig5)
        
        logger.info("Creating parallelization efficiency figure...")
        fig6 = plotter.plot_parallelization_efficiency(ffd_results, hffd_results)
        saved_figures['parallelization_efficiency'] = plotter.save_figure(
            fig6, 'parallelization_efficiency', ['pdf', 'png']
        )
        plt.close(fig6)
    else:
        logger.info("Skipping parallelization figures - no multi-worker data available")
    
    # 6. Memory usage
    logger.info("Creating memory usage figure...")
    fig7 = plotter.plot_memory_usage(ffd_results, hffd_results)
    saved_figures['memory_usage'] = plotter.save_figure(
        fig7, 'memory_usage', ['pdf', 'png']
    )
    plt.close(fig7)
    
    # 7. HFFD depth scaling
    if len(hffd_results) > 0:
        logger.info("Creating HFFD depth scaling figure...")
        fig8 = plotter.plot_hffd_depth_scaling(hffd_results)
        saved_figures['hffd_depth_scaling'] = plotter.save_figure(
            fig8, 'hffd_depth_scaling', ['pdf', 'png']
        )
        plt.close(fig8)
    
    logger.info(f"All figures saved to: {config.figures_dir}")
    return saved_figures