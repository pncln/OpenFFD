#!/usr/bin/env python3
"""
Academic Plotting Utilities for OpenFFD Benchmarks

This module provides publication-ready plotting utilities specifically designed
for academic papers and journals. All plots follow academic standards with
proper styling, typography, and formatting suitable for AIAA Journal submission.

Author: OpenFFD Development Team
Purpose: AIAA Journal Performance Analysis
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogFormatter, LogLocator, FuncFormatter, MaxNLocator
from matplotlib.patches import Rectangle, Circle, Polygon
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

# Suppress specific matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class AcademicPlotStyle:
    """Academic plotting style configuration for publication-quality figures."""
    
    def __init__(self):
        """Initialize academic plotting style."""
        self.setup_style()
        
    def setup_style(self):
        """Configure matplotlib for academic publication standards."""
        # Set the style to a clean base
        plt.style.use('default')
        
        # Academic style parameters optimized for publication quality
        plt.rcParams.update({
            # Font settings - Times Roman for academic publications, larger for readability
            'font.size': 14,  # Increased base font size
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
            'mathtext.fontset': 'cm',  # Computer Modern for math
            
            # Figure settings - optimized for side-by-side placement
            'figure.figsize': [8, 6],  # Slightly smaller for side-by-side layouts
            'figure.dpi': 300,
            'figure.facecolor': 'white',
            'figure.edgecolor': 'none',
            'figure.autolayout': False,
            'figure.subplot.left': 0.12,    # More margin for y-axis labels
            'figure.subplot.bottom': 0.12,  # More margin for x-axis labels
            'figure.subplot.right': 0.95,   # Less right margin
            'figure.subplot.top': 0.90,     # Less top margin
            
            # Axes settings - enhanced visibility
            'axes.linewidth': 1.5,  # Thicker axes lines
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.labelsize': 14,  # Larger axis labels
            'axes.labelweight': 'bold',
            'axes.labelpad': 8,     # More padding for labels
            'axes.titlesize': 16,   # Larger title
            'axes.titleweight': 'bold',
            'axes.titlepad': 15,    # More title padding
            'axes.grid': True,
            'axes.axisbelow': True,
            'axes.prop_cycle': plt.cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']),
            
            # Tick settings - enhanced readability
            'xtick.major.size': 6,   # Larger ticks
            'xtick.minor.size': 4,
            'xtick.major.width': 1.5,
            'xtick.minor.width': 1.0,
            'xtick.direction': 'in',
            'xtick.labelsize': 12,   # Larger tick labels
            'xtick.top': True,
            'xtick.major.pad': 6,    # More padding
            'ytick.major.size': 6,
            'ytick.minor.size': 4,
            'ytick.major.width': 1.5,
            'ytick.minor.width': 1.0,
            'ytick.direction': 'in',
            'ytick.labelsize': 12,
            'ytick.right': True,
            'ytick.major.pad': 6,
            
            # Legend settings - enhanced visibility
            'legend.fontsize': 12,   # Larger legend font
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.shadow': False,
            'legend.framealpha': 0.95,  # Slightly transparent
            'legend.facecolor': 'white',
            'legend.edgecolor': 'black',
            'legend.borderpad': 0.6,
            'legend.columnspacing': 2.2,
            'legend.handlelength': 2.8,  # Longer legend handles
            'legend.handletextpad': 0.8,
            'legend.markerscale': 1.2,   # Larger legend markers
            
            # Line settings - enhanced visibility
            'lines.linewidth': 2.5,  # Thicker lines
            'lines.markersize': 8,   # Larger markers
            'lines.markeredgewidth': 1.2,
            
            # Grid settings - subtle but visible
            'grid.alpha': 0.4,       # More visible grid
            'grid.linewidth': 0.8,
            'grid.linestyle': '-',
            
            # Savefig settings
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.15,  # Slightly more padding
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none',
            'savefig.format': 'pdf'
        })
        
        # Define academic color palette
        self.colors = {
            'primary': '#1f77b4',    # Blue
            'secondary': '#ff7f0e',  # Orange  
            'tertiary': '#2ca02c',   # Green
            'quaternary': '#d62728', # Red
            'quinary': '#9467bd',    # Purple
            'senary': '#8c564b',     # Brown
            'septenary': '#e377c2',  # Pink
            'octonary': '#7f7f7f',   # Gray
            'black': '#000000',
            'gray': '#666666',
            'light_gray': '#cccccc'
        }
        
        # Color sequences for multi-series plots
        self.color_sequences = {
            'qualitative': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
            'sequential_blue': ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef'],
            'sequential_red': ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272', '#fcbba1'],
            'diverging': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
        }


class AcademicPlotter:
    """Main class for creating academic publication-quality plots."""
    
    def __init__(self, style: Optional[AcademicPlotStyle] = None):
        """Initialize academic plotter with style configuration."""
        self.style = style if style is not None else AcademicPlotStyle()
        
    def create_performance_scaling_plot(
        self, 
        data: pd.DataFrame, 
        x_col: str, 
        y_col: str, 
        series_col: str,
        title: str,
        xlabel: str,
        ylabel: str,
        log_scale: Tuple[bool, bool] = (False, False),
        reference_lines: Optional[List[Dict]] = None,
        figsize: Tuple[float, float] = (8, 6),
        tight_layout: bool = True
    ) -> plt.Figure:
        """Create a performance scaling plot with multiple series.
        
        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis data
            y_col: Column name for y-axis data  
            series_col: Column name for different series
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            log_scale: Tuple of (x_log, y_log) boolean flags
            reference_lines: List of reference line specifications
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique series values
        series_values = sorted(data[series_col].unique())
        colors = self.style.color_sequences['qualitative'][:len(series_values)]
        
        # Plot each series with enhanced visibility
        for i, series_val in enumerate(series_values):
            series_data = data[data[series_col] == series_val].sort_values(x_col)
            
            ax.plot(series_data[x_col], series_data[y_col], 
                   'o-', color=colors[i % len(colors)], 
                   label=str(series_val), linewidth=3.0, markersize=8,
                   markerfacecolor='white', markeredgewidth=2.0,
                   markeredgecolor=colors[i % len(colors)])
        
        # Add reference lines if specified
        if reference_lines:
            for ref_line in reference_lines:
                x_ref = ref_line.get('x_data', ax.get_xlim())
                y_ref = ref_line.get('y_data')
                line_style = ref_line.get('style', '--')
                line_color = ref_line.get('color', 'black')
                line_alpha = ref_line.get('alpha', 0.8)  # More visible
                line_label = ref_line.get('label', '')
                
                ax.plot(x_ref, y_ref, line_style, color=line_color, 
                       alpha=line_alpha, label=line_label, linewidth=2.0)  # Thicker reference lines
        
        # Set logarithmic scales if requested
        if log_scale[0]:
            ax.set_xscale('log')
        if log_scale[1]:
            ax.set_yscale('log')
        
        # Enhanced formatting for publication quality
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=14)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=14)
        ax.set_title(title, pad=20, fontweight='bold', fontsize=16)
        
        # Optimize legend placement and style
        legend = ax.legend(loc='best', frameon=True, fancybox=False, shadow=False,
                          framealpha=0.95, edgecolor='black', fontsize=12)
        legend.get_frame().set_linewidth(1.2)
        
        # Enhanced grid
        ax.grid(True, alpha=0.4, linewidth=0.8)
        
        # Auto-adjust axis limits to avoid empty space
        optimize_axis_limits(ax, x_data=data[x_col], y_data=data[y_col], 
                           log_scale=log_scale, margin_factor=0.05, ensure_zero=True)
        
        # Enhanced tick formatting
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1.0, length=4)
        
        if tight_layout:
            plt.tight_layout(pad=1.5)
        return fig
    
    def create_speedup_efficiency_plot(
        self,
        data: pd.DataFrame,
        workers_col: str,
        speedup_col: str,
        efficiency_col: str,
        series_col: Optional[str] = None,
        title: str = "Parallel Efficiency Analysis",
        figsize: Tuple[float, float] = (12, 5)
    ) -> plt.Figure:
        """Create combined speedup and efficiency plots.
        
        Args:
            data: DataFrame containing the data
            workers_col: Column name for number of workers
            speedup_col: Column name for speedup values
            efficiency_col: Column name for efficiency values
            series_col: Optional column for different series
            title: Main title for the figure
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        colors = self.style.color_sequences['qualitative']
        
        if series_col:
            series_values = sorted(data[series_col].unique())
            
            for i, series_val in enumerate(series_values):
                series_data = data[data[series_col] == series_val].sort_values(workers_col)
                color = colors[i % len(colors)]
                
                # Speedup plot
                ax1.plot(series_data[workers_col], series_data[speedup_col],
                        'o-', color=color, label=str(series_val),
                        linewidth=2, markersize=6)
                
                # Efficiency plot  
                efficiency_pct = series_data[efficiency_col] * 100  # Convert to percentage
                ax2.plot(series_data[workers_col], efficiency_pct,
                        'o-', color=color, label=str(series_val),
                        linewidth=2, markersize=6)
        else:
            # Single series
            sorted_data = data.sort_values(workers_col)
            
            ax1.plot(sorted_data[workers_col], sorted_data[speedup_col],
                    'o-', color=colors[0], linewidth=2, markersize=6)
            
            efficiency_pct = sorted_data[efficiency_col] * 100
            ax2.plot(sorted_data[workers_col], efficiency_pct,
                    'o-', color=colors[0], linewidth=2, markersize=6)
        
        # Add ideal speedup line
        max_workers = data[workers_col].max()
        workers_range = np.arange(1, max_workers + 1)
        ax1.plot(workers_range, workers_range, 'k--', alpha=0.7, 
                label='Ideal Speedup', linewidth=1.5)
        
        # Add efficiency thresholds
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, 
                   label='80% Efficiency')
        ax2.axhline(y=50, color='orange', linestyle=':', alpha=0.7,
                   label='50% Efficiency')
        
        # Formatting
        ax1.set_xlabel('Number of Workers')
        ax1.set_ylabel('Speedup')
        ax1.set_title('Parallel Speedup')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Number of Workers')
        ax2.set_ylabel('Parallel Efficiency (%)')
        ax2.set_title('Parallel Efficiency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 105)
        
        plt.tight_layout()
        return fig
    
    def create_complexity_analysis_plot(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        fit_line: Optional[Dict] = None,
        title: str = "Computational Complexity Analysis",
        xlabel: str = "Problem Size",
        ylabel: str = "Execution Time (s)",
        figsize: Tuple[float, float] = (8, 6)
    ) -> plt.Figure:
        """Create computational complexity analysis plot with trend fitting.
        
        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis (problem size)
            y_col: Column name for y-axis (execution time)
            fit_line: Dictionary with fit parameters {'slope', 'intercept', 'r_squared', 'label'}
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot of data points
        ax.loglog(data[x_col], data[y_col], 'o', 
                 color=self.style.colors['primary'], alpha=0.7,
                 markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        
        # Add trend line if provided
        if fit_line:
            x_fit = np.logspace(np.log10(data[x_col].min()), 
                               np.log10(data[x_col].max()), 100)
            y_fit = np.exp(fit_line['intercept']) * x_fit ** fit_line['slope']
            
            label = fit_line.get('label', f"Slope = {fit_line['slope']:.2f}")
            if 'r_squared' in fit_line:
                label += f" (R² = {fit_line['r_squared']:.3f})"
            
            ax.loglog(x_fit, y_fit, '-', color=self.style.colors['secondary'],
                     linewidth=2, label=label)
        
        # Add reference complexity lines
        x_ref = np.array([data[x_col].min(), data[x_col].max()])
        y_base = data[y_col].min()
        
        # O(n) reference
        y_linear = y_base * (x_ref / x_ref[0])
        ax.loglog(x_ref, y_linear, '--', color='gray', alpha=0.5, 
                 linewidth=1, label='O(n)')
        
        # O(n²) reference  
        y_quadratic = y_base * (x_ref / x_ref[0])**2
        ax.loglog(x_ref, y_quadratic, ':', color='gray', alpha=0.5,
                 linewidth=1, label='O(n²)')
        
        # Formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    
    def create_multi_metric_comparison(
        self,
        data: pd.DataFrame,
        categories_col: str,
        metrics: List[str],
        metric_labels: Optional[List[str]] = None,
        title: str = "Multi-Metric Performance Comparison",
        figsize: Tuple[float, float] = (12, 8),
        normalize: bool = True
    ) -> plt.Figure:
        """Create a multi-metric comparison plot (radar chart or bar chart).
        
        Args:
            data: DataFrame containing the data
            categories_col: Column name for categories
            metrics: List of metric column names
            metric_labels: Optional list of metric display labels
            title: Plot title
            figsize: Figure size tuple
            normalize: Whether to normalize metrics to 0-1 range
            
        Returns:
            Matplotlib figure object
        """
        if metric_labels is None:
            metric_labels = metrics
            
        # Calculate mean values for each category and metric
        mean_data = data.groupby(categories_col)[metrics].mean()
        
        # Normalize if requested
        if normalize:
            mean_data = (mean_data - mean_data.min()) / (mean_data.max() - mean_data.min())
        
        # Create subplot for each category
        n_categories = len(mean_data.index)
        n_cols = min(3, n_categories)
        n_rows = (n_categories + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                                subplot_kw=dict(projection='polar'))
        if n_categories == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(-1)
        else:
            axes = axes.flatten()
        
        # Angle for each metric
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = self.style.color_sequences['qualitative']
        
        for i, (category, values) in enumerate(mean_data.iterrows()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Close the plot
            values_plot = values.tolist() + [values.iloc[0]]
            
            # Plot
            ax.plot(angles, values_plot, 'o-', linewidth=2, 
                   color=colors[i % len(colors)], markersize=6)
            ax.fill(angles, values_plot, alpha=0.25, 
                   color=colors[i % len(colors)])
            
            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1 if normalize else mean_data.values.max())
            ax.set_title(str(category), y=1.08)
            ax.grid(True)
        
        # Hide unused subplots
        for i in range(n_categories, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig
    
    def create_error_bar_comparison(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        error_col: str,
        series_col: Optional[str] = None,
        title: str = "Performance Comparison with Error Bars",
        xlabel: str = "Configuration",
        ylabel: str = "Performance Metric",
        figsize: Tuple[float, float] = (10, 6)
    ) -> plt.Figure:
        """Create error bar comparison plot.
        
        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis categories
            y_col: Column name for y-axis values
            error_col: Column name for error values
            series_col: Optional column for different series
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if series_col:
            series_values = sorted(data[series_col].unique())
            colors = self.style.color_sequences['qualitative'][:len(series_values)]
            
            x_positions = np.arange(len(data[x_col].unique()))
            width = 0.8 / len(series_values)
            
            for i, series_val in enumerate(series_values):
                series_data = data[data[series_col] == series_val]
                positions = x_positions + (i - len(series_values)/2 + 0.5) * width
                
                ax.bar(positions, series_data[y_col], width, 
                      yerr=series_data[error_col], capsize=5,
                      label=str(series_val), color=colors[i],
                      alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels(data[x_col].unique())
            ax.legend()
            
        else:
            x_categories = data[x_col].unique()
            colors = self.style.color_sequences['sequential_blue'][:len(x_categories)]
            
            bars = ax.bar(x_categories, data[y_col], yerr=data[error_col],
                         capsize=5, color=colors, alpha=0.8,
                         edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, value, error in zip(bars, data[y_col], data[error_col]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + error,
                       f'{value:.2f}', ha='center', va='bottom')
        
        # Formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def save_figure(
        self, 
        fig: plt.Figure, 
        filename: str, 
        output_dir: str = ".",
        formats: List[str] = ['png', 'pdf'],
        dpi: int = 300,
        remove_title: bool = True,
        optimize_for_print: bool = True
    ) -> List[str]:
        """Save figure in multiple formats for academic publication.
        
        Args:
            fig: Matplotlib figure object
            filename: Base filename (without extension)
            output_dir: Output directory path
            formats: List of output formats
            dpi: Resolution for raster formats
            remove_title: Whether to remove titles from saved figures
            
        Returns:
            List of saved file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        # Store original titles if we need to remove them
        original_titles = {}
        original_suptitle = None
        if remove_title:
            for ax in fig.get_axes():
                original_titles[ax] = ax.get_title()
                ax.set_title('')
            # Also remove figure suptitle
            if hasattr(fig, '_suptitle') and fig._suptitle:
                original_suptitle = fig._suptitle.get_text()
                fig.suptitle('')
        
        for fmt in formats:
            filepath = os.path.join(output_dir, f"{filename}.{fmt}")
            
            # Optimize save parameters for publication quality
            save_kwargs = {
                'format': fmt,
                'dpi': dpi,
                'bbox_inches': 'tight',
                'facecolor': 'white',
                'edgecolor': 'none',
                'pad_inches': 0.15,  # Consistent padding
                'transparent': False
            }
            
            # Additional optimization for print quality
            if optimize_for_print:
                if fmt.lower() == 'pdf':
                    save_kwargs['metadata'] = {'Creator': 'OpenFFD Benchmark Suite'}
                elif fmt.lower() in ['png', 'jpg', 'jpeg']:
                    save_kwargs['dpi'] = max(300, dpi)  # Ensure minimum 300 DPI for raster
            
            fig.savefig(filepath, **save_kwargs)
            saved_files.append(filepath)
        
        # Restore original titles if they were removed
        if remove_title:
            for ax, title in original_titles.items():
                ax.set_title(title)
            if original_suptitle:
                fig.suptitle(original_suptitle)
        
        return saved_files


def create_benchmark_summary_figure(
    benchmark_results: Dict[str, pd.DataFrame],
    output_dir: str = ".",
    figsize: Tuple[float, float] = (16, 12)
) -> plt.Figure:
    """Create a comprehensive benchmark summary figure.
    
    Args:
        benchmark_results: Dictionary mapping benchmark names to DataFrames
        output_dir: Output directory for saving
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    plotter = AcademicPlotter()
    
    # Create 2x2 subplot layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Extract key metrics from each benchmark
    summary_data = {}
    
    for bench_name, df in benchmark_results.items():
        if 'execution_time' in df.columns or 'total_time' in df.columns:
            time_col = 'execution_time' if 'execution_time' in df.columns else 'total_time'
            summary_data[bench_name] = {
                'mean_time': df[time_col].mean(),
                'std_time': df[time_col].std(),
                'min_time': df[time_col].min(),
                'max_time': df[time_col].max()
            }
    
    # Plot 1: Execution time comparison
    ax1 = fig.add_subplot(gs[0, 0])
    if summary_data:
        benchmarks = list(summary_data.keys())
        mean_times = [summary_data[b]['mean_time'] for b in benchmarks]
        std_times = [summary_data[b]['std_time'] for b in benchmarks]
        
        colors = plotter.style.color_sequences['qualitative'][:len(benchmarks)]
        bars = ax1.bar(benchmarks, mean_times, yerr=std_times, capsize=5,
                      color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Benchmark Execution Times')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Performance scaling (if parallel data available)
    ax2 = fig.add_subplot(gs[0, 1])
    parallel_data = None
    for df in benchmark_results.values():
        if 'workers' in df.columns and 'speedup' in df.columns:
            parallel_data = df
            break
    
    if parallel_data is not None:
        speedup_data = parallel_data.groupby('workers')['speedup'].mean()
        ax2.plot(speedup_data.index, speedup_data.values, 'o-', 
                color=plotter.style.colors['primary'], linewidth=2, markersize=6)
        ax2.plot(speedup_data.index, speedup_data.index, 'k--', alpha=0.7, 
                label='Ideal Speedup')
        ax2.set_xlabel('Number of Workers')
        ax2.set_ylabel('Speedup')
        ax2.set_title('Parallel Scalability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Memory usage analysis
    ax3 = fig.add_subplot(gs[1, 0])
    memory_data = None
    for df in benchmark_results.values():
        if 'memory_usage_mb' in df.columns and 'mesh_size' in df.columns:
            memory_data = df
            break
    
    if memory_data is not None:
        mem_analysis = memory_data.groupby('mesh_size')['memory_usage_mb'].mean()
        ax3.loglog(mem_analysis.index, mem_analysis.values, 'o-',
                  color=plotter.style.colors['tertiary'], linewidth=2, markersize=6)
        ax3.set_xlabel('Mesh Size (points)')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Scaling')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Overall performance summary
    ax4 = fig.add_subplot(gs[1, 1])
    if summary_data:
        # Normalize performance metrics for comparison
        max_time = max(summary_data[b]['mean_time'] for b in benchmarks)
        normalized_perf = [1 - (summary_data[b]['mean_time'] / max_time) for b in benchmarks]
        
        ax4.barh(benchmarks, normalized_perf, color=colors, alpha=0.8,
                edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Relative Performance (higher is better)')
        ax4.set_title('Performance Summary')
        ax4.grid(True, alpha=0.3, axis='x')
    
    # Add overall title
    fig.suptitle('OpenFFD Benchmark Summary', fontsize=16, fontweight='bold')
    
    # Save the figure
    saved_files = plotter.save_figure(fig, 'benchmark_summary', output_dir)
    
    return fig


# Utility functions for common academic plotting tasks

def format_scientific_notation(x, pos):
    """Custom formatter for scientific notation in plots."""
    if x == 0:
        return '0'
    else:
        exponent = int(np.floor(np.log10(abs(x))))
        mantissa = x / (10**exponent)
        if exponent == 0:
            return f'{mantissa:.1f}'
        else:
            return f'{mantissa:.1f}×10$^{{{exponent}}}$'


def add_performance_annotations(ax, x_data, y_data, annotations):
    """Add performance annotations to a plot.
    
    Args:
        ax: Matplotlib axes object
        x_data: X-coordinate data
        y_data: Y-coordinate data  
        annotations: List of annotation strings
    """
    for i, (x, y, text) in enumerate(zip(x_data, y_data, annotations)):
        ax.annotate(text, (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def create_publication_colormap(n_colors: int = 10, style: str = 'qualitative'):
    """Create a publication-appropriate colormap.
    
    Args:
        n_colors: Number of distinct colors needed
        style: Color style ('qualitative', 'sequential', 'diverging')
        
    Returns:
        List of color codes
    """
    if style == 'qualitative':
        base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    elif style == 'sequential':
        base_colors = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef',
                      '#deebf7', '#f7fbff']
    elif style == 'diverging':
        base_colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598',
                      '#abdda4', '#66c2a5', '#3288bd']
    else:
        base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Extend or subsample as needed
    if n_colors <= len(base_colors):
        return base_colors[:n_colors]
    else:
        # Interpolate to create more colors
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        cmap = mcolors.ListedColormap(base_colors)
        return [cmap(i / (n_colors - 1)) for i in range(n_colors)]


def optimize_axis_limits(ax, x_data=None, y_data=None, log_scale=(False, False), 
                        margin_factor=0.05, ensure_zero=True):
    """Optimize axis limits to minimize empty space while maintaining readability.
    
    Args:
        ax: Matplotlib axes object
        x_data: X-axis data for limit calculation
        y_data: Y-axis data for limit calculation
        log_scale: Tuple of (x_log, y_log) boolean flags
        margin_factor: Fraction of data range to add as margin
        ensure_zero: Whether to ensure zero is included in linear scales
    """
    if x_data is not None:
        x_min, x_max = np.min(x_data), np.max(x_data)
        
        if log_scale[0]:
            # For log scale, use multiplicative margins
            log_range = np.log10(x_max) - np.log10(x_min)
            log_margin = log_range * margin_factor
            x_lower = 10**(np.log10(x_min) - log_margin)
            x_upper = 10**(np.log10(x_max) + log_margin)
        else:
            # Linear scale logic
            if x_min == x_max:
                x_margin = abs(x_min) * 0.1 if x_min != 0 else 1
            else:
                x_margin = (x_max - x_min) * margin_factor
            
            x_lower = x_min - x_margin
            x_upper = x_max + x_margin
            
            if ensure_zero and x_min > 0 and x_lower > 0:
                x_lower = max(0, x_lower)
                
        ax.set_xlim(x_lower, x_upper)
    
    if y_data is not None:
        y_min, y_max = np.min(y_data), np.max(y_data)
        
        if log_scale[1]:
            # For log scale, use multiplicative margins
            log_range = np.log10(y_max) - np.log10(y_min)
            log_margin = log_range * margin_factor
            y_lower = 10**(np.log10(y_min) - log_margin)
            y_upper = 10**(np.log10(y_max) + log_margin)
        else:
            # Linear scale logic
            if y_min == y_max:
                y_margin = abs(y_min) * 0.1 if y_min != 0 else 1
            else:
                y_margin = (y_max - y_min) * margin_factor
            
            y_lower = y_min - y_margin
            y_upper = y_max + y_margin
            
            if ensure_zero and y_min > 0 and y_lower > 0:
                y_lower = max(0, y_lower)
                
        ax.set_ylim(y_lower, y_upper)


def enhance_plot_visibility(ax, grid_alpha=0.4, spine_width=1.5, tick_length=6):
    """Enhance plot visibility for publication quality.
    
    Args:
        ax: Matplotlib axes object
        grid_alpha: Grid transparency
        spine_width: Width of axis spines
        tick_length: Length of major ticks
    """
    # Enhance spines
    for spine in ax.spines.values():
        if spine.get_visible():
            spine.set_linewidth(spine_width)
    
    # Enhance grid
    ax.grid(True, alpha=grid_alpha, linewidth=0.8)
    
    # Enhance ticks
    ax.tick_params(axis='both', which='major', width=spine_width, 
                   length=tick_length, labelsize=12)
    ax.tick_params(axis='both', which='minor', width=spine_width*0.7, 
                   length=tick_length*0.7)


def create_publication_figure(figsize=(8, 6), tight_layout_pad=1.5):
    """Create a figure optimized for publication quality.
    
    Args:
        figsize: Figure size tuple
        tight_layout_pad: Padding for tight layout
        
    Returns:
        Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Apply publication-quality enhancements
    enhance_plot_visibility(ax)
    
    # Set up for tight layout with appropriate padding
    plt.tight_layout(pad=tight_layout_pad)
    
    return fig, ax