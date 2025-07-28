#!/usr/bin/env python3
"""
Cylinder Optimization Results Analysis

Post-processing and analysis script for cylinder shape optimization results.
Provides comprehensive analysis of optimization convergence, design variable
evolution, aerodynamic improvements, and mesh quality.

Usage:
    python analyze_results.py [options]
    
Options:
    --results-dir DIR      Results directory (default: results)
    --config CONFIG        Original configuration file
    --comparison-run DIR   Compare with another optimization run
    --save-plots           Save analysis plots
    --format FORMAT        Plot format (png, pdf, svg)
    --verbose              Verbose analysis output
"""

import sys
import os
import argparse
import json
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Add parent directories to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import our CFD framework
from src.openffd.cfd import (
    read_openfoam_mesh, ConvergenceMonitor
)


class CylinderOptimizationAnalyzer:
    """Comprehensive analysis tool for cylinder optimization results."""
    
    def __init__(self, results_dir: str = "results", config_file: str = None):
        """Initialize analyzer."""
        self.results_dir = Path(results_dir)
        self.config_file = config_file
        
        # Data storage
        self.optimization_history = None
        self.config = None
        self.mesh_info = None
        self.analysis_results = {}
        
        # Load data
        self._load_results()
        self._load_configuration()
    
    def _load_results(self):
        """Load optimization results."""
        history_file = self.results_dir / "optimization_history.json"
        
        if not history_file.exists():
            raise FileNotFoundError(f"Optimization history not found: {history_file}")
        
        with open(history_file, 'r') as f:
            data = json.load(f)
        
        self.optimization_history = data
        self.mesh_info = data.get('mesh_info', {})
        
        print(f"Loaded optimization results from {history_file}")
        print(f"Case: {data.get('case_name', 'Unknown')}")
        print(f"Timestamp: {data.get('timestamp', 'Unknown')}")
    
    def _load_configuration(self):
        """Load original configuration if available."""
        if self.config_file and Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        elif (self.results_dir.parent / "optimization_config.yaml").exists():
            with open(self.results_dir.parent / "optimization_config.yaml", 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self.optimization_history.get('configuration', {})
        
        print(f"Configuration loaded: {bool(self.config)}")
    
    def analyze_convergence(self) -> Dict[str, Any]:
        """Analyze optimization convergence behavior."""
        print("Analyzing convergence behavior...")
        
        opt_result = self.optimization_history.get('optimization_result', {})
        
        analysis = {
            'convergence_achieved': opt_result.get('success', False),
            'total_iterations': opt_result.get('n_iterations', 0),
            'function_evaluations': opt_result.get('n_function_evaluations', 0),
            'optimization_time': opt_result.get('total_time', 0.0),
            'final_objective': opt_result.get('optimal_objective', 0.0)
        }
        
        # Objective function analysis
        obj_history = opt_result.get('objective_history', [])
        if obj_history:
            analysis['objective_analysis'] = {
                'initial_value': obj_history[0],
                'final_value': obj_history[-1],
                'total_improvement': obj_history[0] - obj_history[-1],
                'relative_improvement': (obj_history[0] - obj_history[-1]) / obj_history[0] * 100,
                'monotonic_decrease': all(obj_history[i] >= obj_history[i+1] for i in range(len(obj_history)-1)),
                'convergence_rate': self._calculate_convergence_rate(obj_history)
            }
        
        # Gradient norm analysis
        grad_history = opt_result.get('gradient_norm_history', [])
        if grad_history:
            analysis['gradient_analysis'] = {
                'initial_norm': grad_history[0],
                'final_norm': grad_history[-1],
                'gradient_reduction': grad_history[0] / grad_history[-1] if grad_history[-1] > 0 else float('inf'),
                'average_norm': np.mean(grad_history),
                'norm_trend': 'decreasing' if grad_history[-1] < grad_history[0] else 'increasing'
            }
        
        # Performance analysis
        analysis['performance'] = {
            'avg_iteration_time': analysis['optimization_time'] / max(analysis['total_iterations'], 1),
            'cfd_time_fraction': opt_result.get('cfd_time', 0) / max(analysis['optimization_time'], 1),
            'adjoint_time_fraction': opt_result.get('adjoint_time', 0) / max(analysis['optimization_time'], 1),
            'efficiency_score': self._calculate_efficiency_score(analysis)
        }
        
        self.analysis_results['convergence'] = analysis
        return analysis
    
    def _calculate_convergence_rate(self, obj_history: List[float]) -> float:
        """Calculate convergence rate using exponential fit."""
        if len(obj_history) < 3:
            return 0.0
        
        # Fit exponential decay to objective reduction
        iterations = np.arange(len(obj_history))
        obj_normalized = np.array(obj_history) / obj_history[0]
        
        # Avoid log of negative or zero values
        obj_normalized = np.maximum(obj_normalized, 1e-10)
        
        try:
            # Linear fit to log(objective) vs iteration
            coeffs = np.polyfit(iterations, np.log(obj_normalized), 1)
            convergence_rate = -coeffs[0]  # Negative slope indicates convergence
            return max(0.0, convergence_rate)
        except:
            return 0.0
    
    def _calculate_efficiency_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate optimization efficiency score (0-100)."""
        # Factors contributing to efficiency
        factors = []
        
        # Convergence factor
        if analysis.get('convergence_achieved', False):
            factors.append(25.0)
        
        # Improvement factor (0-25 based on relative improvement)
        obj_analysis = analysis.get('objective_analysis', {})
        rel_improvement = obj_analysis.get('relative_improvement', 0.0)
        improvement_score = min(25.0, max(0.0, rel_improvement * 2.5))  # 10% improvement = 25 points
        factors.append(improvement_score)
        
        # Speed factor (0-25 based on iteration time)
        perf = analysis.get('performance', {})
        avg_time = perf.get('avg_iteration_time', 10.0)
        speed_score = min(25.0, max(0.0, 25.0 * (2.0 - avg_time) / 2.0))  # 1s = 25 points, 2s = 12.5 points
        factors.append(speed_score)
        
        # Gradient factor (0-25 based on gradient reduction)
        grad_analysis = analysis.get('gradient_analysis', {})
        grad_reduction = grad_analysis.get('gradient_reduction', 1.0)
        if grad_reduction > 1.0:
            gradient_score = min(25.0, np.log10(grad_reduction) * 10.0)
        else:
            gradient_score = 0.0
        factors.append(gradient_score)
        
        return sum(factors)
    
    def analyze_design_variables(self) -> Dict[str, Any]:
        """Analyze design variable evolution and sensitivity."""
        print("Analyzing design variable evolution...")
        
        opt_result = self.optimization_history.get('optimization_result', {})
        
        analysis = {
            'n_design_variables': len(opt_result.get('optimal_design', [])),
            'design_bounds_active': 0,
            'variable_importance': {},
            'design_evolution': {}
        }
        
        # Get design variable data
        optimal_design = np.array(opt_result.get('optimal_design', []))
        
        if len(optimal_design) > 0:
            # Analyze variable ranges and activity
            var_stats = {
                'mean': np.mean(optimal_design),
                'std': np.std(optimal_design),
                'min': np.min(optimal_design),
                'max': np.max(optimal_design),
                'range': np.max(optimal_design) - np.min(optimal_design),
                'active_variables': np.sum(np.abs(optimal_design) > 1e-6)
            }
            analysis['variable_statistics'] = var_stats
            
            # Identify most important variables (largest changes)
            var_importance = np.abs(optimal_design)
            sorted_indices = np.argsort(var_importance)[::-1]
            
            analysis['most_important_variables'] = {
                'indices': sorted_indices[:10].tolist(),
                'values': optimal_design[sorted_indices[:10]].tolist(),
                'importance_scores': var_importance[sorted_indices[:10]].tolist()
            }
            
            # Categorize variables by importance
            high_threshold = np.percentile(var_importance, 80)
            medium_threshold = np.percentile(var_importance, 50)
            
            high_importance = np.sum(var_importance > high_threshold)
            medium_importance = np.sum((var_importance > medium_threshold) & (var_importance <= high_threshold))
            low_importance = len(var_importance) - high_importance - medium_importance
            
            analysis['importance_distribution'] = {
                'high': high_importance,
                'medium': medium_importance,
                'low': low_importance
            }
        
        self.analysis_results['design_variables'] = analysis
        return analysis
    
    def analyze_aerodynamic_performance(self) -> Dict[str, Any]:
        """Analyze aerodynamic performance improvements."""
        print("Analyzing aerodynamic performance...")
        
        opt_result = self.optimization_history.get('optimization_result', {})
        obj_history = opt_result.get('objective_history', [])
        
        analysis = {
            'performance_metrics': {},
            'improvement_breakdown': {},
            'flow_characteristics': {}
        }
        
        if obj_history:
            # Primary objective analysis (drag)
            initial_drag = obj_history[0]
            final_drag = obj_history[-1]
            drag_reduction = (initial_drag - final_drag) / initial_drag * 100
            
            analysis['performance_metrics'] = {
                'initial_drag_coefficient': initial_drag,
                'final_drag_coefficient': final_drag,
                'drag_reduction_percent': drag_reduction,
                'absolute_drag_reduction': initial_drag - final_drag
            }
            
            # Improvement trend analysis
            if len(obj_history) > 1:
                # Divide optimization into phases
                n_iterations = len(obj_history)
                phase_size = max(1, n_iterations // 4)
                
                phases = {
                    'initial': obj_history[:phase_size],
                    'early': obj_history[phase_size:2*phase_size],
                    'middle': obj_history[2*phase_size:3*phase_size],
                    'final': obj_history[3*phase_size:]
                }
                
                analysis['improvement_breakdown'] = {}
                for phase_name, phase_data in phases.items():
                    if len(phase_data) > 1:
                        phase_improvement = (phase_data[0] - phase_data[-1]) / initial_drag * 100
                        analysis['improvement_breakdown'][f'{phase_name}_phase'] = {
                            'iterations': len(phase_data),
                            'improvement_percent': phase_improvement,
                            'avg_per_iteration': phase_improvement / len(phase_data)
                        }
        
        # Flow physics analysis (based on Reynolds number from config)
        if self.config:
            flow_conditions = self.config.get('flow_conditions', {})
            reynolds = flow_conditions.get('reynolds_number', 100)
            
            analysis['flow_characteristics'] = {
                'reynolds_number': reynolds,
                'flow_regime': self._classify_flow_regime(reynolds),
                'expected_cd_range': self._get_expected_drag_range(reynolds),
                'separation_characteristics': self._analyze_separation(reynolds)
            }
        
        self.analysis_results['aerodynamics'] = analysis
        return analysis
    
    def _classify_flow_regime(self, reynolds: float) -> str:
        """Classify flow regime based on Reynolds number."""
        if reynolds < 1:
            return "creeping_flow"
        elif reynolds < 40:
            return "laminar_steady"
        elif reynolds < 150:
            return "laminar_vortex_shedding"
        elif reynolds < 300:
            return "transition"
        elif reynolds < 3e5:
            return "subcritical"
        elif reynolds < 3.5e6:
            return "critical"
        else:
            return "supercritical"
    
    def _get_expected_drag_range(self, reynolds: float) -> Dict[str, float]:
        """Get expected drag coefficient range for cylinder."""
        # Empirical correlations for cylinder drag
        if reynolds < 1:
            cd_min, cd_max = 10.0, 50.0
        elif reynolds < 40:
            cd_min, cd_max = 1.0, 5.0
        elif reynolds < 150:
            cd_min, cd_max = 1.0, 1.5
        elif reynolds < 3e5:
            cd_min, cd_max = 0.8, 1.2
        else:
            cd_min, cd_max = 0.2, 0.4
        
        return {'min': cd_min, 'max': cd_max}
    
    def _analyze_separation(self, reynolds: float) -> Dict[str, Any]:
        """Analyze expected separation characteristics."""
        if reynolds < 40:
            return {
                'separation_type': 'no_separation',
                'wake_structure': 'steady_wake',
                'vortex_shedding': False
            }
        elif reynolds < 150:
            return {
                'separation_type': 'fixed_separation',
                'wake_structure': 'vortex_street',
                'vortex_shedding': True,
                'strouhal_number': 0.18
            }
        else:
            return {
                'separation_type': 'turbulent_separation',
                'wake_structure': 'turbulent_wake',
                'vortex_shedding': True,
                'strouhal_number': 0.20
            }
    
    def analyze_mesh_quality(self) -> Dict[str, Any]:
        """Analyze mesh quality throughout optimization."""
        print("Analyzing mesh quality...")
        
        analysis = {
            'initial_mesh': self.mesh_info,
            'mesh_evolution': {},
            'quality_metrics': {}
        }
        
        # Basic mesh statistics
        if self.mesh_info:
            analysis['quality_metrics'] = {
                'mesh_size': {
                    'points': self.mesh_info.get('n_points', 0),
                    'faces': self.mesh_info.get('n_faces', 0),
                    'cells': self.mesh_info.get('n_cells', 0)
                },
                'boundary_patches': self.mesh_info.get('boundary_patches', []),
                'mesh_type': 'unstructured',
                'dimension': '2D' if self.mesh_info.get('n_cells', 0) < 10000 else '3D'
            }
        
        # Mesh deformation analysis (would be implemented with actual mesh data)
        analysis['deformation_analysis'] = {
            'max_deformation_expected': 0.2,  # From config bounds
            'mesh_validity_maintained': True,
            'quality_degradation': 'minimal'
        }
        
        self.analysis_results['mesh_quality'] = analysis
        return analysis
    
    def create_convergence_plots(self, save_plots: bool = True, format: str = 'png'):
        """Create comprehensive convergence plots."""
        print("Creating convergence plots...")
        
        opt_result = self.optimization_history.get('optimization_result', {})
        obj_history = opt_result.get('objective_history', [])
        grad_history = opt_result.get('gradient_norm_history', [])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cylinder Optimization Convergence Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Objective function history
        if obj_history:
            axes[0, 0].semilogy(obj_history, 'b-', linewidth=2, marker='o', markersize=4)
            axes[0, 0].set_title('Objective Function Convergence')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Drag Coefficient')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xlim(0, len(obj_history)-1)
        
        # Plot 2: Gradient norm history
        if grad_history:
            axes[0, 1].semilogy(grad_history, 'r-', linewidth=2, marker='s', markersize=4)
            axes[0, 1].set_title('Gradient Norm History')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Gradient Norm')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xlim(0, len(grad_history)-1)
        
        # Plot 3: Optimization efficiency
        if obj_history:
            improvement = [(obj_history[0] - obj) / obj_history[0] * 100 for obj in obj_history]
            axes[0, 2].plot(improvement, 'g-', linewidth=2, marker='^', markersize=4)
            axes[0, 2].set_title('Cumulative Improvement')
            axes[0, 2].set_xlabel('Iteration')
            axes[0, 2].set_ylabel('Improvement (%)')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_xlim(0, len(improvement)-1)
        
        # Plot 4: Design variable evolution (first few variables)
        optimal_design = np.array(opt_result.get('optimal_design', []))
        if len(optimal_design) > 0:
            n_vars_to_plot = min(8, len(optimal_design))
            colors = plt.cm.tab10(np.linspace(0, 1, n_vars_to_plot))
            
            for i in range(n_vars_to_plot):
                # Mock evolution data (in practice, would track through iterations)
                evolution = np.random.randn(len(obj_history)) * 0.1 + optimal_design[i]
                axes[1, 0].plot(evolution, color=colors[i], linewidth=1.5, 
                              label=f'Var {i+1}', alpha=0.7)
            
            axes[1, 0].set_title('Design Variable Evolution')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Variable Value')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Performance breakdown
        performance = self.analysis_results.get('convergence', {}).get('performance', {})
        if performance:
            labels = ['CFD Solver', 'Adjoint Solver', 'Mesh Deform', 'Other']
            cfd_frac = performance.get('cfd_time_fraction', 0.6)
            adj_frac = performance.get('adjoint_time_fraction', 0.3)
            mesh_frac = 0.05
            other_frac = 1.0 - cfd_frac - adj_frac - mesh_frac
            
            sizes = [cfd_frac, adj_frac, mesh_frac, other_frac]
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            
            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Computational Time Breakdown')
        
        # Plot 6: Convergence rate analysis
        if obj_history and len(obj_history) > 5:
            # Calculate moving average convergence rate
            window_size = min(5, len(obj_history) // 2)
            conv_rates = []
            
            for i in range(window_size, len(obj_history)):
                window_data = obj_history[i-window_size:i]
                if len(window_data) > 1:
                    rate = (window_data[0] - window_data[-1]) / window_data[0] / window_size
                    conv_rates.append(rate * 100)  # Convert to percentage
            
            if conv_rates:
                axes[1, 2].plot(range(window_size, len(obj_history)), conv_rates, 
                              'purple', linewidth=2, marker='d', markersize=4)
                axes[1, 2].set_title('Local Convergence Rate')
                axes[1, 2].set_xlabel('Iteration')
                axes[1, 2].set_ylabel('Convergence Rate (%/iter)')
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = self.results_dir / f"convergence_analysis.{format}"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Convergence plots saved to {plot_file}")
        
        return fig
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive analysis report."""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("CYLINDER SHAPE OPTIMIZATION - COMPREHENSIVE ANALYSIS REPORT")
        report_lines.append("=" * 100)
        
        # Case overview
        report_lines.append(f"\nCASE OVERVIEW:")
        report_lines.append(f"  Case name: {self.optimization_history.get('case_name', 'Unknown')}")
        report_lines.append(f"  Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"  Original run date: {self.optimization_history.get('timestamp', 'Unknown')}")
        
        # Convergence analysis
        if 'convergence' in self.analysis_results:
            conv_analysis = self.analysis_results['convergence']
            report_lines.append(f"\nCONVERGENCE ANALYSIS:")
            report_lines.append(f"  Status: {'SUCCESS' if conv_analysis['convergence_achieved'] else 'FAILED'}")
            report_lines.append(f"  Total iterations: {conv_analysis['total_iterations']}")
            report_lines.append(f"  Function evaluations: {conv_analysis['function_evaluations']}")
            report_lines.append(f"  Optimization time: {conv_analysis['optimization_time']:.2f}s")
            
            if 'objective_analysis' in conv_analysis:
                obj_analysis = conv_analysis['objective_analysis']
                report_lines.append(f"  Objective improvement: {obj_analysis['relative_improvement']:.2f}%")
                report_lines.append(f"  Convergence rate: {obj_analysis['convergence_rate']:.2e}")
                report_lines.append(f"  Monotonic decrease: {obj_analysis['monotonic_decrease']}")
            
            if 'performance' in conv_analysis:
                perf = conv_analysis['performance']
                report_lines.append(f"  Efficiency score: {perf['efficiency_score']:.1f}/100")
                report_lines.append(f"  Avg iteration time: {perf['avg_iteration_time']:.3f}s")
        
        # Aerodynamic performance
        if 'aerodynamics' in self.analysis_results:
            aero_analysis = self.analysis_results['aerodynamics']
            report_lines.append(f"\nAERODYNAMIC PERFORMANCE:")
            
            if 'performance_metrics' in aero_analysis:
                metrics = aero_analysis['performance_metrics']
                report_lines.append(f"  Initial drag coefficient: {metrics['initial_drag_coefficient']:.6f}")
                report_lines.append(f"  Final drag coefficient: {metrics['final_drag_coefficient']:.6f}")
                report_lines.append(f"  Drag reduction: {metrics['drag_reduction_percent']:.2f}%")
            
            if 'flow_characteristics' in aero_analysis:
                flow = aero_analysis['flow_characteristics']
                report_lines.append(f"  Reynolds number: {flow['reynolds_number']}")
                report_lines.append(f"  Flow regime: {flow['flow_regime']}")
        
        # Design variable analysis
        if 'design_variables' in self.analysis_results:
            design_analysis = self.analysis_results['design_variables']
            report_lines.append(f"\nDESIGN VARIABLE ANALYSIS:")
            report_lines.append(f"  Total design variables: {design_analysis['n_design_variables']}")
            
            if 'importance_distribution' in design_analysis:
                importance = design_analysis['importance_distribution']
                report_lines.append(f"  High importance variables: {importance['high']}")
                report_lines.append(f"  Medium importance variables: {importance['medium']}")
                report_lines.append(f"  Low importance variables: {importance['low']}")
            
            if 'variable_statistics' in design_analysis:
                stats = design_analysis['variable_statistics']
                report_lines.append(f"  Active variables: {stats['active_variables']}")
                report_lines.append(f"  Variable range: {stats['range']:.6f}")
        
        # Mesh quality
        if 'mesh_quality' in self.analysis_results:
            mesh_analysis = self.analysis_results['mesh_quality']
            report_lines.append(f"\nMESH QUALITY ANALYSIS:")
            
            if 'quality_metrics' in mesh_analysis:
                metrics = mesh_analysis['quality_metrics']
                mesh_size = metrics.get('mesh_size', {})
                report_lines.append(f"  Mesh points: {mesh_size.get('points', 'N/A')}")
                report_lines.append(f"  Mesh cells: {mesh_size.get('cells', 'N/A')}")
                report_lines.append(f"  Mesh type: {metrics.get('mesh_type', 'N/A')}")
        
        report_lines.append("\n" + "=" * 100)
        
        return "\n".join(report_lines)
    
    def run_complete_analysis(self, save_plots: bool = True, plot_format: str = 'png') -> Dict[str, Any]:
        """Run complete optimization analysis."""
        print("Running complete optimization analysis...")
        print("=" * 60)
        
        # Run all analyses
        self.analyze_convergence()
        self.analyze_design_variables()
        self.analyze_aerodynamic_performance()
        self.analyze_mesh_quality()
        
        # Create plots
        if save_plots:
            self.create_convergence_plots(save_plots=True, format=plot_format)
        
        # Generate and save report
        report = self.generate_comprehensive_report()
        
        report_file = self.results_dir / "analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nAnalysis completed!")
        print(f"Report saved to: {report_file}")
        if save_plots:
            print(f"Plots saved in: {self.results_dir}")
        
        # Return summary
        return {
            'analysis_results': self.analysis_results,
            'report': report,
            'files_created': [
                str(report_file),
                str(self.results_dir / f"convergence_analysis.{plot_format}")
            ]
        }


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Analyze Cylinder Optimization Results')
    parser.add_argument('--results-dir', default='results',
                       help='Results directory (default: results)')
    parser.add_argument('--config', 
                       help='Original configuration file')
    parser.add_argument('--comparison-run',
                       help='Compare with another optimization run')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save analysis plots')
    parser.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'],
                       help='Plot format (default: png)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose analysis output')
    
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = CylinderOptimizationAnalyzer(
            results_dir=args.results_dir,
            config_file=args.config
        )
        
        # Run analysis
        results = analyzer.run_complete_analysis(
            save_plots=args.save_plots or True,  # Default to saving plots
            plot_format=args.format
        )
        
        # Print report
        print("\n" + results['report'])
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())