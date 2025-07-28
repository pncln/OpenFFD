"""
Convergence Monitoring and Residual Tracking for CFD Simulations

Implements comprehensive convergence monitoring capabilities:
- Multi-variable residual tracking with different norms
- Convergence history analysis and trend detection
- Adaptive convergence criteria and stopping conditions
- Stagnation detection and divergence monitoring
- Real-time convergence visualization and logging
- Performance metrics and iteration efficiency analysis
- Automated convergence assessment for optimization loops

Essential for robust CFD simulations and optimization workflows.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class ResidualNorm(Enum):
    """Enumeration of residual norm types."""
    L1 = "l1"
    L2 = "l2"
    LINF = "linf"
    RMS = "rms"
    SCALED = "scaled"


class ConvergenceStatus(Enum):
    """Enumeration of convergence status."""
    CONVERGING = "converging"
    CONVERGED = "converged"
    STAGNATED = "stagnated"
    DIVERGING = "diverging"
    OSCILLATING = "oscillating"
    UNKNOWN = "unknown"


class StoppingCriterion(Enum):
    """Enumeration of stopping criteria."""
    ABSOLUTE_RESIDUAL = "absolute_residual"
    RELATIVE_RESIDUAL = "relative_residual"
    RESIDUAL_REDUCTION = "residual_reduction"
    MAX_ITERATIONS = "max_iterations"
    STAGNATION = "stagnation"
    DIVERGENCE = "divergence"
    USER_DEFINED = "user_defined"


@dataclass
class ConvergenceConfig:
    """Configuration for convergence monitoring."""
    
    # Residual settings
    residual_norm: ResidualNorm = ResidualNorm.L2
    variable_names: List[str] = field(default_factory=lambda: ["rho", "rho_u", "rho_v", "rho_w", "rho_E"])
    monitor_variables: List[str] = field(default_factory=lambda: ["rho", "rho_u", "rho_v", "rho_w", "rho_E"])
    
    # Convergence criteria
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 1e-3
    reduction_tolerance: float = 1e-3  # Residual reduction factor
    max_iterations: int = 10000
    min_iterations: int = 10
    
    # Stagnation detection
    stagnation_window: int = 50  # Number of iterations to check
    stagnation_tolerance: float = 1e-8  # Minimum change in residual
    oscillation_threshold: float = 0.1  # Relative oscillation detection
    
    # Divergence detection
    divergence_factor: float = 100.0  # Divergence if residual increases by this factor
    max_residual: float = 1e6  # Maximum allowable residual
    
    # History and logging
    history_length: int = 1000  # Number of iterations to keep in memory
    log_frequency: int = 10  # Print convergence info every N iterations
    save_frequency: int = 100  # Save convergence data every N iterations
    
    # Advanced features
    adaptive_criteria: bool = True  # Adapt convergence criteria during solve
    early_stopping: bool = True  # Stop early if convergence is detected
    trend_analysis: bool = True  # Analyze convergence trends
    performance_monitoring: bool = True  # Monitor iteration performance


@dataclass
class ResidualData:
    """Data structure for residual information."""
    
    iteration: int
    time: float
    residuals: Dict[str, float]  # Residual values by variable
    norms: Dict[str, float]  # Different norm calculations
    
    # Additional metrics
    residual_ratio: float = 0.0  # Current/initial residual ratio
    reduction_rate: float = 0.0  # Rate of residual reduction
    iteration_time: float = 0.0  # Time for this iteration
    
    # Convergence indicators
    converged_variables: List[str] = field(default_factory=list)
    status: ConvergenceStatus = ConvergenceStatus.UNKNOWN


@dataclass
class ConvergenceMetrics:
    """Comprehensive convergence metrics."""
    
    # Overall status
    status: ConvergenceStatus = ConvergenceStatus.UNKNOWN
    converged: bool = False
    current_iteration: int = 0
    total_time: float = 0.0
    
    # Residual metrics
    current_residuals: Dict[str, float] = field(default_factory=dict)
    initial_residuals: Dict[str, float] = field(default_factory=dict)
    final_residuals: Dict[str, float] = field(default_factory=dict)
    residual_ratios: Dict[str, float] = field(default_factory=dict)
    
    # Convergence rates
    average_reduction_rate: Dict[str, float] = field(default_factory=dict)
    instantaneous_reduction_rate: Dict[str, float] = field(default_factory=dict)
    asymptotic_reduction_rate: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    average_iteration_time: float = 0.0
    total_solve_time: float = 0.0
    iterations_per_second: float = 0.0
    convergence_efficiency: float = 0.0  # Iterations to convergence / total iterations
    
    # Quality metrics
    smoothness_factor: float = 0.0  # How smooth the convergence is
    oscillation_amplitude: Dict[str, float] = field(default_factory=dict)
    stagnation_iterations: int = 0


class ResidualComputer:
    """
    Computes residuals with different norms and scaling options.
    
    Handles various residual calculation methods for CFD equations.
    """
    
    def __init__(self, config: ConvergenceConfig):
        """Initialize residual computer."""
        self.config = config
        
    def compute_residuals(self,
                         residual_vector: np.ndarray,
                         variable_partition: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Compute residuals for all monitored variables.
        
        Args:
            residual_vector: Flattened residual vector
            variable_partition: Partition indices for different variables
            
        Returns:
            Dictionary of residual values by variable name
        """
        if variable_partition is None:
            # Assume equal partitioning
            n_vars = len(self.config.variable_names)
            n_cells = len(residual_vector) // n_vars
            variable_partition = [n_cells * i for i in range(n_vars + 1)]
        
        residuals = {}
        
        for i, var_name in enumerate(self.config.variable_names):
            if var_name not in self.config.monitor_variables:
                continue
                
            start_idx = variable_partition[i]
            end_idx = variable_partition[i + 1]
            var_residual = residual_vector[start_idx:end_idx]
            
            residuals[var_name] = self._compute_norm(var_residual)
        
        return residuals
    
    def _compute_norm(self, residual_array: np.ndarray) -> float:
        """Compute specified norm of residual array."""
        if len(residual_array) == 0:
            return 0.0
        
        if self.config.residual_norm == ResidualNorm.L1:
            return np.sum(np.abs(residual_array))
        elif self.config.residual_norm == ResidualNorm.L2:
            return np.sqrt(np.sum(residual_array**2))
        elif self.config.residual_norm == ResidualNorm.LINF:
            return np.max(np.abs(residual_array))
        elif self.config.residual_norm == ResidualNorm.RMS:
            return np.sqrt(np.mean(residual_array**2))
        elif self.config.residual_norm == ResidualNorm.SCALED:
            # Scale by maximum absolute value
            max_val = np.max(np.abs(residual_array))
            if max_val > 1e-12:
                return np.sqrt(np.sum((residual_array / max_val)**2))
            else:
                return 0.0
        else:
            return np.sqrt(np.sum(residual_array**2))  # Default to L2
    
    def compute_multiple_norms(self, residual_array: np.ndarray) -> Dict[str, float]:
        """Compute multiple norms for analysis."""
        norms = {}
        
        if len(residual_array) == 0:
            return {norm.value: 0.0 for norm in ResidualNorm}
        
        norms[ResidualNorm.L1.value] = np.sum(np.abs(residual_array))
        norms[ResidualNorm.L2.value] = np.sqrt(np.sum(residual_array**2))
        norms[ResidualNorm.LINF.value] = np.max(np.abs(residual_array))
        norms[ResidualNorm.RMS.value] = np.sqrt(np.mean(residual_array**2))
        
        # Scaled norm
        max_val = np.max(np.abs(residual_array))
        if max_val > 1e-12:
            norms[ResidualNorm.SCALED.value] = np.sqrt(np.sum((residual_array / max_val)**2))
        else:
            norms[ResidualNorm.SCALED.value] = 0.0
        
        return norms


class ConvergenceAnalyzer:
    """
    Analyzes convergence trends and detects convergence patterns.
    
    Provides sophisticated analysis of convergence behavior.
    """
    
    def __init__(self, config: ConvergenceConfig):
        """Initialize convergence analyzer."""
        self.config = config
        
    def analyze_convergence_trend(self, history: List[ResidualData]) -> Dict[str, Any]:
        """Analyze convergence trend from residual history."""
        if len(history) < 5:
            return {"status": "insufficient_data"}
        
        analysis = {}
        
        # Extract residual sequences for each variable
        for var_name in self.config.monitor_variables:
            var_residuals = [data.residuals.get(var_name, 0.0) for data in history]
            analysis[var_name] = self._analyze_variable_trend(var_residuals)
        
        # Overall trend analysis
        analysis["overall"] = self._analyze_overall_trend(history)
        
        return analysis
    
    def _analyze_variable_trend(self, residuals: List[float]) -> Dict[str, Any]:
        """Analyze trend for a single variable."""
        if len(residuals) < 5:
            return {"status": "insufficient_data"}
        
        residuals_array = np.array(residuals)
        
        # Remove zeros and extremely small values for log analysis
        nonzero_residuals = residuals_array[residuals_array > 1e-16]
        
        analysis = {
            "current_residual": residuals[-1],
            "initial_residual": residuals[0],
            "reduction_ratio": residuals[-1] / (residuals[0] + 1e-16),
            "trend": "unknown"
        }
        
        if len(nonzero_residuals) > 2:
            # Linear regression on log residuals
            log_residuals = np.log10(nonzero_residuals)
            x = np.arange(len(log_residuals))
            
            if len(x) > 1:
                slope, intercept = np.polyfit(x, log_residuals, 1)
                analysis["convergence_rate"] = -slope  # Negative slope means convergence
                analysis["r_squared"] = self._compute_r_squared(x, log_residuals, slope, intercept)
                
                # Classify trend
                if slope < -0.01:
                    analysis["trend"] = "converging"
                elif slope > 0.01:
                    analysis["trend"] = "diverging"
                else:
                    analysis["trend"] = "stagnating"
        
        # Oscillation analysis
        if len(residuals) > 10:
            recent_residuals = residuals[-10:]
            analysis["oscillation_amplitude"] = np.std(recent_residuals) / (np.mean(recent_residuals) + 1e-16)
        
        return analysis
    
    def _analyze_overall_trend(self, history: List[ResidualData]) -> Dict[str, Any]:
        """Analyze overall convergence trend."""
        if len(history) < 5:
            return {"status": "insufficient_data"}
        
        # Compute combined residual metric
        combined_residuals = []
        for data in history:
            total_residual = sum(data.residuals.values())
            combined_residuals.append(total_residual)
        
        analysis = self._analyze_variable_trend(combined_residuals)
        
        # Add iteration efficiency
        if len(history) > 1:
            total_time = history[-1].time - history[0].time
            iterations = len(history)
            analysis["iterations_per_second"] = iterations / (total_time + 1e-12)
        
        return analysis
    
    def _compute_r_squared(self, x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
        """Compute R-squared value for linear fit."""
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        
        if ss_tot < 1e-16:
            return 1.0
        
        return 1.0 - (ss_res / ss_tot)
    
    def detect_stagnation(self, history: List[ResidualData]) -> bool:
        """Detect if convergence has stagnated."""
        if len(history) < self.config.stagnation_window:
            return False
        
        recent_history = history[-self.config.stagnation_window:]
        
        for var_name in self.config.monitor_variables:
            var_residuals = [data.residuals.get(var_name, 0.0) for data in recent_history]
            
            if len(var_residuals) > 1:
                max_change = max(var_residuals) - min(var_residuals)
                avg_residual = np.mean(var_residuals)
                
                relative_change = max_change / (avg_residual + 1e-16)
                
                if relative_change > self.config.stagnation_tolerance:
                    return False  # Still changing significantly
        
        return True  # All variables are stagnating
    
    def detect_divergence(self, history: List[ResidualData]) -> bool:
        """Detect if solution is diverging."""
        if len(history) < 2:
            return False
        
        current_data = history[-1]
        
        # Check for extremely large residuals
        for residual in current_data.residuals.values():
            if residual > self.config.max_residual:
                return True
        
        # Check for rapid increase in residuals
        if len(history) > 10:
            for var_name in self.config.monitor_variables:
                recent_residuals = [data.residuals.get(var_name, 0.0) for data in history[-10:]]
                if len(recent_residuals) > 1:
                    initial_residual = recent_residuals[0]
                    current_residual = recent_residuals[-1]
                    
                    if initial_residual > 1e-16 and current_residual / initial_residual > self.config.divergence_factor:
                        return True
        
        return False
    
    def detect_oscillations(self, history: List[ResidualData]) -> Dict[str, bool]:
        """Detect oscillatory behavior in residuals."""
        oscillations = {}
        
        if len(history) < 20:
            return {var: False for var in self.config.monitor_variables}
        
        for var_name in self.config.monitor_variables:
            var_residuals = [data.residuals.get(var_name, 0.0) for data in history[-20:]]
            
            # Simple oscillation detection using variance and trend
            mean_residual = np.mean(var_residuals)
            std_residual = np.std(var_residuals)
            
            relative_oscillation = std_residual / (mean_residual + 1e-16)
            oscillations[var_name] = relative_oscillation > self.config.oscillation_threshold
        
        return oscillations


class ConvergenceMonitor:
    """
    Main convergence monitoring system.
    
    Coordinates residual computation, analysis, and convergence assessment.
    """
    
    def __init__(self, config: ConvergenceConfig):
        """Initialize convergence monitor."""
        self.config = config
        self.residual_computer = ResidualComputer(config)
        self.analyzer = ConvergenceAnalyzer(config)
        
        # History storage
        self.history: deque = deque(maxlen=config.history_length)
        self.metrics = ConvergenceMetrics()
        
        # Initial state
        self.initial_residuals: Optional[Dict[str, float]] = None
        self.start_time = time.time()
        self.last_log_iteration = 0
        self.last_save_iteration = 0
        
        # Convergence tracking
        self.converged_variables = set()
        self.stopping_criteria_met = {}
        
    def update_residuals(self,
                        iteration: int,
                        residual_vector: np.ndarray,
                        variable_partition: Optional[List[int]] = None,
                        iteration_time: float = 0.0) -> ResidualData:
        """
        Update residuals and analyze convergence.
        
        Args:
            iteration: Current iteration number
            residual_vector: Flattened residual vector
            variable_partition: Partition indices for variables
            iteration_time: Time taken for this iteration
            
        Returns:
            Residual data for this iteration
        """
        current_time = time.time()
        
        # Compute residuals
        residuals = self.residual_computer.compute_residuals(residual_vector, variable_partition)
        norms = self.residual_computer.compute_multiple_norms(residual_vector)
        
        # Store initial residuals
        if self.initial_residuals is None:
            self.initial_residuals = residuals.copy()
            self.metrics.initial_residuals = residuals.copy()
        
        # Compute residual ratios
        residual_ratio = max(residuals[var] / (self.initial_residuals[var] + 1e-16) 
                           for var in residuals.keys())
        
        # Compute reduction rate
        reduction_rate = 0.0
        if len(self.history) > 0:
            prev_residual = sum(self.history[-1].residuals.values())
            curr_residual = sum(residuals.values())
            if prev_residual > 1e-16:
                reduction_rate = (prev_residual - curr_residual) / prev_residual
        
        # Create residual data
        residual_data = ResidualData(
            iteration=iteration,
            time=current_time - self.start_time,
            residuals=residuals,
            norms=norms,
            residual_ratio=residual_ratio,
            reduction_rate=reduction_rate,
            iteration_time=iteration_time
        )
        
        # Add to history
        self.history.append(residual_data)
        
        # Analyze convergence
        self._analyze_current_state(residual_data)
        
        # Update metrics
        self._update_metrics()
        
        # Check logging and saving
        self._handle_logging_and_saving(iteration)
        
        return residual_data
    
    def _analyze_current_state(self, residual_data: ResidualData):
        """Analyze current convergence state."""
        # Check individual variable convergence
        converged_vars = []
        for var_name, residual in residual_data.residuals.items():
            if self._is_variable_converged(var_name, residual):
                converged_vars.append(var_name)
                self.converged_variables.add(var_name)
        
        residual_data.converged_variables = converged_vars
        
        # Determine overall status
        if len(self.history) < self.config.min_iterations:
            residual_data.status = ConvergenceStatus.UNKNOWN
        else:
            # Check for divergence first
            if self.analyzer.detect_divergence(list(self.history)):
                residual_data.status = ConvergenceStatus.DIVERGING
            # Check for stagnation
            elif self.analyzer.detect_stagnation(list(self.history)):
                residual_data.status = ConvergenceStatus.STAGNATED
            # Check for convergence
            elif self._check_overall_convergence():
                residual_data.status = ConvergenceStatus.CONVERGED
            # Check for oscillations
            elif any(self.analyzer.detect_oscillations(list(self.history)).values()):
                residual_data.status = ConvergenceStatus.OSCILLATING
            else:
                residual_data.status = ConvergenceStatus.CONVERGING
    
    def _is_variable_converged(self, var_name: str, residual: float) -> bool:
        """Check if a specific variable has converged."""
        if self.initial_residuals is None:
            return False
        
        initial_residual = self.initial_residuals.get(var_name, 1.0)
        
        # Absolute criterion
        if residual < self.config.absolute_tolerance:
            return True
        
        # Relative criterion
        if residual / (initial_residual + 1e-16) < self.config.relative_tolerance:
            return True
        
        # Reduction criterion
        if residual < initial_residual * self.config.reduction_tolerance:
            return True
        
        return False
    
    def _check_overall_convergence(self) -> bool:
        """Check if overall solution has converged."""
        if len(self.converged_variables) >= len(self.config.monitor_variables):
            return True
        
        # Alternative: check if most variables are converged
        convergence_fraction = len(self.converged_variables) / len(self.config.monitor_variables)
        return convergence_fraction >= 0.8  # 80% of variables converged
    
    def _update_metrics(self):
        """Update convergence metrics."""
        if not self.history:
            return
        
        latest_data = self.history[-1]
        
        # Update basic metrics
        self.metrics.current_iteration = latest_data.iteration
        self.metrics.total_time = latest_data.time
        self.metrics.current_residuals = latest_data.residuals.copy()
        self.metrics.status = latest_data.status
        self.metrics.converged = (latest_data.status == ConvergenceStatus.CONVERGED)
        
        # Compute residual ratios
        if self.initial_residuals:
            for var_name in latest_data.residuals:
                initial = self.initial_residuals.get(var_name, 1.0)
                current = latest_data.residuals[var_name]
                self.metrics.residual_ratios[var_name] = current / (initial + 1e-16)
        
        # Performance metrics
        if len(self.history) > 1:
            iteration_times = [data.iteration_time for data in self.history if data.iteration_time > 0]
            if iteration_times:
                self.metrics.average_iteration_time = np.mean(iteration_times)
                self.metrics.iterations_per_second = 1.0 / (self.metrics.average_iteration_time + 1e-12)
        
        # Convergence efficiency
        if self.metrics.converged and self.metrics.current_iteration > 0:
            self.metrics.convergence_efficiency = self.metrics.current_iteration / self.config.max_iterations
    
    def _handle_logging_and_saving(self, iteration: int):
        """Handle periodic logging and saving."""
        # Logging
        if iteration - self.last_log_iteration >= self.config.log_frequency:
            self._log_convergence_status()
            self.last_log_iteration = iteration
        
        # Saving (placeholder for actual file saving)
        if iteration - self.last_save_iteration >= self.config.save_frequency:
            self._save_convergence_data()
            self.last_save_iteration = iteration
    
    def _log_convergence_status(self):
        """Log current convergence status."""
        if not self.history:
            return
        
        latest_data = self.history[-1]
        
        logger.info(f"Iteration {latest_data.iteration}: Status = {latest_data.status.value}")
        
        for var_name, residual in latest_data.residuals.items():
            ratio = residual / (self.initial_residuals.get(var_name, 1.0) + 1e-16) if self.initial_residuals else 1.0
            logger.info(f"  {var_name}: {residual:.2e} (ratio: {ratio:.2e})")
        
        if latest_data.converged_variables:
            logger.info(f"  Converged variables: {', '.join(latest_data.converged_variables)}")
    
    def _save_convergence_data(self):
        """Save convergence data to file (placeholder)."""
        # In practice, would save to JSON or HDF5 format
        pass
    
    def check_stopping_criteria(self) -> Tuple[bool, List[str]]:
        """
        Check all stopping criteria.
        
        Returns:
            Tuple of (should_stop, reasons)
        """
        if not self.history:
            return False, []
        
        latest_data = self.history[-1]
        reasons = []
        should_stop = False
        
        # Convergence criterion
        if latest_data.status == ConvergenceStatus.CONVERGED:
            reasons.append("Solution converged")
            should_stop = True
        
        # Divergence criterion
        elif latest_data.status == ConvergenceStatus.DIVERGING:
            reasons.append("Solution diverging")
            should_stop = True
        
        # Stagnation criterion
        elif latest_data.status == ConvergenceStatus.STAGNATED:
            reasons.append("Solution stagnated")
            should_stop = True
        
        # Maximum iterations
        if latest_data.iteration >= self.config.max_iterations:
            reasons.append("Maximum iterations reached")
            should_stop = True
        
        return should_stop, reasons
    
    def get_convergence_report(self) -> Dict[str, Any]:
        """Generate comprehensive convergence report."""
        if len(self.history) < 2:
            return {"status": "insufficient_data"}
        
        # Trend analysis
        trend_analysis = self.analyzer.analyze_convergence_trend(list(self.history))
        
        # Performance summary
        performance = {
            "total_iterations": self.metrics.current_iteration,
            "total_time": self.metrics.total_time,
            "average_iteration_time": self.metrics.average_iteration_time,
            "iterations_per_second": self.metrics.iterations_per_second
        }
        
        # Convergence summary
        convergence_summary = {
            "status": self.metrics.status.value,
            "converged": self.metrics.converged,
            "converged_variables": list(self.converged_variables),
            "residual_ratios": self.metrics.residual_ratios
        }
        
        # History statistics
        if self.history:
            final_residuals = self.history[-1].residuals
            history_stats = {
                "initial_residuals": self.initial_residuals,
                "final_residuals": final_residuals,
                "best_residuals": self._compute_best_residuals(),
                "worst_residuals": self._compute_worst_residuals()
            }
        else:
            history_stats = {}
        
        return {
            "convergence_summary": convergence_summary,
            "performance": performance,
            "trend_analysis": trend_analysis,
            "history_statistics": history_stats,
            "configuration": {
                "residual_norm": self.config.residual_norm.value,
                "absolute_tolerance": self.config.absolute_tolerance,
                "relative_tolerance": self.config.relative_tolerance,
                "max_iterations": self.config.max_iterations
            }
        }
    
    def _compute_best_residuals(self) -> Dict[str, float]:
        """Compute best (minimum) residuals achieved."""
        if not self.history:
            return {}
        
        best_residuals = {}
        for var_name in self.config.monitor_variables:
            var_residuals = [data.residuals.get(var_name, float('inf')) for data in self.history]
            best_residuals[var_name] = min(var_residuals)
        
        return best_residuals
    
    def _compute_worst_residuals(self) -> Dict[str, float]:
        """Compute worst (maximum) residuals encountered."""
        if not self.history:
            return {}
        
        worst_residuals = {}
        for var_name in self.config.monitor_variables:
            var_residuals = [data.residuals.get(var_name, 0.0) for data in self.history]
            worst_residuals[var_name] = max(var_residuals)
        
        return worst_residuals
    
    def reset(self):
        """Reset convergence monitor for new simulation."""
        self.history.clear()
        self.initial_residuals = None
        self.converged_variables.clear()
        self.stopping_criteria_met.clear()
        self.start_time = time.time()
        self.last_log_iteration = 0
        self.last_save_iteration = 0
        self.metrics = ConvergenceMetrics()


def create_convergence_monitor(variable_names: List[str],
                             absolute_tolerance: float = 1e-6,
                             relative_tolerance: float = 1e-3,
                             max_iterations: int = 10000,
                             config: Optional[ConvergenceConfig] = None) -> ConvergenceMonitor:
    """
    Factory function for creating convergence monitors.
    
    Args:
        variable_names: Names of variables to monitor
        absolute_tolerance: Absolute convergence tolerance
        relative_tolerance: Relative convergence tolerance
        max_iterations: Maximum number of iterations
        config: Optional detailed configuration
        
    Returns:
        Configured convergence monitor
    """
    if config is None:
        config = ConvergenceConfig(
            variable_names=variable_names,
            monitor_variables=variable_names,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations
        )
    
    return ConvergenceMonitor(config)


def test_convergence_monitoring():
    """Test convergence monitoring functionality."""
    print("Testing Convergence Monitoring:")
    
    # Create test configuration
    variable_names = ["rho", "rho_u", "rho_v", "rho_w", "rho_E"]
    config = ConvergenceConfig(
        variable_names=variable_names,
        monitor_variables=variable_names,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-3,
        max_iterations=1000,
        log_frequency=50
    )
    
    # Create convergence monitor
    monitor = create_convergence_monitor(variable_names, config=config)
    
    print(f"  Monitor created for variables: {', '.join(variable_names)}")
    
    # Simulate convergence behavior
    print(f"\n  Simulating convergence process:")
    
    n_cells = 1000
    n_vars = len(variable_names)
    
    # Generate synthetic residual history
    np.random.seed(42)  # For reproducible results
    
    for iteration in range(200):
        # Create synthetic residual vector with decreasing trend
        base_residual = 1.0 * np.exp(-iteration * 0.05)  # Exponential decay
        noise = 0.1 * base_residual * np.random.randn(n_cells * n_vars)
        
        # Add some variable-specific behavior
        residual_vector = np.zeros(n_cells * n_vars)
        for i in range(n_vars):
            start_idx = i * n_cells
            end_idx = (i + 1) * n_cells
            
            # Different convergence rates for different variables
            var_factor = 1.0 + i * 0.2  # Varies from 1.0 to 1.8
            var_residual = base_residual * var_factor + noise[start_idx:end_idx]
            residual_vector[start_idx:end_idx] = var_residual
        
        # Add some oscillations for testing
        if 50 < iteration < 80:
            oscillation = 0.3 * base_residual * np.sin(iteration * 0.5)
            residual_vector += oscillation
        
        # Update monitor
        iteration_time = 0.001 + np.random.rand() * 0.002  # 1-3 ms per iteration
        residual_data = monitor.update_residuals(iteration, residual_vector, 
                                               iteration_time=iteration_time)
        
        # Check stopping criteria
        should_stop, reasons = monitor.check_stopping_criteria()
        if should_stop:
            print(f"    Stopping at iteration {iteration}: {', '.join(reasons)}")
            break
    
    # Get convergence report
    report = monitor.get_convergence_report()
    
    print(f"\n  Convergence Report:")
    print(f"    Status: {report['convergence_summary']['status']}")
    print(f"    Converged: {report['convergence_summary']['converged']}")
    print(f"    Total iterations: {report['performance']['total_iterations']}")
    print(f"    Total time: {report['performance']['total_time']:.3f}s")
    print(f"    Iterations per second: {report['performance']['iterations_per_second']:.1f}")
    
    print(f"\n  Final residual ratios:")
    for var, ratio in report['convergence_summary']['residual_ratios'].items():
        print(f"    {var}: {ratio:.2e}")
    
    if 'trend_analysis' in report:
        print(f"\n  Trend analysis:")
        for var, analysis in report['trend_analysis'].items():
            if isinstance(analysis, dict) and 'trend' in analysis:
                print(f"    {var}: {analysis['trend']} (rate: {analysis.get('convergence_rate', 'N/A')})")
    
    # Test residual computer directly
    print(f"\n  Testing residual computer:")
    residual_computer = ResidualComputer(config)
    
    test_residual = np.random.rand(n_cells * n_vars) * 0.1
    residuals = residual_computer.compute_residuals(test_residual)
    norms = residual_computer.compute_multiple_norms(test_residual)
    
    print(f"    Variable residuals: {residuals}")
    print(f"    Different norms: L2={norms['l2']:.2e}, L1={norms['l1']:.2e}, Linf={norms['linf']:.2e}")
    
    # Test analyzer
    print(f"\n  Testing convergence analyzer:")
    analyzer = ConvergenceAnalyzer(config)
    
    history_list = list(monitor.history)
    trend_analysis = analyzer.analyze_convergence_trend(history_list)
    stagnation = analyzer.detect_stagnation(history_list)
    divergence = analyzer.detect_divergence(history_list)
    oscillations = analyzer.detect_oscillations(history_list)
    
    print(f"    Stagnation detected: {stagnation}")
    print(f"    Divergence detected: {divergence}")
    print(f"    Oscillations detected: {any(oscillations.values())}")
    
    print(f"\n  Convergence monitoring test completed!")


if __name__ == "__main__":
    test_convergence_monitoring()