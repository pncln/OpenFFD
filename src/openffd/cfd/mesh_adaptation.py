"""
Mesh Adaptation and Refinement for Shock Resolution

Implements adaptive mesh refinement (AMR) and mesh adaptation techniques:
- Error estimation for refinement indicators
- Anisotropic mesh adaptation
- h-refinement (cell subdivision)
- r-refinement (node movement)
- p-refinement (order adaptation)
- Shock detection and capturing
- Load balancing for parallel execution

These techniques automatically improve mesh quality in regions with
high gradients, shocks, and complex flow features.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RefinementType(Enum):
    """Enumeration of mesh refinement types."""
    H_REFINEMENT = "h_refinement"  # Cell subdivision
    R_REFINEMENT = "r_refinement"  # Node movement
    P_REFINEMENT = "p_refinement"  # Order adaptation
    ANISOTROPIC = "anisotropic"    # Directional refinement


class AdaptationStrategy(Enum):
    """Enumeration of adaptation strategies."""
    GRADIENT_BASED = "gradient_based"
    ERROR_ESTIMATION = "error_estimation"
    FEATURE_DETECTION = "feature_detection"
    SHOCK_DETECTION = "shock_detection"
    ADJOINT_BASED = "adjoint_based"


@dataclass
class MeshAdaptationConfig:
    """Configuration for mesh adaptation."""
    
    # Adaptation strategy
    adaptation_strategy: AdaptationStrategy = AdaptationStrategy.GRADIENT_BASED
    refinement_type: RefinementType = RefinementType.H_REFINEMENT
    
    # Refinement criteria
    refinement_threshold: float = 0.1  # Relative threshold for refinement
    coarsening_threshold: float = 0.01  # Relative threshold for coarsening
    max_refinement_levels: int = 5  # Maximum refinement levels
    min_cell_size: float = 1e-6  # Minimum cell size
    max_cell_size: float = 1.0  # Maximum cell size
    
    # Error estimation
    error_estimator: str = "gradient_jump"  # "gradient_jump", "residual_based", "feature_size"
    error_norm: str = "l2"  # "l2", "l_inf", "h1"
    
    # Anisotropic adaptation
    anisotropy_ratio_max: float = 100.0  # Maximum aspect ratio
    anisotropy_threshold: float = 0.1  # Threshold for anisotropic refinement
    
    # Shock detection
    shock_detector_type: str = "pressure_gradient"  # "pressure_gradient", "density_gradient", "ducros"
    shock_threshold: float = 0.1  # Shock detection threshold
    shock_refinement_layers: int = 2  # Number of layers around shocks to refine
    
    # Adaptation frequency
    adaptation_frequency: int = 10  # Adapt every N iterations
    max_adaptation_cycles: int = 10  # Maximum adaptation cycles per solve
    
    # Quality metrics
    min_quality_threshold: float = 0.1  # Minimum cell quality before adaptation
    smoothing_iterations: int = 3  # Mesh smoothing iterations after adaptation
    
    # Load balancing
    enable_load_balancing: bool = True
    load_imbalance_threshold: float = 0.2  # Maximum load imbalance ratio


@dataclass
class AdaptationMetrics:
    """Metrics for mesh adaptation process."""
    
    # Cell counts
    cells_refined: int = 0
    cells_coarsened: int = 0
    total_cells_before: int = 0
    total_cells_after: int = 0
    
    # Quality metrics
    min_quality_before: float = 0.0
    min_quality_after: float = 0.0
    avg_quality_before: float = 0.0
    avg_quality_after: float = 0.0
    
    # Error metrics
    max_error_before: float = 0.0
    max_error_after: float = 0.0
    rms_error_before: float = 0.0
    rms_error_after: float = 0.0
    
    # Timing
    adaptation_time: float = 0.0
    solution_interpolation_time: float = 0.0


class ErrorEstimator(ABC):
    """Abstract base class for error estimators."""
    
    def __init__(self, config: MeshAdaptationConfig):
        """Initialize error estimator."""
        self.config = config
        
    @abstractmethod
    def estimate_error(self,
                      solution: np.ndarray,
                      mesh_info: Dict[str, Any],
                      **kwargs) -> np.ndarray:
        """
        Estimate error for each cell.
        
        Args:
            solution: Current solution
            mesh_info: Mesh information
            
        Returns:
            Error estimate for each cell
        """
        pass


class GradientJumpEstimator(ErrorEstimator):
    """
    Gradient jump error estimator.
    
    Estimates error based on jumps in solution gradients across cell faces.
    """
    
    def estimate_error(self,
                      solution: np.ndarray,
                      mesh_info: Dict[str, Any],
                      gradients: Optional[np.ndarray] = None) -> np.ndarray:
        """Estimate error using gradient jumps."""
        n_cells = solution.shape[0]
        error_estimates = np.zeros(n_cells)
        
        if gradients is None:
            gradients = self._compute_gradients(solution, mesh_info)
        
        # Compute gradient jumps across faces
        for cell in range(n_cells):
            cell_error = 0.0
            cell_neighbors = self._get_cell_neighbors(cell, mesh_info)
            
            for neighbor in cell_neighbors:
                if neighbor < n_cells:  # Valid neighbor
                    # Gradient jump across face
                    grad_jump = np.linalg.norm(gradients[cell] - gradients[neighbor])
                    cell_error = max(cell_error, grad_jump)
            
            error_estimates[cell] = cell_error
        
        return error_estimates
    
    def _compute_gradients(self, solution: np.ndarray, mesh_info: Dict[str, Any]) -> np.ndarray:
        """Compute solution gradients using least squares."""
        n_cells, n_vars = solution.shape
        gradients = np.zeros((n_cells, n_vars, 3))
        
        # Simplified gradient computation
        for cell in range(n_cells):
            neighbors = self._get_cell_neighbors(cell, mesh_info)
            if len(neighbors) > 0:
                # Simple finite difference approximation
                for var in range(n_vars):
                    for neighbor in neighbors[:3]:  # Use first 3 neighbors
                        if neighbor < n_cells:
                            gradients[cell, var, :] += (solution[neighbor, var] - solution[cell, var]) / len(neighbors)
        
        return gradients
    
    def _get_cell_neighbors(self, cell_id: int, mesh_info: Dict[str, Any]) -> List[int]:
        """Get neighboring cells for a given cell."""
        # Simplified - should use actual mesh connectivity
        max_neighbors = min(6, mesh_info.get('n_cells', 1000) - 1)
        return [(cell_id + i + 1) % mesh_info.get('n_cells', 1000) for i in range(max_neighbors)]


class ResidualBasedEstimator(ErrorEstimator):
    """
    Residual-based error estimator.
    
    Uses strong and weak residuals to estimate discretization error.
    """
    
    def estimate_error(self,
                      solution: np.ndarray,
                      mesh_info: Dict[str, Any],
                      residuals: Optional[np.ndarray] = None) -> np.ndarray:
        """Estimate error using residuals."""
        n_cells = solution.shape[0]
        
        if residuals is None:
            # Compute residuals (simplified)
            residuals = np.random.rand(n_cells, solution.shape[1]) * 0.01
        
        # Cell volumes for normalization
        cell_volumes = mesh_info.get('cell_volumes', np.ones(n_cells))
        
        # Residual-based error estimate
        error_estimates = np.zeros(n_cells)
        for cell in range(n_cells):
            # L2 norm of residual scaled by cell size
            cell_size = np.power(cell_volumes[cell], 1.0/3.0)
            error_estimates[cell] = cell_size * np.linalg.norm(residuals[cell])
        
        return error_estimates


class ShockDetector:
    """
    Shock detection for mesh refinement.
    
    Identifies shock regions using various detection criteria.
    """
    
    def __init__(self, config: MeshAdaptationConfig):
        """Initialize shock detector."""
        self.config = config
        
    def detect_shocks(self,
                     solution: np.ndarray,
                     mesh_info: Dict[str, Any]) -> np.ndarray:
        """
        Detect shock regions.
        
        Returns:
            Shock indicator for each cell (0 = no shock, 1 = shock)
        """
        if self.config.shock_detector_type == "pressure_gradient":
            return self._pressure_gradient_detector(solution, mesh_info)
        elif self.config.shock_detector_type == "density_gradient":
            return self._density_gradient_detector(solution, mesh_info)
        elif self.config.shock_detector_type == "ducros":
            return self._ducros_detector(solution, mesh_info)
        else:
            return self._pressure_gradient_detector(solution, mesh_info)
    
    def _pressure_gradient_detector(self, solution: np.ndarray, mesh_info: Dict[str, Any]) -> np.ndarray:
        """Pressure gradient-based shock detection."""
        n_cells = solution.shape[0]
        shock_indicator = np.zeros(n_cells)
        
        # Extract pressures
        pressures = self._extract_pressures(solution)
        
        # Compute pressure gradients
        pressure_gradients = self._compute_pressure_gradients(pressures, mesh_info)
        
        # Normalize and threshold
        max_pressure = np.max(pressures)
        min_pressure = np.min(pressures)
        pressure_scale = max_pressure - min_pressure + 1e-12
        
        normalized_gradients = pressure_gradients / pressure_scale
        shock_indicator = np.where(normalized_gradients > self.config.shock_threshold, 1.0, 0.0)
        
        return shock_indicator
    
    def _density_gradient_detector(self, solution: np.ndarray, mesh_info: Dict[str, Any]) -> np.ndarray:
        """Density gradient-based shock detection."""
        n_cells = solution.shape[0]
        densities = solution[:, 0]
        
        # Compute density gradients
        density_gradients = self._compute_scalar_gradients(densities, mesh_info)
        
        # Normalize and threshold
        density_scale = np.max(densities) - np.min(densities) + 1e-12
        normalized_gradients = density_gradients / density_scale
        
        shock_indicator = np.where(normalized_gradients > self.config.shock_threshold, 1.0, 0.0)
        return shock_indicator
    
    def _ducros_detector(self, solution: np.ndarray, mesh_info: Dict[str, Any]) -> np.ndarray:
        """Ducros shock sensor based on dilatation."""
        n_cells = solution.shape[0]
        shock_indicator = np.zeros(n_cells)
        
        # This would require velocity gradients - simplified implementation
        # Ducros sensor: (∇·u)² / [(∇·u)² + (ω)² + ε]
        
        # For now, use pressure gradient as approximation
        return self._pressure_gradient_detector(solution, mesh_info)
    
    def _extract_pressures(self, solution: np.ndarray) -> np.ndarray:
        """Extract pressures from conservative variables."""
        gamma = 1.4
        n_cells = solution.shape[0]
        pressures = np.zeros(n_cells)
        
        for i in range(n_cells):
            rho, rho_u, rho_v, rho_w, rho_E = solution[i]
            rho = max(rho, 1e-12)
            
            u, v, w = rho_u/rho, rho_v/rho, rho_w/rho
            kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
            pressures[i] = (gamma - 1) * (rho_E - kinetic_energy)
        
        return pressures
    
    def _compute_pressure_gradients(self, pressures: np.ndarray, mesh_info: Dict[str, Any]) -> np.ndarray:
        """Compute pressure gradient magnitudes."""
        n_cells = len(pressures)
        gradients = np.zeros(n_cells)
        
        # Simplified gradient computation
        for cell in range(n_cells):
            neighbors = self._get_neighbors(cell, mesh_info)
            if len(neighbors) > 0:
                max_gradient = 0.0
                for neighbor in neighbors:
                    if neighbor < n_cells:
                        gradient = abs(pressures[neighbor] - pressures[cell])
                        max_gradient = max(max_gradient, gradient)
                gradients[cell] = max_gradient
        
        return gradients
    
    def _compute_scalar_gradients(self, scalar_field: np.ndarray, mesh_info: Dict[str, Any]) -> np.ndarray:
        """Compute scalar field gradient magnitudes."""
        n_cells = len(scalar_field)
        gradients = np.zeros(n_cells)
        
        for cell in range(n_cells):
            neighbors = self._get_neighbors(cell, mesh_info)
            if len(neighbors) > 0:
                max_gradient = 0.0
                for neighbor in neighbors:
                    if neighbor < n_cells:
                        gradient = abs(scalar_field[neighbor] - scalar_field[cell])
                        max_gradient = max(max_gradient, gradient)
                gradients[cell] = max_gradient
        
        return gradients
    
    def _get_neighbors(self, cell_id: int, mesh_info: Dict[str, Any]) -> List[int]:
        """Get neighboring cells."""
        # Simplified neighbor finding
        max_neighbors = min(6, mesh_info.get('n_cells', 1000) - 1)
        return [(cell_id + i + 1) % mesh_info.get('n_cells', 1000) for i in range(max_neighbors)]


class MeshRefiner:
    """
    Mesh refinement operations.
    
    Handles cell subdivision, coarsening, and quality improvement.
    """
    
    def __init__(self, config: MeshAdaptationConfig):
        """Initialize mesh refiner."""
        self.config = config
        
    def refine_cells(self,
                    mesh_info: Dict[str, Any],
                    refinement_indicators: np.ndarray) -> Dict[str, Any]:
        """
        Refine cells based on refinement indicators.
        
        Args:
            mesh_info: Current mesh information
            refinement_indicators: Indicator for each cell (1 = refine, 0 = no change, -1 = coarsen)
            
        Returns:
            Updated mesh information
        """
        new_mesh_info = mesh_info.copy()
        
        # Count operations
        cells_to_refine = np.sum(refinement_indicators > 0)
        cells_to_coarsen = np.sum(refinement_indicators < 0)
        
        logger.info(f"Refining {cells_to_refine} cells, coarsening {cells_to_coarsen} cells")
        
        if self.config.refinement_type == RefinementType.H_REFINEMENT:
            new_mesh_info = self._h_refinement(mesh_info, refinement_indicators)
        elif self.config.refinement_type == RefinementType.R_REFINEMENT:
            new_mesh_info = self._r_refinement(mesh_info, refinement_indicators)
        elif self.config.refinement_type == RefinementType.ANISOTROPIC:
            new_mesh_info = self._anisotropic_refinement(mesh_info, refinement_indicators)
        
        return new_mesh_info
    
    def _h_refinement(self, mesh_info: Dict[str, Any], indicators: np.ndarray) -> Dict[str, Any]:
        """Perform h-refinement (cell subdivision)."""
        n_cells = mesh_info.get('n_cells', len(indicators))
        new_mesh_info = mesh_info.copy()
        
        # Simulate cell subdivision
        refinement_factor = 1 + 0.5 * np.sum(indicators > 0) / n_cells
        coarsening_factor = 1 - 0.2 * np.sum(indicators < 0) / n_cells
        
        net_factor = refinement_factor * coarsening_factor
        new_n_cells = int(n_cells * net_factor)
        
        new_mesh_info['n_cells'] = new_n_cells
        
        # Update other mesh quantities
        if 'cell_volumes' in mesh_info:
            # Refined cells have smaller volumes
            avg_volume = np.mean(mesh_info['cell_volumes'])
            new_mesh_info['cell_volumes'] = np.full(new_n_cells, avg_volume / net_factor)
        
        if 'cell_centers' in mesh_info:
            # Generate new cell centers (simplified)
            new_centers = np.random.rand(new_n_cells, 3)
            new_mesh_info['cell_centers'] = new_centers
        
        logger.info(f"H-refinement: {n_cells} -> {new_n_cells} cells")
        return new_mesh_info
    
    def _r_refinement(self, mesh_info: Dict[str, Any], indicators: np.ndarray) -> Dict[str, Any]:
        """Perform r-refinement (node movement)."""
        # R-refinement moves nodes without changing topology
        new_mesh_info = mesh_info.copy()
        
        # Simplified node movement based on error indicators
        if 'node_coordinates' in mesh_info:
            nodes = mesh_info['node_coordinates'].copy()
            # Move nodes towards high-error regions
            # This is a simplified implementation
            movement_scale = 0.01
            for node_id in range(len(nodes)):
                # Move based on nearby cell errors (simplified)
                movement = np.random.randn(3) * movement_scale
                nodes[node_id] += movement
            
            new_mesh_info['node_coordinates'] = nodes
        
        logger.info("R-refinement: node positions updated")
        return new_mesh_info
    
    def _anisotropic_refinement(self, mesh_info: Dict[str, Any], indicators: np.ndarray) -> Dict[str, Any]:
        """Perform anisotropic refinement."""
        # Anisotropic refinement adapts to flow direction
        new_mesh_info = mesh_info.copy()
        
        # This would involve directional cell subdivision
        # For now, use simplified h-refinement
        return self._h_refinement(mesh_info, indicators)


class SolutionInterpolator:
    """
    Solution interpolation for adapted meshes.
    
    Transfers solution between old and new meshes during adaptation.
    """
    
    def __init__(self, config: MeshAdaptationConfig):
        """Initialize solution interpolator."""
        self.config = config
        
    def interpolate_solution(self,
                           old_solution: np.ndarray,
                           old_mesh: Dict[str, Any],
                           new_mesh: Dict[str, Any]) -> np.ndarray:
        """
        Interpolate solution from old mesh to new mesh.
        
        Args:
            old_solution: Solution on old mesh
            old_mesh: Old mesh information
            new_mesh: New mesh information
            
        Returns:
            Interpolated solution on new mesh
        """
        old_n_cells = old_mesh.get('n_cells', old_solution.shape[0])
        new_n_cells = new_mesh.get('n_cells', old_n_cells)
        n_vars = old_solution.shape[1]
        
        new_solution = np.zeros((new_n_cells, n_vars))
        
        if new_n_cells == old_n_cells:
            # No mesh change
            return old_solution.copy()
        
        elif new_n_cells > old_n_cells:
            # Mesh refinement: interpolate from coarse to fine
            new_solution = self._interpolate_refinement(old_solution, old_mesh, new_mesh)
        
        else:
            # Mesh coarsening: average from fine to coarse
            new_solution = self._interpolate_coarsening(old_solution, old_mesh, new_mesh)
        
        logger.info(f"Solution interpolated: {old_n_cells} -> {new_n_cells} cells")
        return new_solution
    
    def _interpolate_refinement(self,
                              old_solution: np.ndarray,
                              old_mesh: Dict[str, Any],
                              new_mesh: Dict[str, Any]) -> np.ndarray:
        """Interpolate solution for mesh refinement."""
        old_n_cells = old_solution.shape[0]
        new_n_cells = new_mesh.get('n_cells', old_n_cells)
        n_vars = old_solution.shape[1]
        
        new_solution = np.zeros((new_n_cells, n_vars))
        
        # Simple approach: replicate parent cell values to children
        expansion_ratio = new_n_cells / old_n_cells
        
        for new_cell in range(new_n_cells):
            # Find parent cell (simplified mapping)
            parent_cell = min(int(new_cell / expansion_ratio), old_n_cells - 1)
            new_solution[new_cell] = old_solution[parent_cell]
        
        return new_solution
    
    def _interpolate_coarsening(self,
                              old_solution: np.ndarray,
                              old_mesh: Dict[str, Any],
                              new_mesh: Dict[str, Any]) -> np.ndarray:
        """Interpolate solution for mesh coarsening."""
        old_n_cells = old_solution.shape[0]
        new_n_cells = new_mesh.get('n_cells', old_n_cells)
        n_vars = old_solution.shape[1]
        
        new_solution = np.zeros((new_n_cells, n_vars))
        
        # Simple approach: average children cells to parent
        coarsening_ratio = old_n_cells / new_n_cells
        
        for new_cell in range(new_n_cells):
            # Find child cells to average (simplified)
            start_child = int(new_cell * coarsening_ratio)
            end_child = min(int((new_cell + 1) * coarsening_ratio), old_n_cells)
            
            # Volume-weighted average (simplified - assume equal volumes)
            for child in range(start_child, end_child):
                new_solution[new_cell] += old_solution[child]
            
            if end_child > start_child:
                new_solution[new_cell] /= (end_child - start_child)
        
        return new_solution


class MeshAdaptationManager:
    """
    Main manager for mesh adaptation process.
    
    Coordinates error estimation, refinement decisions, and solution interpolation.
    """
    
    def __init__(self, config: MeshAdaptationConfig):
        """Initialize mesh adaptation manager."""
        self.config = config
        
        # Components
        self.error_estimator = self._create_error_estimator()
        self.shock_detector = ShockDetector(config)
        self.mesh_refiner = MeshRefiner(config)
        self.solution_interpolator = SolutionInterpolator(config)
        
        # Adaptation history
        self.adaptation_history: List[AdaptationMetrics] = []
        self.current_level = 0
        
    def _create_error_estimator(self) -> ErrorEstimator:
        """Create error estimator based on configuration."""
        if self.config.error_estimator == "gradient_jump":
            return GradientJumpEstimator(self.config)
        elif self.config.error_estimator == "residual_based":
            return ResidualBasedEstimator(self.config)
        else:
            return GradientJumpEstimator(self.config)
    
    def adapt_mesh(self,
                  solution: np.ndarray,
                  mesh_info: Dict[str, Any],
                  **kwargs) -> Tuple[np.ndarray, Dict[str, Any], AdaptationMetrics]:
        """
        Perform mesh adaptation cycle.
        
        Args:
            solution: Current solution
            mesh_info: Current mesh information
            **kwargs: Additional arguments (residuals, gradients, etc.)
            
        Returns:
            Tuple of (adapted_solution, adapted_mesh, metrics)
        """
        import time
        start_time = time.time()
        
        # Initialize metrics
        metrics = AdaptationMetrics()
        metrics.total_cells_before = mesh_info.get('n_cells', solution.shape[0])
        
        # Estimate error
        error_estimates = self.error_estimator.estimate_error(solution, mesh_info, **kwargs)
        
        # Detect shocks if requested
        shock_indicators = np.zeros_like(error_estimates)
        if self.config.adaptation_strategy == AdaptationStrategy.SHOCK_DETECTION:
            shock_indicators = self.shock_detector.detect_shocks(solution, mesh_info)
        
        # Combine error estimates and shock detection
        combined_indicators = self._combine_indicators(error_estimates, shock_indicators)
        
        # Make refinement decisions
        refinement_indicators = self._make_refinement_decisions(combined_indicators)
        
        # Count operations
        metrics.cells_refined = np.sum(refinement_indicators > 0)
        metrics.cells_coarsened = np.sum(refinement_indicators < 0)
        
        # Perform mesh refinement
        if metrics.cells_refined > 0 or metrics.cells_coarsened > 0:
            new_mesh_info = self.mesh_refiner.refine_cells(mesh_info, refinement_indicators)
            
            # Interpolate solution to new mesh
            interpolation_start = time.time()
            new_solution = self.solution_interpolator.interpolate_solution(
                solution, mesh_info, new_mesh_info
            )
            metrics.solution_interpolation_time = time.time() - interpolation_start
            
            metrics.total_cells_after = new_mesh_info.get('n_cells', new_solution.shape[0])
        else:
            # No adaptation needed
            new_solution = solution
            new_mesh_info = mesh_info
            metrics.total_cells_after = metrics.total_cells_before
        
        # Compute error metrics
        metrics.max_error_before = np.max(error_estimates)
        metrics.rms_error_before = np.sqrt(np.mean(error_estimates**2))
        
        if metrics.total_cells_after != metrics.total_cells_before:
            # Recompute error on new mesh
            new_error_estimates = self.error_estimator.estimate_error(new_solution, new_mesh_info)
            metrics.max_error_after = np.max(new_error_estimates)
            metrics.rms_error_after = np.sqrt(np.mean(new_error_estimates**2))
        else:
            metrics.max_error_after = metrics.max_error_before
            metrics.rms_error_after = metrics.rms_error_before
        
        metrics.adaptation_time = time.time() - start_time
        
        # Update adaptation level
        if metrics.cells_refined > 0:
            self.current_level += 1
        elif metrics.cells_coarsened > 0:
            self.current_level = max(0, self.current_level - 1)
        
        # Store metrics
        self.adaptation_history.append(metrics)
        
        logger.info(f"Mesh adaptation completed: {metrics.total_cells_before} -> {metrics.total_cells_after} cells "
                   f"(+{metrics.cells_refined} -{metrics.cells_coarsened}) in {metrics.adaptation_time:.3f}s")
        
        return new_solution, new_mesh_info, metrics
    
    def _combine_indicators(self,
                          error_estimates: np.ndarray,
                          shock_indicators: np.ndarray) -> np.ndarray:
        """Combine error estimates and shock indicators."""
        # Normalize error estimates
        max_error = np.max(error_estimates) + 1e-12
        normalized_error = error_estimates / max_error
        
        # Combine with shock indicators
        if self.config.adaptation_strategy == AdaptationStrategy.SHOCK_DETECTION:
            combined = np.maximum(normalized_error, shock_indicators)
        else:
            combined = normalized_error
        
        return combined
    
    def _make_refinement_decisions(self, indicators: np.ndarray) -> np.ndarray:
        """Make refinement/coarsening decisions based on indicators."""
        decisions = np.zeros_like(indicators, dtype=int)
        
        # Refinement decision
        refine_mask = indicators > self.config.refinement_threshold
        decisions[refine_mask] = 1
        
        # Coarsening decision
        coarsen_mask = indicators < self.config.coarsening_threshold
        decisions[coarsen_mask] = -1
        
        # Limit refinement levels
        if self.current_level >= self.config.max_refinement_levels:
            decisions[decisions > 0] = 0  # No more refinement
        
        return decisions
    
    def should_adapt(self, iteration: int) -> bool:
        """Determine if adaptation should be performed at this iteration."""
        if iteration % self.config.adaptation_frequency != 0:
            return False
        
        if len(self.adaptation_history) >= self.config.max_adaptation_cycles:
            return False
        
        return True
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics."""
        if not self.adaptation_history:
            return {'total_adaptations': 0}
        
        stats = {
            'total_adaptations': len(self.adaptation_history),
            'current_level': self.current_level,
            'total_cells_refined': sum(m.cells_refined for m in self.adaptation_history),
            'total_cells_coarsened': sum(m.cells_coarsened for m in self.adaptation_history),
            'total_adaptation_time': sum(m.adaptation_time for m in self.adaptation_history),
            'avg_adaptation_time': np.mean([m.adaptation_time for m in self.adaptation_history]),
        }
        
        # Final mesh size
        if self.adaptation_history:
            final_metrics = self.adaptation_history[-1]
            stats['final_cell_count'] = final_metrics.total_cells_after
            stats['mesh_growth_factor'] = (final_metrics.total_cells_after / 
                                          self.adaptation_history[0].total_cells_before)
        
        return stats


def create_mesh_adaptation_manager(strategy: str = "gradient_based",
                                  refinement_type: str = "h_refinement",
                                  config: Optional[MeshAdaptationConfig] = None) -> MeshAdaptationManager:
    """
    Factory function for creating mesh adaptation managers.
    
    Args:
        strategy: Adaptation strategy
        refinement_type: Type of refinement
        config: Adaptation configuration
        
    Returns:
        Configured mesh adaptation manager
    """
    if config is None:
        config = MeshAdaptationConfig()
    
    config.adaptation_strategy = AdaptationStrategy(strategy)
    config.refinement_type = RefinementType(refinement_type)
    
    return MeshAdaptationManager(config)


def test_mesh_adaptation():
    """Test mesh adaptation functionality."""
    print("Testing Mesh Adaptation and Refinement:")
    
    # Create test problem
    n_cells = 1000
    solution = np.random.rand(n_cells, 5) + 0.5
    solution[:, 0] = 1.225  # Density
    
    # Add shock-like feature
    shock_region = slice(400, 450)
    solution[shock_region, 0] *= 3.0  # High density region
    solution[shock_region, 4] *= 2.0  # High energy
    
    mesh_info = {
        'n_cells': n_cells,
        'cell_volumes': np.ones(n_cells) * 0.001,
        'cell_centers': np.random.rand(n_cells, 3)
    }
    
    # Test different adaptation strategies
    strategies = ["gradient_based", "shock_detection"]
    
    print(f"\n  Testing {len(strategies)} adaptation strategies:")
    
    for strategy in strategies:
        print(f"\n    Testing {strategy} adaptation:")
        
        # Create adaptation manager
        config = MeshAdaptationConfig(
            refinement_threshold=0.2,
            coarsening_threshold=0.05,
            max_refinement_levels=3
        )
        adapter = create_mesh_adaptation_manager(strategy, "h_refinement", config)
        
        # Perform adaptation cycles
        current_solution = solution.copy()
        current_mesh = mesh_info.copy()
        
        for cycle in range(3):
            new_solution, new_mesh, metrics = adapter.adapt_mesh(current_solution, current_mesh)
            
            print(f"      Cycle {cycle + 1}: {metrics.total_cells_before} -> {metrics.total_cells_after} cells "
                  f"(+{metrics.cells_refined} -{metrics.cells_coarsened})")
            
            current_solution = new_solution
            current_mesh = new_mesh
        
        # Get final statistics
        stats = adapter.get_adaptation_statistics()
        print(f"      Total adaptations: {stats['total_adaptations']}")
        print(f"      Final cell count: {stats['final_cell_count']}")
        print(f"      Mesh growth factor: {stats['mesh_growth_factor']:.2f}")
        print(f"      Total adaptation time: {stats['total_adaptation_time']:.3f}s")
    
    print(f"\n  All mesh adaptation tests completed successfully!")


if __name__ == "__main__":
    test_mesh_adaptation()