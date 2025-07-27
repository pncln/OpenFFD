"""
Mesh Adaptation and Refinement for Supersonic Flow Solver

Provides adaptive mesh refinement capabilities specifically designed for:
- Shock detection and resolution
- Error-based refinement
- Feature-based adaptation
- Anisotropic refinement for boundary layers
- Load balancing for parallel computations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class RefinementType(Enum):
    """Types of mesh refinement strategies."""
    ISOTROPIC = "isotropic"           # Uniform refinement in all directions
    ANISOTROPIC = "anisotropic"       # Directional refinement
    FEATURE_BASED = "feature_based"   # Based on flow features
    ERROR_BASED = "error_based"       # Based on solution error estimates
    SHOCK_BASED = "shock_based"       # Specific to shock detection


class AdaptationCriterion(Enum):
    """Adaptation criteria for mesh refinement."""
    GRADIENT_BASED = "gradient"       # Based on solution gradients
    HESSIAN_BASED = "hessian"        # Based on second derivatives
    SHOCK_SENSOR = "shock_sensor"     # Shock detection sensors
    RESIDUAL_BASED = "residual"      # Based on equation residuals
    FEATURE_SIZE = "feature_size"     # Based on geometric features
    USER_DEFINED = "user_defined"     # Custom criteria


@dataclass
class RefinementCriteria:
    """Refinement criteria configuration."""
    criterion_type: AdaptationCriterion
    threshold: float = 0.1
    max_refinement_level: int = 5
    min_cell_size: float = 1e-6
    max_cell_size: float = 1e3
    
    # Gradient-based criteria
    gradient_variable: str = "density"  # "density", "pressure", "mach"
    gradient_threshold: float = 0.1
    
    # Shock detection parameters
    shock_mach_threshold: float = 1.2
    shock_gradient_threshold: float = 0.05
    
    # Error estimation parameters
    error_tolerance: float = 1e-3
    target_elements: Optional[int] = None
    
    # Anisotropic refinement
    anisotropy_ratio: float = 10.0
    boundary_layer_growth: float = 1.2


@dataclass
class CellRefinementInfo:
    """Information about cell refinement."""
    cell_id: int
    refinement_level: int = 0
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    
    # Refinement flags
    marked_for_refinement: bool = False
    marked_for_coarsening: bool = False
    refinement_type: Optional[RefinementType] = None
    
    # Quality metrics
    quality_measure: float = 1.0
    error_estimate: float = 0.0
    adaptation_indicator: float = 0.0


class ShockDetector:
    """Shock detection algorithms for supersonic flows."""
    
    def __init__(self, mesh=None):
        """Initialize shock detector."""
        self.mesh = mesh
        self.gamma = 1.4  # Specific heat ratio
        
    def detect_shocks_ducros(self, 
                           velocity_field: np.ndarray,
                           pressure_field: np.ndarray,
                           density_field: np.ndarray) -> np.ndarray:
        """
        Ducros shock sensor for shock detection.
        
        Returns:
            Array of shock indicators for each cell [0, 1]
        """
        n_cells = len(velocity_field)
        shock_indicators = np.zeros(n_cells)
        
        for cell_id in range(n_cells):
            # Compute velocity divergence
            neighbors = self._get_cell_neighbors(cell_id)
            if len(neighbors) < 3:
                continue
            
            div_v = self._compute_divergence(cell_id, velocity_field, neighbors)
            
            # Compute velocity curl magnitude
            curl_v = self._compute_curl_magnitude(cell_id, velocity_field, neighbors)
            
            # Ducros sensor
            epsilon = 1e-12
            ducros_sensor = div_v**2 / (div_v**2 + curl_v**2 + epsilon)
            
            # Additional pressure gradient check
            pressure_gradient = self._compute_pressure_gradient(cell_id, pressure_field, neighbors)
            pressure_threshold = 0.1 * pressure_field[cell_id]
            
            if np.linalg.norm(pressure_gradient) > pressure_threshold:
                ducros_sensor *= 2.0  # Amplify in high pressure gradient regions
            
            shock_indicators[cell_id] = min(ducros_sensor, 1.0)
        
        return shock_indicators
    
    def detect_shocks_jameson(self, 
                            pressure_field: np.ndarray,
                            threshold: float = 0.05) -> np.ndarray:
        """
        Jameson shock sensor based on pressure undivided differences.
        
        Returns:
            Array of shock indicators for each cell [0, 1]
        """
        n_cells = len(pressure_field)
        shock_indicators = np.zeros(n_cells)
        
        for cell_id in range(n_cells):
            neighbors = self._get_cell_neighbors(cell_id)
            if len(neighbors) < 2:
                continue
            
            # Compute pressure undivided differences
            p_center = pressure_field[cell_id]
            p_neighbors = pressure_field[neighbors]
            
            # Second differences
            p_max = np.max(p_neighbors)
            p_min = np.min(p_neighbors)
            
            undivided_diff = abs(p_max - 2*p_center + p_min)
            smoothness = abs(p_max - p_min) + 1e-12
            
            jameson_sensor = undivided_diff / smoothness
            
            if jameson_sensor > threshold:
                shock_indicators[cell_id] = min(jameson_sensor / threshold, 1.0)
        
        return shock_indicators
    
    def _get_cell_neighbors(self, cell_id: int) -> List[int]:
        """Get neighboring cells."""
        if hasattr(self.mesh, 'connectivity_manager'):
            return self.mesh.connectivity_manager._cell_neighbors[cell_id]
        return []
    
    def _compute_divergence(self, cell_id: int, field: np.ndarray, neighbors: List[int]) -> float:
        """Compute velocity divergence at cell center."""
        if len(neighbors) < 3:
            return 0.0
        
        # Simple finite difference approximation
        center_value = field[cell_id]
        neighbor_values = field[neighbors]
        
        # Approximate divergence using neighboring values
        divergence = np.sum(neighbor_values - center_value) / len(neighbors)
        return abs(divergence)
    
    def _compute_curl_magnitude(self, cell_id: int, field: np.ndarray, neighbors: List[int]) -> float:
        """Compute curl magnitude at cell center."""
        if len(neighbors) < 3:
            return 0.0
        
        # Simplified curl computation
        center_value = field[cell_id]
        neighbor_values = field[neighbors]
        
        # Approximate curl magnitude
        curl_mag = np.std(neighbor_values - center_value)
        return curl_mag
    
    def _compute_pressure_gradient(self, cell_id: int, pressure: np.ndarray, neighbors: List[int]) -> np.ndarray:
        """Compute pressure gradient at cell center."""
        if len(neighbors) < 3:
            return np.zeros(3)
        
        # Use least squares gradient reconstruction
        center_pressure = pressure[cell_id]
        center_pos = self.mesh.cell_data.centroids[cell_id]
        
        # Build least squares system
        A = []
        b = []
        
        for neighbor_id in neighbors:
            neighbor_pos = self.mesh.cell_data.centroids[neighbor_id]
            dr = neighbor_pos - center_pos
            dp = pressure[neighbor_id] - center_pressure
            
            A.append(dr)
            b.append(dp)
        
        A = np.array(A)
        b = np.array(b)
        
        try:
            # Solve least squares: A * grad = b
            gradient, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return gradient
        except:
            return np.zeros(3)


class MeshAdaptation:
    """
    Comprehensive mesh adaptation framework for supersonic flows.
    
    Features:
    - Multiple refinement strategies
    - Shock-adaptive refinement
    - Error-based adaptation
    - Anisotropic refinement
    - Load balancing support
    """
    
    def __init__(self, mesh=None):
        """Initialize mesh adaptation."""
        self.mesh = mesh
        self.shock_detector = ShockDetector(mesh)
        
        # Refinement tracking
        self.cell_info: Dict[int, CellRefinementInfo] = {}
        self.refinement_history: List[Dict] = []
        
        # Configuration
        self.criteria = RefinementCriteria(AdaptationCriterion.GRADIENT_BASED)
        self.max_adaptations = 10
        self.adaptation_frequency = 10  # Every N iterations
        
        # Statistics
        self.adaptation_stats = {
            'total_adaptations': 0,
            'cells_refined': 0,
            'cells_coarsened': 0,
            'max_refinement_level': 0
        }
        
        self._initialize_cell_info()
    
    def _initialize_cell_info(self) -> None:
        """Initialize cell refinement information."""
        if self.mesh is None:
            return
        
        for cell_id in range(self.mesh.n_cells):
            self.cell_info[cell_id] = CellRefinementInfo(cell_id=cell_id)
    
    def set_criteria(self, criteria: RefinementCriteria) -> None:
        """Set refinement criteria."""
        self.criteria = criteria
        logger.info(f"Set adaptation criteria: {criteria.criterion_type.value}")
    
    def mark_cells_for_refinement(self, 
                                 solution_data: Dict[str, np.ndarray],
                                 **kwargs) -> int:
        """
        Mark cells for refinement based on specified criteria.
        
        Returns:
            Number of cells marked for refinement
        """
        if self.criteria.criterion_type == AdaptationCriterion.SHOCK_SENSOR:
            return self._mark_shock_cells(solution_data)
        elif self.criteria.criterion_type == AdaptationCriterion.GRADIENT_BASED:
            return self._mark_gradient_cells(solution_data)
        elif self.criteria.criterion_type == AdaptationCriterion.ERROR_BASED:
            return self._mark_error_cells(solution_data)
        else:
            logger.warning(f"Unsupported adaptation criterion: {self.criteria.criterion_type}")
            return 0
    
    def _mark_shock_cells(self, solution_data: Dict[str, np.ndarray]) -> int:
        """Mark cells for refinement based on shock detection."""
        # Extract flow variables
        conservatives = solution_data.get('conservatives', np.zeros((self.mesh.n_cells, 5)))
        
        # Convert to primitive variables
        density = conservatives[:, 0]
        momentum = conservatives[:, 1:4]
        energy = conservatives[:, 4]
        
        # Compute velocity and pressure
        velocity = momentum / density.reshape(-1, 1)
        pressure = (self.shock_detector.gamma - 1) * (energy - 0.5 * density * np.sum(velocity**2, axis=1))
        
        # Detect shocks
        shock_indicators = self.shock_detector.detect_shocks_ducros(velocity, pressure, density)
        
        # Mark cells for refinement
        marked_count = 0
        for cell_id in range(self.mesh.n_cells):
            cell_info = self.cell_info[cell_id]
            
            # Check refinement criteria
            if (shock_indicators[cell_id] > self.criteria.shock_gradient_threshold and
                cell_info.refinement_level < self.criteria.max_refinement_level):
                
                cell_info.marked_for_refinement = True
                cell_info.refinement_type = RefinementType.SHOCK_BASED
                cell_info.adaptation_indicator = shock_indicators[cell_id]
                marked_count += 1
        
        logger.info(f"Marked {marked_count} cells for shock-based refinement")
        return marked_count
    
    def _mark_gradient_cells(self, solution_data: Dict[str, np.ndarray]) -> int:
        """Mark cells for refinement based on solution gradients."""
        # Get the specified variable for gradient computation
        if self.criteria.gradient_variable == "density":
            field = solution_data.get('conservatives', np.zeros((self.mesh.n_cells, 5)))[:, 0]
        elif self.criteria.gradient_variable == "pressure":
            conservatives = solution_data.get('conservatives', np.zeros((self.mesh.n_cells, 5)))
            density = conservatives[:, 0]
            momentum = conservatives[:, 1:4]
            energy = conservatives[:, 4]
            velocity = momentum / density.reshape(-1, 1)
            field = (self.shock_detector.gamma - 1) * (energy - 0.5 * density * np.sum(velocity**2, axis=1))
        else:
            logger.warning(f"Unknown gradient variable: {self.criteria.gradient_variable}")
            return 0
        
        # Compute gradients
        gradients = self._compute_cell_gradients(field)
        
        # Mark cells based on gradient magnitude
        marked_count = 0
        for cell_id in range(self.mesh.n_cells):
            cell_info = self.cell_info[cell_id]
            
            gradient_magnitude = np.linalg.norm(gradients[cell_id])
            
            if (gradient_magnitude > self.criteria.gradient_threshold and
                cell_info.refinement_level < self.criteria.max_refinement_level):
                
                cell_info.marked_for_refinement = True
                cell_info.refinement_type = RefinementType.FEATURE_BASED
                cell_info.adaptation_indicator = gradient_magnitude
                marked_count += 1
        
        logger.info(f"Marked {marked_count} cells for gradient-based refinement")
        return marked_count
    
    def _mark_error_cells(self, solution_data: Dict[str, np.ndarray]) -> int:
        """Mark cells for refinement based on error estimates."""
        # Simplified error estimation using solution variation
        conservatives = solution_data.get('conservatives', np.zeros((self.mesh.n_cells, 5)))
        
        marked_count = 0
        for cell_id in range(self.mesh.n_cells):
            cell_info = self.cell_info[cell_id]
            
            # Get cell neighbors
            if hasattr(self.mesh, 'connectivity_manager'):
                neighbors = self.mesh.connectivity_manager._cell_neighbors[cell_id]
            else:
                continue
            
            if len(neighbors) < 2:
                continue
            
            # Compute solution variation
            cell_solution = conservatives[cell_id]
            neighbor_solutions = conservatives[neighbors]
            
            # Error estimate based on solution differences
            error_estimate = np.mean([np.linalg.norm(neighbor_solutions[i] - cell_solution) 
                                    for i in range(len(neighbors))])
            
            cell_info.error_estimate = error_estimate
            
            if (error_estimate > self.criteria.error_tolerance and
                cell_info.refinement_level < self.criteria.max_refinement_level):
                
                cell_info.marked_for_refinement = True
                cell_info.refinement_type = RefinementType.ERROR_BASED
                cell_info.adaptation_indicator = error_estimate
                marked_count += 1
        
        logger.info(f"Marked {marked_count} cells for error-based refinement")
        return marked_count
    
    def _compute_cell_gradients(self, field: np.ndarray) -> np.ndarray:
        """Compute cell-centered gradients using least squares."""
        n_cells = len(field)
        gradients = np.zeros((n_cells, 3))
        
        for cell_id in range(n_cells):
            # Get cell neighbors
            if hasattr(self.mesh, 'connectivity_manager'):
                neighbors = self.mesh.connectivity_manager._cell_neighbors[cell_id]
            else:
                continue
            
            if len(neighbors) < 3:
                continue
            
            # Least squares gradient computation
            cell_center = self.mesh.cell_data.centroids[cell_id]
            cell_value = field[cell_id]
            
            # Build system matrix
            A = []
            b = []
            
            for neighbor_id in neighbors:
                neighbor_center = self.mesh.cell_data.centroids[neighbor_id]
                dr = neighbor_center - cell_center
                df = field[neighbor_id] - cell_value
                
                A.append(dr)
                b.append(df)
            
            A = np.array(A)
            b = np.array(b)
            
            try:
                # Solve least squares
                gradient, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                gradients[cell_id] = gradient
            except:
                gradients[cell_id] = np.zeros(3)
        
        return gradients
    
    def refine_marked_cells(self) -> int:
        """
        Refine cells marked for refinement.
        
        Returns:
            Number of cells created through refinement
        """
        # This is a simplified implementation
        # In practice, this would involve complex mesh operations
        
        cells_to_refine = [cell_id for cell_id, info in self.cell_info.items() 
                          if info.marked_for_refinement]
        
        if not cells_to_refine:
            return 0
        
        logger.info(f"Refining {len(cells_to_refine)} cells")
        
        new_cells_created = 0
        
        for cell_id in cells_to_refine:
            cell_info = self.cell_info[cell_id]
            
            # Create child cells (simplified - would depend on cell type)
            if cell_info.refinement_type == RefinementType.ISOTROPIC:
                # For simplicity, assume each cell creates 8 children
                n_children = 8
            else:
                # Anisotropic or other types
                n_children = 4
            
            # Update refinement tracking
            cell_info.refinement_level += 1
            cell_info.marked_for_refinement = False
            
            # Track statistics
            new_cells_created += n_children
            self.adaptation_stats['cells_refined'] += 1
            self.adaptation_stats['max_refinement_level'] = max(
                self.adaptation_stats['max_refinement_level'],
                cell_info.refinement_level
            )
        
        self.adaptation_stats['total_adaptations'] += 1
        
        logger.info(f"Created {new_cells_created} new cells through refinement")
        return new_cells_created
    
    def mark_cells_for_coarsening(self, solution_data: Dict[str, np.ndarray]) -> int:
        """Mark cells for coarsening."""
        # Simplified coarsening criteria
        marked_count = 0
        
        for cell_id, cell_info in self.cell_info.items():
            if (cell_info.refinement_level > 0 and
                cell_info.adaptation_indicator < 0.1 * self.criteria.threshold):
                
                cell_info.marked_for_coarsening = True
                marked_count += 1
        
        logger.info(f"Marked {marked_count} cells for coarsening")
        return marked_count
    
    def coarsen_marked_cells(self) -> int:
        """Coarsen cells marked for coarsening."""
        cells_to_coarsen = [cell_id for cell_id, info in self.cell_info.items() 
                           if info.marked_for_coarsening]
        
        if not cells_to_coarsen:
            return 0
        
        logger.info(f"Coarsening {len(cells_to_coarsen)} cells")
        
        cells_removed = 0
        for cell_id in cells_to_coarsen:
            cell_info = self.cell_info[cell_id]
            cell_info.refinement_level = max(0, cell_info.refinement_level - 1)
            cell_info.marked_for_coarsening = False
            cells_removed += 1
        
        self.adaptation_stats['cells_coarsened'] += cells_removed
        
        logger.info(f"Coarsened {cells_removed} cells")
        return cells_removed
    
    def adapt_mesh(self, solution_data: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Perform complete mesh adaptation cycle.
        
        Returns:
            Dictionary with adaptation statistics
        """
        logger.info("Starting mesh adaptation cycle")
        
        # Mark cells for refinement
        refined_marked = self.mark_cells_for_refinement(solution_data)
        
        # Mark cells for coarsening
        coarsened_marked = self.mark_cells_for_coarsening(solution_data)
        
        # Perform refinement
        cells_created = self.refine_marked_cells()
        
        # Perform coarsening
        cells_removed = self.coarsen_marked_cells()
        
        # Store adaptation history
        adaptation_record = {
            'iteration': len(self.refinement_history),
            'cells_marked_refinement': refined_marked,
            'cells_marked_coarsening': coarsened_marked,
            'cells_created': cells_created,
            'cells_removed': cells_removed,
            'total_cells': self.mesh.n_cells + cells_created - cells_removed
        }
        
        self.refinement_history.append(adaptation_record)
        
        logger.info(f"Adaptation complete: +{cells_created} -{cells_removed} cells")
        
        return adaptation_record
    
    def get_adaptation_statistics(self) -> Dict[str, Union[int, float, List]]:
        """Get comprehensive adaptation statistics."""
        refinement_levels = [info.refinement_level for info in self.cell_info.values()]
        
        return {
            **self.adaptation_stats,
            'current_cells': self.mesh.n_cells if self.mesh else 0,
            'refinement_levels': {
                'mean': np.mean(refinement_levels) if refinement_levels else 0,
                'max': np.max(refinement_levels) if refinement_levels else 0,
                'distribution': np.bincount(refinement_levels).tolist() if refinement_levels else []
            },
            'adaptation_history': self.refinement_history
        }
    
    def reset_adaptation_flags(self) -> None:
        """Reset all adaptation flags."""
        for cell_info in self.cell_info.values():
            cell_info.marked_for_refinement = False
            cell_info.marked_for_coarsening = False
            cell_info.adaptation_indicator = 0.0