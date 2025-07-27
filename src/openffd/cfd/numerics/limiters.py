"""
Advanced Limiters and Monotonicity Preservers

Implements various limiting strategies for high-resolution schemes:
- Slope limiters for reconstruction
- Flux limiters for TVD schemes
- Monotonicity preservers
- Positivity preservers
- Multi-dimensional limiters
- Shock-based adaptive limiting
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class SlopeLimiter(ABC):
    """Abstract base class for slope limiters."""
    
    @abstractmethod
    def limit_slope(self, 
                   center_value: float,
                   gradient: np.ndarray,
                   neighbor_values: np.ndarray,
                   neighbor_vectors: np.ndarray) -> np.ndarray:
        """
        Limit slope to maintain monotonicity.
        
        Args:
            center_value: Value at cell center
            gradient: Computed gradient vector
            neighbor_values: Values at neighboring cells
            neighbor_vectors: Vectors from center to neighbors
            
        Returns:
            Limited gradient vector
        """
        pass


class BartkLimiter(SlopeLimiter):
    """
    Barth-Jespersen limiter for unstructured meshes.
    
    Ensures that reconstructed values at cell interfaces do not
    exceed the maximum and minimum values in the local neighborhood.
    """
    
    def __init__(self, epsilon: float = 1e-12):
        """
        Initialize Barth-Jespersen limiter.
        
        Args:
            epsilon: Small number to avoid division by zero
        """
        self.epsilon = epsilon
    
    def limit_slope(self, 
                   center_value: float,
                   gradient: np.ndarray,
                   neighbor_values: np.ndarray,
                   neighbor_vectors: np.ndarray) -> np.ndarray:
        """Apply Barth-Jespersen limiting."""
        if len(neighbor_values) == 0:
            return gradient
        
        # Find min and max in neighborhood
        all_values = np.concatenate([[center_value], neighbor_values])
        u_min = np.min(all_values)
        u_max = np.max(all_values)
        
        # Initialize limiter value
        phi = 1.0
        
        # Check each neighbor
        for i, (neighbor_value, neighbor_vector) in enumerate(zip(neighbor_values, neighbor_vectors)):
            # Projected change using gradient
            delta_u = np.dot(gradient, neighbor_vector)
            u_face = center_value + delta_u
            
            # Compute limiter for this face
            if delta_u > self.epsilon:
                phi_i = min(1.0, (u_max - center_value) / delta_u)
            elif delta_u < -self.epsilon:
                phi_i = min(1.0, (u_min - center_value) / delta_u)
            else:
                phi_i = 1.0
            
            # Take minimum over all faces
            phi = min(phi, phi_i)
        
        return phi * gradient


class VenkatakrishnanLimiter(SlopeLimiter):
    """
    Venkatakrishnan limiter - smooth version of Barth-Jespersen.
    
    Provides smooth limiting function that avoids sudden gradient
    cutoff and improves convergence properties.
    """
    
    def __init__(self, K: float = 5.0, epsilon: float = 1e-12):
        """
        Initialize Venkatakrishnan limiter.
        
        Args:
            K: Limiter parameter (controls smoothness)
            epsilon: Small number for numerical stability
        """
        self.K = K
        self.epsilon = epsilon
    
    def limit_slope(self, 
                   center_value: float,
                   gradient: np.ndarray,
                   neighbor_values: np.ndarray,
                   neighbor_vectors: np.ndarray) -> np.ndarray:
        """Apply Venkatakrishnan limiting."""
        if len(neighbor_values) == 0:
            return gradient
        
        # Find min and max in neighborhood
        all_values = np.concatenate([[center_value], neighbor_values])
        u_min = np.min(all_values)
        u_max = np.max(all_values)
        
        # Characteristic mesh size (for smoothness parameter)
        if len(neighbor_vectors) > 0:
            h = np.mean([np.linalg.norm(vec) for vec in neighbor_vectors])
        else:
            h = 1.0
        
        # Smoothness parameter
        epsilon_smooth = (self.K * h)**3
        
        # Initialize limiter
        phi = 1.0
        
        # Check each neighbor
        for neighbor_vector in neighbor_vectors:
            delta_u = np.dot(gradient, neighbor_vector)
            
            if delta_u > self.epsilon:
                delta_max = u_max - center_value
                phi_i = (delta_max**2 + epsilon_smooth + 2*delta_u*delta_max) / \
                        (delta_max**2 + 2*delta_u**2 + delta_u*delta_max + epsilon_smooth)
            elif delta_u < -self.epsilon:
                delta_min = u_min - center_value
                phi_i = (delta_min**2 + epsilon_smooth + 2*delta_u*delta_min) / \
                        (delta_min**2 + 2*delta_u**2 + delta_u*delta_min + epsilon_smooth)
            else:
                phi_i = 1.0
            
            phi = min(phi, phi_i)
        
        return phi * gradient


class MichalakGoochLimiter(SlopeLimiter):
    """
    Michalak-Gooch limiter for improved accuracy.
    
    Provides better accuracy in smooth regions while maintaining
    monotonicity near discontinuities.
    """
    
    def __init__(self, alpha: float = 2.0, epsilon: float = 1e-12):
        """
        Initialize Michalak-Gooch limiter.
        
        Args:
            alpha: Limiter parameter
            epsilon: Small number for numerical stability
        """
        self.alpha = alpha
        self.epsilon = epsilon
    
    def limit_slope(self, 
                   center_value: float,
                   gradient: np.ndarray,
                   neighbor_values: np.ndarray,
                   neighbor_vectors: np.ndarray) -> np.ndarray:
        """Apply Michalak-Gooch limiting."""
        if len(neighbor_values) == 0:
            return gradient
        
        # Modified Barth-Jespersen with improved accuracy
        all_values = np.concatenate([[center_value], neighbor_values])
        u_min = np.min(all_values)
        u_max = np.max(all_values)
        
        phi = 1.0
        
        for neighbor_vector in neighbor_vectors:
            delta_u = np.dot(gradient, neighbor_vector)
            
            if delta_u > self.epsilon:
                delta_max = u_max - center_value
                phi_i = min(1.0, self.alpha * delta_max / delta_u)
            elif delta_u < -self.epsilon:
                delta_min = u_min - center_value
                phi_i = min(1.0, self.alpha * delta_min / delta_u)
            else:
                phi_i = 1.0
            
            phi = min(phi, phi_i)
        
        return phi * gradient


class PositivityPreserver:
    """
    Positivity preserving limiter for physical quantities.
    
    Ensures that certain quantities (like density, pressure)
    remain positive after reconstruction and time stepping.
    """
    
    def __init__(self, 
                 positive_variables: List[int],
                 min_values: Optional[List[float]] = None,
                 scaling_factor: float = 0.1):
        """
        Initialize positivity preserver.
        
        Args:
            positive_variables: Indices of variables that must remain positive
            min_values: Minimum allowed values for each positive variable
            scaling_factor: Safety factor for limiting
        """
        self.positive_variables = positive_variables
        self.min_values = min_values or [1e-12] * len(positive_variables)
        self.scaling_factor = scaling_factor
        
        if len(self.min_values) != len(positive_variables):
            raise ValueError("min_values length must match positive_variables length")
    
    def preserve_positivity(self, 
                          center_values: np.ndarray,
                          reconstructed_values: np.ndarray) -> np.ndarray:
        """
        Preserve positivity of specified variables.
        
        Args:
            center_values: Values at cell center
            reconstructed_values: Reconstructed values at interface
            
        Returns:
            Limited reconstructed values
        """
        limited_values = reconstructed_values.copy()
        
        for i, var_idx in enumerate(self.positive_variables):
            min_val = self.min_values[i]
            
            if reconstructed_values[var_idx] < min_val:
                # Scale back towards center value to maintain positivity
                center_val = center_values[var_idx]
                
                if center_val > min_val:
                    # Compute scaling factor
                    alpha = (min_val - center_val) / (reconstructed_values[var_idx] - center_val + 1e-12)
                    alpha = max(0.0, min(1.0, alpha))
                    
                    # Apply scaling with safety factor
                    limited_values[var_idx] = center_val + alpha * self.scaling_factor * \
                                            (reconstructed_values[var_idx] - center_val)
                else:
                    # Center value itself is problematic
                    limited_values[var_idx] = min_val
        
        return limited_values


class ShockAdaptiveLimiter:
    """
    Shock-adaptive limiter that adjusts limiting based on shock detection.
    
    Uses shock sensors to apply strong limiting near shocks and
    minimal limiting in smooth regions.
    """
    
    def __init__(self, 
                 base_limiter: SlopeLimiter,
                 shock_detector: Optional[Callable] = None,
                 smooth_factor: float = 0.1,
                 shock_factor: float = 1.0):
        """
        Initialize shock-adaptive limiter.
        
        Args:
            base_limiter: Base limiter to use
            shock_detector: Function to detect shocks
            smooth_factor: Limiting factor in smooth regions
            shock_factor: Limiting factor in shock regions
        """
        self.base_limiter = base_limiter
        self.shock_detector = shock_detector
        self.smooth_factor = smooth_factor
        self.shock_factor = shock_factor
    
    def limit_slope_adaptive(self, 
                           center_value: float,
                           gradient: np.ndarray,
                           neighbor_values: np.ndarray,
                           neighbor_vectors: np.ndarray,
                           shock_indicator: float = 0.0) -> np.ndarray:
        """
        Apply adaptive limiting based on shock indicator.
        
        Args:
            center_value: Value at cell center
            gradient: Computed gradient
            neighbor_values: Neighboring values
            neighbor_vectors: Vectors to neighbors
            shock_indicator: Shock indicator [0, 1]
            
        Returns:
            Adaptively limited gradient
        """
        # Apply base limiting
        limited_gradient = self.base_limiter.limit_slope(
            center_value, gradient, neighbor_values, neighbor_vectors
        )
        
        # Adjust based on shock indicator
        limiting_factor = (1.0 - shock_indicator) * self.smooth_factor + \
                         shock_indicator * self.shock_factor
        
        # Blend between unlimited and limited gradients
        adaptive_gradient = (1.0 - limiting_factor) * gradient + \
                           limiting_factor * limited_gradient
        
        return adaptive_gradient


class MultiDimensionalLimiter:
    """
    Multi-dimensional limiter for vector quantities.
    
    Applies limiting to vector fields while preserving
    directional properties and physical constraints.
    """
    
    def __init__(self, 
                 scalar_limiter: SlopeLimiter,
                 vector_treatment: str = "component_wise"):
        """
        Initialize multi-dimensional limiter.
        
        Args:
            scalar_limiter: Base scalar limiter
            vector_treatment: How to treat vectors ("component_wise", "magnitude_direction")
        """
        self.scalar_limiter = scalar_limiter
        self.vector_treatment = vector_treatment
    
    def limit_vector_gradient(self, 
                            center_vector: np.ndarray,
                            gradient_tensor: np.ndarray,
                            neighbor_vectors: np.ndarray,
                            neighbor_positions: np.ndarray) -> np.ndarray:
        """
        Limit gradient tensor for vector quantities.
        
        Args:
            center_vector: Vector at cell center [n_components]
            gradient_tensor: Gradient tensor [n_components, 3]
            neighbor_vectors: Vectors at neighbors [n_neighbors, n_components]
            neighbor_positions: Position vectors to neighbors [n_neighbors, 3]
            
        Returns:
            Limited gradient tensor
        """
        if self.vector_treatment == "component_wise":
            return self._limit_component_wise(center_vector, gradient_tensor,
                                            neighbor_vectors, neighbor_positions)
        elif self.vector_treatment == "magnitude_direction":
            return self._limit_magnitude_direction(center_vector, gradient_tensor,
                                                 neighbor_vectors, neighbor_positions)
        else:
            raise ValueError(f"Unknown vector treatment: {self.vector_treatment}")
    
    def _limit_component_wise(self, 
                            center_vector: np.ndarray,
                            gradient_tensor: np.ndarray,
                            neighbor_vectors: np.ndarray,
                            neighbor_positions: np.ndarray) -> np.ndarray:
        """Limit each vector component independently."""
        limited_gradient = np.zeros_like(gradient_tensor)
        
        for component in range(len(center_vector)):
            center_value = center_vector[component]
            gradient = gradient_tensor[component]
            neighbor_values = neighbor_vectors[:, component]
            
            limited_gradient[component] = self.scalar_limiter.limit_slope(
                center_value, gradient, neighbor_values, neighbor_positions
            )
        
        return limited_gradient
    
    def _limit_magnitude_direction(self, 
                                 center_vector: np.ndarray,
                                 gradient_tensor: np.ndarray,
                                 neighbor_vectors: np.ndarray,
                                 neighbor_positions: np.ndarray) -> np.ndarray:
        """Limit vector magnitude and direction separately."""
        # Compute vector magnitudes
        center_magnitude = np.linalg.norm(center_vector)
        neighbor_magnitudes = np.linalg.norm(neighbor_vectors, axis=1)
        
        # Limit magnitude gradient
        magnitude_gradient = self._compute_magnitude_gradient(
            center_vector, gradient_tensor, neighbor_positions
        )
        
        limited_magnitude_gradient = self.scalar_limiter.limit_slope(
            center_magnitude, magnitude_gradient, neighbor_magnitudes, neighbor_positions
        )
        
        # Reconstruct limited gradient tensor
        # This is a simplified approach - full implementation would be more complex
        if center_magnitude > 1e-12:
            direction = center_vector / center_magnitude
            limited_gradient = np.outer(direction, limited_magnitude_gradient)
        else:
            limited_gradient = gradient_tensor * 0.1  # Small limiting factor
        
        return limited_gradient
    
    def _compute_magnitude_gradient(self, 
                                  center_vector: np.ndarray,
                                  gradient_tensor: np.ndarray,
                                  neighbor_positions: np.ndarray) -> np.ndarray:
        """Compute gradient of vector magnitude."""
        center_magnitude = np.linalg.norm(center_vector)
        
        if center_magnitude < 1e-12:
            return np.zeros(3)
        
        # ∇|v| = (v · ∇v) / |v|
        magnitude_gradient = np.zeros(3)
        for i in range(3):
            magnitude_gradient[i] = np.dot(center_vector, gradient_tensor[:, i]) / center_magnitude
        
        return magnitude_gradient


class FluxLimiter:
    """
    Flux limiter for TVD schemes with enhanced features.
    
    Provides advanced flux limiting with shock detection,
    entropy fixes, and multi-dimensional corrections.
    """
    
    def __init__(self, 
                 limiter_function: Callable[[float], float],
                 entropy_fix: bool = True,
                 multidimensional_correction: bool = False):
        """
        Initialize flux limiter.
        
        Args:
            limiter_function: Base limiter function φ(r)
            entropy_fix: Apply entropy fix for expansion shocks
            multidimensional_correction: Apply multi-dimensional corrections
        """
        self.limiter_function = limiter_function
        self.entropy_fix = entropy_fix
        self.multidimensional_correction = multidimensional_correction
    
    def limit_flux(self, 
                  upwind_flux: np.ndarray,
                  downwind_flux: np.ndarray,
                  gradient_ratio: float,
                  shock_indicator: float = 0.0) -> np.ndarray:
        """
        Apply flux limiting with enhancements.
        
        Args:
            upwind_flux: Upwind flux
            downwind_flux: Downwind flux
            gradient_ratio: Ratio of consecutive gradients
            shock_indicator: Shock detection indicator
            
        Returns:
            Limited flux
        """
        # Apply base limiter
        phi = self.limiter_function(gradient_ratio)
        
        # Entropy fix
        if self.entropy_fix:
            phi = self._apply_entropy_fix(phi, gradient_ratio)
        
        # Shock-based adjustment
        if shock_indicator > 0.5:  # Strong shock region
            phi *= 0.5  # More dissipative
        
        # Compute limited flux
        limited_flux = upwind_flux + phi * (downwind_flux - upwind_flux)
        
        return limited_flux
    
    def _apply_entropy_fix(self, phi: float, r: float) -> float:
        """Apply entropy fix for expansion shocks."""
        # Harten's entropy fix
        if abs(r) < 1e-12:
            return 0.0
        
        # Modify limiter near sonic points
        if 0.8 < r < 1.2:  # Near sonic conditions
            phi *= 0.9  # Slight reduction
        
        return phi


class MonotonicityPreserver:
    """
    Advanced monotonicity preserver with multiple strategies.
    
    Ensures solution monotonicity using various approaches
    including local and global constraints.
    """
    
    def __init__(self, 
                 tolerance: float = 1e-12,
                 global_bounds: bool = False,
                 relaxation_factor: float = 0.9):
        """
        Initialize monotonicity preserver.
        
        Args:
            tolerance: Tolerance for monotonicity checks
            global_bounds: Use global solution bounds
            relaxation_factor: Relaxation factor for constraints
        """
        self.tolerance = tolerance
        self.global_bounds = global_bounds
        self.relaxation_factor = relaxation_factor
    
    def enforce_monotonicity(self, 
                           solution: np.ndarray,
                           connectivity_info: Dict,
                           variable_index: int = 0) -> np.ndarray:
        """
        Enforce monotonicity on entire solution.
        
        Args:
            solution: Solution array [n_cells, n_variables]
            connectivity_info: Mesh connectivity information
            variable_index: Which variable to check
            
        Returns:
            Monotonicity-preserved solution
        """
        if len(solution.shape) == 1:
            values = solution
        else:
            values = solution[:, variable_index]
        
        corrected_values = values.copy()
        
        # Get connectivity
        connectivity = connectivity_info.get('connectivity_manager')
        if not connectivity or not connectivity._cell_neighbors:
            return solution
        
        # Apply local monotonicity constraints
        for cell_id in range(len(values)):
            neighbors = connectivity._cell_neighbors[cell_id]
            
            if len(neighbors) == 0:
                continue
            
            cell_value = values[cell_id]
            neighbor_values = values[neighbors]
            
            # Check monotonicity violation
            min_neighbor = np.min(neighbor_values)
            max_neighbor = np.max(neighbor_values)
            
            # Apply bounds with relaxation
            lower_bound = min_neighbor - self.tolerance
            upper_bound = max_neighbor + self.tolerance
            
            if cell_value < lower_bound:
                corrected_values[cell_id] = lower_bound + \
                    self.relaxation_factor * (cell_value - lower_bound)
            elif cell_value > upper_bound:
                corrected_values[cell_id] = upper_bound + \
                    self.relaxation_factor * (cell_value - upper_bound)
        
        # Update solution
        if len(solution.shape) == 1:
            return corrected_values
        else:
            corrected_solution = solution.copy()
            corrected_solution[:, variable_index] = corrected_values
            return corrected_solution
    
    def check_monotonicity_violation(self, 
                                   solution: np.ndarray,
                                   connectivity_info: Dict) -> Dict[str, float]:
        """
        Check for monotonicity violations in solution.
        
        Returns:
            Dictionary with violation statistics
        """
        if len(solution.shape) == 1:
            n_variables = 1
            values = solution.reshape(-1, 1)
        else:
            n_variables = solution.shape[1]
            values = solution
        
        violations = {
            'total_violations': 0,
            'max_violation': 0.0,
            'variables_violated': []
        }
        
        connectivity = connectivity_info.get('connectivity_manager')
        if not connectivity:
            return violations
        
        for var_idx in range(n_variables):
            var_values = values[:, var_idx]
            var_violations = 0
            max_var_violation = 0.0
            
            for cell_id in range(len(var_values)):
                neighbors = connectivity._cell_neighbors[cell_id] if connectivity._cell_neighbors else []
                
                if len(neighbors) == 0:
                    continue
                
                cell_value = var_values[cell_id]
                neighbor_values = var_values[neighbors]
                
                min_neighbor = np.min(neighbor_values)
                max_neighbor = np.max(neighbor_values)
                
                # Check for violations
                violation = 0.0
                if cell_value < min_neighbor - self.tolerance:
                    violation = min_neighbor - cell_value
                    var_violations += 1
                elif cell_value > max_neighbor + self.tolerance:
                    violation = cell_value - max_neighbor
                    var_violations += 1
                
                max_var_violation = max(max_var_violation, violation)
            
            if var_violations > 0:
                violations['variables_violated'].append(var_idx)
            
            violations['total_violations'] += var_violations
            violations['max_violation'] = max(violations['max_violation'], max_var_violation)
        
        return violations


def create_limiter(limiter_type: str, **kwargs) -> Union[SlopeLimiter, FluxLimiter]:
    """
    Create limiter with specified type and parameters.
    
    Args:
        limiter_type: Type of limiter ("barth", "venkatakrishnan", "michalak_gooch", etc.)
        **kwargs: Additional parameters
        
    Returns:
        Configured limiter
    """
    slope_limiters = {
        "barth": BartkLimiter,
        "venkatakrishnan": VenkatakrishnanLimiter, 
        "michalak_gooch": MichalakGoochLimiter
    }
    
    if limiter_type in slope_limiters:
        return slope_limiters[limiter_type](**kwargs)
    else:
        raise ValueError(f"Unknown limiter type: {limiter_type}")


def test_limiters():
    """Test various limiters on simple cases."""
    print("Testing Limiters:")
    
    # Test data
    center_value = 1.0
    gradient = np.array([0.5, 0.0, 0.0])
    neighbor_values = np.array([0.8, 1.2, 0.9, 1.1])
    neighbor_vectors = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0]
    ])
    
    limiters = {
        "Barth": BartkLimiter(),
        "Venkatakrishnan": VenkatakrishnanLimiter(),
        "Michalak-Gooch": MichalakGoochLimiter()
    }
    
    print(f"Original gradient: {gradient}")
    print(f"Center value: {center_value}")
    print(f"Neighbor values: {neighbor_values}")
    print()
    
    for name, limiter in limiters.items():
        limited_gradient = limiter.limit_slope(center_value, gradient, 
                                             neighbor_values, neighbor_vectors)
        limiting_factor = np.linalg.norm(limited_gradient) / np.linalg.norm(gradient)
        
        print(f"{name}: limited_gradient={limited_gradient}, factor={limiting_factor:.3f}")


if __name__ == "__main__":
    test_limiters()