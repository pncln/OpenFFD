"""
TVD (Total Variation Diminishing) Schemes and Flux Limiters

Implements TVD schemes and various flux limiters for shock-capturing:
- Classical TVD framework
- Flux limiters (MinMod, SuperBee, Van Leer, etc.)
- Slope limiters for high-order reconstruction
- Monotonicity preservation
- Unstructured mesh adaptation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class FluxLimiter(ABC):
    """Abstract base class for flux limiters."""
    
    @abstractmethod
    def limit(self, r: float) -> float:
        """
        Apply flux limiter function.
        
        Args:
            r: Ratio of consecutive gradients
            
        Returns:
            Limited value
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get limiter name."""
        pass


class MinModLimiter(FluxLimiter):
    """
    MinMod limiter - most dissipative but very stable.
    
    φ(r) = max(0, min(1, r))
    
    Provides strong shock-capturing but may be overly dissipative
    in smooth regions.
    """
    
    def limit(self, r: float) -> float:
        """Apply MinMod limiter."""
        return max(0.0, min(1.0, r))
    
    def get_name(self) -> str:
        return "MinMod"


class SuperBeeLimiter(FluxLimiter):
    """
    SuperBee limiter - less dissipative than MinMod.
    
    φ(r) = max(0, min(2r, 1), min(r, 2))
    
    Provides good balance between accuracy and stability.
    """
    
    def limit(self, r: float) -> float:
        """Apply SuperBee limiter."""
        return max(0.0, min(2.0 * r, 1.0), min(r, 2.0))
    
    def get_name(self) -> str:
        return "SuperBee"


class VanLeerLimiter(FluxLimiter):
    """
    Van Leer limiter - smooth and less oscillatory.
    
    φ(r) = (r + |r|) / (1 + |r|)
    
    Provides smooth limiting with good convergence properties.
    """
    
    def limit(self, r: float) -> float:
        """Apply Van Leer limiter."""
        return (r + abs(r)) / (1.0 + abs(r))
    
    def get_name(self) -> str:
        return "VanLeer"


class MonotonizedCentralLimiter(FluxLimiter):
    """
    Monotonized Central (MC) limiter.
    
    φ(r) = max(0, min((1+r)/2, 2, 2r))
    
    Good compromise between accuracy and monotonicity.
    """
    
    def limit(self, r: float) -> float:
        """Apply MC limiter."""
        return max(0.0, min((1.0 + r) / 2.0, 2.0, 2.0 * r))
    
    def get_name(self) -> str:
        return "MC"


class KorenLimiter(FluxLimiter):
    """
    Koren limiter - third-order accurate in smooth regions.
    
    φ(r) = max(0, min(2r, (1+2r)/3, 2))
    
    Provides high accuracy while maintaining TVD property.
    """
    
    def limit(self, r: float) -> float:
        """Apply Koren limiter."""
        return max(0.0, min(2.0 * r, (1.0 + 2.0 * r) / 3.0, 2.0))
    
    def get_name(self) -> str:
        return "Koren"


class OspreLimiter(FluxLimiter):
    """
    OSPRE limiter - optimized for accuracy.
    
    φ(r) = max(0, min(1.5(r^2 + r)/(r^2 + r + 1), 2))
    
    Provides excellent accuracy in smooth regions.
    """
    
    def limit(self, r: float) -> float:
        """Apply OSPRE limiter."""
        if abs(r) < 1e-12:
            return 0.0
        
        numerator = 1.5 * (r**2 + r)
        denominator = r**2 + r + 1.0
        
        return max(0.0, min(numerator / denominator, 2.0))
    
    def get_name(self) -> str:
        return "OSPRE"


class TVDReconstructor:
    """
    TVD (Total Variation Diminishing) reconstructor for unstructured meshes.
    
    Provides high-order accurate reconstruction while maintaining
    monotonicity and preventing spurious oscillations near discontinuities.
    """
    
    def __init__(self, 
                 limiter: FluxLimiter,
                 reconstruction_order: int = 2,
                 epsilon: float = 1e-12):
        """
        Initialize TVD reconstructor.
        
        Args:
            limiter: Flux limiter to use
            reconstruction_order: Order of reconstruction (1 or 2)
            epsilon: Small number to avoid division by zero
        """
        self.limiter = limiter
        self.reconstruction_order = reconstruction_order
        self.epsilon = epsilon
        
        logger.info(f"Initialized TVD reconstructor with {limiter.get_name()} limiter")
    
    def reconstruct_left_right(self, 
                              stencil_values: np.ndarray,
                              stencil_points: np.ndarray,
                              interface_point: np.ndarray,
                              interface_normal: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Reconstruct left and right states at interface using TVD scheme.
        
        Args:
            stencil_values: Solution values at stencil points
            stencil_points: Coordinates of stencil points
            interface_point: Interface coordinate
            interface_normal: Interface normal vector (optional)
            
        Returns:
            (left_state, right_state) at interface
        """
        if len(stencil_values) < 3:
            return self._first_order_reconstruction(stencil_values, stencil_points, interface_point)
        
        # Sort points by distance to interface
        distances = np.linalg.norm(stencil_points - interface_point, axis=1)
        sorted_indices = np.argsort(distances)
        
        # Get closest points for reconstruction
        n_points = min(5, len(stencil_values))  # Use up to 5 points
        indices = sorted_indices[:n_points]
        values = stencil_values[indices]
        points = stencil_points[indices]
        
        if self.reconstruction_order == 1:
            return self._first_order_reconstruction(values, points, interface_point)
        else:
            return self._second_order_tvd_reconstruction(values, points, interface_point, interface_normal)
    
    def _first_order_reconstruction(self, 
                                   values: np.ndarray,
                                   points: np.ndarray,
                                   interface_point: np.ndarray) -> Tuple[float, float]:
        """First-order upwind reconstruction."""
        if len(values) == 0:
            return 0.0, 0.0
        elif len(values) == 1:
            return values[0], values[0]
        
        # Find closest point
        distances = np.linalg.norm(points - interface_point, axis=1)
        closest_idx = np.argmin(distances)
        closest_value = values[closest_idx]
        
        return closest_value, closest_value
    
    def _second_order_tvd_reconstruction(self, 
                                        values: np.ndarray,
                                        points: np.ndarray,
                                        interface_point: np.ndarray,
                                        interface_normal: Optional[np.ndarray]) -> Tuple[float, float]:
        """Second-order TVD reconstruction with slope limiting."""
        if len(values) < 3:
            return self._first_order_reconstruction(values, points, interface_point)
        
        # For unstructured meshes, we need to identify upwind/downwind directions
        # This is challenging without flow information, so we use geometric approach
        
        # Find three closest points
        distances = np.linalg.norm(points - interface_point, axis=1)
        sorted_indices = np.argsort(distances)
        
        # Central point (closest to interface)
        center_idx = sorted_indices[0]
        center_value = values[center_idx]
        center_point = points[center_idx]
        
        # Find upstream and downstream points
        upstream_idx, downstream_idx = self._find_upstream_downstream(
            center_idx, sorted_indices[1:], points, interface_point, interface_normal
        )
        
        if upstream_idx is None or downstream_idx is None:
            # Fall back to first-order
            return center_value, center_value
        
        upstream_value = values[upstream_idx]
        downstream_value = values[downstream_idx]
        upstream_point = points[upstream_idx]
        downstream_point = points[downstream_idx]
        
        # Compute gradients
        upstream_gradient = self._compute_gradient(upstream_value, center_value, 
                                                 upstream_point, center_point)
        downstream_gradient = self._compute_gradient(center_value, downstream_value,
                                                   center_point, downstream_point)
        
        # Compute ratio for limiter
        if abs(downstream_gradient) < self.epsilon:
            r = 0.0
        else:
            r = upstream_gradient / downstream_gradient
        
        # Apply limiter
        phi = self.limiter.limit(r)
        
        # Limited slope
        limited_gradient = phi * downstream_gradient
        
        # Reconstruct left and right states
        distance_to_interface = np.linalg.norm(interface_point - center_point)
        
        # Linear reconstruction with limited slope
        extrapolated_value = center_value + limited_gradient * distance_to_interface
        
        # For left-right states, apply small bias
        left_state = 0.8 * center_value + 0.2 * extrapolated_value
        right_state = 0.2 * center_value + 0.8 * extrapolated_value
        
        return left_state, right_state
    
    def _find_upstream_downstream(self, 
                                 center_idx: int,
                                 candidate_indices: np.ndarray,
                                 points: np.ndarray,
                                 interface_point: np.ndarray,
                                 interface_normal: Optional[np.ndarray]) -> Tuple[Optional[int], Optional[int]]:
        """Find upstream and downstream points for TVD reconstruction."""
        if len(candidate_indices) < 2:
            return None, None
        
        center_point = points[center_idx]
        
        if interface_normal is not None:
            # Use normal vector to determine upstream/downstream
            upstream_idx = None
            downstream_idx = None
            
            for idx in candidate_indices:
                direction = points[idx] - center_point
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm > self.epsilon:
                    direction /= direction_norm
                    dot_product = np.dot(direction, interface_normal)
                    
                    if dot_product < -0.1:  # Upstream
                        upstream_idx = idx
                    elif dot_product > 0.1:  # Downstream
                        downstream_idx = idx
            
            return upstream_idx, downstream_idx
        else:
            # Geometric approach: use closest two points
            return candidate_indices[0], candidate_indices[1]
    
    def _compute_gradient(self, 
                         value1: float, 
                         value2: float,
                         point1: np.ndarray, 
                         point2: np.ndarray) -> float:
        """Compute gradient between two points."""
        distance = np.linalg.norm(point2 - point1)
        if distance < self.epsilon:
            return 0.0
        
        return (value2 - value1) / distance
    
    def reconstruct_all_variables(self, 
                                 stencil_values: np.ndarray,
                                 stencil_points: np.ndarray,
                                 interface_point: np.ndarray,
                                 interface_normal: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct all variables using TVD scheme.
        
        Args:
            stencil_values: Solution values [n_points, n_variables]
            stencil_points: Coordinates [n_points, 3]
            interface_point: Interface coordinate [3]
            interface_normal: Interface normal vector [3] (optional)
            
        Returns:
            (left_states, right_states) for all variables
        """
        if len(stencil_values.shape) == 1:
            # Single variable
            left, right = self.reconstruct_left_right(stencil_values, stencil_points, 
                                                    interface_point, interface_normal)
            return np.array([left]), np.array([right])
        
        n_variables = stencil_values.shape[1]
        left_states = np.zeros(n_variables)
        right_states = np.zeros(n_variables)
        
        for var_idx in range(n_variables):
            values = stencil_values[:, var_idx]
            left, right = self.reconstruct_left_right(values, stencil_points,
                                                    interface_point, interface_normal)
            left_states[var_idx] = left
            right_states[var_idx] = right
        
        return left_states, right_states


class SlopeLimiter:
    """
    Slope limiter for gradient reconstruction.
    
    Applies limiting to computed gradients to maintain monotonicity
    and prevent spurious oscillations.
    """
    
    def __init__(self, limiter: FluxLimiter):
        """
        Initialize slope limiter.
        
        Args:
            limiter: Flux limiter to use for slope limiting
        """
        self.limiter = limiter
    
    def limit_gradient(self, 
                      center_value: float,
                      center_gradient: np.ndarray,
                      neighbor_values: np.ndarray,
                      neighbor_points: np.ndarray,
                      center_point: np.ndarray) -> np.ndarray:
        """
        Limit gradient to maintain monotonicity.
        
        Args:
            center_value: Value at cell center
            center_gradient: Computed gradient at center
            neighbor_values: Values at neighboring cells
            neighbor_points: Coordinates of neighboring cells
            center_point: Coordinate of center cell
            
        Returns:
            Limited gradient vector
        """
        if len(neighbor_values) == 0:
            return center_gradient
        
        limited_gradient = center_gradient.copy()
        
        # Check each neighbor for monotonicity
        for i, (neighbor_value, neighbor_point) in enumerate(zip(neighbor_values, neighbor_points)):
            direction = neighbor_point - center_point
            distance = np.linalg.norm(direction)
            
            if distance < 1e-12:
                continue
            
            direction_unit = direction / distance
            
            # Projected gradient in neighbor direction
            projected_gradient = np.dot(center_gradient, direction_unit)
            
            # Expected value at neighbor using gradient
            expected_value = center_value + projected_gradient * distance
            
            # Actual difference
            actual_difference = neighbor_value - center_value
            expected_difference = expected_value - center_value
            
            # Compute limiter ratio
            if abs(expected_difference) < 1e-12:
                continue
            
            r = actual_difference / expected_difference
            phi = self.limiter.limit(r)
            
            # Apply limiting to projected component
            if phi < 1.0:
                # Reduce gradient in this direction
                reduction_factor = phi
                limited_gradient = (limited_gradient - 
                                  (1.0 - reduction_factor) * projected_gradient * direction_unit)
        
        return limited_gradient


class MonotonicityPreserver:
    """
    Ensures monotonicity preservation in reconstructed values.
    
    Applies additional checks and corrections to maintain
    solution monotonicity properties.
    """
    
    def __init__(self, tolerance: float = 1e-12):
        """
        Initialize monotonicity preserver.
        
        Args:
            tolerance: Tolerance for monotonicity checks
        """
        self.tolerance = tolerance
    
    def enforce_monotonicity(self, 
                           center_value: float,
                           reconstructed_left: float,
                           reconstructed_right: float,
                           neighbor_values: np.ndarray) -> Tuple[float, float]:
        """
        Enforce monotonicity on reconstructed values.
        
        Args:
            center_value: Value at cell center
            reconstructed_left: Left reconstructed value
            reconstructed_right: Right reconstructed value
            neighbor_values: Values at neighboring cells
            
        Returns:
            (corrected_left, corrected_right) values
        """
        if len(neighbor_values) == 0:
            return reconstructed_left, reconstructed_right
        
        # Find min and max of neighbors
        min_neighbor = np.min(neighbor_values)
        max_neighbor = np.max(neighbor_values)
        
        # Extend bounds slightly to include center value
        local_min = min(min_neighbor, center_value) - self.tolerance
        local_max = max(max_neighbor, center_value) + self.tolerance
        
        # Clip reconstructed values to local bounds
        corrected_left = np.clip(reconstructed_left, local_min, local_max)
        corrected_right = np.clip(reconstructed_right, local_min, local_max)
        
        return corrected_left, corrected_right
    
    def check_tvd_property(self, 
                          old_values: np.ndarray,
                          new_values: np.ndarray) -> bool:
        """
        Check if TVD property is satisfied.
        
        Args:
            old_values: Values before update
            new_values: Values after update
            
        Returns:
            True if TVD property is satisfied
        """
        if len(old_values) != len(new_values):
            return False
        
        # Compute total variation
        tv_old = self._compute_total_variation(old_values)
        tv_new = self._compute_total_variation(new_values)
        
        # TVD property: TV should not increase
        return tv_new <= tv_old + self.tolerance
    
    def _compute_total_variation(self, values: np.ndarray) -> float:
        """Compute total variation of solution."""
        if len(values) < 2:
            return 0.0
        
        total_variation = 0.0
        for i in range(len(values) - 1):
            total_variation += abs(values[i + 1] - values[i])
        
        return total_variation


def create_tvd_reconstructor(limiter_type: str = "minmod", **kwargs) -> TVDReconstructor:
    """
    Create TVD reconstructor with specified limiter.
    
    Args:
        limiter_type: Type of limiter ("minmod", "superbee", "vanleer", "mc", "koren", "ospre")
        **kwargs: Additional parameters for reconstructor
        
    Returns:
        Configured TVD reconstructor
    """
    limiter_map = {
        "minmod": MinModLimiter(),
        "superbee": SuperBeeLimiter(),
        "vanleer": VanLeerLimiter(),
        "mc": MonotonizedCentralLimiter(),
        "koren": KorenLimiter(),
        "ospre": OspreLimiter()
    }
    
    if limiter_type not in limiter_map:
        available_limiters = list(limiter_map.keys())
        raise ValueError(f"Unknown limiter type: {limiter_type}. Choose from {available_limiters}")
    
    limiter = limiter_map[limiter_type]
    return TVDReconstructor(limiter, **kwargs)


def plot_limiters():
    """Plot limiter functions for comparison."""
    try:
        import matplotlib.pyplot as plt
        
        r_values = np.linspace(-1, 4, 1000)
        
        limiters = {
            "MinMod": MinModLimiter(),
            "SuperBee": SuperBeeLimiter(),
            "Van Leer": VanLeerLimiter(),
            "MC": MonotonizedCentralLimiter(),
            "Koren": KorenLimiter(),
            "OSPRE": OspreLimiter()
        }
        
        plt.figure(figsize=(12, 8))
        
        for name, limiter in limiters.items():
            phi_values = [limiter.limit(r) for r in r_values]
            plt.plot(r_values, phi_values, label=name, linewidth=2)
        
        # TVD region boundaries
        plt.plot(r_values, np.ones_like(r_values), 'k--', alpha=0.5, label='φ = 1')
        plt.plot(r_values, 2 * r_values, 'k--', alpha=0.5, label='φ = 2r')
        
        plt.xlim(0, 4)
        plt.ylim(0, 2.5)
        plt.xlabel('r')
        plt.ylabel('φ(r)')
        plt.title('Flux Limiter Functions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")


def test_tvd_schemes():
    """Test TVD schemes on simple test cases."""
    # Create test data with discontinuity
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], 
                  [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    
    # Step function
    f = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    interface = np.array([1.5, 0.0, 0.0])
    
    # Test different limiters
    limiters = ["minmod", "superbee", "vanleer", "mc"]
    
    print("TVD Scheme Testing:")
    print("Function: Step function at interface x=1.5")
    print("Expected: smooth transition from 1.0 to 0.0")
    print()
    
    for limiter_type in limiters:
        reconstructor = create_tvd_reconstructor(limiter_type)
        left, right = reconstructor.reconstruct_left_right(f, x, interface)
        
        print(f"{limiter_type.upper()}: left={left:.6f}, right={right:.6f}")


if __name__ == "__main__":
    test_tvd_schemes()
    # plot_limiters()  # Uncomment if matplotlib is available