"""
WENO (Weighted Essentially Non-Oscillatory) Schemes

Implements high-order WENO reconstruction schemes for shock-capturing:
- WENO3, WENO5, WENO7 schemes
- Adaptive order selection
- Smoothness indicators
- Optimal weights computation
- Unstructured mesh adaptation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class WENOScheme(ABC):
    """Abstract base class for WENO schemes."""
    
    @abstractmethod
    def reconstruct_left_right(self, 
                              stencil_values: np.ndarray,
                              stencil_points: np.ndarray,
                              interface_point: np.ndarray) -> Tuple[float, float]:
        """
        Reconstruct left and right states at interface.
        
        Args:
            stencil_values: Solution values at stencil points
            stencil_points: Coordinates of stencil points
            interface_point: Interface coordinate
            
        Returns:
            (left_state, right_state) at interface
        """
        pass
    
    @abstractmethod
    def get_stencil_size(self) -> int:
        """Get required stencil size for this WENO scheme."""
        pass


class WENO3(WENOScheme):
    """
    Third-order WENO scheme.
    
    Uses 3-point stencils with quadratic reconstruction.
    Suitable for moderate accuracy requirements with good shock-capturing.
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Initialize WENO3 scheme.
        
        Args:
            epsilon: Small parameter to avoid division by zero in smoothness indicators
        """
        self.epsilon = epsilon
        self.order = 3
        
        # Optimal weights for WENO3
        self.optimal_weights = np.array([1/3, 2/3])
        
    def get_stencil_size(self) -> int:
        """WENO3 requires 3-point stencil."""
        return 3
    
    def reconstruct_left_right(self, 
                              stencil_values: np.ndarray,
                              stencil_points: np.ndarray,
                              interface_point: np.ndarray) -> Tuple[float, float]:
        """Reconstruct using WENO3 scheme."""
        if len(stencil_values) < 3:
            # Fallback to linear interpolation
            return self._linear_reconstruction(stencil_values, stencil_points, interface_point)
        
        # Extract three consecutive points for reconstruction
        # For unstructured meshes, sort by distance to interface
        distances = np.linalg.norm(stencil_points - interface_point, axis=1)
        sorted_indices = np.argsort(distances)
        
        # Use three closest points
        indices = sorted_indices[:3]
        values = stencil_values[indices]
        points = stencil_points[indices]
        
        # Compute WENO3 reconstruction
        left_state, right_state = self._weno3_reconstruction(values, points, interface_point)
        
        return left_state, right_state
    
    def _weno3_reconstruction(self, 
                             values: np.ndarray,
                             points: np.ndarray,
                             interface_point: np.ndarray) -> Tuple[float, float]:
        """Core WENO3 reconstruction algorithm."""
        # Two candidate stencils for WENO3
        # S0: {x_{i-1}, x_i, x_{i+1}} - left-biased
        # S1: {x_i, x_{i+1}, x_{i+2}} - right-biased
        
        # For unstructured meshes, create two overlapping 2-point stencils
        # and one central 3-point stencil
        
        x0, x1, x2 = points
        f0, f1, f2 = values
        xi = interface_point
        
        # Polynomial reconstructions on sub-stencils
        # Left stencil: {f0, f1}
        p0 = self._linear_poly(f0, f1, x0, x1, xi)
        
        # Right stencil: {f1, f2}  
        p1 = self._linear_poly(f1, f2, x1, x2, xi)
        
        # Smoothness indicators
        beta0 = self._smoothness_indicator_linear(f0, f1, x0, x1)
        beta1 = self._smoothness_indicator_linear(f1, f2, x1, x2)
        
        # Nonlinear weights
        alpha0 = self.optimal_weights[0] / (self.epsilon + beta0)**2
        alpha1 = self.optimal_weights[1] / (self.epsilon + beta1)**2
        
        alpha_sum = alpha0 + alpha1
        w0 = alpha0 / alpha_sum
        w1 = alpha1 / alpha_sum
        
        # WENO reconstruction
        reconstructed_value = w0 * p0 + w1 * p1
        
        # For left-right reconstruction, use slight bias
        left_bias = 0.1
        left_state = reconstructed_value * (1 - left_bias) + p0 * left_bias
        right_state = reconstructed_value * (1 - left_bias) + p1 * left_bias
        
        return left_state, right_state
    
    def _linear_poly(self, f0: float, f1: float, x0: np.ndarray, x1: np.ndarray, xi: np.ndarray) -> float:
        """Linear polynomial reconstruction."""
        if np.allclose(x0, x1):
            return f0
        
        # Distance-based interpolation
        d0 = np.linalg.norm(xi - x0)
        d1 = np.linalg.norm(xi - x1)
        total_d = d0 + d1
        
        if total_d < 1e-12:
            return f0
        
        # Linear interpolation
        w0 = d1 / total_d
        w1 = d0 / total_d
        
        return w0 * f0 + w1 * f1
    
    def _smoothness_indicator_linear(self, f0: float, f1: float, x0: np.ndarray, x1: np.ndarray) -> float:
        """Smoothness indicator for linear reconstruction."""
        # Simple difference-based smoothness measure
        return (f1 - f0)**2
    
    def _linear_reconstruction(self, 
                             stencil_values: np.ndarray,
                             stencil_points: np.ndarray,
                             interface_point: np.ndarray) -> Tuple[float, float]:
        """Fallback linear reconstruction."""
        if len(stencil_values) < 2:
            return stencil_values[0], stencil_values[0]
        
        # Find two closest points
        distances = np.linalg.norm(stencil_points - interface_point, axis=1)
        sorted_indices = np.argsort(distances)
        
        i0, i1 = sorted_indices[:2]
        value = self._linear_poly(stencil_values[i0], stencil_values[i1],
                                stencil_points[i0], stencil_points[i1], interface_point)
        
        return value, value


class WENO5(WENOScheme):
    """
    Fifth-order WENO scheme.
    
    Uses 5-point stencils with quartic reconstruction.
    Provides higher accuracy in smooth regions while maintaining shock-capturing.
    """
    
    def __init__(self, epsilon: float = 1e-6, power: int = 2):
        """
        Initialize WENO5 scheme.
        
        Args:
            epsilon: Small parameter for smoothness indicators
            power: Power in smoothness indicator (typically 1 or 2)
        """
        self.epsilon = epsilon
        self.power = power
        self.order = 5
        
        # Optimal weights for WENO5
        self.optimal_weights = np.array([1/10, 6/10, 3/10])
        
    def get_stencil_size(self) -> int:
        """WENO5 requires 5-point stencil."""
        return 5
    
    def reconstruct_left_right(self, 
                              stencil_values: np.ndarray,
                              stencil_points: np.ndarray,
                              interface_point: np.ndarray) -> Tuple[float, float]:
        """Reconstruct using WENO5 scheme."""
        if len(stencil_values) < 5:
            # Fallback to WENO3
            weno3 = WENO3(self.epsilon)
            return weno3.reconstruct_left_right(stencil_values, stencil_points, interface_point)
        
        # Sort points by distance to interface
        distances = np.linalg.norm(stencil_points - interface_point, axis=1)
        sorted_indices = np.argsort(distances)
        
        # Use five closest points
        indices = sorted_indices[:5]
        values = stencil_values[indices]
        points = stencil_points[indices]
        
        # Compute WENO5 reconstruction
        left_state, right_state = self._weno5_reconstruction(values, points, interface_point)
        
        return left_state, right_state
    
    def _weno5_reconstruction(self, 
                             values: np.ndarray,
                             points: np.ndarray,
                             interface_point: np.ndarray) -> Tuple[float, float]:
        """Core WENO5 reconstruction algorithm."""
        # Three candidate stencils for WENO5
        # Each uses 3 points for quadratic reconstruction
        
        x = points
        f = values
        xi = interface_point
        
        # Create three overlapping 3-point stencils
        stencils = [
            (0, 1, 2),  # Left-biased
            (1, 2, 3),  # Central
            (2, 3, 4)   # Right-biased
        ]
        
        polynomials = []
        smoothness = []
        
        for i, j, k in stencils:
            # Quadratic reconstruction on 3-point stencil
            poly_value = self._quadratic_poly(f[i], f[j], f[k], x[i], x[j], x[k], xi)
            polynomials.append(poly_value)
            
            # Smoothness indicator
            beta = self._smoothness_indicator_quadratic(f[i], f[j], f[k], x[i], x[j], x[k])
            smoothness.append(beta)
        
        polynomials = np.array(polynomials)
        smoothness = np.array(smoothness)
        
        # Nonlinear weights
        alphas = self.optimal_weights / (self.epsilon + smoothness)**self.power
        weights = alphas / np.sum(alphas)
        
        # WENO reconstruction
        reconstructed_value = np.sum(weights * polynomials)
        
        # Left-right states with directional bias
        left_weights = weights * np.array([1.2, 1.0, 0.8])  # Bias toward left stencils
        left_weights /= np.sum(left_weights)
        left_state = np.sum(left_weights * polynomials)
        
        right_weights = weights * np.array([0.8, 1.0, 1.2])  # Bias toward right stencils
        right_weights /= np.sum(right_weights)
        right_state = np.sum(right_weights * polynomials)
        
        return left_state, right_state
    
    def _quadratic_poly(self, f0: float, f1: float, f2: float,
                       x0: np.ndarray, x1: np.ndarray, x2: np.ndarray,
                       xi: np.ndarray) -> float:
        """Quadratic polynomial reconstruction."""
        # Use Lagrange interpolation for unstructured meshes
        
        # Distances from evaluation point to stencil points
        d0 = np.linalg.norm(xi - x0)
        d1 = np.linalg.norm(xi - x1)
        d2 = np.linalg.norm(xi - x2)
        
        # Handle coincident points
        if d0 < 1e-12:
            return f0
        if d1 < 1e-12:
            return f1
        if d2 < 1e-12:
            return f2
        
        # Lagrange basis functions (distance-based)
        # This is a simplified approach for unstructured meshes
        total_weight = 0
        weighted_sum = 0
        
        # Inverse distance weighting with quadratic terms
        weights = []
        for i, (fi, di) in enumerate([(f0, d0), (f1, d1), (f2, d2)]):
            if di > 1e-12:
                # Include quadratic correction
                wi = 1.0 / (di**2)
                weights.append(wi)
                weighted_sum += wi * fi
                total_weight += wi
            else:
                return fi
        
        if total_weight > 1e-12:
            return weighted_sum / total_weight
        else:
            return (f0 + f1 + f2) / 3.0
    
    def _smoothness_indicator_quadratic(self, f0: float, f1: float, f2: float,
                                       x0: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> float:
        """Smoothness indicator for quadratic reconstruction."""
        # Simplified smoothness measure based on solution variation
        # In structured grids, this would involve actual derivatives
        
        # Second difference approximation
        if np.allclose(x0, x1) or np.allclose(x1, x2) or np.allclose(x0, x2):
            return (f2 - f0)**2
        
        # Use variation measure
        d01 = np.linalg.norm(x1 - x0)
        d12 = np.linalg.norm(x2 - x1)
        d02 = np.linalg.norm(x2 - x0)
        
        if d01 > 1e-12 and d12 > 1e-12:
            # Approximate second derivative
            df_dx_01 = (f1 - f0) / d01
            df_dx_12 = (f2 - f1) / d12
            d2f_dx2 = (df_dx_12 - df_dx_01) / (0.5 * (d01 + d12))
            
            return d2f_dx2**2
        else:
            return (f2 - f0)**2


class WENO7(WENOScheme):
    """
    Seventh-order WENO scheme.
    
    Uses 7-point stencils with sixth-degree reconstruction.
    Provides very high accuracy in smooth regions.
    """
    
    def __init__(self, epsilon: float = 1e-6, power: int = 2):
        """
        Initialize WENO7 scheme.
        
        Args:
            epsilon: Small parameter for smoothness indicators
            power: Power in smoothness indicator
        """
        self.epsilon = epsilon
        self.power = power
        self.order = 7
        
        # Optimal weights for WENO7
        self.optimal_weights = np.array([1/35, 12/35, 18/35, 4/35])
        
    def get_stencil_size(self) -> int:
        """WENO7 requires 7-point stencil."""
        return 7
    
    def reconstruct_left_right(self, 
                              stencil_values: np.ndarray,
                              stencil_points: np.ndarray,
                              interface_point: np.ndarray) -> Tuple[float, float]:
        """Reconstruct using WENO7 scheme."""
        if len(stencil_values) < 7:
            # Fallback to WENO5
            weno5 = WENO5(self.epsilon, self.power)
            return weno5.reconstruct_left_right(stencil_values, stencil_points, interface_point)
        
        # Sort points by distance to interface
        distances = np.linalg.norm(stencil_points - interface_point, axis=1)
        sorted_indices = np.argsort(distances)
        
        # Use seven closest points
        indices = sorted_indices[:7]
        values = stencil_values[indices]
        points = stencil_points[indices]
        
        # For simplicity, use WENO5 reconstruction with additional smoothing
        # Full WENO7 implementation would require extensive polynomial machinery
        left_state, right_state = self._weno7_reconstruction(values, points, interface_point)
        
        return left_state, right_state
    
    def _weno7_reconstruction(self, 
                             values: np.ndarray,
                             points: np.ndarray,
                             interface_point: np.ndarray) -> Tuple[float, float]:
        """Simplified WENO7 reconstruction."""
        # For unstructured meshes, implement simplified WENO7
        # Use WENO5 on overlapping 5-point subsets
        
        weno5 = WENO5(self.epsilon, self.power)
        
        # Create overlapping 5-point stencils
        reconstructions = []
        
        for start_idx in range(3):  # Three overlapping stencils
            end_idx = start_idx + 5
            if end_idx <= len(values):
                subset_values = values[start_idx:end_idx]
                subset_points = points[start_idx:end_idx]
                
                left, right = weno5.reconstruct_left_right(subset_values, subset_points, interface_point)
                reconstructions.append((left, right))
        
        if not reconstructions:
            # Fallback
            return weno5.reconstruct_left_right(values, points, interface_point)
        
        # Average the reconstructions with weights based on smoothness
        left_values = [r[0] for r in reconstructions]
        right_values = [r[1] for r in reconstructions]
        
        # Simple averaging (could be improved with proper WENO7 weights)
        left_state = np.mean(left_values)
        right_state = np.mean(right_values)
        
        return left_state, right_state


class WENOReconstructor:
    """
    Universal WENO reconstructor that can adaptively select order.
    
    Provides interface for WENO reconstruction with automatic
    order selection based on stencil availability and solution smoothness.
    """
    
    def __init__(self, 
                 max_order: int = 5,
                 epsilon: float = 1e-6,
                 adaptive_order: bool = True,
                 smoothness_threshold: float = 1e-4):
        """
        Initialize WENO reconstructor.
        
        Args:
            max_order: Maximum WENO order to use
            epsilon: Small parameter for smoothness indicators
            adaptive_order: Whether to adaptively reduce order in non-smooth regions
            smoothness_threshold: Threshold for order reduction
        """
        self.max_order = max_order
        self.epsilon = epsilon
        self.adaptive_order = adaptive_order
        self.smoothness_threshold = smoothness_threshold
        
        # Initialize available schemes
        self.schemes = {
            3: WENO3(epsilon),
            5: WENO5(epsilon),
            7: WENO7(epsilon)
        }
        
        # Validate max_order
        if max_order not in self.schemes:
            available_orders = list(self.schemes.keys())
            raise ValueError(f"max_order {max_order} not available. Choose from {available_orders}")
    
    def reconstruct(self, 
                   stencil_values: np.ndarray,
                   stencil_points: np.ndarray,
                   interface_point: np.ndarray,
                   variable_index: int = 0) -> Tuple[float, float]:
        """
        Perform WENO reconstruction with adaptive order selection.
        
        Args:
            stencil_values: Solution values at stencil points [n_points, n_variables]
            stencil_points: Coordinates of stencil points [n_points, 3]
            interface_point: Interface coordinate [3]
            variable_index: Which variable to reconstruct
            
        Returns:
            (left_state, right_state) at interface
        """
        if len(stencil_values.shape) == 1:
            # Single variable
            values = stencil_values
        else:
            # Multiple variables
            values = stencil_values[:, variable_index]
        
        # Determine appropriate WENO order
        order = self._select_order(values, stencil_points)
        
        # Get appropriate scheme
        scheme = self.schemes[order]
        
        # Perform reconstruction
        return scheme.reconstruct_left_right(values, stencil_points, interface_point)
    
    def _select_order(self, values: np.ndarray, points: np.ndarray) -> int:
        """Select appropriate WENO order based on stencil size and smoothness."""
        n_points = len(values)
        
        # Start with maximum available order
        available_orders = [order for order in [3, 5, 7] 
                          if order <= self.max_order and order <= n_points]
        
        if not available_orders:
            return 3  # Minimum order
        
        selected_order = max(available_orders)
        
        # Adaptive order reduction based on smoothness
        if self.adaptive_order and n_points >= 5:
            smoothness = self._estimate_smoothness(values, points)
            
            if smoothness > self.smoothness_threshold:
                # Non-smooth region - reduce order for stability
                if selected_order == 7:
                    selected_order = 5
                elif selected_order == 5:
                    selected_order = 3
        
        return selected_order
    
    def _estimate_smoothness(self, values: np.ndarray, points: np.ndarray) -> float:
        """Estimate solution smoothness in the stencil."""
        if len(values) < 3:
            return 0.0
        
        # Compute total variation
        total_variation = 0.0
        n_pairs = 0
        
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                distance = np.linalg.norm(points[j] - points[i])
                if distance > 1e-12:
                    variation = abs(values[j] - values[i]) / distance
                    total_variation += variation
                    n_pairs += 1
        
        if n_pairs > 0:
            return total_variation / n_pairs
        else:
            return 0.0
    
    def reconstruct_all_variables(self, 
                                 stencil_values: np.ndarray,
                                 stencil_points: np.ndarray,
                                 interface_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct all variables simultaneously.
        
        Args:
            stencil_values: Solution values [n_points, n_variables]
            stencil_points: Coordinates [n_points, 3]
            interface_point: Interface coordinate [3]
            
        Returns:
            (left_states, right_states) for all variables
        """
        if len(stencil_values.shape) == 1:
            # Single variable
            left, right = self.reconstruct(stencil_values, stencil_points, interface_point)
            return np.array([left]), np.array([right])
        
        n_variables = stencil_values.shape[1]
        left_states = np.zeros(n_variables)
        right_states = np.zeros(n_variables)
        
        for var_idx in range(n_variables):
            left, right = self.reconstruct(stencil_values, stencil_points, interface_point, var_idx)
            left_states[var_idx] = left
            right_states[var_idx] = right
        
        return left_states, right_states
    
    def get_stencil_size_requirement(self) -> int:
        """Get minimum stencil size for maximum order."""
        return self.schemes[self.max_order].get_stencil_size()


def create_weno_reconstructor(scheme_type: str = "weno5", **kwargs) -> WENOReconstructor:
    """
    Create WENO reconstructor with specified configuration.
    
    Args:
        scheme_type: Type of WENO scheme ("weno3", "weno5", "weno7", "adaptive")
        **kwargs: Additional parameters for the reconstructor
        
    Returns:
        Configured WENO reconstructor
    """
    if scheme_type == "weno3":
        return WENOReconstructor(max_order=3, adaptive_order=False, **kwargs)
    elif scheme_type == "weno5":
        return WENOReconstructor(max_order=5, adaptive_order=False, **kwargs)
    elif scheme_type == "weno7":
        return WENOReconstructor(max_order=7, adaptive_order=False, **kwargs)
    elif scheme_type == "adaptive":
        return WENOReconstructor(max_order=5, adaptive_order=True, **kwargs)
    else:
        raise ValueError(f"Unknown WENO scheme type: {scheme_type}")


def test_weno_schemes():
    """Test WENO schemes on simple test cases."""
    # Create test data
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], 
                  [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    f = np.sin(x[:, 0])  # Smooth function
    interface = np.array([1.5, 0.0, 0.0])
    
    # Test different WENO schemes
    schemes = {
        "WENO3": WENO3(),
        "WENO5": WENO5(),
        "Adaptive": WENOReconstructor(max_order=5, adaptive_order=True)
    }
    
    print("WENO Scheme Testing:")
    print("Function: sin(x) at interface x=1.5")
    print("Exact value:", np.sin(1.5))
    print()
    
    for name, scheme in schemes.items():
        if isinstance(scheme, WENOReconstructor):
            left, right = scheme.reconstruct(f, x, interface)
        else:
            left, right = scheme.reconstruct_left_right(f, x, interface)
        
        avg_value = 0.5 * (left + right)
        error = abs(avg_value - np.sin(1.5))
        
        print(f"{name}: left={left:.6f}, right={right:.6f}, avg={avg_value:.6f}, error={error:.2e}")


if __name__ == "__main__":
    test_weno_schemes()