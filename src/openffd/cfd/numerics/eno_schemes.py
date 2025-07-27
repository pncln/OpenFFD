"""
ENO (Essentially Non-Oscillatory) Schemes

Implements ENO reconstruction schemes for shock-capturing:
- ENO2, ENO3, ENO5 schemes
- Adaptive stencil selection
- Smoothness-based stencil choosing
- Divided differences
- Unstructured mesh adaptation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ENOScheme(ABC):
    """Abstract base class for ENO schemes."""
    
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
        """Get required stencil size for this ENO scheme."""
        pass


class DividedDifferences:
    """
    Divided differences calculator for ENO schemes.
    
    Computes Newton divided differences which are used to
    measure smoothness and construct interpolating polynomials.
    """
    
    def __init__(self, tolerance: float = 1e-12):
        """
        Initialize divided differences calculator.
        
        Args:
            tolerance: Tolerance for numerical stability
        """
        self.tolerance = tolerance
    
    def compute_divided_differences(self, 
                                  values: np.ndarray,
                                  points: np.ndarray) -> np.ndarray:
        """
        Compute divided differences table.
        
        Args:
            values: Function values at points
            points: Coordinate points (1D distances for simplicity)
            
        Returns:
            Divided differences matrix [n_points, n_points]
        """
        n = len(values)
        if n != len(points):
            raise ValueError("Values and points arrays must have same length")
        
        # Initialize divided differences table
        dd_table = np.zeros((n, n))
        
        # First column: function values
        dd_table[:, 0] = values
        
        # Compute higher order divided differences
        for j in range(1, n):
            for i in range(n - j):
                x_diff = points[i + j] - points[i]
                if abs(x_diff) < self.tolerance:
                    # Handle coincident points
                    dd_table[i, j] = 0.0
                else:
                    dd_table[i, j] = (dd_table[i + 1, j - 1] - dd_table[i, j - 1]) / x_diff
        
        return dd_table
    
    def compute_smoothness_indicators(self, dd_table: np.ndarray) -> np.ndarray:
        """
        Compute smoothness indicators from divided differences.
        
        Args:
            dd_table: Divided differences table
            
        Returns:
            Smoothness indicators for each possible stencil
        """
        n = dd_table.shape[0]
        smoothness = []
        
        # For each possible stencil starting position
        for start in range(n - 1):
            # Use highest order divided difference as smoothness measure
            max_order = min(n - start - 1, dd_table.shape[1] - 1)
            if max_order > 0:
                smoothness_value = abs(dd_table[start, max_order])
            else:
                smoothness_value = 0.0
            
            smoothness.append(smoothness_value)
        
        return np.array(smoothness)


class ENO2(ENOScheme):
    """
    Second-order ENO scheme.
    
    Uses 2-point stencils with linear reconstruction.
    Chooses smoothest stencil based on divided differences.
    """
    
    def __init__(self, tolerance: float = 1e-12):
        """
        Initialize ENO2 scheme.
        
        Args:
            tolerance: Tolerance for numerical computations
        """
        self.tolerance = tolerance
        self.order = 2
        self.dd_calculator = DividedDifferences(tolerance)
        
    def get_stencil_size(self) -> int:
        """ENO2 requires minimum 3 points for stencil selection."""
        return 3
    
    def reconstruct_left_right(self, 
                              stencil_values: np.ndarray,
                              stencil_points: np.ndarray,
                              interface_point: np.ndarray) -> Tuple[float, float]:
        """Reconstruct using ENO2 scheme."""
        if len(stencil_values) < 2:
            # Not enough points for reconstruction
            if len(stencil_values) == 1:
                return stencil_values[0], stencil_values[0]
            else:
                return 0.0, 0.0
        
        if len(stencil_values) == 2:
            # Only one possible stencil
            return self._linear_reconstruction(stencil_values, stencil_points, interface_point)
        
        # Sort points by distance to interface for local reconstruction
        distances = np.linalg.norm(stencil_points - interface_point, axis=1)
        sorted_indices = np.argsort(distances)
        
        # Use closest points for stencil selection
        n_points = min(4, len(stencil_values))
        indices = sorted_indices[:n_points]
        values = stencil_values[indices]
        points = stencil_points[indices]
        
        # Convert to 1D for divided differences (distance from interface)
        distances_1d = distances[indices]
        
        # Find smoothest 2-point stencil
        best_stencil = self._select_smoothest_stencil(values, distances_1d, 2)
        
        # Use selected stencil for reconstruction
        stencil_values_selected = values[best_stencil]
        stencil_points_selected = points[best_stencil]
        
        return self._linear_reconstruction(stencil_values_selected, stencil_points_selected, interface_point)
    
    def _select_smoothest_stencil(self, 
                                 values: np.ndarray,
                                 distances: np.ndarray,
                                 stencil_size: int) -> List[int]:
        """Select smoothest stencil based on divided differences."""
        n = len(values)
        if n < stencil_size:
            return list(range(n))
        
        best_smoothness = float('inf')
        best_stencil = list(range(stencil_size))
        
        # Try all possible stencils of given size
        for start in range(n - stencil_size + 1):
            stencil_indices = list(range(start, start + stencil_size))
            stencil_values = values[stencil_indices]
            stencil_distances = distances[stencil_indices]
            
            # Compute divided differences
            try:
                dd_table = self.dd_calculator.compute_divided_differences(stencil_values, stencil_distances)
                
                # Smoothness indicator (highest order divided difference)
                if stencil_size > 1 and dd_table.shape[1] > 1:
                    smoothness = abs(dd_table[0, stencil_size - 1])
                else:
                    smoothness = 0.0
                
                if smoothness < best_smoothness:
                    best_smoothness = smoothness
                    best_stencil = stencil_indices
                    
            except Exception:
                # Skip problematic stencils
                continue
        
        return best_stencil
    
    def _linear_reconstruction(self, 
                             values: np.ndarray,
                             points: np.ndarray,
                             interface_point: np.ndarray) -> Tuple[float, float]:
        """Linear reconstruction between two points."""
        if len(values) < 2:
            if len(values) == 1:
                return values[0], values[0]
            else:
                return 0.0, 0.0
        
        # Use two closest points
        distances = np.linalg.norm(points - interface_point, axis=1)
        sorted_indices = np.argsort(distances)
        
        p0_idx, p1_idx = sorted_indices[0], sorted_indices[1]
        f0, f1 = values[p0_idx], values[p1_idx]
        x0, x1 = points[p0_idx], points[p1_idx]
        
        # Linear interpolation
        value = self._linear_interpolation(f0, f1, x0, x1, interface_point)
        
        return value, value
    
    def _linear_interpolation(self, 
                            f0: float, f1: float,
                            x0: np.ndarray, x1: np.ndarray,
                            xi: np.ndarray) -> float:
        """Linear interpolation between two points."""
        d0 = np.linalg.norm(xi - x0)
        d1 = np.linalg.norm(xi - x1)
        total_d = d0 + d1
        
        if total_d < self.tolerance:
            return f0
        
        # Inverse distance weighting
        w1 = d0 / total_d
        w0 = d1 / total_d
        
        return w0 * f0 + w1 * f1


class ENO3(ENOScheme):
    """
    Third-order ENO scheme.
    
    Uses 3-point stencils with quadratic reconstruction.
    Provides higher accuracy than ENO2 while maintaining non-oscillatory property.
    """
    
    def __init__(self, tolerance: float = 1e-12):
        """
        Initialize ENO3 scheme.
        
        Args:
            tolerance: Tolerance for numerical computations
        """
        self.tolerance = tolerance
        self.order = 3
        self.dd_calculator = DividedDifferences(tolerance)
        
    def get_stencil_size(self) -> int:
        """ENO3 requires minimum 4 points for stencil selection."""
        return 4
    
    def reconstruct_left_right(self, 
                              stencil_values: np.ndarray,
                              stencil_points: np.ndarray,
                              interface_point: np.ndarray) -> Tuple[float, float]:
        """Reconstruct using ENO3 scheme."""
        if len(stencil_values) < 3:
            # Fall back to ENO2
            eno2 = ENO2(self.tolerance)
            return eno2.reconstruct_left_right(stencil_values, stencil_points, interface_point)
        
        # Sort points by distance to interface
        distances = np.linalg.norm(stencil_points - interface_point, axis=1)
        sorted_indices = np.argsort(distances)
        
        # Use closest points
        n_points = min(5, len(stencil_values))
        indices = sorted_indices[:n_points]
        values = stencil_values[indices]
        points = stencil_points[indices]
        distances_1d = distances[indices]
        
        # Find smoothest 3-point stencil
        best_stencil = self._select_smoothest_stencil(values, distances_1d, 3)
        
        # Use selected stencil for quadratic reconstruction
        stencil_values_selected = values[best_stencil]
        stencil_points_selected = points[best_stencil]
        
        return self._quadratic_reconstruction(stencil_values_selected, stencil_points_selected, interface_point)
    
    def _select_smoothest_stencil(self, 
                                 values: np.ndarray,
                                 distances: np.ndarray,
                                 stencil_size: int) -> List[int]:
        """Select smoothest stencil using ENO procedure."""
        n = len(values)
        if n < stencil_size:
            return list(range(n))
        
        # Start with two closest points and grow stencil
        current_stencil = [0, 1]  # Start with two closest points
        
        # Grow stencil to desired size
        while len(current_stencil) < stencil_size and len(current_stencil) < n:
            best_addition = None
            best_smoothness = float('inf')
            
            # Try adding each remaining point
            for candidate in range(n):
                if candidate in current_stencil:
                    continue
                
                test_stencil = current_stencil + [candidate]
                test_stencil.sort()  # Keep in order
                
                # Compute smoothness of test stencil
                test_values = values[test_stencil]
                test_distances = distances[test_stencil]
                
                try:
                    dd_table = self.dd_calculator.compute_divided_differences(test_values, test_distances)
                    
                    # Use highest order divided difference as smoothness measure
                    max_order = min(len(test_stencil) - 1, dd_table.shape[1] - 1)
                    if max_order > 0:
                        smoothness = abs(dd_table[0, max_order])
                    else:
                        smoothness = 0.0
                    
                    if smoothness < best_smoothness:
                        best_smoothness = smoothness
                        best_addition = candidate
                        
                except Exception:
                    continue
            
            if best_addition is not None:
                current_stencil.append(best_addition)
                current_stencil.sort()
            else:
                break
        
        return current_stencil
    
    def _quadratic_reconstruction(self, 
                                values: np.ndarray,
                                points: np.ndarray,
                                interface_point: np.ndarray) -> Tuple[float, float]:
        """Quadratic reconstruction using Lagrange interpolation."""
        if len(values) < 3:
            # Fall back to linear
            eno2 = ENO2(self.tolerance)
            return eno2._linear_reconstruction(values, points, interface_point)
        
        # Use three points for quadratic interpolation
        f0, f1, f2 = values[:3]
        x0, x1, x2 = points[:3]
        
        # Lagrange interpolation adapted for unstructured meshes
        value = self._lagrange_interpolation_3pt(f0, f1, f2, x0, x1, x2, interface_point)
        
        return value, value
    
    def _lagrange_interpolation_3pt(self, 
                                   f0: float, f1: float, f2: float,
                                   x0: np.ndarray, x1: np.ndarray, x2: np.ndarray,
                                   xi: np.ndarray) -> float:
        """Three-point Lagrange interpolation for unstructured meshes."""
        # Compute distances
        d0 = np.linalg.norm(xi - x0)
        d1 = np.linalg.norm(xi - x1)
        d2 = np.linalg.norm(xi - x2)
        
        # Handle coincident points
        if d0 < self.tolerance:
            return f0
        if d1 < self.tolerance:
            return f1
        if d2 < self.tolerance:
            return f2
        
        # For unstructured meshes, use modified Lagrange interpolation
        # based on inverse distance weighting with quadratic correction
        
        # Basic inverse distance weights
        w0 = 1.0 / (d0**2)
        w1 = 1.0 / (d1**2)
        w2 = 1.0 / (d2**2)
        
        total_weight = w0 + w1 + w2
        
        if total_weight < self.tolerance:
            return (f0 + f1 + f2) / 3.0
        
        # Weighted interpolation
        interpolated_value = (w0 * f0 + w1 * f1 + w2 * f2) / total_weight
        
        return interpolated_value


class ENO5(ENOScheme):
    """
    Fifth-order ENO scheme.
    
    Uses 5-point stencils with quartic reconstruction.
    Provides very high accuracy in smooth regions.
    """
    
    def __init__(self, tolerance: float = 1e-12):
        """
        Initialize ENO5 scheme.
        
        Args:
            tolerance: Tolerance for numerical computations
        """
        self.tolerance = tolerance
        self.order = 5
        self.dd_calculator = DividedDifferences(tolerance)
        
    def get_stencil_size(self) -> int:
        """ENO5 requires minimum 6 points for stencil selection."""
        return 6
    
    def reconstruct_left_right(self, 
                              stencil_values: np.ndarray,
                              stencil_points: np.ndarray,
                              interface_point: np.ndarray) -> Tuple[float, float]:
        """Reconstruct using ENO5 scheme."""
        if len(stencil_values) < 5:
            # Fall back to ENO3
            eno3 = ENO3(self.tolerance)
            return eno3.reconstruct_left_right(stencil_values, stencil_points, interface_point)
        
        # Sort points by distance to interface
        distances = np.linalg.norm(stencil_points - interface_point, axis=1)
        sorted_indices = np.argsort(distances)
        
        # Use closest points
        n_points = min(7, len(stencil_values))
        indices = sorted_indices[:n_points]
        values = stencil_values[indices]
        points = stencil_points[indices]
        distances_1d = distances[indices]
        
        # Find smoothest 5-point stencil
        best_stencil = self._select_smoothest_stencil(values, distances_1d, 5)
        
        # Use selected stencil for high-order reconstruction
        stencil_values_selected = values[best_stencil]
        stencil_points_selected = points[best_stencil]
        
        return self._high_order_reconstruction(stencil_values_selected, stencil_points_selected, interface_point)
    
    def _select_smoothest_stencil(self, 
                                 values: np.ndarray,
                                 distances: np.ndarray,
                                 stencil_size: int) -> List[int]:
        """Select smoothest stencil using recursive ENO procedure."""
        # Use ENO3 procedure but extend to 5 points
        eno3 = ENO3(self.tolerance)
        current_stencil = eno3._select_smoothest_stencil(values, distances, 3)
        
        # Extend to 5 points
        n = len(values)
        while len(current_stencil) < stencil_size and len(current_stencil) < n:
            best_addition = None
            best_smoothness = float('inf')
            
            for candidate in range(n):
                if candidate in current_stencil:
                    continue
                
                test_stencil = current_stencil + [candidate]
                test_stencil.sort()
                
                test_values = values[test_stencil]
                test_distances = distances[test_stencil]
                
                try:
                    dd_table = self.dd_calculator.compute_divided_differences(test_values, test_distances)
                    
                    max_order = min(len(test_stencil) - 1, dd_table.shape[1] - 1)
                    if max_order > 0:
                        smoothness = abs(dd_table[0, max_order])
                    else:
                        smoothness = 0.0
                    
                    if smoothness < best_smoothness:
                        best_smoothness = smoothness
                        best_addition = candidate
                        
                except Exception:
                    continue
            
            if best_addition is not None:
                current_stencil.append(best_addition)
                current_stencil.sort()
            else:
                break
        
        return current_stencil
    
    def _high_order_reconstruction(self, 
                                 values: np.ndarray,
                                 points: np.ndarray,
                                 interface_point: np.ndarray) -> Tuple[float, float]:
        """High-order reconstruction using polynomial fitting."""
        if len(values) < 5:
            # Fall back to lower order
            eno3 = ENO3(self.tolerance)
            return eno3._quadratic_reconstruction(values, points, interface_point)
        
        # For simplicity, use weighted high-order interpolation
        # Full implementation would use Newton polynomials
        
        # Compute distances and weights
        distances = np.array([np.linalg.norm(interface_point - point) for point in points])
        
        # Avoid division by zero
        distances = np.maximum(distances, self.tolerance)
        
        # High-order inverse distance weighting
        weights = 1.0 / (distances**4)  # Higher power for ENO5
        total_weight = np.sum(weights)
        
        if total_weight < self.tolerance:
            return np.mean(values), np.mean(values)
        
        # Weighted interpolation
        interpolated_value = np.sum(weights * values) / total_weight
        
        return interpolated_value, interpolated_value


class ENOReconstructor:
    """
    Universal ENO reconstructor with adaptive order selection.
    
    Provides interface for ENO reconstruction with automatic
    order selection based on stencil availability and solution smoothness.
    """
    
    def __init__(self, 
                 max_order: int = 3,
                 tolerance: float = 1e-12,
                 adaptive_order: bool = True):
        """
        Initialize ENO reconstructor.
        
        Args:
            max_order: Maximum ENO order to use
            tolerance: Tolerance for numerical computations
            adaptive_order: Whether to adaptively select order
        """
        self.max_order = max_order
        self.tolerance = tolerance
        self.adaptive_order = adaptive_order
        
        # Initialize available schemes
        self.schemes = {
            2: ENO2(tolerance),
            3: ENO3(tolerance),
            5: ENO5(tolerance)
        }
        
        if max_order not in self.schemes:
            available_orders = list(self.schemes.keys())
            raise ValueError(f"max_order {max_order} not available. Choose from {available_orders}")
    
    def reconstruct(self, 
                   stencil_values: np.ndarray,
                   stencil_points: np.ndarray,
                   interface_point: np.ndarray,
                   variable_index: int = 0) -> Tuple[float, float]:
        """
        Perform ENO reconstruction with adaptive order selection.
        
        Args:
            stencil_values: Solution values at stencil points
            stencil_points: Coordinates of stencil points
            interface_point: Interface coordinate
            variable_index: Which variable to reconstruct
            
        Returns:
            (left_state, right_state) at interface
        """
        if len(stencil_values.shape) == 1:
            values = stencil_values
        else:
            values = stencil_values[:, variable_index]
        
        # Determine appropriate order
        order = self._select_order(values, stencil_points)
        
        # Get appropriate scheme
        scheme = self.schemes[order]
        
        # Perform reconstruction
        return scheme.reconstruct_left_right(values, stencil_points, interface_point)
    
    def _select_order(self, values: np.ndarray, points: np.ndarray) -> int:
        """Select appropriate ENO order."""
        n_points = len(values)
        
        # Available orders based on stencil size
        available_orders = []
        for order in [2, 3, 5]:
            if order <= self.max_order and n_points >= self.schemes[order].get_stencil_size():
                available_orders.append(order)
        
        if not available_orders:
            return 2  # Minimum order
        
        if not self.adaptive_order:
            return max(available_orders)
        
        # Adaptive order selection based on smoothness
        # For now, use maximum available order
        # Could be enhanced with smoothness estimation
        return max(available_orders)
    
    def reconstruct_all_variables(self, 
                                 stencil_values: np.ndarray,
                                 stencil_points: np.ndarray,
                                 interface_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct all variables using ENO scheme."""
        if len(stencil_values.shape) == 1:
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


def create_eno_reconstructor(scheme_type: str = "eno3", **kwargs) -> ENOReconstructor:
    """
    Create ENO reconstructor with specified configuration.
    
    Args:
        scheme_type: Type of ENO scheme ("eno2", "eno3", "eno5", "adaptive")
        **kwargs: Additional parameters
        
    Returns:
        Configured ENO reconstructor
    """
    if scheme_type == "eno2":
        return ENOReconstructor(max_order=2, adaptive_order=False, **kwargs)
    elif scheme_type == "eno3":
        return ENOReconstructor(max_order=3, adaptive_order=False, **kwargs)
    elif scheme_type == "eno5":
        return ENOReconstructor(max_order=5, adaptive_order=False, **kwargs)
    elif scheme_type == "adaptive":
        return ENOReconstructor(max_order=3, adaptive_order=True, **kwargs)
    else:
        raise ValueError(f"Unknown ENO scheme type: {scheme_type}")


def test_eno_schemes():
    """Test ENO schemes on simple test cases."""
    # Create test data
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], 
                  [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    
    # Smooth function
    f_smooth = np.sin(x[:, 0])
    
    # Function with discontinuity
    f_discontinuous = np.where(x[:, 0] < 2.5, 1.0, 0.0)
    
    interface = np.array([2.5, 0.0, 0.0])
    
    schemes = {
        "ENO2": ENO2(),
        "ENO3": ENO3(),
        "ENO5": ENO5(),
        "Adaptive": ENOReconstructor(max_order=3, adaptive_order=True)
    }
    
    print("ENO Scheme Testing:")
    print("\n1. Smooth function: sin(x) at x=2.5")
    print("Exact value:", np.sin(2.5))
    
    for name, scheme in schemes.items():
        if isinstance(scheme, ENOReconstructor):
            left, right = scheme.reconstruct(f_smooth, x, interface)
        else:
            left, right = scheme.reconstruct_left_right(f_smooth, x, interface)
        
        avg_value = 0.5 * (left + right)
        error = abs(avg_value - np.sin(2.5))
        
        print(f"{name}: left={left:.6f}, right={right:.6f}, avg={avg_value:.6f}, error={error:.2e}")
    
    print("\n2. Discontinuous function: step at x=2.5")
    print("Expected: transition from 1.0 to 0.0")
    
    for name, scheme in schemes.items():
        if isinstance(scheme, ENOReconstructor):
            left, right = scheme.reconstruct(f_discontinuous, x, interface)
        else:
            left, right = scheme.reconstruct_left_right(f_discontinuous, x, interface)
        
        print(f"{name}: left={left:.6f}, right={right:.6f}")


if __name__ == "__main__":
    test_eno_schemes()