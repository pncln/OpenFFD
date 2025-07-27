"""
Slope Limiters and Flux Limiters for Monotonicity Preservation

Implements various limiting techniques for high-resolution shock-capturing schemes:
- Slope limiters (minmod, superbee, van Leer, MUSCL)
- Flux limiters for TVD schemes
- Multi-dimensional limiters for unstructured grids
- Shock detection and adaptive limiting

These limiters ensure monotonicity and prevent spurious oscillations
near shocks and other discontinuities in supersonic flows.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LimiterConfig:
    """Configuration for slope and flux limiters."""
    
    # Limiter selection
    limiter_type: str = "minmod"  # "minmod", "superbee", "vanleer", "muscl", "venkatakrishnan"
    flux_limiter_type: str = "superbee"  # For TVD schemes
    
    # Parameters
    epsilon: float = 1e-12  # Small number to avoid division by zero
    kappa: float = 1.0/3.0  # MUSCL parameter (1/3 for 3rd order, 0 for 2nd order)
    venkat_k: float = 5.0  # Venkatakrishnan limiter parameter
    
    # Multi-dimensional limiting
    use_multidim_limiting: bool = True
    directional_bias: bool = False  # Bias limiting in flow direction
    
    # Shock detection
    use_shock_detection: bool = True
    shock_threshold: float = 0.1  # Pressure gradient threshold for shock detection
    disable_limiting_smooth: bool = True  # Disable limiting in smooth regions
    
    # Performance
    vectorized_computation: bool = True
    use_precomputed_ratios: bool = True


class SlopeLimiter(ABC):
    """Abstract base class for slope limiters."""
    
    def __init__(self, config: Optional[LimiterConfig] = None):
        """Initialize slope limiter."""
        self.config = config or LimiterConfig()
        self.name = "base_limiter"
        
    @abstractmethod
    def limit_slope(self, 
                   left_gradient: np.ndarray,
                   right_gradient: np.ndarray,
                   cell_values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply slope limiting to gradients.
        
        Args:
            left_gradient: Gradient from left neighbor
            right_gradient: Gradient from right neighbor
            cell_values: Cell values for additional limiting criteria
            
        Returns:
            Limited gradient
        """
        pass
    
    def compute_ratio(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute ratio r = a/b with division by zero protection."""
        epsilon = self.config.epsilon
        denominator = np.where(np.abs(b) < epsilon, epsilon, b)
        return a / denominator


class MinmodLimiter(SlopeLimiter):
    """
    Minmod limiter - most diffusive but robust.
    
    φ(r) = max(0, min(1, r))
    """
    
    def __init__(self, config: Optional[LimiterConfig] = None):
        super().__init__(config)
        self.name = "minmod"
    
    def limit_slope(self,
                   left_gradient: np.ndarray,
                   right_gradient: np.ndarray,
                   cell_values: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply minmod limiting."""
        # Minmod function: min(|a|, |b|) * sign(a) if sign(a) == sign(b), else 0
        limited = np.where(
            left_gradient * right_gradient > 0,
            np.where(
                np.abs(left_gradient) < np.abs(right_gradient),
                left_gradient,
                right_gradient
            ),
            0.0
        )
        return limited


class SuperbeeLimiter(SlopeLimiter):
    """
    Superbee limiter - less diffusive, sharper shock resolution.
    
    φ(r) = max(0, min(2r, 1), min(r, 2))
    """
    
    def __init__(self, config: Optional[LimiterConfig] = None):
        super().__init__(config)
        self.name = "superbee"
    
    def limit_slope(self,
                   left_gradient: np.ndarray,
                   right_gradient: np.ndarray,
                   cell_values: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply superbee limiting."""
        r = self.compute_ratio(left_gradient, right_gradient)
        
        # Superbee function
        phi = np.maximum(0, np.minimum(2*r, 1))
        phi = np.maximum(phi, np.minimum(r, 2))
        
        return phi * right_gradient


class VanLeerLimiter(SlopeLimiter):
    """
    Van Leer limiter - smooth, good balance of accuracy and monotonicity.
    
    φ(r) = (r + |r|) / (1 + |r|)
    """
    
    def __init__(self, config: Optional[LimiterConfig] = None):
        super().__init__(config)
        self.name = "van_leer"
    
    def limit_slope(self,
                   left_gradient: np.ndarray,
                   right_gradient: np.ndarray,
                   cell_values: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply Van Leer limiting."""
        r = self.compute_ratio(left_gradient, right_gradient)
        
        # Van Leer function
        phi = (r + np.abs(r)) / (1 + np.abs(r))
        
        return phi * right_gradient


class MUSCLLimiter(SlopeLimiter):
    """
    MUSCL limiter with κ parameter for order control.
    
    Provides variable order accuracy (2nd to 3rd order).
    """
    
    def __init__(self, config: Optional[LimiterConfig] = None):
        super().__init__(config)
        self.name = "muscl"
        self.kappa = config.kappa if config else 1.0/3.0
    
    def limit_slope(self,
                   left_gradient: np.ndarray,
                   right_gradient: np.ndarray,
                   cell_values: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply MUSCL limiting."""
        # MUSCL extrapolation with limiting
        r = self.compute_ratio(left_gradient, right_gradient)
        
        # MUSCL limiter function
        phi = np.maximum(0, np.minimum(2*r, np.minimum((1 + r*self.kappa)/(1 + self.kappa), 2)))
        
        return phi * right_gradient


class VenkatakrishnanLimiter(SlopeLimiter):
    """
    Venkatakrishnan limiter - differentiable, good for unstructured grids.
    
    Particularly suitable for adjoint computations due to smoothness.
    """
    
    def __init__(self, config: Optional[LimiterConfig] = None):
        super().__init__(config)
        self.name = "venkatakrishnan"
        self.K = config.venkat_k if config else 5.0
    
    def limit_slope(self,
                   left_gradient: np.ndarray,
                   right_gradient: np.ndarray,
                   cell_values: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply Venkatakrishnan limiting."""
        if cell_values is None:
            # Fallback to Van Leer if no cell values provided
            return VanLeerLimiter(self.config).limit_slope(left_gradient, right_gradient)
        
        # Venkatakrishnan limiter requires local mesh size information
        delta_max = np.max(cell_values) - cell_values
        delta_min = cell_values - np.min(cell_values)
        
        # Cell size estimation (simplified)
        h_squared = np.ones_like(cell_values)  # Should be actual cell size
        
        eps_squared = (self.K * h_squared)**3
        
        # Apply Venkatakrishnan function
        limited = np.where(
            left_gradient * right_gradient > 0,
            self._venkat_function(left_gradient, right_gradient, eps_squared),
            0.0
        )
        
        return limited
    
    def _venkat_function(self, a: np.ndarray, b: np.ndarray, eps_sq: np.ndarray) -> np.ndarray:
        """Venkatakrishnan limiting function."""
        r = self.compute_ratio(a, b)
        phi = (r**2 + 2*r + eps_sq) / (r**2 + r + 2 + eps_sq)
        return phi * b


class MultiDimensionalLimiter:
    """
    Multi-dimensional limiter for unstructured grids.
    
    Extends 1D limiters to multi-dimensional cases with proper
    treatment of grid irregularities and flow directionality.
    """
    
    def __init__(self, base_limiter: SlopeLimiter, config: Optional[LimiterConfig] = None):
        """Initialize multi-dimensional limiter."""
        self.base_limiter = base_limiter
        self.config = config or LimiterConfig()
        
    def limit_gradients(self,
                       cell_gradients: np.ndarray,
                       neighbor_gradients: np.ndarray,
                       face_normals: np.ndarray,
                       cell_values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply multi-dimensional limiting to cell gradients.
        
        Args:
            cell_gradients: Gradients at cell centers [n_cells, n_vars, 3]
            neighbor_gradients: Gradients at neighbors [n_cells, n_neighbors, n_vars, 3]
            face_normals: Face normal vectors [n_cells, n_neighbors, 3]
            cell_values: Cell values for additional criteria
            
        Returns:
            Limited gradients [n_cells, n_vars, 3]
        """
        n_cells, n_vars, n_dims = cell_gradients.shape
        limited_gradients = cell_gradients.copy()
        
        for cell in range(n_cells):
            for var in range(n_vars):
                if self.config.directional_bias:
                    # Apply directional limiting (flow-aligned)
                    limited_gradients[cell, var] = self._limit_directional(
                        cell_gradients[cell, var],
                        neighbor_gradients[cell, :, var],
                        face_normals[cell],
                        cell_values[cell, var] if cell_values is not None else None
                    )
                else:
                    # Apply component-wise limiting
                    for dim in range(n_dims):
                        limited_gradients[cell, var, dim] = self._limit_component(
                            cell_gradients[cell, var, dim],
                            neighbor_gradients[cell, :, var, dim]
                        )
        
        return limited_gradients
    
    def _limit_directional(self,
                          cell_grad: np.ndarray,
                          neighbor_grads: np.ndarray,
                          normals: np.ndarray,
                          cell_value: Optional[float] = None) -> np.ndarray:
        """Apply directional limiting based on face normals."""
        limited_grad = cell_grad.copy()
        
        for neighbor_idx, normal in enumerate(normals):
            if neighbor_idx >= len(neighbor_grads):
                continue
                
            # Project gradients onto face normal
            cell_proj = np.dot(cell_grad, normal)
            neighbor_proj = np.dot(neighbor_grads[neighbor_idx], normal)
            
            # Apply 1D limiting to projected values
            limited_proj = self.base_limiter.limit_slope(
                np.array([cell_proj]), 
                np.array([neighbor_proj]),
                np.array([cell_value]) if cell_value is not None else None
            )[0]
            
            # Reconstruct gradient (simplified - should use proper reconstruction)
            if np.abs(cell_proj) > self.config.epsilon:
                scale_factor = limited_proj / cell_proj
                limited_grad *= scale_factor
        
        return limited_grad
    
    def _limit_component(self, cell_comp: float, neighbor_comps: np.ndarray) -> float:
        """Apply component-wise limiting."""
        if len(neighbor_comps) == 0:
            return cell_comp
        
        # Use average of neighbors as reference
        avg_neighbor = np.mean(neighbor_comps)
        
        limited = self.base_limiter.limit_slope(
            np.array([cell_comp]),
            np.array([avg_neighbor])
        )[0]
        
        return limited


class ShockDetector:
    """
    Shock detection for adaptive limiting.
    
    Identifies regions with strong gradients/shocks to selectively
    apply limiting only where needed.
    """
    
    def __init__(self, config: Optional[LimiterConfig] = None):
        """Initialize shock detector."""
        self.config = config or LimiterConfig()
        
    def detect_shocks(self,
                     conservative_vars: np.ndarray,
                     gradients: np.ndarray,
                     mesh_info: Dict[str, Any]) -> np.ndarray:
        """
        Detect shock regions in the flow field.
        
        Args:
            conservative_vars: Conservative variables [n_cells, 5]
            gradients: Solution gradients [n_cells, 5, 3]
            mesh_info: Mesh connectivity information
            
        Returns:
            Shock indicator [n_cells] (0 = smooth, 1 = shock)
        """
        n_cells = conservative_vars.shape[0]
        shock_indicator = np.zeros(n_cells)
        
        # Extract pressures
        pressures = self._extract_pressures(conservative_vars)
        
        # Compute pressure gradients
        pressure_gradients = gradients[:, 4, :]  # Assuming energy is last variable
        pressure_grad_magnitude = np.linalg.norm(pressure_gradients, axis=1)
        
        # Shock detection based on pressure gradient
        max_pressure = np.max(pressures)
        min_pressure = np.min(pressures)
        pressure_scale = max_pressure - min_pressure + self.config.epsilon
        
        # Normalized pressure gradient
        normalized_grad = pressure_grad_magnitude / pressure_scale
        
        # Mark cells with large pressure gradients as shocks
        shock_indicator = np.where(
            normalized_grad > self.config.shock_threshold,
            1.0,
            0.0
        )
        
        # Additional shock detection criteria
        if hasattr(mesh_info, 'cell_volumes'):
            # Scale by cell size
            cell_sizes = np.power(mesh_info['cell_volumes'], 1.0/3.0)
            scaled_grad = pressure_grad_magnitude * cell_sizes
            shock_indicator = np.maximum(shock_indicator,
                                       np.where(scaled_grad > self.config.shock_threshold, 1.0, 0.0))
        
        return shock_indicator
    
    def _extract_pressures(self, conservative_vars: np.ndarray) -> np.ndarray:
        """Extract pressures from conservative variables."""
        gamma = 1.4
        pressures = np.zeros(conservative_vars.shape[0])
        
        for i in range(len(pressures)):
            rho, rho_u, rho_v, rho_w, rho_E = conservative_vars[i]
            rho = max(rho, 1e-12)
            
            u, v, w = rho_u/rho, rho_v/rho, rho_w/rho
            kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
            pressures[i] = (gamma - 1) * (rho_E - kinetic_energy)
        
        return pressures


class AdaptiveLimiter:
    """
    Adaptive limiter that adjusts limiting strength based on local flow conditions.
    
    Combines shock detection with variable limiting to achieve optimal
    balance between accuracy and monotonicity.
    """
    
    def __init__(self, config: Optional[LimiterConfig] = None):
        """Initialize adaptive limiter."""
        self.config = config or LimiterConfig()
        
        # Create base limiters
        self.shock_limiter = MinmodLimiter(config)  # Conservative for shocks
        self.smooth_limiter = VanLeerLimiter(config)  # Less diffusive for smooth regions
        
        # Shock detector
        self.shock_detector = ShockDetector(config)
        
        # Multi-dimensional limiter
        self.multidim_limiter = MultiDimensionalLimiter(self.shock_limiter, config)
        
    def apply_adaptive_limiting(self,
                              conservative_vars: np.ndarray,
                              gradients: np.ndarray,
                              neighbor_gradients: np.ndarray,
                              face_normals: np.ndarray,
                              mesh_info: Dict[str, Any]) -> np.ndarray:
        """
        Apply adaptive limiting based on local flow conditions.
        
        Args:
            conservative_vars: Conservative variables
            gradients: Cell gradients
            neighbor_gradients: Neighbor gradients
            face_normals: Face normal vectors
            mesh_info: Mesh information
            
        Returns:
            Adaptively limited gradients
        """
        # Detect shock regions
        shock_indicator = self.shock_detector.detect_shocks(
            conservative_vars, gradients, mesh_info
        )
        
        # Initialize limited gradients
        limited_gradients = gradients.copy()
        
        # Apply different limiting strategies
        if self.config.use_multidim_limiting:
            # Multi-dimensional limiting
            shock_cells = shock_indicator > 0.5
            smooth_cells = ~shock_cells
            
            if np.any(shock_cells):
                # Conservative limiting in shock regions
                self.multidim_limiter.base_limiter = self.shock_limiter
                limited_gradients[shock_cells] = self.multidim_limiter.limit_gradients(
                    gradients[shock_cells],
                    neighbor_gradients[shock_cells],
                    face_normals[shock_cells],
                    conservative_vars[shock_cells] if self.config.disable_limiting_smooth else None
                )
            
            if np.any(smooth_cells) and not self.config.disable_limiting_smooth:
                # Less diffusive limiting in smooth regions
                self.multidim_limiter.base_limiter = self.smooth_limiter
                limited_gradients[smooth_cells] = self.multidim_limiter.limit_gradients(
                    gradients[smooth_cells],
                    neighbor_gradients[smooth_cells],
                    face_normals[smooth_cells],
                    conservative_vars[smooth_cells]
                )
        else:
            # Component-wise limiting
            for cell in range(len(conservative_vars)):
                limiter = self.shock_limiter if shock_indicator[cell] > 0.5 else self.smooth_limiter
                
                for var in range(gradients.shape[1]):
                    for dim in range(gradients.shape[2]):
                        if len(neighbor_gradients[cell]) > 0:
                            avg_neighbor = np.mean(neighbor_gradients[cell, :, var, dim])
                            limited_gradients[cell, var, dim] = limiter.limit_slope(
                                np.array([gradients[cell, var, dim]]),
                                np.array([avg_neighbor])
                            )[0]
        
        return limited_gradients


class FluxLimiter:
    """
    Flux limiters for TVD (Total Variation Diminishing) schemes.
    
    Applied at cell faces to ensure TVD property and prevent
    oscillations in high-resolution schemes.
    """
    
    def __init__(self, limiter_type: str = "superbee", config: Optional[LimiterConfig] = None):
        """Initialize flux limiter."""
        self.limiter_type = limiter_type
        self.config = config or LimiterConfig()
        
    def apply_flux_limiting(self,
                          face_fluxes: np.ndarray,
                          upwind_gradients: np.ndarray,
                          downwind_gradients: np.ndarray) -> np.ndarray:
        """
        Apply flux limiting to face fluxes.
        
        Args:
            face_fluxes: Computed face fluxes
            upwind_gradients: Gradients from upwind side
            downwind_gradients: Gradients from downwind side
            
        Returns:
            Limited face fluxes
        """
        n_faces, n_vars = face_fluxes.shape
        limited_fluxes = face_fluxes.copy()
        
        for face in range(n_faces):
            for var in range(n_vars):
                # Compute flux limiter
                limiter_value = self._compute_flux_limiter(
                    upwind_gradients[face, var],
                    downwind_gradients[face, var]
                )
                
                # Apply limiting
                limited_fluxes[face, var] *= limiter_value
        
        return limited_fluxes
    
    def _compute_flux_limiter(self, upwind_grad: float, downwind_grad: float) -> float:
        """Compute flux limiter value."""
        epsilon = self.config.epsilon
        
        # Compute ratio
        if abs(downwind_grad) < epsilon:
            r = 0.0
        else:
            r = upwind_grad / downwind_grad
        
        # Apply limiter function
        if self.limiter_type == "minmod":
            return max(0, min(1, r))
        elif self.limiter_type == "superbee":
            return max(0, min(2*r, 1), min(r, 2))
        elif self.limiter_type == "van_leer":
            return (r + abs(r)) / (1 + abs(r))
        elif self.limiter_type == "mc":  # Monotonized Central
            return max(0, min(2*r, (1+r)/2, 2))
        else:
            # Default to Van Leer
            return (r + abs(r)) / (1 + abs(r))


def create_slope_limiter(limiter_type: str, config: Optional[LimiterConfig] = None) -> SlopeLimiter:
    """
    Factory function for creating slope limiters.
    
    Args:
        limiter_type: Type of limiter
        config: Limiter configuration
        
    Returns:
        Configured slope limiter
    """
    if limiter_type == "minmod":
        return MinmodLimiter(config)
    elif limiter_type == "superbee":
        return SuperbeeLimiter(config)
    elif limiter_type == "van_leer":
        return VanLeerLimiter(config)
    elif limiter_type == "muscl":
        return MUSCLLimiter(config)
    elif limiter_type == "venkatakrishnan":
        return VenkatakrishnanLimiter(config)
    else:
        raise ValueError(f"Unknown limiter type: {limiter_type}")


def test_limiters():
    """Test limiter functionality."""
    print("Testing Slope and Flux Limiters:")
    
    # Test data
    left_grad = np.array([1.0, -0.5, 2.0, 0.1, -1.5])
    right_grad = np.array([0.8, -0.3, 1.5, 0.15, -1.2])
    
    # Test different limiters
    limiters = {
        "Minmod": MinmodLimiter(),
        "Superbee": SuperbeeLimiter(),
        "Van Leer": VanLeerLimiter(),
        "MUSCL": MUSCLLimiter(),
        "Venkatakrishnan": VenkatakrishnanLimiter()
    }
    
    print("\n  Testing 1D Slope Limiters:")
    for name, limiter in limiters.items():
        limited = limiter.limit_slope(left_grad, right_grad)
        print(f"    {name}: {limited}")
    
    # Test adaptive limiter
    print("\n  Testing Adaptive Limiter:")
    config = LimiterConfig(use_shock_detection=True, shock_threshold=0.1)
    adaptive = AdaptiveLimiter(config)
    
    # Mock data
    n_cells = 100
    conservative_vars = np.random.rand(n_cells, 5) + 0.5
    gradients = np.random.rand(n_cells, 5, 3) * 0.1
    neighbor_gradients = np.random.rand(n_cells, 6, 5, 3) * 0.1
    face_normals = np.random.rand(n_cells, 6, 3)
    
    # Normalize face normals
    for i in range(n_cells):
        for j in range(6):
            norm = np.linalg.norm(face_normals[i, j])
            if norm > 1e-12:
                face_normals[i, j] /= norm
    
    mesh_info = {'cell_volumes': np.ones(n_cells)}
    
    limited_gradients = adaptive.apply_adaptive_limiting(
        conservative_vars, gradients, neighbor_gradients, face_normals, mesh_info
    )
    
    print(f"    Applied adaptive limiting to {n_cells} cells")
    print(f"    Original gradient norm: {np.linalg.norm(gradients):.6f}")
    print(f"    Limited gradient norm: {np.linalg.norm(limited_gradients):.6f}")
    
    # Test flux limiter
    print("\n  Testing Flux Limiter:")
    flux_limiter = FluxLimiter("superbee")
    n_faces = 50
    face_fluxes = np.random.rand(n_faces, 5)
    upwind_grads = np.random.rand(n_faces, 5)
    downwind_grads = np.random.rand(n_faces, 5)
    
    limited_fluxes = flux_limiter.apply_flux_limiting(face_fluxes, upwind_grads, downwind_grads)
    print(f"    Applied flux limiting to {n_faces} faces")
    print(f"    Original flux norm: {np.linalg.norm(face_fluxes):.6f}")
    print(f"    Limited flux norm: {np.linalg.norm(limited_fluxes):.6f}")


if __name__ == "__main__":
    test_limiters()