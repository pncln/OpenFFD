"""
Shock Detection Algorithms for High-Resolution Schemes

Provides various shock sensors and discontinuity detectors for:
- Switching between high-order and low-order schemes
- Adaptive limiting in shock regions
- Mesh adaptation indicators
- Solution quality assessment
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ShockDetector(ABC):
    """Abstract base class for shock detection algorithms."""
    
    @abstractmethod
    def detect_shocks(self, 
                     solution: np.ndarray,
                     mesh_info: Dict,
                     **kwargs) -> np.ndarray:
        """
        Detect shocks in the solution field.
        
        Args:
            solution: Solution array [n_cells, n_variables]
            mesh_info: Mesh connectivity and geometric information
            
        Returns:
            Shock indicator array [n_cells] with values in [0, 1]
        """
        pass


class DucrosShockSensor(ShockDetector):
    """
    Ducros shock sensor based on velocity divergence and vorticity.
    
    The sensor is defined as:
    χ = (∇·v)² / [(∇·v)² + (∇×v)² + ε]
    
    where χ ≈ 1 in shock regions and χ ≈ 0 in vortical regions.
    """
    
    def __init__(self, 
                 epsilon: float = 1e-12,
                 gamma: float = 1.4,
                 pressure_amplification: bool = True,
                 threshold: float = 0.65):
        """
        Initialize Ducros shock sensor.
        
        Args:
            epsilon: Small number to avoid division by zero
            gamma: Specific heat ratio
            pressure_amplification: Use pressure gradient amplification
            threshold: Threshold for shock detection
        """
        self.epsilon = epsilon
        self.gamma = gamma
        self.pressure_amplification = pressure_amplification
        self.threshold = threshold
        
    def detect_shocks(self, 
                     solution: np.ndarray,
                     mesh_info: Dict,
                     **kwargs) -> np.ndarray:
        """Detect shocks using Ducros sensor."""
        n_cells = solution.shape[0]
        shock_indicators = np.zeros(n_cells)
        
        # Extract connectivity information
        connectivity = mesh_info.get('connectivity_manager')
        if not connectivity:
            logger.warning("No connectivity manager provided for shock detection")
            return shock_indicators
        
        # Convert to primitive variables
        conservatives = solution[:, :5]  # [rho, rho*u, rho*v, rho*w, rho*E]
        primitives = self._conservatives_to_primitives(conservatives)
        
        for cell_id in range(n_cells):
            neighbors = connectivity._cell_neighbors[cell_id] if connectivity._cell_neighbors else []
            
            if len(neighbors) < 4:  # Need sufficient neighbors for gradient computation
                continue
            
            # Compute velocity divergence and curl
            div_v, curl_v_mag = self._compute_velocity_divergence_curl(
                cell_id, primitives, neighbors, mesh_info
            )
            
            # Ducros sensor
            ducros_value = div_v**2 / (div_v**2 + curl_v_mag**2 + self.epsilon)
            
            # Optional pressure gradient amplification
            if self.pressure_amplification:
                pressure_gradient = self._compute_pressure_gradient_magnitude(
                    cell_id, primitives, neighbors, mesh_info
                )
                
                # Amplify sensor in high pressure gradient regions
                pressure_threshold = 0.1 * primitives[cell_id, 4]  # 10% of local pressure
                if pressure_gradient > pressure_threshold:
                    ducros_value = min(1.0, ducros_value * 1.5)
            
            shock_indicators[cell_id] = ducros_value
        
        return shock_indicators
    
    def _conservatives_to_primitives(self, conservatives: np.ndarray) -> np.ndarray:
        """Convert conservative to primitive variables."""
        n_cells = conservatives.shape[0]
        primitives = np.zeros((n_cells, 6))  # [rho, u, v, w, p, T]
        
        for i in range(n_cells):
            rho, rho_u, rho_v, rho_w, rho_E = conservatives[i]
            
            # Avoid division by zero
            rho = max(rho, 1e-12)
            
            # Velocities
            u = rho_u / rho
            v = rho_v / rho
            w = rho_w / rho
            
            # Pressure
            kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
            p = (self.gamma - 1) * (rho_E - kinetic_energy)
            p = max(p, 1e-6)  # Ensure positive pressure
            
            # Temperature (ideal gas)
            R = 287.0  # Gas constant for air
            T = p / (rho * R)
            
            primitives[i] = [rho, u, v, w, p, T]
        
        return primitives
    
    def _compute_velocity_divergence_curl(self, 
                                        cell_id: int,
                                        primitives: np.ndarray,
                                        neighbors: List[int],
                                        mesh_info: Dict) -> Tuple[float, float]:
        """Compute velocity divergence and curl magnitude."""
        cell_center = mesh_info['centroids'][cell_id]
        cell_velocity = primitives[cell_id, 1:4]
        
        # Compute velocity gradients using least squares
        A = []
        b_u, b_v, b_w = [], [], []
        
        for neighbor_id in neighbors:
            neighbor_center = mesh_info['centroids'][neighbor_id]
            neighbor_velocity = primitives[neighbor_id, 1:4]
            
            dr = neighbor_center - cell_center
            du = neighbor_velocity - cell_velocity
            
            A.append(dr)
            b_u.append(du[0])
            b_v.append(du[1])
            b_w.append(du[2])
        
        if len(A) < 3:
            return 0.0, 0.0
        
        A = np.array(A)
        b_u = np.array(b_u)
        b_v = np.array(b_v)
        b_w = np.array(b_w)
        
        try:
            # Solve least squares for velocity gradients
            grad_u = np.linalg.lstsq(A, b_u, rcond=None)[0]
            grad_v = np.linalg.lstsq(A, b_v, rcond=None)[0]
            grad_w = np.linalg.lstsq(A, b_w, rcond=None)[0]
            
            # Velocity divergence
            div_v = grad_u[0] + grad_v[1] + grad_w[2]
            
            # Velocity curl magnitude
            curl_x = grad_w[1] - grad_v[2]
            curl_y = grad_u[2] - grad_w[0]
            curl_z = grad_v[0] - grad_u[1]
            curl_v_mag = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
            
            return abs(div_v), curl_v_mag
            
        except np.linalg.LinAlgError:
            return 0.0, 0.0
    
    def _compute_pressure_gradient_magnitude(self, 
                                           cell_id: int,
                                           primitives: np.ndarray,
                                           neighbors: List[int],
                                           mesh_info: Dict) -> float:
        """Compute pressure gradient magnitude."""
        cell_center = mesh_info['centroids'][cell_id]
        cell_pressure = primitives[cell_id, 4]
        
        A = []
        b = []
        
        for neighbor_id in neighbors:
            neighbor_center = mesh_info['centroids'][neighbor_id]
            neighbor_pressure = primitives[neighbor_id, 4]
            
            dr = neighbor_center - cell_center
            dp = neighbor_pressure - cell_pressure
            
            A.append(dr)
            b.append(dp)
        
        if len(A) < 3:
            return 0.0
        
        A = np.array(A)
        b = np.array(b)
        
        try:
            grad_p = np.linalg.lstsq(A, b, rcond=None)[0]
            return np.linalg.norm(grad_p)
        except np.linalg.LinAlgError:
            return 0.0


class JamesonShockSensor(ShockDetector):
    """
    Jameson shock sensor based on pressure undivided differences.
    
    The sensor detects shocks using normalized pressure differences:
    ν = |p_{i+1} - 2p_i + p_{i-1}| / (p_{i+1} + 2p_i + p_{i-1})
    """
    
    def __init__(self, 
                 threshold: float = 0.05,
                 gamma: float = 1.4):
        """
        Initialize Jameson shock sensor.
        
        Args:
            threshold: Threshold for shock detection
            gamma: Specific heat ratio
        """
        self.threshold = threshold
        self.gamma = gamma
    
    def detect_shocks(self, 
                     solution: np.ndarray,
                     mesh_info: Dict,
                     **kwargs) -> np.ndarray:
        """Detect shocks using Jameson sensor."""
        n_cells = solution.shape[0]
        shock_indicators = np.zeros(n_cells)
        
        # Extract connectivity information
        connectivity = mesh_info.get('connectivity_manager')
        if not connectivity:
            return shock_indicators
        
        # Convert to primitive variables
        conservatives = solution[:, :5]
        primitives = self._conservatives_to_primitives(conservatives)
        
        for cell_id in range(n_cells):
            neighbors = connectivity._cell_neighbors[cell_id] if connectivity._cell_neighbors else []
            
            if len(neighbors) < 2:
                continue
            
            # Get pressure values
            p_center = primitives[cell_id, 4]
            p_neighbors = primitives[neighbors, 4]
            
            # Compute undivided differences
            if len(p_neighbors) >= 2:
                p_max = np.max(p_neighbors)
                p_min = np.min(p_neighbors)
                
                # Second difference approximation
                undivided_diff = abs(p_max - 2*p_center + p_min)
                smoothness = abs(p_max - p_min) + 1e-12
                
                jameson_sensor = undivided_diff / smoothness
                
                # Normalize and apply threshold
                if jameson_sensor > self.threshold:
                    shock_indicators[cell_id] = min(1.0, jameson_sensor / self.threshold)
        
        return shock_indicators
    
    def _conservatives_to_primitives(self, conservatives: np.ndarray) -> np.ndarray:
        """Convert conservative to primitive variables."""
        return DucrosShockSensor._conservatives_to_primitives(self, conservatives)


class PressureJumpDetector(ShockDetector):
    """
    Pressure jump detector for shock identification.
    
    Detects shocks based on pressure ratios across cell interfaces.
    """
    
    def __init__(self, 
                 pressure_ratio_threshold: float = 1.3,
                 gamma: float = 1.4):
        """
        Initialize pressure jump detector.
        
        Args:
            pressure_ratio_threshold: Minimum pressure ratio to detect shock
            gamma: Specific heat ratio
        """
        self.pressure_ratio_threshold = pressure_ratio_threshold
        self.gamma = gamma
    
    def detect_shocks(self, 
                     solution: np.ndarray,
                     mesh_info: Dict,
                     **kwargs) -> np.ndarray:
        """Detect shocks using pressure jumps."""
        n_cells = solution.shape[0]
        shock_indicators = np.zeros(n_cells)
        
        connectivity = mesh_info.get('connectivity_manager')
        if not connectivity:
            return shock_indicators
        
        # Convert to primitive variables
        conservatives = solution[:, :5]
        primitives = self._conservatives_to_primitives(conservatives)
        
        for cell_id in range(n_cells):
            neighbors = connectivity._cell_neighbors[cell_id] if connectivity._cell_neighbors else []
            
            if not neighbors:
                continue
            
            p_center = primitives[cell_id, 4]
            max_pressure_ratio = 1.0
            
            # Check pressure ratios with all neighbors
            for neighbor_id in neighbors:
                p_neighbor = primitives[neighbor_id, 4]
                
                # Pressure ratio (larger/smaller)
                pressure_ratio = max(p_center, p_neighbor) / (min(p_center, p_neighbor) + 1e-12)
                max_pressure_ratio = max(max_pressure_ratio, pressure_ratio)
            
            # Shock indicator based on pressure ratio
            if max_pressure_ratio > self.pressure_ratio_threshold:
                shock_indicators[cell_id] = min(1.0, 
                    (max_pressure_ratio - 1.0) / (self.pressure_ratio_threshold - 1.0))
        
        return shock_indicators
    
    def _conservatives_to_primitives(self, conservatives: np.ndarray) -> np.ndarray:
        """Convert conservative to primitive variables."""
        return DucrosShockSensor._conservatives_to_primitives(self, conservatives)


class MachNumberDetector(ShockDetector):
    """
    Mach number-based shock detector.
    
    Identifies supersonic regions and potential shock locations
    based on local Mach numbers and Mach number gradients.
    """
    
    def __init__(self, 
                 mach_threshold: float = 1.1,
                 mach_gradient_threshold: float = 0.5,
                 gamma: float = 1.4,
                 R: float = 287.0):
        """
        Initialize Mach number detector.
        
        Args:
            mach_threshold: Minimum Mach number for supersonic detection
            mach_gradient_threshold: Threshold for Mach number gradients
            gamma: Specific heat ratio
            R: Gas constant
        """
        self.mach_threshold = mach_threshold
        self.mach_gradient_threshold = mach_gradient_threshold
        self.gamma = gamma
        self.R = R
    
    def detect_shocks(self, 
                     solution: np.ndarray,
                     mesh_info: Dict,
                     **kwargs) -> np.ndarray:
        """Detect shocks using Mach number analysis."""
        n_cells = solution.shape[0]
        shock_indicators = np.zeros(n_cells)
        
        connectivity = mesh_info.get('connectivity_manager')
        if not connectivity:
            return shock_indicators
        
        # Convert to primitive variables
        conservatives = solution[:, :5]
        primitives = self._conservatives_to_primitives(conservatives)
        
        # Compute Mach numbers
        mach_numbers = self._compute_mach_numbers(primitives)
        
        for cell_id in range(n_cells):
            neighbors = connectivity._cell_neighbors[cell_id] if connectivity._cell_neighbors else []
            
            if not neighbors:
                continue
            
            mach_center = mach_numbers[cell_id]
            
            # Check if in supersonic region
            supersonic_indicator = 0.0
            if mach_center > self.mach_threshold:
                supersonic_indicator = min(1.0, (mach_center - 1.0) / self.mach_threshold)
            
            # Compute Mach number gradient
            mach_gradient = self._compute_mach_gradient(
                cell_id, mach_numbers, neighbors, mesh_info
            )
            
            gradient_indicator = 0.0
            if mach_gradient > self.mach_gradient_threshold:
                gradient_indicator = min(1.0, mach_gradient / self.mach_gradient_threshold)
            
            # Combined indicator
            shock_indicators[cell_id] = max(supersonic_indicator, gradient_indicator)
        
        return shock_indicators
    
    def _conservatives_to_primitives(self, conservatives: np.ndarray) -> np.ndarray:
        """Convert conservative to primitive variables."""
        return DucrosShockSensor._conservatives_to_primitives(self, conservatives)
    
    def _compute_mach_numbers(self, primitives: np.ndarray) -> np.ndarray:
        """Compute Mach numbers from primitive variables."""
        n_cells = primitives.shape[0]
        mach_numbers = np.zeros(n_cells)
        
        for i in range(n_cells):
            rho, u, v, w, p, T = primitives[i]
            
            # Velocity magnitude
            velocity_mag = np.sqrt(u**2 + v**2 + w**2)
            
            # Speed of sound
            speed_of_sound = np.sqrt(self.gamma * p / rho)
            
            # Mach number
            mach_numbers[i] = velocity_mag / (speed_of_sound + 1e-12)
        
        return mach_numbers
    
    def _compute_mach_gradient(self, 
                             cell_id: int,
                             mach_numbers: np.ndarray,
                             neighbors: List[int],
                             mesh_info: Dict) -> float:
        """Compute Mach number gradient magnitude."""
        if len(neighbors) < 3:
            return 0.0
        
        cell_center = mesh_info['centroids'][cell_id]
        mach_center = mach_numbers[cell_id]
        
        A = []
        b = []
        
        for neighbor_id in neighbors:
            neighbor_center = mesh_info['centroids'][neighbor_id]
            mach_neighbor = mach_numbers[neighbor_id]
            
            dr = neighbor_center - cell_center
            dm = mach_neighbor - mach_center
            
            A.append(dr)
            b.append(dm)
        
        A = np.array(A)
        b = np.array(b)
        
        try:
            grad_mach = np.linalg.lstsq(A, b, rcond=None)[0]
            return np.linalg.norm(grad_mach)
        except np.linalg.LinAlgError:
            return 0.0


class CompositeShockDetector(ShockDetector):
    """
    Composite shock detector combining multiple detection methods.
    
    Provides robust shock detection by combining different sensors
    with weighted contributions.
    """
    
    def __init__(self, 
                 detectors: List[Tuple[ShockDetector, float]],
                 combination_method: str = "max"):
        """
        Initialize composite detector.
        
        Args:
            detectors: List of (detector, weight) tuples
            combination_method: "max", "weighted_average", or "product"
        """
        self.detectors = detectors
        self.combination_method = combination_method
        
        # Normalize weights
        if combination_method == "weighted_average":
            total_weight = sum(weight for _, weight in detectors)
            self.detectors = [(detector, weight/total_weight) 
                            for detector, weight in detectors]
    
    def detect_shocks(self, 
                     solution: np.ndarray,
                     mesh_info: Dict,
                     **kwargs) -> np.ndarray:
        """Detect shocks using composite method."""
        if not self.detectors:
            return np.zeros(solution.shape[0])
        
        # Compute indicators from all detectors
        all_indicators = []
        weights = []
        
        for detector, weight in self.detectors:
            indicators = detector.detect_shocks(solution, mesh_info, **kwargs)
            all_indicators.append(indicators)
            weights.append(weight)
        
        all_indicators = np.array(all_indicators)
        weights = np.array(weights)
        
        # Combine indicators
        if self.combination_method == "max":
            return np.max(all_indicators, axis=0)
        elif self.combination_method == "weighted_average":
            return np.average(all_indicators, axis=0, weights=weights)
        elif self.combination_method == "product":
            result = np.ones(solution.shape[0])
            for indicators in all_indicators:
                result *= indicators
            return result
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")


def create_default_shock_detector(shock_type: str = "ducros") -> ShockDetector:
    """
    Create default shock detector configuration.
    
    Args:
        shock_type: Type of detector ("ducros", "jameson", "pressure_jump", "mach", "composite")
        
    Returns:
        Configured shock detector
    """
    if shock_type == "ducros":
        return DucrosShockSensor(threshold=0.65, pressure_amplification=True)
    
    elif shock_type == "jameson":
        return JamesonShockSensor(threshold=0.05)
    
    elif shock_type == "pressure_jump":
        return PressureJumpDetector(pressure_ratio_threshold=1.3)
    
    elif shock_type == "mach":
        return MachNumberDetector(mach_threshold=1.1, mach_gradient_threshold=0.5)
    
    elif shock_type == "composite":
        # Create composite detector with multiple methods
        ducros = DucrosShockSensor(threshold=0.65)
        jameson = JamesonShockSensor(threshold=0.05)
        pressure_jump = PressureJumpDetector(pressure_ratio_threshold=1.3)
        
        return CompositeShockDetector([
            (ducros, 0.5),
            (jameson, 0.3),
            (pressure_jump, 0.2)
        ], combination_method="max")
    
    else:
        raise ValueError(f"Unknown shock detector type: {shock_type}")


def smooth_shock_indicator(indicators: np.ndarray,
                          mesh_info: Dict,
                          smoothing_passes: int = 2) -> np.ndarray:
    """
    Smooth shock indicators to avoid sharp transitions.
    
    Args:
        indicators: Raw shock indicators
        mesh_info: Mesh connectivity information
        smoothing_passes: Number of smoothing iterations
        
    Returns:
        Smoothed shock indicators
    """
    connectivity = mesh_info.get('connectivity_manager')
    if not connectivity:
        return indicators
    
    smoothed = indicators.copy()
    
    for _ in range(smoothing_passes):
        new_smoothed = smoothed.copy()
        
        for cell_id in range(len(indicators)):
            neighbors = connectivity._cell_neighbors[cell_id] if connectivity._cell_neighbors else []
            
            if neighbors:
                # Average with neighbors
                neighbor_values = smoothed[neighbors]
                avg_value = np.mean(neighbor_values)
                new_smoothed[cell_id] = 0.7 * smoothed[cell_id] + 0.3 * avg_value
        
        smoothed = new_smoothed
    
    return smoothed