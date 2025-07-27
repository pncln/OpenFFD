"""
Conservative and Primitive Variable Conversion for Compressible Flow

Provides robust conversion between conservative and primitive variables
with special handling for supersonic flows and extreme conditions.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConservativeVariables:
    """Conservative variables for compressible flow [rho, rho*u, rho*v, rho*w, rho*E]."""
    density: float
    momentum_x: float
    momentum_y: float
    momentum_z: float
    total_energy: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.density, self.momentum_x, self.momentum_y, 
                        self.momentum_z, self.total_energy])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'ConservativeVariables':
        """Create from numpy array."""
        return cls(array[0], array[1], array[2], array[3], array[4])


@dataclass
class PrimitiveVariables:
    """Primitive variables for compressible flow [rho, u, v, w, p, T]."""
    density: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    pressure: float
    temperature: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.density, self.velocity_x, self.velocity_y,
                        self.velocity_z, self.pressure, self.temperature])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'PrimitiveVariables':
        """Create from numpy array."""
        return cls(array[0], array[1], array[2], array[3], array[4], array[5])
    
    def velocity_magnitude(self) -> float:
        """Compute velocity magnitude."""
        return np.sqrt(self.velocity_x**2 + self.velocity_y**2 + self.velocity_z**2)
    
    def velocity_vector(self) -> np.ndarray:
        """Get velocity as vector."""
        return np.array([self.velocity_x, self.velocity_y, self.velocity_z])
    
    def mach_number(self, gamma: float, R: float) -> float:
        """Compute Mach number."""
        speed_of_sound = np.sqrt(gamma * R * self.temperature)
        return self.velocity_magnitude() / speed_of_sound


class VariableConverter:
    """
    Robust converter between conservative and primitive variables.
    
    Features:
    - Handles extreme conditions (vacuum, high Mach numbers)
    - Entropy-consistent conversions
    - Bounded temperature and pressure
    - Special handling for supersonic flows
    """
    
    def __init__(self, 
                 gamma: float = 1.4,
                 R: float = 287.0,
                 min_pressure: float = 1e-6,
                 min_temperature: float = 1e-6,
                 min_density: float = 1e-12):
        """
        Initialize variable converter.
        
        Args:
            gamma: Specific heat ratio
            R: Gas constant [J/kg/K]
            min_pressure: Minimum allowed pressure
            min_temperature: Minimum allowed temperature
            min_density: Minimum allowed density
        """
        self.gamma = gamma
        self.R = R
        self.min_pressure = min_pressure
        self.min_temperature = min_temperature
        self.min_density = min_density
        
        # Derived constants
        self.gamma_minus_1 = gamma - 1.0
        self.gamma_over_gamma_minus_1 = gamma / self.gamma_minus_1
        
        # Bounds for physical variables
        self.max_velocity = 5000.0  # m/s
        self.max_temperature = 10000.0  # K
        self.max_pressure = 1e8  # Pa
        
    def conservatives_to_primitives(self, 
                                  conservatives: np.ndarray,
                                  check_bounds: bool = True) -> np.ndarray:
        """
        Convert conservative to primitive variables.
        
        Args:
            conservatives: [rho, rho*u, rho*v, rho*w, rho*E]
            check_bounds: Apply physical bounds checking
            
        Returns:
            primitives: [rho, u, v, w, p, T]
        """
        rho, rho_u, rho_v, rho_w, rho_E = conservatives
        
        # Density bounds
        if check_bounds:
            rho = max(rho, self.min_density)
        
        # Avoid division by zero
        rho_safe = max(rho, self.min_density)
        
        # Velocities
        u = rho_u / rho_safe
        v = rho_v / rho_safe
        w = rho_w / rho_safe
        
        # Apply velocity bounds
        if check_bounds:
            velocity_mag = np.sqrt(u**2 + v**2 + w**2)
            if velocity_mag > self.max_velocity:
                scale = self.max_velocity / velocity_mag
                u *= scale
                v *= scale
                w *= scale
        
        # Kinetic energy
        kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
        
        # Internal energy
        internal_energy = rho_E - kinetic_energy
        
        # Pressure from ideal gas law
        pressure = self.gamma_minus_1 * internal_energy
        
        # Apply pressure bounds
        if check_bounds:
            pressure = max(pressure, self.min_pressure)
            pressure = min(pressure, self.max_pressure)
        
        # Temperature from ideal gas law
        temperature = pressure / (rho_safe * self.R)
        
        # Apply temperature bounds
        if check_bounds:
            temperature = max(temperature, self.min_temperature)
            temperature = min(temperature, self.max_temperature)
        
        return np.array([rho, u, v, w, pressure, temperature])
    
    def primitives_to_conservatives(self, 
                                  primitives: np.ndarray,
                                  check_bounds: bool = True) -> np.ndarray:
        """
        Convert primitive to conservative variables.
        
        Args:
            primitives: [rho, u, v, w, p, T]
            check_bounds: Apply physical bounds checking
            
        Returns:
            conservatives: [rho, rho*u, rho*v, rho*w, rho*E]
        """
        rho, u, v, w, p, T = primitives
        
        # Apply bounds
        if check_bounds:
            rho = max(rho, self.min_density)
            p = max(p, self.min_pressure)
            p = min(p, self.max_pressure)
            T = max(T, self.min_temperature)
            T = min(T, self.max_temperature)
            
            # Velocity bounds
            velocity_mag = np.sqrt(u**2 + v**2 + w**2)
            if velocity_mag > self.max_velocity:
                scale = self.max_velocity / velocity_mag
                u *= scale
                v *= scale
                w *= scale
        
        # Conservative variables
        rho_u = rho * u
        rho_v = rho * v
        rho_w = rho * w
        
        # Total energy
        kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
        internal_energy = p / self.gamma_minus_1
        rho_E = internal_energy + kinetic_energy
        
        return np.array([rho, rho_u, rho_v, rho_w, rho_E])
    
    def compute_speed_of_sound(self, primitives: np.ndarray) -> float:
        """Compute speed of sound from primitive variables."""
        rho, u, v, w, p, T = primitives
        return np.sqrt(self.gamma * p / rho)
    
    def compute_mach_number(self, primitives: np.ndarray) -> float:
        """Compute Mach number from primitive variables."""
        rho, u, v, w, p, T = primitives
        velocity_mag = np.sqrt(u**2 + v**2 + w**2)
        speed_of_sound = self.compute_speed_of_sound(primitives)
        return velocity_mag / speed_of_sound
    
    def compute_total_enthalpy(self, primitives: np.ndarray) -> float:
        """Compute total enthalpy per unit mass."""
        rho, u, v, w, p, T = primitives
        kinetic_energy = 0.5 * (u**2 + v**2 + w**2)
        specific_enthalpy = self.gamma_over_gamma_minus_1 * p / rho
        return specific_enthalpy + kinetic_energy
    
    def compute_entropy(self, primitives: np.ndarray) -> float:
        """Compute entropy (for monitoring purposes)."""
        rho, u, v, w, p, T = primitives
        # Dimensionless entropy: s = ln(p/rho^gamma)
        entropy = np.log(p / (rho**self.gamma))
        return entropy
    
    def validate_state(self, 
                      conservatives: Optional[np.ndarray] = None,
                      primitives: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        Validate physical consistency of flow state.
        
        Returns:
            (is_valid, error_message)
        """
        if conservatives is not None:
            rho, rho_u, rho_v, rho_w, rho_E = conservatives
            
            # Check for negative density
            if rho <= 0:
                return False, f"Negative density: {rho}"
            
            # Check for negative total energy
            if rho_E <= 0:
                return False, f"Negative total energy: {rho_E}"
            
            # Convert to primitives for further checks
            try:
                prim = self.conservatives_to_primitives(conservatives, check_bounds=False)
                return self.validate_state(primitives=prim)
            except:
                return False, "Failed to convert to primitives"
        
        elif primitives is not None:
            rho, u, v, w, p, T = primitives
            
            # Check bounds
            if rho <= 0:
                return False, f"Non-positive density: {rho}"
            if p <= 0:
                return False, f"Non-positive pressure: {p}"
            if T <= 0:
                return False, f"Non-positive temperature: {T}"
            
            # Check for extremely high values
            velocity_mag = np.sqrt(u**2 + v**2 + w**2)
            if velocity_mag > self.max_velocity:
                return False, f"Velocity too high: {velocity_mag}"
            if T > self.max_temperature:
                return False, f"Temperature too high: {T}"
            if p > self.max_pressure:
                return False, f"Pressure too high: {p}"
            
            # Check thermodynamic consistency
            expected_T = p / (rho * self.R)
            if abs(T - expected_T) / T > 0.01:  # 1% tolerance
                return False, f"Thermodynamic inconsistency: T={T}, expected={expected_T}"
        
        return True, "Valid state"
    
    def fix_unphysical_state(self, 
                           conservatives: np.ndarray,
                           reference_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Attempt to fix unphysical flow states.
        
        Args:
            conservatives: Conservative variables to fix
            reference_state: Reference state for fallback
            
        Returns:
            Fixed conservative variables
        """
        rho, rho_u, rho_v, rho_w, rho_E = conservatives.copy()
        
        # Fix negative density
        if rho <= 0:
            if reference_state is not None:
                rho = reference_state[0]
            else:
                rho = self.min_density
            logger.warning(f"Fixed negative density to {rho}")
        
        # Fix negative total energy
        if rho_E <= 0:
            # Set minimum internal energy
            kinetic_energy = 0.5 * (rho_u**2 + rho_v**2 + rho_w**2) / rho
            min_internal_energy = self.min_pressure / self.gamma_minus_1
            rho_E = min_internal_energy + kinetic_energy
            logger.warning(f"Fixed negative total energy to {rho_E}")
        
        # Check if internal energy is positive
        kinetic_energy = 0.5 * (rho_u**2 + rho_v**2 + rho_w**2) / rho
        internal_energy = rho_E - kinetic_energy
        
        if internal_energy <= 0:
            # Reduce kinetic energy to maintain positive internal energy
            min_internal_energy = self.min_pressure / self.gamma_minus_1
            max_kinetic_energy = rho_E - min_internal_energy
            
            if max_kinetic_energy > 0:
                scale = np.sqrt(max_kinetic_energy / kinetic_energy)
                rho_u *= scale
                rho_v *= scale
                rho_w *= scale
            else:
                # Set very small velocities
                velocity_scale = np.sqrt(2 * min_internal_energy / (3 * rho))
                rho_u = rho * velocity_scale * 0.1
                rho_v = rho * velocity_scale * 0.1
                rho_w = rho * velocity_scale * 0.1
                rho_E = min_internal_energy + 0.5 * (rho_u**2 + rho_v**2 + rho_w**2) / rho
            
            logger.warning("Fixed negative internal energy")
        
        return np.array([rho, rho_u, rho_v, rho_w, rho_E])
    
    def compute_derived_quantities(self, primitives: np.ndarray) -> dict:
        """Compute useful derived quantities from primitive variables."""
        rho, u, v, w, p, T = primitives
        
        # Basic quantities
        velocity_mag = np.sqrt(u**2 + v**2 + w**2)
        speed_of_sound = np.sqrt(self.gamma * p / rho)
        mach_number = velocity_mag / speed_of_sound
        
        # Thermodynamic quantities
        specific_enthalpy = self.gamma_over_gamma_minus_1 * p / rho
        total_enthalpy = specific_enthalpy + 0.5 * velocity_mag**2
        
        # Stagnation conditions
        total_temperature = T * (1 + 0.5 * self.gamma_minus_1 * mach_number**2)
        total_pressure = p * (total_temperature / T)**(self.gamma / self.gamma_minus_1)
        
        # Entropy
        entropy = np.log(p / (rho**self.gamma))
        
        return {
            'velocity_magnitude': velocity_mag,
            'speed_of_sound': speed_of_sound,
            'mach_number': mach_number,
            'specific_enthalpy': specific_enthalpy,
            'total_enthalpy': total_enthalpy,
            'total_temperature': total_temperature,
            'total_pressure': total_pressure,
            'entropy': entropy
        }


# Utility functions for array operations
def convert_conservatives_to_primitives_array(conservatives_array: np.ndarray,
                                            converter: VariableConverter) -> np.ndarray:
    """Convert array of conservative variables to primitives."""
    n_cells = conservatives_array.shape[0]
    primitives_array = np.zeros((n_cells, 6))
    
    for i in range(n_cells):
        primitives_array[i] = converter.conservatives_to_primitives(conservatives_array[i])
    
    return primitives_array


def convert_primitives_to_conservatives_array(primitives_array: np.ndarray,
                                            converter: VariableConverter) -> np.ndarray:
    """Convert array of primitive variables to conservatives."""
    n_cells = primitives_array.shape[0]
    conservatives_array = np.zeros((n_cells, 5))
    
    for i in range(n_cells):
        conservatives_array[i] = converter.primitives_to_conservatives(primitives_array[i])
    
    return conservatives_array