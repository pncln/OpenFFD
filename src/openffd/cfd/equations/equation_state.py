"""
Equation of State for Compressible Flow

Provides thermodynamic relationships for various gas models
including perfect gas, real gas effects, and high-temperature effects.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class EquationOfState(ABC):
    """Abstract base class for equation of state models."""
    
    @abstractmethod
    def pressure(self, density: float, temperature: float) -> float:
        """Compute pressure from density and temperature."""
        pass
    
    @abstractmethod
    def temperature(self, density: float, pressure: float) -> float:
        """Compute temperature from density and pressure."""
        pass
    
    @abstractmethod
    def speed_of_sound(self, density: float, pressure: float) -> float:
        """Compute speed of sound."""
        pass
    
    @abstractmethod
    def specific_enthalpy(self, temperature: float) -> float:
        """Compute specific enthalpy."""
        pass
    
    @abstractmethod
    def specific_entropy(self, density: float, temperature: float) -> float:
        """Compute specific entropy."""
        pass


@dataclass
class GasProperties:
    """Gas properties for thermodynamic calculations."""
    molecular_weight: float  # kg/mol
    specific_heat_ratio: float  # gamma = cp/cv
    gas_constant: float  # J/kg/K
    
    # Temperature-dependent properties (if applicable)
    reference_temperature: float = 298.15  # K
    reference_pressure: float = 101325.0  # Pa
    
    # Sutherland's law coefficients for viscosity
    sutherland_temperature: float = 110.4  # K for air
    reference_viscosity: float = 1.716e-5  # Pa·s for air at 273K
    
    # Prandtl number
    prandtl_number: float = 0.72  # for air


class PerfectGas(EquationOfState):
    """
    Perfect gas equation of state: p = ρRT
    
    Suitable for most supersonic flow applications where
    real gas effects are negligible.
    """
    
    def __init__(self, gas_properties: GasProperties):
        """Initialize perfect gas model."""
        self.props = gas_properties
        self.gamma = gas_properties.specific_heat_ratio
        self.R = gas_properties.gas_constant
        self.gamma_minus_1 = self.gamma - 1.0
        self.gamma_over_gamma_minus_1 = self.gamma / self.gamma_minus_1
        
        # Specific heats
        self.cv = self.R / self.gamma_minus_1  # Specific heat at constant volume
        self.cp = self.gamma * self.cv  # Specific heat at constant pressure
        
        logger.info(f"Initialized perfect gas: γ={self.gamma}, R={self.R} J/kg/K")
    
    def pressure(self, density: float, temperature: float) -> float:
        """Compute pressure: p = ρRT."""
        return density * self.R * temperature
    
    def temperature(self, density: float, pressure: float) -> float:
        """Compute temperature: T = p/(ρR)."""
        return pressure / (density * self.R)
    
    def speed_of_sound(self, density: float, pressure: float) -> float:
        """Compute speed of sound: a = √(γp/ρ)."""
        return np.sqrt(self.gamma * pressure / density)
    
    def speed_of_sound_from_temperature(self, temperature: float) -> float:
        """Compute speed of sound from temperature: a = √(γRT)."""
        return np.sqrt(self.gamma * self.R * temperature)
    
    def specific_enthalpy(self, temperature: float) -> float:
        """Compute specific enthalpy: h = cpT."""
        return self.cp * temperature
    
    def specific_internal_energy(self, temperature: float) -> float:
        """Compute specific internal energy: e = cvT."""
        return self.cv * temperature
    
    def specific_entropy(self, density: float, temperature: float) -> float:
        """Compute specific entropy: s = cv*ln(T) - R*ln(ρ) + constant."""
        # Using reference state for absolute entropy
        s_ref = 0.0  # Reference entropy
        T_ref = self.props.reference_temperature
        rho_ref = self.props.reference_pressure / (self.R * T_ref)
        
        return s_ref + self.cv * np.log(temperature / T_ref) - self.R * np.log(density / rho_ref)
    
    def total_enthalpy(self, density: float, pressure: float, velocity_squared: float) -> float:
        """Compute total enthalpy: H = h + V²/2."""
        temperature = self.temperature(density, pressure)
        static_enthalpy = self.specific_enthalpy(temperature)
        return static_enthalpy + 0.5 * velocity_squared
    
    def isentropic_relations(self, 
                           mach_number: float) -> Dict[str, float]:
        """
        Compute isentropic flow relations.
        
        Returns ratios relative to stagnation conditions.
        """
        M = mach_number
        gamma = self.gamma
        gamma_m1 = self.gamma_minus_1
        
        # Temperature ratio
        temp_ratio = 1.0 / (1.0 + 0.5 * gamma_m1 * M**2)
        
        # Pressure ratio
        pressure_ratio = temp_ratio**(gamma / gamma_m1)
        
        # Density ratio
        density_ratio = temp_ratio**(1.0 / gamma_m1)
        
        # Area ratio (for nozzle flows)
        area_ratio = (1.0 / M) * ((2.0 + gamma_m1 * M**2) / (gamma + 1))**((gamma + 1) / (2 * gamma_m1))
        
        return {
            'temperature_ratio': temp_ratio,
            'pressure_ratio': pressure_ratio,
            'density_ratio': density_ratio,
            'area_ratio': area_ratio
        }
    
    def shock_relations(self, 
                       mach_upstream: float) -> Dict[str, float]:
        """
        Compute normal shock relations (Rankine-Hugoniot).
        
        Returns downstream conditions relative to upstream.
        """
        M1 = mach_upstream
        gamma = self.gamma
        gamma_m1 = self.gamma_minus_1
        gamma_p1 = gamma + 1
        
        # Downstream Mach number
        M2_squared = (M1**2 + 2/gamma_m1) / (2*gamma*M1**2/gamma_m1 - 1)
        M2 = np.sqrt(M2_squared)
        
        # Pressure ratio
        pressure_ratio = 1 + 2*gamma/gamma_p1 * (M1**2 - 1)
        
        # Density ratio
        density_ratio = gamma_p1 * M1**2 / (gamma_m1 * M1**2 + 2)
        
        # Temperature ratio
        temperature_ratio = pressure_ratio / density_ratio
        
        return {
            'mach_downstream': M2,
            'pressure_ratio': pressure_ratio,
            'density_ratio': density_ratio,
            'temperature_ratio': temperature_ratio
        }
    
    def oblique_shock_relations(self, 
                              mach_upstream: float,
                              shock_angle: float) -> Dict[str, float]:
        """
        Compute oblique shock relations.
        
        Args:
            mach_upstream: Upstream Mach number
            shock_angle: Shock angle in radians
            
        Returns:
            Dictionary with shock relations
        """
        M1 = mach_upstream
        beta = shock_angle
        gamma = self.gamma
        
        # Normal component of upstream Mach number
        M1n = M1 * np.sin(beta)
        
        # Normal shock relations for normal component
        normal_relations = self.shock_relations(M1n)
        M2n = normal_relations['mach_downstream']
        
        # Downstream Mach number
        M2 = M2n / np.sin(beta - np.arctan(2/np.tan(beta) * (M1**2 * np.sin(beta)**2 - 1) / 
                                          (M1**2 * (gamma + np.cos(2*beta)) + 2)))
        
        # Deflection angle
        delta = np.arctan(2/np.tan(beta) * (M1**2 * np.sin(beta)**2 - 1) / 
                         (M1**2 * (gamma + np.cos(2*beta)) + 2))
        
        return {
            'mach_downstream': M2,
            'deflection_angle': delta,
            'pressure_ratio': normal_relations['pressure_ratio'],
            'density_ratio': normal_relations['density_ratio'],
            'temperature_ratio': normal_relations['temperature_ratio']
        }
    
    def compute_viscosity(self, temperature: float) -> float:
        """Compute dynamic viscosity using Sutherland's law."""
        T = temperature
        T_ref = self.props.reference_temperature
        mu_ref = self.props.reference_viscosity
        S = self.props.sutherland_temperature
        
        # Sutherland's law
        mu = mu_ref * (T / T_ref)**(3/2) * (T_ref + S) / (T + S)
        
        return mu
    
    def compute_thermal_conductivity(self, temperature: float) -> float:
        """Compute thermal conductivity from viscosity and Prandtl number."""
        mu = self.compute_viscosity(temperature)
        Pr = self.props.prandtl_number
        
        # k = μ * cp / Pr
        k = mu * self.cp / Pr
        
        return k
    
    def compute_stagnation_conditions(self, 
                                    density: float,
                                    pressure: float,
                                    velocity: np.ndarray) -> Dict[str, float]:
        """
        Compute stagnation (total) conditions.
        
        Args:
            density: Static density
            pressure: Static pressure  
            velocity: Velocity vector
            
        Returns:
            Dictionary with stagnation conditions
        """
        # Static temperature
        T_static = self.temperature(density, pressure)
        
        # Velocity magnitude
        V_mag = np.linalg.norm(velocity)
        
        # Mach number
        a_static = self.speed_of_sound_from_temperature(T_static)
        M = V_mag / a_static
        
        # Stagnation temperature
        T_total = T_static * (1 + 0.5 * self.gamma_minus_1 * M**2)
        
        # Stagnation pressure
        p_total = pressure * (T_total / T_static)**(self.gamma / self.gamma_minus_1)
        
        # Stagnation density
        rho_total = density * (T_total / T_static)**(1 / self.gamma_minus_1)
        
        return {
            'total_temperature': T_total,
            'total_pressure': p_total,
            'total_density': rho_total,
            'mach_number': M
        }


class RealGas(EquationOfState):
    """
    Real gas equation of state for high-temperature applications.
    
    Uses virial equation or other real gas models for applications
    where perfect gas assumptions break down.
    """
    
    def __init__(self, gas_properties: GasProperties):
        """Initialize real gas model."""
        self.props = gas_properties
        self.perfect_gas = PerfectGas(gas_properties)
        
        # Virial coefficients (temperature-dependent)
        self.virial_coefficients = {
            'B': -1.5e-4,  # Second virial coefficient [m³/mol]
            'C': 1.0e-8    # Third virial coefficient [m⁶/mol²]
        }
        
        logger.info("Initialized real gas equation of state")
    
    def pressure(self, density: float, temperature: float) -> float:
        """Compute pressure using virial equation."""
        # Convert to molar density
        rho_molar = density / self.props.molecular_weight  # mol/m³
        
        # Virial equation: p = ρRT(1 + Bρ + Cρ²)
        B = self.virial_coefficients['B']
        C = self.virial_coefficients['C']
        
        Z = 1 + B * rho_molar + C * rho_molar**2  # Compressibility factor
        
        return density * self.props.gas_constant * temperature * Z
    
    def temperature(self, density: float, pressure: float) -> float:
        """Compute temperature (requires iteration for real gas)."""
        # Initial guess using perfect gas
        T_guess = self.perfect_gas.temperature(density, pressure)
        
        # Newton-Raphson iteration
        for _ in range(10):
            p_computed = self.pressure(density, T_guess)
            dp_dT = self._pressure_derivative_temperature(density, T_guess)
            
            if abs(dp_dT) < 1e-12:
                break
                
            T_new = T_guess - (p_computed - pressure) / dp_dT
            
            if abs(T_new - T_guess) < 1e-6:
                break
                
            T_guess = T_new
        
        return T_guess
    
    def _pressure_derivative_temperature(self, density: float, temperature: float) -> float:
        """Compute dp/dT for Newton-Raphson iteration."""
        rho_molar = density / self.props.molecular_weight
        B = self.virial_coefficients['B']
        C = self.virial_coefficients['C']
        
        Z = 1 + B * rho_molar + C * rho_molar**2
        
        return density * self.props.gas_constant * Z
    
    def speed_of_sound(self, density: float, pressure: float) -> float:
        """Compute speed of sound for real gas."""
        # Simplified approach - use compressibility factor correction
        temperature = self.temperature(density, pressure)
        a_perfect = self.perfect_gas.speed_of_sound_from_temperature(temperature)
        
        # Real gas correction (simplified)
        rho_molar = density / self.props.molecular_weight
        B = self.virial_coefficients['B']
        
        correction = 1 + 0.5 * B * rho_molar
        
        return a_perfect * np.sqrt(correction)
    
    def specific_enthalpy(self, temperature: float) -> float:
        """Compute specific enthalpy for real gas."""
        # Include departure from ideal gas
        h_ideal = self.perfect_gas.specific_enthalpy(temperature)
        
        # Real gas correction (simplified)
        h_departure = 0.0  # Would include detailed thermodynamic calculations
        
        return h_ideal + h_departure
    
    def specific_entropy(self, density: float, temperature: float) -> float:
        """Compute specific entropy for real gas."""
        # Include compressibility factor effects
        s_ideal = self.perfect_gas.specific_entropy(density, temperature)
        
        # Real gas correction
        rho_molar = density / self.props.molecular_weight
        B = self.virial_coefficients['B']
        
        s_departure = -self.props.gas_constant * B * rho_molar
        
        return s_ideal + s_departure


# Predefined gas properties
AIR_PROPERTIES = GasProperties(
    molecular_weight=28.97e-3,  # kg/mol
    specific_heat_ratio=1.4,
    gas_constant=287.0,  # J/kg/K
    sutherland_temperature=110.4,  # K
    reference_viscosity=1.716e-5,  # Pa·s
    prandtl_number=0.72
)

NITROGEN_PROPERTIES = GasProperties(
    molecular_weight=28.014e-3,  # kg/mol
    specific_heat_ratio=1.4,
    gas_constant=296.8,  # J/kg/K
    sutherland_temperature=111.0,  # K
    reference_viscosity=1.663e-5,  # Pa·s
    prandtl_number=0.71
)

ARGON_PROPERTIES = GasProperties(
    molecular_weight=39.948e-3,  # kg/mol
    specific_heat_ratio=1.67,
    gas_constant=208.1,  # J/kg/K
    sutherland_temperature=144.0,  # K
    reference_viscosity=2.125e-5,  # Pa·s
    prandtl_number=0.67
)