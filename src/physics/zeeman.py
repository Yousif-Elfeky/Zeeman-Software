"""
Physics calculations for Zeeman effect analysis.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# Constants
PLANCK = 6.62607015e-34  # Planck constant in J⋅s
LIGHT_SPEED = 2.99792458e8  # Speed of light in m/s
FOCAL_LENGTH = 0.150  # Focal length in meters
SILICA_INDEX = 1.46  # Refractive index of fused silica
EV_TO_JOULE = 1.602176634e-19  # Conversion factor from eV to Joules

@dataclass
class ZeemanMeasurement:
    """Class to hold measurement data for one image."""
    B_field: float  # Magnetic field strength in Tesla
    wavelength: float  # Central wavelength in meters
    R_center: Optional[float] = None  # Center radius in mm
    R_inner: Optional[float] = None   # Inner radius in mm
    R_outer: Optional[float] = None   # Outer radius in mm
    
    # Calculated values
    alpha_c: Optional[float] = None  # Center incident angle
    alpha_i: Optional[float] = None  # Inner incident angle
    alpha_o: Optional[float] = None  # Outer incident angle
    
    beta_c: Optional[float] = None   # Center refracted angle
    beta_i: Optional[float] = None   # Inner refracted angle
    beta_o: Optional[float] = None   # Outer refracted angle
    
    delta_lambda_i: Optional[float] = None  # Inner wavelength shift
    delta_lambda_o: Optional[float] = None  # Outer wavelength shift
    
    delta_E_i: Optional[float] = None  # Inner energy shift in Joules
    delta_E_o: Optional[float] = None  # Outer energy shift in Joules
    delta_E_avg: Optional[float] = None  # Average energy shift magnitude

def calculate_incident_angle(radius_mm: float) -> float:
    """Calculate incident angle α using focal length."""
    return np.arctan(radius_mm / 1000 / FOCAL_LENGTH)

def calculate_refracted_angle(alpha: float) -> float:
    """Calculate refracted angle β using Snell's law."""
    return np.arcsin(np.sin(alpha) / SILICA_INDEX)

def calculate_wavelength_shift(beta: float, beta_c: float, wavelength: float) -> float:
    """Calculate wavelength shift Δλ using Fabry-Perot relation."""
    return wavelength * (np.cos(beta_c) / np.cos(beta) - 1)

def calculate_energy_shift(delta_lambda: float, wavelength: float) -> float:
    """Calculate energy shift ΔE in Joules."""
    return PLANCK * LIGHT_SPEED * delta_lambda / (wavelength ** 2)

def process_measurement(measurement: ZeemanMeasurement) -> ZeemanMeasurement:
    """Process a single measurement to calculate all derived values."""
    # Skip if missing any radius measurement
    if any(v is None for v in [measurement.R_center, measurement.R_inner, measurement.R_outer]):
        return measurement
    
    # Calculate incident angles (α)
    measurement.alpha_c = calculate_incident_angle(measurement.R_center)
    measurement.alpha_i = calculate_incident_angle(measurement.R_inner)
    measurement.alpha_o = calculate_incident_angle(measurement.R_outer)
    
    # Calculate refracted angles (β)
    measurement.beta_c = calculate_refracted_angle(measurement.alpha_c)
    measurement.beta_i = calculate_refracted_angle(measurement.alpha_i)
    measurement.beta_o = calculate_refracted_angle(measurement.alpha_o)
    
    # Calculate wavelength shifts (Δλ)
    measurement.delta_lambda_i = calculate_wavelength_shift(measurement.beta_i, measurement.beta_c, measurement.wavelength)
    measurement.delta_lambda_o = calculate_wavelength_shift(measurement.beta_o, measurement.beta_c, measurement.wavelength)
    
    # Calculate energy shifts (ΔE)
    measurement.delta_E_i = calculate_energy_shift(measurement.delta_lambda_i, measurement.wavelength)
    measurement.delta_E_o = calculate_energy_shift(measurement.delta_lambda_o, measurement.wavelength)
    
    # Calculate average energy shift magnitude
    measurement.delta_E_avg = (abs(measurement.delta_E_i) + abs(measurement.delta_E_o)) / 2
    
    return measurement

def calculate_bohr_magneton(measurements: List[ZeemanMeasurement]) -> tuple[float, float, float, float, float, float]:
    """
    Calculate Bohr magneton from measurements using both inner and outer shifts.
    Returns (bohr_magneton_inner, bohr_magneton_outer, bohr_magneton_avg, 
            specific_charge_inner, specific_charge_outer, specific_charge_avg)
    bohr_magneton in J/T
    specific_charge in C/kg
    """
    # Extract B and ΔE values for valid measurements
    valid_measurements = [m for m in measurements if m.delta_E_i is not None and m.delta_E_o is not None]
    if not valid_measurements:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    B_values = np.array([m.B_field for m in valid_measurements])
    E_i_values = np.array([abs(m.delta_E_i) for m in valid_measurements])
    E_o_values = np.array([abs(m.delta_E_o) for m in valid_measurements])
    
    # Normalize values to avoid poor conditioning
    B_mean = np.mean(B_values)
    B_std = np.std(B_values) if len(B_values) > 1 else 1.0
    B_norm = (B_values - B_mean) / B_std if B_std != 0 else B_values
    
    E_i_mean = np.mean(E_i_values)
    E_i_std = np.std(E_i_values) if len(E_i_values) > 1 else 1.0
    E_i_norm = (E_i_values - E_i_mean) / E_i_std if E_i_std != 0 else E_i_values
    
    E_o_mean = np.mean(E_o_values)
    E_o_std = np.std(E_o_values) if len(E_o_values) > 1 else 1.0
    E_o_norm = (E_o_values - E_o_mean) / E_o_std if E_o_std != 0 else E_o_values
    
    # Fit |ΔE| vs B to get Bohr magneton (slope) for inner and outer shifts
    # Convert back to original scale after fit
    bohr_magneton_inner = np.polyfit(B_norm, E_i_norm, 1)[0] * (E_i_std / B_std) if B_std != 0 and E_i_std != 0 else 0.0
    bohr_magneton_outer = np.polyfit(B_norm, E_o_norm, 1)[0] * (E_o_std / B_std) if B_std != 0 and E_o_std != 0 else 0.0
    
    # Average the magnitudes
    bohr_magneton_avg = (abs(bohr_magneton_inner) + abs(bohr_magneton_outer)) / 2
    
    # Calculate specific charge (e/m) using μB = eℏ/2m
    h_bar = PLANCK / (2 * np.pi)
    specific_charge_inner = 2 * abs(bohr_magneton_inner) / h_bar
    specific_charge_outer = 2 * abs(bohr_magneton_outer) / h_bar
    specific_charge_avg = 2 * bohr_magneton_avg / h_bar
    
    return (bohr_magneton_inner, bohr_magneton_outer, bohr_magneton_avg,
            specific_charge_inner, specific_charge_outer, specific_charge_avg)
