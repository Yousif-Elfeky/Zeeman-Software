"""
Physics calculations for Zeeman effect analysis.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# Constants
PLANCK = 6.62607015e-34  
LIGHT_SPEED = 2.99792458e8  
FOCAL_LENGTH = 0.150  
SILICA_INDEX = 1.46  
EV_TO_JOULE = 1.602176634e-19  

@dataclass
class ZeemanMeasurement:
    B_field: float  
    wavelength: float  
    R_center: Optional[float] = None  
    R_inner: Optional[float] = None   
    R_outer: Optional[float] = None   
    
    alpha_c: Optional[float] = None  
    alpha_i: Optional[float] = None  
    alpha_o: Optional[float] = None  
    
    beta_c: Optional[float] = None   
    beta_i: Optional[float] = None   
    beta_o: Optional[float] = None   
    
    delta_lambda_i: Optional[float] = None  
    delta_lambda_o: Optional[float] = None  
    
    delta_E_i: Optional[float] = None  
    delta_E_o: Optional[float] = None  
    delta_E_avg: Optional[float] = None  

def calculate_incident_angle(radius_mm: float) -> float:
    return np.arctan(radius_mm / 1000 / FOCAL_LENGTH)

def calculate_refracted_angle(alpha: float) -> float:
    return np.arcsin(np.sin(alpha) / SILICA_INDEX)

def calculate_wavelength_shift(beta: float, beta_c: float, wavelength: float) -> float:
    return wavelength * (np.cos(beta_c) / np.cos(beta) - 1)

def calculate_energy_shift(delta_lambda: float, wavelength: float) -> float:
    return PLANCK * LIGHT_SPEED * delta_lambda / (wavelength ** 2)

def process_measurement(measurement: ZeemanMeasurement) -> ZeemanMeasurement:
    if any(v is None for v in [measurement.R_center, measurement.R_inner, measurement.R_outer]):
        return measurement
    
    measurement.alpha_c = calculate_incident_angle(measurement.R_center)
    measurement.alpha_i = calculate_incident_angle(measurement.R_inner)
    measurement.alpha_o = calculate_incident_angle(measurement.R_outer)
    
    measurement.beta_c = calculate_refracted_angle(measurement.alpha_c)
    measurement.beta_i = calculate_refracted_angle(measurement.alpha_i)
    measurement.beta_o = calculate_refracted_angle(measurement.alpha_o)
    
    measurement.delta_lambda_i = calculate_wavelength_shift(measurement.beta_i, measurement.beta_c, measurement.wavelength)
    measurement.delta_lambda_o = calculate_wavelength_shift(measurement.beta_o, measurement.beta_c, measurement.wavelength)
    
    measurement.delta_E_i = calculate_energy_shift(measurement.delta_lambda_i, measurement.wavelength)
    measurement.delta_E_o = calculate_energy_shift(measurement.delta_lambda_o, measurement.wavelength)
    
    measurement.delta_E_avg = (abs(measurement.delta_E_i) + abs(measurement.delta_E_o)) / 2
    
    return measurement

def calculate_bohr_magneton(measurements: List[ZeemanMeasurement]) -> tuple[float, float, float, float, float, float]:
    valid_measurements = [m for m in measurements if m.delta_E_i is not None and m.delta_E_o is not None]
    if not valid_measurements:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    B_values = np.array([m.B_field for m in valid_measurements])
    E_i_values = np.array([abs(m.delta_E_i) for m in valid_measurements])
    E_o_values = np.array([abs(m.delta_E_o) for m in valid_measurements])
    
    B_mean = np.mean(B_values)
    B_std = np.std(B_values) if len(B_values) > 1 else 1.0
    B_norm = (B_values - B_mean) / B_std if B_std != 0 else B_values
    
    E_i_mean = np.mean(E_i_values)
    E_i_std = np.std(E_i_values) if len(E_i_values) > 1 else 1.0
    E_i_norm = (E_i_values - E_i_mean) / E_i_std if E_i_std != 0 else E_i_values
    
    E_o_mean = np.mean(E_o_values)
    E_o_std = np.std(E_o_values) if len(E_o_values) > 1 else 1.0
    E_o_norm = (E_o_values - E_o_mean) / E_o_std if E_o_std != 0 else E_o_values
    
    bohr_magneton_inner = np.polyfit(B_norm, E_i_norm, 1)[0] * (E_i_std / B_std) if B_std != 0 and E_i_std != 0 else 0.0
    bohr_magneton_outer = np.polyfit(B_norm, E_o_norm, 1)[0] * (E_o_std / B_std) if B_std != 0 and E_o_std != 0 else 0.0
    
    bohr_magneton_avg = (abs(bohr_magneton_inner) + abs(bohr_magneton_outer)) / 2
    
    h_bar = PLANCK / (2 * np.pi)
    specific_charge_inner = 2 * abs(bohr_magneton_inner) / h_bar
    specific_charge_outer = 2 * abs(bohr_magneton_outer) / h_bar
    specific_charge_avg = 2 * bohr_magneton_avg / h_bar
    
    return (bohr_magneton_inner, bohr_magneton_outer, bohr_magneton_avg,
            specific_charge_inner, specific_charge_outer, specific_charge_avg)
