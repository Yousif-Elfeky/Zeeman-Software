# Zeeman Physics Module

This document provides a detailed explanation of the Zeeman Physics module implemented in the Zeeman Effect Analysis Software. The module is responsible for performing the physics calculations related to the Zeeman effect, including wavelength shifts, energy shifts, and Bohr magneton calculations.

## Overview

The Zeeman Physics module provides the theoretical foundation for analyzing the Zeeman effect in spectral lines. It implements the mathematical models and physical constants needed to convert measured spectral line positions into meaningful physical quantities such as wavelength shifts, energy shifts, and the Bohr magneton.

## Class Structure

The module is built around the `ZeemanMeasurement` data class and several supporting functions:

1. **ZeemanMeasurement Class**: Stores all relevant data for a single Zeeman effect measurement
2. **Angle Calculation Functions**: Calculate incident and refracted angles
3. **Wavelength and Energy Shift Functions**: Calculate shifts based on measured angles
4. **Measurement Processing Function**: Processes raw measurements into physical quantities
5. **Bohr Magneton Calculation Function**: Calculates the Bohr magneton from a series of measurements

## Physical Constants

The module defines several physical constants used in the calculations:

```python
# Constants
PLANCK = 6.62607015e-34  # Planck's constant in J·s
LIGHT_SPEED = 2.99792458e8  # Speed of light in m/s
FOCAL_LENGTH = 0.150  # Focal length in meters
SILICA_INDEX = 1.46  # Refractive index of silica
EV_TO_JOULE = 1.602176634e-19  # Conversion factor from eV to Joule
```

These constants are used throughout the calculations to ensure accurate results.

## Implementation Details

### 1. ZeemanMeasurement Data Class

The `ZeemanMeasurement` data class stores all relevant data for a single Zeeman effect measurement:

```python
@dataclass
class ZeemanMeasurement:
    B_field: float  # Magnetic field strength in Tesla
    wavelength: float  # Central wavelength in meters
    R_center: Optional[float] = None  # Radius of central line in mm
    R_inner: Optional[float] = None   # Radius of inner line in mm
    R_outer: Optional[float] = None   # Radius of outer line in mm
    
    alpha_c: Optional[float] = None  # Incident angle for central line
    alpha_i: Optional[float] = None  # Incident angle for inner line
    alpha_o: Optional[float] = None  # Incident angle for outer line
    
    beta_c: Optional[float] = None   # Refracted angle for central line
    beta_i: Optional[float] = None   # Refracted angle for inner line
    beta_o: Optional[float] = None   # Refracted angle for outer line
    
    delta_lambda_i: Optional[float] = None  # Wavelength shift for inner line
    delta_lambda_o: Optional[float] = None  # Wavelength shift for outer line
    
    delta_E_i: Optional[float] = None  # Energy shift for inner line
    delta_E_o: Optional[float] = None  # Energy shift for outer line
    delta_E_avg: Optional[float] = None  # Average energy shift
```

This class provides a structured way to store and access all the data related to a Zeeman effect measurement, from the raw measurements (radii) to the calculated physical quantities (wavelength shifts, energy shifts).

### 2. Angle Calculation Functions

The module includes functions to calculate the incident and refracted angles based on the measured radii:

```python
def calculate_incident_angle(radius_mm: float) -> float:
    """
    Calculate the incident angle based on the measured radius.
    
    Args:
        radius_mm: Radius in millimeters
        
    Returns:
        Incident angle in radians
    """
    return np.arctan(radius_mm / 1000 / FOCAL_LENGTH)

def calculate_refracted_angle(alpha: float) -> float:
    """
    Calculate the refracted angle based on the incident angle.
    
    Args:
        alpha: Incident angle in radians
        
    Returns:
        Refracted angle in radians
    """
    return np.arcsin(np.sin(alpha) / SILICA_INDEX)
```

These functions implement the optical principles that relate the measured radii to the angles of light propagation through the optical system.

### 3. Wavelength and Energy Shift Functions

The module includes functions to calculate the wavelength and energy shifts based on the measured angles:

```python
def calculate_wavelength_shift(beta: float, beta_c: float, wavelength: float) -> float:
    """
    Calculate the wavelength shift based on the refracted angles.
    
    Args:
        beta: Refracted angle for the shifted line in radians
        beta_c: Refracted angle for the central line in radians
        wavelength: Central wavelength in meters
        
    Returns:
        Wavelength shift in meters
    """
    return wavelength * (np.cos(beta_c) / np.cos(beta) - 1)

def calculate_energy_shift(delta_lambda: float, wavelength: float) -> float:
    """
    Calculate the energy shift based on the wavelength shift.
    
    Args:
        delta_lambda: Wavelength shift in meters
        wavelength: Central wavelength in meters
        
    Returns:
        Energy shift in Joules
    """
    return PLANCK * LIGHT_SPEED * delta_lambda / (wavelength ** 2)
```

These functions implement the physical principles that relate the measured angles to the wavelength and energy shifts caused by the Zeeman effect.

### 4. Measurement Processing Function

The module includes a function to process a raw measurement into physical quantities:

```python
def process_measurement(measurement: ZeemanMeasurement) -> ZeemanMeasurement:
    """
    Process a raw measurement into physical quantities.
    
    Args:
        measurement: A ZeemanMeasurement object with raw measurements
        
    Returns:
        The same ZeemanMeasurement object with calculated physical quantities
    """
    if any(v is None for v in [measurement.R_center, measurement.R_inner, measurement.R_outer]):
        return measurement
    
    # Calculate incident angles
    measurement.alpha_c = calculate_incident_angle(measurement.R_center)
    measurement.alpha_i = calculate_incident_angle(measurement.R_inner)
    measurement.alpha_o = calculate_incident_angle(measurement.R_outer)
    
    # Calculate refracted angles
    measurement.beta_c = calculate_refracted_angle(measurement.alpha_c)
    measurement.beta_i = calculate_refracted_angle(measurement.alpha_i)
    measurement.beta_o = calculate_refracted_angle(measurement.alpha_o)
    
    # Calculate wavelength shifts
    measurement.delta_lambda_i = calculate_wavelength_shift(measurement.beta_i, measurement.beta_c, measurement.wavelength)
    measurement.delta_lambda_o = calculate_wavelength_shift(measurement.beta_o, measurement.beta_c, measurement.wavelength)
    
    # Calculate energy shifts
    measurement.delta_E_i = calculate_energy_shift(measurement.delta_lambda_i, measurement.wavelength)
    measurement.delta_E_o = calculate_energy_shift(measurement.delta_lambda_o, measurement.wavelength)
    
    # Calculate average energy shift
    measurement.delta_E_avg = (abs(measurement.delta_E_i) + abs(measurement.delta_E_o)) / 2
    
    return measurement
```

This function takes a `ZeemanMeasurement` object with raw measurements (radii) and calculates all the derived physical quantities, including incident angles, refracted angles, wavelength shifts, and energy shifts.

### 5. Bohr Magneton Calculation Function

The module includes a function to calculate the Bohr magneton from a series of measurements:

```python
def calculate_bohr_magneton(measurements: List[ZeemanMeasurement]) -> tuple[float, float, float, float, float, float]:
    """
    Calculate the Bohr magneton from a series of measurements.
    
    Args:
        measurements: A list of ZeemanMeasurement objects
        
    Returns:
        A tuple containing (bohr_magneton_inner, bohr_magneton_outer, bohr_magneton_avg,
                          specific_charge_inner, specific_charge_outer, specific_charge_avg)
    """
    valid_measurements = [m for m in measurements if m.delta_E_i is not None and m.delta_E_o is not None]
    if not valid_measurements:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    # Extract magnetic field and energy shift values
    B_values = np.array([m.B_field for m in valid_measurements])
    E_i_values = np.array([abs(m.delta_E_i) for m in valid_measurements])
    E_o_values = np.array([abs(m.delta_E_o) for m in valid_measurements])
    
    # Normalize the data for better numerical stability
    B_mean = np.mean(B_values)
    B_std = np.std(B_values) if len(B_values) > 1 else 1.0
    B_norm = (B_values - B_mean) / B_std if B_std != 0 else B_values
    
    E_i_mean = np.mean(E_i_values)
    E_i_std = np.std(E_i_values) if len(E_i_values) > 1 else 1.0
    E_i_norm = (E_i_values - E_i_mean) / E_i_std if E_i_std != 0 else E_i_values
    
    E_o_mean = np.mean(E_o_values)
    E_o_std = np.std(E_o_values) if len(E_o_values) > 1 else 1.0
    E_o_norm = (E_o_values - E_o_mean) / E_o_std if E_o_std != 0 else E_o_values
    
    # Calculate Bohr magneton using linear regression
    bohr_magneton_inner = np.polyfit(B_norm, E_i_norm, 1)[0] * (E_i_std / B_std) if B_std != 0 and E_i_std != 0 else 0.0
    bohr_magneton_outer = np.polyfit(B_norm, E_o_norm, 1)[0] * (E_o_std / B_std) if B_std != 0 and E_o_std != 0 else 0.0
    
    bohr_magneton_avg = (abs(bohr_magneton_inner) + abs(bohr_magneton_outer)) / 2
    
    # Calculate specific charge (e/m)
    h_bar = PLANCK / (2 * np.pi)
    specific_charge_inner = 2 * abs(bohr_magneton_inner) / h_bar
    specific_charge_outer = 2 * abs(bohr_magneton_outer) / h_bar
    specific_charge_avg = 2 * bohr_magneton_avg / h_bar
    
    return (bohr_magneton_inner, bohr_magneton_outer, bohr_magneton_avg,
            specific_charge_inner, specific_charge_outer, specific_charge_avg)
```

This function takes a list of `ZeemanMeasurement` objects and calculates the Bohr magneton and specific charge (e/m) using linear regression. The function:

1. Filters out invalid measurements
2. Extracts magnetic field and energy shift values
3. Normalizes the data for better numerical stability
4. Calculates the Bohr magneton using linear regression
5. Calculates the specific charge (e/m)
6. Returns the results for both inner and outer lines, as well as the average

## Physical Principles

The Zeeman Physics module is based on several key physical principles:

1. **Zeeman Effect**: The splitting of spectral lines in the presence of a magnetic field
2. **Optical Principles**: The relationship between measured radii and angles of light propagation
3. **Wavelength-Energy Relationship**: The relationship between wavelength shifts and energy shifts
4. **Bohr Magneton**: The fundamental unit of magnetic moment for an electron

These principles are implemented in the module's functions to provide a comprehensive analysis of the Zeeman effect.

## Integration with the Measurement Workflow

The Zeeman Physics module is integrated into the measurement workflow through the Main Window and Results Window, which use the module to process measurements and calculate results:

```python
# Example integration in the Results Window
measurements = []
for img_data in self.mw.images:
    if 'measurement' in img_data and img_data['measurement'] is not None:
        # Create and process a ZeemanMeasurement object
        measurement = zeeman.ZeemanMeasurement(
            B_field=img_data['B_field'],
            wavelength=img_data['wavelength'],
            R_center=img_data['measurement']['radii']['middle'],
            R_inner=img_data['measurement']['radii']['inner'],
            R_outer=img_data['measurement']['radii']['outer']
        )
        measurement = zeeman.process_measurement(measurement)
        measurements.append(measurement)

# Calculate Bohr magneton
results = zeeman.calculate_bohr_magneton(measurements)
```

## Benefits

The Zeeman Physics module offers several advantages:

1. **Accurate Calculations**: Implements precise physical models and constants for accurate results
2. **Comprehensive Analysis**: Provides a complete analysis of the Zeeman effect, from raw measurements to physical quantities
3. **Structured Data**: Uses a well-defined data class to store and access measurement data
4. **Flexible Integration**: Can be easily integrated into different parts of the software
5. **Educational Value**: Implements the physics of the Zeeman effect in a clear and understandable way
