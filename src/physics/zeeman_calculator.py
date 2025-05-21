import numpy as np

class ZeemanCalculator:
    def __init__(self):
        # Physical constants
        self.PLANCKS_CONSTANT = 6.62607015e-34  # in J⋅s
        self.ELECTRON_CHARGE = 1.602176634e-19   # in C
        self.ELECTRON_MASS = 9.1093837015e-31    # in kg
    
    def calculate_bohr_magneton(self, delta_lambda, wavelength, magnetic_field):
        c = 2.99792458e8  # speed of light in m/s
        bohr_magneton = (self.PLANCKS_CONSTANT * c * delta_lambda) / (2 * wavelength**2 * magnetic_field)
        return bohr_magneton
    
    def calculate_specific_charge(self, bohr_magneton):
        h_bar = self.PLANCKS_CONSTANT / (2 * np.pi)
        specific_charge = 2 * bohr_magneton / h_bar
        return specific_charge
    
    def calculate_uncertainties(self, measurements, magnetic_field_uncertainty):
        pass
