"""
Psychrometric data module - Lookup tables for saturation properties.
Platform agnostic. Used for fast psychrometric calculations.
"""

import numpy as np
from core_calc import humedad_saturacion, entalpia_saturacion

class PsicroLUT:
    """Psychrometric lookup table for saturation properties.
    
    Pre-computes saturation humidity ratio and enthalpy over a temperature range
    for fast lookups during simulation.
    
    Attributes:
        T_min: Minimum temperature (°C)
        T_max: Maximum temperature (°C)
        step: Temperature step for grid (°C)
        P_atm: Atmospheric pressure (Pa)
        T_grid: Array of temperatures
        ws_lut: Saturation humidity ratio lookup table
        hs_lut: Saturation enthalpy lookup table
    """
    
    def __init__(self, T_min=-15.0, T_max=65.0, step=0.1, P_atm=101325.0):
        """Initialize psychrometric lookup table.
        
        Args:
            T_min: Minimum temperature (°C, default: -15)
            T_max: Maximum temperature (°C, default: 65)
            step: Temperature step for grid (°C, default: 0.1)
            P_atm: Atmospheric pressure (Pa, default: 101325)
        """
        self.T_min = T_min
        self.T_max = T_max
        self.step = step
        self.P_atm = P_atm
        
        # Create temperature grid
        self.T_grid = np.arange(T_min, T_max + step, step)
        self.num_pts = len(self.T_grid)
        
        # Initialize lookup tables
        self.ws_lut = np.zeros(self.num_pts)
        self.hs_lut = np.zeros(self.num_pts)
        
        # Populate lookup tables
        for idx, T in enumerate(self.T_grid):
            ws = humedad_saturacion(T, P_atm)
            self.ws_lut[idx] = ws
            self.hs_lut[idx] = entalpia_saturacion(T, ws, P_atm)

    def get_ws_hs(self, T):
        """Get saturation humidity ratio and enthalpy at temperature T.
        
        Performs linear interpolation between grid points for sub-grid precision.
        
        Args:
            T: Temperature in °C
            
        Returns:
            Tuple of (saturation_humidity_ratio_kg_kg, saturation_enthalpy_kJ_kg_da)
        """
        # Clamp temperature to grid range
        T_clamped = max(self.T_min, min(self.T_max, float(T)))
        
        # Calculate index (0-based)
        idx_exact = (T_clamped - self.T_min) / self.step
        idx_low = int(np.floor(idx_exact))
        idx_high = int(np.ceil(idx_exact))
        
        # Bounds checking
        if idx_low < 0:
            idx_low = 0
        if idx_high >= self.num_pts:
            idx_high = self.num_pts - 1
        if idx_low >= self.num_pts:
            idx_low = self.num_pts - 1
        
        # If exact match or at boundary, return without interpolation
        if idx_low == idx_high:
            return self.ws_lut[idx_low], self.hs_lut[idx_low]
        
        # Linear interpolation
        frac = idx_exact - idx_low
        ws = self.ws_lut[idx_low] * (1 - frac) + self.ws_lut[idx_high] * frac
        hs = self.hs_lut[idx_low] * (1 - frac) + self.hs_lut[idx_high] * frac
        
        return float(ws), float(hs)
