"""
Core thermodynamic calculation module - Platform agnostic.
No PyQt5 dependencies. Can be used in web applications, CLI, or other UIs.

Exports:
- Thermodynamic property functions (pressure, humidity, enthalpy)
- Tower 2D matrix simulation engine
- Operating point resolver
"""

import numpy as np
from scipy.optimize import root_scalar
from functools import lru_cache

# ==========================================
# THERMODYNAMIC CONSTANTS
# ==========================================
CP_WATER_DEFAULT = 4.184      # kJ/kg.K
CP_AIR_DEFAULT = 1.006        # kJ/kg.K
CP_VAPOR_DEFAULT = 1.86       # kJ/kg.K
H_FG0_DEFAULT = 2501.0        # kJ/kg (latent heat of vaporization at 0°C)

# Optional CoolProp import
HAS_COOLPROP = False
try:
    import CoolProp.CoolProp as CP
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False

# ==========================================
# ATMOSPHERIC PROPERTIES
# ==========================================
def obtener_presion_barometrica(altitud_m):
    """Calculate barometric pressure at given altitude (m).
    
    Args:
        altitud_m: Altitude in meters
        
    Returns:
        Pressure in Pa
    """
    P0 = 101325.0  # Pa
    return P0 * (1.0 - 0.6875e-5 * float(altitud_m))**5.2561

# ==========================================
# WATER PROPERTIES
# ==========================================
@lru_cache(maxsize=4096)
def cp_agua_local_fast(T_round):
    """Cached specific heat of water at temperature T (fast lookup).
    
    Args:
        T_round: Temperature in °C (pre-rounded to 0.1°C)
        
    Returns:
        Specific heat in kJ/kg.K
    """
    if HAS_COOLPROP:
        try:
            return float(CP.PropsSI('C', 'T', T_round + 273.15, 'P', 101325, 'Water') / 1000.0)
        except Exception:
            pass
    return CP_WATER_DEFAULT

def cp_agua_local(T_celcius):
    """Specific heat of water at temperature T.
    
    Args:
        T_celcius: Temperature in °C
        
    Returns:
        Specific heat in kJ/kg.K
    """
    T_clamped = max(-10.0, min(95.0, float(T_celcius)))
    return cp_agua_local_fast(round(T_clamped, 1))

# ==========================================
# PSYCHROMETRIC PROPERTIES
# ==========================================
@lru_cache(maxsize=4096)
def humedad_saturacion_fast(T_round, P_atm_round):
    """Cached saturation humidity ratio at temperature T (fast lookup).
    
    Args:
        T_round: Temperature in °C (pre-rounded to 0.1°C)
        P_atm_round: Pressure in Pa (pre-rounded to 100 Pa)
        
    Returns:
        Saturation humidity ratio in kg/kg
    """
    if HAS_COOLPROP:
        try:
            val = CP.HAPropsSI('W', 'T', T_round + 273.15, 'R', 1.0, 'P', P_atm_round)
            if not np.isnan(val) and val > 0:
                return float(val)
        except Exception:
            pass
    
    # Fallback: Magnus formula for saturation vapor pressure
    P_atm_kPa = P_atm_round / 1000.0
    den_temp = T_round + 237.3
    if abs(den_temp) < 1e-4:
        den_temp = 1e-4
    P_sat = 0.61078 * np.exp(max(-50.0, min(50.0, 17.27 * T_round / den_temp)))
    den_press = P_atm_kPa - P_sat
    if den_press <= 1e-4:
        den_press = 1e-4
    return float(0.622 * P_sat / den_press)

def humedad_saturacion(T, P_atm=101325.0):
    """Saturation humidity ratio at temperature T.
    
    Args:
        T: Temperature in °C
        P_atm: Atmospheric pressure in Pa (default: 101325 Pa)
        
    Returns:
        Saturation humidity ratio in kg/kg
    """
    T_clamped = max(-20.0, min(95.0, float(T)))
    return humedad_saturacion_fast(round(T_clamped, 1), round(float(P_atm), -2))

def factor_lewis(w_sw, w):
    """Lewis factor for mass-heat transfer analogy.
    
    Typically around 0.865^(2/3) ≈ 0.926
    
    Args:
        w_sw: Saturation humidity ratio at wet-bulb temperature (kg/kg)
        w: Current humidity ratio (kg/kg)
        
    Returns:
        Lewis factor (dimensionless)
    """
    w_sw_c = max(0.0, float(w_sw))
    w_c = max(0.0, float(w))
    
    if w_c >= w_sw_c or abs(w_sw_c - w_c) < 1e-6:
        return 0.865**(2/3)
    
    arg = (w_sw_c + 0.622) / (w_c + 0.622)
    if arg <= 1.0 + 1e-7:
        return 0.865**(2/3)
    
    num = arg - 1.0
    den = np.log(arg)
    if den <= 1e-7:
        return 0.865**(2/3)
    
    return (0.865**(2/3)) * (num / den)

# ==========================================
# AIR PROPERTIES (MOIST AIR)
# ==========================================
@lru_cache(maxsize=4096)
def entalpia_saturacion_fast(T_round, w_sat_round, P_atm_round):
    """Cached saturation enthalpy at temperature T (fast lookup).
    
    Args:
        T_round: Temperature in °C (pre-rounded to 0.1°C)
        w_sat_round: Saturation humidity ratio (pre-rounded to 0.0001 kg/kg)
        P_atm_round: Pressure in Pa (pre-rounded to 100 Pa)
        
    Returns:
        Saturation enthalpy in kJ/kg_da
    """
    if HAS_COOLPROP:
        try:
            val = CP.HAPropsSI('H', 'T', T_round + 273.15, 'W', w_sat_round, 'P', P_atm_round) / 1000.0
            if not np.isnan(val):
                return float(val)
        except Exception:
            pass
    
    # Fallback: polynomial approximation
    return float(CP_AIR_DEFAULT * T_round + w_sat_round * (H_FG0_DEFAULT + CP_VAPOR_DEFAULT * T_round))

def entalpia_saturacion(T, w_sat, P_atm=101325.0):
    """Saturation enthalpy at temperature T.
    
    Args:
        T: Temperature in °C
        w_sat: Saturation humidity ratio in kg/kg
        P_atm: Atmospheric pressure in Pa (default: 101325 Pa)
        
    Returns:
        Saturation enthalpy in kJ/kg_da
    """
    T_clamped = max(-20.0, min(95.0, float(T)))
    w_clamped = max(0.0, float(w_sat))
    return entalpia_saturacion_fast(round(T_clamped, 1), round(w_clamped, 4), round(float(P_atm), -2))

def temp_aire_desde_entalpia(h_a, w_a, P_atm=101325.0):
    """Calculate dry-bulb temperature from enthalpy and humidity ratio.
    
    Args:
        h_a: Enthalpy in kJ/kg_da
        w_a: Humidity ratio in kg/kg
        P_atm: Atmospheric pressure in Pa (default: 101325 Pa)
        
    Returns:
        Dry-bulb temperature in °C
    """
    h_c = max(-50.0, min(500.0, float(h_a)))
    w_c = max(0.0, min(0.1, float(w_a)))
    
    if HAS_COOLPROP:
        try:
            T_kelvin = CP.HAPropsSI('T', 'H', h_c * 1000.0, 'W', w_c, 'P', float(P_atm))
            if not np.isnan(T_kelvin):
                return float(T_kelvin - 273.15)
        except Exception:
            pass
    
    # Fallback: polynomial approximation
    den = CP_AIR_DEFAULT + w_c * CP_VAPOR_DEFAULT
    if abs(den) < 1e-5:
        den = CP_AIR_DEFAULT
    return float((h_c - w_c * H_FG0_DEFAULT) / den)

# ==========================================
# TOWER SIMULATION ENGINE
# ==========================================
def simular_torre_2d_matriz(NTU_actual, T_w_in, m_w_total, h_a_in, w_a_in, m_a_total, 
                             P_atm=101325.0, Nx=6, Ny=6, lut=None):
    """2D matrix tower simulation using Poppe method.
    
    Simulates counter-flow cooling tower with discretized water/air cells.
    Tracks temperature, humidity, and fog formation.
    
    Args:
        NTU_actual: Number of Transfer Units (current operating point)
        T_w_in: Water inlet temperature (°C)
        m_w_total: Total water mass flow (kg/s)
        h_a_in: Air inlet enthalpy (kJ/kg_da)
        w_a_in: Air inlet humidity ratio (kg/kg)
        m_a_total: Total air mass flow (kg/s)
        P_atm: Atmospheric pressure (Pa, default: 101325)
        Nx: Number of vertical cells (default: 6)
        Ny: Number of horizontal cells (default: 6)
        lut: PsicroLUT object for fast lookups (optional)
        
    Returns:
        Tuple of:
        - T_w_out: Water outlet temperature (°C)
        - m_w_evap: Water evaporated (kg/s)
        - T_w_matrix: Water temperature matrix (Ny, Nx)
        - w_a_matrix: Air humidity ratio matrix (Ny, Nx)
        - T_a_matrix: Air temperature matrix (Ny, Nx)
        - fog_matrix: Boolean fog occurrence matrix (Ny, Nx)
    """
    m_a_total_safe = max(1e-4, float(m_a_total))
    
    dm_w = m_w_total / Nx  
    dm_a = m_a_total_safe / Ny  
    K_dA = (NTU_actual * m_w_total) / (Nx * Ny) 
    
    # Initialize matrices
    T_w = np.zeros((Ny + 1, Nx))
    m_w = np.zeros((Ny + 1, Nx))
    h_a = np.zeros((Ny, Nx + 1))
    w_a = np.zeros((Ny, Nx + 1))
    
    fog_matrix = np.zeros((Ny, Nx), dtype=bool)
    T_a_matrix = np.zeros((Ny, Nx))
    
    # Boundary conditions
    T_w[0, :] = T_w_in
    m_w[0, :] = dm_w
    h_a[:, 0] = h_a_in
    w_a[:, 0] = w_a_in
    
    # Main simulation loop
    for i in range(Ny):      
        for j in range(Nx):  
            T_water_cell = T_w[i, j]
            m_water_cell = m_w[i, j]
            h_air_cell = h_a[i, j]
            w_air_cell = w_a[i, j]
            
            cp_w_local = cp_agua_local(T_water_cell)
            
            # Get saturation properties
            if lut is not None:
                w_sw, h_sw = lut.get_ws_hs(T_water_cell)
            else:
                w_sw = humedad_saturacion(T_water_cell, P_atm)
                h_sw = entalpia_saturacion(T_water_cell, w_sw, P_atm)
                
            h_v = H_FG0_DEFAULT + CP_VAPOR_DEFAULT * T_water_cell
            Le = factor_lewis(w_sw, w_air_cell)
            
            # Calculate transfer potentials
            potencial_w = max(0.0, w_sw - w_air_cell)
            potencial_h = (h_sw - h_air_cell) + (Le - 1) * (h_sw - h_air_cell - potencial_w * h_v) + potencial_w * cp_w_local * T_water_cell
            
            # Mass and heat transfer
            agua_evap_celda = K_dA * potencial_w
            calor_transferido = K_dA * potencial_h
            
            # Update air properties
            w_a_next = w_air_cell + (agua_evap_celda / dm_a)
            h_a_next = h_air_cell + (calor_transferido / dm_a)
            
            w_a[i, j+1] = w_a_next
            h_a[i, j+1] = h_a_next
            
            # Calculate air temperature
            T_a_next = temp_aire_desde_entalpia(h_a_next, w_a_next, P_atm)
            T_a_matrix[i, j] = T_a_next
            
            # Check for fog formation (supersaturation)
            w_sat_local = lut.get_ws_hs(T_a_next)[0] if lut is not None else humedad_saturacion(T_a_next, P_atm)
            
            if w_a_next > w_sat_local:
                fog_matrix[i, j] = True
            
            # Update water properties
            m_w_next = max(1e-6, m_water_cell - agua_evap_celda)
            m_w[i+1, j] = m_w_next
            
            den_energia = m_w_next * cp_w_local
            if abs(den_energia) < 1e-6:
                den_energia = 1e-6
            T_w[i+1, j] = (m_water_cell * cp_w_local * T_water_cell - calor_transferido) / den_energia

    # Calculate outlet conditions
    T_w_salida_final = np.average(T_w[Ny, :], weights=m_w[Ny, :])
    agua_evaporada_total = max(0.0, m_w_total - np.sum(m_w[Ny, :]))
    
    return T_w_salida_final, agua_evaporada_total, T_w[:-1, :], w_a[:, 1:], T_a_matrix, fog_matrix

# ==========================================
# CALIBRATION ENGINE
# ==========================================
def resolver_punto_operacion(datos_p, N_celdas, worker_ref, pct_base, pct_span):
    """Resolve tower operating point for given conditions.
    
    Uses binary search to find NTU value that matches target water outlet temp.
    
    Args:
        datos_p: Dictionary with tower parameters and climate data
        N_celdas: Number of matrix cells (Nx=Ny for square grid)
        worker_ref: Worker object with logging/progress signals (optional)
        pct_base: Base percentage for progress reporting
        pct_span: Percentage span for progress reporting
        
    Returns:
        Dictionary with operating point results and tower performance
    """
    from psychro_data import PsicroLUT
    
    P_atm = obtener_presion_barometrica(datos_p['altitud'])
    m_w_total = datos_p['caudal_w'] * 1000.0 / 3600.0 
    m_a_total = datos_p['caudal_a'] * datos_p['densidad_a'] 
    
    T_db = datos_p['T_db_in']
    T_wb = datos_p['T_wb_in']
    T_piscina = datos_p['T_w_in']
    
    # Calculate w_a_in and h_a_in from wet bulb temperature if not provided
    if 'w_a_in' not in datos_p or 'h_a_in' not in datos_p:
        if HAS_COOLPROP:
            try:
                w_a_in = CP.HAPropsSI('W', 'T', T_db + 273.15, 'B', T_wb + 273.15, 'P', P_atm)
                h_a_in = CP.HAPropsSI('H', 'T', T_db + 273.15, 'B', T_wb + 273.15, 'P', P_atm) / 1000.0
            except Exception:
                w_sat_wb = humedad_saturacion(T_wb, P_atm)
                w_a_in = ((H_FG0_DEFAULT - (CP_WATER_DEFAULT - CP_VAPOR_DEFAULT) * T_wb) * w_sat_wb - CP_AIR_DEFAULT * (T_db - T_wb)) / (H_FG0_DEFAULT + CP_VAPOR_DEFAULT * T_db - CP_WATER_DEFAULT * T_wb)
                h_a_in = CP_AIR_DEFAULT * T_db + w_a_in * (H_FG0_DEFAULT + CP_VAPOR_DEFAULT * T_db)
        else:
            w_sat_wb = humedad_saturacion(T_wb, P_atm)
            w_a_in = ((H_FG0_DEFAULT - (CP_WATER_DEFAULT - CP_VAPOR_DEFAULT) * T_wb) * w_sat_wb - CP_AIR_DEFAULT * (T_db - T_wb)) / (H_FG0_DEFAULT + CP_VAPOR_DEFAULT * T_db - CP_WATER_DEFAULT * T_wb)
            h_a_in = CP_AIR_DEFAULT * T_db + w_a_in * (H_FG0_DEFAULT + CP_VAPOR_DEFAULT * T_db)
    else:
        w_a_in = datos_p['w_a_in']
        h_a_in = datos_p['h_a_in']
    
    lut = PsicroLUT(T_min=-20.0, T_max=80.0, step=0.1, P_atm=P_atm)
    
    def objetivo_ntu(NTU_guess):
        """Objective function: target outlet temp - calculated outlet temp"""
        NTU_safe = max(0.01, min(50.0, float(NTU_guess)))
        
        T_sal, _, _, _, _, _ = simular_torre_2d_matriz(
            NTU_safe, T_piscina, m_w_total, h_a_in, w_a_in, m_a_total, 
            P_atm, Nx=N_celdas, Ny=N_celdas, lut=lut
        )
        
        return T_sal - datos_p['T_w_out_target']
    
    try:
        NTU_solucion = root_scalar(objetivo_ntu, bracket=[0.01, 50.0], method='brentq').root
    except Exception:
        NTU_solucion = 1.0
    
    # Calculate full tower performance at solution point
    T_sal_final, evap_kg, Matriz_T_w, Matriz_w_a, Matriz_T_a, Matriz_niebla = simular_torre_2d_matriz(
        NTU_solucion, T_piscina, m_w_total, h_a_in, w_a_in, m_a_total,
        P_atm, Nx=N_celdas, Ny=N_celdas, lut=lut
    )
    
    evap_m3h = evap_kg * 3600.0 / 1000.0
    pct_evap = (evap_m3h / datos_p['caudal_w']) * 100.0
    range_w = T_piscina - T_sal_final
    approach_w = T_sal_final - T_wb
    
    # Calculate thermal power output
    cp_medio = cp_agua_local((T_piscina + T_sal_final) / 2.0)
    Q_kW = m_w_total * cp_medio * range_w
    Q_MWt = Q_kW / 1000.0
    Q_TR = Q_kW / 3.517
    
    L_G_ratio = m_w_total / m_a_total
    
    return {
        'NTU': NTU_solucion,
        'T_salida': T_sal_final,
        'evaporacion_m3h': evap_m3h,
        'pct_evap': pct_evap,
        'range_w': range_w,
        'approach_w': approach_w,
        'Q_MWt': Q_MWt,
        'Q_TR': Q_TR,
        'L_G_ratio': L_G_ratio,
        'Matriz_T_w': Matriz_T_w,
        'Matriz_w_a': Matriz_w_a * 1000.0,
        'Matriz_T_a': Matriz_T_a,
        'Matriz_niebla': Matriz_niebla,
        'hay_niebla': bool(np.any(Matriz_niebla)),
        'T_w_in': T_piscina,
        'num_celdas': N_celdas
    }
