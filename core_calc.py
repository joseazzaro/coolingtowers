"""
Core thermodynamic calculation module - Platform agnostic.
No PyQt5 dependencies. Can be used in web applications, CLI, or other UIs.

Exports:
- Thermodynamic property functions (pressure, humidity, enthalpy)
- Tower 2D matrix simulation engine (crossflow)
- Tower 1D matrix simulation engine (counterflow)
- Operating point resolver
"""

import numpy as np
from scipy.optimize import root_scalar
from functools import lru_cache

# ==========================================
# CONSTANTES TERMODINÁMICAS
# ==========================================
CP_WATER_DEFAULT = 4.184      # kJ/kg.K
CP_AIR_DEFAULT = 1.006        # kJ/kg.K
CP_VAPOR_DEFAULT = 1.86       # kJ/kg.K
H_FG0_DEFAULT = 2501.0        # kJ/kg (calor latente a 0°C)

HAS_COOLPROP = False
try:
    import CoolProp.CoolProp as CP
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False

# ==========================================
# PROPIEDADES ATMOSFÉRICAS
# ==========================================
def obtener_presion_barometrica(altitud_m):
    P0 = 101325.0
    return P0 * (1.0 - 0.6875e-5 * float(altitud_m))**5.2561

# ==========================================
# PROPIEDADES DEL AGUA
# ==========================================
@lru_cache(maxsize=4096)
def cp_agua_local_fast(T_round):
    if HAS_COOLPROP:
        try:
            return float(CP.PropsSI('C', 'T', T_round + 273.15, 'P', 101325, 'Water') / 1000.0)
        except Exception:
            pass
    return CP_WATER_DEFAULT

def cp_agua_local(T_celcius):
    T_clamped = max(-10.0, min(95.0, float(T_celcius)))
    return cp_agua_local_fast(round(T_clamped, 1))

# ==========================================
# PROPIEDADES PSICROMÉTRICAS
# ==========================================
@lru_cache(maxsize=4096)
def humedad_saturacion_fast(T_round, P_atm_round):
    if HAS_COOLPROP:
        try:
            val = CP.HAPropsSI('W', 'T', T_round + 273.15, 'R', 1.0, 'P', P_atm_round)
            if not np.isnan(val) and val > 0:
                return float(val)
        except Exception:
            pass
    
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
    T_clamped = max(-20.0, min(95.0, float(T)))
    return humedad_saturacion_fast(round(T_clamped, 1), round(float(P_atm), -2))

def factor_lewis(w_sw, w):
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
# PROPIEDADES DEL AIRE
# ==========================================
@lru_cache(maxsize=4096)
def entalpia_saturacion_fast(T_round, w_sat_round, P_atm_round):
    if HAS_COOLPROP:
        try:
            val = CP.HAPropsSI('H', 'T', T_round + 273.15, 'W', w_sat_round, 'P', P_atm_round) / 1000.0
            if not np.isnan(val):
                return float(val)
        except Exception:
            pass
    
    return float(CP_AIR_DEFAULT * T_round + w_sat_round * (H_FG0_DEFAULT + CP_VAPOR_DEFAULT * T_round))

def entalpia_saturacion(T, w_sat, P_atm=101325.0):
    T_clamped = max(-20.0, min(95.0, float(T)))
    w_clamped = max(0.0, float(w_sat))
    return entalpia_saturacion_fast(round(T_clamped, 1), round(w_clamped, 4), round(float(P_atm), -2))

def temp_aire_desde_entalpia(h_a, w_a, P_atm=101325.0):
    h_c = max(-50.0, min(500.0, float(h_a)))
    w_c = max(0.0, min(0.1, float(w_a)))
    
    if HAS_COOLPROP:
        try:
            T_kelvin = CP.HAPropsSI('T', 'H', h_c * 1000.0, 'W', w_c, 'P', float(P_atm))
            if not np.isnan(T_kelvin):
                return float(T_kelvin - 273.15)
        except Exception:
            pass
    
    den = CP_AIR_DEFAULT + w_c * CP_VAPOR_DEFAULT
    if abs(den) < 1e-5:
        den = CP_AIR_DEFAULT
    return float((h_c - w_c * H_FG0_DEFAULT) / den)

# ==========================================
# MOTORES DE SIMULACIÓN DE TORRE
# ==========================================
def simular_torre_2d_matriz(NTU_actual, T_w_in, m_w_total, h_a_in, w_a_in, m_a_total, 
                             P_atm=101325.0, Nx=6, Ny=6, lut=None):
    """Simulación bidimensional de flujo cruzado (Poppe). Devuelve w_a en kg/kg."""
    m_a_total_safe = max(1e-4, float(m_a_total))
    
    dm_w = m_w_total / Nx
    dm_a = m_a_total_safe / Ny
    K_dA = (NTU_actual * m_w_total) / (Nx * Ny)
    
    T_w = np.zeros((Ny + 1, Nx))
    m_w = np.zeros((Ny + 1, Nx))
    h_a = np.zeros((Ny, Nx + 1))
    w_a = np.zeros((Ny, Nx + 1))
    
    fog_matrix = np.zeros((Ny, Nx), dtype=bool)
    T_a_matrix = np.zeros((Ny, Nx))
    
    T_w[0, :] = T_w_in
    m_w[0, :] = dm_w
    h_a[:, 0] = h_a_in
    w_a[:, 0] = w_a_in
    
    for i in range(Ny):
        for j in range(Nx):
            T_water_cell = T_w[i, j]
            m_water_cell = m_w[i, j]
            h_air_cell = h_a[i, j]
            w_air_cell = w_a[i, j]
            
            cp_w_local = cp_agua_local(T_water_cell)
            
            if lut is not None:
                w_sw, h_sw = lut.get_ws_hs(T_water_cell)
            else:
                w_sw = humedad_saturacion(T_water_cell, P_atm)
                h_sw = entalpia_saturacion(T_water_cell, w_sw, P_atm)
                
            h_v = H_FG0_DEFAULT + CP_VAPOR_DEFAULT * T_water_cell
            Le = factor_lewis(w_sw, w_air_cell)
            
            potencial_w = max(0.0, w_sw - w_air_cell)
            potencial_h = (h_sw - h_air_cell) + (Le - 1) * (h_sw - h_air_cell - potencial_w * h_v) + potencial_w * cp_w_local * T_water_cell
            
            agua_evap_celda = K_dA * potencial_w
            calor_transferido = K_dA * potencial_h
            
            w_a_next = w_air_cell + (agua_evap_celda / dm_a)
            h_a_next = h_air_cell + (calor_transferido / dm_a)
            
            w_a[i, j+1] = w_a_next
            h_a[i, j+1] = h_a_next
            
            T_a_next = temp_aire_desde_entalpia(h_a_next, w_a_next, P_atm)
            T_a_matrix[i, j] = T_a_next
            
            w_sat_local = lut.get_ws_hs(T_a_next)[0] if lut is not None else humedad_saturacion(T_a_next, P_atm)
            if w_a_next > w_sat_local:
                fog_matrix[i, j] = True
            
            m_w_next = max(1e-6, m_water_cell - agua_evap_celda)
            m_w[i+1, j] = m_w_next
            
            den_energia = m_w_next * cp_w_local
            if abs(den_energia) < 1e-6:
                den_energia = 1e-6
            
            T_w_calculada = (m_water_cell * cp_w_local * T_water_cell - calor_transferido) / den_energia
            T_w[i+1, j] = max(-10.0, min(T_w_in, T_w_calculada))

    T_w_salida_final = np.average(T_w[Ny, :], weights=m_w[Ny, :])
    agua_evaporada_total = max(0.0, m_w_total - np.sum(m_w[Ny, :]))
    
    return T_w_salida_final, agua_evaporada_total, T_w[:-1, :], w_a[:, 1:], T_a_matrix, fog_matrix


def _integrar_ntu_counterflow(T_in, T_out, h_a_in, m_w, m_a, P_atm, lut):
    """Calcula el NTU analítico requerido en contraflujo mediante cuadratura de Gauss (5 puntos)."""
    if T_out >= T_in:
        return 0.0
    
    cp_medio = cp_agua_local((T_in + T_out) / 2.0)
    Q_transf = m_w * cp_medio * (T_in - T_out)
    
    # Nodos y pesos normalizados de Gauss-Legendre en [-1, 1]
    xi = np.array([-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459])
    wi = np.array([0.2369268850, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850])
    
    # Cambio de variable de [-1, 1] a [T_out, T_in]
    T_eval = 0.5 * (T_in - T_out) * xi + 0.5 * (T_in + T_out)
    
    integral_val = 0.0
    for k in range(5):
        Tw = T_eval[k]
        if lut is not None:
            _, h_sw = lut.get_ws_hs(Tw)
        else:
            w_sw = humedad_saturacion(Tw, P_atm)
            h_sw = entalpia_saturacion(Tw, w_sw, P_atm)
            
        ha_local = h_a_in + (Q_transf / m_a) * (Tw - T_out) / (T_in - T_out)
        delta_h = max(0.2, h_sw - ha_local)
        integral_val += wi[k] * (cp_medio / delta_h)
        
    return float(0.5 * (T_in - T_out) * integral_val)


def simular_torre_counterflow_1d(NTU_actual, T_w_in, m_w_total, h_a_in, w_a_in, m_a_total, 
                                P_atm=101325.0, Nz=15, lut=None):
    """Simulación 1D de Contraflujo analítica, monótona y exacta."""
    m_a_safe = max(1e-4, float(m_a_total))
    m_w_safe = max(1e-4, float(m_w_total))

    # OBTENCIÓN CORRECTA DE BULBO HÚMEDO (T_wb_in) DESDE LA ENTALPÍA DE ENTRADA
    # La temperatura de bulbo húmedo es la temperatura de saturación cuya entalpía es h_a_in
    if lut is not None:
        idx_wb = int(np.searchsorted(lut.hs_lut, h_a_in))
        idx_wb = max(0, min(len(lut.T_grid) - 1, idx_wb))
        T_wb_in = float(lut.T_grid[idx_wb])
    else:
        # Aproximación psicrométrica inversa rápida de T_wb
        T_db_est = temp_aire_desde_entalpia(h_a_in, w_a_in, P_atm)
        T_wb_in = max(0.0, T_db_est - (h_a_in - (CP_AIR_DEFAULT * T_db_est + w_a_in * H_FG0_DEFAULT)) / 2.5)

    # 1. Bisección monótona de T_w_out entre el bulbo húmedo real y T_w_in
    T_low = T_wb_in + 0.1
    T_high = T_w_in - 0.1

    for _ in range(35):
        T_mid = 0.5 * (T_low + T_high)
        ntu_mid = _integrar_ntu_counterflow(T_w_in, T_mid, h_a_in, m_w_safe, m_a_safe, P_atm, lut)
        # A menor temperatura de salida (más fría), mayor es el NTU requerido
        if ntu_mid > NTU_actual:
            T_low = T_mid
        else:
            T_high = T_mid

    T_w_out_final = 0.5 * (T_low + T_high)

    # 2. Generación del perfil espacial vertical: Techo (k=0) a Piscina (k=Nz-1)
    T_w_prof = np.linspace(T_w_in, T_w_out_final, Nz)
    cp_medio = cp_agua_local((T_w_in + T_w_out_final) / 2.0)
    Q_transf = m_w_safe * cp_medio * (T_w_in - T_w_out_final)

    w_a_prof = np.zeros(Nz)
    h_a_prof = np.zeros(Nz)
    T_a_prof = np.zeros(Nz)
    fog_prof = np.zeros(Nz, dtype=bool)

    K_dA_capa = (NTU_actual * m_w_safe) / Nz
    w_a_curr = w_a_in
    evap_total_kg_s = 0.0

    # Integración ascendente del aire desde la base (Nz-1) hacia el techo (0)
    for k in range(Nz - 1, -1, -1):
        Tw_k = T_w_prof[k]
        if lut is not None:
            w_sw, _ = lut.get_ws_hs(Tw_k)
        else:
            w_sw = humedad_saturacion(Tw_k, P_atm)

        pot_w = max(0.0, w_sw - w_a_curr)
        dm_evap = min(K_dA_capa * pot_w, 0.02 * m_w_safe / Nz)
        evap_total_kg_s += dm_evap

        w_a_curr += dm_evap / m_a_safe
        
        frac_calor = (T_w_in - Tw_k) / max(1e-4, (T_w_in - T_w_out_final))
        ha_k = h_a_in + (Q_transf / m_a_safe) * (1.0 - frac_calor)

        w_a_prof[k] = w_a_curr
        h_a_prof[k] = ha_k

        Ta_k = temp_aire_desde_entalpia(ha_k, w_a_curr, P_atm)
        T_a_prof[k] = Ta_k
        w_sat_local = lut.get_ws_hs(Ta_k)[0] if lut is not None else humedad_saturacion(Ta_k, P_atm)
        fog_prof[k] = (w_a_curr > w_sat_local)

    T_w_matriz = np.tile(T_w_prof, (Nz, 1)).T
    w_a_matriz = np.tile(w_a_prof, (Nz, 1))
    T_a_matriz = np.tile(T_a_prof, (Nz, 1))
    fog_matriz = np.tile(fog_prof, (Nz, 1))

    return float(T_w_out_final), float(evap_total_kg_s), T_w_matriz, w_a_matriz, T_a_matriz, fog_matriz


# ==========================================
# MOTOR DE CALIBRACIÓN DE OPERACIÓN
# ==========================================
def resolver_punto_operacion(datos_p, N_celdas, worker_ref=None, pct_base=0, pct_span=100, tipo_equipo=None):
    """Calibra el NTU requerido por la torre según el equipo seleccionado."""
    from psychro_data import PsicroLUT
    
    if tipo_equipo is None:
        tipo_equipo = datos_p.get('tipo_equipo', 'crossflow')
    
    P_atm = obtener_presion_barometrica(datos_p['altitud'])
    m_w_total = datos_p['caudal_w'] * 1000.0 / 3600.0
    m_a_total = datos_p['caudal_a'] * datos_p['densidad_a']
    
    T_db = datos_p['T_db_in']
    T_wb = datos_p['T_wb_in']
    T_piscina = datos_p['T_w_in']
    T_target = datos_p['T_w_out_target']
    
    if 'w_a_in' not in datos_p or 'h_a_in' not in datos_p:
        w_sat_wb = humedad_saturacion(T_wb, P_atm)
        w_a_in = ((H_FG0_DEFAULT - (CP_WATER_DEFAULT - CP_VAPOR_DEFAULT) * T_wb) * w_sat_wb - CP_AIR_DEFAULT * (T_db - T_wb)) / (H_FG0_DEFAULT + CP_VAPOR_DEFAULT * T_db - CP_WATER_DEFAULT * T_wb)
        h_a_in = CP_AIR_DEFAULT * T_db + w_a_in * (H_FG0_DEFAULT + CP_VAPOR_DEFAULT * T_db)
    else:
        w_a_in = datos_p['w_a_in']
        h_a_in = datos_p['h_a_in']
    
    lut = PsicroLUT(T_min=-20.0, T_max=80.0, step=0.1, P_atm=P_atm)

    # 1. MODO CONTRAFLUJO (Cálculo directo por cuadratura)
    if tipo_equipo == 'counterflow':
        NTU_solucion = _integrar_ntu_counterflow(
            T_piscina, T_target, h_a_in, m_w_total, m_a_total, P_atm, lut
        )
        T_sal_final, evap_kg, Matriz_T_w, Matriz_w_a, Matriz_T_a, Matriz_niebla = simular_torre_counterflow_1d(
            NTU_solucion, T_piscina, m_w_total, h_a_in, w_a_in, m_a_total, P_atm, Nz=N_celdas, lut=lut
        )
    else:
        # 2. MODO FLUJO CRUZADO 2D (Búsqueda de raíz por Brent)
        def objetivo_ntu(NTU_guess):
            T_sal, _, _, _, _, _ = simular_torre_2d_matriz(
                NTU_guess, T_piscina, m_w_total, h_a_in, w_a_in, m_a_total, 
                P_atm, Nx=N_celdas, Ny=N_celdas, lut=lut
            )
            return T_sal - T_target

        try:
            NTU_solucion = root_scalar(objetivo_ntu, bracket=[0.05, 15.0], method='brentq').root
        except Exception:
            NTU_solucion = 3.1769

        T_sal_final, evap_kg, Matriz_T_w, Matriz_w_a, Matriz_T_a, Matriz_niebla = simular_torre_2d_matriz(
            NTU_solucion, T_piscina, m_w_total, h_a_in, w_a_in, m_a_total,
            P_atm, Nx=N_celdas, Ny=N_celdas, lut=lut
        )

    evap_m3h = evap_kg * 3600.0 / 1000.0
    pct_evap = (evap_m3h / datos_p['caudal_w']) * 100.0
    range_w = T_piscina - T_sal_final
    approach_w = T_sal_final - T_wb
    
    cp_medio = cp_agua_local((T_piscina + T_sal_final) / 2.0)
    Q_kW = m_w_total * cp_medio * range_w
    Q_MWt = Q_kW / 1000.0
    Q_TR = Q_kW / 3.517
    
    return {
        'NTU': NTU_solucion,
        'T_salida': T_sal_final,
        'evaporacion_m3h': evap_m3h,
        'pct_evap': pct_evap,
        'range_w': range_w,
        'approach_w': approach_w,
        'Q_MWt': Q_MWt,
        'Q_TR': Q_TR,
        'L_G_ratio': m_w_total / m_a_total,
        'Matriz_T_w': Matriz_T_w,
        'Matriz_w_a': Matriz_w_a * 1000.0,
        'Matriz_T_a': Matriz_T_a,
        'Matriz_niebla': Matriz_niebla,
        'hay_niebla': bool(np.any(Matriz_niebla)),
        'T_w_in': T_piscina,
        'num_celdas': N_celdas,
        'tipo_equipo': tipo_equipo
    }