import sys
import os
import csv
from datetime import datetime, timedelta
from functools import lru_cache
import numpy as np
from scipy.optimize import root_scalar

# Intento de importación de CoolProp con Fallback Automático
HAS_COOLPROP = False
try:
    import CoolProp.CoolProp as CP
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False

# Importaciones de PyQt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QStatusBar,
    QSplitter, QMessageBox, QComboBox, QProgressDialog, QDialog, 
    QDialogButtonBox, QFileDialog, QCheckBox, QDateEdit, QMenuBar, QAction
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QLocale, QDate
from PyQt5.QtGui import QFont, QDoubleValidator, QIntValidator

# Importaciones de Matplotlib para PyQt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# ==========================================
# 1. CONSTANTES TERMODINÁMICAS Y PSICROMETRÍA
# ==========================================
cp_w_def = 4.184     # kJ/kg.K
cp_a_def = 1.006     # kJ/kg.K
cp_v_def = 1.86      # kJ/kg.K
h_fg0_def = 2501.0   # kJ/kg

def obtener_presion_barometrica(altitud_m):
    P0 = 101325.0  # Pa
    return P0 * (1.0 - 0.6875e-5 * float(altitud_m))**5.2561

@lru_cache(maxsize=4096)
def cp_agua_local_fast(T_round):
    if HAS_COOLPROP:
        try:
            return float(CP.PropsSI('C', 'T', T_round + 273.15, 'P', 101325, 'Water') / 1000.0)
        except Exception:
            pass
    return cp_w_def

def cp_agua_local(T_celcius):
    T_clamped = max(-10.0, min(95.0, float(T_celcius)))
    return cp_agua_local_fast(round(T_clamped, 1))

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

@lru_cache(maxsize=4096)
def entalpia_saturacion_fast(T_round, w_sat_round, P_atm_round):
    if HAS_COOLPROP:
        try:
            val = CP.HAPropsSI('H', 'T', T_round + 273.15, 'W', w_sat_round, 'P', P_atm_round) / 1000.0
            if not np.isnan(val):
                return float(val)
        except Exception:
            pass
    return float(cp_a_def * T_round + w_sat_round * (h_fg0_def + cp_v_def * T_round))

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
    den = cp_a_def + w_c * cp_v_def
    if abs(den) < 1e-5:
        den = cp_a_def
    return float((h_c - w_c * h_fg0_def) / den)

class PsicroLUT:
    def __init__(self, T_min=-15.0, T_max=65.0, step=0.1, P_atm=101325.0):
        self.T_min = T_min
        self.T_max = T_max
        self.step = step
        self.P_atm = P_atm
        
        self.T_grid = np.arange(T_min, T_max + step, step)
        self.num_pts = len(self.T_grid)
        
        self.ws_lut = np.zeros(self.num_pts)
        self.hs_lut = np.zeros(self.num_pts)
        
        for idx, T in enumerate(self.T_grid):
            ws = humedad_saturacion(T, P_atm)
            self.ws_lut[idx] = ws
            self.hs_lut[idx] = entalpia_saturacion(T, ws, P_atm)

    def get_ws_hs(self, T):
        idx = int((T - self.T_min) / self.step)
        if idx < 0:
            idx = 0
        elif idx >= self.num_pts:
            idx = self.num_pts - 1
        return self.ws_lut[idx], self.hs_lut[idx]

# ==========================================
# 2. MOTOR POPPE 2D OPTIMIZADO
# ==========================================
def simular_torre_2d_matriz(NTU_actual, T_w_in, m_w_total, h_a_in, w_a_in, m_a_total, P_atm=101325.0, Nx=6, Ny=6, lut=None):
    m_a_total_safe = max(1e-4, float(m_a_total))
    
    dm_w = m_w_total / Nx  
    dm_a = m_a_total_safe / Ny  
    K_dA = (NTU_actual * m_w_total) / (Nx * Ny) 
    
    T_w = np.zeros((Ny + 1, Nx))
    m_w = np.zeros((Ny + 1, Nx))
    h_a = np.zeros((Ny, Nx + 1))
    w_a = np.zeros((Ny, Nx + 1))
    
    matriz_niebla = np.zeros((Ny, Nx), dtype=bool)
    matriz_T_aire = np.zeros((Ny, Nx))
    
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
                
            h_v = h_fg0_def + cp_v_def * T_water_cell
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
            matriz_T_aire[i, j] = T_a_next
            
            w_sat_local = lut.get_ws_hs(T_a_next)[0] if lut is not None else humedad_saturacion(T_a_next, P_atm)
            
            if w_a_next > w_sat_local:
                matriz_niebla[i, j] = True
            
            m_w_next = max(1e-6, m_water_cell - agua_evap_celda)
            m_w[i+1, j] = m_w_next
            
            den_energia = m_w_next * cp_w_local
            if abs(den_energia) < 1e-6:
                den_energia = 1e-6
            T_w[i+1, j] = (m_water_cell * cp_w_local * T_water_cell - calor_transferido) / den_energia

    T_w_salida_final = np.average(T_w[Ny, :], weights=m_w[Ny, :])
    agua_evaporada_total = max(0.0, m_w_total - np.sum(m_w[Ny, :]))
    
    return T_w_salida_final, agua_evaporada_total, T_w[:-1, :], w_a[:, 1:], matriz_T_aire, matriz_niebla

def resolver_punto_operacion(datos_p, N_celdas, worker_ref, pct_base, pct_span):
    P_atm = obtener_presion_barometrica(datos_p['altitud'])
    m_w_total = datos_p['caudal_w'] * 1000.0 / 3600.0 
    m_a_total = datos_p['caudal_a'] * datos_p['densidad_a'] 
    
    T_db = datos_p['T_db_in']
    T_wb = datos_p['T_wb_in']
    
    if HAS_COOLPROP:
        try:
            w_a_in = CP.HAPropsSI('W', 'T', T_db + 273.15, 'B', T_wb + 273.15, 'P', P_atm)
            h_a_in = CP.HAPropsSI('H', 'T', T_db + 273.15, 'B', T_wb + 273.15, 'P', P_atm) / 1000.0
        except Exception:
            w_sat_wb = humedad_saturacion(T_wb, P_atm)
            w_a_in = ((h_fg0_def - (cp_w_def - cp_v_def) * T_wb) * w_sat_wb - cp_a_def * (T_db - T_wb)) / (h_fg0_def + cp_v_def * T_db - cp_w_def * T_wb)
            h_a_in = cp_a_def * T_db + w_a_in * (h_fg0_def + cp_v_def * T_db)
    else:
        w_sat_wb = humedad_saturacion(T_wb, P_atm)
        w_a_in = ((h_fg0_def - (cp_w_def - cp_v_def) * T_wb) * w_sat_wb - cp_a_def * (T_db - T_wb)) / (h_fg0_def + cp_v_def * T_db - cp_w_def * T_wb)
        h_a_in = cp_a_def * T_db + w_a_in * (h_fg0_def + cp_v_def * T_db)

    num_puntos = 25
    ntu_puntos = np.linspace(0.1, 10.0, num_puntos)
    errores = []

    for idx, ntu_val in enumerate(ntu_puntos):
        if worker_ref._is_cancelled:
            return None

        pct = pct_base + int((idx / num_puntos) * pct_span)
        worker_ref.progreso_signal.emit(f"Evaluando malla ({N_celdas}x{N_celdas}) - Paso {idx+1}/{num_puntos}...", pct)
        
        T_calc, _, _, _, _, _ = simular_torre_2d_matriz(
            ntu_val, datos_p['T_w_in'], m_w_total, h_a_in, w_a_in, m_a_total, P_atm,
            Nx=N_celdas, Ny=N_celdas
        )
        errores.append(T_calc - datos_p['T_w_out_target'])

    def objetivo_ntu(NTU_guess):
        T_calc, _, _, _, _, _ = simular_torre_2d_matriz(
            NTU_guess, datos_p['T_w_in'], m_w_total, h_a_in, w_a_in, m_a_total, P_atm,
            Nx=N_celdas, Ny=N_celdas
        )
        return T_calc - datos_p['T_w_out_target']

    bracket_encontrado = None
    for i in range(len(errores) - 1):
        if errores[i] * errores[i+1] <= 0:
            bracket_encontrado = [ntu_puntos[i], ntu_puntos[i+1]]
            break

    if bracket_encontrado is not None:
        res = root_scalar(objetivo_ntu, bracket=bracket_encontrado, method='brentq')
        NTU_calibrado = res.root
    else:
        res = root_scalar(objetivo_ntu, x0=3.0, x1=4.0, method='secant')
        NTU_calibrado = res.root

    T_sal, evap_kg, Matriz_T_w, Matriz_w_a, Matriz_T_a, Matriz_niebla = simular_torre_2d_matriz(
        NTU_calibrado, datos_p['T_w_in'], m_w_total, h_a_in, w_a_in, m_a_total, P_atm,
        Nx=N_celdas, Ny=N_celdas
    )

    evap_m3h = evap_kg * 3600.0 / 1000.0
    pct_evap = (evap_m3h / datos_p['caudal_w']) * 100.0
    range_w = datos_p['T_w_in'] - T_sal
    approach_w = T_sal - T_wb
    
    cp_medio = cp_agua_local((datos_p['T_w_in'] + T_sal) / 2.0)
    Q_kW = m_w_total * cp_medio * range_w
    Q_MWt = Q_kW / 1000.0
    Q_TR = Q_kW / 3.517
    
    L_G_ratio = m_w_total / m_a_total

    return {
        'NTU': NTU_calibrado,
        'T_salida': T_sal,
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
        'T_w_in': datos_p['T_w_in'],
        'num_celdas': N_celdas
    }

# ==========================================
# 3. CONTROLADOR PID DISCRETO
# ==========================================
class ControladorPID:
    def __init__(self, Kp=4.0, Ti=300.0, Td=5.0, u_min=0.0, u_max=100.0):
        self.Kp = float(Kp)
        self.Ti = max(float(Ti), 1.0)
        self.Td = float(Td)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.integral = 0.0
        self.prev_error = None

    def calcular(self, setpoint, medido, dt):
        error = float(medido - setpoint)
        if np.isnan(error) or np.isinf(error):
            error = 0.0

        if self.prev_error is None:
            self.prev_error = error

        P = self.Kp * error
        self.integral += error * dt
        if np.isnan(self.integral) or np.isinf(self.integral):
            self.integral = 0.0

        I = (self.Kp / self.Ti) * self.integral
        D = self.Kp * self.Td * (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        
        u_raw = P + I + D
        if np.isnan(u_raw) or np.isinf(u_raw):
            u_raw = self.u_min

        u_sat = max(self.u_min, min(self.u_max, u_raw))
        
        if u_raw != u_sat:
            self.integral -= error * dt
            if np.isnan(self.integral) or np.isinf(self.integral):
                self.integral = 0.0
            
        return u_sat

# ==========================================
# 4. PARSER LIGERO DE ARCHIVOS EPW
# ==========================================
def leer_archivo_epw(path_epw):
    datos_clima = []
    with open(path_epw, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header_count = 0
        for row in reader:
            if header_count < 8:
                header_count += 1
                continue
            if len(row) > 21:
                try:
                    anio = int(row[0])
                    mes = int(row[1])
                    dia = int(row[2])
                    hora = int(row[3]) - 1
                    
                    dt = datetime(2024, mes, dia, hora)
                    tdb = float(row[6])
                    tdew = float(row[7])
                    rh = float(row[8]) / 100.0
                    p_atm = float(row[9])
                    
                    # Intentar leer velocidad del viento (columna 21) si existe
                    try:
                        u_viento = float(row[21])
                    except (IndexError, ValueError):
                        u_viento = 3.5  # m/s por defecto
                    
                    twb = tdb * np.arctan(0.151977 * (rh * 100.0 + 8.313659)**0.5) + np.arctan(tdb + rh * 100.0) - np.arctan(rh * 100.0 - 1.676331) + 0.00391838 * (rh * 100.0)**1.5 * np.arctan(0.023101 * rh * 100.0) - 4.686035
                    
                    datos_clima.append({
                        'dt': dt,
                        'tdb': tdb,
                        'twb': twb,
                        'rh': rh,
                        'patm': p_atm,
                        'u_viento': u_viento
                    })
                except (ValueError, IndexError):
                    continue
    return datos_clima

# ==========================================
# 5. HILO DE SIMULACIÓN DINÁMICA
# ==========================================
class CalibracionWorker(QThread):
    progreso_signal = pyqtSignal(str, int)
    exito_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    cancelado_signal = pyqtSignal()

    def __init__(self, datos_input_p1, datos_input_p2=None):
        super().__init__()
        self.d1 = datos_input_p1
        self.d2 = datos_input_p2
        self._is_cancelled = False

    def cancelar(self):
        self._is_cancelled = True

    def run(self):
        try:
            N_celdas = self.d1['num_celdas']

            if self.d2 is None:
                res1 = resolver_punto_operacion(self.d1, N_celdas, self, pct_base=5, pct_span=85)
                if res1 is None or self._is_cancelled:
                    self.cancelado_signal.emit()
                    return
                
                res1['es_dual'] = False
                self.progreso_signal.emit("Completado", 100)
                self.exito_signal.emit(res1)

            else:
                self.progreso_signal.emit("Calibrando Punto 1...", 5)
                res1 = resolver_punto_operacion(self.d1, N_celdas, self, pct_base=5, pct_span=40)
                if res1 is None or self._is_cancelled:
                    self.cancelado_signal.emit()
                    return

                self.progreso_signal.emit("Calibrando Punto 2...", 50)
                res2 = resolver_punto_operacion(self.d2, N_celdas, self, pct_base=50, pct_span=40)
                if res2 is None or self._is_cancelled:
                    self.cancelado_signal.emit()
                    return

                lg1 = res1['L_G_ratio']
                lg2 = res2['L_G_ratio']
                ntu1 = res1['NTU']
                ntu2 = res2['NTU']

                if abs(lg1 - lg2) < 1e-5:
                    m_exp = 0.6
                else:
                    m_exp = - np.log(ntu1 / ntu2) / np.log(lg1 / lg2)

                c_coef = ntu1 / (lg1 ** (-m_exp))

                res1['es_dual'] = True
                res1['c_coef'] = c_coef
                res1['m_exp'] = m_exp
                res1['p2'] = res2

                self.progreso_signal.emit("Ajuste de 2 puntos completado", 100)
                self.exito_signal.emit(res1)

        except Exception as e:
            if not self._is_cancelled:
                self.error_signal.emit(str(e))

class SimularDinamicaWorker(QThread):
    progreso_signal = pyqtSignal(str, int)
    exito_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    cancelado_signal = pyqtSignal()

    def __init__(self, config_sim):
        super().__init__()
        self.cfg = config_sim
        self._is_cancelled = False

    def cancelar(self):
        self._is_cancelled = True

    def run(self):
        try:
            self.progreso_signal.emit("Cargando y procesando archivo climático EPW...", 5)
            clima = leer_archivo_epw(self.cfg['path_epw'])
            if not clima:
                raise ValueError("No se pudieron extraer datos válidos del archivo EPW.")

            f_ini = self.cfg['fecha_inicio']
            f_fin = self.cfg['fecha_fin']
            clima_filtrado = [c for c in clima if f_ini <= c['dt'] <= f_fin]
            
            if not clima_filtrado:
                raise ValueError("El rango de fechas seleccionado no coincide con el archivo EPW.")

            dt_sim_sec = self.cfg['dt_sim_sec']
            
            t_epw_sec = np.array([(c['dt'] - clima_filtrado[0]['dt']).total_seconds() for c in clima_filtrado])
            tdb_epw = np.array([c['tdb'] for c in clima_filtrado])
            twb_epw = np.array([c['twb'] for c in clima_filtrado])
            patm_epw = np.array([c['patm'] for c in clima_filtrado])
            uviento_epw = np.array([c['u_viento'] for c in clima_filtrado])

            t_total_sec = t_epw_sec[-1]
            time_steps_sec = np.arange(0, t_total_sec + dt_sim_sec, dt_sim_sec)
            total_pasos = len(time_steps_sec)

            self.progreso_signal.emit("Pre-interpolando vectores climáticos con NumPy...", 7)
            tdb_vec = np.interp(time_steps_sec, t_epw_sec, tdb_epw)
            twb_vec = np.interp(time_steps_sec, t_epw_sec, twb_epw)
            patm_vec = np.interp(time_steps_sec, t_epw_sec, patm_epw)
            uviento_vec = np.interp(time_steps_sec, t_epw_sec, uviento_epw)

            self.progreso_signal.emit("Inicializando Tabla Psicrométrica Fast-LUT...", 9)
            lut = PsicroLUT(T_min=-15.0, T_max=65.0, step=0.1, P_atm=patm_epw[0])

            w_sat_wb_vec = np.array([lut.get_ws_hs(twb_vec[k])[0] for k in range(total_pasos)])
            w_a_in_vec = ((h_fg0_def - (cp_w_def - cp_v_def) * twb_vec) * w_sat_wb_vec - cp_a_def * (tdb_vec - twb_vec)) / (h_fg0_def + cp_v_def * tdb_vec - cp_w_def * twb_vec)
            h_a_in_vec = cp_a_def * tdb_vec + w_a_in_vec * (h_fg0_def + cp_v_def * tdb_vec)

            pid = ControladorPID(
                Kp=self.cfg['kp'], Ti=self.cfg['ti'], Td=self.cfg['td'],
                u_min=self.cfg['speed_min'], u_max=100.0
            )

            caudal_w_m3h = self.cfg['caudal_w_m3h']
            m_w_nom = caudal_w_m3h * 1000.0 / 3600.0
            m_a_nom = self.cfg['caudal_a_m3s'] * self.cfg['densidad_a']
            NTU_ref = self.cfg['ntu_ref']
            T_set = self.cfg['t_setpoint']

            pct_drift = float(self.cfg.get('pct_drift', 0.005))
            coc = float(self.cfg.get('coc', 4.0))

            v_estanque_m3 = self.cfg['vol_estanque_m3']
            m_estanque_kg = max(100.0, v_estanque_m3 * 1000.0)
            tau_sec = m_estanque_kg / m_w_nom
            
            delta_T_proceso = self.cfg['t_w_in_nom'] - T_set

            p_motor_nom_kw = self.cfg['p_motor_kw']
            eta_glob = max(0.1, min(1.0, self.cfg['eta_fan_pct'] / 100.0))

            T_piscina = T_set + 0.2
            T_w_in_dinamica = T_piscina + delta_T_proceso

            tdb_0 = tdb_vec[0]
            twb_0 = twb_vec[0]
            patm_0 = patm_vec[0]
            w_a_in0 = w_a_in_vec[0]
            h_a_in0 = h_a_in_vec[0]

            for _ in range(20):
                u_init = pid.calcular(T_set, T_piscina, dt_sim_sec)
                m_a_init = max(1e-4, m_a_nom * (u_init / 100.0))
                T_sal_init, _, _, _, _, _ = simular_torre_2d_matriz(
                    NTU_ref, T_w_in_dinamica, m_w_nom, h_a_in0, w_a_in0, m_a_init, patm_0, Nx=6, Ny=6, lut=lut
                )
                T_sal_init = max(twb_0, min(T_w_in_dinamica, T_sal_init))
                T_piscina = T_piscina + min(1.0, dt_sim_sec / tau_sec) * (T_sal_init - T_piscina)
                T_w_in_dinamica = T_piscina + delta_T_proceso

            times, t_out_arr, t_in_arr, fan_speed_arr, t_wb_arr, t_db_arr, t_a_out_arr, evap_arr, q_mwt_arr, niebla_arr = [], [], [], [], [], [], [], [], [], []
            power_kw_arr = []
            
            energia_acum_kwh = 0.0
            energia_disipada_mwh_t = 0.0
            agua_evap_total_m3 = 0.0
            agua_drift_total_m3 = 0.0
            agua_purga_total_m3 = 0.0

            fraccion_renovacion = min(1.0, dt_sim_sec / tau_sec)
            drift_m3h = caudal_w_m3h * (pct_drift / 100.0)

            for idx in range(total_pasos):
                if self._is_cancelled:
                    self.cancelado_signal.emit()
                    return

                if idx % max(1, total_pasos // 50) == 0:
                    pct = int(10 + (idx / total_pasos) * 85)
                    sec = time_steps_sec[idx]
                    dt_actual = clima_filtrado[0]['dt'] + timedelta(seconds=float(sec))
                    self.progreso_signal.emit(f"Simulando {dt_actual.strftime('%d/%m %H:%M')}...", pct)

                tdb_k = tdb_vec[idx]
                twb_k = twb_vec[idx]
                patm_k = patm_vec[idx]
                w_a_in_k = w_a_in_vec[idx]
                h_a_in_k = h_a_in_vec[idx]

                velocidad_pct = pid.calcular(T_set, T_piscina, dt_sim_sec)
                m_a_actual = max(1e-4, m_a_nom * (velocidad_pct / 100.0))

                u_ratio = velocidad_pct / 100.0
                p_elec_instantanea_kw = (p_motor_nom_kw / eta_glob) * (u_ratio ** 3)
                energia_acum_kwh += p_elec_instantanea_kw * (dt_sim_sec / 3600.0)

                T_salida_inst, evap_kg_s, _, _, Matriz_T_a, Matriz_niebla = simular_torre_2d_matriz(
                    NTU_ref, T_w_in_dinamica, m_w_nom, h_a_in_k, w_a_in_k, m_a_actual, patm_k,
                    Nx=6, Ny=6, lut=lut
                )

                T_salida_inst = max(twb_k, min(T_w_in_dinamica, T_salida_inst))
                T_a_out_prom = np.mean(Matriz_T_a[:, -1])
                hay_niebla_paso = bool(np.any(Matriz_niebla))

                T_piscina = T_piscina + fraccion_renovacion * (T_salida_inst - T_piscina)
                T_piscina = max(twb_k, min(T_w_in_dinamica, T_piscina))

                T_w_in_dinamica = T_piscina + delta_T_proceso

                evap_m3h = max(0.0, evap_kg_s * 3600.0 / 1000.0)
                purga_m3h = max(0.0, (evap_m3h - drift_m3h * (coc - 1.0)) / (coc - 1.0))
                
                dt_horas = (dt_sim_sec / 3600.0)
                agua_evap_total_m3 += evap_m3h * dt_horas
                agua_drift_total_m3 += drift_m3h * dt_horas
                agua_purga_total_m3 += purga_m3h * dt_horas

                Q_MWt = (m_w_nom * cp_w_def * delta_T_proceso) / 1000.0
                energia_disipada_mwh_t += Q_MWt * dt_horas

                dt_actual = clima_filtrado[0]['dt'] + timedelta(seconds=float(time_steps_sec[idx]))
                times.append(dt_actual)
                t_out_arr.append(T_piscina)
                t_in_arr.append(T_w_in_dinamica)
                fan_speed_arr.append(velocidad_pct)
                t_wb_arr.append(twb_k)
                t_db_arr.append(tdb_k)
                t_a_out_arr.append(T_a_out_prom)
                evap_arr.append(evap_m3h)
                q_mwt_arr.append(Q_MWt)
                niebla_arr.append(hay_niebla_paso)
                power_kw_arr.append(p_elec_instantanea_kw)

            agua_total_makeup_m3 = agua_evap_total_m3 + agua_drift_total_m3 + agua_purga_total_m3
            energia_disipada_kwh_t = energia_disipada_mwh_t * 1000.0
            
            cop_torre = (energia_disipada_kwh_t / energia_acum_kwh) if energia_acum_kwh > 0 else 0.0
            intensidad_agua_m3_mwh = (agua_total_makeup_m3 / energia_disipada_mwh_t) if energia_disipada_mwh_t > 0 else 0.0
            intensidad_agua_m3_kwhe = (agua_total_makeup_m3 / energia_acum_kwh) if energia_acum_kwh > 0 else 0.0
            
            vel_promedio_pct = float(np.mean(fan_speed_arr))

            res = {
                'times': times,
                't_out': t_out_arr,
                't_in': t_in_arr,
                'fan_speed': fan_speed_arr,
                't_wb': t_wb_arr,
                't_db': t_db_arr,
                't_a_out': t_a_out_arr,
                'evap': evap_arr,
                'q_mwt': q_mwt_arr,
                'niebla': niebla_arr,
                'power_kw': power_kw_arr,
                'energia_total_kwh': energia_acum_kwh,
                'energia_disipada_mwh_t': energia_disipada_mwh_t,
                'agua_evap_m3': agua_evap_total_m3,
                'agua_drift_m3': agua_drift_total_m3,
                'agua_purga_m3': agua_purga_total_m3,
                'agua_total_makeup_m3': agua_total_makeup_m3,
                'cop_torre': cop_torre,
                'intensidad_agua_m3_mwh': intensidad_agua_m3_mwh,
                'intensidad_agua_m3_kwhe': intensidad_agua_m3_kwhe,
                'vel_promedio_pct': vel_promedio_pct,
                't_setpoint': T_set,
                'viento_medio': float(np.mean(uviento_vec)),
                'caudal_a_m3s': self.cfg['caudal_a_m3s'],
                'p_motor_kw': self.cfg['p_motor_kw']
            }

            self.progreso_signal.emit("Finalizado", 100)
            self.exito_signal.emit(res)

        except Exception as e:
            if not self._is_cancelled:
                self.error_signal.emit(str(e))

# ==========================================
# 6. DIÁLOGO EMERGENTE DE PERFIL DE PLUMA BRIGGS 2D
# ==========================================
# ==========================================
# DIÁLOGO EMERGENTE PARA EL 2º PUNTO
# ==========================================
class DialogoSegundoPunto(QDialog):
    def __init__(self, parent=None, datos_p1=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración del 2º Punto de Funcionamiento")
        self.setFixedSize(380, 420)
        self.datos_p1 = datos_p1
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        lbl_info = QLabel("Ingrese las condiciones medidas para la 2ª prueba operativa:")
        lbl_info.setFont(QFont("Segoe UI", 9))
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color: #2C3E50;")
        layout.addWidget(lbl_info)

        gb = QGroupBox("Condiciones del Punto 2")
        gb.setStyleSheet("""
            QGroupBox {
                font-size: 11px; font-weight: bold; color: #2C3E50;
                border: 1px solid #DCDCDC; border-radius: 4px; margin-top: 6px; padding-top: 10px;
            }
        """)
        grid = QGridLayout(gb)
        grid.setVerticalSpacing(4)

        Tw1_def = f"{self.datos_p1['T_w_in']:.1f}" if self.datos_p1 else "30.0"
        Tw2_def = f"{self.datos_p1['T_w_out_target'] - 1.0:.1f}" if self.datos_p1 else "21.0"
        Cw_def = f"{self.datos_p1['caudal_w'] * 0.85:.1f}" if self.datos_p1 else "1000.0"
        Tdb_def = f"{self.datos_p1['T_db_in']:.1f}" if self.datos_p1 else "28.0"
        Twb_def = f"{self.datos_p1['T_wb_in']:.1f}" if self.datos_p1 else "16.5"
        Ca_def = f"{self.datos_p1['caudal_a']:.1f}" if self.datos_p1 else "474.1"

        self.txt_Tw_in = self.crear_field("Temp. Entrada Agua (T_w1):", Tw1_def, "°C", grid, 0)
        self.txt_Tw_out = self.crear_field("Temp. Salida Agua (T_w2):", Tw2_def, "°C", grid, 1)
        self.txt_caudal_w = self.crear_field("Caudal Volumétrico Agua:", Cw_def, "m³/h", grid, 2)
        self.txt_Tdb_in = self.crear_field("Temp. Bulbo Seco (T_db):", Tdb_def, "°C", grid, 3)
        self.txt_Twb_in = self.crear_field("Temp. Bulbo Húmedo (T_wb):", Twb_def, "°C", grid, 4)
        self.txt_caudal_a = self.crear_field("Caudal Aire Ventilador:", Ca_def, "m³/s", grid, 5)

        layout.addWidget(gb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.button(QDialogButtonBox.Ok).setText("Calibrar Ambas Condiciones")
        buttons.button(QDialogButtonBox.Ok).setStyleSheet("background-color: #34495E; color: white; padding: 5px 10px;")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def crear_field(self, label, default, unit, grid, row):
        lbl = QLabel(label)
        lbl.setFont(QFont("Segoe UI", 9))
        txt = QLineEdit(default)
        txt.setFont(QFont("Segoe UI", 9))
        val = QDoubleValidator()
        val.setLocale(QLocale("C"))
        txt.setValidator(val)
        lbl_u = QLabel(unit)
        lbl_u.setFont(QFont("Segoe UI", 8))
        lbl_u.setStyleSheet("color: #777777;")

        grid.addWidget(lbl, row, 0)
        grid.addWidget(txt, row, 1)
        grid.addWidget(lbl_u, row, 2)
        return txt

    def obtener_datos_p2(self):
        return {
            'T_w_in': float(self.txt_Tw_in.text().replace(',', '.')),
            'T_w_out_target': float(self.txt_Tw_out.text().replace(',', '.')),
            'caudal_w': float(self.txt_caudal_w.text().replace(',', '.')),
            'T_db_in': float(self.txt_Tdb_in.text().replace(',', '.')),
            'T_wb_in': float(self.txt_Twb_in.text().replace(',', '.')),
            'caudal_a': float(self.txt_caudal_a.text().replace(',', '.')),
            'densidad_a': self.datos_p1['densidad_a'],
            'altitud': self.datos_p1['altitud'],
            'num_celdas': self.datos_p1['num_celdas']
        }
    
class DialogoPerfilPluma(QDialog):
    def __init__(self, parent=None, datos_sim=None):
        super().__init__(parent)
        self.setWindowTitle("Perfil Atmosférico de Pluma y Dispersión (Modelo Briggs 2D)")
        self.resize(950, 620)
        self.datos_sim = datos_sim
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Título e Info
        lbl_info = QLabel("Modelo de Dispersión de Pluma Térmica (Briggs - US EPA) y Condensación Psicrométrica")
        lbl_info.setFont(QFont("Segoe UI", 10, QFont.Bold))
        lbl_info.setStyleSheet("color: #2C3E50;")
        layout.addWidget(lbl_info)

        # Canvas Matplotlib para Dibujar la Pluma
        self.fig = Figure(figsize=(8, 4.5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # KPIs de la Pluma
        self.lbl_kpis_pluma = QLabel("Calculando trayectoria de la pluma...")
        self.lbl_kpis_pluma.setFont(QFont("Segoe UI", 9))
        self.lbl_kpis_pluma.setStyleSheet("background-color: #F4F6F7; padding: 8px; border-radius: 4px; color: #1A252F;")
        layout.addWidget(self.lbl_kpis_pluma)

        self.simular_y_graficar_pluma()

    def simular_y_graficar_pluma(self):
        if not self.datos_sim:
            return

        # Extraer variables térmicas medias
        T_a_out = float(np.mean(self.datos_sim['t_a_out']))
        T_db_amb = float(np.mean(self.datos_sim['t_db']))
        T_wb_amb = float(np.mean(self.datos_sim['t_wb']))
        u_wind = max(0.5, float(self.datos_sim.get('viento_medio', 3.5)))
        
        caudal_a = float(self.datos_sim['caudal_a_m3s'])
        vel_fan_pct = float(self.datos_sim['vel_promedio_pct'])
        caudal_a_actual = caudal_a * (vel_fan_pct / 100.0)

        # Geometría estimada de la torre
        H_torre = 12.0  # metros
        D_boca = 4.0    # metros de diámetro de salida
        A_boca = np.pi * (D_boca / 2.0)**2
        w_salida_m_s = caudal_a_actual / A_boca if A_boca > 0 else 2.0

        # Modelo de Elevación de Pluma de Briggs (Flotabilidad Fb y Momento Fm)
        g = 9.81
        T_kelvin_out = T_a_out + 273.15
        T_kelvin_amb = T_db_amb + 273.15
        
        # Parámetro de flotabilidad térmica
        F_b = g * w_salida_m_s * (D_boca**2 / 4.0) * max(0.001, (T_kelvin_out - T_kelvin_amb) / T_kelvin_out)
        
        # Grilla horizontal X (m) de 0 a 150 metros
        x_vec = np.linspace(0.1, 150.0, 300)
        
        # Trayectoria central Z(x) según Briggs
        z_centron = H_torre + (3.0 * F_b * (x_vec**2) / (2.0 * 0.6**2 * (u_wind**3)))**(1.0 / 3.0)
        
        # Ancho de dispersión de la pluma en X (expansión gaussiana)
        sigma_z = 0.1 * (x_vec**0.9) + (D_boca / 2.0)

        # Evaluación Psicrométrica de Mezcla para determinar visibilidad
        w_amb = humedad_saturacion(T_wb_amb)
        w_out = humedad_saturacion(T_a_out)
        
        # Perfil 2D de cota superior e inferior visibles
        z_top = z_centron + sigma_z
        z_bot = np.maximum(0.0, z_centron - sigma_z)

        # Determinar punto de disipación de la pluma (Hum. Relativa < 100%)
        # A mayor x, la pluma se diluye con el aire ambiental
        frac_mezcla = np.exp(-x_vec / 40.0)  # Factor de dilución
        w_pluma_vec = w_amb + (w_out - w_amb) * frac_mezcla
        T_pluma_vec = T_db_amb + (T_a_out - T_db_amb) * frac_mezcla
        
        w_sat_pluma = np.array([humedad_saturacion(t) for t in T_pluma_vec])
        es_visible = (w_pluma_vec >= w_sat_pluma * 0.98) # Margen de saturación

        x_vis = x_vec[es_visible]
        z_top_vis = z_top[es_visible]
        z_bot_vis = z_bot[es_visible]

        L_pluma_vis = float(x_vis[-1]) if len(x_vis) > 0 else 0.0
        H_max_vis = float(np.max(z_top_vis)) if len(z_top_vis) > 0 else H_torre

        # GRAFICADO EN CANVAS
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Estructura física de la torre de enfriamiento
        ax.add_patch(matplotlib.patches.Rectangle((-D_boca, 0), D_boca, H_torre, color='#34495E', alpha=0.8, label='Torre de Enfriamiento'))
        ax.plot([-D_boca/2.0, -D_boca/2.0], [H_torre, H_torre + 1.5], color='#2C3E50', linewidth=3)
        ax.plot([0, 0], [H_torre, H_torre + 1.5], color='#2C3E50', linewidth=3)

        # Pluma Visible (Sombra/Niebla)
        if len(x_vis) > 0:
            ax.fill_between(x_vis, z_bot_vis, z_top_vis, color='#BDC3C7', alpha=0.6, label='Pluma Visible (Niebla/Condensación)')
            ax.plot(x_vis, z_centron[es_visible], color='#7F8C8D', linestyle='--', linewidth=1.5, label='Eje Central de Pluma')

        # Pluma Disipada (Aire Seco)
        ax.plot(x_vec[~es_visible], z_centron[~es_visible], color='#3498DB', linestyle=':', alpha=0.5, label='Eje de Dispersión Térmica (Incalculable a simple vista)')

        # Vector de Viento
        ax.annotate('', xy=(15, H_torre + 12), xytext=(2, H_torre + 12),
                    arrowprops=dict(facecolor='#E74C3C', edgecolor='#E74C3C', arrowstyle='->', lw=2))
        ax.text(8, H_torre + 13.5, f"Viento Ambient.: {u_wind:.1f} m/s", color='#C0392B', fontsize=8, fontweight='bold')

        ax.set_xlim(-10, 140)
        ax.set_ylim(0, max(45, H_max_vis + 10))
        ax.set_xlabel("Distancia Horizontal en Dirección del Viento (m)", fontsize=9)
        ax.set_ylabel("Altura sobre el Nivel del Suelo (m)", fontsize=9)
        ax.set_title(f"Perfil 2D de Dispersión de Pluma Atmosférica - T_salida={T_a_out:.1f}°C | T_amb={T_db_amb:.1f}°C", fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

        # Actualizar Etiquetas de KPIs
        downwash_risk = "ALTO (Viento Fuerte agazapa la pluma)" if u_wind > 6.0 else "BAJO (Ascenso térmico seguro)"
        self.lbl_kpis_pluma.setText(
            f"<b>Longitud Máxima Pluma Visible:</b> {L_pluma_vis:.1f} m | "
            f"<b>Altura Máxima Alcanzada:</b> {H_max_vis:.1f} m | "
            f"<b>Riesgo de Recirculación / Downwash:</b> {downwash_risk}"
        )

# ==========================================
# 7. VENTANA EMERGENTE DE SIMULACIÓN DINÁMICA
# ==========================================
class VentanaSimulacionDinamica(QDialog):
    def __init__(self, parent=None, datos_torre=None):
        super().__init__(parent)
        self.setWindowTitle("Simulación Dinámica Anual / Climática (.EPW) con PID y Balance Hídrico")
        
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint | Qt.WindowMaximizeButtonHint)
        self.resize(1340, 820)
        self.setMinimumSize(980, 640)
        
        self.datos_torre = datos_torre
        self.res_sim = None

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        panel_cfg = QWidget()
        layout_cfg = QVBoxLayout(panel_cfg)
        layout_cfg.setContentsMargins(5, 5, 5, 5)

        estilo_gb = "QGroupBox { font-size: 11px; font-weight: bold; color: #2C3E50; border: 1px solid #DCDCDC; border-radius: 4px; margin-top: 6px; padding-top: 10px; }"

        gb_epw = QGroupBox("1. Archivo Climático EPW")
        gb_epw.setStyleSheet(estilo_gb)
        grid_epw = QGridLayout(gb_epw)

        self.txt_epw_path = QLineEdit()
        self.txt_epw_path.setPlaceholderText("Seleccione archivo .epw...")
        self.txt_epw_path.setFont(QFont("Segoe UI", 8))
        btn_epw = QPushButton("Examinar...")
        btn_epw.setFont(QFont("Segoe UI", 8))
        btn_epw.clicked.connect(self.examinar_epw)

        grid_epw.addWidget(self.txt_epw_path, 0, 0)
        grid_epw.addWidget(btn_epw, 0, 1)
        layout_cfg.addWidget(gb_epw)

        gb_tiempo = QGroupBox("2. Rango Temporal, Estanque y Purga")
        gb_tiempo.setStyleSheet(estilo_gb)
        grid_tiempo = QGridLayout(gb_tiempo)

        self.date_ini = QDateEdit(QDate(2024, 1, 1))
        self.date_ini.setDisplayFormat("dd/MM/yyyy")
        self.date_fin = QDateEdit(QDate(2024, 1, 7))
        self.date_fin.setDisplayFormat("dd/MM/yyyy")

        self.txt_dt_sim = QLineEdit("300")
        self.txt_vol_estanque = QLineEdit("15.0")
        self.txt_coc = QLineEdit("4.0")           
        self.txt_drift = QLineEdit("0.005")       

        for txt in [self.txt_dt_sim, self.txt_vol_estanque, self.txt_coc, self.txt_drift]:
            txt.setFont(QFont("Segoe UI", 9))

        grid_tiempo.addWidget(QLabel("Fecha Inicio:"), 0, 0)
        grid_tiempo.addWidget(self.date_ini, 0, 1)
        grid_tiempo.addWidget(QLabel("Fecha Fin:"), 1, 0)
        grid_tiempo.addWidget(self.date_fin, 1, 1)
        grid_tiempo.addWidget(QLabel("Paso Tiempo Δt:"), 2, 0)
        grid_tiempo.addWidget(self.txt_dt_sim, 2, 1)
        grid_tiempo.addWidget(QLabel("seg"), 2, 2)
        grid_tiempo.addWidget(QLabel("Vol. Estanque:"), 3, 0)
        grid_tiempo.addWidget(self.txt_vol_estanque, 3, 1)
        grid_tiempo.addWidget(QLabel("m³"), 3, 2)
        grid_tiempo.addWidget(QLabel("Ciclos Concentración (COC):"), 4, 0)
        grid_tiempo.addWidget(self.txt_coc, 4, 1)
        grid_tiempo.addWidget(QLabel("Arrastre / Drift:"), 5, 0)
        grid_tiempo.addWidget(self.txt_drift, 5, 1)
        grid_tiempo.addWidget(QLabel("%"), 5, 2)

        layout_cfg.addWidget(gb_tiempo)

        gb_pid = QGroupBox("3. Controlador PID del Ventilador")
        gb_pid.setStyleSheet(estilo_gb)
        grid_pid = QGridLayout(gb_pid)

        self.txt_t_set = QLineEdit("20.6")
        self.txt_kp = QLineEdit("4.0")
        self.txt_ti = QLineEdit("300")
        self.txt_td = QLineEdit("5")
        self.txt_speed_min = QLineEdit("20.0")

        for txt in [self.txt_t_set, self.txt_kp, self.txt_ti, self.txt_td, self.txt_speed_min]:
            txt.setFont(QFont("Segoe UI", 9))

        grid_pid.addWidget(QLabel("Setpoint Temp. Agua:"), 0, 0)
        grid_pid.addWidget(self.txt_t_set, 0, 1)
        grid_pid.addWidget(QLabel("°C"), 0, 2)

        grid_pid.addWidget(QLabel("Ganancia Kp:"), 1, 0)
        grid_pid.addWidget(self.txt_kp, 1, 1)

        grid_pid.addWidget(QLabel("Tiempo Integral Ti:"), 2, 0)
        grid_pid.addWidget(self.txt_ti, 2, 1)
        grid_pid.addWidget(QLabel("s"), 2, 2)

        grid_pid.addWidget(QLabel("Tiempo Derivativo Td:"), 3, 0)
        grid_pid.addWidget(self.txt_td, 3, 1)
        grid_pid.addWidget(QLabel("s"), 3, 2)

        grid_pid.addWidget(QLabel("Velocidad Mínima:"), 4, 0)
        grid_pid.addWidget(self.txt_speed_min, 4, 1)
        grid_pid.addWidget(QLabel("%"), 4, 2)

        layout_cfg.addWidget(gb_pid)

        gb_motor = QGroupBox("4. Motor y Eficiencia")
        gb_motor.setStyleSheet(estilo_gb)
        grid_motor = QGridLayout(gb_motor)

        self.txt_p_motor = QLineEdit("150.0")
        self.txt_p_motor.setFont(QFont("Segoe UI", 9))
        self.txt_eta_fan = QLineEdit("75.0")
        self.txt_eta_fan.setFont(QFont("Segoe UI", 9))

        grid_motor.addWidget(QLabel("Potencia Motor:"), 0, 0)
        grid_motor.addWidget(self.txt_p_motor, 0, 1)
        grid_motor.addWidget(QLabel("kW"), 0, 2)

        grid_motor.addWidget(QLabel("Eficiencia Global:"), 1, 0)
        grid_motor.addWidget(self.txt_eta_fan, 1, 1)
        grid_motor.addWidget(QLabel("%"), 1, 2)

        layout_cfg.addWidget(gb_motor)

        self.btn_ejecutar = QPushButton("🚀 Ejecutar Simulación Dinámica")
        self.btn_ejecutar.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.btn_ejecutar.setStyleSheet("background-color: #27AE60; color: white; padding: 8px; border-radius: 3px;")
        self.btn_ejecutar.clicked.connect(self.ejecutar_simulacion)
        layout_cfg.addWidget(self.btn_ejecutar)

        gb_kpi = QGroupBox("📊 Balance de Agua y KPIs del Período")
        gb_kpi.setStyleSheet(estilo_gb)
        layout_kpi = QVBoxLayout(gb_kpi)
        layout_kpi.setSpacing(3)

        self.lbl_q_disipada = QLabel("Energía Disipada: -- MWh_t")
        self.lbl_kwh_total = QLabel("Energía Consumida: -- kWh_e")
        self.lbl_m3_evap = QLabel("Agua Evaporada (E): -- m³")
        self.lbl_m3_purga = QLabel("Agua Purga (B): -- m³")
        self.lbl_m3_drift = QLabel("Agua Arrastre (D): -- m³")
        self.lbl_m3_total = QLabel("Reposición Total (Make-up): -- m³")
        self.lbl_cop = QLabel("Rendimiento (COP): -- kWh_t/kWh_e")
        self.lbl_int_agua_mwh = QLabel("Consumo Espec. Agua: -- m³/MWh_t")

        for lbl in [self.lbl_q_disipada, self.lbl_kwh_total, self.lbl_m3_evap, self.lbl_m3_purga, 
                    self.lbl_m3_drift, self.lbl_m3_total, self.lbl_cop, self.lbl_int_agua_mwh]:
            lbl.setFont(QFont("Segoe UI", 8))
            lbl.setStyleSheet("color: #1A252F;")
            layout_kpi.addWidget(lbl)

        layout_cfg.addWidget(gb_kpi)
        layout_cfg.addStretch()

        panel_grafica = QWidget()
        layout_graf = QVBoxLayout(panel_grafica)
        layout_graf.setContentsMargins(5, 5, 5, 5)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout_graf.addWidget(self.toolbar)
        layout_graf.addWidget(self.canvas)

        panel_derecho = QWidget()
        layout_der = QVBoxLayout(panel_derecho)
        layout_der.setContentsMargins(5, 5, 5, 5)

        gb_vars = QGroupBox("Variables a Graficar")
        gb_vars.setStyleSheet(estilo_gb)
        layout_chk = QVBoxLayout(gb_vars)
        layout_chk.setSpacing(6)

        self.chk_tin = QCheckBox("Temp. Entrada Agua Tw1 (°C)")
        self.chk_tin.setChecked(True)
        self.chk_tout = QCheckBox("Temp. Salida Agua Tw2 (°C)")
        self.chk_tout.setChecked(True)
        self.chk_speed = QCheckBox("Velocidad Ventilador (%)")
        self.chk_speed.setChecked(True)
        self.chk_power = QCheckBox("Potencia Eléctrica (kW)")
        self.chk_power.setChecked(False)
        self.chk_twb = QCheckBox("Bulbo Húmedo Twb (°C)")
        self.chk_twb.setChecked(True)
        self.chk_tdb = QCheckBox("Bulbo Seco Ext. Tdb (°C)")
        self.chk_tdb.setChecked(False)
        self.chk_taout = QCheckBox("Temp. Salida Aire Ta,out (°C)")
        self.chk_taout.setChecked(False)
        self.chk_niebla = QCheckBox("Presencia Niebla (Sombra)")
        self.chk_niebla.setChecked(True)
        self.chk_q = QCheckBox("Carga Térmica (MWt)")
        self.chk_q.setChecked(True)
        self.chk_evap = QCheckBox("Evaporación (m³/h)")
        self.chk_evap.setChecked(True)

        for chk in [self.chk_tin, self.chk_tout, self.chk_speed, self.chk_power, self.chk_twb, 
                    self.chk_tdb, self.chk_taout, self.chk_niebla, self.chk_q, self.chk_evap]:
            chk.setFont(QFont("Segoe UI", 8))
            chk.stateChanged.connect(self.replot)
            layout_chk.addWidget(chk)

        layout_chk.addStretch()
        gb_vars.setLayout(layout_chk)
        layout_der.addWidget(gb_vars)

        main_layout.addWidget(panel_cfg, stretch=2)
        main_layout.addWidget(panel_grafica, stretch=7)
        main_layout.addWidget(panel_derecho, stretch=2)

    def examinar_epw(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo climático EPW", "", "Archivos EPW (*.epw);;Todos los archivos (*.*)"
        )
        if file_path:
            self.txt_epw_path.setText(file_path)

    def ejecutar_simulacion(self):
        path_epw = self.txt_epw_path.text()
        if not path_epw or not os.path.exists(path_epw):
            QMessageBox.warning(self, "Archivo Faltante", "Por favor seleccione un archivo .epw válido.")
            return

        try:
            d_ini = self.date_ini.date()
            d_fin = self.date_fin.date()

            cfg = {
                'path_epw': path_epw,
                'fecha_inicio': datetime(2024, d_ini.month(), d_ini.day(), 0, 0),
                'fecha_fin': datetime(2024, d_fin.month(), d_fin.day(), 23, 59),
                'dt_sim_sec': float(self.txt_dt_sim.text()),
                'vol_estanque_m3': float(self.txt_vol_estanque.text()),
                'coc': float(self.txt_coc.text()),
                'pct_drift': float(self.txt_drift.text()),
                't_setpoint': float(self.txt_t_set.text()),
                'kp': float(self.txt_kp.text()),
                'ti': float(self.txt_ti.text()),
                'td': float(self.txt_td.text()),
                'speed_min': float(self.txt_speed_min.text()),
                'p_motor_kw': float(self.txt_p_motor.text()),
                'eta_fan_pct': float(self.txt_eta_fan.text()),
                'caudal_w_m3h': self.datos_torre['caudal_w'],
                'caudal_a_m3s': self.datos_torre['caudal_a'],
                'densidad_a': self.datos_torre['densidad_a'],
                't_w_in_nom': self.datos_torre['T_w_in'],
                'ntu_ref': self.datos_torre['NTU']
            }

            self.progress = QProgressDialog("Iniciando simulación temporal...", "Cancelar", 0, 100, self)
            self.progress.setWindowTitle("Simulación Dinámica PID")
            self.progress.setWindowModality(Qt.WindowModal)

            self.worker = SimularDinamicaWorker(cfg)
            self.worker.progreso_signal.connect(self.actualizar_progreso)
            self.worker.exito_signal.connect(self.procesar_exito)
            self.worker.error_signal.connect(self.procesar_error)
            self.worker.cancelado_signal.connect(self.procesar_cancelado)

            self.progress.canceled.connect(self.worker.cancelar)
            self.btn_ejecutar.setEnabled(False)
            self.worker.start()

        except ValueError:
            QMessageBox.warning(self, "Entrada Inválida", "Por favor revise los parámetros numéricos ingresados.")

    def actualizar_progreso(self, msg, pct):
        if hasattr(self, 'progress') and self.progress:
            self.progress.setLabelText(msg)
            self.progress.setValue(pct)

    def procesar_exito(self, res):
        if hasattr(self, 'progress') and self.progress:
            self.progress.close()
        self.btn_ejecutar.setEnabled(True)
        self.res_sim = res

        self.lbl_q_disipada.setText(f"Energía Disipada: <b>{res['energia_disipada_mwh_t']:.2f} MWh_t</b>")
        self.lbl_kwh_total.setText(f"Energía Consumida: <b>{res['energia_total_kwh']:.2f} kWh_e</b>")
        self.lbl_m3_evap.setText(f"Agua Evaporada (E): <b>{res['agua_evap_m3']:.2f} m³</b>")
        self.lbl_m3_purga.setText(f"Agua Purga (B): <b>{res['agua_purga_m3']:.2f} m³</b>")
        self.lbl_m3_drift.setText(f"Agua Arrastre (D): <b>{res['agua_drift_m3']:.2f} m³</b>")
        self.lbl_m3_total.setText(f"Reposición Total (Make-up): <b style='color:#D35400;'>{res['agua_total_makeup_m3']:.2f} m³</b>")
        self.lbl_cop.setText(f"Rendimiento (COP): <b>{res['cop_torre']:.2f} kWh_t/kWh_e</b>")
        self.lbl_int_agua_mwh.setText(f"Consumo Espec. Agua: <b>{res['intensidad_agua_m3_mwh']:.3f} m³/MWh_t</b>")

        # Notificar a la ventana principal para activar el menú de Pluma
        if self.parent() and hasattr(self.parent(), 'actualizar_resultado_dinamico'):
            self.parent().actualizar_resultado_dinamico(res)

        self.replot()

    def procesar_cancelado(self):
        if hasattr(self, 'progress') and self.progress:
            self.progress.close()
        self.btn_ejecutar.setEnabled(True)

    def procesar_error(self, err):
        if hasattr(self, 'progress') and self.progress:
            self.progress.close()
        self.btn_ejecutar.setEnabled(True)
        QMessageBox.critical(self, "Error de Simulación", f"Ocurrió un error:\n{err}")

    def replot(self):
        if self.res_sim is None:
            return

        self.fig.clear()
        times = self.res_sim['times']

        mostrar_temperaturas = (
            self.chk_tout.isChecked() or self.chk_tin.isChecked() or 
            self.chk_twb.isChecked() or self.chk_tdb.isChecked() or self.chk_taout.isChecked()
        )
        mostrar_velocidad = self.chk_speed.isChecked()
        mostrar_potencia = self.chk_power.isChecked()
        mostrar_carga = self.chk_q.isChecked()
        mostrar_evap = self.chk_evap.isChecked()

        if not (mostrar_temperaturas or mostrar_velocidad or mostrar_potencia or mostrar_carga or mostrar_evap):
            self.canvas.draw()
            return

        usar_panel_inferior = mostrar_carga or mostrar_evap
        
        if usar_panel_inferior and (mostrar_temperaturas or mostrar_velocidad or mostrar_potencia):
            ax_top = self.fig.add_subplot(211)
            ax_bot = self.fig.add_subplot(212, sharex=ax_top)
        elif usar_panel_inferior:
            ax_top = None
            ax_bot = self.fig.add_subplot(111)
        else:
            ax_top = self.fig.add_subplot(111)
            ax_bot = None

        lines = []

        if ax_top is not None:
            if self.chk_niebla.isChecked() and 'niebla' in self.res_sim:
                niebla_mask = np.array(self.res_sim['niebla'], dtype=bool)
                if np.any(niebla_mask):
                    diff = np.diff(niebla_mask.astype(int))
                    inicios = np.where(diff == 1)[0] + 1
                    fines = np.where(diff == -1)[0] + 1
                    
                    if niebla_mask[0]:
                        inicios = np.insert(inicios, 0, 0)
                    if niebla_mask[-1]:
                        fines = np.append(fines, len(niebla_mask) - 1)
                        
                    for idx_f, (i, f) in enumerate(zip(inicios, fines)):
                        lbl_niebla = 'Pluma/Niebla Activa' if idx_f == 0 else ""
                        ax_top.axvspan(times[i], times[min(f, len(times)-1)], color='#7F8C8D', alpha=0.20, linewidth=0, label=lbl_niebla)

            if mostrar_temperaturas:
                if self.chk_tin.isChecked():
                    l_tin, = ax_top.plot(times, self.res_sim['t_in'], color='#D35400', label='Temp. Entrada Agua Tw1 (°C)', linewidth=1.3, linestyle='--')
                    lines.append(l_tin)
                if self.chk_tout.isChecked():
                    l1, = ax_top.plot(times, self.res_sim['t_out'], color='#C0392B', label='Temp. Salida Agua Tw2 (°C)', linewidth=1.5)
                    lines.append(l1)
                    l_set, = ax_top.plot(times, [self.res_sim['t_setpoint']]*len(times), color='#C0392B', linestyle=':', alpha=0.6, label='Setpoint Agua')
                    lines.append(l_set)
                if self.chk_twb.isChecked():
                    l2, = ax_top.plot(times, self.res_sim['t_wb'], color='#2980B9', linestyle=':', label='Bulbo Húmedo Twb (°C)')
                    lines.append(l2)
                if self.chk_tdb.isChecked():
                    l_tdb, = ax_top.plot(times, self.res_sim['t_db'], color='#16A085', linestyle='-.', label='Bulbo Seco Tdb (°C)', alpha=0.85)
                    lines.append(l_tdb)
                if self.chk_taout.isChecked():
                    l_taout, = ax_top.plot(times, self.res_sim['t_a_out'], color='#8E44AD', linestyle='-', label='Temp. Salida Aire Ta,out (°C)', linewidth=1.2)
                    lines.append(l_taout)

                ax_top.set_ylabel("Temperatura (°C)", color='#222222', fontsize=8)
                ax_top.tick_params(labelsize=8)

            if mostrar_velocidad or mostrar_potencia:
                ax_sec = ax_top.twinx() if mostrar_temperaturas else ax_top
                if mostrar_velocidad:
                    l3, = ax_sec.plot(times, self.res_sim['fan_speed'], color='#27AE60', label='Velocidad Ventilador (%)', linewidth=1.2)
                    lines.append(l3)
                    ax_sec.set_ylabel("Velocidad (%)", color='#27AE60', fontsize=8)
                    ax_sec.set_ylim(-5, 105)
                if mostrar_potencia:
                    l_pow, = ax_sec.plot(times, self.res_sim['power_kw'], color='#2980B9', linestyle='--', label='Potencia Eléctrica (kW)', linewidth=1.2)
                    lines.append(l_pow)
                    if not mostrar_velocidad:
                        ax_sec.set_ylabel("Potencia (kW)", color='#2980B9', fontsize=8)

                ax_sec.tick_params(labelsize=8)

            labels = [l.get_label() for l in lines]
            ax_top.legend(lines, labels, loc='upper right', fontsize=8, framealpha=0.85)

        if ax_bot is not None:
            lines_bot = []
            if mostrar_carga:
                l_q, = ax_bot.plot(times, self.res_sim['q_mwt'], color='#8E44AD', label='Carga Térmica (MWt)', linewidth=1.4)
                lines_bot.append(l_q)
                ax_bot.set_ylabel("Carga (MWt)", color='#8E44AD', fontsize=8)
                ax_bot.tick_params(labelsize=8)

            if mostrar_evap:
                ax_evap = ax_bot.twinx() if mostrar_carga else ax_bot
                l_ev, = ax_evap.plot(times, self.res_sim['evap'], color='#E67E22', linestyle='-.', label='Evaporación (m³/h)', linewidth=1.2)
                lines_bot.append(l_ev)
                ax_evap.set_ylabel("Evaporación (m³/h)", color='#E67E22', fontsize=8)
                ax_evap.tick_params(labelsize=8)

            labels_bot = [l.get_label() for l in lines_bot]
            ax_bot.legend(lines_bot, labels_bot, loc='upper right', fontsize=8, framealpha=0.85)
            ax_bot.set_xlabel("Fecha / Hora", fontsize=8)

        if ax_top is not None and ax_bot is None:
            ax_top.set_xlabel("Fecha / Hora", fontsize=8)

        self.fig.autofmt_xdate()
        self.fig.tight_layout()
        self.canvas.draw()

# ==========================================
# 8. CANVA DE MATPLOTLIB 2D
# ==========================================
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def graficar_matriz(self, datos_res, capa_seleccionada="Temperatura del Agua (Tw)"):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        if capa_seleccionada == "Temperatura del Agua (Tw)":
            Matriz_plot = datos_res['Matriz_T_w']
            cmap_use = 'coolwarm'
            label_cbar = 'Temperatura del Agua (°C)'
        elif capa_seleccionada == "Humedad Absoluta del Aire (wa)":
            Matriz_plot = datos_res['Matriz_w_a']
            cmap_use = 'Blues'
            label_cbar = 'Humedad Absoluta (g vapor / kg aire)'
        else:
            Matriz_plot = datos_res['Matriz_T_a']
            cmap_use = 'YlOrRd'
            label_cbar = 'Temp. Bulbo Seco Aire (°C)'

        Ny, Nx = Matriz_plot.shape

        im = self.ax.imshow(Matriz_plot, cmap=cmap_use, origin='upper', aspect='auto')
        colorbar = self.fig.colorbar(im, ax=self.ax, pad=0.03)
        colorbar.set_label(label_cbar, fontsize=9, color='#333333', labelpad=8)
        colorbar.ax.tick_params(labelsize=8)

        if datos_res['hay_niebla']:
            Matriz_niebla = datos_res['Matriz_niebla']
            capa_niebla = np.zeros((Ny, Nx, 4))
            capa_niebla[Matriz_niebla] = [0.2, 0.2, 0.2, 0.35] 
            
            self.ax.imshow(capa_niebla, origin='upper', aspect='auto')
            self.ax.contour(Matriz_niebla, levels=[0.5], colors=['#222222'], linestyles=['--'], linewidths=[1.5])
            
            self.ax.plot([], [], color='#666666', alpha=0.5, linewidth=6, label='Zona de Niebla')
            self.ax.plot([], [], color='#222222', linestyle='--', linewidth=1.5, label='Frente de Condensación')
            self.ax.legend(loc='lower left', fontsize=8, framealpha=0.85)

        motor_str = "CoolProp Engine" if HAS_COOLPROP else "ASHRAE Standard Engine"
        N = datos_res['num_celdas']
        titulo_texto = (
            f"Mapa 2D ({N}x{N}): {capa_seleccionada}   (NTU = {datos_res['NTU']:.4f})  [{motor_str}]\n"
            f"Entrada Techo: {datos_res['T_w_in']:.1f} °C   |   Piscina Mezclada: {datos_res['T_salida']:.2f} °C"
        )
        self.ax.set_title(titulo_texto, fontsize=10, fontweight='bold', color='#222222', pad=12)

        self.ax.set_xlabel('Entrada Aire Ambiente   →   Dirección del Flujo de Aire   →   Salida', fontsize=9, color='#444444', labelpad=8)
        self.ax.set_ylabel('← Caída del Agua (Techo a Piscina) →', fontsize=9, color='#444444', labelpad=8)
        self.ax.tick_params(labelsize=8)

        self.fig.tight_layout()
        self.draw()

# ==========================================
# 9. VENTANA PRINCIPAL DE PyQt5 CON OPCIÓN DE PLUMA EN MENÚ
# ==========================================
class TorreCoolingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemelo Digital 2D - Torre de Enfriamiento (Poppe)")
        self.setGeometry(100, 100, 1180, 750)
        self.ultimo_resultado = None
        self.ultimo_resultado_dinamico = None

        self.init_menu()
        self.init_ui()

    def init_menu(self):
        menubar = self.menuBar()
        menu_simulacion = menubar.addMenu("Simulación")

        self.action_sim_dinamica = QAction("Simulación Dinámica EPW (Control PID)...", self)
        self.action_sim_dinamica.setStatusTip("Ejecutar simulación dinámica anual con archivo EPW y control PID del ventilador")
        self.action_sim_dinamica.setEnabled(False)
        self.action_sim_dinamica.triggered.connect(self.abrir_simulacion_dinamica)
        menu_simulacion.addAction(self.action_sim_dinamica)

        # NUEVA OPCIÓN: PERFIL DE PLUMA BRIGGS 2D
        self.action_ver_pluma = QAction("Ver Perfil de Pluma Atmosférica (Briggs 2D)...", self)
        self.action_ver_pluma.setStatusTip("Visualizar elevación y dispersión de la pluma de humedad según modelo Briggs 2D")
        self.action_ver_pluma.setEnabled(False)
        self.action_ver_pluma.triggered.connect(self.abrir_ventana_pluma)
        menu_simulacion.addAction(self.action_ver_pluma)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # SECCIÓN IZQUIERDA
        panel_izquierdo = QWidget()
        layout_izq = QVBoxLayout(panel_izquierdo)
        layout_izq.setContentsMargins(10, 10, 10, 10)
        layout_izq.setSpacing(10)

        estilo_gb = "QGroupBox { font-size: 11px; font-weight: bold; color: #2C3E50; border: 1px solid #DCDCDC; border-radius: 4px; margin-top: 8px; padding-top: 10px; }"

        gb_agua = QGroupBox("Parámetros del Agua (Punto 1)")
        gb_agua.setStyleSheet(estilo_gb)
        grid_agua = QGridLayout()
        grid_agua.setVerticalSpacing(4)

        self.txt_Tw_in = self.crear_input("31.7", "°C", grid_agua, 0, "Temp. Entrada Agua (T_w1):")
        self.txt_Tw_out = self.crear_input("20.6", "°C", grid_agua, 1, "Temp. Salida Deseada (T_w2):")
        self.txt_caudal_w = self.crear_input("1174.0", "m³/h", grid_agua, 2, "Caudal Volumétrico Agua:")
        gb_agua.setLayout(grid_agua)
        layout_izq.addWidget(gb_agua)

        gb_aire = QGroupBox("Condiciones Ambientales y Malla")
        gb_aire.setStyleSheet(estilo_gb)
        grid_aire = QGridLayout()
        grid_aire.setVerticalSpacing(4)

        self.txt_Tdb_in = self.crear_input("30.0", "°C", grid_aire, 0, "Temp. Bulbo Seco (T_db):")
        self.txt_Twb_in = self.crear_input("17.8", "°C", grid_aire, 1, "Temp. Bulbo Húmedo (T_wb):")
        self.txt_caudal_a = self.crear_input("474.1", "m³/s", grid_aire, 2, "Caudal Aire Ventilador:")
        self.txt_densidad_a = self.crear_input("1.177", "kg/m³", grid_aire, 3, "Densidad del Aire:", precision=3)
        self.txt_altitud = self.crear_input("0.0", "m", grid_aire, 4, "Altitud del Sitio:")
        self.txt_num_celdas = self.crear_input_entero("15", "celdas", grid_aire, 5, "Resolución Malla (NxN):")
        gb_aire.setLayout(grid_aire)
        layout_izq.addWidget(gb_aire)

        # BOTONES: CALIBRAR 1 PUNTO Y AJUSTE 2 PUNTOS
        layout_botones = QHBoxLayout()
        self.btn_calcular = QPushButton("Calibrar NTU")
        self.btn_calcular.setFont(QFont("Segoe UI", 9))
        self.btn_calcular.setCursor(Qt.PointingHandCursor)
        self.btn_calcular.setStyleSheet("QPushButton { background-color: #34495E; color: #FFFFFF; border: none; padding: 6px 8px; border-radius: 3px; } QPushButton:hover { background-color: #2C3E50; }")
        self.btn_calcular.clicked.connect(self.ejecutar_calibracion_1p)

        self.btn_dos_puntos = QPushButton("Ajuste 2 Puntos")
        self.btn_dos_puntos.setFont(QFont("Segoe UI", 9))
        self.btn_dos_puntos.setCursor(Qt.PointingHandCursor)
        self.btn_dos_puntos.setStyleSheet("QPushButton { background-color: #27AE60; color: #FFFFFF; border: none; padding: 6px 8px; border-radius: 3px; } QPushButton:hover { background-color: #219653; }")
        self.btn_dos_puntos.clicked.connect(self.abrir_dialogo_2puntos)

        layout_botones.addWidget(self.btn_calcular)
        layout_botones.addWidget(self.btn_dos_puntos)
        layout_izq.addLayout(layout_botones)

        # Grupo Resultados
        gb_res = QGroupBox("Resultados de Diagnóstico Térmico")
        gb_res.setStyleSheet(estilo_gb)
        layout_res = QVBoxLayout()
        layout_res.setSpacing(3)

        self.lbl_ntu_res = QLabel("NTU Calibrado:  --")
        self.lbl_merkel_res = QLabel("Constantes Relleno:  --")
        self.lbl_q_res = QLabel("Carga Térmica:  --")
        self.lbl_range_res = QLabel("Range (ΔTw):  --")
        self.lbl_approach_res = QLabel("Approach:  --")
        self.lbl_lg_res = QLabel("Relación Masa (L/G):  --")
        self.lbl_evap_res = QLabel("Evaporación:  --")
        self.lbl_niebla_res = QLabel("Estado Pluma/Niebla:  --")

        for lbl in [self.lbl_ntu_res, self.lbl_merkel_res, self.lbl_q_res, self.lbl_range_res, 
                    self.lbl_approach_res, self.lbl_lg_res, self.lbl_evap_res, self.lbl_niebla_res]:
            lbl.setFont(QFont("Segoe UI", 9))
            lbl.setStyleSheet("color: #1A252F; padding: 1px 0px;")

        layout_res.addWidget(self.lbl_ntu_res)
        layout_res.addWidget(self.lbl_merkel_res)
        layout_res.addWidget(self.lbl_q_res)
        layout_res.addWidget(self.lbl_range_res)
        layout_res.addWidget(self.lbl_approach_res)
        layout_res.addWidget(self.lbl_lg_res)
        layout_res.addWidget(self.lbl_evap_res)
        layout_res.addWidget(self.lbl_niebla_res)

        gb_res.setLayout(layout_res)
        layout_izq.addWidget(gb_res)
        layout_izq.addStretch()

        # SECCIÓN DERECHA
        panel_derecho = QWidget()
        layout_der = QVBoxLayout(panel_derecho)
        layout_der.setContentsMargins(5, 5, 5, 5)

        top_der_layout = QHBoxLayout()
        lbl_combo = QLabel("Variable a Visualizar en la Matriz 2D:")
        lbl_combo.setFont(QFont("Segoe UI", 9))
        
        self.combo_capa = QComboBox()
        self.combo_capa.setFont(QFont("Segoe UI", 9))
        self.combo_capa.addItems(["Temperatura del Agua (Tw)", "Humedad Absoluta del Aire (wa)", "Temperatura del Aire (Ta)"])
        self.combo_capa.currentTextChanged.connect(self.cambiar_capa_grafico)

        top_der_layout.addWidget(lbl_combo)
        top_der_layout.addWidget(self.combo_capa)
        top_der_layout.addStretch()

        layout_der.addLayout(top_der_layout)

        self.canvas = MplCanvas(self, width=6, height=6, dpi=100)
        layout_der.addWidget(self.canvas)

        splitter.addWidget(panel_izquierdo)
        splitter.addWidget(panel_derecho)
        splitter.setSizes([310, 850])

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("font-size: 11px; color: #555555; background-color: #FAFAFA;")
        self.setStatusBar(self.status_bar)
        
        engine_msg = "CoolProp (NIST)" if HAS_COOLPROP else "ASHRAE Standard"
        self.status_bar.showMessage(f"Primero presione 'Calibrar NTU' para habilitar la simulación dinámica EPW. [{engine_msg}]")

    def crear_input(self, valor_defecto, unidad, grid_layout, fila, label_texto, precision=1):
        lbl = QLabel(label_texto)
        lbl.setFont(QFont("Segoe UI", 9))
        txt = QLineEdit(valor_defecto)
        txt.setFont(QFont("Segoe UI", 9))
        validator = QDoubleValidator()
        validator.setLocale(QLocale("C"))
        txt.setValidator(validator)

        def formatear_decimales():
            try:
                val = self.parse_float(txt.text())
                txt.setText(f"{val:.{precision}f}")
            except ValueError:
                pass

        txt.editingFinished.connect(formatear_decimales)
        lbl_unit = QLabel(unidad)
        lbl_unit.setFont(QFont("Segoe UI", 8))
        grid_layout.addWidget(lbl, fila, 0)
        grid_layout.addWidget(txt, fila, 1)
        grid_layout.addWidget(lbl_unit, fila, 2)
        return txt

    def crear_input_entero(self, valor_defecto, unidad, grid_layout, fila, label_texto):
        lbl = QLabel(label_texto)
        lbl.setFont(QFont("Segoe UI", 9))
        txt = QLineEdit(valor_defecto)
        txt.setFont(QFont("Segoe UI", 9))
        txt.setValidator(QIntValidator(5, 100))
        lbl_unit = QLabel(unidad)
        lbl_unit.setFont(QFont("Segoe UI", 8))
        grid_layout.addWidget(lbl, fila, 0)
        grid_layout.addWidget(txt, fila, 1)
        grid_layout.addWidget(lbl_unit, fila, 2)
        return txt

    def parse_float(self, text):
        return float(text.replace(',', '.'))

    def obtener_datos_pantalla_p1(self):
        return {
            'T_w_in': self.parse_float(self.txt_Tw_in.text()),
            'T_w_out_target': self.parse_float(self.txt_Tw_out.text()),
            'caudal_w': self.parse_float(self.txt_caudal_w.text()),
            'T_db_in': self.parse_float(self.txt_Tdb_in.text()),
            'T_wb_in': self.parse_float(self.txt_Twb_in.text()),
            'caudal_a': self.parse_float(self.txt_caudal_a.text()),
            'densidad_a': self.parse_float(self.txt_densidad_a.text()),
            'altitud': self.parse_float(self.txt_altitud.text()),
            'num_celdas': int(self.txt_num_celdas.text())
        }

    def lanzar_worker(self, datos_p1, datos_p2=None):
        self.progress_calib = QProgressDialog("Calibrando torre...", "Cancelar", 0, 100, self)
        self.progress_calib.setWindowTitle("Calibración Térmica en Proceso")
        self.progress_calib.setWindowModality(Qt.WindowModal)

        self.btn_calcular.setEnabled(False)
        self.btn_dos_puntos.setEnabled(False)

        self.worker_cal = CalibracionWorker(datos_p1, datos_p2)
        self.worker_cal.progreso_signal.connect(self.actualizar_progreso_calib)
        self.worker_cal.exito_signal.connect(self.procesar_exito_calib)
        self.worker_cal.error_signal.connect(self.procesar_error_calib)
        self.worker_cal.cancelado_signal.connect(self.procesar_cancelacion_calib)
        
        self.progress_calib.canceled.connect(self.worker_cal.cancelar)
        self.worker_cal.start()

    def ejecutar_calibracion_1p(self):
        try:
            d1 = self.obtener_datos_pantalla_p1()
            self.lanzar_worker(d1, datos_p2=None)
        except ValueError:
            QMessageBox.warning(self, "Entrada Inválida", "Verifique que todos los campos contengan números válidos.")

    def abrir_dialogo_2puntos(self):
        try:
            d1 = self.obtener_datos_pantalla_p1()
            dlg = DialogoSegundoPunto(self, datos_p1=d1)
            if dlg.exec_() == QDialog.Accepted:
                d2 = dlg.obtener_datos_p2()
                self.lanzar_worker(d1, d2)
        except ValueError:
            QMessageBox.warning(self, "Entrada Inválida", "Verifique que todos los campos del Punto 1 sean válidos.")

    def actualizar_progreso_calib(self, msg, pct):
        if hasattr(self, 'progress_calib') and self.progress_calib:
            self.progress_calib.setLabelText(msg)
            self.progress_calib.setValue(pct)
        self.status_bar.showMessage(f"{msg} ({pct}%)")

    def procesar_exito_calib(self, res):
        if hasattr(self, 'progress_calib') and self.progress_calib:
            self.progress_calib.close()

        self.btn_calcular.setEnabled(True)
        self.btn_dos_puntos.setEnabled(True)
        self.ultimo_resultado = res
        
        self.action_sim_dinamica.setEnabled(True)
        
        self.lbl_ntu_res.setText(f"NTU Calibrado (P1): <b style='font-size:10.5pt; color:#2980B9;'>{res['NTU']:.4f}</b>")

        if res['es_dual']:
            c = res['c_coef']
            m = res['m_exp']
            self.lbl_merkel_res.setText(f"Modelo Merkel: <b style='color:#8E44AD;'>c = {c:.3f}, m = {m:.3f}</b>")
            self.status_bar.showMessage(f"Ajuste de 2 Puntos Exitoso! c={c:.3f}, m={m:.3f}. Simulación activada.")
        else:
            self.lbl_merkel_res.setText("Modelo Merkel: <b>(Ajuste 1 Punto)</b>")
            self.status_bar.showMessage(f"Calibración exitosa. NTU = {res['NTU']:.4f}. Simulación activada.")

        self.lbl_q_res.setText(f"Carga Térmica: <b>{res['Q_MWt']:.2f} MWt</b> ({res['Q_TR']:.0f} TR)")
        self.lbl_range_res.setText(f"Range (ΔTw): <b>{res['range_w']:.2f} °C</b>")
        self.lbl_approach_res.setText(f"Approach: <b>{res['approach_w']:.2f} °C</b>")
        self.lbl_lg_res.setText(f"Relación Masa (L/G): <b>{res['L_G_ratio']:.3f}</b>")
        self.lbl_evap_res.setText(f"Evaporación: <b>{res['evaporacion_m3h']:.2f} m³/h</b> ({res['pct_evap']:.2f}%)")

        if res['hay_niebla']:
            self.lbl_niebla_res.setText("Estado Pluma/Niebla: <b style='color:#C0392B;'>DETECTADA (Supersaturación)</b>")
        else:
            self.lbl_niebla_res.setText("Estado Pluma/Niebla: <b style='color:#27AE60;'>Sin Niebla (Aire no saturado)</b>")

        self.canvas.graficar_matriz(res, self.combo_capa.currentText())

    def procesar_cancelacion_calib(self):
        if hasattr(self, 'progress_calib') and self.progress_calib:
            self.progress_calib.close()

        self.btn_calcular.setEnabled(True)
        self.btn_dos_puntos.setEnabled(True)
        self.status_bar.showMessage("Calibración cancelada por el usuario.")

    def procesar_error_calib(self, err):
        if hasattr(self, 'progress_calib') and self.progress_calib:
            self.progress_calib.close()

        self.btn_calcular.setEnabled(True)
        self.btn_dos_puntos.setEnabled(True)
        QMessageBox.critical(self, "Error de Calibración", f"No se pudo calibrar:\n{err}")

    def cambiar_capa_grafico(self, nueva_capa):
        if self.ultimo_resultado is not None and 'Matriz_T_w' in self.ultimo_resultado:
            self.canvas.graficar_matriz(self.ultimo_resultado, nueva_capa)

    def abrir_simulacion_dinamica(self):
        if self.ultimo_resultado is None or 'NTU' not in self.ultimo_resultado:
            QMessageBox.warning(self, "Calibración Requerida", "Debe calibrar la torre en la pantalla principal antes de iniciar la simulación dinámica.")
            return

        d_torre = self.obtener_datos_pantalla_p1()
        d_torre['NTU'] = self.ultimo_resultado['NTU']

        dlg = VentanaSimulacionDinamica(self, datos_torre=d_torre)
        dlg.exec_()

    def actualizar_resultado_dinamico(self, res_din):
        self.ultimo_resultado_dinamico = res_din
        self.action_ver_pluma.setEnabled(True)

    def abrir_ventana_pluma(self):
        if self.ultimo_resultado_dinamico is None:
            QMessageBox.warning(self, "Simulación Requerida", "Debe ejecutar una simulación dinámica antes de visualizar el perfil de pluma.")
            return

        dlg = DialogoPerfilPluma(self, datos_sim=self.ultimo_resultado_dinamico)
        dlg.exec_()

# ==========================================
# 10. PUNTO DE ENTRADA
# ==========================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = TorreCoolingApp()
    window.show()
    sys.exit(app.exec_())