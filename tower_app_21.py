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
    QDialogButtonBox, QFileDialog, QCheckBox, QDateEdit, QMenuBar, QAction,
    QActionGroup, QRadioButton, QButtonGroup, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QLocale, QDate, QSettings
from PyQt5.QtGui import QFont, QDoubleValidator, QIntValidator, QIcon

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
    """Lee un archivo EPW de forma robusta.

    Busca filas de datos detectando si la primera cuatro columnas son año/mes/día/hora
    y acepta diferentes longitudes de encabezado. Devuelve una lista de diccionarios
    con las claves: 'dt','tdb','twb','rh','patm','u_viento'.
    """
    datos_clima = []

    def _safe_int(s):
        try:
            return int(str(s).strip())
        except Exception:
            return None

    def _safe_float(s, default=None):
        try:
            return float(str(s).strip().replace(',', '.'))
        except Exception:
            return default

    with open(path_epw, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 4:
                continue

            # limpiar posibles caracteres BOM y espacios
            try:
                row0 = row[0].lstrip('\ufeff').strip()
            except Exception:
                row0 = str(row[0]).strip()

            anio = _safe_int(row0)
            mes = _safe_int(row[1])
            dia = _safe_int(row[2])
            hora = _safe_int(row[3])

            # aceptar únicamente filas que parezcan datos (rangos plausibles)
            if anio is None or mes is None or dia is None or hora is None:
                continue
            if not (1900 <= anio <= 2100 and 1 <= mes <= 12 and 1 <= dia <= 31 and 1 <= hora <= 24):
                continue

            # convertir hora EPW (1..24) a 0..23
            hora_idx = max(0, min(23, hora - 1))

            try:
                dt = datetime(anio, mes, dia, hora_idx)
            except Exception:
                continue

            # columnas típicas de EPW (si faltan, se usan valores por defecto razonables)
            tdb = _safe_float(row[6], None) if len(row) > 6 else None
            # intentar columna de punto de rocío/humedad si está disponible
            tdew = _safe_float(row[7], None) if len(row) > 7 else None
            rh_raw = _safe_float(row[8], None) if len(row) > 8 else None
            patm = _safe_float(row[9], None) if len(row) > 9 else None

            if tdb is None or rh_raw is None:
                # si faltan datos críticos, saltar la fila
                continue

            # RH en EPW viene en % (0..100)
            rh = rh_raw / 100.0 if rh_raw is not None else 0.0

            u_viento = _safe_float(row[21], None) if len(row) > 21 else None
            if u_viento is None:
                # intentar columnas alternativas comunes (ej. 20)
                if len(row) > 20:
                    u_viento = _safe_float(row[20], 3.5)
                else:
                    u_viento = 3.5

            # aproximación rápida de Twb si no hay columna directa
            try:
                twb = float(tdb) * np.arctan(0.151977 * (rh * 100.0 + 8.313659) ** 0.5) + np.arctan(float(tdb) + rh * 100.0) - np.arctan(rh * 100.0 - 1.676331) + 0.00391838 * (rh * 100.0) ** 1.5 * np.arctan(0.023101 * rh * 100.0) - 4.686035
            except Exception:
                twb = float(tdb)

            datos_clima.append({
                'dt': dt,
                'tdb': float(tdb),
                'twb': float(twb),
                'rh': float(rh),
                'patm': float(patm) if patm is not None else 101325.0,
                'u_viento': float(u_viento)
            })

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

            # Si la configuración solicita normalizar a un único año, aplicar aquí
            if self.cfg.get('epw_normalize'):
                target_year = int(self.cfg.get('epw_normalize_year') or 2000)
                clima_norm = []
                for r in clima:
                    dt0 = r['dt']
                    m = dt0.month
                    d = dt0.day
                    h = dt0.hour
                    try:
                        ndt = datetime(target_year, m, d, h)
                    except ValueError:
                        # fallback: try common valid day choices (handle Feb 29)
                        ndt = None
                        for dd in (28, 27, 26, 25):
                            try:
                                ndt = datetime(target_year, m, dd, h)
                                break
                            except Exception:
                                ndt = None
                        if ndt is None:
                            ndt = datetime(target_year, m, max(1, min(28, d)), h)
                    nr = dict(r)
                    nr['dt'] = ndt
                    clima_norm.append(nr)
                clima = clima_norm

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
                'u_viento_vec': uviento_vec,  # <--- ¡AQUÍ ESTÁ EL CAMBIO CLAVE!
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

def parse_float_local(text):
    return float(text.replace(',', '.'))

def conectar_formato_precision(txt_widget, precision=1):
    """Reformatea el contenido de un QLineEdit numérico a una precisión fija
    (mínimo 1 decimal) cada vez que el campo pierde el foco."""
    def _formatear():
        try:
            val = parse_float_local(txt_widget.text())
            txt_widget.setText(f"{val:.{max(1, precision)}f}")
        except ValueError:
            pass
    txt_widget.editingFinished.connect(_formatear)
    return _formatear

def traducir(idioma, key, **kwargs):
    texto = TRANSLATIONS[idioma][key]
    return texto.format(**kwargs) if kwargs else texto

# ==========================================
# 6. DIÁLOGO EMERGENTE DE PERFIL DE PLUMA BRIGGS 2D
# ==========================================
# ==========================================
# DIÁLOGO EMERGENTE PARA EL 2º PUNTO
# ==========================================
class DialogoSegundoPunto(QDialog):
    def __init__(self, parent=None, datos_p1=None, idioma='es'):
        super().__init__(parent)
        self.idioma = idioma
        self.setWindowTitle(self.tr_txt('dlg2p_title'))
        self.setFixedSize(380, 420)
        self.datos_p1 = datos_p1
        self.init_ui()

    def tr_txt(self, key, **kwargs):
        return traducir(self.idioma, key, **kwargs)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        lbl_info = QLabel(self.tr_txt('dlg2p_info'))
        lbl_info.setFont(QFont("Segoe UI", 9))
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color: #2C3E50;")
        layout.addWidget(lbl_info)

        gb = QGroupBox(self.tr_txt('dlg2p_gb'))
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

        self.txt_Tw_in = self.crear_field(self.tr_txt('lbl_Tw_in'), Tw1_def, "°C", grid, 0)
        self.txt_Tw_out = self.crear_field(self.tr_txt('dlg2p_Tw_out'), Tw2_def, "°C", grid, 1)
        self.txt_caudal_w = self.crear_field(self.tr_txt('lbl_caudal_w'), Cw_def, "m³/h", grid, 2)
        self.txt_Tdb_in = self.crear_field(self.tr_txt('lbl_Tdb_in'), Tdb_def, "°C", grid, 3)
        self.txt_Twb_in = self.crear_field(self.tr_txt('lbl_Twb_in'), Twb_def, "°C", grid, 4)
        self.txt_caudal_a = self.crear_field(self.tr_txt('lbl_caudal_a'), Ca_def, "m³/s", grid, 5)

        layout.addWidget(gb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.button(QDialogButtonBox.Ok).setText(self.tr_txt('dlg2p_btn_ok'))
        buttons.button(QDialogButtonBox.Ok).setStyleSheet("background-color: #34495E; color: white; padding: 5px 10px;")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def crear_field(self, label, default, unit, grid, row, precision=1):
        lbl = QLabel(label)
        lbl.setFont(QFont("Segoe UI", 9))
        txt = QLineEdit(default)
        txt.setFont(QFont("Segoe UI", 9))
        val = QDoubleValidator()
        val.setLocale(QLocale("C"))
        txt.setValidator(val)
        conectar_formato_precision(txt, precision)
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
    
# ==========================================
# 6. DIÁLOGO EMERGENTE CON SELECTOR HORA A HORA DE PLUMA (BRIGGS 2D)
# ==========================================
from PyQt5.QtWidgets import QSlider

# ==========================================
# 6. DIÁLOGO EMERGENTE DE PLUMA CON UI GEOMÉTRICA COMPLETA
# ==========================================
class DialogoPerfilPluma(QDialog):
    def __init__(self, parent=None, datos_sim=None, idioma='es'):
        super().__init__(parent)
        self.idioma = idioma
        self.setWindowTitle(self.tr_txt('pluma_title'))
        self.resize(1000, 720)
        self.datos_sim = datos_sim
        self.init_ui()

    def tr_txt(self, key, **kwargs):
        return traducir(self.idioma, key, **kwargs)

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Encabezado
        lbl_info = QLabel(self.tr_txt('pluma_info'))
        lbl_info.setFont(QFont("Segoe UI", 10, QFont.Bold))
        lbl_info.setStyleSheet("color: #2C3E50;")
        layout.addWidget(lbl_info)

        # --- AQUI SE AGREGA LA SECCIÓN VISUAL DE GEOMETRÍA DE LA TORRE ---
        gb_geom = QGroupBox(self.tr_txt('pluma_gb_geom'))
        gb_geom.setStyleSheet("""
            QGroupBox {
                font-size: 11px; font-weight: bold; color: #2C3E50;
                border: 1px solid #DCDCDC; border-radius: 4px; margin-top: 4px; padding-top: 8px;
            }
        """)
        layout_geom = QHBoxLayout(gb_geom)

        lbl_d = QLabel(self.tr_txt('pluma_lbl_diametro'))
        lbl_d.setFont(QFont("Segoe UI", 8))
        self.txt_diametro_boca = QLineEdit("3.60")
        self.txt_diametro_boca.setFont(QFont("Segoe UI", 8))
        self.txt_diametro_boca.setFixedWidth(60)
        val_d = QDoubleValidator(0.5, 20.0, 2)
        val_d.setLocale(QLocale("C"))
        self.txt_diametro_boca.setValidator(val_d)
        conectar_formato_precision(self.txt_diametro_boca, 2)
        lbl_u_d = QLabel("m")
        lbl_u_d.setFont(QFont("Segoe UI", 8))

        lbl_h = QLabel(self.tr_txt('pluma_lbl_altura'))
        lbl_h.setFont(QFont("Segoe UI", 8))
        self.txt_altura_torre = QLineEdit("10.00")
        self.txt_altura_torre.setFont(QFont("Segoe UI", 8))
        self.txt_altura_torre.setFixedWidth(60)
        val_h = QDoubleValidator(1.0, 100.0, 2)
        val_h.setLocale(QLocale("C"))
        self.txt_altura_torre.setValidator(val_h)
        conectar_formato_precision(self.txt_altura_torre, 2)
        lbl_u_h = QLabel("m")
        lbl_u_h.setFont(QFont("Segoe UI", 8))

        # Reconectar cambios para actualizar la gráfica al editar geometría
        self.txt_diametro_boca.editingFinished.connect(self.recalcular_actual)
        self.txt_altura_torre.editingFinished.connect(self.recalcular_actual)

        layout_geom.addWidget(lbl_d)
        layout_geom.addWidget(self.txt_diametro_boca)
        layout_geom.addWidget(lbl_u_d)
        layout_geom.addSpacing(20)
        layout_geom.addWidget(lbl_h)
        layout_geom.addWidget(self.txt_altura_torre)
        layout_geom.addWidget(lbl_u_h)
        layout_geom.addStretch()

        layout.addWidget(gb_geom)

        # Canvas Matplotlib
        self.fig = Figure(figsize=(8, 4.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # KPIs del Instante
        self.lbl_kpis_pluma = QLabel(self.tr_txt('pluma_kpis_default'))
        self.lbl_kpis_pluma.setFont(QFont("Segoe UI", 9))
        self.lbl_kpis_pluma.setStyleSheet("background-color: #F4F6F7; padding: 8px; border-radius: 4px; color: #1A252F;")
        layout.addWidget(self.lbl_kpis_pluma)

        # PANEL DE CONTROL TEMPORAL
        gb_control = QGroupBox(self.tr_txt('pluma_gb_control'))
        layout_ctrl = QHBoxLayout(gb_control)

        self.lbl_fecha_actual = QLabel(self.tr_txt('pluma_fecha_default'))
        self.lbl_fecha_actual.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.lbl_fecha_actual.setStyleSheet("color: #2980B9; min-width: 180px;")

        total_pasos = len(self.datos_sim['times']) if self.datos_sim else 1
        
        self.slider_tiempo = QSlider(Qt.Horizontal)
        self.slider_tiempo.setMinimum(0)
        self.slider_tiempo.setMaximum(total_pasos - 1)
        self.slider_tiempo.setValue(0)
        self.slider_tiempo.valueChanged.connect(self.actualizar_instante)

        self.btn_worst_case = QPushButton(self.tr_txt('pluma_btn_worst'))
        self.btn_worst_case.setFont(QFont("Segoe UI", 8, QFont.Bold))
        self.btn_worst_case.setStyleSheet("background-color: #E67E22; color: white; padding: 5px 10px; border-radius: 3px;")
        self.btn_worst_case.clicked.connect(self.ir_a_peor_caso)

        layout_ctrl.addWidget(self.lbl_fecha_actual)
        layout_ctrl.addWidget(self.slider_tiempo, stretch=1)
        layout_ctrl.addWidget(self.btn_worst_case)

        layout.addWidget(gb_control)

        # Cargar primer instante
        self.actualizar_instante(0)

    def recalcular_actual(self):
        self.actualizar_instante(self.slider_tiempo.value())

    def ir_a_peor_caso(self):
        if not self.datos_sim:
            return
        
        t_a_out_arr = np.array(self.datos_sim['t_a_out'])
        t_db_arr = np.array(self.datos_sim['t_db'])
        t_wb_arr = np.array(self.datos_sim['t_wb'])
        
        humedad_rel_aprox = np.maximum(0.1, (t_wb_arr + 20) / (t_db_arr + 20))
        idx_peor = int(np.argmax(humedad_rel_aprox))
        
        self.slider_tiempo.setValue(idx_peor)

    def actualizar_instante(self, idx):
        if not self.datos_sim or idx >= len(self.datos_sim['times']):
            return

        dt_instante = self.datos_sim['times'][idx]
        self.lbl_fecha_actual.setText(self.tr_txt('pluma_fecha_texto', fecha=dt_instante.strftime('%d/%m/%Y %H:%M')))

        T_a_out = float(self.datos_sim['t_a_out'][idx])
        T_db_amb = float(self.datos_sim['t_db'][idx])
        T_wb_amb = float(self.datos_sim['t_wb'][idx])
        vel_fan_pct = float(self.datos_sim['fan_speed'][idx])
        
        if 'u_viento_vec' in self.datos_sim and idx < len(self.datos_sim['u_viento_vec']):
            u_wind = max(0.5, float(self.datos_sim['u_viento_vec'][idx]))
        else:
            u_wind = max(0.5, float(self.datos_sim.get('viento_medio', 3.5)))

        caudal_a = float(self.datos_sim['caudal_a_m3s'])
        caudal_a_actual = max(0.1, caudal_a * (vel_fan_pct / 100.0))

        # LECTURA SEGURO DE GEOMETRÍA CON FALLBACK SI EL CAMPO QUEDA VACÍO
        try:
            H_torre = float(self.txt_altura_torre.text().replace(',', '.'))
        except ValueError:
            H_torre = 10.0

        try:
            D_boca = float(self.txt_diametro_boca.text().replace(',', '.'))
        except ValueError:
            D_boca = 3.6

        A_boca = (np.pi / 4.0) * (D_boca ** 2)
        w_salida_m_s = caudal_a_actual / A_boca if A_boca > 0 else 1.0

        g = 9.81
        T_kelvin_out = T_a_out + 273.15
        T_kelvin_amb = T_db_amb + 273.15
        
        F_b = g * w_salida_m_s * (D_boca**2 / 4.0) * max(0.0001, (T_kelvin_out - T_kelvin_amb) / T_kelvin_out)
        
        x_vec = np.linspace(0.1, 160.0, 300)
        z_centron = (H_torre + D_boca / 2.0) + (3.0 * F_b * (x_vec**2) / (2.0 * 0.6**2 * (u_wind**3)))**(1.0 / 3.0)
        # 2. Expansión progresiva que nace exacta en el borde de la boca (x = 0)
        sigma_z = 0.12 * (x_vec**0.88) + (D_boca / 2.0) * (1.0 - np.exp(-x_vec / 2.0))

        z_top = z_centron + sigma_z
        z_bot = np.maximum(H_torre, z_centron - sigma_z) # La cota inferior nunca cae por debajo del techo en x=0

        w_amb = humedad_saturacion(T_wb_amb)
        w_out = humedad_saturacion(T_a_out)

        frac_mezcla = np.exp(-x_vec / max(8.0, 12.0 * u_wind))
        w_pluma_vec = w_amb + (w_out - w_amb) * frac_mezcla
        T_pluma_vec = T_db_amb + (T_a_out - T_db_amb) * frac_mezcla
        
        w_sat_pluma = np.array([humedad_saturacion(t) for t in T_pluma_vec])
        es_visible = (w_pluma_vec >= w_sat_pluma * 0.985)

        x_vis = x_vec[es_visible]
        z_top_vis = z_top[es_visible]
        z_bot_vis = z_bot[es_visible]

        L_pluma_vis = float(x_vis[-1]) if len(x_vis) > 0 else 0.0
        H_max_vis = float(np.max(z_top_vis)) if len(z_top_vis) > 0 else H_torre

        # DIBUJAR EN MATPLOTLIB
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Centra la estructura entre X = -D_boca/2 y X = D_boca/2 para que coincida con x = 0
        ax.add_patch(matplotlib.patches.Rectangle((-D_boca / 2.0, 0), D_boca, H_torre, color='#34495E', alpha=0.85, label=self.tr_txt('pluma_torre_label')))
        ax.plot([-D_boca / 2.0, -D_boca / 2.0], [H_torre, H_torre + 1.0], color='#1A252F', linewidth=2.5)
        ax.plot([D_boca / 2.0, D_boca / 2.0], [H_torre, H_torre + 1.0], color='#1A252F', linewidth=2.5)
        if len(x_vis) > 0:
            ax.fill_between(x_vis, z_bot_vis, z_top_vis, color='#95A5A6', alpha=0.55, label=self.tr_txt('pluma_visible_label'))
            ax.plot(x_vis, z_centron[es_visible], color='#7F8C8D', linestyle='--', linewidth=1.5, label=self.tr_txt('pluma_eje_central'))

        ax.plot(x_vec[~es_visible], z_centron[~es_visible], color='#3498DB', linestyle=':', alpha=0.4, label=self.tr_txt('pluma_eje_dispersion'))

        ax.annotate('', xy=(20, H_torre + 12), xytext=(2, H_torre + 12),
                    arrowprops=dict(facecolor='#C0392B', edgecolor='#C0392B', arrowstyle='->', lw=2))
        ax.text(8, H_torre + 13.5, self.tr_txt('pluma_viento_inst', u=u_wind), color='#C0392B', fontsize=8, fontweight='bold')

        ax.set_xlim(-10, 150)
        ax.set_ylim(0, max(45, H_max_vis + 8))
        ax.set_xlabel(self.tr_txt('pluma_xlabel'), fontsize=9)
        ax.set_ylabel(self.tr_txt('pluma_ylabel'), fontsize=9)
        ax.set_title(self.tr_txt('pluma_titulo', fecha=dt_instante.strftime('%d/%m/%Y %H:%M'), v0=w_salida_m_s, tsal=T_a_out, tamb=T_db_amb), fontsize=9, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

        if u_wind > 5.5 and vel_fan_pct < 35.0:
            downwash_risk = f"<b style='color:#C0392B;'>{self.tr_txt('pluma_riesgo_critico')}</b>"
        elif u_wind > 4.0:
            downwash_risk = f"<b style='color:#E67E22;'>{self.tr_txt('pluma_riesgo_moderado')}</b>"
        else:
            downwash_risk = f"<b style='color:#27AE60;'>{self.tr_txt('pluma_riesgo_bajo')}</b>"

        self.lbl_kpis_pluma.setText(
            self.tr_txt('pluma_kpi_texto', v0=w_salida_m_s, l=L_pluma_vis, h=H_max_vis, riesgo=downwash_risk)
        )

# ==========================================
# 7. VENTANA EMERGENTE DE SIMULACIÓN DINÁMICA
# ==========================================
class DialogoEpwChoice(QDialog):
    def __init__(self, parent=None, years=None, idioma='es'):
        super().__init__(parent)
        self.idioma = idioma
        self.setWindowTitle(self.tr_txt('epw_multi_title'))
        self.setModal(True)
        self.years = sorted(years) if years else []
        self.choice = {'action': 'preserve', 'year': None}
        self._init_ui()

    def tr_txt(self, key, **kwargs):
        return traducir(self.idioma, key, **kwargs)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        lbl = QLabel(self.tr_txt('epw_multi_info'))
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        self.rb_preserve = QRadioButton(self.tr_txt('epw_multi_preserve'))
        self.rb_normalize = QRadioButton(self.tr_txt('epw_multi_normalize'))
        self.rb_preserve.setChecked(True)

        layout.addWidget(self.rb_preserve)
        h = QHBoxLayout()
        h.addWidget(self.rb_normalize)
        self.spin_year = QSpinBox()
        self.spin_year.setRange(1900, 2100)
        self.spin_year.setValue(2000)
        h.addWidget(self.spin_year)
        h.addStretch()
        layout.addLayout(h)

        if self.years:
            years_text = ', '.join(str(y) for y in self.years[:10])
            info = QLabel(self.tr_txt('epw_multi_years_present').format(years=years_text))
            info.setStyleSheet('color: #555555; font-size: 11px;')
            layout.addWidget(info)

        # Remember choice checkbox
        self.chk_remember = QCheckBox(self.tr_txt('epw_multi_remember'))
        layout.addWidget(self.chk_remember)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        if self.rb_normalize.isChecked():
            self.choice = {'action': 'normalize', 'year': int(self.spin_year.value())}
        else:
            self.choice = {'action': 'preserve', 'year': None}
        # persist if requested
        try:
            if self.chk_remember.isChecked():
                settings = QSettings('cooling_towers', 'tower_app')
                settings.setValue('epw_choice_action', self.choice['action'])
                settings.setValue('epw_choice_year', self.choice['year'] if self.choice['year'] is not None else '')
        except Exception:
            pass
        super().accept()

    def get_choice(self):
        return self.choice

class VentanaSimulacionDinamica(QDialog):
    def __init__(self, parent=None, datos_torre=None, idioma='es', estado_previo=None):
        super().__init__(parent)
        self.idioma = idioma
        self.setWindowTitle(self.tr_txt('sim_title'))
        
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint | Qt.WindowMaximizeButtonHint)
        self.resize(1340, 900)
        self.setMinimumSize(980, 720)
        
        self.datos_torre = datos_torre
        self.res_sim = None

        self.init_ui()

        if estado_previo is not None:
            self.restaurar_estado(estado_previo)

    def tr_txt(self, key, **kwargs):
        return traducir(self.idioma, key, **kwargs)

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        panel_cfg = QWidget()
        layout_cfg = QVBoxLayout(panel_cfg)
        layout_cfg.setContentsMargins(5, 5, 5, 5)

        estilo_gb = "QGroupBox { font-size: 11px; font-weight: bold; color: #2C3E50; border: 1px solid #DCDCDC; border-radius: 4px; margin-top: 6px; padding-top: 10px; }"

        gb_epw = QGroupBox(self.tr_txt('sim_gb_epw'))
        gb_epw.setStyleSheet(estilo_gb)
        grid_epw = QGridLayout(gb_epw)

        self.txt_epw_path = QLineEdit()
        self.txt_epw_path.setPlaceholderText(self.tr_txt('sim_epw_placeholder'))
        self.txt_epw_path.setFont(QFont("Segoe UI", 9))
        btn_epw = QPushButton(self.tr_txt('sim_btn_examinar'))
        btn_epw.setFont(QFont("Segoe UI", 9))
        btn_epw.clicked.connect(self.examinar_epw)

        grid_epw.addWidget(self.txt_epw_path, 0, 0)
        grid_epw.addWidget(btn_epw, 0, 1)
        layout_cfg.addWidget(gb_epw)

        gb_tiempo = QGroupBox(self.tr_txt('sim_gb_tiempo'))
        gb_tiempo.setStyleSheet(estilo_gb)
        grid_tiempo = QGridLayout(gb_tiempo)
        grid_tiempo.setColumnMinimumWidth(0, 155)
        grid_tiempo.setColumnMinimumWidth(1, 95)

        self.date_ini = QDateEdit(QDate(2024, 1, 1))
        self.date_ini.setDisplayFormat("dd/MM/yyyy")
        self.date_ini.setFont(QFont("Segoe UI", 9))
        self.date_fin = QDateEdit(QDate(2024, 1, 7))
        self.date_fin.setDisplayFormat("dd/MM/yyyy")
        self.date_fin.setFont(QFont("Segoe UI", 9))

        self.txt_dt_sim = QLineEdit("300.0")
        self.txt_vol_estanque = QLineEdit("15.0")
        self.txt_coc = QLineEdit("4.0")           
        self.txt_drift = QLineEdit("0.005")       

        for txt in [self.txt_dt_sim, self.txt_vol_estanque, self.txt_coc, self.txt_drift]:
            txt.setFont(QFont("Segoe UI", 9))
            txt.setValidator(QDoubleValidator())
            txt.setFixedWidth(90)

        conectar_formato_precision(self.txt_dt_sim, 1)
        conectar_formato_precision(self.txt_vol_estanque, 1)
        conectar_formato_precision(self.txt_coc, 1)
        conectar_formato_precision(self.txt_drift, 3)

        grid_tiempo.addWidget(QLabel(self.tr_txt('sim_lbl_fecha_ini')), 0, 0)
        grid_tiempo.addWidget(self.date_ini, 0, 1)
        grid_tiempo.addWidget(QLabel(self.tr_txt('sim_lbl_fecha_fin')), 1, 0)
        grid_tiempo.addWidget(self.date_fin, 1, 1)
        grid_tiempo.addWidget(QLabel(self.tr_txt('sim_lbl_dt')), 2, 0)
        grid_tiempo.addWidget(self.txt_dt_sim, 2, 1)
        grid_tiempo.addWidget(QLabel("seg"), 2, 2)
        grid_tiempo.addWidget(QLabel(self.tr_txt('sim_lbl_vol_estanque')), 3, 0)
        grid_tiempo.addWidget(self.txt_vol_estanque, 3, 1)
        grid_tiempo.addWidget(QLabel("m³"), 3, 2)
        grid_tiempo.addWidget(QLabel(self.tr_txt('sim_lbl_coc')), 4, 0)
        grid_tiempo.addWidget(self.txt_coc, 4, 1)
        grid_tiempo.addWidget(QLabel(self.tr_txt('sim_lbl_drift')), 5, 0)
        grid_tiempo.addWidget(self.txt_drift, 5, 1)
        grid_tiempo.addWidget(QLabel("%"), 5, 2)

        for i in range(grid_tiempo.count()):
            w = grid_tiempo.itemAt(i).widget()
            if isinstance(w, QLabel):
                w.setFont(QFont("Segoe UI", 9))

        layout_cfg.addWidget(gb_tiempo)

        gb_pid = QGroupBox(self.tr_txt('sim_gb_pid'))
        gb_pid.setStyleSheet(estilo_gb)
        grid_pid = QGridLayout(gb_pid)
        grid_pid.setColumnMinimumWidth(0, 155)
        grid_pid.setColumnMinimumWidth(1, 95)

        self.txt_t_set = QLineEdit("20.6")
        self.txt_kp = QLineEdit("4.0")
        self.txt_ti = QLineEdit("300.0")
        self.txt_td = QLineEdit("5.0")
        self.txt_speed_min = QLineEdit("20.0")

        for txt in [self.txt_t_set, self.txt_kp, self.txt_ti, self.txt_td, self.txt_speed_min]:
            txt.setFont(QFont("Segoe UI", 9))
            txt.setValidator(QDoubleValidator())
            txt.setFixedWidth(90)

        for txt in [self.txt_t_set, self.txt_kp, self.txt_ti, self.txt_td, self.txt_speed_min]:
            conectar_formato_precision(txt, 1)

        grid_pid.addWidget(QLabel(self.tr_txt('sim_lbl_setpoint')), 0, 0)
        grid_pid.addWidget(self.txt_t_set, 0, 1)
        grid_pid.addWidget(QLabel("°C"), 0, 2)

        grid_pid.addWidget(QLabel(self.tr_txt('sim_lbl_kp')), 1, 0)
        grid_pid.addWidget(self.txt_kp, 1, 1)

        grid_pid.addWidget(QLabel(self.tr_txt('sim_lbl_ti')), 2, 0)
        grid_pid.addWidget(self.txt_ti, 2, 1)
        grid_pid.addWidget(QLabel("s"), 2, 2)

        grid_pid.addWidget(QLabel(self.tr_txt('sim_lbl_td')), 3, 0)
        grid_pid.addWidget(self.txt_td, 3, 1)
        grid_pid.addWidget(QLabel("s"), 3, 2)

        grid_pid.addWidget(QLabel(self.tr_txt('sim_lbl_speed_min')), 4, 0)
        grid_pid.addWidget(self.txt_speed_min, 4, 1)
        grid_pid.addWidget(QLabel("%"), 4, 2)

        for i in range(grid_pid.count()):
            w = grid_pid.itemAt(i).widget()
            if isinstance(w, QLabel):
                w.setFont(QFont("Segoe UI", 9))

        layout_cfg.addWidget(gb_pid)

        gb_motor = QGroupBox(self.tr_txt('sim_gb_motor'))
        gb_motor.setStyleSheet(estilo_gb)
        grid_motor = QGridLayout(gb_motor)
        grid_motor.setColumnMinimumWidth(0, 155)
        grid_motor.setColumnMinimumWidth(1, 95)

        self.txt_p_motor = QLineEdit("150.0")
        self.txt_p_motor.setFont(QFont("Segoe UI", 9))
        self.txt_p_motor.setValidator(QDoubleValidator())
        self.txt_p_motor.setFixedWidth(90)
        conectar_formato_precision(self.txt_p_motor, 1)
        self.txt_eta_fan = QLineEdit("75.0")
        self.txt_eta_fan.setFont(QFont("Segoe UI", 9))
        self.txt_eta_fan.setValidator(QDoubleValidator())
        self.txt_eta_fan.setFixedWidth(90)
        conectar_formato_precision(self.txt_eta_fan, 1)

        grid_motor.addWidget(QLabel(self.tr_txt('sim_lbl_p_motor')), 0, 0)
        grid_motor.addWidget(self.txt_p_motor, 0, 1)
        grid_motor.addWidget(QLabel("kW"), 0, 2)

        grid_motor.addWidget(QLabel(self.tr_txt('sim_lbl_eta_fan')), 1, 0)
        grid_motor.addWidget(self.txt_eta_fan, 1, 1)
        grid_motor.addWidget(QLabel("%"), 1, 2)

        for i in range(grid_motor.count()):
            w = grid_motor.itemAt(i).widget()
            if isinstance(w, QLabel):
                w.setFont(QFont("Segoe UI", 9))

        layout_cfg.addWidget(gb_motor)

        self.btn_ejecutar = QPushButton(self.tr_txt('sim_btn_ejecutar'))
        self.btn_ejecutar.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.btn_ejecutar.setMinimumHeight(32)
        self.btn_ejecutar.setCursor(Qt.PointingHandCursor)
        self.btn_ejecutar.setStyleSheet("QPushButton { background-color: #27AE60; color: white; padding: 8px; border-radius: 3px; } QPushButton:hover { background-color: #219653; }")
        self.btn_ejecutar.clicked.connect(self.ejecutar_simulacion)

        self.btn_csv = QPushButton(self.tr_txt('sim_btn_csv'))
        self.btn_csv.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.btn_csv.setMinimumHeight(32)
        self.btn_csv.setCursor(Qt.PointingHandCursor)
        self.btn_csv.setStyleSheet("QPushButton { background-color: #2980B9; color: white; padding: 8px; border-radius: 3px; } QPushButton:hover { background-color: #21618C; }")
        self.btn_csv.clicked.connect(self.exportar_csv)

        layout_botones_sim = QHBoxLayout()
        layout_botones_sim.addWidget(self.btn_ejecutar)
        layout_botones_sim.addWidget(self.btn_csv)
        layout_cfg.addLayout(layout_botones_sim)

        gb_kpi = QGroupBox(self.tr_txt('sim_gb_kpi'))
        gb_kpi.setStyleSheet(estilo_gb)
        layout_kpi = QVBoxLayout(gb_kpi)
        layout_kpi.setSpacing(3)

        self.lbl_q_disipada = QLabel(f"{self.tr_txt('sim_kpi_q_disipada')} -- MWh_t")
        self.lbl_kwh_total = QLabel(f"{self.tr_txt('sim_kpi_kwh_total')} -- kWh_e")
        self.lbl_m3_evap = QLabel(f"{self.tr_txt('sim_kpi_m3_evap')} -- m³")
        self.lbl_m3_purga = QLabel(f"{self.tr_txt('sim_kpi_m3_purga')} -- m³")
        self.lbl_m3_drift = QLabel(f"{self.tr_txt('sim_kpi_m3_drift')} -- m³")
        self.lbl_m3_total = QLabel(f"{self.tr_txt('sim_kpi_m3_total')} -- m³")
        self.lbl_cop = QLabel(f"{self.tr_txt('sim_kpi_cop')} -- kWh_t/kWh_e")
        self.lbl_int_agua_mwh = QLabel(f"{self.tr_txt('sim_kpi_int_agua')} -- m³/MWh_t")

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

        gb_vars = QGroupBox(self.tr_txt('sim_gb_vars'))
        gb_vars.setStyleSheet(estilo_gb)
        layout_chk = QVBoxLayout(gb_vars)
        layout_chk.setSpacing(6)

        self.chk_tin = QCheckBox(self.tr_txt('chk_tin'))
        self.chk_tin.setChecked(True)
        self.chk_tout = QCheckBox(self.tr_txt('chk_tout'))
        self.chk_tout.setChecked(True)
        self.chk_speed = QCheckBox(self.tr_txt('chk_speed'))
        self.chk_speed.setChecked(True)
        self.chk_power = QCheckBox(self.tr_txt('chk_power'))
        self.chk_power.setChecked(False)
        self.chk_twb = QCheckBox(self.tr_txt('chk_twb'))
        self.chk_twb.setChecked(True)
        self.chk_tdb = QCheckBox(self.tr_txt('chk_tdb'))
        self.chk_tdb.setChecked(False)
        self.chk_taout = QCheckBox(self.tr_txt('chk_taout'))
        self.chk_taout.setChecked(False)
        self.chk_niebla = QCheckBox(self.tr_txt('chk_niebla'))
        self.chk_niebla.setChecked(True)
        self.chk_q = QCheckBox(self.tr_txt('chk_q'))
        self.chk_q.setChecked(True)
        self.chk_evap = QCheckBox(self.tr_txt('chk_evap'))
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

    # Dialog to allow user choice when EPW contains multiple source years
    def _prompt_epw_year_choice(self, years):
        # Check saved preference first
        try:
            settings = QSettings('cooling_towers', 'tower_app')
            saved_action = settings.value('epw_choice_action', '')
            saved_year = settings.value('epw_choice_year', '')
            if saved_action in ('preserve', 'normalize'):
                year_val = int(saved_year) if saved_year not in (None, '', 'None') else None
                return {'action': saved_action, 'year': year_val}
        except Exception:
            pass

        dlg = DialogoEpwChoice(self, years, idioma=self.idioma)
        res = dlg.exec_()
        if res == QDialog.Accepted:
            return dlg.get_choice()
        return {'action': 'preserve', 'year': None}

    def examinar_epw(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, self.tr_txt('sim_dlg_examinar_title'), "", self.tr_txt('sim_dlg_examinar_filter')
        )
        if file_path:
            self.txt_epw_path.setText(file_path)
            try:
                clima = leer_archivo_epw(file_path)
                if clima:
                    years = sorted(set(c['dt'].year for c in clima))
                    # si hay múltiples años, preguntar al usuario cómo tratarlos
                    if len(years) > 1:
                        choice = self._prompt_epw_year_choice(years)
                    else:
                        choice = {'action': 'preserve', 'year': None}

                    # aplicar normalización solo para la vista previa (y recordar decisión para la simulación)
                    self.epw_normalize = (choice['action'] == 'normalize')
                    self.epw_normalize_year = int(choice['year']) if choice.get('year') else None

                    if self.epw_normalize and self.epw_normalize_year:
                        # crear copia normalizada de las entradas para la vista
                        clima_preview = []
                        for r in clima:
                            dt0 = r['dt']
                            y = self.epw_normalize_year
                            m = dt0.month
                            d = dt0.day
                            h = dt0.hour
                            # ajustar días inválidos (e.g., Feb 29)
                            valid_dt = None
                            try:
                                valid_dt = datetime(y, m, d, h)
                            except ValueError:
                                # intentar retroceder hasta fecha válida
                                for dd in (28, 27, 26, 25):
                                    try:
                                        valid_dt = datetime(y, m, dd, h)
                                        break
                                    except Exception:
                                        valid_dt = None
                            if valid_dt is None:
                                valid_dt = datetime(y, m, max(1, min(28, d)), h)
                            nr = dict(r)
                            nr['dt'] = valid_dt
                            clima_preview.append(nr)
                        clima_use = clima_preview
                    else:
                        clima_use = clima

                    dt_min = min(c['dt'] for c in clima_use)
                    dt_max = max(c['dt'] for c in clima_use)
                    # usar la fecha mínima encontrada como inicio
                    q_ini = QDate(dt_min.year, dt_min.month, dt_min.day)
                    # por defecto mostrar una ventana de 7 días o hasta la fecha máxima disponible
                    dt_fin_def = dt_min + timedelta(days=6)
                    dt_fin_use = dt_fin_def if dt_fin_def <= dt_max else dt_max
                    q_fin = QDate(dt_fin_use.year, dt_fin_use.month, dt_fin_use.day)
                    self.date_ini.setDate(q_ini)
                    self.date_fin.setDate(q_fin)
            except Exception:
                pass

    def ejecutar_simulacion(self):
        path_epw = self.txt_epw_path.text()
        if not path_epw or not os.path.exists(path_epw):
            QMessageBox.warning(self, self.tr_txt('title_archivo_faltante'), self.tr_txt('msg_archivo_faltante'))
            return

        try:
            d_ini = self.date_ini.date()
            d_fin = self.date_fin.date()

            cfg = {
                'path_epw': path_epw,
                'fecha_inicio': datetime(d_ini.year(), d_ini.month(), d_ini.day(), 0, 0),
                'fecha_fin': datetime(d_fin.year(), d_fin.month(), d_fin.day(), 23, 59),
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
            # incluir la preferencia de normalización EPW si fue seleccionada
            cfg['epw_normalize'] = getattr(self, 'epw_normalize', False)
            cfg['epw_normalize_year'] = getattr(self, 'epw_normalize_year', None)

            self.progress = QProgressDialog(self.tr_txt('sim_iniciando'), "Cancelar", 0, 100, self)
            self.progress.setWindowTitle(self.tr_txt('title_sim_pid'))
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
            QMessageBox.warning(self, self.tr_txt('title_entrada_invalida'), self.tr_txt('msg_entrada_invalida_sim'))

    def actualizar_progreso(self, msg, pct):
        if hasattr(self, 'progress') and self.progress:
            self.progress.setLabelText(msg)
            self.progress.setValue(pct)

    def procesar_exito(self, res):
        if hasattr(self, 'progress') and self.progress:
            self.progress.close()
        self.btn_ejecutar.setEnabled(True)
        self.res_sim = res

        self.actualizar_labels_kpi(res)

        # Notificar a la ventana principal para activar el menú de Pluma
        if self.parent() and hasattr(self.parent(), 'actualizar_resultado_dinamico'):
            self.parent().actualizar_resultado_dinamico(res)

        self.replot()

    def actualizar_labels_kpi(self, res):
        self.lbl_q_disipada.setText(f"{self.tr_txt('sim_kpi_q_disipada')} <b>{res['energia_disipada_mwh_t']:.2f} MWh_t</b>")
        self.lbl_kwh_total.setText(f"{self.tr_txt('sim_kpi_kwh_total')} <b>{res['energia_total_kwh']:.2f} kWh_e</b>")
        self.lbl_m3_evap.setText(f"{self.tr_txt('sim_kpi_m3_evap')} <b>{res['agua_evap_m3']:.2f} m³</b>")
        self.lbl_m3_purga.setText(f"{self.tr_txt('sim_kpi_m3_purga')} <b>{res['agua_purga_m3']:.2f} m³</b>")
        self.lbl_m3_drift.setText(f"{self.tr_txt('sim_kpi_m3_drift')} <b>{res['agua_drift_m3']:.2f} m³</b>")
        self.lbl_m3_total.setText(f"{self.tr_txt('sim_kpi_m3_total')} <b style='color:#D35400;'>{res['agua_total_makeup_m3']:.2f} m³</b>")
        self.lbl_cop.setText(f"{self.tr_txt('sim_kpi_cop')} <b>{res['cop_torre']:.2f} kWh_t/kWh_e</b>")
        self.lbl_int_agua_mwh.setText(f"{self.tr_txt('sim_kpi_int_agua')} <b>{res['intensidad_agua_m3_mwh']:.3f} m³/MWh_t</b>")

    def obtener_config_actual(self):
        return {
            'epw_path': self.txt_epw_path.text(),
            'date_ini': self.date_ini.date(),
            'date_fin': self.date_fin.date(),
            'dt_sim': self.txt_dt_sim.text(),
            'vol_estanque': self.txt_vol_estanque.text(),
            'coc': self.txt_coc.text(),
            'drift': self.txt_drift.text(),
            't_set': self.txt_t_set.text(),
            'kp': self.txt_kp.text(),
            'ti': self.txt_ti.text(),
            'td': self.txt_td.text(),
            'speed_min': self.txt_speed_min.text(),
            'p_motor': self.txt_p_motor.text(),
            'eta_fan': self.txt_eta_fan.text(),
            'chk_tin': self.chk_tin.isChecked(),
            'chk_tout': self.chk_tout.isChecked(),
            'chk_speed': self.chk_speed.isChecked(),
            'chk_power': self.chk_power.isChecked(),
            'chk_twb': self.chk_twb.isChecked(),
            'chk_tdb': self.chk_tdb.isChecked(),
            'chk_taout': self.chk_taout.isChecked(),
            'chk_niebla': self.chk_niebla.isChecked(),
            'chk_q': self.chk_q.isChecked(),
            'chk_evap': self.chk_evap.isChecked(),
            'res_sim': self.res_sim,
        }

    def restaurar_estado(self, estado):
        self.txt_epw_path.setText(estado['epw_path'])
        self.date_ini.setDate(estado['date_ini'])
        self.date_fin.setDate(estado['date_fin'])
        self.txt_dt_sim.setText(estado['dt_sim'])
        self.txt_vol_estanque.setText(estado['vol_estanque'])
        self.txt_coc.setText(estado['coc'])
        self.txt_drift.setText(estado['drift'])
        self.txt_t_set.setText(estado['t_set'])
        self.txt_kp.setText(estado['kp'])
        self.txt_ti.setText(estado['ti'])
        self.txt_td.setText(estado['td'])
        self.txt_speed_min.setText(estado['speed_min'])
        self.txt_p_motor.setText(estado['p_motor'])
        self.txt_eta_fan.setText(estado['eta_fan'])

        self.chk_tin.setChecked(estado['chk_tin'])
        self.chk_tout.setChecked(estado['chk_tout'])
        self.chk_speed.setChecked(estado['chk_speed'])
        self.chk_power.setChecked(estado['chk_power'])
        self.chk_twb.setChecked(estado['chk_twb'])
        self.chk_tdb.setChecked(estado['chk_tdb'])
        self.chk_taout.setChecked(estado['chk_taout'])
        self.chk_niebla.setChecked(estado['chk_niebla'])
        self.chk_q.setChecked(estado['chk_q'])
        self.chk_evap.setChecked(estado['chk_evap'])

        if estado.get('res_sim') is not None:
            self.res_sim = estado['res_sim']
            self.actualizar_labels_kpi(self.res_sim)
            self.replot()

    def exportar_csv(self):
        if self.res_sim is None:
            QMessageBox.warning(self, self.tr_txt('title_sin_datos'), self.tr_txt('msg_sin_datos_csv'))
            return

        variables = [
            (self.chk_tin, 't_in', 'chk_tin'),
            (self.chk_tout, 't_out', 'chk_tout'),
            (self.chk_twb, 't_wb', 'chk_twb'),
            (self.chk_tdb, 't_db', 'chk_tdb'),
            (self.chk_taout, 't_a_out', 'chk_taout'),
            (self.chk_speed, 'fan_speed', 'chk_speed'),
            (self.chk_power, 'power_kw', 'chk_power'),
            (self.chk_q, 'q_mwt', 'chk_q'),
            (self.chk_evap, 'evap', 'chk_evap'),
            (self.chk_niebla, 'niebla', 'chk_niebla'),
        ]
        claves_datos = [data_key for chk, data_key, label_key in variables if chk.isChecked() and data_key in self.res_sim]
        encabezados = [self.tr_txt('csv_col_fecha')] + [self.tr_txt(label_key) for chk, data_key, label_key in variables if chk.isChecked() and data_key in self.res_sim]

        if not claves_datos:
            QMessageBox.warning(self, self.tr_txt('title_sin_datos'), self.tr_txt('msg_sin_variables_csv'))
            return

        file_path, _ = QFileDialog.getSaveFileName(self, self.tr_txt('sim_csv_dialog_title'), "", self.tr_txt('sim_csv_filter'))
        if not file_path:
            return
        if not file_path.lower().endswith('.csv'):
            file_path += '.csv'

        try:
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(encabezados)
                times = self.res_sim['times']
                for i in range(len(times)):
                    fila = [times[i].strftime('%d/%m/%Y %H:%M:%S')]
                    for key in claves_datos:
                        fila.append(self.res_sim[key][i])
                    writer.writerow(fila)
            QMessageBox.information(self, self.tr_txt('title_csv_exportado'), self.tr_txt('msg_csv_exportado', path=file_path))
        except Exception as e:
            QMessageBox.critical(self, self.tr_txt('title_error_csv'), self.tr_txt('msg_error_csv', err=e))

    def procesar_cancelado(self):
        if hasattr(self, 'progress') and self.progress:
            self.progress.close()
        self.btn_ejecutar.setEnabled(True)

    def procesar_error(self, err):
        if hasattr(self, 'progress') and self.progress:
            self.progress.close()
        self.btn_ejecutar.setEnabled(True)
        QMessageBox.critical(self, self.tr_txt('title_error_sim'), self.tr_txt('msg_error_sim', err=err))

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
                        lbl_niebla = self.tr_txt('plot_niebla_activa') if idx_f == 0 else ""
                        ax_top.axvspan(times[i], times[min(f, len(times)-1)], color='#7F8C8D', alpha=0.20, linewidth=0, label=lbl_niebla)

            if mostrar_temperaturas:
                if self.chk_tin.isChecked():
                    l_tin, = ax_top.plot(times, self.res_sim['t_in'], color='#D35400', label=self.tr_txt('plot_tin'), linewidth=1.3, linestyle='--')
                    lines.append(l_tin)
                if self.chk_tout.isChecked():
                    l1, = ax_top.plot(times, self.res_sim['t_out'], color='#C0392B', label=self.tr_txt('plot_tout'), linewidth=1.5)
                    lines.append(l1)
                    l_set, = ax_top.plot(times, [self.res_sim['t_setpoint']]*len(times), color='#C0392B', linestyle=':', alpha=0.6, label=self.tr_txt('plot_setpoint'))
                    lines.append(l_set)
                if self.chk_twb.isChecked():
                    l2, = ax_top.plot(times, self.res_sim['t_wb'], color='#2980B9', linestyle=':', label=self.tr_txt('plot_twb'))
                    lines.append(l2)
                if self.chk_tdb.isChecked():
                    l_tdb, = ax_top.plot(times, self.res_sim['t_db'], color='#16A085', linestyle='-.', label=self.tr_txt('plot_tdb'), alpha=0.85)
                    lines.append(l_tdb)
                if self.chk_taout.isChecked():
                    l_taout, = ax_top.plot(times, self.res_sim['t_a_out'], color='#8E44AD', linestyle='-', label=self.tr_txt('plot_taout'), linewidth=1.2)
                    lines.append(l_taout)

                ax_top.set_ylabel(self.tr_txt('plot_ylabel_temp'), color='#222222', fontsize=8)
                ax_top.tick_params(labelsize=8)

            if mostrar_velocidad or mostrar_potencia:
                ax_sec = ax_top.twinx() if mostrar_temperaturas else ax_top
                if mostrar_velocidad:
                    l3, = ax_sec.plot(times, self.res_sim['fan_speed'], color='#27AE60', label=self.tr_txt('plot_speed'), linewidth=1.2)
                    lines.append(l3)
                    ax_sec.set_ylabel(self.tr_txt('plot_ylabel_vel'), color='#27AE60', fontsize=8)
                    ax_sec.set_ylim(-5, 105)
                if mostrar_potencia:
                    l_pow, = ax_sec.plot(times, self.res_sim['power_kw'], color='#2980B9', linestyle='--', label=self.tr_txt('plot_power'), linewidth=1.2)
                    lines.append(l_pow)
                    if not mostrar_velocidad:
                        ax_sec.set_ylabel(self.tr_txt('plot_ylabel_pow'), color='#2980B9', fontsize=8)

                ax_sec.tick_params(labelsize=8)

            labels = [l.get_label() for l in lines]
            ax_top.legend(lines, labels, loc='upper right', fontsize=8, framealpha=0.85)

        if ax_bot is not None:
            lines_bot = []
            if mostrar_carga:
                l_q, = ax_bot.plot(times, self.res_sim['q_mwt'], color='#8E44AD', label=self.tr_txt('plot_q'), linewidth=1.4)
                lines_bot.append(l_q)
                ax_bot.set_ylabel(self.tr_txt('plot_ylabel_carga'), color='#8E44AD', fontsize=8)
                ax_bot.tick_params(labelsize=8)

            if mostrar_evap:
                ax_evap = ax_bot.twinx() if mostrar_carga else ax_bot
                l_ev, = ax_evap.plot(times, self.res_sim['evap'], color='#E67E22', linestyle='-.', label=self.tr_txt('plot_evap'), linewidth=1.2)
                lines_bot.append(l_ev)
                ax_evap.set_ylabel(self.tr_txt('plot_ylabel_evap'), color='#E67E22', fontsize=8)
                ax_evap.tick_params(labelsize=8)

            labels_bot = [l.get_label() for l in lines_bot]
            ax_bot.legend(lines_bot, labels_bot, loc='upper right', fontsize=8, framealpha=0.85)
            ax_bot.set_xlabel(self.tr_txt('plot_xlabel_fecha'), fontsize=8)

        if ax_top is not None and ax_bot is None:
            ax_top.set_xlabel(self.tr_txt('plot_xlabel_fecha'), fontsize=8)

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

    def graficar_matriz(self, datos_res, capa_seleccionada='Tw', idioma='es'):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        nombres_capa = {
            'Tw': traducir(idioma, 'combo_tw'),
            'wa': traducir(idioma, 'combo_wa'),
            'Ta': traducir(idioma, 'combo_ta'),
        }

        if capa_seleccionada == 'Tw':
            Matriz_plot = datos_res['Matriz_T_w']
            cmap_use = 'coolwarm'
            label_cbar = traducir(idioma, 'mapa2d_cbar_tw')
        elif capa_seleccionada == 'wa':
            Matriz_plot = datos_res['Matriz_w_a']
            cmap_use = 'Blues'
            label_cbar = traducir(idioma, 'mapa2d_cbar_wa')
        else:
            Matriz_plot = datos_res['Matriz_T_a']
            cmap_use = 'YlOrRd'
            label_cbar = traducir(idioma, 'mapa2d_cbar_ta')

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
            
            self.ax.plot([], [], color='#666666', alpha=0.5, linewidth=6, label=traducir(idioma, 'mapa2d_zona_niebla'))
            self.ax.plot([], [], color='#222222', linestyle='--', linewidth=1.5, label=traducir(idioma, 'mapa2d_frente_condensacion'))
            self.ax.legend(loc='lower left', fontsize=8, framealpha=0.85)

        motor_str = "CoolProp Engine" if HAS_COOLPROP else "ASHRAE Standard Engine"
        N = datos_res['num_celdas']
        titulo_texto = traducir(
            idioma, 'mapa2d_titulo',
            n=N, capa=nombres_capa.get(capa_seleccionada, capa_seleccionada),
            ntu=datos_res['NTU'], motor=motor_str,
            tin=datos_res['T_w_in'], tsal=datos_res['T_salida']
        )
        self.ax.set_title(titulo_texto, fontsize=10, fontweight='bold', color='#222222', pad=12)

        self.ax.set_xlabel(traducir(idioma, 'mapa2d_xlabel'), fontsize=9, color='#444444', labelpad=8)
        self.ax.set_ylabel(traducir(idioma, 'mapa2d_ylabel'), fontsize=9, color='#444444', labelpad=8)
        self.ax.tick_params(labelsize=8)

        self.fig.tight_layout()
        self.draw()

# ==========================================
# TRADUCCIONES DE LA INTERFAZ (ES / EN)
# ==========================================
TRANSLATIONS = {
    'es': {
        'title': "CTSim - Gemelo Digital 2D - Torre de Enfriamiento (Poppe)",
        'menu_simulacion': "Simulación",
        'menu_idioma': "Idioma",
        'idioma_es': "Español",
        'idioma_en': "English",
        'accion_sim_dinamica': "Simulación Dinámica EPW (Control PID)...",
        'tip_sim_dinamica': "Ejecutar simulación dinámica anual con archivo EPW y control PID del ventilador",
        'accion_ver_pluma': "Ver Perfil de Pluma Atmosférica (Briggs 2D)...",
        'tip_ver_pluma': "Visualizar elevación y dispersión de la pluma de humedad según modelo Briggs 2D",
        'gb_agua': "Parámetros del Agua (Punto 1)",
        'gb_aire': "Condiciones Ambientales y Malla",
        'gb_res': "Resultados de Diagnóstico Térmico",
        'lbl_Tw_in': "Temp. Entrada Agua (T_w1):",
        'lbl_Tw_out': "Temp. Salida Deseada (T_w2):",
        'lbl_caudal_w': "Caudal Volumétrico Agua:",
        'lbl_Tdb_in': "Temp. Bulbo Seco (T_db):",
        'lbl_Twb_in': "Temp. Bulbo Húmedo (T_wb):",
        'lbl_caudal_a': "Caudal Aire Ventilador:",
        'lbl_densidad_a': "Densidad del Aire:",
        'lbl_altitud': "Altitud del Sitio:",
        'lbl_num_celdas': "Resolución Malla (NxN):",
        'btn_calcular': "Calibrar NTU",
        'btn_dos_puntos': "Ajuste 2 Puntos",
        'lbl_combo': "Variable a Visualizar en la Matriz 2D:",
        'combo_tw': "Temperatura del Agua (Tw)",
        'combo_wa': "Humedad Absoluta del Aire (wa)",
        'combo_ta': "Temperatura del Aire (Ta)",
        'status_default': "Primero presione 'Calibrar NTU' para habilitar la simulación dinámica EPW. [{engine}]",
        'res_ntu_label': "NTU Calibrado (P1):",
        'res_merkel_label': "Modelo Merkel:",
        'res_merkel_1p': "(Ajuste 1 Punto)",
        'res_q_label': "Carga Térmica:",
        'res_range_label': "Range (ΔTw):",
        'res_approach_label': "Approach:",
        'res_lg_label': "Relación Masa (L/G):",
        'res_evap_label': "Evaporación:",
        'res_niebla_label': "Estado Pluma/Niebla:",
        'res_niebla_si': "DETECTADA (Supersaturación)",
        'res_niebla_no': "Sin Niebla (Aire no saturado)",
        'msg_2p_exito': "Ajuste de 2 Puntos Exitoso! c={c:.3f}, m={m:.3f}. Simulación activada.",
        'msg_1p_exito': "Calibración exitosa. NTU = {ntu:.4f}. Simulación activada.",
        'msg_cancelado': "Calibración cancelada por el usuario.",
        'title_error_calib': "Error de Calibración",
        'msg_error_calib': "No se pudo calibrar:\n{err}",
        'title_entrada_invalida': "Entrada Inválida",
        'msg_entrada_invalida_1p': "Verifique que todos los campos contengan números válidos.",
        'msg_entrada_invalida_2p': "Verifique que todos los campos del Punto 1 sean válidos.",
        'title_calibracion_requerida': "Calibración Requerida",
        'msg_calibracion_requerida': "Debe calibrar la torre en la pantalla principal antes de iniciar la simulación dinámica.",
        'title_simulacion_requerida': "Simulación Requerida",
        'msg_simulacion_requerida': "Debe ejecutar una simulación dinámica antes de visualizar el perfil de pluma.",

        # --- Diálogo 2º Punto ---
        'dlg2p_title': "Configuración del 2º Punto de Funcionamiento",
        'dlg2p_info': "Ingrese las condiciones medidas para la 2ª prueba operativa:",
        'dlg2p_gb': "Condiciones del Punto 2",
        'dlg2p_Tw_out': "Temp. Salida Agua (T_w2):",
        'dlg2p_btn_ok': "Calibrar Ambas Condiciones",

        # --- Diálogo Perfil de Pluma ---
        'pluma_title': "Perfil Atmosférico de Pluma y Dispersión Hora a Hora (Briggs 2D)",
        'pluma_info': "Análisis Dinámico de Dispersión de Pluma Atmosférica y Riesgo de Recirculación",
        'pluma_gb_geom': "Parámetros Geométricos de la Estructura",
        'pluma_lbl_diametro': "Diámetro Boca Ventilador:",
        'pluma_lbl_altura': "Altura Estructura Torre:",
        'pluma_kpis_default': "Seleccione una hora para evaluar el perfil...",
        'pluma_gb_control': "Navegación Temporal en el Período Simulado",
        'pluma_fecha_default': "Fecha/Hora: --/--/---- --:--",
        'pluma_btn_worst': "📍 Ir a Máxima Pluma",
        'pluma_torre_label': "Torre de Enfriamiento",
        'pluma_visible_label': "Pluma Visible (Saturación/Niebla)",
        'pluma_eje_central': "Eje Central",
        'pluma_eje_dispersion': "Eje de Dispersión Térmica",
        'pluma_viento_inst': "Viento Inst.: {u:.1f} m/s",
        'pluma_xlabel': "Distancia Horizontal en Dirección del Viento (m)",
        'pluma_ylabel': "Altura sobre el Suelo (m)",
        'pluma_titulo': "Instante: {fecha} | v0={v0:.1f} m/s | T_salida={tsal:.1f}°C | T_amb={tamb:.1f}°C",
        'pluma_riesgo_critico': "CRÍTICO (Viento vence el tiro del ventilador)",
        'pluma_riesgo_moderado': "MODERADO (Deflexión severa)",
        'pluma_riesgo_bajo': "BAJO (Flotabilidad térmica estable)",
        'pluma_kpi_texto': "<b>Velocidad Descarga (v0):</b> {v0:.1f} m/s | <b>Longitud Pluma Visible:</b> {l:.1f} m | <b>Altura Máxima:</b> {h:.1f} m | <b>Riesgo Recirculación:</b> {riesgo}",
        'pluma_fecha_texto': "📅 {fecha}",

        # --- Ventana Simulación Dinámica EPW ---
        'sim_title': "Simulación Dinámica Anual / Climática (.EPW) con PID y Balance Hídrico",
        'sim_gb_epw': "1. Archivo Climático EPW",
        'sim_epw_placeholder': "Seleccione archivo .epw...",
        'sim_btn_examinar': "Examinar...",
        'sim_gb_tiempo': "2. Rango Temporal, Estanque y Purga",
        'sim_lbl_fecha_ini': "Fecha Inicio:",
        'sim_lbl_fecha_fin': "Fecha Fin:",
        'sim_lbl_dt': "Paso Tiempo Δt:",
        'sim_lbl_vol_estanque': "Vol. Estanque:",
        'sim_lbl_coc': "Ciclos Concentración (COC):",
        'sim_lbl_drift': "Arrastre / Drift:",
        'sim_gb_pid': "3. Controlador PID del Ventilador",
        'sim_lbl_setpoint': "Setpoint Temp. Agua:",
        'sim_lbl_kp': "Ganancia Kp:",
        'sim_lbl_ti': "Tiempo Integral Ti:",
        'sim_lbl_td': "Tiempo Derivativo Td:",
        'sim_lbl_speed_min': "Velocidad Mínima:",
        'sim_gb_motor': "4. Motor y Eficiencia",
        'sim_lbl_p_motor': "Potencia Motor:",
        'sim_lbl_eta_fan': "Eficiencia Global:",
        'sim_btn_ejecutar': "Simular",
        'sim_gb_kpi': "📊 Balance de Agua y KPIs del Período",
        'sim_kpi_q_disipada': "Energía Disipada:",
        'sim_kpi_kwh_total': "Energía Consumida:",
        'sim_kpi_m3_evap': "Agua Evaporada (E):",
        'sim_kpi_m3_purga': "Agua Purga (B):",
        'sim_kpi_m3_drift': "Agua Arrastre (D):",
        'sim_kpi_m3_total': "Reposición Total (Make-up):",
        'sim_kpi_cop': "Rendimiento (COP):",
        'sim_kpi_int_agua': "Consumo Espec. Agua:",
        'sim_gb_vars': "Variables a Graficar",
        'chk_tin': "Temp. Entrada Agua Tw1 (°C)",
        'chk_tout': "Temp. Salida Agua Tw2 (°C)",
        'chk_speed': "Velocidad Ventilador (%)",
        'chk_power': "Potencia Eléctrica (kW)",
        'chk_twb': "Bulbo Húmedo Twb (°C)",
        'chk_tdb': "Bulbo Seco Ext. Tdb (°C)",
        'chk_taout': "Temp. Salida Aire Ta,out (°C)",
        'chk_niebla': "Presencia Niebla (Sombra)",
        'chk_q': "Carga Térmica (MWt)",
        'chk_evap': "Evaporación (m³/h)",
        'sim_dlg_examinar_title': "Seleccionar archivo climático EPW",
        'sim_dlg_examinar_filter': "Archivos EPW (*.epw);;Todos los archivos (*.*)",
        'epw_multi_title': "EPW: Archivo con Múltiples Años Detectado",
        'epw_multi_info': "El archivo EPW contiene filas originadas en varios años. ¿Cómo desea tratarlas?",
        'epw_multi_preserve': "Preservar años originales (mantener datetimes tal cual)",
        'epw_multi_normalize': "Normalizar a un solo año:",
        'epw_multi_years_present': "Años presentes en el archivo: {years}",
        'epw_multi_remember': "Recordar mi elección",
        'epw_choice_cleared_title': "Preferencia EPW borrada",
        'epw_choice_cleared_msg': "La preferencia guardada para archivos EPW ha sido eliminada.",
        'epw_choice_cleared_err': "No se pudo eliminar la preferencia guardada.",
        'sim_clear_epw_choice': "Borrar preferencia EPW guardada",
        'menu_settings': "Configuración",
        'sim_reset_prefs': "Restablecer todas las preferencias",
        'reset_prefs_confirm_title': "Restablecer Preferencias",
        'reset_prefs_confirm_msg': "¿Desea restablecer todas las preferencias de la aplicación a sus valores por defecto?",
        'reset_prefs_done_msg': "Preferencias restablecidas correctamente.",
        'reset_prefs_err_msg': "No se pudieron restablecer las preferencias.",
        'title_archivo_faltante': "Archivo Faltante",
        'msg_archivo_faltante': "Por favor seleccione un archivo .epw válido.",
        'title_sim_pid': "Simulación Dinámica PID",
        'sim_iniciando': "Iniciando simulación temporal...",
        'msg_entrada_invalida_sim': "Por favor revise los parámetros numéricos ingresados.",
        'title_error_sim': "Error de Simulación",
        'msg_error_sim': "Ocurrió un error:\n{err}",

        # --- Exportación CSV ---
        'sim_btn_csv': "💾 Exportar",
        'sim_csv_dialog_title': "Guardar Resultados como CSV",
        'sim_csv_filter': "Archivos CSV (*.csv)",
        'title_sin_datos': "Sin Datos",
        'msg_sin_datos_csv': "Debe ejecutar una simulación antes de exportar resultados.",
        'msg_sin_variables_csv': "Seleccione al menos una variable para exportar.",
        'title_csv_exportado': "Exportación Exitosa",
        'msg_csv_exportado': "Resultados exportados exitosamente a:\n{path}",
        'title_error_csv': "Error de Exportación",
        'msg_error_csv': "No se pudo exportar el archivo:\n{err}",
        'csv_col_fecha': "Fecha/Hora",

        # --- Gráficos dinámicos (replot) ---
        'plot_niebla_activa': "Pluma/Niebla Activa",
        'plot_tin': "Temp. Entrada Agua Tw1 (°C)",
        'plot_tout': "Temp. Salida Agua Tw2 (°C)",
        'plot_setpoint': "Setpoint Agua",
        'plot_twb': "Bulbo Húmedo Twb (°C)",
        'plot_tdb': "Bulbo Seco Tdb (°C)",
        'plot_taout': "Temp. Salida Aire Ta,out (°C)",
        'plot_ylabel_temp': "Temperatura (°C)",
        'plot_speed': "Velocidad Ventilador (%)",
        'plot_ylabel_vel': "Velocidad (%)",
        'plot_power': "Potencia Eléctrica (kW)",
        'plot_ylabel_pow': "Potencia (kW)",
        'plot_q': "Carga Térmica (MWt)",
        'plot_ylabel_carga': "Carga (MWt)",
        'plot_evap': "Evaporación (m³/h)",
        'plot_ylabel_evap': "Evaporación (m³/h)",
        'plot_xlabel_fecha': "Fecha / Hora",

        # --- Mapa 2D (MplCanvas) ---
        'mapa2d_cbar_tw': "Temperatura del Agua (°C)",
        'mapa2d_cbar_wa': "Humedad Absoluta (g vapor / kg aire)",
        'mapa2d_cbar_ta': "Temp. Bulbo Seco Aire (°C)",
        'mapa2d_zona_niebla': "Zona de Niebla",
        'mapa2d_frente_condensacion': "Frente de Condensación",
        'mapa2d_titulo': "Mapa 2D ({n}x{n}): {capa}   (NTU = {ntu:.4f})  [{motor}]\nEntrada Techo: {tin:.1f} °C   |   Piscina Mezclada: {tsal:.2f} °C",
        'mapa2d_xlabel': "Entrada Aire Ambiente   →   Dirección del Flujo de Aire   →   Salida",
        'mapa2d_ylabel': "← Caída del Agua (Techo a Piscina) →",
    },
    'en': {
        'title': "CTSim - 2D Digital Twin - Cooling Tower (Poppe)",
        'menu_simulacion': "Simulation",
        'menu_idioma': "Language",
        'idioma_es': "Español",
        'idioma_en': "English",
        'accion_sim_dinamica': "Dynamic EPW Simulation (PID Control)...",
        'tip_sim_dinamica': "Run annual dynamic simulation with EPW file and fan PID control",
        'accion_ver_pluma': "View Atmospheric Plume Profile (Briggs 2D)...",
        'tip_ver_pluma': "Visualize elevation and dispersion of the moisture plume using the Briggs 2D model",
        'gb_agua': "Water Parameters (Point 1)",
        'gb_aire': "Ambient Conditions and Grid",
        'gb_res': "Thermal Diagnostic Results",
        'lbl_Tw_in': "Water Inlet Temp. (T_w1):",
        'lbl_Tw_out': "Desired Outlet Temp. (T_w2):",
        'lbl_caudal_w': "Water Volumetric Flow:",
        'lbl_Tdb_in': "Dry Bulb Temp. (T_db):",
        'lbl_Twb_in': "Wet Bulb Temp. (T_wb):",
        'lbl_caudal_a': "Fan Air Flow:",
        'lbl_densidad_a': "Air Density:",
        'lbl_altitud': "Site Altitude:",
        'lbl_num_celdas': "Grid Resolution (NxN):",
        'btn_calcular': "Calibrate NTU",
        'btn_dos_puntos': "2-Point Fit",
        'lbl_combo': "Variable to Display in 2D Grid:",
        'combo_tw': "Water Temperature (Tw)",
        'combo_wa': "Air Humidity Ratio (wa)",
        'combo_ta': "Air Temperature (Ta)",
        'status_default': "First press 'Calibrate NTU' to enable the dynamic EPW simulation. [{engine}]",
        'res_ntu_label': "Calibrated NTU (P1):",
        'res_merkel_label': "Merkel Model:",
        'res_merkel_1p': "(1-Point Fit)",
        'res_q_label': "Thermal Load:",
        'res_range_label': "Range (ΔTw):",
        'res_approach_label': "Approach:",
        'res_lg_label': "Mass Ratio (L/G):",
        'res_evap_label': "Evaporation:",
        'res_niebla_label': "Plume/Fog Status:",
        'res_niebla_si': "DETECTED (Supersaturation)",
        'res_niebla_no': "No Fog (Unsaturated Air)",
        'msg_2p_exito': "2-Point Fit Successful! c={c:.3f}, m={m:.3f}. Simulation enabled.",
        'msg_1p_exito': "Calibration successful. NTU = {ntu:.4f}. Simulation enabled.",
        'msg_cancelado': "Calibration cancelled by the user.",
        'title_error_calib': "Calibration Error",
        'msg_error_calib': "Could not calibrate:\n{err}",
        'title_entrada_invalida': "Invalid Input",
        'msg_entrada_invalida_1p': "Please check that all fields contain valid numbers.",
        'msg_entrada_invalida_2p': "Please check that all Point 1 fields are valid.",
        'title_calibracion_requerida': "Calibration Required",
        'msg_calibracion_requerida': "You must calibrate the tower on the main screen before starting the dynamic simulation.",
        'title_simulacion_requerida': "Simulation Required",
        'msg_simulacion_requerida': "You must run a dynamic simulation before viewing the plume profile.",

        # --- Point 2 Dialog ---
        'dlg2p_title': "Operating Point 2 Configuration",
        'dlg2p_info': "Enter the conditions measured for the 2nd operating test:",
        'dlg2p_gb': "Point 2 Conditions",
        'dlg2p_Tw_out': "Water Outlet Temp. (T_w2):",
        'dlg2p_btn_ok': "Calibrate Both Conditions",

        # --- Plume Profile Dialog ---
        'pluma_title': "Atmospheric Plume Profile and Hourly Dispersion (Briggs 2D)",
        'pluma_info': "Dynamic Analysis of Atmospheric Plume Dispersion and Recirculation Risk",
        'pluma_gb_geom': "Structure Geometric Parameters",
        'pluma_lbl_diametro': "Fan Outlet Diameter:",
        'pluma_lbl_altura': "Tower Structure Height:",
        'pluma_kpis_default': "Select an hour to evaluate the profile...",
        'pluma_gb_control': "Time Navigation in the Simulated Period",
        'pluma_fecha_default': "Date/Time: --/--/---- --:--",
        'pluma_btn_worst': "📍 Go to Maximum Plume",
        'pluma_torre_label': "Cooling Tower",
        'pluma_visible_label': "Visible Plume (Saturation/Fog)",
        'pluma_eje_central': "Central Axis",
        'pluma_eje_dispersion': "Thermal Dispersion Axis",
        'pluma_viento_inst': "Instant Wind: {u:.1f} m/s",
        'pluma_xlabel': "Horizontal Distance in Wind Direction (m)",
        'pluma_ylabel': "Height Above Ground (m)",
        'pluma_titulo': "Time: {fecha} | v0={v0:.1f} m/s | T_out={tsal:.1f}°C | T_amb={tamb:.1f}°C",
        'pluma_riesgo_critico': "CRITICAL (Wind overcomes fan draft)",
        'pluma_riesgo_moderado': "MODERATE (Severe deflection)",
        'pluma_riesgo_bajo': "LOW (Stable thermal buoyancy)",
        'pluma_kpi_texto': "<b>Discharge Velocity (v0):</b> {v0:.1f} m/s | <b>Visible Plume Length:</b> {l:.1f} m | <b>Maximum Height:</b> {h:.1f} m | <b>Recirculation Risk:</b> {riesgo}",
        'pluma_fecha_texto': "📅 {fecha}",

        # --- EPW Dynamic Simulation Window ---
        'sim_title': "Annual / Climatic Dynamic Simulation (.EPW) with PID and Water Balance",
        'sim_gb_epw': "1. EPW Climate File",
        'sim_epw_placeholder': "Select .epw file...",
        'sim_btn_examinar': "Browse...",
        'sim_gb_tiempo': "2. Time Range, Pond and Blowdown",
        'sim_lbl_fecha_ini': "Start Date:",
        'sim_lbl_fecha_fin': "End Date:",
        'sim_lbl_dt': "Time Step Δt:",
        'sim_lbl_vol_estanque': "Pond Volume:",
        'sim_lbl_coc': "Cycles of Concentration (COC):",
        'sim_lbl_drift': "Drift:",
        'sim_gb_pid': "3. Fan PID Controller",
        'sim_lbl_setpoint': "Water Temp. Setpoint:",
        'sim_lbl_kp': "Gain Kp:",
        'sim_lbl_ti': "Integral Time Ti:",
        'sim_lbl_td': "Derivative Time Td:",
        'sim_lbl_speed_min': "Minimum Speed:",
        'sim_gb_motor': "4. Motor and Efficiency",
        'sim_lbl_p_motor': "Motor Power:",
        'sim_lbl_eta_fan': "Overall Efficiency:",
        'sim_btn_ejecutar': "Simulate",
        'sim_gb_kpi': "📊 Water Balance and Period KPIs",
        'sim_kpi_q_disipada': "Dissipated Energy:",
        'sim_kpi_kwh_total': "Consumed Energy:",
        'sim_kpi_m3_evap': "Evaporated Water (E):",
        'sim_kpi_m3_purga': "Blowdown Water (B):",
        'sim_kpi_m3_drift': "Drift Water (D):",
        'sim_kpi_m3_total': "Total Make-up:",
        'sim_kpi_cop': "Performance (COP):",
        'sim_kpi_int_agua': "Specific Water Consumption:",
        'sim_gb_vars': "Variables to Plot",
        'chk_tin': "Water Inlet Temp. Tw1 (°C)",
        'chk_tout': "Water Outlet Temp. Tw2 (°C)",
        'chk_speed': "Fan Speed (%)",
        'chk_power': "Electrical Power (kW)",
        'chk_twb': "Wet Bulb Twb (°C)",
        'chk_tdb': "Ext. Dry Bulb Tdb (°C)",
        'chk_taout': "Air Outlet Temp. Ta,out (°C)",
        'chk_niebla': "Fog Presence (Shading)",
        'chk_q': "Thermal Load (MWt)",
        'chk_evap': "Evaporation (m³/h)",
        'sim_dlg_examinar_title': "Select EPW Climate File",
        'sim_dlg_examinar_filter': "EPW Files (*.epw);;All Files (*.*)",
        'epw_multi_title': "EPW: Multiple Years Detected",
        'epw_multi_info': "The EPW file contains rows from multiple source years. How would you like to treat the dates?",
        'epw_multi_preserve': "Preserve original years (keep datetimes as-is)",
        'epw_multi_normalize': "Normalize to a single year:",
        'epw_multi_years_present': "Years present in file: {years}",
        'epw_multi_remember': "Remember my choice",
        'epw_choice_cleared_title': "EPW Preference Cleared",
        'epw_choice_cleared_msg': "Saved preference for EPW files has been removed.",
        'epw_choice_cleared_err': "Could not remove saved EPW preference.",
        'sim_clear_epw_choice': "Clear saved EPW preference",
        'menu_settings': "Settings",
        'sim_reset_prefs': "Reset all preferences",
        'reset_prefs_confirm_title': "Reset Preferences",
        'reset_prefs_confirm_msg': "Reset all application preferences to their defaults?",
        'reset_prefs_done_msg': "Preferences reset successfully.",
        'reset_prefs_err_msg': "Could not reset preferences.",
        'title_archivo_faltante': "Missing File",
        'msg_archivo_faltante': "Please select a valid .epw file.",
        'title_sim_pid': "Dynamic PID Simulation",
        'sim_iniciando': "Starting time simulation...",
        'msg_entrada_invalida_sim': "Please check the entered numeric parameters.",
        'title_error_sim': "Simulation Error",
        'msg_error_sim': "An error occurred:\n{err}",

        # --- CSV Export ---
        'sim_btn_csv': "💾 Export",
        'sim_csv_dialog_title': "Save Results as CSV",
        'sim_csv_filter': "CSV Files (*.csv)",
        'title_sin_datos': "No Data",
        'msg_sin_datos_csv': "You must run a simulation before exporting results.",
        'msg_sin_variables_csv': "Select at least one variable to export.",
        'title_csv_exportado': "Export Successful",
        'msg_csv_exportado': "Results successfully exported to:\n{path}",
        'title_error_csv': "Export Error",
        'msg_error_csv': "Could not export the file:\n{err}",
        'csv_col_fecha': "Date/Time",

        # --- Dynamic Charts (replot) ---
        'plot_niebla_activa': "Active Plume/Fog",
        'plot_tin': "Water Inlet Temp. Tw1 (°C)",
        'plot_tout': "Water Outlet Temp. Tw2 (°C)",
        'plot_setpoint': "Water Setpoint",
        'plot_twb': "Wet Bulb Twb (°C)",
        'plot_tdb': "Dry Bulb Tdb (°C)",
        'plot_taout': "Air Outlet Temp. Ta,out (°C)",
        'plot_ylabel_temp': "Temperature (°C)",
        'plot_speed': "Fan Speed (%)",
        'plot_ylabel_vel': "Speed (%)",
        'plot_power': "Electrical Power (kW)",
        'plot_ylabel_pow': "Power (kW)",
        'plot_q': "Thermal Load (MWt)",
        'plot_ylabel_carga': "Load (MWt)",
        'plot_evap': "Evaporation (m³/h)",
        'plot_ylabel_evap': "Evaporation (m³/h)",
        'plot_xlabel_fecha': "Date / Time",

        # --- 2D Map (MplCanvas) ---
        'mapa2d_cbar_tw': "Water Temperature (°C)",
        'mapa2d_cbar_wa': "Humidity Ratio (g vapor / kg air)",
        'mapa2d_cbar_ta': "Air Dry Bulb Temp. (°C)",
        'mapa2d_zona_niebla': "Fog Zone",
        'mapa2d_frente_condensacion': "Condensation Front",
        'mapa2d_titulo': "2D Map ({n}x{n}): {capa}   (NTU = {ntu:.4f})  [{motor}]\nRoof Inlet: {tin:.1f} °C   |   Mixed Basin: {tsal:.2f} °C",
        'mapa2d_xlabel': "Ambient Air Inlet   →   Air Flow Direction   →   Outlet",
        'mapa2d_ylabel': "← Water Fall (Roof to Basin) →",
    },
}

# ==========================================
# 9. VENTANA PRINCIPAL DE PyQt5 CON OPCIÓN DE PLUMA EN MENÚ
# ==========================================
class TorreCoolingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.idioma = 'es'
        self._campos_labels = []
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "icono_torre.png")))
        self.setGeometry(100, 100, 1180, 750)
        self.ultimo_resultado = None
        self.ultimo_resultado_dinamico = None
        self.ultimo_estado_sim = None

        self.init_menu()
        self.init_ui()
        self.retranslate_ui()

    def tr_txt(self, key, **kwargs):
        texto = TRANSLATIONS[self.idioma][key]
        return texto.format(**kwargs) if kwargs else texto

    def cambiar_idioma(self, lang):
        if self.idioma == lang:
            return
        self.idioma = lang
        self.retranslate_ui()

    def init_menu(self):
        menubar = self.menuBar()
        self.menu_simulacion = menubar.addMenu("Simulación")

        self.action_sim_dinamica = QAction(self)
        self.action_sim_dinamica.setEnabled(False)
        self.action_sim_dinamica.triggered.connect(self.abrir_simulacion_dinamica)
        self.menu_simulacion.addAction(self.action_sim_dinamica)

        # NUEVA OPCIÓN: PERFIL DE PLUMA BRIGGS 2D
        self.action_ver_pluma = QAction(self)
        self.action_ver_pluma.setEnabled(False)
        self.action_ver_pluma.triggered.connect(self.abrir_ventana_pluma)
        self.menu_simulacion.addAction(self.action_ver_pluma)

        # Action to clear saved EPW preference (placed under Settings menu)
        self.action_clear_epw_choice = QAction(self)
        self.action_clear_epw_choice.triggered.connect(self._clear_saved_epw_choice)

        # MENÚ DE IDIOMA
        self.menu_idioma = menubar.addMenu("Idioma")
        self.action_idioma_es = QAction(self)
        self.action_idioma_es.setCheckable(True)
        self.action_idioma_es.setChecked(True)
        self.action_idioma_es.triggered.connect(lambda: self.cambiar_idioma('es'))
        self.action_idioma_en = QAction(self)
        self.action_idioma_en.setCheckable(True)
        self.action_idioma_en.triggered.connect(lambda: self.cambiar_idioma('en'))

        grupo_idioma = QActionGroup(self)
        grupo_idioma.addAction(self.action_idioma_es)
        grupo_idioma.addAction(self.action_idioma_en)

        self.menu_idioma.addAction(self.action_idioma_es)
        self.menu_idioma.addAction(self.action_idioma_en)
        # SETTINGS MENU
        self.menu_settings = menubar.addMenu(self.tr_txt('menu_settings') if 'menu_settings' in TRANSLATIONS[self.idioma] else 'Settings')
        self.menu_settings.addAction(self.action_clear_epw_choice)
        # Reset all preferences action
        self.action_reset_prefs = QAction(self)
        self.action_reset_prefs.triggered.connect(self._reset_all_preferences)
        self.menu_settings.addAction(self.action_reset_prefs)
        # set texts for language actions and simulation menu items
        self.action_idioma_es.setText(self.tr_txt('idioma_es'))
        self.action_idioma_en.setText(self.tr_txt('idioma_en'))
        # simulation menu items text
        self.action_sim_dinamica.setText(self.tr_txt('accion_sim_dinamica'))
        self.action_sim_dinamica.setToolTip(self.tr_txt('tip_sim_dinamica'))
        self.action_ver_pluma.setText(self.tr_txt('accion_ver_pluma'))
        self.action_ver_pluma.setToolTip(self.tr_txt('tip_ver_pluma'))
        self.action_clear_epw_choice.setText(self.tr_txt('sim_clear_epw_choice') if 'sim_clear_epw_choice' in TRANSLATIONS[self.idioma] else 'Clear saved EPW choice')
        # reset prefs menu item
        self.action_reset_prefs.setText(self.tr_txt('sim_reset_prefs') if 'sim_reset_prefs' in TRANSLATIONS[self.idioma] else 'Reset all preferences')

    def _clear_saved_epw_choice(self):
        try:
            settings = QSettings('cooling_towers', 'tower_app')
            settings.remove('epw_choice_action')
            settings.remove('epw_choice_year')
            QMessageBox.information(self, self.tr_txt('epw_choice_cleared_title'), self.tr_txt('epw_choice_cleared_msg'))
        except Exception:
            QMessageBox.warning(self, self.tr_txt('epw_choice_cleared_title'), self.tr_txt('epw_choice_cleared_err'))

    def _reset_all_preferences(self):
        reply = QMessageBox.question(
            self,
            self.tr_txt('reset_prefs_confirm_title'),
            self.tr_txt('reset_prefs_confirm_msg'),
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                settings = QSettings('cooling_towers', 'tower_app')
                settings.clear()
                QMessageBox.information(self, self.tr_txt('reset_prefs_confirm_title'), self.tr_txt('reset_prefs_done_msg'))
            except Exception:
                QMessageBox.warning(self, self.tr_txt('reset_prefs_confirm_title'), self.tr_txt('reset_prefs_err_msg'))

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

        gb_agua = QGroupBox()
        self.gb_agua = gb_agua
        gb_agua.setStyleSheet(estilo_gb)
        grid_agua = QGridLayout()
        grid_agua.setVerticalSpacing(4)

        self.txt_Tw_in = self.crear_input("31.7", "°C", grid_agua, 0, 'lbl_Tw_in')
        self.txt_Tw_out = self.crear_input("20.6", "°C", grid_agua, 1, 'lbl_Tw_out')
        self.txt_caudal_w = self.crear_input("1174.0", "m³/h", grid_agua, 2, 'lbl_caudal_w')
        gb_agua.setLayout(grid_agua)
        layout_izq.addWidget(gb_agua)

        gb_aire = QGroupBox()
        self.gb_aire = gb_aire
        gb_aire.setStyleSheet(estilo_gb)
        grid_aire = QGridLayout()
        grid_aire.setVerticalSpacing(4)

        self.txt_Tdb_in = self.crear_input("30.0", "°C", grid_aire, 0, 'lbl_Tdb_in')
        self.txt_Twb_in = self.crear_input("17.8", "°C", grid_aire, 1, 'lbl_Twb_in')
        self.txt_caudal_a = self.crear_input("474.1", "m³/s", grid_aire, 2, 'lbl_caudal_a')
        self.txt_densidad_a = self.crear_input("1.177", "kg/m³", grid_aire, 3, 'lbl_densidad_a', precision=3)
        self.txt_altitud = self.crear_input("0.0", "m", grid_aire, 4, 'lbl_altitud')
        self.txt_num_celdas = self.crear_input_entero("15", "celdas", grid_aire, 5, 'lbl_num_celdas')
        gb_aire.setLayout(grid_aire)
        layout_izq.addWidget(gb_aire)

        # BOTONES: CALIBRAR 1 PUNTO Y AJUSTE 2 PUNTOS
        layout_botones = QHBoxLayout()
        self.btn_calcular = QPushButton()
        self.btn_calcular.setFont(QFont("Segoe UI", 9))
        self.btn_calcular.setCursor(Qt.PointingHandCursor)
        self.btn_calcular.setStyleSheet("QPushButton { background-color: #34495E; color: #FFFFFF; border: none; padding: 6px 8px; border-radius: 3px; } QPushButton:hover { background-color: #2C3E50; }")
        self.btn_calcular.clicked.connect(self.ejecutar_calibracion_1p)

        self.btn_dos_puntos = QPushButton()
        self.btn_dos_puntos.setFont(QFont("Segoe UI", 9))
        self.btn_dos_puntos.setCursor(Qt.PointingHandCursor)
        self.btn_dos_puntos.setStyleSheet("QPushButton { background-color: #27AE60; color: #FFFFFF; border: none; padding: 6px 8px; border-radius: 3px; } QPushButton:hover { background-color: #219653; }")
        self.btn_dos_puntos.clicked.connect(self.abrir_dialogo_2puntos)

        layout_botones.addWidget(self.btn_calcular)
        layout_botones.addWidget(self.btn_dos_puntos)
        layout_izq.addLayout(layout_botones)
 

        # Grupo Resultados
        gb_res = QGroupBox()
        self.gb_res = gb_res
        gb_res.setStyleSheet(estilo_gb)
        layout_res = QVBoxLayout()
        layout_res.setSpacing(3)

        self.lbl_ntu_res = QLabel()
        self.lbl_merkel_res = QLabel()
        self.lbl_q_res = QLabel()
        self.lbl_range_res = QLabel()
        self.lbl_approach_res = QLabel()
        self.lbl_lg_res = QLabel()
        self.lbl_evap_res = QLabel()
        self.lbl_niebla_res = QLabel()

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
        self.lbl_combo = QLabel()
        self.lbl_combo.setFont(QFont("Segoe UI", 9))
        
        self.combo_capa = QComboBox()
        self.combo_capa.setFont(QFont("Segoe UI", 9))
        self.combo_capa.addItem("", 'Tw')
        self.combo_capa.addItem("", 'wa')
        self.combo_capa.addItem("", 'Ta')
        self.combo_capa.currentIndexChanged.connect(self.cambiar_capa_grafico)

        top_der_layout.addWidget(self.lbl_combo)
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

    def crear_input(self, valor_defecto, unidad, grid_layout, fila, label_key, precision=1):
        lbl = QLabel()
        lbl.setFont(QFont("Segoe UI", 9))
        self._campos_labels.append((lbl, label_key))
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

    def crear_input_entero(self, valor_defecto, unidad, grid_layout, fila, label_key):
        lbl = QLabel()
        lbl.setFont(QFont("Segoe UI", 9))
        self._campos_labels.append((lbl, label_key))
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

    def retranslate_ui(self):
        self.setWindowTitle(self.tr_txt('title'))

        self.menu_simulacion.setTitle(self.tr_txt('menu_simulacion'))
        self.action_sim_dinamica.setText(self.tr_txt('accion_sim_dinamica'))
        self.action_sim_dinamica.setStatusTip(self.tr_txt('tip_sim_dinamica'))
        self.action_ver_pluma.setText(self.tr_txt('accion_ver_pluma'))
        self.action_ver_pluma.setStatusTip(self.tr_txt('tip_ver_pluma'))

        self.menu_idioma.setTitle(self.tr_txt('menu_idioma'))
        self.action_idioma_es.setText(self.tr_txt('idioma_es'))
        self.action_idioma_en.setText(self.tr_txt('idioma_en'))

        self.gb_agua.setTitle(self.tr_txt('gb_agua'))
        self.gb_aire.setTitle(self.tr_txt('gb_aire'))
        self.gb_res.setTitle(self.tr_txt('gb_res'))

        for lbl, key in self._campos_labels:
            lbl.setText(self.tr_txt(key))

        self.btn_calcular.setText(self.tr_txt('btn_calcular'))
        self.btn_dos_puntos.setText(self.tr_txt('btn_dos_puntos'))

        self.lbl_combo.setText(self.tr_txt('lbl_combo'))
        idx_actual = self.combo_capa.currentIndex()
        self.combo_capa.blockSignals(True)
        self.combo_capa.setItemText(0, self.tr_txt('combo_tw'))
        self.combo_capa.setItemText(1, self.tr_txt('combo_wa'))
        self.combo_capa.setItemText(2, self.tr_txt('combo_ta'))
        self.combo_capa.setCurrentIndex(idx_actual)
        self.combo_capa.blockSignals(False)

        if self.ultimo_resultado is not None and 'NTU' in self.ultimo_resultado:
            self.actualizar_labels_resultado(self.ultimo_resultado)
            self.canvas.graficar_matriz(self.ultimo_resultado, self.combo_capa.currentData(), idioma=self.idioma)
        else:
            self.lbl_ntu_res.setText(f"{self.tr_txt('res_ntu_label')}  --")
            self.lbl_merkel_res.setText(f"{self.tr_txt('res_merkel_label')}  --")
            self.lbl_q_res.setText(f"{self.tr_txt('res_q_label')}  --")
            self.lbl_range_res.setText(f"{self.tr_txt('res_range_label')}  --")
            self.lbl_approach_res.setText(f"{self.tr_txt('res_approach_label')}  --")
            self.lbl_lg_res.setText(f"{self.tr_txt('res_lg_label')}  --")
            self.lbl_evap_res.setText(f"{self.tr_txt('res_evap_label')}  --")
            self.lbl_niebla_res.setText(f"{self.tr_txt('res_niebla_label')}  --")
            engine_msg = "CoolProp (NIST)" if HAS_COOLPROP else "ASHRAE Standard"
            self.status_bar.showMessage(self.tr_txt('status_default', engine=engine_msg))

    def actualizar_labels_resultado(self, res):
        self.lbl_ntu_res.setText(f"{self.tr_txt('res_ntu_label')} <b style='font-size:10.5pt; color:#2980B9;'>{res['NTU']:.4f}</b>")

        if res['es_dual']:
            c = res['c_coef']
            m = res['m_exp']
            self.lbl_merkel_res.setText(f"{self.tr_txt('res_merkel_label')} <b style='color:#8E44AD;'>c = {c:.3f}, m = {m:.3f}</b>")
        else:
            self.lbl_merkel_res.setText(f"{self.tr_txt('res_merkel_label')} <b>{self.tr_txt('res_merkel_1p')}</b>")

        self.lbl_q_res.setText(f"{self.tr_txt('res_q_label')} <b>{res['Q_MWt']:.2f} MWt</b> ({res['Q_TR']:.0f} TR)")
        self.lbl_range_res.setText(f"{self.tr_txt('res_range_label')} <b>{res['range_w']:.2f} °C</b>")
        self.lbl_approach_res.setText(f"{self.tr_txt('res_approach_label')} <b>{res['approach_w']:.2f} °C</b>")
        self.lbl_lg_res.setText(f"{self.tr_txt('res_lg_label')} <b>{res['L_G_ratio']:.3f}</b>")
        self.lbl_evap_res.setText(f"{self.tr_txt('res_evap_label')} <b>{res['evaporacion_m3h']:.2f} m³/h</b> ({res['pct_evap']:.2f}%)")

        if res['hay_niebla']:
            self.lbl_niebla_res.setText(f"{self.tr_txt('res_niebla_label')} <b style='color:#C0392B;'>{self.tr_txt('res_niebla_si')}</b>")
        else:
            self.lbl_niebla_res.setText(f"{self.tr_txt('res_niebla_label')} <b style='color:#27AE60;'>{self.tr_txt('res_niebla_no')}</b>")

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
            QMessageBox.warning(self, self.tr_txt('title_entrada_invalida'), self.tr_txt('msg_entrada_invalida_1p'))

    def abrir_dialogo_2puntos(self):
        try:
            d1 = self.obtener_datos_pantalla_p1()
            dlg = DialogoSegundoPunto(self, datos_p1=d1, idioma=self.idioma)
            if dlg.exec_() == QDialog.Accepted:
                d2 = dlg.obtener_datos_p2()
                self.lanzar_worker(d1, d2)
        except ValueError:
            QMessageBox.warning(self, self.tr_txt('title_entrada_invalida'), self.tr_txt('msg_entrada_invalida_2p'))

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

        self.actualizar_labels_resultado(res)

        if res['es_dual']:
            c = res['c_coef']
            m = res['m_exp']
            self.status_bar.showMessage(self.tr_txt('msg_2p_exito', c=c, m=m))
        else:
            self.status_bar.showMessage(self.tr_txt('msg_1p_exito', ntu=res['NTU']))

        self.canvas.graficar_matriz(res, self.combo_capa.currentData(), idioma=self.idioma)

    def procesar_cancelacion_calib(self):
        if hasattr(self, 'progress_calib') and self.progress_calib:
            self.progress_calib.close()

        self.btn_calcular.setEnabled(True)
        self.btn_dos_puntos.setEnabled(True)
        self.status_bar.showMessage(self.tr_txt('msg_cancelado'))

    def procesar_error_calib(self, err):
        if hasattr(self, 'progress_calib') and self.progress_calib:
            self.progress_calib.close()

        self.btn_calcular.setEnabled(True)
        self.btn_dos_puntos.setEnabled(True)
        QMessageBox.critical(self, self.tr_txt('title_error_calib'), self.tr_txt('msg_error_calib', err=err))

    def cambiar_capa_grafico(self, index):
        if self.ultimo_resultado is not None and 'Matriz_T_w' in self.ultimo_resultado:
            self.canvas.graficar_matriz(self.ultimo_resultado, self.combo_capa.itemData(index), idioma=self.idioma)

    def abrir_simulacion_dinamica(self):
        if self.ultimo_resultado is None or 'NTU' not in self.ultimo_resultado:
            QMessageBox.warning(self, self.tr_txt('title_calibracion_requerida'), self.tr_txt('msg_calibracion_requerida'))
            return

        d_torre = self.obtener_datos_pantalla_p1()
        d_torre['NTU'] = self.ultimo_resultado['NTU']

        dlg = VentanaSimulacionDinamica(self, datos_torre=d_torre, idioma=self.idioma, estado_previo=self.ultimo_estado_sim)
        dlg.exec_()
        self.ultimo_estado_sim = dlg.obtener_config_actual()

    def actualizar_resultado_dinamico(self, res_din):
        self.ultimo_resultado_dinamico = res_din
        self.action_ver_pluma.setEnabled(True)

    def abrir_ventana_pluma(self):
        if self.ultimo_resultado_dinamico is None:
            QMessageBox.warning(self, self.tr_txt('title_simulacion_requerida'), self.tr_txt('msg_simulacion_requerida'))
            return

        dlg = DialogoPerfilPluma(self, datos_sim=self.ultimo_resultado_dinamico, idioma=self.idioma)
        dlg.exec_()

# ==========================================
# 10. PUNTO DE ENTRADA
# ==========================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "icono_torre.png")))

    window = TorreCoolingApp()
    window.show()
    sys.exit(app.exec_())