import sys
import os
import csv
from datetime import datetime, timedelta
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
    return P0 * (1.0 - 0.6875e-5 * altitud_m)**5.2561

def cp_agua_local(T_celcius):
    if HAS_COOLPROP:
        try:
            return CP.PropsSI('C', 'T', T_celcius + 273.15, 'P', 101325, 'Water') / 1000.0
        except Exception:
            pass
    return cp_w_def

def humedad_saturacion(T, P_atm=101325.0):
    if HAS_COOLPROP:
        try:
            return CP.HAPropsSI('W', 'T', T + 273.15, 'R', 1.0, 'P', P_atm)
        except Exception:
            pass
    P_atm_kPa = P_atm / 1000.0
    P_sat = 0.61078 * np.exp(17.27 * T / (T + 237.3)) 
    return 0.622 * P_sat / (P_atm_kPa - P_sat)

def factor_lewis(w_sw, w):
    if w >= w_sw or abs(w_sw - w) < 1e-6:
        return 0.865**(2/3)
    arg = (w_sw + 0.622) / (w + 0.622)
    if arg <= 1.0 + 1e-7:
        return 0.865**(2/3)
    num = arg - 1.0
    den = np.log(arg)
    if den <= 1e-7:
        return 0.865**(2/3)
    return (0.865**(2/3)) * (num / den)

def entalpia_saturacion(T, w_sat, P_atm=101325.0):
    if HAS_COOLPROP:
        try:
            return CP.HAPropsSI('H', 'T', T + 273.15, 'W', w_sat, 'P', P_atm) / 1000.0
        except Exception:
            pass
    return cp_a_def * T + w_sat * (h_fg0_def + cp_v_def * T)

def temp_aire_desde_entalpia(h_a, w_a, P_atm=101325.0):
    if HAS_COOLPROP:
        try:
            T_kelvin = CP.HAPropsSI('T', 'H', h_a * 1000.0, 'W', w_a, 'P', P_atm)
            return T_kelvin - 273.15
        except Exception:
            pass
    return (h_a - w_a * h_fg0_def) / (cp_a_def + w_a * cp_v_def)

# ==========================================
# 2. MOTOR POPPE 2D
# ==========================================
def simular_torre_2d_matriz(NTU_actual, T_w_in, m_w_total, h_a_in, w_a_in, m_a_total, P_atm=101325.0, Nx=20, Ny=20):
    dm_w = m_w_total / Nx  
    dm_a = m_a_total / Ny  
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
            w_sw = humedad_saturacion(T_water_cell, P_atm)
            h_sw = entalpia_saturacion(T_water_cell, w_sw, P_atm)
            h_v = h_fg0_def + cp_v_def * T_water_cell
            Le = factor_lewis(w_sw, w_air_cell)
            
            potencial_w = w_sw - w_air_cell
            potencial_h = (h_sw - h_air_cell) + (Le - 1) * (h_sw - h_air_cell - potencial_w * h_v) + potencial_w * cp_w_local * T_water_cell
            
            agua_evap_celda = K_dA * potencial_w
            calor_transferido = K_dA * potencial_h
            
            w_a_next = w_air_cell + (agua_evap_celda / dm_a)
            h_a_next = h_air_cell + (calor_transferido / dm_a)
            
            w_a[i, j+1] = w_a_next
            h_a[i, j+1] = h_a_next
            
            T_a_next = temp_aire_desde_entalpia(h_a_next, w_a_next, P_atm)
            matriz_T_aire[i, j] = T_a_next
            w_sat_local = humedad_saturacion(T_a_next, P_atm)
            
            if w_a_next > w_sat_local:
                matriz_niebla[i, j] = True
            
            m_w[i+1, j] = m_water_cell - agua_evap_celda
            T_w[i+1, j] = (m_water_cell * cp_w_local * T_water_cell - calor_transferido) / (m_water_cell * cp_w_local)

    T_w_salida_final = np.average(T_w[Ny, :], weights=m_w[Ny, :])
    agua_evaporada_total = m_w_total - np.sum(m_w[Ny, :])
    
    return T_w_salida_final, agua_evaporada_total, T_w[:-1, :], w_a[:, 1:], matriz_T_aire, matriz_niebla

def simular_torre_2d_directo(NTU_fijo, T_w_in, m_w_total, T_db_in, T_wb_in, m_a_total, P_atm=101325.0, Nx=15, Ny=15):
    if HAS_COOLPROP:
        try:
            w_a_in = CP.HAPropsSI('W', 'T', T_db_in + 273.15, 'B', T_wb_in + 273.15, 'P', P_atm)
            h_a_in = CP.HAPropsSI('H', 'T', T_db_in + 273.15, 'B', T_wb_in + 273.15, 'P', P_atm) / 1000.0
        except Exception:
            w_sat_wb = humedad_saturacion(T_wb_in, P_atm)
            w_a_in = ((h_fg0_def - (cp_w_def - cp_v_def) * T_wb_in) * w_sat_wb - cp_a_def * (T_db_in - T_wb_in)) / (h_fg0_def + cp_v_def * T_db_in - cp_w_def * T_wb_in)
            h_a_in = cp_a_def * T_db_in + w_a_in * (h_fg0_def + cp_v_def * T_db_in)
    else:
        w_sat_wb = humedad_saturacion(T_wb_in, P_atm)
        w_a_in = ((h_fg0_def - (cp_w_def - cp_v_def) * T_wb_in) * w_sat_wb - cp_a_def * (T_db_in - T_wb_in)) / (h_fg0_def + cp_v_def * T_db_in - cp_w_def * T_wb_in)
        h_a_in = cp_a_def * T_db_in + w_a_in * (h_fg0_def + cp_v_def * T_db_in)

    dm_w = m_w_total / Nx  
    dm_a = m_a_total / Ny  
    K_dA = (NTU_fijo * m_w_total) / (Nx * Ny) 
    
    T_w = np.zeros((Ny + 1, Nx))
    m_w = np.zeros((Ny + 1, Nx))
    h_a = np.zeros((Ny, Nx + 1))
    w_a = np.zeros((Ny, Nx + 1))
    
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
            w_sw = humedad_saturacion(T_water_cell, P_atm)
            h_sw = entalpia_saturacion(T_water_cell, w_sw, P_atm)
            h_v = h_fg0_def + cp_v_def * T_water_cell
            Le = factor_lewis(w_sw, w_air_cell)
            
            potencial_w = w_sw - w_air_cell
            potencial_h = (h_sw - h_air_cell) + (Le - 1) * (h_sw - h_air_cell - potencial_w * h_v) + potencial_w * cp_w_local * T_water_cell
            
            agua_evap_celda = K_dA * potencial_w
            calor_transferido = K_dA * potencial_h
            
            w_a[i, j+1] = w_air_cell + (agua_evap_celda / dm_a)
            h_a[i, j+1] = h_air_cell + (calor_transferido / dm_a)
            
            m_w[i+1, j] = m_water_cell - agua_evap_celda
            T_w[i+1, j] = (m_water_cell * cp_w_local * T_water_cell - calor_transferido) / (m_water_cell * cp_w_local)

    T_w_salida_final = np.average(T_w[Ny, :], weights=m_w[Ny, :])
    agua_evaporada_total = m_w_total - np.sum(m_w[Ny, :])
    
    return T_w_salida_final, agua_evaporada_total

# Subrutina auxiliar para evaluar un punto de operación
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
# 3. CONTROLADOR PID DISCRETO CON SATURACIÓN
# ==========================================
# ==========================================
# CONTROLADOR PID Y WORKER DINÁMICO CORREGIDO
# ==========================================
class ControladorPID:
    def __init__(self, Kp=2.0, Ti=120.0, Td=0.0, u_min=20.0, u_max=100.0):
        self.Kp = Kp
        self.Ti = max(Ti, 1e-3)
        self.Td = Td
        self.u_min = u_min
        self.u_max = u_max
        self.integral = 0.0
        self.prev_error = 0.0

    def calcular(self, setpoint, medido, dt):
        # En enfriamiento: Si T_medida > T_setpoint, error POSITIVO para AUMENTAR ventilador
        error = medido - setpoint
        
        # Proporcional
        P = self.Kp * error
        
        # Integral
        self.integral += error * dt
        I = (self.Kp / self.Ti) * self.integral
        
        # Derivativo
        D = self.Kp * self.Td * (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        
        u_raw = P + I + D
        u_sat = max(self.u_min, min(self.u_max, u_raw))
        
        # Anti-windup estricto para evitar saturación plana
        if u_raw != u_sat:
            if (u_raw > self.u_max and error > 0) or (u_raw < self.u_min and error < 0):
                self.integral -= error * dt
                
        return u_sat

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
            
            # --- INTERPOLACIÓN TEMPORAL DE CLIMA SEGÚN DELTA T ---
            t_epw_sec = np.array([(c['dt'] - clima_filtrado[0]['dt']).total_seconds() for c in clima_filtrado])
            tdb_epw = np.array([c['tdb'] for c in clima_filtrado])
            twb_epw = np.array([c['twb'] for c in clima_filtrado])
            patm_epw = np.array([c['patm'] for c in clima_filtrado])

            t_total_sec = t_epw_sec[-1]
            time_steps_sec = np.arange(0, t_total_sec + dt_sim_sec, dt_sim_sec)

            tdb_interp = np.interp(time_steps_sec, t_epw_sec, tdb_epw)
            twb_interp = np.interp(time_steps_sec, t_epw_sec, twb_epw)
            patm_interp = np.interp(time_steps_sec, t_epw_sec, patm_epw)

            # Inicialización de Controlador PID
            pid = ControladorPID(
                Kp=self.cfg['kp'], Ti=self.cfg['ti'], Td=self.cfg['td'],
                u_min=self.cfg['speed_min'], u_max=100.0
            )

            m_w_nom = self.cfg['caudal_w_m3h'] * 1000.0 / 3600.0
            m_a_nom = self.cfg['caudal_a_m3s'] * self.cfg['densidad_a']
            NTU_ref = self.cfg['ntu_ref']
            T_set = self.cfg['t_setpoint']

            times, t_out_arr, fan_speed_arr, t_wb_arr, evap_arr, q_mwt_arr = [], [], [], [], [], []

            T_out_actual = self.cfg['t_w_in_nom'] - 5.0
            total_pasos = len(time_steps_sec)

            for idx, sec in enumerate(time_steps_sec):
                if self._is_cancelled:
                    self.cancelado_signal.emit()
                    return

                if idx % max(1, total_pasos // 50) == 0:
                    pct = int(10 + (idx / total_pasos) * 85)
                    dt_actual = clima_filtrado[0]['dt'] + timedelta(seconds=float(sec))
                    self.progreso_signal.emit(f"Simulando {dt_actual.strftime('%d/%m %H:%M')}...", pct)

                tdb_k = tdb_interp[idx]
                twb_k = twb_interp[idx]
                patm_k = patm_interp[idx]

                # Calcular nueva velocidad del ventilador (%) con el PID
                velocidad_pct = pid.calcular(T_set, T_out_actual, dt_sim_sec)
                
                # Caudal de aire instantáneo proporcional a la velocidad
                m_a_actual = m_a_nom * (velocidad_pct / 100.0)

                # Simulación térmica 2D de la torre
                T_out_nueva, evap_kg_s = simular_torre_2d_directo(
                    NTU_ref, self.cfg['t_w_in_nom'], m_w_nom, tdb_k, twb_k, m_a_actual, patm_k,
                    Nx=10, Ny=10
                )

                # Inercia térmica básica para evitar saltos irreales de 1 segundo
                inercia = min(1.0, dt_sim_sec / 300.0) # constante de tiempo ~5 min
                T_out_actual = T_out_actual + inercia * (T_out_nueva - T_out_actual)

                evap_m3h = evap_kg_s * 3600.0 / 1000.0
                Q_MWt = (m_w_nom * cp_w_def * (self.cfg['t_w_in_nom'] - T_out_actual)) / 1000.0

                dt_actual = clima_filtrado[0]['dt'] + timedelta(seconds=float(sec))
                times.append(dt_actual)
                t_out_arr.append(T_out_actual)
                fan_speed_arr.append(velocidad_pct)
                t_wb_arr.append(twb_k)
                evap_arr.append(evap_m3h)
                q_mwt_arr.append(Q_MWt)

            res = {
                'times': times,
                't_out': t_out_arr,
                'fan_speed': fan_speed_arr,
                't_wb': t_wb_arr,
                'evap': evap_arr,
                'q_mwt': q_mwt_arr,
                't_setpoint': T_set
            }

            self.progreso_signal.emit("Finalizado", 100)
            self.exito_signal.emit(res)

        except Exception as e:
            if not self._is_cancelled:
                self.error_signal.emit(str(e))

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
                    
                    twb = tdb * np.arctan(0.151977 * (rh * 100.0 + 8.313659)**0.5) + np.arctan(tdb + rh * 100.0) - np.arctan(rh * 100.0 - 1.676331) + 0.00391838 * (rh * 100.0)**1.5 * np.arctan(0.023101 * rh * 100.0) - 4.686035
                    
                    datos_clima.append({
                        'dt': dt,
                        'tdb': tdb,
                        'twb': twb,
                        'rh': rh,
                        'patm': p_atm
                    })
                except (ValueError, IndexError):
                    continue
    return datos_clima

# ==========================================
# 5. HILOS DE CÁLCULO EN SEGUNDO PLANO
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

            if self.d2 is None: # CALIBRACIÓN 1 PUNTO
                res1 = resolver_punto_operacion(self.d1, N_celdas, self, pct_base=5, pct_span=85)
                if res1 is None or self._is_cancelled:
                    self.cancelado_signal.emit()
                    return
                
                res1['es_dual'] = False
                self.progreso_signal.emit("Completado", 100)
                self.exito_signal.emit(res1)

            else: # AJUSTE DUAL DE 2 PUNTOS (MERKEL c, m)
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
            self.progreso_signal.emit("Cargando archivo climático EPW...", 5)
            clima = leer_archivo_epw(self.cfg['path_epw'])
            if not clima:
                raise ValueError("No se pudieron extraer datos válidos del archivo EPW.")

            f_ini = self.cfg['fecha_inicio']
            f_fin = self.cfg['fecha_fin']
            clima_filtrado = [c for c in clima if f_ini <= c['dt'] <= f_fin]
            
            if not clima_filtrado:
                raise ValueError("El rango de fechas seleccionado no coincide con los datos del archivo EPW.")

            pid = ControladorPID(
                Kp=self.cfg['kp'], Ti=self.cfg['ti'], Td=self.cfg['td'],
                u_min=self.cfg['speed_min'], u_max=100.0
            )

            dt_sim_sec = self.cfg['dt_sim_sec']
            m_w_nom = self.cfg['caudal_w_m3h'] * 1000.0 / 3600.0
            m_a_nom = self.cfg['caudal_a_m3s'] * self.cfg['densidad_a']
            NTU_ref = self.cfg['ntu_ref']
            T_set = self.cfg['t_setpoint']

            times, t_out_arr, fan_speed_arr, t_wb_arr, evap_arr, q_mwt_arr = [], [], [], [], [], []

            total_pasos = len(clima_filtrado)
            for idx, c in enumerate(clima_filtrado):
                if self._is_cancelled:
                    self.cancelado_signal.emit()
                    return

                pct = int(10 + (idx / total_pasos) * 85)
                if idx % 10 == 0:
                    self.progreso_signal.emit(f"Simulando {c['dt'].strftime('%d/%m %H:00')} - Paso {idx+1}/{total_pasos}...", pct)

                T_out_actual = t_out_arr[-1] if t_out_arr else self.cfg['t_w_in_nom'] - 5.0
                velocidad_pct = pid.calcular(T_set, T_out_actual, dt_sim_sec)
                m_a_actual = m_a_nom * (velocidad_pct / 100.0)

                T_out_nueva, evap_kg_s = simular_torre_2d_directo(
                    NTU_ref, self.cfg['t_w_in_nom'], m_w_nom, c['tdb'], c['twb'], m_a_actual, c['patm'],
                    Nx=12, Ny=12
                )

                evap_m3h = evap_kg_s * 3600.0 / 1000.0
                Q_MWt = (m_w_nom * cp_w_def * (self.cfg['t_w_in_nom'] - T_out_nueva)) / 1000.0

                times.append(c['dt'])
                t_out_arr.append(T_out_nueva)
                fan_speed_arr.append(velocidad_pct)
                t_wb_arr.append(c['twb'])
                evap_arr.append(evap_m3h)
                q_mwt_arr.append(Q_MWt)

            res = {
                'times': times,
                't_out': t_out_arr,
                'fan_speed': fan_speed_arr,
                't_wb': t_wb_arr,
                'evap': evap_arr,
                'q_mwt': q_mwt_arr,
                't_setpoint': T_set
            }

            self.progreso_signal.emit("Finalizado", 100)
            self.exito_signal.emit(res)

        except Exception as e:
            if not self._is_cancelled:
                self.error_signal.emit(str(e))

# ==========================================
# 6. DIÁLOGO EMERGENTE 2º PUNTO
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

# ==========================================
# 7. VENTANA EMERGENTE DE SIMULACIÓN DINÁMICA (REDIMENSIONABLE Y MAXIMIZABLE)
# ==========================================
class VentanaSimulacionDinamica(QDialog):
    def __init__(self, parent=None, datos_torre=None):
        super().__init__(parent)
        self.setWindowTitle("Simulación Dinámica Anual / Climática (.EPW) con PID")
        
        # Permitir cambiar tamaño y maximizar/minimizar
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint | Qt.WindowMaximizeButtonHint)
        self.resize(1200, 750)
        self.setMinimumSize(900, 600)
        
        self.datos_torre = datos_torre
        self.res_sim = None

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        panel_cfg = QWidget()
        layout_cfg = QVBoxLayout(panel_cfg)
        layout_cfg.setContentsMargins(5, 5, 5, 5)

        estilo_gb = "QGroupBox { font-size: 11px; font-weight: bold; color: #2C3E50; border: 1px solid #DCDCDC; border-radius: 4px; margin-top: 6px; padding-top: 10px; }"

        # 1. Archivo EPW
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

        # 2. Rango Temporal
        gb_tiempo = QGroupBox("2. Rango Temporal y Paso")
        gb_tiempo.setStyleSheet(estilo_gb)
        grid_tiempo = QGridLayout(gb_tiempo)

        self.date_ini = QDateEdit(QDate(2024, 1, 1))
        self.date_ini.setDisplayFormat("dd/MM/yyyy")
        self.date_fin = QDateEdit(QDate(2024, 1, 7))
        self.date_fin.setDisplayFormat("dd/MM/yyyy")

        self.txt_dt_sim = QLineEdit("3600")
        self.txt_dt_sim.setFont(QFont("Segoe UI", 9))

        grid_tiempo.addWidget(QLabel("Fecha Inicio:"), 0, 0)
        grid_tiempo.addWidget(self.date_ini, 0, 1)
        grid_tiempo.addWidget(QLabel("Fecha Fin:"), 1, 0)
        grid_tiempo.addWidget(self.date_fin, 1, 1)
        grid_tiempo.addWidget(QLabel("Paso Tiempo Δt:"), 2, 0)
        grid_tiempo.addWidget(self.txt_dt_sim, 2, 1)
        grid_tiempo.addWidget(QLabel("seg"), 2, 2)

        layout_cfg.addWidget(gb_tiempo)

        # 3. PID (Valores recomendados para dt=3600s)
        gb_pid = QGroupBox("3. Controlador PID del Ventilador")
        gb_pid.setStyleSheet(estilo_gb)
        grid_pid = QGridLayout(gb_pid)

        self.txt_t_set = QLineEdit("20.6")
        self.txt_kp = QLineEdit("0.8")
        self.txt_ti = QLineEdit("3600")
        self.txt_td = QLineEdit("0")
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

        self.btn_ejecutar = QPushButton("🚀 Ejecutar Simulación Dinámica")
        self.btn_ejecutar.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.btn_ejecutar.setStyleSheet("background-color: #27AE60; color: white; padding: 8px; border-radius: 3px;")
        self.btn_ejecutar.clicked.connect(self.ejecutar_simulacion)
        layout_cfg.addWidget(self.btn_ejecutar)

        layout_cfg.addStretch()

        # PANEL DERECHO: GRÁFICAS
        panel_grafica = QWidget()
        layout_graf = QVBoxLayout(panel_grafica)
        layout_graf.setContentsMargins(5, 5, 5, 5)

        gb_vars = QGroupBox("Variables a Visualizar en la Gráfica Temporal")
        gb_vars.setStyleSheet(estilo_gb)
        layout_chk = QHBoxLayout(gb_vars)

        self.chk_tout = QCheckBox("Temp. Salida Agua (°C)")
        self.chk_tout.setChecked(True)
        self.chk_speed = QCheckBox("Velocidad Ventilador (%)")
        self.chk_speed.setChecked(True)
        self.chk_twb = QCheckBox("Bulbo Húmedo (°C)")
        self.chk_twb.setChecked(True)
        self.chk_q = QCheckBox("Carga Térmica (MWt)")
        self.chk_q.setChecked(True)
        self.chk_evap = QCheckBox("Evaporación (m³/h)")
        self.chk_evap.setChecked(True)

        for chk in [self.chk_tout, self.chk_speed, self.chk_twb, self.chk_q, self.chk_evap]:
            chk.setFont(QFont("Segoe UI", 8))
            chk.stateChanged.connect(self.replot)
            layout_chk.addWidget(chk)

        layout_graf.addWidget(gb_vars)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout_graf.addWidget(self.toolbar)
        layout_graf.addWidget(self.canvas)

        main_layout.addWidget(panel_cfg, stretch=1)
        main_layout.addWidget(panel_grafica, stretch=3)

    def examinar_epw(self):
        """Método que busca el archivo .epw"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Seleccionar archivo climático EPW", 
            "", 
            "Archivos EPW (*.epw);;Todos los archivos (*.*)"
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
                't_setpoint': float(self.txt_t_set.text()),
                'kp': float(self.txt_kp.text()),
                'ti': float(self.txt_ti.text()),
                'td': float(self.txt_td.text()),
                'speed_min': float(self.txt_speed_min.text()),
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

        mostrar_temperaturas = self.chk_tout.isChecked() or self.chk_twb.isChecked()
        mostrar_velocidad = self.chk_speed.isChecked()
        mostrar_carga = self.chk_q.isChecked()
        mostrar_evap = self.chk_evap.isChecked()

        if not (mostrar_temperaturas or mostrar_velocidad or mostrar_carga or mostrar_evap):
            self.canvas.draw()
            return

        usar_panel_inferior = mostrar_carga or mostrar_evap
        
        if usar_panel_inferior and (mostrar_temperaturas or mostrar_velocidad):
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
            if mostrar_temperaturas:
                if self.chk_tout.isChecked():
                    l1, = ax_top.plot(times, self.res_sim['t_out'], color='#C0392B', label='Temp. Agua Salida (°C)', linewidth=1.5)
                    lines.append(l1)
                    l_set, = ax_top.plot(times, [self.res_sim['t_setpoint']]*len(times), color='#C0392B', linestyle='--', alpha=0.6, label='Setpoint Agua')
                    lines.append(l_set)
                if self.chk_twb.isChecked():
                    l2, = ax_top.plot(times, self.res_sim['t_wb'], color='#2980B9', linestyle=':', label='Bulbo Húmedo (°C)')
                    lines.append(l2)
                ax_top.set_ylabel("Temperatura (°C)", color='#222222', fontsize=8.5)
                ax_top.tick_params(labelsize=8)

            if mostrar_velocidad:
                ax_speed = ax_top.twinx() if mostrar_temperaturas else ax_top
                l3, = ax_speed.plot(times, self.res_sim['fan_speed'], color='#27AE60', label='Velocidad Ventilador (%)', linewidth=1.2)
                lines.append(l3)
                ax_speed.set_ylabel("Velocidad (%)", color='#27AE60', fontsize=8.5)
                ax_speed.set_ylim(-5, 105)
                ax_speed.tick_params(labelsize=8)

            labels = [l.get_label() for l in lines]
            ax_top.legend(lines, labels, loc='upper right', fontsize=8, framealpha=0.85)

        if ax_bot is not None:
            lines_bot = []
            if mostrar_carga:
                l_q, = ax_bot.plot(times, self.res_sim['q_mwt'], color='#8E44AD', label='Carga Térmica (MWt)', linewidth=1.4)
                lines_bot.append(l_q)
                ax_bot.set_ylabel("Carga (MWt)", color='#8E44AD', fontsize=8.5)
                ax_bot.tick_params(labelsize=8)

            if mostrar_evap:
                ax_evap = ax_bot.twinx() if mostrar_carga else ax_bot
                l_ev, = ax_evap.plot(times, self.res_sim['evap'], color='#D35400', linestyle='-.', label='Evaporación (m³/h)', linewidth=1.2)
                lines_bot.append(l_ev)
                ax_evap.set_ylabel("Evaporación (m³/h)", color='#D35400', fontsize=8.5)
                ax_evap.tick_params(labelsize=8)

            labels_bot = [l.get_label() for l in lines_bot]
            ax_bot.legend(lines_bot, labels_bot, loc='upper right', fontsize=8, framealpha=0.85)
            ax_bot.set_xlabel("Fecha / Hora", fontsize=8.5)

        if ax_top is not None and ax_bot is None:
            ax_top.set_xlabel("Fecha / Hora", fontsize=8.5)

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
# 9. VENTANA PRINCIPAL DE PyQt5
# ==========================================
class TorreCoolingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemelo Digital 2D - Torre de Enfriamiento (Poppe)")
        self.setGeometry(100, 100, 1180, 750)
        self.ultimo_resultado = None

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

# ==========================================
# 10. PUNTO DE ENTRADA
# ==========================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = TorreCoolingApp()
    window.show()
    sys.exit(app.exec_())