"""
Tower simulation engines - Business logic for calibration and dynamic simulation.

NOTE: These classes initially inherit from QThread for compatibility with tower_app_21.py.
For web/mobile platforms, extract the run() method logic into pure functions.
"""

import numpy as np
from datetime import datetime, timedelta
from core_calc import (
    simular_torre_2d_matriz, resolver_punto_operacion, obtener_presion_barometrica,
    humedad_saturacion, temp_aire_desde_entalpia,
    CP_WATER_DEFAULT, CP_AIR_DEFAULT, CP_VAPOR_DEFAULT, H_FG0_DEFAULT,
)
from psychro_data import PsicroLUT
from utils import leer_archivo_epw

# ==========================================
# PID CONTROLLER
# ==========================================
class ControladorPID:
    """PID controller for fan speed regulation.
    
    Uses proportional, integral, and derivative terms to control output (0-100%).
    """
    
    def __init__(self, Kp=4.0, Ti=300.0, Td=5.0, u_min=0.0, u_max=100.0):
        """Initialize PID controller.
        
        Args:
            Kp: Proportional gain (default: 4.0)
            Ti: Integral time constant in seconds (default: 300.0)
            Td: Derivative time constant in seconds (default: 5.0)
            u_min: Output minimum (default: 0.0)
            u_max: Output maximum (default: 100.0)
        """
        self.Kp = float(Kp)
        self.Ti = max(float(Ti), 1.0)
        self.Td = float(Td)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.integral = 0.0
        self.prev_error = None

    def calcular(self, setpoint, medido, dt):
        """Calculate PID output.
        
        Args:
            setpoint: Target value
            medido: Measured value
            dt: Time step in seconds
            
        Returns:
            Saturated output (u_min to u_max)
        """
        error = float(medido - setpoint)
        if np.isnan(error) or np.isinf(error):
            error = 0.0

        if self.prev_error is None:
            self.prev_error = error

        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        if np.isnan(self.integral) or np.isinf(self.integral):
            self.integral = 0.0
        I = (self.Kp / self.Ti) * self.integral
        
        # Derivative term
        D = self.Kp * self.Td * (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        
        # Unsaturated output
        u_raw = P + I + D
        if np.isnan(u_raw) or np.isinf(u_raw):
            u_raw = self.u_min

        # Saturate
        u_sat = max(self.u_min, min(self.u_max, u_raw))
        
        # Anti-windup: if saturated, reduce integral accumulation
        if u_raw != u_sat:
            self.integral -= error * dt
            if np.isnan(self.integral) or np.isinf(self.integral):
                self.integral = 0.0
            
        return u_sat

# ==========================================
# SIMULATION ENGINES (with QThread for PyQt5 compatibility)
# ==========================================

try:
    from PyQt5.QtCore import QThread, pyqtSignal
    HAS_PYQT5 = True
except ImportError:
    HAS_PYQT5 = False
    # Fallback: Create dummy QThread class for non-PyQt environments
    class QThread:
        def __init__(self):
            pass
        def run(self):
            pass
    class pyqtSignal:
        def __init__(self, *args):
            pass
        def emit(self, *args):
            pass

class CalibracionWorker(QThread if HAS_PYQT5 else object):
    """Calibration worker - Finds tower NTU values for given operating points.
    
    Performs single or dual-point calibration to determine tower characteristics.
    Inherits from QThread for PyQt5 compatibility.
    """
    
    if HAS_PYQT5:
        progreso_signal = pyqtSignal(str, int)
        exito_signal = pyqtSignal(dict)
        error_signal = pyqtSignal(str)
        cancelado_signal = pyqtSignal()

    def __init__(self, datos_input_p1, datos_input_p2=None):
        """Initialize calibration worker.
        
        Args:
            datos_input_p1: Dictionary with point 1 data (temp, flow, humidity, etc.)
            datos_input_p2: Optional dictionary with point 2 data for dual-point calibration
        """
        super().__init__()
        self.d1 = datos_input_p1
        self.d2 = datos_input_p2
        self._is_cancelled = False

    def cancelar(self):
        """Request cancellation."""
        self._is_cancelled = True

    def run(self):
        """Execute calibration (called by QThread.start())."""
        try:
            N_celdas = self.d1.get('num_celdas', 6)

            if self.d2 is None:
                # Single-point calibration
                if HAS_PYQT5:
                    self.progreso_signal.emit("Calibrando Punto 1...", 50)
                
                res1 = resolver_punto_operacion(self.d1, N_celdas, self, pct_base=5, pct_span=85)
                
                if res1 is None or self._is_cancelled:
                    if HAS_PYQT5:
                        self.cancelado_signal.emit()
                    return
                
                res1['es_dual'] = False
                
                if HAS_PYQT5:
                    self.progreso_signal.emit("Completado", 100)
                    self.exito_signal.emit(res1)

            else:
                # Dual-point calibration
                if HAS_PYQT5:
                    self.progreso_signal.emit("Calibrando Punto 1...", 5)
                
                res1 = resolver_punto_operacion(self.d1, N_celdas, self, pct_base=5, pct_span=40)
                if res1 is None or self._is_cancelled:
                    if HAS_PYQT5:
                        self.cancelado_signal.emit()
                    return

                if HAS_PYQT5:
                    self.progreso_signal.emit("Calibrando Punto 2...", 50)
                
                res2 = resolver_punto_operacion(self.d2, N_celdas, self, pct_base=50, pct_span=40)
                if res2 is None or self._is_cancelled:
                    if HAS_PYQT5:
                        self.cancelado_signal.emit()
                    return

                # Calculate characteristic curve coefficients
                lg1 = res1.get('L_G_ratio', 1.0)
                lg2 = res2.get('L_G_ratio', 1.0)
                ntu1 = res1.get('NTU', 1.0)
                ntu2 = res2.get('NTU', 1.0)

                if abs(lg1 - lg2) < 1e-5:
                    m_exp = 0.6
                else:
                    m_exp = -np.log(ntu1 / ntu2) / np.log(lg1 / lg2)

                c_coef = ntu1 / (lg1 ** (-m_exp))

                res1['es_dual'] = True
                res1['c_coef'] = c_coef
                res1['m_exp'] = m_exp
                res1['p2'] = res2

                if HAS_PYQT5:
                    self.progreso_signal.emit("Ajuste de 2 puntos completado", 100)
                    self.exito_signal.emit(res1)

        except Exception as e:
            if not self._is_cancelled and HAS_PYQT5:
                self.error_signal.emit(str(e))

class SimularDinamicaWorker(QThread if HAS_PYQT5 else object):
    """Dynamic simulation worker - Simulates tower over climate profile with time steps.
    
    Reads EPW climate file, runs transient simulation with PID control,
    tracks evaporation, drift, and fog formation over simulation period.
    Inherits from QThread for PyQt5 compatibility.
    """
    
    if HAS_PYQT5:
        progreso_signal = pyqtSignal(str, int)
        exito_signal = pyqtSignal(dict)
        error_signal = pyqtSignal(str)
        cancelado_signal = pyqtSignal()

    def __init__(self, config_sim):
        """Initialize dynamic simulation worker.
        
        Args:
            config_sim: Dictionary with simulation configuration
        """
        super().__init__()
        self.cfg = config_sim
        self._is_cancelled = False

    def cancelar(self):
        """Request cancellation."""
        self._is_cancelled = True

    def run(self):
        """Execute dynamic simulation (called by QThread.start())."""
        try:
            if HAS_PYQT5:
                self.progreso_signal.emit("Cargando y procesando archivo climático EPW...", 5)

            clima = leer_archivo_epw(self.cfg['path_epw'])
            if not clima:
                raise ValueError("No se pudieron extraer datos válidos del archivo EPW.")

            # Normalize to a single canonical year if requested
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

            if HAS_PYQT5:
                self.progreso_signal.emit("Pre-interpolando vectores climáticos con NumPy...", 7)
            tdb_vec = np.interp(time_steps_sec, t_epw_sec, tdb_epw)
            twb_vec = np.interp(time_steps_sec, t_epw_sec, twb_epw)
            patm_vec = np.interp(time_steps_sec, t_epw_sec, patm_epw)
            uviento_vec = np.interp(time_steps_sec, t_epw_sec, uviento_epw)

            if HAS_PYQT5:
                self.progreso_signal.emit("Inicializando Tabla Psicrométrica Fast-LUT...", 9)
            lut = PsicroLUT(T_min=-15.0, T_max=65.0, step=0.1, P_atm=patm_epw[0])

            w_sat_wb_vec = np.array([lut.get_ws_hs(twb_vec[k])[0] for k in range(total_pasos)])
            w_a_in_vec = ((H_FG0_DEFAULT - (CP_WATER_DEFAULT - CP_VAPOR_DEFAULT) * twb_vec) * w_sat_wb_vec - CP_AIR_DEFAULT * (tdb_vec - twb_vec)) / (H_FG0_DEFAULT + CP_VAPOR_DEFAULT * tdb_vec - CP_WATER_DEFAULT * twb_vec)
            h_a_in_vec = CP_AIR_DEFAULT * tdb_vec + w_a_in_vec * (H_FG0_DEFAULT + CP_VAPOR_DEFAULT * tdb_vec)

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
            w_a_out_arr = []
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
                    if HAS_PYQT5:
                        self.cancelado_signal.emit()
                    return

                if idx % max(1, total_pasos // 50) == 0:
                    pct = int(10 + (idx / total_pasos) * 85)
                    sec = time_steps_sec[idx]
                    dt_actual = clima_filtrado[0]['dt'] + timedelta(seconds=float(sec))
                    if HAS_PYQT5:
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

                T_salida_inst, evap_kg_s, _, Matriz_w_a, Matriz_T_a, Matriz_niebla = simular_torre_2d_matriz(
                    NTU_ref, T_w_in_dinamica, m_w_nom, h_a_in_k, w_a_in_k, m_a_actual, patm_k,
                    Nx=6, Ny=6, lut=lut
                )

                T_salida_inst = max(twb_k, min(T_w_in_dinamica, T_salida_inst))
                T_a_out_prom = np.mean(Matriz_T_a[:, -1])
                w_a_out_prom = np.mean(Matriz_w_a[:, -1])
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

                Q_MWt = (m_w_nom * CP_WATER_DEFAULT * delta_T_proceso) / 1000.0
                energia_disipada_mwh_t += Q_MWt * dt_horas

                dt_actual = clima_filtrado[0]['dt'] + timedelta(seconds=float(time_steps_sec[idx]))
                times.append(dt_actual)
                t_out_arr.append(T_piscina)
                t_in_arr.append(T_w_in_dinamica)
                fan_speed_arr.append(velocidad_pct)
                t_wb_arr.append(twb_k)
                t_db_arr.append(tdb_k)
                t_a_out_arr.append(T_a_out_prom)
                w_a_out_arr.append(w_a_out_prom)
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

            resultado = {
                'times': times,
                't_out': t_out_arr,
                't_in': t_in_arr,
                'fan_speed': fan_speed_arr,
                't_wb': t_wb_arr,
                't_db': t_db_arr,
                't_a_out': t_a_out_arr,
                'w_a_out': w_a_out_arr,
                'evap': evap_arr,
                'q_mwt': q_mwt_arr,
                'niebla': niebla_arr,
                'power_kw': power_kw_arr,
                'u_viento_vec': uviento_vec,
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
                'p_motor_kw': self.cfg['p_motor_kw'],
            }

            if HAS_PYQT5:
                self.progreso_signal.emit("Finalizado", 100)
                self.exito_signal.emit(resultado)

        except Exception as e:
            if not self._is_cancelled and HAS_PYQT5:
                self.error_signal.emit(str(e))
