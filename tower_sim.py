"""
Tower simulation engines - Business logic for calibration and dynamic simulation.
Platform-agnostic simulation workers compatible with PyQt5 threads.
"""

import numpy as np
from datetime import datetime, timedelta
from psychro_data import PsicroLUT
from utils import leer_archivo_epw
from core_calc import (
    simular_torre_2d_matriz,
    simular_torre_counterflow_1d,
    resolver_punto_operacion,
    obtener_presion_barometrica,
    humedad_saturacion,
    temp_aire_desde_entalpia,
    CP_WATER_DEFAULT, CP_AIR_DEFAULT, CP_VAPOR_DEFAULT, H_FG0_DEFAULT,
)


# ==========================================
# CONTROLADOR PID (ACCIÓN DIRECTA - REFRIGERACIÓN)
# ==========================================
class ControladorPID:
    def __init__(self, Kp=1.0, Ti=1800.0, Td=0.0, u_min=20.0, u_max=100.0, deadband=0.3, max_rate=5.0):
        self.Kp = float(Kp)
        self.Ti = max(float(Ti), 1.0)
        self.Td = float(Td)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.deadband = max(0.0, float(deadband))
        self.max_rate = float(max_rate)
        self.integral = 0.0
        self.prev_error = None
        self.prev_u = None

    def reset(self, u_inicial=20.0):
        self.integral = 0.0
        self.prev_error = None
        self.prev_u = float(u_inicial)

    def calcular(self, setpoint, medido, dt):
        error_real = float(medido - setpoint)
        if np.isnan(error_real) or np.isinf(error_real):
            error_real = 0.0

        if abs(error_real) <= self.deadband:
            error_calc = 0.0
        else:
            error_calc = error_real

        if self.prev_error is None:
            self.prev_error = error_real

        P = self.Kp * error_calc
        D = self.Kp * self.Td * (error_real - self.prev_error) / dt if dt > 0 else 0.0

        u_tentativo = P + (self.Kp / self.Ti) * self.integral + D
        
        bloquear_integral = False
        if u_tentativo >= self.u_max and error_calc > 0:
            bloquear_integral = True
        elif u_tentativo <= self.u_min and error_calc < 0:
            bloquear_integral = True

        if not bloquear_integral:
            self.integral += error_calc * dt

        I = (self.Kp / self.Ti) * self.integral
        u_unsat = P + I + D
        u_sat = max(self.u_min, min(self.u_max, u_unsat))

        if self.prev_u is not None and self.max_rate > 0:
            delta_u = u_sat - self.prev_u
            if abs(delta_u) > self.max_rate:
                u_sat = self.prev_u + np.sign(delta_u) * self.max_rate

        self.prev_error = error_real
        self.prev_u = u_sat
        return u_sat


# ==========================================
# WORKERS DE CÁLCULO
# ==========================================
try:
    from PyQt5.QtCore import QThread, pyqtSignal
    HAS_PYQT5 = True
except ImportError:
    HAS_PYQT5 = False
    class QThread:
        def __init__(self): pass
        def run(self): pass
    class pyqtSignal:
        def __init__(self, *args): pass
        def emit(self, *args): pass


class CalibracionWorker(QThread if HAS_PYQT5 else object):
    if HAS_PYQT5:
        progreso_signal = pyqtSignal(str, int)
        exito_signal = pyqtSignal(dict)
        error_signal = pyqtSignal(str)
        cancelado_signal = pyqtSignal()

    def __init__(self, datos_input_p1, datos_input_p2=None, tipo_equipo='crossflow'):
        super().__init__()
        self.d1 = datos_input_p1
        self.d2 = datos_input_p2
        self.tipo_equipo = tipo_equipo
        self._is_cancelled = False

    def cancelar(self):
        self._is_cancelled = True

    def run(self):
        try:
            N_celdas = self.d1.get('num_celdas', 6)

            if self.d2 is None:
                if HAS_PYQT5:
                    self.progreso_signal.emit("Calibrando Punto 1...", 50)
                
                res1 = resolver_punto_operacion(
                    self.d1, N_celdas, self, pct_base=5, pct_span=85, tipo_equipo=self.tipo_equipo
                )
                if res1 is None or self._is_cancelled:
                    if HAS_PYQT5:
                        self.cancelado_signal.emit()
                    return
                
                res1['es_dual'] = False
                res1['c_coef'] = None
                res1['m_exp'] = None
                if HAS_PYQT5:
                    self.progreso_signal.emit("Completado", 100)
                    self.exito_signal.emit(res1)
            else:
                if HAS_PYQT5:
                    self.progreso_signal.emit("Calibrando Punto 1...", 5)
                
                res1 = resolver_punto_operacion(
                    self.d1, N_celdas, self, pct_base=5, pct_span=40, tipo_equipo=self.tipo_equipo
                )
                if res1 is None or self._is_cancelled:
                    if HAS_PYQT5:
                        self.cancelado_signal.emit()
                    return

                if HAS_PYQT5:
                    self.progreso_signal.emit("Calibrando Punto 2...", 50)
                
                res2 = resolver_punto_operacion(
                    self.d2, N_celdas, self, pct_base=50, pct_span=40, tipo_equipo=self.tipo_equipo
                )
                if res2 is None or self._is_cancelled:
                    if HAS_PYQT5:
                        self.cancelado_signal.emit()
                    return

                lg1 = float(res1.get('L_G_ratio', 1.0))
                lg2 = float(res2.get('L_G_ratio', 1.0))
                ntu1 = float(res1.get('NTU', 1.0))
                ntu2 = float(res2.get('NTU', 1.0))

                if abs(lg1 - lg2) < 1e-5:
                    m_exp = 0.6
                else:
                    m_exp = float(-np.log(ntu1 / ntu2) / np.log(lg1 / lg2))

                c_coef = float(ntu1 / (lg1 ** (-m_exp)))

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
    if HAS_PYQT5:
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
            if HAS_PYQT5:
                self.progreso_signal.emit("Cargando y procesando archivo climático EPW...", 5)

            año_ref = self.cfg.get('año_referencia', 2021)
            clima = leer_archivo_epw(self.cfg['path_epw'], año_referencia=año_ref)
            if not clima:
                raise ValueError("No se pudieron extraer datos válidos del archivo EPW.")

            f_ini = self.cfg['fecha_inicio']
            f_fin = self.cfg['fecha_fin']
            clima_filtrado = [c for c in clima if f_ini <= c['dt'] <= f_fin]

            if not clima_filtrado:
                raise ValueError("El rango de fechas seleccionado no coincide con el archivo EPW procesado.")

            dt_sim_sec = self.cfg['dt_sim_sec']

            t_epw_sec = np.array([(c['dt'] - clima_filtrado[0]['dt']).total_seconds() for c in clima_filtrado])
            tdb_epw = np.array([c['tdb'] for c in clima_filtrado])
            twb_epw = np.array([c['twb'] for c in clima_filtrado])
            patm_epw = np.array([c['patm'] for c in clima_filtrado])
            uviento_epw = np.array([c['u_viento'] for c in clima_filtrado])

            t_total_sec = t_epw_sec[-1]
            time_steps_sec = np.arange(0, t_total_sec + dt_sim_sec, dt_sim_sec)
            total_pasos = len(time_steps_sec)

            tdb_vec = np.interp(time_steps_sec, t_epw_sec, tdb_epw)
            twb_vec = np.interp(time_steps_sec, t_epw_sec, twb_epw)
            patm_vec = np.interp(time_steps_sec, t_epw_sec, patm_epw)
            uviento_vec = np.interp(time_steps_sec, t_epw_sec, uviento_epw)

            lut = PsicroLUT(T_min=-15.0, T_max=65.0, step=0.1, P_atm=patm_epw[0])

            w_sat_wb_vec = np.array([lut.get_ws_hs(twb_vec[k])[0] for k in range(total_pasos)])
            w_a_in_vec = ((H_FG0_DEFAULT - (CP_WATER_DEFAULT - CP_VAPOR_DEFAULT) * twb_vec) * w_sat_wb_vec - CP_AIR_DEFAULT * (tdb_vec - twb_vec)) / (H_FG0_DEFAULT + CP_VAPOR_DEFAULT * tdb_vec - CP_WATER_DEFAULT * twb_vec)
            h_a_in_vec = CP_AIR_DEFAULT * tdb_vec + w_a_in_vec * (H_FG0_DEFAULT + CP_VAPOR_DEFAULT * tdb_vec)

            # CONTROLADORES PID INDEPENDIENTES
            fan_min_spd = float(self.cfg['speed_min'])
            pump_min_spd = float(self.cfg.get('speed_min_bomba', 50.0))

            pid_fan = ControladorPID(
                Kp=self.cfg['kp'], Ti=self.cfg['ti'], Td=self.cfg['td'],
                u_min=fan_min_spd, u_max=100.0,
                deadband=self.cfg.get('deadband', 0.3),
                max_rate=self.cfg.get('max_rate', 5.0)
            )

            pid_pump = ControladorPID(
                Kp=self.cfg.get('kp_pump', 1.0),
                Ti=self.cfg.get('ti_pump', 1800.0),
                Td=self.cfg.get('td_pump', 0.0),
                u_min=pump_min_spd, u_max=100.0,
                deadband=self.cfg.get('deadband_pump', 0.3),
                max_rate=self.cfg.get('max_rate_pump', 3.0)
            )

            caudal_w_nom_m3h = self.cfg['caudal_w_m3h']
            m_w_nom = caudal_w_nom_m3h * 1000.0 / 3600.0
            m_a_nom = self.cfg['caudal_a_m3s'] * self.cfg['densidad_a']
            NTU_ref = self.cfg['ntu_ref']
            T_set = self.cfg['t_setpoint']

            # Configuración de 2 puntos
            es_dual = self.cfg.get('es_dual', False)
            c_coef = self.cfg.get('c_coef', None)
            m_exp = self.cfg.get('m_exp', None)

            pct_drift = float(self.cfg.get('pct_drift', 0.005))
            coc = float(self.cfg.get('coc', 4.0))

            v_estanque_m3 = self.cfg['vol_estanque_m3']
            m_estanque_kg = max(100.0, v_estanque_m3 * 1000.0)

            p_motor_fan_kw = self.cfg['p_motor_kw']
            eta_fan_glob = max(0.1, min(1.0, self.cfg['eta_fan_pct'] / 100.0))

            p_motor_pump_kw = float(self.cfg.get('p_bomba_kw', 45.0))
            eta_pump_glob = max(0.1, min(1.0, float(self.cfg.get('eta_bomba_pct', 80.0)) / 100.0))

            delta_T_nom = float(self.cfg['t_w_in_nom']) - T_set
            Q_proceso_nominal_kW = m_w_nom * CP_WATER_DEFAULT * delta_T_nom
            tipo_equipo = self.cfg.get('tipo_equipo', 'crossflow')

            tdb_0 = tdb_vec[0]
            twb_0 = twb_vec[0]
            patm_0 = patm_vec[0]
            w_a_in0 = w_a_in_vec[0]
            h_a_in0 = h_a_in_vec[0]

            T_piscina = max(twb_0 + 2.0, T_set)
            pid_fan.reset(u_inicial=fan_min_spd)
            pid_pump.reset(u_inicial=100.0)

            # Warm-up numérico inicial
            for _ in range(30):
                v_fan_0 = pid_fan.calcular(T_set, T_piscina, dt_sim_sec)
                if v_fan_0 <= (fan_min_spd + 0.5):
                    v_pump_0 = pid_pump.calcular(T_set, T_piscina, dt_sim_sec)
                else:
                    v_pump_0 = 100.0
                    pid_pump.reset(u_inicial=100.0)

                m_w_inst0 = max(0.20 * m_w_nom, m_w_nom * (v_pump_0 / 100.0))
                m_a_inst0 = max(0.05 * m_a_nom, m_a_nom * (v_fan_0 / 100.0))

                dT_proc0 = Q_proceso_nominal_kW / (m_w_inst0 * CP_WATER_DEFAULT)
                T_w_in_din0 = T_piscina + dT_proc0

                if es_dual and c_coef is not None and m_exp is not None:
                    lg_0 = m_w_inst0 / m_a_inst0
                    ntu_0 = float(c_coef * (lg_0 ** (-m_exp)))
                else:
                    ntu_0 = NTU_ref

                if tipo_equipo == 'counterflow':
                    T_sal_init, _, _, _, _, _ = simular_torre_counterflow_1d(
                        ntu_0, T_w_in_din0, m_w_inst0, h_a_in0, w_a_in0, m_a_inst0, patm_0, 
                        Nz=16, lut=lut
                    )
                else:
                    T_sal_init, _, _, _, _, _ = simular_torre_2d_matriz(
                        ntu_0, T_w_in_din0, m_w_inst0, h_a_in0, w_a_in0, m_a_inst0, patm_0, 
                        Nx=6, Ny=6, lut=lut
                    )

                T_sal_init = max(twb_0, min(T_w_in_din0, T_sal_init))
                tau_0 = m_estanque_kg / m_w_inst0
                renov_0 = min(0.20, dt_sim_sec / tau_0)
                T_piscina = T_piscina + renov_0 * (T_sal_init - T_piscina)

            times = []
            t_out_arr, t_in_arr = [], []
            fan_speed_arr, pump_speed_arr = [], []
            t_wb_arr, t_db_arr, t_a_out_arr = [], [], []
            evap_arr, q_mwt_arr, niebla_arr = [], [], []
            w_a_out_arr = []
            power_fan_kw_arr, power_pump_kw_arr, power_tot_kw_arr = [], [], []
            agua_total_arr = []

            energia_fan_kwh = 0.0
            energia_pump_kwh = 0.0
            energia_disipada_mwh_t = 0.0
            agua_evap_total_m3 = 0.0
            agua_drift_total_m3 = 0.0
            agua_purga_total_m3 = 0.0

            # Bucle de simulación temporal dinámico
            for idx in range(total_pasos):
                if self._is_cancelled:
                    if HAS_PYQT5:
                        self.cancelado_signal.emit()
                    return

                if HAS_PYQT5 and idx % max(1, total_pasos // 50) == 0:
                    pct = int(10 + (idx / total_pasos) * 85)
                    sec = time_steps_sec[idx]
                    dt_actual = clima_filtrado[0]['dt'] + timedelta(seconds=float(sec))
                    self.progreso_signal.emit(f"Simulando {dt_actual.strftime('%d/%m %H:%M')}...", pct)

                tdb_k = tdb_vec[idx]
                twb_k = twb_vec[idx]
                patm_k = patm_vec[idx]
                w_a_in_k = w_a_in_vec[idx]
                h_a_in_k = h_a_in_vec[idx]

                # LÓGICA SECUENCIAL SPLIT-RANGE CON DOS PID
                vel_fan_k = pid_fan.calcular(T_set, T_piscina, dt_sim_sec)
                
                # Si el ventilador ya está en su velocidad mínima y la piscina sigue más fría que el setpoint:
                if vel_fan_k <= (fan_min_spd + 0.3):
                    vel_pump_k = pid_pump.calcular(T_set, T_piscina, dt_sim_sec)
                else:
                    vel_pump_k = 100.0
                    pid_pump.reset(u_inicial=100.0)

                # Caudales másicos modulantes
                m_a_actual = max(0.05 * m_a_nom, m_a_nom * (vel_fan_k / 100.0))
                m_w_actual = max(0.20 * m_w_nom, m_w_nom * (vel_pump_k / 100.0))
                caudal_w_inst_m3h = caudal_w_nom_m3h * (vel_pump_k / 100.0)

                # Salto térmico del proceso para disipar la carga nominal de diseño
                delta_T_proceso = Q_proceso_nominal_kW / (m_w_actual * CP_WATER_DEFAULT)
                T_w_in_dinamica = T_piscina + delta_T_proceso

                # NTU modulante según L/G instantáneo
                if es_dual and c_coef is not None and m_exp is not None:
                    lg_actual = m_w_actual / m_a_actual
                    ntu_actual_paso = float(c_coef * (lg_actual ** (-m_exp)))
                else:
                    ntu_actual_paso = NTU_ref

                # Potencias eléctricas instantáneas
                u_fan_r = vel_fan_k / 100.0
                p_fan_kw = (p_motor_fan_kw / eta_fan_glob) * (u_fan_r ** 3)

                u_pump_r = vel_pump_k / 100.0
                p_pump_kw = (p_motor_pump_kw / eta_pump_glob) * (u_pump_r ** 3)
                p_tot_kw = p_fan_kw + p_pump_kw

                dt_horas = dt_sim_sec / 3600.0
                energia_fan_kwh += p_fan_kw * dt_horas
                energia_pump_kwh += p_pump_kw * dt_horas

                # Cálculo térmico de la torre
                if tipo_equipo == 'counterflow':
                    T_salida_inst, evap_kg_s, _, Matriz_w_a, Matriz_T_a, Matriz_niebla = simular_torre_counterflow_1d(
                        ntu_actual_paso, T_w_in_dinamica, m_w_actual, h_a_in_k, w_a_in_k, m_a_actual, patm_k,
                        Nz=16, lut=lut
                    )
                else:
                    T_salida_inst, evap_kg_s, _, Matriz_w_a, Matriz_T_a, Matriz_niebla = simular_torre_2d_matriz(
                        ntu_actual_paso, T_w_in_dinamica, m_w_actual, h_a_in_k, w_a_in_k, m_a_actual, patm_k,
                        Nx=6, Ny=6, lut=lut
                    )

                T_salida_inst = max(twb_k, min(T_w_in_dinamica, T_salida_inst))
                T_a_out_prom = float(np.mean(Matriz_T_a[:, -1]))
                w_a_out_prom = float(np.mean(Matriz_w_a[:, -1]))
                hay_niebla_paso = bool(np.any(Matriz_niebla))

                # Balance térmico del estanque de acumulación
                tau_sec = m_estanque_kg / m_w_actual
                fraccion_renov = min(0.20, dt_sim_sec / tau_sec)
                T_piscina = T_piscina + fraccion_renov * (T_salida_inst - T_piscina)
                T_piscina = max(twb_k, min(T_w_in_dinamica, T_piscina))

                # Balance de agua
                drift_m3h = caudal_w_inst_m3h * (pct_drift / 100.0)
                evap_m3h = max(0.0, evap_kg_s * 3600.0 / 1000.0)
                purga_m3h = max(0.0, (evap_m3h - drift_m3h * (coc - 1.0)) / (coc - 1.0))

                agua_evap_total_m3 += evap_m3h * dt_horas
                agua_drift_total_m3 += drift_m3h * dt_horas
                agua_purga_total_m3 += purga_m3h * dt_horas

                Q_MWt = (m_w_actual * CP_WATER_DEFAULT * delta_T_proceso) / 1000.0
                energia_disipada_mwh_t += Q_MWt * dt_horas

                dt_actual = clima_filtrado[0]['dt'] + timedelta(seconds=float(time_steps_sec[idx]))
                times.append(dt_actual)
                t_out_arr.append(T_piscina)
                t_in_arr.append(T_w_in_dinamica)
                fan_speed_arr.append(vel_fan_k)
                pump_speed_arr.append(vel_pump_k)
                t_wb_arr.append(twb_k)
                t_db_arr.append(tdb_k)
                t_a_out_arr.append(T_a_out_prom)
                w_a_out_arr.append(w_a_out_prom)
                evap_arr.append(evap_m3h)
                q_mwt_arr.append(Q_MWt)
                niebla_arr.append(hay_niebla_paso)
                power_fan_kw_arr.append(p_fan_kw)
                power_pump_kw_arr.append(p_pump_kw)
                power_tot_kw_arr.append(p_tot_kw)
                agua_total_arr.append(evap_m3h + drift_m3h + purga_m3h)

            energia_total_kwh = energia_fan_kwh + energia_pump_kwh
            agua_total_makeup_m3 = agua_evap_total_m3 + agua_drift_total_m3 + agua_purga_total_m3
            energia_disipada_kwh_t = energia_disipada_mwh_t * 1000.0

            cop_torre = (energia_disipada_kwh_t / energia_total_kwh) if energia_total_kwh > 0 else 0.0
            intensidad_agua_m3_mwh = (agua_total_makeup_m3 / energia_disipada_mwh_t) if energia_disipada_mwh_t > 0 else 0.0
            intensidad_agua_m3_kwhe = (agua_total_makeup_m3 / energia_total_kwh) if energia_total_kwh > 0 else 0.0

            resultado = {
                'times': times,
                't_out': t_out_arr,
                't_in': t_in_arr,
                'fan_speed': fan_speed_arr,
                'pump_speed': pump_speed_arr,
                't_wb': t_wb_arr,
                't_db': t_db_arr,
                't_a_out': t_a_out_arr,
                'w_a_out': w_a_out_arr,
                'evap': evap_arr,
                'q_mwt': q_mwt_arr,
                'niebla': niebla_arr,
                'power_fan_kw': power_fan_kw_arr,
                'power_pump_kw': power_pump_kw_arr,
                'power_kw': power_tot_kw_arr,
                'agua_total_m3h': agua_total_arr,
                'dt_sim_sec': dt_sim_sec,
                'u_viento_vec': uviento_vec,
                'energia_fan_kwh': energia_fan_kwh,
                'energia_pump_kwh': energia_pump_kwh,
                'energia_total_kwh': energia_total_kwh,
                'energia_disipada_mwh_t': energia_disipada_mwh_t,
                'agua_evap_m3': agua_evap_total_m3,
                'agua_drift_m3': agua_drift_total_m3,
                'agua_purga_m3': agua_purga_total_m3,
                'agua_total_makeup_m3': agua_total_makeup_m3,
                'cop_torre': cop_torre,
                'intensidad_agua_m3_mwh': intensidad_agua_m3_mwh,
                'intensidad_agua_m3_kwhe': intensidad_agua_m3_kwhe,
                'vel_promedio_pct': float(np.mean(fan_speed_arr)),
                'vel_promedio_pump_pct': float(np.mean(pump_speed_arr)),
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