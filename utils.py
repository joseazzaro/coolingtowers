"""
Utility functions - Platform independent helpers for parsing, formatting, and translation.
"""

import csv
from datetime import datetime, timedelta
import numpy as np

# ==========================================
# NUMERIC PARSING AND FORMATTING
# ==========================================
def parse_float_local(text):
    """Parse float from text with local decimal separator support.
    
    Handles both '.' and ',' as decimal separators.
    
    Args:
        text: Text to parse
        
    Returns:
        float value
    """
    return float(text.replace(',', '.'))

def conectar_formato_precision(txt_widget, precision=1):
    """Format numeric input widget to fixed precision on focus loss (PyQt5 helper).
    
    Args:
        txt_widget: QLineEdit widget
        precision: Number of decimal places (default: 1)
    """
    def _formatear():
        try:
            val = parse_float_local(txt_widget.text())
            txt_widget.setText(f"{val:.{max(1, precision)}f}")
        except ValueError:
            pass
    
    txt_widget.editingFinished.connect(_formatear)

# ==========================================
# EPW FILE PARSING
# ==========================================
def leer_archivo_epw(path_epw):
    """Parse EPW (EnergyPlus Weather) file robustly.
    
    Detects data rows by checking if the first four columns are plausible
    year/month/day/hour values (rather than relying on a fixed header size).
    Estimates wet-bulb temperature from dry-bulb and relative humidity when
    no direct wet-bulb column is present.
    
    Args:
        path_epw: Path to EPW file
        
    Returns:
        List of dictionaries with hourly climate data, or None on failure:
        - dt: datetime object
        - tdb: Dry-bulb temperature (°C)
        - twb: Wet-bulb temperature (°C)
        - rh: Relative humidity (0-1 fraction)
        - patm: Atmospheric pressure (Pa)
        - u_viento: Wind speed (m/s)
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

    try:
        with open(path_epw, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 4:
                    continue

                try:
                    row0 = row[0].lstrip('\ufeff').strip()
                except Exception:
                    row0 = str(row[0]).strip()

                anio = _safe_int(row0)
                mes = _safe_int(row[1])
                dia = _safe_int(row[2])
                hora = _safe_int(row[3])

                # Only accept rows that look like data (plausible ranges)
                if anio is None or mes is None or dia is None or hora is None:
                    continue
                if not (1900 <= anio <= 2100 and 1 <= mes <= 12 and 1 <= dia <= 31 and 1 <= hora <= 24):
                    continue

                # EPW hour is 1..24, convert to 0..23
                hora_idx = max(0, min(23, hora - 1))

                try:
                    dt = datetime(anio, mes, dia, hora_idx)
                except Exception:
                    continue

                # Typical EPW columns (use reasonable defaults if missing)
                tdb = _safe_float(row[6], None) if len(row) > 6 else None
                rh_raw = _safe_float(row[8], None) if len(row) > 8 else None
                patm = _safe_float(row[9], None) if len(row) > 9 else None

                if tdb is None or rh_raw is None:
                    continue

                # RH in EPW is a percentage (0..100)
                rh = rh_raw / 100.0

                u_viento = _safe_float(row[21], None) if len(row) > 21 else None
                if u_viento is None:
                    u_viento = _safe_float(row[20], 3.5) if len(row) > 20 else 3.5

                # Fast wet-bulb approximation from dry-bulb and RH
                try:
                    twb = (float(tdb) * np.arctan(0.151977 * (rh * 100.0 + 8.313659) ** 0.5)
                           + np.arctan(float(tdb) + rh * 100.0) - np.arctan(rh * 100.0 - 1.676331)
                           + 0.00391838 * (rh * 100.0) ** 1.5 * np.arctan(0.023101 * rh * 100.0) - 4.686035)
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
    except FileNotFoundError:
        return None
    except Exception:
        return None

    return datos_clima if datos_clima else None

def obtener_rango_epw(datos_clima):
    """Extract date range from climate data.
    
    Args:
        datos_clima: List of climate data dictionaries (from leer_archivo_epw)
        
    Returns:
        Tuple of (fecha_min, fecha_max) as datetime objects, or (None, None) if empty
    """
    if not datos_clima:
        return None, None
    
    fechas = [d['dt'] for d in datos_clima if 'dt' in d]
    if not fechas:
        return None, None
    
    return min(fechas), max(fechas)

def detectar_multianio_epw(datos_clima):
    """Detect if EPW contains multiple years (TMYx composite).
    
    Args:
        datos_clima: List of climate data dictionaries
        
    Returns:
        List of unique years, or None if single year
    """
    if not datos_clima:
        return None
    
    años = set()
    for d in datos_clima:
        if 'dt' in d:
            años.add(d['dt'].year)
    
    return sorted(list(años)) if len(años) > 1 else None

def normalizar_epw_a_año_canonico(datos_clima, año_canonico=2017):
    """Remap EPW data from multiple years to a single canonical year.
    
    Used for TMYx/composite files to create a single-year representative dataset.
    
    Args:
        datos_clima: List of climate data dictionaries (from leer_archivo_epw)
        año_canonico: Target year for all records (default: 2017)
        
    Returns:
        List of climate data remapped to canonical year
    """
    resultado = []
    for d in datos_clima:
        d_copia = d.copy()
        if 'dt' in d_copia:
            dt_original = d_copia['dt']
            dt_nuevo = dt_original.replace(year=año_canonico)
            d_copia['dt'] = dt_nuevo
        resultado.append(d_copia)
    return resultado

# ==========================================
# TRANSLATION SYSTEM
# ==========================================
TRADUCCIONES = {
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
        'plot_saturation': "Saturación (ws)",
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
        'accion_ver_psicrometrico': "Ver Carta Psicrométrica...",
        'tip_ver_psicrometrico': "Mostrar la evolución en la carta psicrométrica con control por tiempo",
        'accion_ver_duracion': "Ver Curva de Duración Acumulada...",
        'tip_ver_duracion': "Mostrar la curva de duración acumulada de consumo de agua, electricidad y carga térmica",
        'dur_title': "Curva de Duración Acumulada",
        'dur_chk_water': "Consumo de Agua (m³/h)",
        'dur_chk_power': "Consumo Eléctrico (kW)",
        'dur_chk_thermal': "Carga Térmica (MWt)",
        'dur_xlabel': "Horas de Operación Acumuladas (h)",
        'dur_ylabel_water': "Consumo de Agua (m³/h)",
        'dur_ylabel_power': "Potencia Eléctrica (kW)",
        'dur_ylabel_thermal': "Carga Térmica (MWt)",
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
        'plot_saturation': "Saturation (ws)",
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
        'plot_tdb': "Dry Bulb Tdb (°C)",
        'plot_setpoint': "Setpoint",
        'mapa2d_cbar_wa': "Humidity Ratio (g vapor / kg air)",
        'mapa2d_cbar_ta': "Air Dry Bulb Temp. (°C)",
        'mapa2d_zona_niebla': "Fog Zone",
        'mapa2d_frente_condensacion': "Condensation Front",
        'mapa2d_titulo': "2D Map ({n}x{n}): {capa}   (NTU = {ntu:.4f})  [{motor}]\nRoof Inlet: {tin:.1f} °C   |   Mixed Basin: {tsal:.2f} °C",
        'mapa2d_xlabel': "Ambient Air Inlet   →   Air Flow Direction   →   Outlet",
        'mapa2d_ylabel': "← Water Fall (Roof to Basin) →",
        'accion_ver_psicrometrico': "View Psychrometric Chart...",
        'tip_ver_psicrometrico': "Show simulation evolution in psychrometric chart with time slider",
        'accion_ver_duracion': "View Cumulative Duration Curve...",
        'tip_ver_duracion': "Show the cumulative load duration curve for water, electricity, and thermal load",
        'dur_title': "Cumulative Duration Curve",
        'dur_chk_water': "Water Consumption (m³/h)",
        'dur_chk_power': "Electricity Consumption (kW)",
        'dur_chk_thermal': "Thermal Load (MWt)",
        'dur_xlabel': "Accumulated Operating Hours (h)",
        'dur_ylabel_water': "Water Consumption (m³/h)",
        'dur_ylabel_power': "Electrical Power (kW)",
        'dur_ylabel_thermal': "Thermal Load (MWt)",
    },
}

def traducir(idioma, key, **kwargs):
    """Translate key to specified language with optional parameter substitution.
    
    Args:
        idioma: Language code ('es', 'en', etc.)
        key: Translation key
        **kwargs: Parameters for format string substitution
        
    Returns:
        Translated string, or key itself if not found
    """
    if idioma not in TRADUCCIONES:
        idioma = 'en'
    
    texto = TRADUCCIONES[idioma].get(key, key)
    
    try:
        return texto.format(**kwargs)
    except (KeyError, ValueError):
        return texto
