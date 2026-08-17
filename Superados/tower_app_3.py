import sys
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
    QSplitter, QMessageBox, QComboBox, QProgressDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QLocale
from PyQt5.QtGui import QFont, QDoubleValidator, QIntValidator

# Importaciones de Matplotlib para PyQt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==========================================
# 1. FUNCIONES TERMODINÁMICAS Y PSICROMETRÍA (CoolProp / ASHRAE)
# ==========================================
cp_w_def = 4.184     # kJ/kg.K
cp_a_def = 1.006     # kJ/kg.K
cp_v_def = 1.86      # kJ/kg.K
h_fg0_def = 2501.0   # kJ/kg

def obtener_presion_barometrica(altitud_m):
    P0 = 101325.0  # Pa
    return P0 * (1.0 - 0.6875e-5 * altitud_m)**5.2561

def cp_agua_local(T_celcius):
    """Calor específico del agua en kJ/kg.K dinámico."""
    if HAS_COOLPROP:
        try:
            return CP.PropsSI('C', 'T', T_celcius + 273.15, 'P', 101325, 'Water') / 1000.0
        except Exception:
            pass
    return cp_w_def

def humedad_saturacion(T, P_atm=101325.0):
    """Humedad de saturación (kg/kg) usando CoolProp o ASHRAE."""
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
    """Entalpía de saturación (kJ/kg) dinámicamente ajustada."""
    if HAS_COOLPROP:
        try:
            return CP.HAPropsSI('H', 'T', T + 273.15, 'W', w_sat, 'P', P_atm) / 1000.0
        except Exception:
            pass
    return cp_a_def * T + w_sat * (h_fg0_def + cp_v_def * T)

def temp_aire_desde_entalpia(h_a, w_a, P_atm=101325.0):
    """Calcula T (°C) a partir de entalpía h (kJ/kg) y humedad w (kg/kg)."""
    if HAS_COOLPROP:
        try:
            T_kelvin = CP.HAPropsSI('T', 'H', h_a * 1000.0, 'W', w_a, 'P', P_atm)
            return T_kelvin - 273.15
        except Exception:
            pass
    return (h_a - w_a * h_fg0_def) / (cp_a_def + w_a * cp_v_def)

# ==========================================
# 2. MOTOR DE SIMULACIÓN POPPE 2D
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

# ==========================================
# 3. HILO DE CÁLCULO CON INTERRUPCIÓN SEGURA
# ==========================================
class CalibracionWorker(QThread):
    progreso_signal = pyqtSignal(str, int)  # Mensaje, porcentaje (0-100)
    exito_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    cancelado_signal = pyqtSignal()

    def __init__(self, datos_input):
        super().__init__()
        self.d = datos_input
        self._is_cancelled = False

    def cancelar(self):
        """Solicita la interrupción limpia del proceso."""
        self._is_cancelled = True

    def run(self):
        try:
            self.progreso_signal.emit("Calculando propiedades psicrométricas iniciales...", 5)
            P_atm = obtener_presion_barometrica(self.d['altitud'])
            
            m_w_total = self.d['caudal_w'] * 1000.0 / 3600.0 
            m_a_total = self.d['caudal_a'] * self.d['densidad_a'] 
            
            T_db = self.d['T_db_in']
            T_wb = self.d['T_wb_in']
            N_celdas = self.d['num_celdas']
            
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
                # CORRECCIÓN AQUÍ: Usamos cp_v_def en lugar de cp_v no definida
                w_a_in = ((h_fg0_def - (cp_w_def - cp_v_def) * T_wb) * w_sat_wb - cp_a_def * (T_db - T_wb)) / (h_fg0_def + cp_v_def * T_db - cp_w_def * T_wb)
                h_a_in = cp_a_def * T_db + w_a_in * (h_fg0_def + cp_v_def * T_db)

            num_puntos = 30
            ntu_puntos = np.linspace(0.1, 10.0, num_puntos)
            errores = []

            for idx, ntu_val in enumerate(ntu_puntos):
                if self._is_cancelled:
                    self.cancelado_signal.emit()
                    return

                pct = int(10 + (idx / num_puntos) * 60)
                self.progreso_signal.emit(f"Evaluando matriz 2D ({N_celdas}x{N_celdas}) - Paso {idx+1}/{num_puntos}...", pct)
                
                T_calc, _, _, _, _, _ = simular_torre_2d_matriz(
                    ntu_val, self.d['T_w_in'], m_w_total, h_a_in, w_a_in, m_a_total, P_atm,
                    Nx=N_celdas, Ny=N_celdas
                )
                errores.append(T_calc - self.d['T_w_out_target'])

            if self._is_cancelled:
                self.cancelado_signal.emit()
                return

            self.progreso_signal.emit("Ajustando raíz de NTU exacta (Brent/Secant)...", 75)

            def objetivo_ntu(NTU_guess):
                T_calc, _, _, _, _, _ = simular_torre_2d_matriz(
                    NTU_guess, self.d['T_w_in'], m_w_total, h_a_in, w_a_in, m_a_total, P_atm,
                    Nx=N_celdas, Ny=N_celdas
                )
                return T_calc - self.d['T_w_out_target']

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

            if self._is_cancelled:
                self.cancelado_signal.emit()
                return

            self.progreso_signal.emit(f"Generando matriz final 2D ({N_celdas}x{N_celdas})...", 90)

            T_sal, evap_kg, Matriz_T_w, Matriz_w_a, Matriz_T_a, Matriz_niebla = simular_torre_2d_matriz(
                NTU_calibrado, self.d['T_w_in'], m_w_total, h_a_in, w_a_in, m_a_total, P_atm,
                Nx=N_celdas, Ny=N_celdas
            )

            evap_m3h = evap_kg * 3600.0 / 1000.0
            pct_evap = (evap_m3h / self.d['caudal_w']) * 100.0
            range_w = self.d['T_w_in'] - T_sal
            approach_w = T_sal - T_wb
            
            cp_medio = cp_agua_local((self.d['T_w_in'] + T_sal) / 2.0)
            Q_kW = m_w_total * cp_medio * range_w
            Q_MWt = Q_kW / 1000.0
            Q_TR = Q_kW / 3.517
            
            L_G_ratio = m_w_total / m_a_total

            resultado = {
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
                'T_w_in': self.d['T_w_in'],
                'num_celdas': N_celdas
            }

            self.progreso_signal.emit("Completado", 100)
            self.exito_signal.emit(resultado)

        except Exception as e:
            if not self._is_cancelled:
                self.error_signal.emit(str(e))

# ==========================================
# 4. CANVA DE MATPLOTLIB
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
# 5. VENTANA PRINCIPAL DE PyQt5
# ==========================================
class TorreCoolingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemelo Digital 2D - Torre de Enfriamiento (Poppe)")
        self.setGeometry(100, 100, 1180, 750)
        self.ultimo_resultado = None

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --------------------------------------------------
        # SECCIÓN IZQUIERDA: INPUTS SOBRIOS + CONFIG CELDAS
        # --------------------------------------------------
        panel_izquierdo = QWidget()
        layout_izq = QVBoxLayout(panel_izquierdo)
        layout_izq.setContentsMargins(10, 10, 10, 10)
        layout_izq.setSpacing(10)

        estilo_gb = """
            QGroupBox {
                font-size: 11px;
                font-weight: bold;
                color: #2C3E50;
                border: 1px solid #DCDCDC;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """

        # Grupo 1: Datos de Agua
        gb_agua = QGroupBox("Parámetros del Agua (Rating)")
        gb_agua.setStyleSheet(estilo_gb)
        grid_agua = QGridLayout()
        grid_agua.setVerticalSpacing(4)

        self.txt_Tw_in = self.crear_input("31.7", "°C", grid_agua, 0, "Temp. Entrada Agua (T_w1):")
        self.txt_Tw_out = self.crear_input("20.6", "°C", grid_agua, 1, "Temp. Salida Deseada (T_w2):")
        self.txt_caudal_w = self.crear_input("1174.0", "m³/h", grid_agua, 2, "Caudal Volumétrico Agua:")
        gb_agua.setLayout(grid_agua)
        layout_izq.addWidget(gb_agua)

        # Grupo 2: Datos del Aire, Clima y Resolución
        gb_aire = QGroupBox("Condiciones Ambientales y Malla")
        gb_aire.setStyleSheet(estilo_gb)
        grid_aire = QGridLayout()
        grid_aire.setVerticalSpacing(4)

        self.txt_Tdb_in = self.crear_input("30.0", "°C", grid_aire, 0, "Temp. Bulbo Seco (T_db):")
        self.txt_Twb_in = self.crear_input("17.8", "°C", grid_aire, 1, "Temp. Bulbo Húmedo (T_wb):")
        self.txt_caudal_a = self.crear_input("474.1", "m³/s", grid_aire, 2, "Caudal Aire Ventilador:")
        self.txt_densidad_a = self.crear_input("1.177", "kg/m³", grid_aire, 3, "Densidad del Aire:", precision=3)
        self.txt_altitud = self.crear_input("0.0", "m", grid_aire, 4, "Altitud del Sitio:")
        
        # ENTRY: Número de celdas NxN
        self.txt_num_celdas = self.crear_input_entero("20", "celdas", grid_aire, 5, "Resolución Malla (NxN):")
        gb_aire.setLayout(grid_aire)
        layout_izq.addWidget(gb_aire)

        # Botón compacto y sobrio
        self.btn_calcular = QPushButton("Calibrar NTU")
        self.btn_calcular.setFont(QFont("Segoe UI", 9))
        self.btn_calcular.setCursor(Qt.PointingHandCursor)
        self.btn_calcular.setStyleSheet("""
            QPushButton {
                background-color: #34495E;
                color: #FFFFFF;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #2C3E50;
            }
            QPushButton:pressed {
                background-color: #1A252F;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
                color: #7F8C8D;
            }
        """)
        self.btn_calcular.clicked.connect(self.ejecutar_calibracion)
        layout_izq.addWidget(self.btn_calcular)

        # Grupo 3: Cuadro de Resultados
        gb_res = QGroupBox("Resultados de Diagnóstico Térmico")
        gb_res.setStyleSheet(estilo_gb)
        layout_res = QVBoxLayout()
        layout_res.setSpacing(4)

        self.lbl_ntu_res = QLabel("NTU Calibrado:  --")
        self.lbl_q_res = QLabel("Carga Térmica:  --")
        self.lbl_range_res = QLabel("Range (ΔTw):  --")
        self.lbl_approach_res = QLabel("Approach:  --")
        self.lbl_lg_res = QLabel("Relación Masa (L/G):  --")
        self.lbl_evap_res = QLabel("Evaporación:  --")
        self.lbl_niebla_res = QLabel("Estado Pluma/Niebla:  --")

        for lbl in [self.lbl_ntu_res, self.lbl_q_res, self.lbl_range_res, self.lbl_approach_res, 
                    self.lbl_lg_res, self.lbl_evap_res, self.lbl_niebla_res]:
            lbl.setFont(QFont("Segoe UI", 9))
            lbl.setStyleSheet("color: #1A252F; padding: 1px 0px;")

        layout_res.addWidget(self.lbl_ntu_res)
        layout_res.addWidget(self.lbl_q_res)
        layout_res.addWidget(self.lbl_range_res)
        layout_res.addWidget(self.lbl_approach_res)
        layout_res.addWidget(self.lbl_lg_res)
        layout_res.addWidget(self.lbl_evap_res)
        layout_res.addWidget(self.lbl_niebla_res)

        gb_res.setLayout(layout_res)
        layout_izq.addWidget(gb_res)

        layout_izq.addStretch()

        # --------------------------------------------------
        # SECCIÓN DERECHA: GRÁFICA MAXIMIZADA
        # --------------------------------------------------
        panel_derecho = QWidget()
        layout_der = QVBoxLayout(panel_derecho)
        layout_der.setContentsMargins(5, 5, 5, 5)

        top_der_layout = QHBoxLayout()
        lbl_combo = QLabel("Variable a Visualizar en la Matriz 2D:")
        lbl_combo.setFont(QFont("Segoe UI", 9))
        lbl_combo.setStyleSheet("color: #444444;")

        self.combo_capa = QComboBox()
        self.combo_capa.setFont(QFont("Segoe UI", 9))
        self.combo_capa.addItems([
            "Temperatura del Agua (Tw)", 
            "Humedad Absoluta del Aire (wa)", 
            "Temperatura del Aire (Ta)"
        ])
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

        # --------------------------------------------------
        # BARRA DE ESTADO INFERIOR
        # --------------------------------------------------
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("font-size: 11px; color: #555555; background-color: #FAFAFA;")
        self.setStatusBar(self.status_bar)
        
        engine_msg = "Motor: CoolProp (NIST)" if HAS_COOLPROP else "Motor: ASHRAE Standard (CoolProp no instalado)"
        self.status_bar.showMessage(f"Listo. Ingrese las condiciones y presione 'Calibrar NTU'.  [{engine_msg}]")

    def crear_input(self, valor_defecto, unidad, grid_layout, fila, label_texto, precision=1):
        lbl = QLabel(label_texto)
        lbl.setFont(QFont("Segoe UI", 9))
        lbl.setStyleSheet("color: #444444;")
        
        txt = QLineEdit(valor_defecto)
        txt.setFont(QFont("Segoe UI", 9))
        
        txt.setStyleSheet("""
            QLineEdit {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 3px 6px;
                background-color: #FFFFFF;
                color: #222222;
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2;
                background-color: #FAFCFF;
            }
        """)
        
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
        lbl_unit.setStyleSheet("color: #777777;")

        grid_layout.addWidget(lbl, fila, 0)
        grid_layout.addWidget(txt, fila, 1)
        grid_layout.addWidget(lbl_unit, fila, 2)
        return txt

    def crear_input_entero(self, valor_defecto, unidad, grid_layout, fila, label_texto):
        lbl = QLabel(label_texto)
        lbl.setFont(QFont("Segoe UI", 9))
        lbl.setStyleSheet("color: #444444;")
        
        txt = QLineEdit(valor_defecto)
        txt.setFont(QFont("Segoe UI", 9))
        txt.setValidator(QIntValidator(5, 100))
        
        txt.setStyleSheet("""
            QLineEdit {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 3px 6px;
                background-color: #FFFFFF;
                color: #222222;
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2;
                background-color: #FAFCFF;
            }
        """)
        
        lbl_unit = QLabel(unidad)
        lbl_unit.setFont(QFont("Segoe UI", 8))
        lbl_unit.setStyleSheet("color: #777777;")

        grid_layout.addWidget(lbl, fila, 0)
        grid_layout.addWidget(txt, fila, 1)
        grid_layout.addWidget(lbl_unit, fila, 2)
        return txt

    def parse_float(self, text):
        return float(text.replace(',', '.'))

    def ejecutar_calibracion(self):
        try:
            datos = {
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

            self.progress_dialog = QProgressDialog("Iniciando simulación 2D...", "Cancelar", 0, 100, self)
            self.progress_dialog.setWindowTitle("Calibración Térmica en Proceso")
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.setValue(0)

            self.btn_calcular.setEnabled(False)
            self.status_bar.showMessage("Ejecutando subrutina de calibración...")

            self.worker = CalibracionWorker(datos)
            self.worker.progreso_signal.connect(self.actualizar_progreso_dialogo)
            self.worker.exito_signal.connect(self.procesar_exito)
            self.worker.error_signal.connect(self.procesar_error)
            self.worker.cancelado_signal.connect(self.procesar_cancelacion)
            
            # Al hacer clic en Cancelar, llamamos al método de interrupción suave
            self.progress_dialog.canceled.connect(self.worker.cancelar)
            
            self.worker.start()

        except ValueError:
            QMessageBox.warning(self, "Entrada Inválida", "Verifique que todos los campos contengan números válidos.")
            self.btn_calcular.setEnabled(True)

    def actualizar_progreso_dialogo(self, mensaje, porcentaje):
        if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
            self.progress_dialog.setLabelText(mensaje)
            self.progress_dialog.setValue(porcentaje)
        self.status_bar.showMessage(f"{mensaje} ({porcentaje}%)")

    def procesar_exito(self, res):
        if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
            self.progress_dialog.setValue(100)
            self.progress_dialog.close()

        self.btn_calcular.setEnabled(True)
        self.ultimo_resultado = res
        self.status_bar.showMessage(f"Calibración exitosa. NTU = {res['NTU']:.4f}")

        self.lbl_ntu_res.setText(f"NTU Calibrado: <b style='font-size:10.5pt; color:#2980B9;'>{res['NTU']:.4f}</b>")
        self.lbl_q_res.setText(f"Carga Térmica: <b>{res['Q_MWt']:.2f} MWt</b> ({res['Q_TR']:.0f} TR)")
        self.lbl_range_res.setText(f"Range (ΔTw): <b>{res['range_w']:.2f} °C</b>")
        self.lbl_approach_res.setText(f"Approach: <b>{res['approach_w']:.2f} °C</b>")
        self.lbl_lg_res.setText(f"Relación Masa (L/G): <b>{res['L_G_ratio']:.3f}</b>")
        self.lbl_evap_res.setText(f"Evaporación: <b>{res['evaporacion_m3h']:.2f} m³/h</b> ({res['pct_evap']:.2f}%)")

        if res['hay_niebla']:
            self.lbl_niebla_res.setText("Estado Pluma/Niebla: <b style='color:#C0392B;'>DETECTADA (Supersaturación)</b>")
        else:
            self.lbl_niebla_res.setText("Estado Pluma/Niebla: <b style='color:#27AE60;'>Sin Niebla (Aire no saturado)</b>")

        capa_activa = self.combo_capa.currentText()
        self.canvas.graficar_matriz(res, capa_activa)

    def procesar_cancelacion(self):
        """Maneja el evento cuando el usuario presiona Cancelar sin congelar la app."""
        if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
            self.progress_dialog.close()

        self.btn_calcular.setEnabled(True)
        self.status_bar.showMessage("Simulación cancelada por el usuario. Listo para un nuevo cálculo.")

    def cambiar_capa_grafico(self, nueva_capa):
        if self.ultimo_resultado is not None:
            self.canvas.graficar_matriz(self.ultimo_resultado, nueva_capa)

    def procesar_error(self, mensaje_error):
        if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
            self.progress_dialog.close()

        self.btn_calcular.setEnabled(True)
        self.status_bar.showMessage(f"Error: {mensaje_error}")
        QMessageBox.critical(self, "Error de Simulación", f"Error durante la simulación:\n{mensaje_error}")

# ==========================================
# 6. PUNTO DE ENTRADA
# ==========================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = TorreCoolingApp()
    window.show()
    sys.exit(app.exec_())