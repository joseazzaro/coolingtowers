import sys
import numpy as np
from scipy.optimize import root_scalar

# Importaciones de PyQt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QStatusBar,
    QSplitter, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QLocale
from PyQt5.QtGui import QFont, QDoubleValidator

# Importaciones de Matplotlib para PyQt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==========================================
# 1. CONSTANTES TERMODINÁMICAS Y PSICROMETRÍA
# ==========================================
cp_w = 4.184     # kJ/kg.K
cp_a = 1.006     # kJ/kg.K
cp_v = 1.86      # kJ/kg.K
h_fg0 = 2501.0   # kJ/kg

def obtener_presion_barometrica(altitud_m):
    P0 = 101.325 
    return P0 * (1.0 - 0.6875e-5 * altitud_m)**5.2561

def humedad_saturacion(T, P_atm=101.325):
    P_sat = 0.61078 * np.exp(17.27 * T / (T + 237.3)) 
    return 0.622 * P_sat / (P_atm - P_sat)

def factor_lewis(w_sw, w):
    """Factor de Lewis dinámico de Bosnjakovic blindado contra valores nulos/negativos."""
    # Si la humedad del aire alcanza o supera la saturación, usar el límite constante
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

def entalpia_saturacion(T, w_sat):
    return cp_a * T + w_sat * (h_fg0 + cp_v * T)

def temp_aire_desde_entalpia(h_a, w_a):
    return (h_a - w_a * h_fg0) / (cp_a + w_a * cp_v)

# ==========================================
# 2. MOTOR DE SIMULACIÓN POPPE 2D
# ==========================================
def simular_torre_2d_matriz(NTU_actual, T_w_in, m_w_total, h_a_in, w_a_in, m_a_total, P_atm=101.325, Nx=20, Ny=20):
    dm_w = m_w_total / Nx  
    dm_a = m_a_total / Ny  
    K_dA = (NTU_actual * m_w_total) / (Nx * Ny) 
    
    T_w = np.zeros((Ny + 1, Nx))
    m_w = np.zeros((Ny + 1, Nx))
    h_a = np.zeros((Ny, Nx + 1))
    w_a = np.zeros((Ny, Nx + 1))
    
    matriz_niebla = np.zeros((Ny, Nx), dtype=bool)
    
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
            
            w_sw = humedad_saturacion(T_water_cell, P_atm)
            h_sw = entalpia_saturacion(T_water_cell, w_sw)
            h_v = h_fg0 + cp_v * T_water_cell
            Le = factor_lewis(w_sw, w_air_cell)
            
            potencial_w = w_sw - w_air_cell
            potencial_h = (h_sw - h_air_cell) + (Le - 1) * (h_sw - h_air_cell - potencial_w * h_v) + potencial_w * cp_w * T_water_cell
            
            agua_evap_celda = K_dA * potencial_w
            calor_transferido = K_dA * potencial_h
            
            w_a_next = w_air_cell + (agua_evap_celda / dm_a)
            h_a_next = h_air_cell + (calor_transferido / dm_a)
            
            w_a[i, j+1] = w_a_next
            h_a[i, j+1] = h_a_next
            
            T_a_next = temp_aire_desde_entalpia(h_a_next, w_a_next)
            w_sat_local = humedad_saturacion(T_a_next, P_atm)
            
            if w_a_next > w_sat_local:
                matriz_niebla[i, j] = True
            
            m_w[i+1, j] = m_water_cell - agua_evap_celda
            T_w[i+1, j] = (m_water_cell * cp_w * T_water_cell - calor_transferido) / (m_water_cell * cp_w)

    T_w_salida_final = np.average(T_w[Ny, :], weights=m_w[Ny, :])
    agua_evaporada_total = m_w_total - np.sum(m_w[Ny, :])
    
    return T_w_salida_final, agua_evaporada_total, T_w[:-1, :], matriz_niebla

# ==========================================
# 3. HILO DE CÁLCULO (QTHREAD) EN SEGUNDO PLANO
# ==========================================
class CalibracionWorker(QThread):
    progreso_signal = pyqtSignal(str)
    exito_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, datos_input):
        super().__init__()
        self.d = datos_input

    def run(self):
        try:
            self.progreso_signal.emit("Calculando presión barométrica y balances de masa...")
            P_atm = obtener_presion_barometrica(self.d['altitud'])
            
            # Caudales masivos (kg/s)
            m_w_total = self.d['caudal_w'] * 1000.0 / 3600.0  # m3/h -> kg/s
            m_a_total = self.d['caudal_a'] * self.d['densidad_a'] # m3/s * kg/m3 -> kg/s
            
            # Psicrometría rigurosa para Tdb y Twb
            T_db = self.d['T_db_in']
            T_wb = self.d['T_wb_in']
            
            # Humedad de saturación a la temperatura de bulbo húmedo
            w_sat_wb = humedad_saturacion(T_wb, P_atm)
            
            # Humedad absoluta real del aire ambiente (Ecuación Psicrométrica de Carrier)
            w_a_in = ((h_fg0 - (cp_w - cp_v) * T_wb) * w_sat_wb - cp_a * (T_db - T_wb)) / (h_fg0 + cp_v * T_db - cp_w * T_wb)
            
            # Entalpía real de entrada
            h_a_in = cp_a * T_db + w_a_in * (h_fg0 + cp_v * T_db)

            self.progreso_signal.emit("Iniciando búsqueda iterativa del NTU real...")

            def objetivo_ntu(NTU_guess):
                T_calc, _, _, _ = simular_torre_2d_matriz(
                    NTU_guess, self.d['T_w_in'], m_w_total, h_a_in, w_a_in, m_a_total, P_atm
                )
                return T_calc - self.d['T_w_out_target']

            # Búsqueda robusta de bracket
            ntu_puntos = np.linspace(0.1, 10.0, 40)
            errores = [objetivo_ntu(val) for val in ntu_puntos]
            
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

            self.progreso_signal.emit(f"¡NTU Calibrado con éxito: {NTU_calibrado:.4f}! Generando matriz 2D...")

            T_sal, evap_kg, Matriz_T, Matriz_niebla = simular_torre_2d_matriz(
                NTU_calibrado, self.d['T_w_in'], m_w_total, h_a_in, w_a_in, m_a_total, P_atm
            )

            evap_m3h = evap_kg * 3600.0 / 1000.0

            resultado = {
                'NTU': NTU_calibrado,
                'T_salida': T_sal,
                'evaporacion_m3h': evap_m3h,
                'Matriz_T': Matriz_T,
                'Matriz_niebla': Matriz_niebla,
                'T_w_in': self.d['T_w_in']
            }

            self.exito_signal.emit(resultado)

        except Exception as e:
            self.error_signal.emit(str(e))

# ==========================================
# 4. CANVA DE MATPLOTLIB (PANEL DERECHO MEJORADO)
# ==========================================
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.colorbar = None

    def graficar_matriz_agua(self, Matriz_T, Matriz_niebla, T_w_in, T_w_out, NTU):
        self.ax.clear()
        
        # Eliminar colorbar previa para evitar duplicación
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

        Ny, Nx = Matriz_T.shape

        # 1. Mapa de Calor Base de la Temperatura del Agua
        im = self.ax.imshow(Matriz_T, cmap='coolwarm', origin='upper', aspect='auto')
        self.colorbar = self.fig.colorbar(im, ax=self.ax, pad=0.03)
        self.colorbar.set_label('Temperatura Local del Agua (°C)', fontsize=10, fontweight='bold', labelpad=10)

        # 2. Sombra y Contorno de Niebla
        if np.any(Matriz_niebla):
            capa_niebla = np.zeros((Ny, Nx, 4))
            capa_niebla[Matriz_niebla] = [0.3, 0.3, 0.3, 0.45] # Gris semitransparente
            
            self.ax.imshow(capa_niebla, origin='upper', aspect='auto')
            self.ax.contour(Matriz_niebla, levels=[0.5], colors=['black'], linestyles=['--'], linewidths=[2])
            
            self.ax.plot([], [], color='gray', alpha=0.6, linewidth=8, label='Zona de Niebla (Supersaturación)')
            self.ax.plot([], [], color='black', linestyle='--', linewidth=2, label='Frente de Condensación')
            self.ax.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

        # 3. TÍTULO CON INFORMACIÓN TÉRMICA ORGANIZADA (SIN SOLAPAMIENTO)
        titulo_texto = (
            f"Perfil Térmico 2D del Agua (NTU = {NTU:.4f})\n"
            f"Entrada Techo: {T_w_in:.1f} °C  →  Piscina Mezclada: {T_w_out:.2f} °C"
        )
        self.ax.set_title(titulo_texto, fontsize=11, fontweight='bold', pad=15)

        # 4. ETIQUETAS DE EJES CON DIRECCIONES CLARAS
        self.ax.set_xlabel('Entrada Aire Ambiente  →  Dirección del Flujo de Aire  →  Salida', fontsize=9.5, labelpad=10)
        self.ax.set_ylabel('← Caída del Agua (Techo a Piscina) →', fontsize=9.5, labelpad=10)

        # Ajuste inteligente de márgenes de Matplotlib
        self.fig.tight_layout()
        self.draw()

# ==========================================
# 5. VENTANA PRINCIPAL DE PyQt5
# ==========================================
class TorreCoolingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemelo Digital de Torre de Enfriamiento 2D (Teoría de Poppe)")
        self.setGeometry(100, 100, 1200, 750)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --------------------------------------------------
        # SECCIÓN IZQUIERDA: INPUTS DE LA TORRE
        # --------------------------------------------------
        panel_izquierdo = QWidget()
        layout_izq = QVBoxLayout(panel_izquierdo)

        font_titulo = QFont("Arial", 11, QFont.Bold)

        # Grupo 1: Datos de Agua
        gb_agua = QGroupBox("Parámetros del Agua (Condición de Rating)")
        gb_agua.setFont(font_titulo)
        grid_agua = QGridLayout()

        self.txt_Tw_in = self.crear_input("31.7", "°C", grid_agua, 0, "Temp. Entrada Agua (T_w1):")
        self.txt_Tw_out = self.crear_input("20.6", "°C", grid_agua, 1, "Temp. Salida Deseada (T_w2):")
        self.txt_caudal_w = self.crear_input("1174.0", "m³/h", grid_agua, 2, "Caudal Volumétrico Agua:")
        gb_agua.setLayout(grid_agua)
        layout_izq.addWidget(gb_agua)

        # Grupo 2: Datos del Aire y Clima
        gb_aire = QGroupBox("Condiciones del Aire y Clima Ambientales")
        gb_aire.setFont(font_titulo)
        grid_aire = QGridLayout()

        self.txt_Tdb_in = self.crear_input("30.0", "°C", grid_aire, 0, "Temp. Bulbo Seco (T_db):")
        self.txt_Twb_in = self.crear_input("17.8", "°C", grid_aire, 1, "Temp. Bulbo Húmedo (T_wb):")
        self.txt_caudal_a = self.crear_input("474.1", "m³/s", grid_aire, 2, "Caudal Aire Ventilador:")
        self.txt_densidad_a = self.crear_input("1.177", "kg/m³", grid_aire, 3, "Densidad del Aire:")
        self.txt_altitud = self.crear_input("0.0", "m.s.n.m.", grid_aire, 4, "Altitud del Sitio:")
        gb_aire.setLayout(grid_aire)
        layout_izq.addWidget(gb_aire)

        # Botón de Cálculo y Calibración
        self.btn_calcular = QPushButton("🔥 CALIBRAR NTU Y GENERAR MODELO 2D")
        self.btn_calcular.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_calcular.setStyleSheet("background-color: #007ACC; color: white; padding: 12px; border-radius: 5px;")
        self.btn_calcular.clicked.connect(self.ejecutar_calibracion)
        layout_izq.addWidget(self.btn_calcular)

        # Grupo 3: Cuadro de Resultados de Calibración
        gb_res = QGroupBox("Resultados de la Calibración")
        gb_res.setFont(font_titulo)
        layout_res = QVBoxLayout()

        self.lbl_ntu_res = QLabel("NTU Calibrado: --")
        self.lbl_ntu_res.setFont(QFont("Arial", 10, QFont.Bold))
        self.lbl_evap_res = QLabel("Evaporación Estimada: -- m³/h")
        self.lbl_evap_res.setFont(QFont("Arial", 10))

        layout_res.addWidget(self.lbl_ntu_res)
        layout_res.addWidget(self.lbl_evap_res)
        gb_res.setLayout(layout_res)
        layout_izq.addWidget(gb_res)

        layout_izq.addStretch()

        # --------------------------------------------------
        # SECCIÓN DERECHA: GRÁFICA DE MATPLOTLIB (MATRIZ 2D)
        # --------------------------------------------------
        panel_derecho = QWidget()
        layout_der = QVBoxLayout(panel_derecho)

        self.canvas = MplCanvas(self, width=6, height=6, dpi=100)
        layout_der.addWidget(self.canvas)

        splitter.addWidget(panel_izquierdo)
        splitter.addWidget(panel_derecho)
        splitter.setSizes([400, 800])

        # --------------------------------------------------
        # BARRA INFERIOR DE ESTADO (STATUS BAR)
        # --------------------------------------------------
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo. Ingrese las condiciones de operación y presione 'Calibrar'.")

    def crear_input(self, valor_defecto, unidad, grid_layout, fila, label_texto):
        lbl = QLabel(label_texto)
        lbl.setFont(QFont("Arial", 9))
        
        txt = QLineEdit(valor_defecto)
        txt.setFont(QFont("Arial", 9))
        
        # Seteamos un validador con locale 'C' neutral para evitar la confusión del punto/coma
        validator = QDoubleValidator()
        validator.setLocale(QLocale("C"))
        txt.setValidator(validator)
        
        lbl_unit = QLabel(unidad)
        lbl_unit.setFont(QFont("Arial", 9, QFont.Bold))
        lbl_unit.setStyleSheet("color: #555555;")

        grid_layout.addWidget(lbl, fila, 0)
        grid_layout.addWidget(txt, fila, 1)
        grid_layout.addWidget(lbl_unit, fila, 2)
        return txt

    def parse_float(self, text):
        """Convierte texto limpiando comas por puntos por compatibilidad."""
        return float(text.replace(',', '.'))

    def ejecutar_calibracion(self):
        try:
            # Lectura sanitizada reemplazando comas por puntos
            datos = {
                'T_w_in': self.parse_float(self.txt_Tw_in.text()),
                'T_w_out_target': self.parse_float(self.txt_Tw_out.text()),
                'caudal_w': self.parse_float(self.txt_caudal_w.text()),
                'T_db_in': self.parse_float(self.txt_Tdb_in.text()),
                'T_wb_in': self.parse_float(self.txt_Twb_in.text()),
                'caudal_a': self.parse_float(self.txt_caudal_a.text()),
                'densidad_a': self.parse_float(self.txt_densidad_a.text()),
                'altitud': self.parse_float(self.txt_altitud.text())
            }

            self.btn_calcular.setEnabled(False)
            self.status_bar.showMessage("Iniciando subrutina de calibración termodinámica...")

            self.worker = CalibracionWorker(datos)
            self.worker.progreso_signal.connect(self.actualizar_status)
            self.worker.exito_signal.connect(self.procesar_exito)
            self.worker.error_signal.connect(self.procesar_error)
            self.worker.start()

        except ValueError:
            QMessageBox.warning(self, "Error de Entrada", "Por favor, ingrese valores numéricos válidos en todos los campos.")
            self.btn_calcular.setEnabled(True)

    def actualizar_status(self, mensaje):
        self.status_bar.showMessage(mensaje)

    def procesar_exito(self, res):
        self.btn_calcular.setEnabled(True)
        self.status_bar.showMessage(f"Cálculo finalizado exitosamente. NTU = {res['NTU']:.4f}")

        self.lbl_ntu_res.setText(f"NTU Calibrado: {res['NTU']:.4f}")
        self.lbl_evap_res.setText(f"Evaporación Estimada: {res['evaporacion_m3h']:.2f} m³/h")

        self.canvas.graficar_matriz_agua(
            res['Matriz_T'], res['Matriz_niebla'], res['T_w_in'], res['T_salida'], res['NTU']
        )

    def procesar_error(self, mensaje_error):
        self.btn_calcular.setEnabled(True)
        self.status_bar.showMessage(f"Error durante la simulación: {mensaje_error}")
        QMessageBox.critical(self, "Error de Simulación", f"Ocurrió un error en el cálculo:\n{mensaje_error}")

# ==========================================
# 6. PUNTO DE ENTRADA
# ==========================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = TorreCoolingApp()
    window.show()
    sys.exit(app.exec_())