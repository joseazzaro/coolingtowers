import sys
import os
import csv
import math
from datetime import datetime, timedelta
import numpy as np

# Importaciones de módulos refactorizados (separación de responsabilidades)
from core_calc import (
    obtener_presion_barometrica,
    cp_agua_local,
    humedad_saturacion,
    entalpia_saturacion,
    temp_aire_desde_entalpia,
    simular_torre_2d_matriz,
    resolver_punto_operacion,
    CP_WATER_DEFAULT, CP_AIR_DEFAULT, CP_VAPOR_DEFAULT, H_FG0_DEFAULT,
    HAS_COOLPROP
)
from psychro_data import PsicroLUT
from tower_sim import ControladorPID, CalibracionWorker, SimularDinamicaWorker
from utils import (
    leer_archivo_epw,
    traducir,
    parse_float_local,
    conectar_formato_precision,
    obtener_rango_epw,
    detectar_multianio_epw,
    normalizar_epw_a_año_canonico
)

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
import matplotlib.ticker as mticker

# ==========================================
# CONSTANTES LOCALES (Alias para compatibilidad)
# ==========================================
cp_w_def = CP_WATER_DEFAULT       # 4.184 kJ/kg.K
cp_a_def = CP_AIR_DEFAULT         # 1.006 kJ/kg.K
cp_v_def = CP_VAPOR_DEFAULT       # 1.86 kJ/kg.K
h_fg0_def = H_FG0_DEFAULT         # 2501.0 kJ/kg



# ==========================================
# UI DIALOG CLASSES - Main application logic
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


class DialogoPsicrometrico(QDialog):
    def __init__(self, parent=None, datos_sim=None, idioma='es'):
        super().__init__(parent)
        self.idioma = idioma
        self.datos_sim = datos_sim or {}
        self.setWindowTitle('Psychrometric Chart')
        self.resize(700, 520)
        self.init_ui()

    def tr_txt(self, key, **kwargs):
        return traducir(self.idioma, key, **kwargs)

    def init_ui(self):
        layout = QVBoxLayout(self)
        # Matplotlib canvas
        self.fig = Figure(figsize=(6.5, 4.5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Controls
        ctrl = QHBoxLayout()
        self.lbl_fecha = QLabel(self.tr_txt('pluma_fecha_default'))
        self.slider = QSlider(Qt.Horizontal)
        total = len(self.datos_sim.get('times', []))
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, total - 1))
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.actualizar_instante)
        ctrl.addWidget(self.lbl_fecha)
        ctrl.addWidget(self.slider, stretch=1)
        layout.addLayout(ctrl)

        # Initial draw
        self.actualizar_instante(0)

    def actualizar_instante(self, idx):
        if not self.datos_sim or idx >= len(self.datos_sim.get('times', [])):
            return
        t = self.datos_sim['times'][idx]
        self.lbl_fecha.setText(self.tr_txt('pluma_fecha_texto', fecha=t.strftime('%d/%m/%Y %H:%M')))

        # Prepare psychrometric plot: x = T_a_out (tower outlet), y = w_a_out (outlet humidity ratio)
        # Fog forms when outlet air reaches saturation curve
        T_a_out = np.array(self.datos_sim.get('t_a_out', []))
        w_a_out = np.array(self.datos_sim.get('w_a_out', []))
        
        # Convert outlet humidity from kg/kg to g/kg for display
        P_atm = 101325.0
        lut = PsicroLUT(T_min=-20.0, T_max=80.0, step=0.1, P_atm=P_atm)
        w_a_out_g = w_a_out * 1000.0  # Convert to g/kg

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # saturation curve (ws) from LUT (convert to g/kg for display)
        T_sat = lut.T_grid
        w_sat = lut.ws_lut
        w_sat_g = w_sat * 1000.0
        ax.plot(T_sat, w_sat_g, color='#2980B9', linestyle='--', linewidth=1.0, label=self.tr_txt('plot_saturation'))

        # plot trajectory: tower outlet temperature vs outlet humidity ratio (in g/kg)
        if len(T_a_out) > 0:
            ax.plot(T_a_out, w_a_out_g, color='#7F8C8D', alpha=0.6, label='Outlet Air')
        # mark current instant
        if idx < len(T_a_out):
            ax.scatter([T_a_out[idx]], [w_a_out_g[idx]], color='#C0392B', s=60, zorder=10)

        # X-axis limits: Fixed range 15-40°C for typical tower outlet conditions
        ax.set_xlim(15.0, 40.0)

        visible_sat = w_sat_g[(T_sat >= 15.0) & (T_sat <= 40.0)]
        visible_outlet = w_a_out_g[np.isfinite(w_a_out_g)]
        y_values = np.concatenate((visible_sat, visible_outlet))
        y_values = y_values[np.isfinite(y_values)]
        if y_values.size:
            y_min = min(0.0, float(np.min(y_values)))
            y_max = float(np.max(y_values))
            y_margin = max(0.5, (y_max - y_min) * 0.08)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

        ax.set_xlabel('Tower Outlet Temperature (°C)')
        ax.set_ylabel('Humidity Ratio (g/kg)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_title('Psychrometric Chart')
        ax.legend(loc='upper right', fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()


class DialogoDuracionAcumulada(QDialog):
    """Cumulative (load) duration curve: metrics sorted descending vs accumulated operating hours."""

    def __init__(self, parent=None, datos_sim=None, idioma='es'):
        super().__init__(parent)
        self.idioma = idioma
        self.datos_sim = datos_sim or {}
        self.setWindowTitle(self.tr_txt('dur_title'))
        self.resize(750, 550)
        self.init_ui()

    def tr_txt(self, key, **kwargs):
        return traducir(self.idioma, key, **kwargs)

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.fig = Figure(figsize=(7.0, 5.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        ctrl = QHBoxLayout()
        self.chk_water = QCheckBox(self.tr_txt('dur_chk_water'))
        self.chk_water.setChecked(True)
        self.chk_power = QCheckBox(self.tr_txt('dur_chk_power'))
        self.chk_power.setChecked(True)
        self.chk_thermal = QCheckBox(self.tr_txt('dur_chk_thermal'))
        self.chk_thermal.setChecked(False)

        for chk in (self.chk_water, self.chk_power, self.chk_thermal):
            chk.stateChanged.connect(self.replot)
            ctrl.addWidget(chk)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.replot()

    def _curva_duracion(self, values):
        """Sort values descending and pair them with accumulated operating hours."""
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr, arr
        dt_h = float(self.datos_sim.get('dt_sim_sec', 3600.0)) / 3600.0
        x_horas = np.arange(1, arr.size + 1) * dt_h
        return x_horas, np.sort(arr)[::-1]

    def replot(self):
        self.fig.clear()

        mostrar_agua = self.chk_water.isChecked()
        mostrar_power = self.chk_power.isChecked()
        mostrar_thermal = self.chk_thermal.isChecked()

        if not (mostrar_agua or mostrar_power or mostrar_thermal):
            self.canvas.draw()
            return

        ax = self.fig.add_subplot(111)
        lines = []
        eje_primario_usado = False
        ejes_extra = []

        if mostrar_agua:
            x, y = self._curva_duracion(self.datos_sim.get('agua_total_m3h', []))
            l, = ax.plot(x, y, color='#2980B9', label=self.tr_txt('dur_chk_water'), linewidth=1.4)
            lines.append(l)
            ax.set_ylabel(self.tr_txt('dur_ylabel_water'), color='#2980B9', fontsize=9)
            ax.tick_params(axis='y', labelcolor='#2980B9')
            eje_primario_usado = True

        if mostrar_power:
            x, y = self._curva_duracion(self.datos_sim.get('power_kw', []))
            ax_power = ax if not eje_primario_usado else ax.twinx()
            if ax_power is not ax:
                ejes_extra.append(ax_power)
            l, = ax_power.plot(x, y, color='#27AE60', label=self.tr_txt('dur_chk_power'), linewidth=1.4, linestyle='--')
            lines.append(l)
            ax_power.set_ylabel(self.tr_txt('dur_ylabel_power'), color='#27AE60', fontsize=9)
            ax_power.tick_params(axis='y', labelcolor='#27AE60')
            eje_primario_usado = True

        if mostrar_thermal:
            x, y = self._curva_duracion(self.datos_sim.get('q_mwt', []))
            if not eje_primario_usado:
                ax_thermal = ax
            else:
                ax_thermal = ax.twinx()
                if ejes_extra:
                    ax_thermal.spines['right'].set_position(('outward', 55))
                ejes_extra.append(ax_thermal)
            l, = ax_thermal.plot(x, y, color='#8E44AD', label=self.tr_txt('dur_chk_thermal'), linewidth=1.4, linestyle='-.')
            lines.append(l)
            ax_thermal.set_ylabel(self.tr_txt('dur_ylabel_thermal'), color='#8E44AD', fontsize=9)
            ax_thermal.tick_params(axis='y', labelcolor='#8E44AD')

        ax.set_xlabel(self.tr_txt('dur_xlabel'), fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=8, framealpha=0.85)
        self.fig.tight_layout()
        self.canvas.draw()


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
        self.txt_kp = QLineEdit("1.0")
        self.txt_ti = QLineEdit("1800.0")
        self.txt_td = QLineEdit("0.0")
        self.txt_speed_min = QLineEdit("20.0")
        # --- NUEVOS CAMPOS: Banda Muerta y Rampa Máxima ---
        self.txt_deadband = QLineEdit("0.3")
        self.txt_max_rate = QLineEdit("5.0")

        for txt in [self.txt_t_set, self.txt_kp, self.txt_ti, self.txt_td, 
                    self.txt_speed_min, self.txt_deadband, self.txt_max_rate]:
            txt.setFont(QFont("Segoe UI", 9))
            txt.setValidator(QDoubleValidator())
            txt.setFixedWidth(90)
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

        # --- AGREGAR A LA GRILLA ---
        grid_pid.addWidget(QLabel("Banda Muerta (Deadband):"), 5, 0)
        grid_pid.addWidget(self.txt_deadband, 5, 1)
        grid_pid.addWidget(QLabel("°C"), 5, 2)

        grid_pid.addWidget(QLabel("Rampa Máx. (Max Rate):"), 6, 0)
        grid_pid.addWidget(self.txt_max_rate, 6, 1)
        grid_pid.addWidget(QLabel("%/paso"), 6, 2)

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
                'deadband': float(self.txt_deadband.text()),
                'max_rate': float(self.txt_max_rate.text()),
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
            'deadband': self.txt_deadband.text(),
            'max_rate': self.txt_max_rate.text(),
            'p_motor': self.txt_p_motor.text(),
            'eta_fan': self.txt_eta_fan.text(),
            'epw_normalize': getattr(self, 'epw_normalize', False),
            'epw_normalize_year': getattr(self, 'epw_normalize_year', None),
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
        self.txt_deadband.setText(estado.get('deadband', '0.3'))
        self.txt_max_rate.setText(estado.get('max_rate', '5.0'))
        self.txt_p_motor.setText(estado['p_motor'])
        self.txt_eta_fan.setText(estado['eta_fan'])
        self.epw_normalize = estado.get('epw_normalize', False)
        self.epw_normalize_year = estado.get('epw_normalize_year')

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
        return traducir(self.idioma, key, **kwargs)

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

        # Psychrometric chart action
        self.action_psicrometrico = QAction(self)
        self.action_psicrometrico.setEnabled(False)
        self.action_psicrometrico.triggered.connect(self.abrir_ventana_psicrometrico)
        self.menu_simulacion.addAction(self.action_psicrometrico)

        # Cumulative duration curve action
        self.action_duracion = QAction(self)
        self.action_duracion.setEnabled(False)
        self.action_duracion.triggered.connect(self.abrir_ventana_duracion)
        self.menu_simulacion.addAction(self.action_duracion)

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
        self.menu_settings = menubar.addMenu(self.tr_txt('menu_settings'))
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
        self.action_psicrometrico.setText(self.tr_txt('accion_ver_psicrometrico'))
        self.action_psicrometrico.setToolTip(self.tr_txt('tip_ver_psicrometrico'))
        self.action_duracion.setText(self.tr_txt('accion_ver_duracion'))
        self.action_duracion.setToolTip(self.tr_txt('tip_ver_duracion'))
        self.action_clear_epw_choice.setText(self.tr_txt('sim_clear_epw_choice'))
        # reset prefs menu item
        self.action_reset_prefs.setText(self.tr_txt('sim_reset_prefs'))

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
        self.action_psicrometrico.setEnabled(True)
        self.action_duracion.setEnabled(True)

    def abrir_ventana_pluma(self):
        if self.ultimo_resultado_dinamico is None:
            QMessageBox.warning(self, self.tr_txt('title_simulacion_requerida'), self.tr_txt('msg_simulacion_requerida'))
            return

        dlg = DialogoPerfilPluma(self, datos_sim=self.ultimo_resultado_dinamico, idioma=self.idioma)
        dlg.exec_()

    def abrir_ventana_psicrometrico(self):
        if self.ultimo_resultado_dinamico is None:
            QMessageBox.warning(self, self.tr_txt('title_simulacion_requerida'), self.tr_txt('msg_simulacion_requerida'))
            return
        dlg = DialogoPsicrometrico(self, datos_sim=self.ultimo_resultado_dinamico, idioma=self.idioma)
        dlg.exec_()

    def abrir_ventana_duracion(self):
        if self.ultimo_resultado_dinamico is None:
            QMessageBox.warning(self, self.tr_txt('title_simulacion_requerida'), self.tr_txt('msg_simulacion_requerida'))
            return
        dlg = DialogoDuracionAcumulada(self, datos_sim=self.ultimo_resultado_dinamico, idioma=self.idioma)
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