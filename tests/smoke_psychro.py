from PyQt5.QtWidgets import QApplication
from datetime import datetime, timedelta
import sys

# Import the dialog class from the application
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tower_app_21 import DialogoPsicrometrico


def run_smoke():
    app = QApplication(sys.argv)

    base = datetime(2009, 2, 1, 0, 0)
    times = [base + timedelta(hours=i) for i in range(6)]

    res_sim = {
        'times': times,
        't_db': [16.5, 17.0, 17.5, 18.0, 18.5, 19.0],
        't_wb': [10.0, 10.2, 10.4, 10.6, 10.8, 11.0],
        't_a_out': [20.0, 20.1, 20.2, 20.3, 20.4, 20.5],
        'w_a_out': [0.0080, 0.0085, 0.0090, 0.0095, 0.0100, 0.0105],  # Outlet humidity (kg/kg)
        'fan_speed': [50, 55, 60, 65, 70, 75],
        'caudal_a_m3s': 1.0
    }

    dlg = DialogoPsicrometrico(None, datos_sim=res_sim, idioma='es')
    # Call actualizar_instante for a few indexes to exercise plotting
    for i in range(len(times)):
        dlg.actualizar_instante(i)
    print('SMOKE_OK')


if __name__ == '__main__':
    run_smoke()
