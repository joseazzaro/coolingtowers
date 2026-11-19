"""
Script para generar el gráfico de consumo energético anual del ventilador
y sobreconsumo por contrapresión según la altura de la malla (z).
"""

import numpy as np
import matplotlib.pyplot as plt

def generar_grafico_energia_malla(
    guardar=True,
    nombre_archivo='fan_energy_penalty_comparison.png'
):
    # Elevaciones evaluadas z (m)
    z_elevations = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.7, 5.0])
    
    # Consumo anual del ventilador (MWh/año)
    e_fan_mwh = np.array([112.59, 113.24, 113.02, 112.88, 112.78, 112.72, 112.68, 112.64, 112.61])
    
    # Incremento energético anual sobre la línea base (kWh/año)
    delta_e_kwh = np.array([0.0, 658.0, 437.8, 292.9, 197.0, 133.0, 90.2, 52.7, 19.8])
    
    # Configuración de figura con dos subplots compartiendo eje X
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)

    # --- Panel Superior: Consumo Eléctrico Anual (MWh/año) ---
    ax1.plot(z_elevations, e_fan_mwh, color='#1F77B4', linewidth=2.2, label='Annual Fan Energy (MWh/yr)')
    ax1.scatter(z_elevations, e_fan_mwh, color='#1F77B4', s=50, zorder=5)
    ax1.axhline(112.59, color='#7F8C8D', linestyle='--', linewidth=1.4, label='Baseline (No Mesh: 112.59 MWh/yr)')
    ax1.set_ylabel('Fan Energy (MWh/yr)', fontsize=10, fontweight='bold', color='#1F77B4')
    ax1.tick_params(axis='y', labelcolor='#1F77B4')
    ax1.set_ylim(112.4, 113.5)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    ax1.set_title('Annual Fan Energy Consumption vs. Collector Mesh Elevation', fontsize=11, fontweight='bold')

    # --- Panel Inferior: Penalización Energética ΔE (kWh/año) ---
    ax2.plot(z_elevations, delta_e_kwh, color='#D62728', linewidth=2.2, label='Incremental Energy ΔE (kWh/yr)')
    ax2.scatter(z_elevations, delta_e_kwh, color='#D62728', s=50, zorder=5)
    ax2.axvline(1.0, color='#7F8C8D', linestyle=':', linewidth=1.5, label='Current Proposal (z = 1.0 m)')
    ax2.axvline(3.7, color='#FF7F0E', linestyle='-', linewidth=1.8, label='Optimal Height (z ≈ 3.7 m)')
    ax2.axvspan(2.0, 3.5, color='#F39C12', alpha=0.15, label='Design Window (2.0 to 3.5 m)')

    # Destacado en z = 1.0 m
    ax2.annotate('z = 1.0 m\n+437.8 kWh/yr\n(+0.39%)', xy=(1.0, 437.8), xytext=(1.25, 480),
                 fontsize=8.5, fontweight='bold', color='#C0392B',
                 arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#FDEDEC', edgecolor='#E6B0AA', alpha=0.9))

    # Destacado en z = 3.7 m (óptimo)
    ax2.annotate('z = 3.7 m\n+52.7 kWh/yr\n(-88% penalty)', xy=(3.7, 52.7), xytext=(3.85, 160),
                 fontsize=8.5, fontweight='bold', color='#D35400',
                 arrowprops=dict(arrowstyle='->', color='#D35400', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#FEF9E7', edgecolor='#F39C12', alpha=0.9))

    ax2.set_xlabel('Vertical Distance Above Stack Discharge z (m)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Incremental Energy ΔE (kWh/yr)', fontsize=10, fontweight='bold', color='#D62728')
    ax2.tick_params(axis='y', labelcolor='#D62728')
    ax2.set_ylim(-10, 720)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    ax2.set_title('Aerodynamic Penalty Attenuation with Distance', fontsize=11, fontweight='bold')

    plt.tight_layout()
    if guardar:
        plt.savefig(nombre_archivo, dpi=300)
        print(f"Gráfico guardado exitosamente como: {nombre_archivo}")
    plt.show()

if __name__ == '__main__':
    generar_grafico_energia_malla()