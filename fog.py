"""
Modeling and optimization script for fog-harvesting mesh installed above a cooling tower.
Computes Liquid Water Content (LWC), backpressure transmitted to fan blades,
and net water collection rate as a function of vertical distance z.
Outputs publication-quality dual plots in English with annotated key values.
"""

import numpy as np
import matplotlib.pyplot as plt

def p_sat(T_celsius):
    """Saturation vapor pressure using Tetens equation (in kPa)."""
    return 0.61078 * np.exp(17.27 * T_celsius / (T_celsius + 237.3))

def w_sat(T_celsius, P_kPa=101.325):
    """Air saturation humidity ratio in g vapor / kg dry air."""
    ps = p_sat(T_celsius)
    return 0.622 * ps / (P_kPa - ps) * 1000.0

def simulate_mesh_height_optimization(
    D_boca=3.0,          # Fan stack discharge diameter (m)
    v0=4.0,              # Mean air discharge velocity at stack exit (m/s)
    T_out=26.0,          # Air dry bulb temperature at fill outlet (°C)
    w_out=21.5,          # Air humidity ratio at fill outlet (g/kg dry air)
    T_amb=4.0,           # Winter ambient dry bulb temperature (°C)
    w_amb=4.2,           # Ambient humidity ratio (g/kg, RH ~85%)
    D_malla=3.8,         # Effective collector mesh diameter (m)
    P_atm_kPa=101.325,   # Barometric pressure (kPa)
    save_figure=True,
    filename='mesh_height_optimization_en.png'
):
    # Vertical distance vector above stack outlet (0 to 10 meters)
    z_vals = np.linspace(0.01, 10.0, 500)
    
    # 1. Turbulent jet mass entrainment ratio (axisymmetric free jet)
    m_ratio = 1.0 + 0.25 * (z_vals / D_boca) + 0.05 * (z_vals / D_boca)**2
    f_amb = (m_ratio - 1.0) / m_ratio
    f_out = 1.0 - f_amb
    
    # 2. Psychrometric mixing at distance z
    T_mix = f_out * T_out + f_amb * T_amb
    w_mix = f_out * w_out + f_amb * w_amb
    ws_mix = np.array([w_sat(t, P_atm_kPa) for t in T_mix])
    
    # 3. Supersaturated condensed liquid water (LWC)
    dw_cond = np.maximum(0.0, w_mix - ws_mix)  # g liquid / kg dry air
    rho_a = P_atm_kPa / (0.287 * (T_mix + 273.15))
    LWC = dw_cond * rho_a  # g liquid water / m³ moist air
    
    # 4. Total condensed droplet flow in the plume (L/h)
    A_boca = np.pi * (D_boca / 2.0)**2
    Q0 = A_boca * v0  # m³/s air flow
    m_dot_liq_total = (LWC / 1000.0) * (Q0 * m_ratio) * 3600.0  # L/h
    
    # 5. Core velocity decay and backpressure transmitted upstream to fan blades
    v_mean = v0 / (1.0 + 0.35 * (z_vals / D_boca))
    Delta_P_malla = 2.5 * 0.5 * rho_a * (v_mean**2)  # Mesh head loss (Pa)
    factor_transmision = np.exp(-1.8 * (z_vals / D_boca))
    Delta_P_fan = Delta_P_malla * factor_transmision  # Backpressure on fan blades (Pa)
    
    # 6. Inertial impaction capture efficiency (Stokes number dependency)
    eta_captura = 0.40 * (v_mean / v0)**0.65
    
    # 7. Geometric plume expansion (~12° half-angle cone)
    theta_rad = np.radians(12.0)
    D_pluma = D_boca + 2.0 * z_vals * np.tan(theta_rad)
    frac_interceptada = np.minimum(1.0, (D_malla / D_pluma)**2)
    
    # 8. Net collected water rate (liters/h)
    Agua_rec = m_dot_liq_total * eta_captura * frac_interceptada
    
    # Hydraulic-aerodynamic trade-off optimization index
    indice_opt = np.argmax(Agua_rec / (1.0 + 0.05 * Delta_P_fan))
    z_opt = z_vals[indice_opt]
    
    # Extract benchmark values at z = 1.0 m and optimal height
    idx_1m = np.argmin(np.abs(z_vals - 1.0))
    lwc_1m = LWC[idx_1m]
    dp_1m = Delta_P_fan[idx_1m]
    rec_1m = Agua_rec[idx_1m]
    
    lwc_opt = LWC[indice_opt]
    dp_opt = Delta_P_fan[indice_opt]
    rec_opt = Agua_rec[indice_opt]
    gain_pct = ((rec_opt / rec_1m) - 1.0) * 100.0

    # --- Plotting Configuration ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)

    # Top Subplot: Plume Thermodynamics vs. Aerodynamic Penalty
    l1 = ax1.plot(z_vals, LWC, color='#1F77B4', linewidth=2.2, label='Plume LWC (g/m³)')
    ax1.set_ylabel('Liquid Water Content - LWC (g/m³)', color='#1F77B4', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1F77B4')
    ax1.set_ylim(0, 3.2)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Annotations on LWC curve
    ax1.plot(1.0, lwc_1m, marker='o', markersize=6, color='#1F77B4')
    ax1.annotate(f'{lwc_1m:.2f} g/m³', xy=(1.0, lwc_1m), xytext=(1.15, lwc_1m - 0.25),
                 fontsize=8.5, color='#1F77B4', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#1F77B4', lw=1.2))

    ax1.plot(z_opt, lwc_opt, marker='o', markersize=6, color='#1F77B4')
    ax1.annotate(f'{lwc_opt:.2f} g/m³', xy=(z_opt, lwc_opt), xytext=(z_opt + 0.2, lwc_opt - 0.2),
                 fontsize=8.5, color='#1F77B4', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#1F77B4', lw=1.2))

    # Right axis: Backpressure on fan
    ax1_twin = ax1.twinx()
    l2 = ax1_twin.plot(z_vals, Delta_P_fan, color='#D62728', linewidth=2.2, linestyle='--', label='Fan Backpressure ΔP (Pa)')
    ax1_twin.set_ylabel('Backpressure on Fan Blades ΔP (Pa)', color='#D62728', fontsize=10, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor='#D62728')
    ax1_twin.set_ylim(0, 25)

    # Annotations on Backpressure curve
    ax1_twin.plot(1.0, dp_1m, marker='s', markersize=6, color='#D62728')
    ax1_twin.annotate(f'{dp_1m:.1f} Pa', xy=(1.0, dp_1m), xytext=(1.2, dp_1m + 2.0),
                      fontsize=8.5, color='#D62728', fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color='#D62728', lw=1.2))

    ax1_twin.plot(z_opt, dp_opt, marker='s', markersize=6, color='#D62728')
    ax1_twin.annotate(f'{dp_opt:.1f} Pa', xy=(z_opt, dp_opt), xytext=(z_opt + 0.2, dp_opt + 2.0),
                      fontsize=8.5, color='#D62728', fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color='#D62728', lw=1.2))

    lines1 = l1 + l2
    labels1 = [line.get_label() for line in lines1]
    ax1.legend(lines1, labels1, loc='center right', fontsize=8.5, framealpha=0.9)
    ax1.set_title('Plume Thermodynamics vs. Aerodynamic Fan Penalty', fontsize=11, fontweight='bold')

    # Bottom Subplot: Net Fog Water Collection
    ax2.plot(z_vals, Agua_rec, color='#2CA02C', linewidth=2.2, label='Water Recovery Rate (L/h)')
    ax2.axvline(1.0, color='#7F8C8D', linestyle=':', linewidth=1.6, label='Current Proposal (z = 1.0 m)')
    ax2.axvline(z_opt, color='#FF7F0E', linestyle='-', linewidth=2.0, label=f'Optimal Height (z ≈ {z_opt:.1f} m)')
    ax2.axvspan(2.0, 3.5, color='#F39C12', alpha=0.15, label='Design Compromise Window (2.0 to 3.5 m)')

    # Callout at z = 1.0 m
    ax2.plot(1.0, rec_1m, marker='o', markersize=6, color='#2CA02C')
    ax2.annotate(f'z = 1.0 m\n{rec_1m:.1f} L/h', xy=(1.0, rec_1m), xytext=(0.2, rec_1m + 15),
                 fontsize=8.5, fontweight='bold', color='#2C3E50',
                 arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#ECF0F1', edgecolor='#BDC3C7', alpha=0.9))

    # Callout at Optimal Height
    ax2.plot(z_opt, rec_opt, marker='*', markersize=10, color='#D35400')
    ax2.annotate(f'Optimal: z = {z_opt:.1f} m\n{rec_opt:.1f} L/h (+{gain_pct:.0f}%)',
                 xy=(z_opt, rec_opt), xytext=(z_opt + 0.35, rec_opt - 2.0),
                 fontsize=8.5, fontweight='bold', color='#D35400',
                 arrowprops=dict(arrowstyle='->', color='#D35400', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#FEF9E7', edgecolor='#F39C12', alpha=0.9))

    ax2.set_xlabel('Vertical Distance Above Stack Discharge z (m)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Recovered Water (L/h)', color='#2CA02C', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#2CA02C')
    ax2.set_ylim(0, 95)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    ax2.set_title(f'Net Fog Water Collection with Feasible Mesh (D ≈ {D_malla} m)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    if save_figure:
        plt.savefig(filename, dpi=300)
        print(f"Chart successfully saved as: {filename}")
    plt.show()

if __name__ == '__main__':
    simulate_mesh_height_optimization()