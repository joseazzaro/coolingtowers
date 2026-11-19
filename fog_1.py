"""
Modeling and optimization script for fog-harvesting mesh installed above a cooling tower.
Computes Liquid Water Content (LWC), backpressure transmitted to fan blades,
and net water collection rate as a function of vertical distance z.
Outputs publication-quality dual plots in English with annotated key values for multiple mesh yields.
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
    D_boca=3.0,          
    v0=4.0,              
    T_out=26.0,          
    w_out=21.5,          
    T_amb=4.0,           
    w_amb=4.2,           
    D_malla=3.8,         
    P_atm_kPa=101.325,
    rendimientos_base=[0.30, 0.40, 0.50], # Distintos rendimientos base de malla (30%, 40%, 50%)
    save_figure=True,
    filename='mesh_height_optimization_yields.png'
):
    z_vals = np.linspace(0.01, 10.0, 500)
    
    m_ratio = 1.0 + 0.25 * (z_vals / D_boca) + 0.05 * (z_vals / D_boca)**2
    f_amb = (m_ratio - 1.0) / m_ratio
    f_out = 1.0 - f_amb
    
    T_mix = f_out * T_out + f_amb * T_amb
    w_mix = f_out * w_out + f_amb * w_amb
    ws_mix = np.array([w_sat(t, P_atm_kPa) for t in T_mix])
    
    dw_cond = np.maximum(0.0, w_mix - ws_mix)
    rho_a = P_atm_kPa / (0.287 * (T_mix + 273.15))
    LWC = dw_cond * rho_a
    
    A_boca = np.pi * (D_boca / 2.0)**2
    Q0 = A_boca * v0
    m_dot_liq_total = (LWC / 1000.0) * (Q0 * m_ratio) * 3600.0
    
    v_mean = v0 / (1.0 + 0.35 * (z_vals / D_boca))
    Delta_P_malla = 2.5 * 0.5 * rho_a * (v_mean**2)
    factor_transmision = np.exp(-1.8 * (z_vals / D_boca))
    Delta_P_fan = Delta_P_malla * factor_transmision
    
    theta_rad = np.radians(12.0)
    D_pluma = D_boca + 2.0 * z_vals * np.tan(theta_rad)
    frac_interceptada = np.minimum(1.0, (D_malla / D_pluma)**2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 8.5), sharex=True)

    # Top Subplot: Plume Thermodynamics vs. Aerodynamic Penalty
    l1 = ax1.plot(z_vals, LWC, color='#1F77B4', linewidth=2.2, label='Plume LWC (g/m³)')
    ax1.set_ylabel('Liquid Water Content - LWC (g/m³)', color='#1F77B4', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1F77B4')
    ax1.set_ylim(0, 3.2)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax1_twin = ax1.twinx()
    l2 = ax1_twin.plot(z_vals, Delta_P_fan, color='#D62728', linewidth=2.2, linestyle='--', label='Fan Backpressure ΔP (Pa)')
    ax1_twin.set_ylabel('Backpressure on Fan Blades ΔP (Pa)', color='#D62728', fontsize=10, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor='#D62728')
    ax1_twin.set_ylim(0, 25)

    lines1 = l1 + l2
    labels1 = [line.get_label() for line in lines1]
    ax1.legend(lines1, labels1, loc='center right', fontsize=8.5, framealpha=0.9)
    ax1.set_title('Plume Thermodynamics vs. Aerodynamic Fan Penalty', fontsize=11, fontweight='bold')

    # Bottom Subplot: Net Fog Water Collection for multiple yields
    colores = ['#2CA02C', '#9467BD', '#8C564B']
    
    for idx, eta_base in enumerate(rendimientos_base):
        eta_captura = eta_base * (v_mean / v0)**0.65
        Agua_rec = m_dot_liq_total * eta_captura * frac_interceptada
        
        indice_opt = np.argmax(Agua_rec / (1.0 + 0.05 * Delta_P_fan))
        z_opt = z_vals[indice_opt]
        rec_opt = Agua_rec[indice_opt]
        
        ax2.plot(z_vals, Agua_rec, color=colores[idx % len(colores)], linewidth=2.2, 
                 label=f'Yield {int(eta_base*100)}% (Max: {rec_opt:.1f} L/h at z={z_opt:.1f}m)')
        
        # Mark optimal point for each yield
        ax2.plot(z_opt, rec_opt, marker='*', markersize=8, color=colores[idx % len(colores)])

    ax2.axvline(1.0, color='#7F8C8D', linestyle=':', linewidth=1.6, label='Current Proposal (z = 1.0 m)')
    ax2.axvspan(2.0, 3.5, color='#F39C12', alpha=0.15, label='Design Compromise Window (2.0 - 3.5 m)')

    ax2.set_xlabel('Vertical Distance Above Stack Discharge z (m)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Recovered Water (L/h)', color='#333333', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 120)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    ax2.set_title(f'Net Fog Water Collection by Mesh Yield (D ≈ {D_malla} m)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    if save_figure:
        plt.savefig(filename, dpi=300)
        print(f"Chart successfully saved as: {filename}")
    plt.show()

if __name__ == '__main__':
    simulate_mesh_height_optimization()