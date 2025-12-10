import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "figure.figsize": (6.0, 10.0), # use (4.0, 10.0) for side by side in A4 paper
    "lines.linewidth": 1.4,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.dpi": 300
})

def plot_signal(x, y, ylabel='', xlabel='Time (s)', color='b'):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, color=color, linewidth=1.5)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True)
    
def plot_results(df_controller, name='thermostat', dt=1.0, ): # Added dt argument
    time = df_controller['time']
    # Extract data
    Q_cool = df_controller['Q_cool']
    # Q_gen = df_controller['Q_gen'] # Un-comment if you want to plot this too
    T_batt = df_controller['T_batt']
    T_clnt = df_controller['T_clnt']
    w_pump = df_controller['w_pump']
    w_comp = df_controller['w_comp']
    P_cooling = df_controller['P_cooling']

    fig, axs = plt.subplots(5, 1, sharex=True, layout='constrained')

    # --- 1. Heat Transfer ---
    axs[0].plot(time, Q_cool, 'b', lw=1.5)
    axs[0].set_ylabel(r'Heat Removed' + '\n' + r'($\dot{Q}_{cool}$) [W]') 
    axs[0].set_xlim(0, len(time))
    axs[0].set_ylim(0, 2000)
    
    # --- 2. Pump Speed ---
    axs[1].plot(time, w_pump, 'r')
    axs[1].set_ylabel(r'Pump Speed' + '\n' + r'($\omega_{pump}$) [RPM]')
    axs[1].set_xlim(0, len(time))
    axs[1].set_ylim(0, 10000)

    # --- 3. Compressor Speed ---
    axs[2].plot(time, w_comp, 'k')
    axs[2].set_ylabel(r'Comp Speed' + '\n' + r'($\omega_{comp}$) [RPM]')
    axs[2].set_xlim(0, len(time))
    axs[2].set_ylim(0, 10000)
    
    # --- 4. Temperatures ---
    axs[3].plot(time, T_batt, 'r', label='Battery ($T_{batt}$)')
    axs[3].plot(time, T_clnt, 'b--', label='Coolant ($T_{clnt}$)')
    axs[3].set_ylabel('Temperature\n[$^\circ$C]')
    axs[3].legend(loc='upper left', frameon=True) 
    axs[3].set_xlim(0, len(time))
    axs[3].set_ylim(28, 35)

    # --- 5. Energy Consumption ---
    joules_to_kJ = 1/1000
    energy_cooling_kJ = np.cumsum(P_cooling) * dt * joules_to_kJ
    
    axs[4].plot(time, energy_cooling_kJ, 'g')
    axs[4].set_xlabel('Time (s)')
    axs[4].set_ylabel('Cooling Energy\nConsumed [kJ]')
    axs[4].grid(True, which='both')
    axs[4].set_xlim(0, len(time))
    axs[4].set_ylim(0, 400)

    save_dir = Path(r"C:\Users\super\Desktop\Vasudeva\Manifold\engineering\predictive-control\xresults_btm")    
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{name}_controller.pdf", bbox_inches='tight')