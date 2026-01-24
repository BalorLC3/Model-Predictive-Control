import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei"], # Change this to tex True & unicode to True
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "figure.figsize": (4.0, 10.0), # TO paste in the Thesis is okay to use (4.0, 10.0), but (6.0, 10.0) is better for normal visualizaton
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
    plt.show()
    
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
    axs[0].set_ylabel(r'Calor Removido' + '\n' + r'($\dot{Q}_{cool}$) [W]') 
    axs[0].set_xlim(0, len(time))
    axs[0].set_ylim(0, 2000)
    
    # --- 2. Pump Speed ---
    axs[1].plot(time, w_pump, 'r')
    axs[1].set_ylabel(r'Vel. Bomba' + '\n' + r'($\omega_{pump}$) [RPM]')
    axs[1].set_xlim(0, len(time))
    axs[1].set_ylim(0, 10000)

    # --- 3. Compressor Speed ---
    axs[2].plot(time, w_comp, 'k')
    axs[2].set_ylabel(r'Vel. Compresor' + '\n' + r'($\omega_{comp}$) [RPM]')
    axs[2].set_xlim(0, len(time))
    axs[2].set_ylim(0, 10000)
    
    # --- 4. Temperatures ---
    axs[3].plot(time, T_batt, 'r', label='Bateria ($T_{batt}$)')
    axs[3].plot(time, T_clnt, 'b--', label='Refrigerante ($T_{clnt}$)')
    axs[3].set_ylabel(r'Temperatura' + '\n' + r'($T$) [$^\circ$C]')
    axs[3].legend(loc='upper left', frameon=True) 
    axs[3].set_xlim(0, len(time))
    axs[3].set_ylim(28, 35)

    # --- 5. Energy Consumption ---
    joules_to_kJ = 1/1000
    energy_cooling_kJ = np.cumsum(P_cooling) * dt * joules_to_kJ
    
    axs[4].plot(time, energy_cooling_kJ, 'g')
    axs[4].set_xlabel('Tiempo (s)')
    axs[4].set_ylabel('Energia de Enf.' + '\n' + r'($P_{cool}$) [kJ]')
    axs[4].grid(True, which='both')
    axs[4].set_xlim(0, len(time))
    axs[4].set_ylim(0, 400)

    save_dir = Path(r"C:\Users\super\Desktop\Vasudeva\Manifold\engineering\predictive-control\xresults_btm")    
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{name}_controller.png", bbox_inches='tight', dpi=300)
    plt.show()

def show_results(
        controller_name,
        N, 
        states_hist, 
        ctrl_hist, 
        diag_hist
    ):
    df = pd.DataFrame({
        'time': np.arange(N),
        # Estados
        'T_batt': states_hist[:, 0],
        'T_clnt': states_hist[:, 1],
        # Controles
        'w_comp': ctrl_hist[:, 0],
        'w_pump': ctrl_hist[:, 1],
        # Diagnósticos 
        'P_cooling': diag_hist[:, 0],
        'Q_gen':     diag_hist[:, 4],
        'Q_cool':    diag_hist[:, 5]
    })
    print(f"Total Energy: {(df['P_cooling'].sum()/1000):.4f} kJ")
    print(f"Final T_batt: \n{df[['time','T_batt']].tail(3)}")
    
    plot_results(df, controller_name)

def plot_learning_history(history):
    episodes = np.arange(len(history['ep_rewards']))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    ax1.plot(episodes, history['ep_energy_kj'], color='dodgerblue')
    ax1.set_ylabel('Energia'+ '\n' + r'Consumida [kJ]')

    ax2.plot(episodes, history['ep_avg_temp'], color='red')
    ax2.set_ylabel(r'Promedio $T_{batt}$ [°C]')
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(episodes[5:], history['ep_rewards'][5:], color='seagreen')
    ax.set_ylabel('Recompensa'+ '\n' + r'Cumulativa ($R$)')
    ax.set_xlabel('Episodio')
    ax.ticklabel_format(axis='y')
    plt.tight_layout()
    plt.show()




        


        