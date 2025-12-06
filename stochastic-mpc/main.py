import matplotlib.pyplot as plt
import numpy as np
from sys_dynamics_casadi import BatteryThermalSystem, SystemParameters
from setup import SimConfiguration, run_simulation
from controllers import Thermostat, NMPC    
from plot_utils import plot_results

if __name__ == "__main__":
    try: 
        driving_data = np.load('driving_energy.npy', mmap_mode='r')
        driving_data_sim = driving_data[:1000] 
        print('Datos cargados.')
    except:
        t_synth = np.arange(0, 1000)
        driving_data_sim = np.abs(np.sin(t_synth/50)) * 20000
        print('Usando datos sintéticos.')

    config = SimConfiguration(
        driving_data = driving_data,
        T_amb = 40.0,
        dt = 1.0
    )

    params = SystemParameters()
    init_state = {'T_batt': 30.0, 'T_clnt': 30.0, 'soc': 0.8}

    # ==========================================
    # 1. SIMULACIÓN TERMOSTATO (BASELINE)
    # ==========================================
    print("\n--- Ejecutando Termostato ---")
    env_thermo = BatteryThermalSystem(init_state, params) # Instancia nueva
    ctrl_thermo = Thermostat()
    
    df_thermo = run_simulation(env_thermo, ctrl_thermo, config)
    
    plot_results(df_thermo)
    plt.savefig('thermostat_control.pdf')

    plt.show()
    
    # ==========================================
    # 2. SIMULACIÓN NMPC (DETERMINISTA)
    # ==========================================
    print("\n--- Ejecutando NMPC ---")    
    ctrl_NMPC = NMPC(
        driving_data=driving_data, 
        dt=1.0, 
        horizon=5, 
    )
    env_mpc = BatteryThermalSystem(init_state, params)    
    df_mpc = run_simulation(env_mpc, ctrl_NMPC, config)
    
    # ==========================================
    # 3. COMPARACIÓN DE RESULTADOS
    # ==========================================
    print("\nGenerando comparativa...")    
    e_thermo = df_thermo['P_cooling'].sum() * 1.0 / 3.6e6
    e_mpc = df_mpc['P_cooling'].sum() * 1.0 / 3.6e6
    
    print(f"Energía Termostato: {e_thermo:.4f} kWh")
    print(f"Energía SMPC:       {e_mpc:.4f} kWh")
    print(f"Ahorro:             {(1 - e_mpc/e_thermo)*100:.2f}%")

    plot_results(df_mpc)
    plt.savefig('nmpc_control.pdf')


    plt.show()