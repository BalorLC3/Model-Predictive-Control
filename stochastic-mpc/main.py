import matplotlib.pyplot as plt
import numpy as np
from sys_dynamics_casadi import BatteryThermalSystem, SystemParameters
from setup import SimConfiguration, run_simulation
from controllers import Thermostat, DMPC, SMPC
from plot_utils import plot_results

if __name__ == "__main__":
    try: 
        driving_data = np.load('driving_energy.npy', mmap_mode='r')
        velocity_data = np.load('driving_velocity.npy', mmap_mode='r')
        print('Datos cargados.')
    except:
        print("Change directory, files not found")
    config = SimConfiguration(
        driving_data = driving_data,
        velocity_data = velocity_data,
        T_amb = 40.0,
        dt = 1.0
    )

    params = SystemParameters()
    init_state = {'T_batt': 30.0, 'T_clnt': 30.0, 'soc': 0.8}
    T_des = 33.0
    horizon = 5
    # ==========================================
    # SIMULACIÓN TERMOSTATO (BASELINE)
    # ==========================================
    print("\n--- Ejecutando Termostato ---")
    env_thermo = BatteryThermalSystem(init_state, params) # Instancia nueva
    ctrl_thermo = Thermostat()
    
    df_thermo = run_simulation(env_thermo, ctrl_thermo, config)
    
    # plot_results(df_thermo)

    
    # ==========================================
    # SIMULACIÓN NMPC (DETERMINISTA)
    # ==========================================
    # print("\n--- Ejecutando DMPC ---")    
    # ctrl_DMPC = DMPC(
    #     dt=1.0, 
    #     T_des=T_des,
    #     horizon=horizon,     # Before horizon 5
    #     alpha=0.065,
    #     avg_window=15
    # )
    # env_dmpc = BatteryThermalSystem(init_state, params)    
    # df_dmpc = run_simulation(env_dmpc, ctrl_DMPC, config)
    # plot_results(df_dmpc, 'dmpc')
    
    # ==========================================
    # SIMULACIÓN SMPC (ESTOCASTICO)
    # ==========================================
    print("\n--- Ejecutando SMPC ---")
    ctrl_SMPC = SMPC(
        driving_data,
        velocity_data,
        dt = 1.0,
        T_des=T_des,
        horizon=horizon,
        alpha=0.0016, # 0.016 yields good results
        n_clusters=4
    )

    env_smpc = BatteryThermalSystem(init_state, params)
    df_smpc = run_simulation(env_smpc, ctrl_SMPC, config)
    plot_results(df_smpc, 'smpc')
    plt.show()

    # ==========================================
    # COMPARACIÓN DE RESULTADOS
    # ==========================================
    print("\nGenerando comparativa...")    
    e_thermo = df_thermo['P_cooling'].sum() * 1.0 / 3.6e6
    # e_dmpc = df_dmpc['P_cooling'].sum() * 1.0 / 3.6e6
    e_smpc = df_smpc['P_cooling'].sum() * 1.0 / 3.6e6


    print(f"Energía Termostato: {e_thermo:.4f} kWh")
    # print(f"Energía DMPC:       {e_dmpc:.4f} kWh")
    print(f"Energía SMPC:       {e_smpc:.4f} kWh")
