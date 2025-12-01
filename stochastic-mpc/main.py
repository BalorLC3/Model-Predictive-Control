import matplotlib.pyplot as plt
import numpy as np
from sys_dynamics import BatteryThermalSystem, SystemParameters
from setup import SimConfiguration, run_simulation
from controllers import Thermostat
from plot_utils import plot_results, plot_signal

if __name__ == "__main__":
    try: 
        driving_data = np.load('driving_energy.npy', mmap_mode='r')
    except:
        t_synth = np.arange(0, 2740)
        driving_data = np.abs(np.sin(t_synth/50)) * 20000

    config = SimConfiguration(
        driving_data = driving_data,
        T_amb = 40.0,
        dt = 1.0
    )

    params = SystemParameters()
    init_state = {'T_batt': 30.0, 'T_clnt':30.0, 'soc':0.8}

    env = BatteryThermalSystem(init_state, params)
    ctrl_thermo = Thermostat()
    df_thermo = run_simulation(env, ctrl_thermo, config)

    plot_results(df_thermo)
