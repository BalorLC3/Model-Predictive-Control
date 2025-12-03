import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import pandas as pd
import time

from sys_dynamics_jax import SystemParameters
from setup import run_simulation
from plot_utils import plot_results
from thermostat import thermostat_logic_jax # Assuming you saved it here

if __name__ == "__main__":
    # 1. Data Loading
    try: 
        raw_data = np.load('driving_energy.npy', mmap_mode='r')
        driving_data = jnp.array(raw_data)
        print("Driving cycle loaded.")
    except:
        print("Using synthetic data.")
        t_synth = jnp.arange(0, 2740)
        driving_data = jnp.abs(jnp.sin(t_synth/50.0)) * 20000.0

    # 2. Prepare Inputs
    N = len(driving_data)
    T_amb_seq = jnp.full((N, 1), 40.0) # 40°C Ambient
    P_driv_seq = driving_data.reshape(-1, 1)
    
    # Shape (N, 2)
    disturbances_array = jnp.hstack([P_driv_seq, T_amb_seq])

    # 3. Setup System
    params = SystemParameters()
    init_state = jnp.array([30.0, 30.0, 0.8])

    # 4. Run (Compilation happens here on first run)
    print("Simulation Thermostat Jax")
    start = time.time()
    history = run_simulation(init_state, thermostat_logic_jax, disturbances_array, params, 1.0)
    print(f"Done in {time.time()-start:.2f}s")
    
    # Force computation and move to CPU for plotting
    # (Pandas prefers numpy arrays over JAX device arrays)
    state_hist = np.array(history['state'])
    ctrl_hist = np.array(history['controls'])
    diag_matrix = np.array(history['diagnostics'])
    
    # 5. Process Results
    results = {
        'time': np.arange(N),
        'T_batt': state_hist[:, 0],
        'T_clnt': state_hist[:, 1],
        'w_comp': ctrl_hist[:, 0],
        'w_pump': ctrl_hist[:, 1],
        'P_cooling': diag_matrix[:, 0],
        'Q_gen': diag_matrix[:, 4],
        'Q_cool': diag_matrix[:, 5],
    }
    
    df = pd.DataFrame(results)
    
    print("Plotting...")
    plot_results(df)
    plt.show()