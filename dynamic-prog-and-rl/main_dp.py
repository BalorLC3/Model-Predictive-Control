import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import jax

from sys_dynamics_jax import SystemParameters
from dp_core import run_dp_offline, make_dp_controller_fn
from setup import run_simulation
from plot_utils import plot_results


if __name__ == "__main__":
    # --- SETUP ---
    try:
        raw = np.load('driving_energy.npy', mmap_mode='r')
        data = jnp.array(raw)
    except:
        data = jnp.abs(jnp.sin(jnp.arange(600)/50.0)) * 20000.0
        
    N = len(data)
    dist = jnp.hstack([data.reshape(-1,1), jnp.full((N,1), 40.0)])
    params = SystemParameters()
    
    policy_cube = run_dp_offline(dist, params, alpha=0.05)
    
    dp_ctrl_fn = make_dp_controller_fn(policy_cube)
    
    init_state = jnp.array([30.0, 30.0, 0.8])
    history = run_simulation(init_state, dp_ctrl_fn, dist, params, 1.0)
    
    # Extraemos las matrices de JAX a Numpy para Pandas
    states_hist = np.array(history['state'])
    ctrl_hist = np.array(history['controls'])
    diag_hist = np.array(history['diagnostics'])
    
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
    
    print(f"Total Energy: {(df['P_cooling'].sum()/3600/1000):.4f} kWh")
    
    plot_results(df, 'dp')
    plt.show()