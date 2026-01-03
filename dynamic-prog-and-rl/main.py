# All time modules
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
# Own modules
from system.sys_dynamics_jax import SystemParameters
from controllers.dynamic_programming import run_dp_offline, make_dp_controller_fn
from controllers.thermostat import thermostat_logic_jax
from utils.setup import run_simulation
from utils.plot_funs import plot_results

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
    
    plot_results(df, controller_name)
    plt.show()


if __name__ == "__main__":
    # ===============================================================
    # SETUP (N, data, disturbances and parameters)
    # ===============================================================
    try:
        raw = np.load('driving_energy.npy', mmap_mode='r')
        data = jnp.array(raw)
        print("Driving cycle loaded.")
    except:
        data = jnp.abs(jnp.sin(jnp.arange(600)/50.0)) * 20000.0
    N = len(data)
    dist = jnp.hstack([data.reshape(-1,1), jnp.full((N,1), 40.0)])
    params = SystemParameters()
    
    # ===============================================================
    # CONTROLLER 
    # ===============================================================
    controller_name = "thermostat"
    T_des = 33

    if controller_name == "dp":
        policy_cube = run_dp_offline(dist, params, alpha=0.05, T_des=T_des)
        controller_func = make_dp_controller_fn(policy_cube)
    elif controller_name == "thermostat":
        controller_func = thermostat_logic_jax
    # ===============================================================
    # SHARED FUNS (initial state and always must yield history)
    # ===============================================================
    init_state = jnp.array([30.0, 30.0, 0.8])
    history = run_simulation(init_state, controller_func, dist, params, 1.0)
    # ===============================================================
    # RESULTS 
    # ===============================================================
    states_hist = np.array(history['state'])
    ctrl_hist = np.array(history['controls'])
    diag_hist = np.array(history['diagnostics'])
    
    show_results(controller_name, N, states_hist, ctrl_hist, diag_hist)
