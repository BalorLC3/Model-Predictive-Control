import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import time
from functools import partial
import jax

from sys_dynamics_jax import SystemParameters
from dp_core import bellman_update, TB_N, TC_N, U_GRID, get_normalized_coords
from setup import run_simulation
from plot_utils import plot_results

# 1. THE SOLVER LOOP
def run_dp_offline(disturbances, params, dt=1.0):
    N = len(disturbances)
    J_next = jnp.zeros((TB_N, TC_N))
    policy_history = []
    
    print(f"Solving DP Backward ({N} steps)...")
    start = time.time()
    
    for k in range(N - 1, -1, -1):
        J_curr, Pol_idx = bellman_update(J_next, disturbances[k], params, dt)
        policy_history.append(Pol_idx)
        J_next = J_curr
        if k % 500 == 0: print(f"Step {k}")
            
    print(f"Done in {time.time()-start:.2f}s")
    return jnp.stack(policy_history[::-1]) # Shape (N, TB_N, TC_N)

# 2. THE CONTROLLER (Uses the Solved Policy)
def make_dp_controller_fn(policy_cube):
    """Returns a function compatible with run_simulation"""
    
    def dp_controller(state, carry, k, params):
        # state: [T_batt, T_clnt, soc]
        # k: time index
        
        # 1. Get Coordinates
        coords = get_normalized_coords(state[0], state[1])
        
        # 2. Lookup Optimal Action Index for THIS time step k
        # We slice the cube at time k -> (TB_N, TC_N)
        policy_at_k = policy_cube[k]
        
        # We round to nearest integer index
        u_idx_float = jax.scipy.ndimage.map_coordinates(
            policy_at_k, coords, order=0, mode='nearest'
        )
        u_idx = u_idx_float.astype(int)
        
        # 3. Map Index to RPMs
        controls = U_GRID[u_idx]
        
        return controls, carry # Carry unused

    return dp_controller

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
    
    policy_cube = run_dp_offline(dist, params)
    
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
    
    plot_results(df)
    plt.savefig('dp_solver.pdf')
    plt.show()