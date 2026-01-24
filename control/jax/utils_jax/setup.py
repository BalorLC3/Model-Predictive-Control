# setup.py
import jax
import jax.numpy as jnp
from functools import partial
from system.jax_ode_solver import rk4_step
import numpy as np

def load_driving_cycle(path="data/driving_energy.npy", fallback_len=600):
    try:
        raw = np.load(path, mmap_mode="r")
        print("Driving cycle loaded.")
        data = jnp.array(raw)
    except Exception:
        print("Driving cycle failed, using fallback.")
        data = jnp.abs(jnp.sin(jnp.arange(fallback_len) / 50.0)) * 20000.0

    N = len(data)
    dist = jnp.hstack([
        data.reshape(-1, 1),
        jnp.full((N, 1), 40.0),
    ])
    return dist


@partial(jax.jit, static_argnames=['dt', 'controller'])
def run_simulation(init_state_vec, controller, disturbances_array, params, dt):
    """
    Args:
        controller: Function with signature (state, env_carry, time_idx, params) -> (controls, new_carry)
    """
    
    N = disturbances_array.shape[0]
    time_indices = jnp.arange(N)
    
    # We zip inputs: ((P_driv, T_amb), time_idx)
    scan_inputs = (disturbances_array, time_indices)
    
    init_carry = (init_state_vec, 0.0) 

    def step_fn(carry, inputs):
        state, ctrl_carry = carry
        disturbance, k = inputs # Unpack time index k
        
        # ---  Pass Full State and Time to Controller ---
        controls, new_ctrl_carry = controller(state, ctrl_carry, k, params)
        
        # Physics Step
        next_state, diagnostics = rk4_step(state, controls, disturbance, params, dt)
        
        output = {
            'state': state,
            'controls': controls,
            'diagnostics': diagnostics
        }
        
        return (next_state, new_ctrl_carry), output

    _, history = jax.lax.scan(step_fn, init_carry, scan_inputs)
    
    return history

