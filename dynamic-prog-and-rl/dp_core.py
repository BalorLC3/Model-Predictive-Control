# dp_core.py
import jax
import jax.numpy as jnp
from functools import partial
from jax_ode_solver import rk4_step

# --- 1. CONFIGURATION ---
TB_MIN, TB_MAX, TB_N = 25.0, 45.0, 51   # Finer grid (51 points)
TC_MIN, TC_MAX, TC_N = 20.0, 40.0, 41

TB_GRID = jnp.linspace(TB_MIN, TB_MAX, TB_N)
TC_GRID = jnp.linspace(TC_MIN, TC_MAX, TC_N)

# Action Space (16 combinations)
W_COMP_OPTS = jnp.linspace(0.0, 5000.0, 8)
W_PUMP_OPTS = jnp.linspace(0.0, 6000.0, 8)
U_GRID = jnp.dstack(jnp.meshgrid(W_COMP_OPTS, W_PUMP_OPTS)).reshape(-1, 2)

def get_normalized_coords(T_batt, T_clnt):
    """Convert physical Temp to grid coordinates (0.0 to N-1.0)"""
    # Clip to ensure we don't go out of bounds
    T_b_safe = jnp.clip(T_batt, TB_MIN, TB_MAX)
    T_c_safe = jnp.clip(T_clnt, TC_MIN, TC_MAX)
    
    idx_b = (T_b_safe - TB_MIN) / (TB_MAX - TB_MIN) * (TB_N - 1)
    idx_c = (T_c_safe - TC_MIN) / (TC_MAX - TC_MIN) * (TC_N - 1)
    return jnp.stack([idx_b, idx_c])

@partial(jax.jit, static_argnames=['dt'])
def bellman_update(cost_to_go_next, disturbance, params, dt):
    
    # Create State Grid
    tb_mesh, tc_mesh = jnp.meshgrid(TB_GRID, TC_GRID, indexing='ij')
    # Assume SOC = 0.5 (Midpoint) for planning
    states_flat = jnp.stack([tb_mesh.flatten(), tc_mesh.flatten(), jnp.full(tb_mesh.size, 0.5)], axis=1)

    def evaluate_state_action(state, action):
        next_state, diag = rk4_step(state, action, disturbance, params, dt)
        
        P_batt = diag[7] # Watts
        P_comp = diag[8] 
        P_total = P_comp + P_batt
        T_next = next_state[0]
        
        # --- COST FUNCTION ---
        # 1. Pure Energy (Joules -> normalized)
        # We scale it so 1 kWh ~ 1.0 cost unit roughly, to balance with penalty
        J_energy = P_total * dt / 3.6e3 
        
        # 2. Barrier Penalty (Safety)
        # If T > 34, cost explodes.
        J_safety = 1e5 * jnp.maximum(0.0, T_next - 35.0)**2
        
        # 3. Reference very weak pull to 25C to prefer cold start
        J_ref = 0.0014 * (T_next - 27.0)**2
        
        stage_cost = J_energy + J_safety + J_ref*0
        
        # --- INTERPOLATION ---
        coords = get_normalized_coords(next_state[0], next_state[1])
        # order=1 means Bilinear Interpolation (Smooth)
        future_cost = jax.scipy.ndimage.map_coordinates(cost_to_go_next, coords, order=1, mode='nearest')
        
        return stage_cost + future_cost

    # Vectorize
    evaluate_all = jax.vmap(jax.vmap(evaluate_state_action, in_axes=(None, 0)), in_axes=(0, None))
    all_costs = evaluate_all(states_flat, U_GRID)
    
    # Min operation
    best_indices = jnp.argmin(all_costs, axis=1)
    best_costs = jnp.min(all_costs, axis=1)
    
    return best_costs.reshape(TB_N, TC_N), best_indices.reshape(TB_N, TC_N)