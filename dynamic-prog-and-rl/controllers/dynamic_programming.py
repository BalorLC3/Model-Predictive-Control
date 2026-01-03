import jax
import jax.numpy as jnp
from functools import partial
from system.jax_ode_solver import rk4_step
import time

# --- 1. CONFIGURATION ---
TB_MIN_GRID, TB_MAX_GRID, TB_N = 25.0, 45.0, 51   # The GRID limits
TC_MIN_GRID, TC_MAX_GRID, TC_N = 20.0, 40.0, 41

TB_GRID = jnp.linspace(TB_MIN_GRID, TB_MAX_GRID, TB_N)
TC_GRID = jnp.linspace(TC_MIN_GRID, TC_MAX_GRID, TC_N)

# Action Space
W_COMP_OPTS = jnp.linspace(0.0, 10000.0, 8)
W_PUMP_OPTS = jnp.linspace(0.0, 10000.0, 8)
U_GRID = jnp.dstack(jnp.meshgrid(W_COMP_OPTS, W_PUMP_OPTS)).reshape(-1, 2)

# --- CONSTRAINT CONFIG (Matching SMPC) ---
T_LIMIT_MIN = 30.0
T_LIMIT_MAX = 35.0
RHO_SOFT = 0.2     # Same penalty weight as SMPC

def get_normalized_coords(T_batt, T_clnt):
    """Convert physical Temp to grid coordinates (0.0 to N-1.0)"""
    T_b_safe = jnp.clip(T_batt, TB_MIN_GRID, TB_MAX_GRID)
    T_c_safe = jnp.clip(T_clnt, TC_MIN_GRID, TC_MAX_GRID)
    
    idx_b = (T_b_safe - TB_MIN_GRID) / (TB_MAX_GRID - TB_MIN_GRID) * (TB_N - 1)
    idx_c = (T_c_safe - TC_MIN_GRID) / (TC_MAX_GRID - TC_MIN_GRID) * (TC_N - 1)
    return jnp.stack([idx_b, idx_c])

@partial(jax.jit, static_argnames=['dt'])
def bellman_update(cost_to_go_next, disturbance, params, dt):
    # Create mesh of all states [TB, TC]
    tb_mesh, tc_mesh = jnp.meshgrid(TB_GRID, TC_GRID, indexing='ij')
    states_flat = jnp.stack([
        tb_mesh.flatten(),
        tc_mesh.flatten(),
        jnp.full(tb_mesh.size, 0.5) 
    ], axis=1)

    def evaluate_state_action(state, action):
        next_state, diag = rk4_step(state, action, disturbance, params, dt)
        
        # 1. Energy Cost (kJ = kW * s) 
        # Matching SMPC: (P_batt + P_comp) * dt
        # diag[1] is P_batt (W), diag[7] is P_comp (W) -> (W + W)/1000 = kW
        P_total_kW = (diag[1] + diag[7]) / 1000.0 
        energy_cost = P_total_kW * (dt / 3600.0)

        # 2. Soft Constraint Penalty (Slack) 
        # Violation = max(0, T - T_max) + max(0, T_min - T)
        # Using softplus or relu for differentiability isn't needed in DP (discrete lookup), 
        # jnp.maximum is fast.
        T_next = next_state[0]
        
        viol_upper = jnp.maximum(0.0, T_next - T_LIMIT_MAX)
        viol_lower = jnp.maximum(0.0, T_LIMIT_MIN - T_next)
        
        # Slack Cost = rho * (S^2)
        # Note: In SMPC we had separate slacks S_up and S_low.
        slack_cost = RHO_SOFT * (viol_upper**2 + viol_lower**2)

        stage_cost = energy_cost + slack_cost

        # 3. Future Cost (Interpolation) 
        coords = get_normalized_coords(next_state[0], next_state[1])
        future_cost = jax.scipy.ndimage.map_coordinates(
            cost_to_go_next, coords, order=1, mode='nearest'
        )
        
        return stage_cost + future_cost

    # Vectorize: (States x Actions)
    evaluate_all = jax.vmap(jax.vmap(evaluate_state_action, in_axes=(None, 0)), in_axes=(0, None))
    
    # matrix shape: (N_states, N_actions)
    all_costs = evaluate_all(states_flat, U_GRID)
    
    # Min over actions
    best_indices = jnp.argmin(all_costs, axis=1)
    best_costs = jnp.min(all_costs, axis=1)
    
    return best_costs.reshape(TB_N, TC_N), best_indices.reshape(TB_N, TC_N)

def run_dp_offline(disturbances, params, dt=1.0, alpha=1.0, T_des=33.0):
    """
    alpha: Terminal cost weight (set to 0.0 to match constraint-riding behavior)
    """
    N = len(disturbances)
    # Terminal cost
    J_next = alpha * (TB_GRID[:, None] - T_des)**2
    
    policy_history = []
    
    print(f"Solving DP Backward ({N} steps)...")
    start = time.time()
    
    for k in range(N - 1, -1, -1):
        # Pass dynamic disturbance[k]
        J_curr, Pol_idx = bellman_update(J_next, disturbances[k], params, dt)
        policy_history.append(Pol_idx)
        J_next = J_curr
        if k % 1000 == 0: print(f"Step {k}")
            
    print(f"Done in {time.time()-start:.2f}s")
    # Reverse policy because we solved backward
    return jnp.stack(policy_history[::-1]) 

def make_dp_controller_fn(policy_cube):
    def dp_controller(state, carry, k, params):
        coords = get_normalized_coords(state[0], state[1])
        policy_at_k = policy_cube[k]
        u_idx_float = jax.scipy.ndimage.map_coordinates(
            policy_at_k, coords, order=0, mode='nearest'
        )
        u_idx = u_idx_float.astype(int)
        controls = U_GRID[u_idx]
        return controls, carry
    return dp_controller