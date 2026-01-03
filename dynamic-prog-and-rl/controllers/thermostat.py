import jax.numpy as jnp

def thermostat_logic_jax(state, prev_carry, T_des, k, params):
    """
    Implements Thermostat with Hysteresis in pure JAX.
    Signature matches the Universal Runner used for DP.
    
    Args:
        state: [T_batt, T_clnt, soc]
        prev_carry: 1.0 (On) or 0.0 (Off) from previous step
        k: Current time index (unused here, but required by signature)
        params: SystemParameters object (unused here)
        
    Returns:
        controls: [w_comp, w_pump]
        next_carry: The new cooling state (0.0 or 1.0)
    """
    # 1. Unpack Inputs
    T_batt = state[0]
    cooling_state_prev = prev_carry 
    
    # 2. Thresholds
    T_upper = T_des
    T_lower = 32.5
    
    # 3. Hysteresis Logic
    should_turn_on = T_batt >= T_upper
    should_turn_off = T_batt <= T_lower
    
    # Next State = 1 if High, 0 if Low, else Previous
    cooling_state_next = jnp.where(should_turn_on, 1.0, 
                         jnp.where(should_turn_off, 0.0, cooling_state_prev))
    
    # 4. Actuation
    w_comp = 3000.0 * cooling_state_next
    w_pump = 2000.0 * cooling_state_next
    
    # 5. Return
    return jnp.array([w_comp, w_pump]), cooling_state_next