import jax
import jax.numpy as jnp
import numpy as np # Used only for loading data initially
import os

# ===============================================================
# JAX BATTERY MODELS MODULE
# Optimized for MPC / DP / RL
# ===============================================================

# --- 1. GLOBAL PARAMETERS (Converted to JAX Arrays) ---

# OCV Model Data (Table 7)
OCV_TEMPS = jnp.array([5.0, 15.0, 25.0, 45.0])

# Shape: (4 temps, 5 parameters)
OCV_PARAMS_CHARGE = jnp.array([
    [3.734, -0.2756, 8.013e-5, -0.001155, -0.06674],
    [3.629, -0.3191, 6.952e-5, -0.0005411, -0.1493],
    [3.637, -0.3091, 7.033e-5, -0.0005477, -0.1366],
    [3.584, -0.4868, 6.212e-5, -0.0001919, -0.2181],
])

OCV_PARAMS_DISCHARGE = jnp.array([
    [3.804, -0.3487, 8.838e-5, -0.001618, -0.04724],
    [3.599, -0.2933, 6.9e-5,   -0.0004687, -0.1434],
    [3.604, -0.2803, 6.871e-5, -0.0004523, -0.1341],
    [3.55,  -0.4573, 6.02e-5,  -6.241e-5,  -0.221],
])

# Resistance & Capacity Data (Table 2)
RES_TEMPS = jnp.array([5.0, 15.0, 25.0, 45.0])
CNOM_TABLE = jnp.array([17.17, 19.24, 20.0, 21.6])
R0_TABLE = jnp.array([0.007, 0.0047, 0.003, 0.0019])
R1_TABLE = jnp.array([0.0042, 0.0018, 0.00065, 0.00054])


# ===============================================================
# --- 2. ENTROPIC HEAT DATA  ---
# ===============================================================

def _load_entropy_constants():
    """
    Loads the .npz file using standard NumPy and converts to JAX arrays.
    These act as immutable constants for the JIT compiler.
    """
    soc_grid = jnp.array([-1.000e-04, 4.990e-02, 9.990e-02, 1.499e-01, 1.999e-01, 2.500e-01,
            3.000e-01, 3.500e-01, 4.000e-01, 4.500e-01, 5.000e-01, 5.500e-01,
            6.000e-01, 6.500e-01, 7.000e-01, 7.500e-01, 8.000e-01, 8.500e-01,
            9.000e-01, 9.500e-01, 1.000e+00])

    dvdt_grid = jnp.array([-5.05259901e-04, -3.86484957e-04, -1.83862780e-04, -1.45203914e-04,
             -1.01362908e-04, -7.95978631e-05, -2.97455572e-05, 6.21858344e-05,
             1.06130487e-04, 1.28724671e-04, 1.39088981e-04, 1.38363481e-04,
             1.33595895e-04, 7.88723628e-05, -3.42022076e-06, -1.48209565e-05,
             -2.12468258e-05, -2.45634037e-05, -3.57568533e-05, -3.33730641e-05,
             -3.50313530e-05])

    return soc_grid, dvdt_grid

# Initialize the constants at import time
# This ensures they are loaded once and baked into the JIT graph
SOC_GRID, DVDT_GRID = _load_entropy_constants()

# ===============================================================
# --- 3. JAX FUNCTIONS ---
# ===============================================================

@jax.jit
def get_dvdt_jax(soc):
    """
    Calculates entropic heat coefficient (dV/dT) [V/K].
    JAX's interp is fully differentiable, so this works for MPC/Gradient Descent.
    
    Args:
        soc: State of Charge (0.0 to 1.0)
    """
    # jnp.interp(x, xp, fp)
    return jnp.interp(soc, SOC_GRID, DVDT_GRID)

@jax.jit
def get_ocv_jax(soc, temp, p_batt_total):
    """
    JAX-compatible OCV calculation.
    
    Args:
        soc: State of charge (0.0 to 1.0)
        temp: Temperature [C]
        p_batt_total: Total battery power [W]. Used to determine charge/discharge.
                      (Negative power = Charging)
    """
    soc_percent = soc * 100.0
    
    # Check mode: True if Charging (Power < 0)
    is_charging = p_batt_total < 0
    
    # We must calculate parameters for BOTH cases because JAX 
    # executes all branches in the graph,. we select the result at the end.
    
    # Helper to interpolate a column of params
    def interp_param(col_idx, table):
        return jnp.interp(temp, OCV_TEMPS, table[:, col_idx])

    # 1. Get charge params [p0, p1, p2, a1, a2]
    p_c = [interp_param(i, OCV_PARAMS_CHARGE) for i in range(5)]
    
    # 2. Get Discharge Params
    p_d = [interp_param(i, OCV_PARAMS_DISCHARGE) for i in range(5)]
    
    # 3. Select Parameters based on mode
    # jnp.where(condition, if_true, if_false)
    p0 = jnp.where(is_charging, p_c[0], p_d[0])
    p1 = jnp.where(is_charging, p_c[1], p_d[1])
    p2 = jnp.where(is_charging, p_c[2], p_d[2])
    a1 = jnp.where(is_charging, p_c[3], p_d[3])
    a2 = jnp.where(is_charging, p_c[4], p_d[4])
    
    # 4. Calculate OCV
    ocv = p0 * jnp.exp(a1 * soc_percent) + \
          p1 * jnp.exp(a2 * soc_percent) + \
          p2 * (soc_percent ** 2)
          
    return ocv

@jax.jit
def get_rbatt_jax(soc, temp):
    """
    Calculates internal resistance (R_pack).
    """
    SCALE_FACTOR = 3.0
    
    # Interpolate R0 and R1
    r0_cell = jnp.interp(temp, RES_TEMPS, R0_TABLE)
    r1_cell = jnp.interp(temp, RES_TEMPS, R1_TABLE)
    
    r_base_cell = r0_cell + r1_cell    
    r_total_cell = r_base_cell 
    r_total_pack = r_total_cell / SCALE_FACTOR
    
    return r_total_pack

@jax.jit
def get_cnom_jax(temp):
    """
    Calculates nominal capacity [Ah].
    """
    return jnp.interp(temp, RES_TEMPS, CNOM_TABLE)