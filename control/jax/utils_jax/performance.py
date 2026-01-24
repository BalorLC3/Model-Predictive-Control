import time
from utils.setup import run_simulation
import jax

def time_total_simulation(
    init_state_vec,
    controller,
    disturbances_array,
    params,
    dt,
):
    # --- Warm-up (JIT compilation) ---
    out = run_simulation(
        init_state_vec,
        controller,
        disturbances_array,
        params,
        dt,
    )
    out['state'].block_until_ready()

    # --- Timed run ---
    t0 = time.perf_counter()

    out = run_simulation(
        init_state_vec,
        controller,
        disturbances_array,
        params,
        dt,
    )
    out['state'].block_until_ready()

    t1 = time.perf_counter()

    return t1 - t0
