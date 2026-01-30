# All time modules
import numpy as np
import jax
import jax.numpy as jnp
# Own modules
from control.jax.system.sys_dynamics_jax import SystemParameters
from control.jax.controllers.dynamic_programming import run_dp_offline, make_dp_controller_fn
from control.jax.controllers.thermostat import thermostat_logic_jax
from control.jax.reinforcement_learning.sac import SBXActor
from control.jax.utils.setup import run_simulation, load_driving_cycle
from control.jax.utils.performance import time_total_simulation
from control.utils.plot_helper import show_results 
from control.jax.env.env_batt import ObservationConfig
import pickle

def make_controller(controller_name, dist, params, horizon):
    """
    Returns: controller_fn, controller_metadata (dict)
    """
    if controller_name == "dp":
        policy_cube = run_dp_offline(dist, params, alpha=100.0)
        return make_dp_controller_fn(policy_cube), {}

    if controller_name == "thermostat":
        return thermostat_logic_jax, {}

    if controller_name.startswith("sac"):
        with open(f"control/jax/results/{controller_name}/actor_weights.pkl", "rb") as f:
            params_nn = pickle.load(f)

        actor = SBXActor(n_actions=2)
        obs_config = ObservationConfig(horizon=horizon)
        def get_obs(state, disturbance, preview): 
            raw = jnp.concatenate([state, disturbance, preview]) 
            mean = jnp.concatenate([ 
                obs_config.obs_mean,                     
                jnp.full((obs_config.horizon,), 10000.0) 
            ]) 
            scale = jnp.concatenate([ 
                obs_config.obs_scale, jnp.full((obs_config.horizon,), 10000.0) 
            ]) 
            return (raw - mean) / scale

        def controller_fn(state, carry, k, params_sys):
            d_curr = dist[k]

            preview = jnp.zeros((obs_config.horizon,))
            if obs_config.horizon > 0:
                preview = jax.lax.dynamic_slice(
                    dist,
                    (k + 1, 0),
                    (obs_config.horizon, 1),
                ).reshape(-1)

            obs = get_obs(state, d_curr, preview)
            mean = actor.apply(params_nn, obs)
            action = jnp.tanh(mean)

            controls = (action + 1.0) * 5000.0
            return controls, carry

        return controller_fn, {"actor": actor}
    
    raise ValueError(f"Unknown controller: {controller_name}")

def run_and_time(controller_fn, init_state, dist, params, dt):
    total_time = time_total_simulation(
        init_state,
        controller_fn,
        dist,
        params,
        dt,
    )

    history = run_simulation(init_state, controller_fn, dist, params, dt)

    return history, total_time

if __name__ == "__main__":

    # ---------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------
    controller_name = "thermostat"; horizon = 0
    dt = 1.0

    dist = load_driving_cycle()
    params = SystemParameters()

    init_state = jnp.array([30.0, 30.0, 0.8])

    # ---------------------------------------------------------------
    # Controller
    # ---------------------------------------------------------------
    controller_fn, ctrl_meta = make_controller(
        controller_name,
        dist,
        params,
        horizon
    )

    # ---------------------------------------------------------------
    # Simulation + timing
    # ---------------------------------------------------------------
    history, sim_time = run_and_time(
        controller_fn,
        init_state,
        dist,
        params,
        dt,
    )

    print(
        f"[{controller_name}] "
        f"Total simulation time: {sim_time:.3f} s | "
        f"Average time per step (dt = {dt} s): {sim_time/len(dist):.6f} s"
    )

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------
    states = np.asarray(history["state"])
    controls = np.asarray(history["controls"])
    diagnostics = np.asarray(history["diagnostics"])

    show_results(
        states_hist=states, 
        ctrl_hist=controls, 
        diag_hist=diagnostics, 
        controller_name=controller_name, 
        config='horizontal'
    )




