import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax
import jax.numpy as jnp
from system.sys_dynamics_jax import SystemParameters
from system.jax_ode_solver import rk4_step
from dataclasses import dataclass, field


@dataclass
class ObservationConfig:
    horizon: int = field(
        default_factory=lambda: 20
    )
    obs_mean: jnp.ndarray = field(
        default_factory=lambda: jnp.array([32.0, 32.0, 0.5, 10000.0, 25.0])
    )
    obs_scale: jnp.ndarray = field(
        default_factory=lambda: jnp.array([15.0, 15.0, 0.5, 10000.0, 15.0])
    )


class BatteryCoolingEnv(gym.Env):
    """
    Standard Gymnasium Environment for Stable Baselines3.
    """

    def __init__(self, render_mode=None, def_horizon=0):
        super().__init__()

        try:
            raw = np.load("data/driving_energy.npy", mmap_mode="r")
            self.driving_data = jnp.array(raw)
        except Exception:
            time = jnp.arange(0, 3600)
            self.driving_data = jnp.abs(jnp.sin(time / 50.0)) * 20000.0
        self.def_horizon = def_horizon
        self.params = SystemParameters()
        self.obs_config = ObservationConfig(horizon=self.def_horizon)
        self.dt = 1.0
        self.N_data = len(self.driving_data)
        self.horizon = self.obs_config.horizon
        print(f"Training with l-step lookahead, l = {self.horizon}")


        self.action_space = spaces.Box(
            low=0.0, high=10_000.0, shape=(2,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5 + self.horizon,), dtype=np.float32
        )

        # Close over params + dt to avoid recompilation
        self._jit_step = jax.jit(
            lambda s, a, d: self._core_logic(s, a, d, self.params, self.dt)
        )

        self.state = None
        self.t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        rng = self.np_random

        base_temp = rng.uniform(30, 33)
        self.state = jnp.array([base_temp, base_temp, 0.8])

        self.t = rng.integers(0, self.N_data - 2000)

        d = self._get_disturbance(self.t)
        preview = self._get_receding_horizon(self.t)
        obs = self._get_obs(self.state, d, preview)
        return obs, {}

    def step(self, action):
        d = self._get_disturbance(self.t)

        next_state, reward, terminated, info = self._jit_step(
            self.state, jnp.array(action), d,
        )

        self.state = next_state
        self.t += 1

        truncated = self.t >= self.N_data - 1

        d_next = self._get_disturbance(self.t)
        preview = self._get_receding_horizon(self.t)
        # Observation for next step
        obs = self._get_obs(next_state, d_next, preview)

        return obs, reward, terminated, truncated, info

    def _get_receding_horizon(self, t):
        idxs = jnp.arange(t + 1, t + self.horizon + 1)
        idxs = jnp.clip(idxs, 0, self.N_data - 1)
        return self.driving_data[idxs]

    def _get_disturbance(self, k):
        return jnp.array([self.driving_data[k], 40.0])

    def _get_obs(self, state, disturbance, preview):
        raw = jnp.concatenate([state, disturbance, preview])
        mean = jnp.concatenate([
            self.obs_config.obs_mean,
            jnp.full((self.horizon,), 10000.0)
        ])
        scale = jnp.concatenate([
            self.obs_config.obs_scale,
            jnp.full((self.horizon,), 10000.0)
        ])
        return (raw - mean) / scale

    @staticmethod
    @jax.jit
    def _core_logic(state, action, disturbance, params, dt):
        # Action mapping
        controls = action 

        next_state, diag = rk4_step(state, controls, disturbance, params, dt)
        T_next = next_state[0]

        MAX_TEMP_DEVIATION = 15.0 
        MAX_POWER = 50.0

        # --- Reward ---
        cost_energy = (diag[0] / 1000.0) / MAX_POWER  # P_cool in kWh

        # Penalty weight
        T_des = 33.0
        T_MAX = 35.0
        T_MIN = 15.0

        viol_up = jnp.maximum(0.0, T_next - T_MAX)
        viol_low = jnp.maximum(0.0, T_MIN - T_next)

        cost_constraint = (viol_up + viol_low) / MAX_TEMP_DEVIATION

        T_des = 33.0
        
        # Only apply this cost at the very last step
        cost_temp_deviation = ((T_next - T_des) / MAX_TEMP_DEVIATION)**2
        w_energy = 20.0
        w_temp = 3.0 # 5.0, or 3.0 for less importance 
        w_constraint = 2.0

        total_cost = (
            w_energy * cost_energy + 
            w_temp * cost_temp_deviation + 
            w_constraint * cost_constraint
        )
        reward = -total_cost

        # Check fail state
        terminated = (T_next > 45.0) | (T_next < 15.0)



        return next_state, reward, terminated, {
            "time": dt,
            "P_cooling": diag[0],
            "T_batt": T_next,
            "cost_energy": cost_energy,
            "cost_temp": cost_temp_deviation,
            "cost_constraint": cost_constraint,
        }
