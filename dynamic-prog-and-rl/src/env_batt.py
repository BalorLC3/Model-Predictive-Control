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
    obs_mean: jnp.ndarray = field(
        default_factory=lambda: jnp.array([32.0, 32.0, 0.5, 10000.0, 25.0])
    )
    obs_scale: jnp.ndarray = field(
        default_factory=lambda: jnp.array([15.0, 15.0, 0.5, 10000.0, 15.0])
    )
# Don't forget to update in _core_logic because jax can't access to class attributes

class BatteryCoolingEnv(gym.Env):
    """
    Standard Gymnasium Environment for Stable Baselines3.
    """
    
    def __init__(self, render_mode=None):
        super().__init__()
        try:
            raw = np.load('data/driving_energy.npy', mmap_mode='r')
            self.driving_data = jnp.array(raw) 
        except:
            t = jnp.arange(0, 3600)
            self.driving_data = jnp.abs(jnp.sin(t/50.0)) * 20000.0
            
        self.params = SystemParameters()
        self.obs_config = ObservationConfig()
        self.dt = 1.0
        self.N_data = len(self.driving_data)
        # Continuous: [Compressor, Pump] normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # [T_batt, T_clnt, SOC, P_driv, T_amb]
        # Normalized roughly to [-1, 1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self._jit_step = jax.jit(self._core_logic)
        
        self.state = None
        self.k = 0 # Time index

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize State [T_batt, T_clnt, SOC]
        base_temp = np.random.uniform(28.0, 34.5) 
        self.state = jnp.array([base_temp, base_temp, 0.8])
        
        # Random start time in the driving cycle (Data Augmentation)
        self.k = np.random.randint(0, self.N_data - 2000)
        
        d = self._get_disturbance(self.k)
        
        obs = self._get_obs(self.state, d)
        return obs, {}

    def step(self, action):
        d = self._get_disturbance(self.k)
        
        obs, next_state, reward, terminated, info = self._jit_step(
            self.state, jnp.array(action), d, self.params, self.dt
        )
        
        self.state = next_state
        self.k += 1

        truncated = self.k >= self.N_data - 1



        return obs, reward, terminated, truncated, info

    def _get_disturbance(self, k):
        return jnp.array([self.driving_data[k], 40.0])

    def _get_obs(self, state, disturbance):
        # Normalization
        raw = jnp.concatenate([state, disturbance])
        mean = self.obs_config.obs_mean
        scale = self.obs_config.obs_scale
        return (raw - mean) / scale

    @staticmethod
    @jax.jit
    def _core_logic(state, action, disturbance, params, dt):
        # Map Action [-1, 1] -> [0, 5000] RPM
        controls = (jnp.tanh(action) + 1.0) * 2500.0
        
        # Physics Step
        next_state, diag = rk4_step(state, controls, disturbance, params, dt)
        T_next = next_state[0]
        
        # --- REWARD FUNCTION ---
        # Minimize Cooling Power (Index 0)
        # diag[8] is P_pump (W), diag[7] is P_comp (W) -> (W + W)/1000 = kW.
        # diag[0] is P_cool = P_pump + P_comp
        P_cool_kW = diag[0] / 1000.0
        cost_energy = P_cool_kW * dt  # in kWh
        
        # Penalty Weight
        T_des = 33.0
        T_MAX = 34.5
        T_MIN = 30.0
        
        viol_up = jnp.maximum(0.0, T_next - T_MAX)
        viol_low = jnp.maximum(0.0, T_MIN - T_next)

        total_viol = viol_up + viol_low
        cost_constraint = 20.0 * (total_viol ** 2)
        cost_des = 0.03 * (T_next - T_des)**2 

        reward = -(cost_constraint + cost_energy + cost_des)
        
        # Observation for next step
        raw_obs = jnp.concatenate([next_state, disturbance])
        mean = jnp.array([32.0, 32.0, 0.5, 10000.0, 25.0])
        scale = jnp.array([15.0, 15.0, 0.5, 10000.0, 15.0])
        obs = (raw_obs - mean) / scale
        
        # Check fail state
        terminated = (T_next > 45.0) | (T_next < 15.0)# Blow up
        
        reward = jnp.where(terminated, reward - 20000.0, reward)

        return obs, next_state, reward, terminated, {
            "time": dt,
            "P_cooling": diag[0],
            "T_batt": next_state[0]
        }