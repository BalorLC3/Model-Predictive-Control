import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei"], # Change this to tex True & unicode to True
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "figure.figsize": (6.0, 10.0),
    "lines.linewidth": 1.4,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.dpi": 300
})

class DetailedMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Metrics containers
        self.history = {
            'ep_rewards': [],
            'ep_energy_kj': [], # The real KPI
            'ep_avg_temp': [],
            'ep_max_temp': [],
            'ep_length': [],
            'ep_cost_total': [],
            'ep_cost_energy': [],
            'ep_cost_temp': [],
            'ep_cost_constraint': []
        }
        
        # Accumulators for current episode
        self.cur_reward = 0
        self.cur_energy_joules = 0
        self.cur_temp_sum = 0
        self.cur_temp_max = -np.inf
        self.cur_steps = 0

        self.cur_total_cost_sum = 0
        self.cur_cost_energy_sum = 0
        self.cur_cost_temp_sum = 0
        self.cur_cost_constraint_sum = 0

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]
        
        self.cur_reward += reward
        self.cur_steps += 1
        
        # P_cooling (Watts) * dt (1.0s) = KiloJoules
        p_cool_w = info.get('P_cooling', 0.0) 
        self.cur_energy_joules += p_cool_w * 1.0 
        
        # Temp Stats
        t_batt = info.get('T_batt', 0.0)
        self.cur_temp_sum += t_batt
        self.cur_temp_max = max(self.cur_temp_max, t_batt)

        
        cost_energy = info.get('cost_energy', 0)
        cost_temp = info.get('cost_temp')
        cost_constraint = info.get('cost_constraint', 0)

        self.cur_cost_energy_sum += cost_energy
        self.cur_cost_temp_sum += cost_temp
        self.cur_cost_constraint_sum += cost_constraint

        if done:
            # Store Metrics
            self.history['ep_rewards'].append(self.cur_reward)
            self.history['ep_energy_kj'].append(self.cur_energy_joules / 1000.0) # Convert J -> kJ
            self.history['ep_avg_temp'].append(self.cur_temp_sum / self.cur_steps)
            self.history['ep_max_temp'].append(self.cur_temp_max)
            self.history['ep_length'].append(self.cur_steps)
            
            # Print status every episode because we have so few of them
            ep_num = len(self.history['ep_rewards'])
            print(f"  > Ep {ep_num}: Energy={self.history['ep_energy_kj'][-1]:.1f}kJ, "
                  f"AvgT={self.cur_temp_sum / self.cur_steps:.1f}C, Reward={self.cur_reward:.1f}")
            
            print(f"   | AvgCostEnergy = {self.cur_cost_energy_sum / self.cur_steps:.4f}, AvgCostTemp = {self.cur_cost_temp_sum / self.cur_steps:.4f}, AvgCostConstraint = {self.cur_cost_constraint_sum / self.cur_steps:.4f}")

            # Reset
            self.cur_reward = 0
            self.cur_energy_joules = 0
            self.cur_temp_sum = 0
            self.cur_temp_max = -np.inf
            self.cur_steps = 0
            

            self.cur_total_cost_sum = 0
            self.cur_cost_energy_sum = 0
            self.cur_cost_temp_sum = 0
            self.cur_cost_constraint_sum = 0
        return True

class TrainExport:
    def __init__(self, model, env, path_prefix: str):
        self.model = model
        self.env = env
        self.path_prefix = path_prefix
        os.makedirs(self.path_prefix, exist_ok=True)

    def train(self, total_timesteps: int, **kwargs):
        print(f"Starting Training ({total_timesteps} steps)...")
        
        # Use the Detailed Callback
        callback = DetailedMetricsCallback()
        
        self.model.learn(
            total_timesteps=total_timesteps, 
            progress_bar=True, 
            callback=callback,
            **kwargs
        )
        
        self._save_artifacts(callback)
        self.plot_kpis(callback.history)
        
        return callback.history

    def _save_artifacts(self, callback):
        # Save Model
        self.model.save(f"{self.path_prefix}/model.zip")
        
        # Save JAX Weights
        if hasattr(self.model.policy, 'actor_state'):
            with open(f"{self.path_prefix}/actor_weights.pkl", 'wb') as f:
                pickle.dump(self.model.policy.actor_state.params, f)
                print(f"Weights saved at {self.path_prefix}/actor_weights.pkl")

        # Save History
        with open(f"{self.path_prefix}/history.pkl", 'wb') as f:
            pickle.dump(callback.history, f)
            print(f"History saved at {self.path_prefix}/history.pkl")

    def plot_kpis(self, history):
        """Plots Energy, Temperature, and Reward per Episode"""
        episodes = np.arange(len(history['ep_rewards']))
        
        fig, axs = plt.subplots(3, 1, figsize=(6, 5), sharex=True)
        
        axs[0].plot(episodes, history['ep_energy_kj'], color='dodgerblue')
        axs[0].set_ylabel('Energia'+ '\n' + r'Consumida [kJ]')

        axs[1].plot(episodes, history['ep_avg_temp'], color='red')
        axs[1].set_ylabel(r'Promedio $T_{batt}$ [°C]')

        axs[2].plot(episodes, history['ep_rewards'], color='seagreen')
        axs[2].set_ylabel('Recompensa'+ '\n' + r'Cumulativa ($R$)')
        axs[2].set_xlabel('Episodio')
        axs[2].ticklabel_format(axis='y', style='sci', scilimits=(4,4))
        
        
        plt.tight_layout()
        plt.savefig(f"{self.path_prefix}/metrics.png")
        plt.show()