from src.env_batt import BatteryCoolingEnv
from sbx import SAC, TQC
from utils.trainer import TrainExport
from stable_baselines3.sac.policies import SACPolicy

import os 

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

env = BatteryCoolingEnv()

name = "sac"
model = SAC(
        env=env, 
        gamma=0.99,
        learning_rate=3e-4,
        policy= "MlpPolicy",
        buffer_size=100_000, # Large buffer for slow dynamics
        learning_starts=1_000, # Learning starts after 1000 iterations of the system dynamics
        tau=0.005
)

# Reward Function:
# R(s, a) = -[\lambda * P_cool \beta * (\max(0, T_batt' - T_max) + \max(0, T_min - T_batt'))**2 + \alpha * (T_batt' - T_batt)**2] - punish
# where punish = 20000, T_batt is a entry of s and P_cool(u) is function of the actions.
env = BatteryCoolingEnv()
trainer = TrainExport(
    model, 
    env, 
    path_prefix="results/" + name
)

trainer.train(total_timesteps=50_000)

# WINNER: lambda = 20.0, beta = 200.0, alpha = 0.02