from src.env_batt import BatteryCoolingEnv
from sbx import SAC, TQC
from utils.trainer import TrainExport
from stable_baselines3.sac.policies import SACPolicy

name = "sac"  # CAMBIAR "sac" por "tqc"  

env = BatteryCoolingEnv()

if name == "tqc":
    model = TQC(
        "MlpPolicy", 
        env, 
        buffer_size=100_000,
        learning_starts=1_000,
        verbose=1, 
)

elif name == "sac":
    model = SAC(
        "MlpPolicy", 
        env, 
        buffer_size=100_000,
        learning_starts=1_000,
        verbose=1, 
)

trainer = TrainExport(model, env, path_prefix="results/" + name)
trainer.train(total_timesteps=65_000)