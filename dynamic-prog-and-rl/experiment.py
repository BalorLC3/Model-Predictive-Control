import os
# CRITICAL: Prevent JAX from hogging 90% VRAM instantly, 
# allowing multiple sequential runs without crashing.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from src.env_batt import BatteryCoolingEnv
from sbx import SAC
from utils.trainer import TrainExport
import gc

def run_batch():
    # Common settings for all runs
    COMMON_PARAMS = {
        "policy": "MlpPolicy",
        "buffer_size": 100_000, # Large buffer for slow dynamics
        "learning_starts": 1_000,
        "verbose": 1,
    }

    # Define the 4 Experiments
    experiments = [
        # --- EXP 1: Baseline ---
        # Standard Gamma (0.99) implies ~100 step horizon.
        {
            "name": "sac1",
            "kwargs": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
            }
        },

        # --- EXP 2: Long Horizon (Recommended for Thermal) ---
        # Gamma 0.995 implies ~200 step horizon. Better for slow temp changes.
        {#NOTE:WINNER, with higher gamma is converging slower, which physically means that will take on consideration thermal inertia
            "name": "sac2",
            "kwargs": {
                "learning_rate": 3e-4,
                "gamma": 0.995, 
                "tau": 0.005,
            }
        },

        # --- EXP 3: Batched Updates (Speed & Stability) ---
        # Collect 32 steps, then train 32 times. 
        # Often learns better features due to batch correlation breaking.
        {
            "name": "sac3",
            "kwargs": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "train_freq": 32,
                "gradient_steps": 32,
            }
        },

        # --- EXP 4: Aggressive Learner ---
        # Higher LR to learn faster in limited 80k steps.
        # Reduced Tau (0.02) to update target networks faster.
        {
            "name": "sac4",
            "kwargs": {
                "learning_rate": 1e-3, 
                "gamma": 0.99,
                "tau": 0.02, 
            }
        },
    ]

    TOTAL_TIMESTEPS = 80_000

    print(f"=== Starting Batch Experiment (4 Configurations, {TOTAL_TIMESTEPS} steps each) ===\n")

    for i, exp in enumerate(experiments):
        exp_name = exp["name"]
        print(f"\n[{i+1}/4] Running Experiment: {exp_name}")
        print(f"Params: {exp['kwargs']}")

        # 1. Init Environment
        # Re-initialize for every run to ensure clean state
        env = BatteryCoolingEnv()

        # 2. Merge Parameters
        model_params = {**COMMON_PARAMS, **exp['kwargs']}
        
        # 3. Init Model
        model = SAC(env=env, **model_params)

        # 4. Train
        trainer = TrainExport(
            model, 
            env, 
            path_prefix=f"results/{exp_name}"
        )
        
        try:
            trainer.train(total_timesteps=TOTAL_TIMESTEPS)
        except Exception as e:
            print(f"!!! Error in {exp_name}: {e}")
        
        # 5. Cleanup to free VRAM for next run
        del model
        del env
        del trainer
        gc.collect() # Force Garbage Collection
        
        print(f"Finished {exp_name}. Memory cleaned.\n" + "-"*50)

if __name__ == "__main__":
    run_batch()