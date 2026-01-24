import argparse
from src.env_batt import BatteryCoolingEnv, ObservationConfig
from sbx import SAC
from utils.trainer import TrainExport
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def main():
    parser = argparse.ArgumentParser(description="Train SAC with different horizons")
    parser.add_argument("--name", type=str, default="sac", help="Experiment name")
    parser.add_argument("--horizon", type=int, default=0, help="Lookahead horizon steps")
    parser.add_argument("--total_steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    print(f"Training with l-step lookahead, l = {args.horizon}")
    
    # Create environment with specified horizon
    env = BatteryCoolingEnv(def_horizon=args.horizon)
    
    name = f"{args.name}_h{args.horizon}"
    
    model = SAC(
        env=env, 
        gamma=0.99,
        learning_rate=args.lr,
        policy="MlpPolicy",
        buffer_size=1_000_000,
        learning_starts=2_740,
        ent_coef='auto',
        tau=0.005,
        verbose=1,
        seed=args.seed
    )
    
    trainer = TrainExport(
        model, 
        env, 
        path_prefix=f"results/{name}"
    )
    
    trainer.train(total_timesteps=args.total_steps)

if __name__ == "__main__":
    main()