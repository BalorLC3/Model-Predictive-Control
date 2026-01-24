import argparse
from src.env_batt import BatteryCoolingEnv
from sbx import SAC
from utils.trainer import TrainExport
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def main():
    """
    Fine-tunes a pre-trained SAC model.
    Loads an existing model and continues training with a new (usually lower) learning rate.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained SAC model")
    
    parser.add_argument("--name", type=str, default="sac", help="Base name of the experiment to load (e.g., 'sac')")
    parser.add_argument("--horizon", type=int, required=True, help="Lookahead horizon of the model to load (e.g., 0 or 10)")
    
    parser.add_argument("--finetune_steps", type=int, default=200_000, help="Number of additional steps to train for")
    parser.add_argument("--lr", type=float, default=3e-5, help="New, lower learning rate for fine-tuning")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for the fine-tuning environment")
    
    args = parser.parse_args()
    
    model_name_to_load = f"{args.name}_h{args.horizon}"
    model_path_to_load = f"results/{model_name_to_load}/model.zip"

    print(f"--- Starting Fine-Tuning ---")
    print(f"Loading model from: {model_path_to_load}")
    print(f"Fine-tuning for: {args.finetune_steps} additional steps")
    print(f"Using new learning rate: {args.lr}")
    print("-" * 30)

    if not os.path.exists(model_path_to_load):
        print(f"ERROR: Model file not found at {model_path_to_load}")
        return

    env = BatteryCoolingEnv(def_horizon=args.horizon)

    model = SAC.load(
        model_path_to_load,
        env=env,
        learning_rate=args.lr,
        verbose=1,
        seed=args.seed
    )

    fine_tune_path_prefix = f"results/{model_name_to_load}/finetune"
    trainer = TrainExport(
        model, 
        env, 
        path_prefix=fine_tune_path_prefix
    )
    
    trainer.train(
        total_timesteps=args.finetune_steps,
        reset_num_timesteps=False 
    )
    
    print(f"--- Fine-Tuning Complete ---")
    print(f"Final fine-tuned model saved in: {fine_tune_path_prefix}")


if __name__ == "__main__":
    main()