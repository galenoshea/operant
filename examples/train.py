"""Simple training loop example using Operant.

Demonstrates the basic API for running vectorized environments with inline logging.

Usage:
    poetry run python examples/train_ppo.py
"""

import numpy as np
from operant import PyCartPoleVecEnv, Logger


def main():
    num_envs = 4096
    num_steps = 10000

    # Create vectorized environment and logger
    env = PyCartPoleVecEnv(num_envs)
    logger = Logger(csv_path="training.csv")
    obs = env.reset()

    print(f"Running {num_envs} environments for {num_steps} steps...")
    print()

    # Training loop
    for step in range(num_steps):
        # Your policy would go here - using random actions for demo
        actions = np.random.randint(0, 2, size=num_envs, dtype=np.int32)
        obs, rewards, dones, truncated = env.step(actions)

        # Log metrics (inline console + CSV)
        logs = env.get_logs()
        logger.log(
            steps=num_envs,
            reward=logs["mean_reward"],
            length=logs["mean_length"],
        )
        env.clear_logs()

    logger.close()
    print("Training complete. Metrics saved to training.csv")


if __name__ == "__main__":
    main()
