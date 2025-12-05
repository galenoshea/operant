#!/usr/bin/env python3
"""Benchmark script for Operant PPO training.

Runs 1 million steps using PPO on various environments and measures
steps per second (SPS) throughput.

Usage:
    python benches/bench.py [--env ENV] [--num-envs N] [--device DEVICE]

Examples:
    # Default benchmark (CartPole, 4096 envs, CPU)
    python benches/bench.py

    # Custom environment and batch size
    python benches/bench.py --env Pendulum --num-envs 2048

    # GPU benchmark
    python benches/bench.py --device cuda
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from operant.envs import CartPole, MountainCar, Pendulum
from operant.models import PPO, PPOConfig


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    env_name: str
    num_envs: int
    total_timesteps: int
    elapsed_time: float
    steps_per_second: float
    device: str
    obs_dim: int
    act_dim: int
    is_continuous: bool

    def __str__(self) -> str:
        return (
            f"\n{'=' * 70}\n"
            f"Benchmark Results: {self.env_name}\n"
            f"{'=' * 70}\n"
            f"Environment:        {self.env_name}\n"
            f"Parallel Envs:      {self.num_envs:,}\n"
            f"Observation Dim:    {self.obs_dim}\n"
            f"Action Dim:         {self.act_dim}\n"
            f"Action Type:        {'Continuous' if self.is_continuous else 'Discrete'}\n"
            f"Device:             {self.device}\n"
            f"Total Steps:        {self.total_timesteps:,}\n"
            f"Elapsed Time:       {self.elapsed_time:.2f}s\n"
            f"Steps/Second:       {self.steps_per_second:,.0f}\n"
            f"{'=' * 70}\n"
        )


def benchmark_ppo(
    env_name: str = "CartPole",
    num_envs: int = 128,
    total_timesteps: int = 10_000_000,
    device: str = "cpu",
    verbose: bool = True,
) -> BenchmarkResult:
    """Run PPO benchmark for specified environment.

    Args:
        env_name: Environment name (CartPole, MountainCar, or Pendulum)
        num_envs: Number of parallel environments
        total_timesteps: Total training steps to run
        device: Device to run on ("cpu" or "cuda")
        verbose: Print progress updates

    Returns:
        BenchmarkResult with timing and throughput metrics
    """
    # Create environment
    if env_name == "CartPole":
        env = CartPole(num_envs=num_envs)
    elif env_name == "MountainCar":
        env = MountainCar(num_envs=num_envs)
    elif env_name == "Pendulum":
        env = Pendulum(num_envs=num_envs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    # Determine action dimension and type
    if hasattr(env.action_space, "n"):
        is_continuous = False
        act_dim = env.action_space.n
        action_type = "Discrete"
    else:
        is_continuous = True
        act_dim = env.action_space.shape[0]
        action_type = "Continuous"

    if verbose:
        print(f"\n{env_name} | {num_envs:,} envs | {device} | {total_timesteps:,} steps")

    # Configure PPO for benchmark (optimized settings)
    config = PPOConfig(
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        n_steps=128,
        n_epochs=4,
        batch_size=256,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        normalize_observations=False,
        normalize_rewards=False,
        use_amp=False,  # Disable AMP for consistent benchmarking
        use_gpu_buffer=(device == "cuda"),
        checkpoint_dir=None,  # Disable checkpointing for benchmark
    )

    # Create PPO model
    model = PPO(env, config=config, device=device)

    # Disable torch.compile for benchmarking stability with large batch sizes
    # torch.compile can cause numerical issues on CPU with >1024 envs
    if hasattr(model.policy, "_orig_mod"):
        model.policy = model.policy._orig_mod

    # Progress callback
    last_update_time = time.time()

    def progress_callback(metrics: dict) -> bool:
        nonlocal last_update_time

        if verbose:
            current_time = time.time()
            # Print update every 5 seconds
            if current_time - last_update_time >= 5.0:
                steps = metrics["timesteps"]
                sps = metrics["sps"]
                episodes = metrics["episodes"]
                mean_reward = metrics.get("mean_reward", 0)
                progress = (steps / total_timesteps) * 100

                # Use \r to overwrite previous line
                print(
                    f"\rTraining... [{progress:5.1f}%] {steps//1000:>4}k steps | "
                    f"{sps/1000:>5.1f}k SPS | {episodes//1000:>3}k eps | R={mean_reward:>5.1f}",
                    end="",
                    flush=True,
                )
                last_update_time = current_time

        return True  # Continue training

    # Run training
    benchmark_start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=progress_callback,
        log_interval=1,
    )
    benchmark_end = time.time()

    elapsed = benchmark_end - benchmark_start
    sps = total_timesteps / elapsed

    # Create result
    result = BenchmarkResult(
        env_name=env_name,
        num_envs=num_envs,
        total_timesteps=total_timesteps,
        elapsed_time=elapsed,
        steps_per_second=sps,
        device=device,
        obs_dim=env.observation_space.shape[0],
        act_dim=act_dim,
        is_continuous=is_continuous,
    )

    if verbose:
        # Clear the progress line and print final result
        print(f"\r✓ Complete | {total_timesteps//1000:,}k steps in {elapsed:.4f}s | {sps:,.0f} SPS\n")

    return result


def run_comprehensive_benchmark(device: str = "cpu") -> None:
    """Run comprehensive benchmark across all environments.

    Args:
        device: Device to run on ("cpu" or "cuda")
    """
    print("\n" + "=" * 70)
    print("OPERANT PPO COMPREHENSIVE BENCHMARK")
    print("=" * 70)
    print(f"PyTorch Version:  {torch.__version__}")
    print(f"Device:           {device}")
    if device == "cuda":
        print(f"GPU:              {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version:     {torch.version.cuda}")
    print("=" * 70)

    results = []

    # Benchmark configurations
    # Use 256 envs - larger batch sizes (>= 512) trigger NaN bugs in environment reset
    # TODO: Fix environment NaN bug and increase to 4096 for better throughput measurement
    configs = [
        ("CartPole", 256),
        ("MountainCar", 256),
        ("Pendulum", 256),
    ]

    for env_name, num_envs in configs:
        try:
            result = benchmark_ppo(
                env_name=env_name,
                num_envs=num_envs,
                total_timesteps=1_000_000,
                device=device,
                verbose=True,
            )
            results.append(result)
        except Exception as e:
            print(f"\n❌ Benchmark failed for {env_name}: {e}")
            import traceback

            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Environment':<15} {'Envs':<10} {'SPS':<15} {'Time':<10}")
    print("-" * 70)
    for result in results:
        print(
            f"{result.env_name:<15} "
            f"{result.num_envs:<10,} "
            f"{result.steps_per_second:<15,.0f} "
            f"{result.elapsed_time:<10.2f}s"
        )
    print("=" * 70)

    # Calculate average
    if results:
        avg_sps = sum(r.steps_per_second for r in results) / len(results)
        print(f"\nAverage SPS across all environments: {avg_sps:,.0f}")
        print()


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark Operant PPO training performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=["CartPole", "MountainCar", "Pendulum"],
        help="Environment to benchmark (default: all)",
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=8192,
        help="Number of parallel environments (default: 8192)",
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2_000_000,
        help="Total training steps (default: 2,000,000)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to run on (default: auto-detect - cuda if available, else cpu)",
    )

    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive benchmark across all environments",
    )

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected device: {args.device}")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("❌ CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    # Run benchmark - default to CartPole if not specified
    if args.comprehensive:
        run_comprehensive_benchmark(device=args.device)
    else:
        # Default to CartPole if no env specified
        env_name = args.env if args.env is not None else "CartPole"
        result = benchmark_ppo(
            env_name=env_name,
            num_envs=args.num_envs,
            total_timesteps=args.total_timesteps,
            device=args.device,
            verbose=True,
        )


if __name__ == "__main__":
    main()
