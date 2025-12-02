"""Benchmark comparing Gymnasium and Operant CartPole performance.

This benchmark measures steps per second (SPS) for vectorized environments
at various parallelization levels.

Usage:
    python benches/cartpole_benchmark.py          # Quick Operant benchmark (4096 envs)
    python benches/cartpole_benchmark.py --all    # Full benchmark suite (Gymnasium vs Operant)
"""

import argparse
import time
import numpy as np
import gymnasium as gym

from operant.envs import CartPoleVecEnv


def benchmark_gymnasium(num_envs: int, num_steps: int, num_trials: int = 5) -> dict:
    """Benchmark gymnasium SyncVectorEnv with CartPole."""
    results = []

    for trial in range(num_trials):
        # Create vectorized environment
        env = gym.vector.SyncVectorEnv(
            [lambda: gym.make("CartPole-v1") for _ in range(num_envs)]
        )

        # Reset
        obs, _ = env.reset(seed=42 + trial)

        # Warmup
        for _ in range(100):
            actions = env.action_space.sample()
            obs, rewards, terminated, truncated, infos = env.step(actions)

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_steps):
            actions = env.action_space.sample()
            obs, rewards, terminated, truncated, infos = env.step(actions)
        elapsed = time.perf_counter() - start

        total_steps = num_steps * num_envs
        sps = total_steps / elapsed
        results.append(sps)

        env.close()

    return {
        "mean_sps": np.mean(results),
        "std_sps": np.std(results),
        "min_sps": np.min(results),
        "max_sps": np.max(results),
        "trials": results,
    }



def benchmark_operant(num_envs: int, num_steps: int, num_trials: int = 5) -> dict:
    """Benchmark Operant Rust CartPole."""
    results = []

    for trial in range(num_trials):
        # Create vectorized environment
        env = CartPoleVecEnv(num_envs)

        # Reset
        obs = env.reset(seed=42 + trial)

        # Warmup
        for _ in range(100):
            actions = np.random.randint(0, 2, num_envs, dtype=np.int32)
            obs, rewards, terminals, truncations = env.step(actions)

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_steps):
            actions = np.random.randint(0, 2, num_envs, dtype=np.int32)
            obs, rewards, terminals, truncations = env.step(actions)
        elapsed = time.perf_counter() - start

        total_steps = num_steps * num_envs
        sps = total_steps / elapsed
        results.append(sps)

    return {
        "mean_sps": np.mean(results),
        "std_sps": np.std(results),
        "min_sps": np.min(results),
        "max_sps": np.max(results),
        "trials": results,
    }


def run_benchmark():
    """Run full benchmark suite."""
    print("=" * 80)
    print("CartPole Benchmark: Gymnasium vs Operant (Rust)")
    print("=" * 80)
    print()

    # Test configurations: (num_envs, num_steps)
    configs = [
        (1, 10000),      # Single env
        (16, 5000),      # Small batch
        (256, 2000),     # Medium batch
        (1024, 1000),    # Large batch
        (4096, 500),     # Very large batch
    ]

    num_trials = 5

    print(f"Running {num_trials} trials per configuration")
    print(f"Configurations: {[f'{n} envs' for n, _ in configs]}")
    print()

    # Results storage
    all_results = []

    for num_envs, num_steps in configs:
        print(f"--- {num_envs} environments, {num_steps} steps/env ---")
        print()

        # Gymnasium benchmark
        print(f"  Gymnasium...    ", end="", flush=True)
        gym_results = benchmark_gymnasium(num_envs, num_steps, num_trials)
        print(f"{gym_results['mean_sps']:>12,.0f} SPS (+/- {gym_results['std_sps']:,.0f})")

        # Operant benchmark
        print(f"  Operant (Rust)...", end="", flush=True)
        op_results = benchmark_operant(num_envs, num_steps, num_trials)
        print(f"{op_results['mean_sps']:>12,.0f} SPS (+/- {op_results['std_sps']:,.0f})")

        # Calculate speedups
        speedup_vs_gym = op_results['mean_sps'] / gym_results['mean_sps']
        print(f"  Speedup vs Gym:  {speedup_vs_gym:>11.1f}x")
        print()

        all_results.append({
            "num_envs": num_envs,
            "num_steps": num_steps,
            "gymnasium": gym_results,
            "operant": op_results,
            "speedup_vs_gym": speedup_vs_gym,
        })

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()

    print(f"{'Envs':>6} | {'Gymnasium':>14} | {'Operant':>14} | {'Speedup':>8}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['num_envs']:>6} | {r['gymnasium']['mean_sps']:>14,.0f} | {r['operant']['mean_sps']:>14,.0f} | {r['speedup_vs_gym']:>7.1f}x")

    print()

    # Average speedups
    avg_speedup_gym = np.mean([r['speedup_vs_gym'] for r in all_results])
    print(f"Average speedup vs Gymnasium: {avg_speedup_gym:.1f}x")

    return all_results


def run_fast_benchmark():
    """Quick benchmark at 4096 envs for Operant only."""
    num_envs = 4096
    num_steps = 2000
    num_trials = 3

    print("=" * 60)
    print("CartPole Benchmark - Operant (4096 envs)")
    print("=" * 60)
    print()

    # Operant
    print("Operant...     ", end="", flush=True)
    op_results = benchmark_operant(num_envs, num_steps, num_trials)
    print(f"{op_results['mean_sps']/1e6:>6.2f}M steps/sec")

    print()
    print(f"Total steps: {num_steps * num_envs * num_trials:,}")
    print(f"Trials: {num_trials}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CartPole benchmark')
    parser.add_argument('--all', action='store_true',
                        help='Full benchmark suite at multiple environment counts')
    args = parser.parse_args()

    if args.all:
        run_benchmark()
    else:
        run_fast_benchmark()
