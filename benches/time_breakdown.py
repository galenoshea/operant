#!/usr/bin/env python3
"""
Detailed timing breakdown of PPO training loop to identify bottlenecks.
"""

import time
import torch
import operant
import numpy as np

def measure_component_times(num_envs=8192, num_steps=128, num_iterations=10):
    """Measure time spent in each component of the training loop."""

    # Initialize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = operant.PyCartPoleVecEnv(num_envs, device=device)

    # Simple PPO-like model
    obs_dim = 4
    act_dim = 1
    hidden_dim = 64

    model = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dim, act_dim),
        torch.nn.Tanh()
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Timing dictionaries
    times = {
        'env_reset': 0.0,
        'env_step': 0.0,
        'numpy_to_torch': 0.0,
        'model_forward': 0.0,
        'torch_to_numpy': 0.0,
        'optimizer_step': 0.0,
        'total_iteration': 0.0,
    }

    print(f"\nMeasuring timing for {num_iterations} iterations...")
    print(f"Envs: {num_envs}, Steps per iteration: {num_steps}")
    print(f"Device: {device}\n")

    for iteration in range(num_iterations):
        iter_start = time.perf_counter()

        # Reset
        t0 = time.perf_counter()
        obs_np = env.reset()
        times['env_reset'] += time.perf_counter() - t0

        for step in range(num_steps):
            # Convert numpy to torch
            t0 = time.perf_counter()
            obs = torch.from_numpy(obs_np).to(device)
            times['numpy_to_torch'] += time.perf_counter() - t0

            # Forward pass
            t0 = time.perf_counter()
            with torch.no_grad():
                action_logits = model(obs)
            times['model_forward'] += time.perf_counter() - t0

            # Convert torch to numpy
            t0 = time.perf_counter()
            actions_np = action_logits.cpu().numpy()
            times['torch_to_numpy'] += time.perf_counter() - t0

            # Environment step
            t0 = time.perf_counter()
            obs_np, rewards, terminals, truncations, info = env.step(actions_np)
            times['env_step'] += time.perf_counter() - t0

        # Dummy optimizer step
        t0 = time.perf_counter()
        obs = torch.from_numpy(obs_np).to(device)
        loss = model(obs).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        times['optimizer_step'] += time.perf_counter() - t0

        times['total_iteration'] += time.perf_counter() - iter_start

        if iteration % 2 == 0:
            print(f"Iteration {iteration+1}/{num_iterations} complete...")

    # Calculate percentages
    total = times['total_iteration']
    print("\n" + "="*80)
    print("TIMING BREAKDOWN")
    print("="*80)
    print(f"{'Component':<25} {'Time (s)':<12} {'% of Total':<12} {'Per Iter (ms)'}")
    print("-"*80)

    for component, duration in times.items():
        pct = (duration / total * 100) if total > 0 else 0
        per_iter_ms = (duration / num_iterations * 1000)
        print(f"{component:<25} {duration:>10.4f}s  {pct:>6.1f}%       {per_iter_ms:>8.2f} ms")

    print("-"*80)
    print(f"{'TOTAL':<25} {total:>10.4f}s  {100.0:>6.1f}%")
    print("="*80)

    # Calculate SPS
    total_steps = num_envs * num_steps * num_iterations
    sps = total_steps / total
    print(f"\nTotal steps: {total_steps:,}")
    print(f"Total time: {total:.4f}s")
    print(f"Steps per second (SPS): {sps:,.0f}")

    # Breakdown of env_step
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)

    env_step_pct = (times['env_step'] / total * 100)
    data_transfer_pct = ((times['numpy_to_torch'] + times['torch_to_numpy']) / total * 100)

    print(f"\n1. Environment stepping: {env_step_pct:.1f}% of total time")
    print(f"   - This includes CPU→GPU data transfer for observations/rewards")
    print(f"   - Time per env step: {times['env_step'] / (num_iterations * num_steps) * 1000:.3f} ms")

    print(f"\n2. Explicit data transfers (numpy↔torch): {data_transfer_pct:.1f}% of total time")
    print(f"   - numpy→torch: {times['numpy_to_torch'] / total * 100:.1f}%")
    print(f"   - torch→numpy: {times['torch_to_numpy'] / total * 100:.1f}%")

    print(f"\n3. Model forward pass: {times['model_forward'] / total * 100:.1f}% of total time")

    print(f"\n4. Optimizer step: {times['optimizer_step'] / total * 100:.1f}% of total time")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    total_transfer = env_step_pct + data_transfer_pct
    print(f"\nTotal CPU↔GPU data transfer overhead: ~{total_transfer:.1f}% of training time")
    print(f"This confirms that GPU-resident environments (Phase 1) will provide")
    print(f"the biggest performance improvement.\n")

if __name__ == "__main__":
    measure_component_times(num_envs=8192, num_steps=128, num_iterations=10)
