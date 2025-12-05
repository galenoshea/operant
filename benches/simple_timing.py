#!/usr/bin/env python3
"""
Simple timing measurement of key operations.
"""

import time
import torch
import numpy as np
from operant.envs import CartPole

def measure_operations(num_envs=8192, num_iterations=1000):
    """Measure timing of individual operations."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Envs: {num_envs:,}")
    print(f"Iterations: {num_iterations:,}\n")

    # Create environment
    env = CartPole(num_envs=num_envs)

    # Timing storage
    times = {
        'env_reset': [],
        'env_step': [],
        'cpu_to_gpu': [],
        'gpu_to_cpu': [],
    }

    # Warm-up
    obs = env.reset()
    for _ in range(10):
        actions = np.random.randint(0, 2, size=(num_envs,), dtype=np.int64)
        obs, _, _, _, _ = env.step(actions)

    print("Running timing measurements...")

    # Measure reset
    for i in range(min(num_iterations, 100)):
        t0 = time.perf_counter()
        obs = env.reset()
        times['env_reset'].append(time.perf_counter() - t0)

    # Measure step
    for i in range(num_iterations):
        actions = np.random.randint(0, 2, size=(num_envs,), dtype=np.int64)

        t0 = time.perf_counter()
        obs, rewards, terminals, truncations, info = env.step(actions)
        times['env_step'].append(time.perf_counter() - t0)

        if i % 200 == 0:
            print(f"  Progress: {i}/{num_iterations}")

    # Measure CPU→GPU transfer
    print("\nMeasuring data transfer...")
    test_data = np.random.randn(num_envs, 4).astype(np.float32)
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        gpu_data = torch.from_numpy(test_data).to(device)
        torch.cuda.synchronize() if device == "cuda" else None
        times['cpu_to_gpu'].append(time.perf_counter() - t0)

    # Measure GPU→CPU transfer
    gpu_data = torch.randn(num_envs, 4, device=device)
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        cpu_data = gpu_data.cpu().numpy()
        times['gpu_to_cpu'].append(time.perf_counter() - t0)

    # Calculate statistics
    print("\n" + "="*80)
    print("TIMING RESULTS (microseconds)")
    print("="*80)
    print(f"{'Operation':<20} {'Mean':<12} {'Median':<12} {'Min':<12} {'Max':<12}")
    print("-"*80)

    for op, timings in times.items():
        if not timings:
            continue
        mean = np.mean(timings) * 1e6
        median = np.median(timings) * 1e6
        min_t = np.min(timings) * 1e6
        max_t = np.max(timings) * 1e6
        print(f"{op:<20} {mean:>10.1f} μs {median:>10.1f} μs {min_t:>10.1f} μs {max_t:>10.1f} μs")

    print("="*80)

    # Calculate equivalent SPS
    step_time_s = np.mean(times['env_step'])
    sps = num_envs / step_time_s

    print(f"\nAverage env.step() time: {step_time_s*1000:.3f} ms")
    print(f"Equivalent SPS: {sps:,.0f}")

    # Breakdown
    cpu_to_gpu_us = np.mean(times['cpu_to_gpu']) * 1e6
    gpu_to_cpu_us = np.mean(times['gpu_to_cpu']) * 1e6
    step_us = step_time_s * 1e6

    transfer_time = (cpu_to_gpu_us + gpu_to_cpu_us)
    transfer_pct = (transfer_time / step_us * 100) if step_us > 0 else 0

    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)
    print(f"\nData transfer time per step:")
    print(f"  CPU→GPU: {cpu_to_gpu_us:>8.1f} μs")
    print(f"  GPU→CPU: {gpu_to_cpu_us:>8.1f} μs")
    print(f"  Total:   {transfer_time:>8.1f} μs ({transfer_pct:.1f}% of step time)")
    print(f"\nenv.step() time: {step_us:.1f} μs")
    print(f"\nData transfer overhead: {transfer_pct:.1f}% of env.step() time")

    if transfer_pct > 30:
        print("\n⚠️  HIGH TRANSFER OVERHEAD DETECTED!")
        print("    Recommendation: Implement GPU-resident environments (Phase 1)")

    print("="*80)

if __name__ == "__main__":
    measure_operations(num_envs=8192, num_iterations=1000)
