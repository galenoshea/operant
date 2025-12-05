#!/usr/bin/env python3
"""
Profile benchmark to identify bottlenecks in training loop.
"""

import cProfile
import pstats
import io
from pstats import SortKey
import sys

# Import the benchmark
sys.path.insert(0, ".")
from benches.bench import main

if __name__ == "__main__":
    # Profile the benchmark
    profiler = cProfile.Profile()
    profiler.enable()

    # Run benchmark with reduced steps for profiling
    import sys
    sys.argv = [
        "profile_bench.py",
        "--env", "CartPole",
        "--num-envs", "8192",
        "--total-timesteps", "500000",  # Shorter for profiling
    ]

    try:
        main()
    except SystemExit:
        pass

    profiler.disable()

    # Print stats
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats(SortKey.CUMULATIVE)

    print("\n" + "="*80)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
    print("="*80)
    stats.print_stats(30)

    print("\n" + "="*80)
    print("TOP 30 FUNCTIONS BY TOTAL TIME")
    print("="*80)
    stats.sort_stats(SortKey.TIME)
    stats.print_stats(30)

    # Save to file
    with open("profile_results.txt", "w") as f:
        f.write(s.getvalue())

    print("\nFull profile saved to profile_results.txt")
