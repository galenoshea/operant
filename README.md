# Operant

[![PyPI version](https://badge.fury.io/py/operant.svg)](https://badge.fury.io/py/operant)
[![Crates.io](https://img.shields.io/crates/v/operant.svg)](https://crates.io/crates/operant)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

High-performance SIMD-optimized Gymnasium-compatible reinforcement learning environments in Rust with Python bindings.

**~600x faster than Gymnasium** for vectorized environments.

## What is This?

Operant provides **native Rust implementations** of Gymnasium environments with:
- **SIMD vectorization**: Process 8 environments simultaneously per instruction (AVX2)
- **Struct-of-Arrays layout**: Cache-friendly memory access patterns
- **Zero-copy numpy**: Direct array access without Python overhead
- **Gymnasium compatibility**: Drop-in replacement for standard Gym environments

Unlike [PufferLib](https://github.com/PufferAI/PufferLib) which wraps existing Gymnasium environments for vectorization, Operant implements environments **natively in Rust** for maximum performance.

## Supported Environments

| Environment | State Dim | Action Space | Physics | Reward |
|-------------|-----------|--------------|---------|---------|
| CartPole | 4 | Discrete(2) | Inverted pendulum balance | +1 per step alive |
| MountainCar | 2 | Discrete(3) | Sparse reward climbing | -1 per step |
| Pendulum | 3 | Continuous(1) | Swing-up control | Cost minimization |

All environments provide Gymnasium-compatible `observation_space` and `action_space` properties for easy integration with RL frameworks.

## Performance

```
CartPole Benchmark (4096 envs)
============================================================
Operant...     97.54M steps/sec
Gymnasium...    0.16M steps/sec

Speedup: ~600x faster than Gymnasium
```

## Requirements

- Python 3.10+

## Installation

### Python (PyPI)

```bash
pip install operant
```

### Rust (crates.io)

```bash
cargo add operant
```

### From Source (Development)

Requires Rust nightly and Poetry:

```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 3. Setup project
poetry install
poetry run maturin develop --release
```

## Usage

### Python

#### CartPole (Discrete Actions)

```python
import numpy as np
from operant.envs import CartPoleVecEnv

# Create 4096 parallel environments
num_envs = 4096
env = CartPoleVecEnv(num_envs)
obs, info = env.reset(seed=42)  # Shape: (4096, 4)

for step in range(10000):
    actions = np.random.randint(0, 2, size=num_envs, dtype=np.int32)
    obs, rewards, terminals, truncations, info = env.step(actions)
```

#### Multi-threaded Execution

For heavier environments or large batch sizes, enable parallel execution:

```python
# Use 4 worker threads for parallel step execution
env = CartPoleVecEnv(num_envs=8192, workers=4)
```

#### MountainCar (Discrete Actions)

```python
from operant.envs import MountainCarVecEnv

num_envs = 4096
env = MountainCarVecEnv(num_envs)
obs, info = env.reset(seed=42)  # Shape: (4096, 2)

for step in range(10000):
    actions = np.random.randint(0, 3, size=num_envs, dtype=np.int32)
    obs, rewards, terminals, truncations, info = env.step(actions)
```

#### Pendulum (Continuous Actions)

```python
from operant.envs import PendulumVecEnv

num_envs = 4096
env = PendulumVecEnv(num_envs)
obs, info = env.reset(seed=42)  # Shape: (4096, 3) - [cos(θ), sin(θ), θ_dot]

for step in range(10000):
    actions = np.random.uniform(-2.0, 2.0, size=num_envs).astype(np.float32)
    obs, rewards, terminals, truncations, info = env.step(actions)
```

### Rust

```rust
use operant::{CartPole, VecEnv};

fn main() {
    // Create 1024 parallel environments
    let mut env = CartPole::new(1024);

    // Reset all environments
    let obs = env.reset();

    // Step with actions
    let actions = vec![0; 1024];
    let (obs, rewards, terminals, truncations) = env.step(&actions);
}
```

### Logging and Metrics

```python
from operant.utils import Logger

# Context manager automatically handles cleanup
with Logger(csv_path="training.csv") as logger:
    for step in range(1000):
        # ... training loop ...
        logger.log(steps=num_envs, reward=mean_reward, length=mean_length)
```

## Migration from v0.1.x

**Old imports (deprecated)**:
```python
from operant import PyCartPoleVecEnv, Logger
```

**New imports (recommended)**:
```python
from operant.envs import CartPoleVecEnv
from operant.utils import Logger
```

The old import style will continue to work until v0.4.0, but will emit deprecation warnings.

## Benchmarks

### Quick Benchmark

Compare Operant at 4096 environments:

```bash
poetry run python benches/cartpole_benchmark.py
```

### Full Benchmark

Test across multiple environment counts (1, 16, 256, 1024, 4096):

```bash
poetry run python benches/cartpole_benchmark.py --all
```

## Architecture

Operant uses a Struct-of-Arrays (SoA) memory layout with SIMD vectorization:

- **f32x8 SIMD**: Processes 8 environments simultaneously per instruction
- **SoA Layout**: Cache-friendly memory access patterns
- **Zero-copy**: Direct numpy array access without Python overhead
- **Rust + PyO3**: Native performance with Python ergonomics

## Development

```bash
# Run tests
poetry run pytest

# Build in debug mode (faster compilation)
poetry run maturin develop

# Build in release mode (faster runtime)
poetry run maturin develop --release
```

## License

MIT
