# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Operant is a high-performance SIMD-optimized reinforcement learning library providing Gymnasium-compatible environments implemented in Rust with Python bindings. **Goal: Achieve 1-2M+ SPS to surpass PufferLib** as the fastest RL library.

**Current Performance**: ~40K SPS (training with PPO on 3090 GPU + 5950X CPU)
**Target Performance**: 1-2M+ SPS (GPU-resident environments + optimizations)

**Key technologies:**
- Rust with nightly features (portable SIMD via `f32x8`, targeting AVX-512)
- CUDA kernels for GPU-resident environments (roadmap)
- PyO3 for zero-copy buffer protocol with potential CUDA IPC
- Ratatui TUI with decoupled observer pattern (IPC via mmap)
- Poetry for Python dependency management
- Maturin for Rust-Python builds

**Performance Philosophy**: "Optimize every single aspect" - from memory layout to kernel fusion to async execution

## Common Development Commands

### Building

```bash
# Install Python dependencies
poetry install

# Build Rust extension in debug mode (faster compilation, slower runtime)
poetry run maturin develop

# Build Rust extension in release mode (slower compilation, faster runtime)
poetry run maturin develop --release

# Always use --release for benchmarking or performance testing
```

### Testing

```bash
# Run Python tests
poetry run pytest

# Run Python tests with verbose output
poetry run pytest -v

# Run specific test file
poetry run pytest python/tests/test_integration.py

# Run Rust tests (for individual crates)
cd crates/operant-envs && cargo test
cd crates/operant-core && cargo test
```

### Benchmarking

```bash
# Quick benchmark (4096 envs only)
poetry run python benches/cartpole_benchmark.py

# Full benchmark across multiple environment counts
poetry run python benches/cartpole_benchmark.py --all
```

## Architecture

### Crate Structure (Rust Workspace)

The project uses a Rust workspace with five crates (six planned):

1. **`operant-core`** (`crates/operant-core/`): Core traits and abstractions
   - `VecEnvironment` trait: Defines interface for vectorized environments
   - `LogData` trait: Metrics tracking interface
   - No dependencies beyond rand

2. **`operant-envs`** (`crates/operant-envs/`): CPU SIMD-optimized environment implementations
   - Gymnasium environments: CartPole, MountainCar, Pendulum
   - SIMD vectorization using `f32x8` (processes 8 envs per instruction)
   - Struct-of-Arrays (SoA) memory layout for cache efficiency
   - Optional features: `simd` (default), `parallel` (rayon-based multi-threading)
   - SIMD helpers in `src/shared/simd.rs`: Fast Taylor series approximations for trig functions
   - **Status**: Production-ready, 150M+ SPS (environment-only)

3. **`operant-cuda`** (`crates/operant-cuda/`) 🚧 **ROADMAP**
   - GPU-resident environment kernels for 10-15x training speedup
   - CUDA implementations: CartPole, MountainCar, Pendulum
   - Eliminates CPU↔GPU data transfer bottleneck
   - SoA memory layout on GPU (coalesced access)
   - Integration via `cudarc` or raw CUDA + bindgen
   - **Target**: Phase 1 of optimization plan, 500K+ SPS

4. **`operant-python`** (`crates/operant-python/`): PyO3 bindings
   - PyClass wrappers: `PyCartPoleVecEnv`, `PyMountainCarVecEnv`, `PyPendulumVecEnv`
   - Zero-copy buffer protocol via custom `Buffer1D`, `Buffer2D`, `Buffer3D` types
   - RL utilities: `RolloutBuffer` (SoA GAE computation), `RunningNormalizer`, `AsyncEnvPool`
   - Optional `tui` feature: Terminal UI logger with GPU monitoring (atomic metrics)
   - All Python-visible types must use `#[pyclass]` and be exposed in `lib.rs`
   - **Roadmap**: Add CUDA IPC for zero-copy tensors (Phase 2)

5. **`operant`** (`crates/operant/`): Pure Rust API (re-exports from operant-envs)

6. **`operant-monitor`** (binary) 🚧 **ROADMAP**
   - Standalone TUI binary for decoupled monitoring
   - Reads from memory-mapped metrics files
   - CLI: `operant monitor <run_id>` or `operant monitor --latest`
   - **Target**: Phase 3 of optimization plan, UX enhancement

### Python Package Structure

The Python package (`python/operant/`) wraps the Rust extension with Gymnasium compatibility:

- **`__init__.py`**: Module facade system
  - Wraps raw Rust envs with `_VecEnvWrapper` for Gymnasium API compatibility
  - Provides `BoxSpace` and `DiscreteSpace` classes (mimic Gymnasium spaces)
  - Lazy-loads `operant.models` (requires PyTorch)
  - Submodule registration: `operant.envs`, `operant.utils`, `operant.models`

- **`operant.envs`**: Clean environment API
  - `CartPoleVecEnv(num_envs, workers=1)` - Discrete action space
  - `MountainCarVecEnv(num_envs, workers=1)` - Discrete action space
  - `PendulumVecEnv(num_envs, workers=1)` - Continuous action space
  - All envs expose `.observation_space`, `.action_space`, `.num_envs`

- **`operant.utils`**: Training utilities
  - `Logger` (alias for `TUILogger`): Context manager for CSV logging with TUI

- **`operant.models`**: PyTorch-based RL algorithms (lazy loaded)
  - `PPO`: Proximal Policy Optimization with Rust-backed RolloutBuffer
  - Neural networks: `ActorCritic`, `DiscreteActorCritic`, `ContinuousActorCritic`
  - Intrinsic motivation: `RND` (Random Network Distillation), `ICM` (Intrinsic Curiosity Module)
  - Value normalization: `PopArtValueHead`

- **`operant.buffers`**: GPU-accelerated buffers (new in 0.3.x)
  - `GpuBuffer`: PyTorch CUDA buffer for fast environment rollouts

### Memory Layout: Struct-of-Arrays (SoA)

Environments use SoA instead of Array-of-Structs for SIMD efficiency:

```rust
// BAD: Array of Structs (AoS) - poor cache locality
struct State { x: f32, y: f32, vx: f32, vy: f32 }
states: Vec<State>  // [s0, s1, s2, ...] - accessing all x values requires scattered reads

// GOOD: Struct of Arrays (SoA) - vectorizable
struct States {
    x: Vec<f32>,   // [x0, x1, x2, x3, x4, x5, x6, x7, ...]
    y: Vec<f32>,   // [y0, y1, y2, y3, y4, y5, y6, y7, ...]
    // ...
}
// Can load 8 consecutive x values into f32x8 for parallel processing
```

### SIMD Implementation Details

- **Target**: AVX2 with `f32x8` (8-wide SIMD vectors)
- **Requires**: Rust nightly with `#![feature(portable_simd)]`
- **Key pattern**: Chunk environments into groups of 8, process with SIMD, handle remainder serially
- **SIMD helpers** (`crates/operant-envs/src/shared/simd.rs`):
  - `simd_sin`, `simd_cos`: Fast Taylor series approximations (order 4-5)
  - Trade perfect accuracy for performance (sufficient for RL physics)

### Parallel Execution

When `workers > 1` is specified, environments use rayon for multi-threaded parallelism:

- **Safety**: `SyncPtr` wrapper asserts non-overlapping access across threads
- **Strategy**: Split environment range across worker threads
- **Use case**: Heavy environments or very large batch sizes (e.g., 8192+ envs)
- **Implementation**: See `crates/operant-envs/src/shared/parallel.rs`

### TUI Architecture (Decoupled Observer Pattern)

**Current Status**: TUI runs in-process with atomic metrics (zero overhead)
**Roadmap**: Decouple into separate monitor process for crash isolation

#### Current Design (v0.4.0)
```rust
// Training writes to atomics (hot path - ~free cost)
metrics_buffer.sps.store(current_sps.to_bits(), Ordering::Relaxed);

// TUI renders in separate thread (same process)
// See: crates/operant-python/src/tui.rs
```

#### Target Design (v0.5.0+)
```
Training Process                Monitor Process
┌─────────────┐                ┌──────────────┐
│  PPO.learn  │                │ Ratatui TUI  │
│     ↓       │   Memory-Map   │    Reads     │
│  Metrics    │───────────────→│   Renders    │
│  (Atomic)   │   IPC/mmap     │  (Detached)  │
└─────────────┘                └──────────────┘
```

**Benefits**:
1. **Crash Isolation**: TUI crash doesn't kill training
2. **Detach/Reattach**: Monitor remote training sessions
3. **Zero Overhead**: No shared process resources
4. **Scalability**: Multiple monitors can attach to same training

**CLI UX**:
```bash
# Auto-spawn TUI in separate process
operant train --config ppo.toml --monitor

# Headless training (server/cluster)
operant train --config ppo.toml --detach

# Attach to running/completed training
operant monitor --latest
operant monitor <run_id>
```

**Implementation Files**:
- `crates/operant-monitor/` (new binary crate)
- `crates/operant-python/src/metrics_writer.rs` (mmap writer)
- `~/.operant/metrics/<run_id>.mmap` (IPC file)

### Zero-Copy Buffer Protocol

Python bindings use custom buffer types (`Buffer1D`, `Buffer2D`, `Buffer3D`) that implement Python's buffer protocol:

- **Direct access**: PyTorch can wrap buffers with `torch.as_tensor()` without copying
- **Layout**: C-contiguous, f32 dtype
- **Usage pattern**:
  ```python
  obs, _ = env.reset(seed=42)  # Returns Buffer2D
  obs_tensor = torch.as_tensor(obs)  # Zero-copy!
  ```

### Deprecation Strategy (v0.1.x → v0.3.x)

Old imports emit warnings but still work until v0.4.0:
- ❌ `from operant import PyCartPoleVecEnv`
- ✅ `from operant.envs import CartPoleVecEnv`

When modifying Python code, prefer the new import style.

## Development Workflow

### Adding a New Environment

1. Implement `VecEnvironment` trait in `crates/operant-envs/src/gymnasium/your_env.rs`
2. Use SoA layout with SIMD vectorization (see `cartpole.rs` as reference)
3. Add PyO3 wrapper in `crates/operant-python/src/lib.rs` as `PyYourEnvVecEnv`
4. Register in Python facade: `python/operant/__init__.py` → `_EnvsModule`
5. Add tests in `python/tests/test_integration.py`

### Modifying SIMD Code

- **Feature gate**: All SIMD code must be behind `#[cfg(feature = "simd")]`
- **Testing**: Test both SIMD and scalar paths (disable feature to test scalar)
- **Safety**: SIMD operations are safe, but ensure chunk alignment handling is correct

### Working with PyO3 Bindings

- **Return types**: Use `PyResult<T>` for fallible operations
- **Arrays**: Use `numpy::PyArray*` for input, custom `Buffer*` types for output
- **Memory**: Buffers must outlive references; clone data when returning to Python
- **GIL**: PyO3 handles GIL automatically in `#[pymethods]`

### Rust Nightly Features

This project requires nightly Rust for:
- `portable_simd`: SIMD vectorization (`f32x8`)

Ensure rustup is set to nightly:
```bash
rustup default nightly
# Or use rust-toolchain.toml file
```

## Key Performance Considerations

### Critical Path Optimizations
1. **GPU-Resident Environments** (HIGHEST IMPACT):
   - Target: Port environments to CUDA kernels
   - Eliminates CPU↔GPU data transfer bottleneck
   - Expected: 10-15x speedup over current architecture
   - Status: Roadmap (see `OPTIMIZATION_PLAN.md`)

2. **Zero-Copy Buffers**:
   - Current: `obs.numpy()` → `torch.from_numpy()` every step
   - Target: Direct CUDA IPC between Rust and PyTorch
   - Use `torch.cuda.from_dlpack()` or raw device pointers
   - Expected: 2-3x speedup

3. **SIMD & Memory Layout**:
   - Current: f32x8 (AVX2) - process 8 envs per instruction
   - Target: f32x16 (AVX-512) for 5950X - 2x wider vectors
   - Always maintain SoA (Struct-of-Arrays) for cache efficiency
   - Alignment: Process environments in chunks of 8/16

4. **Batch Size Strategy**:
   - Current sweet spot: 4096-8192 envs for single-threaded GPU training
   - CPU-only: 1024-2048 envs optimal
   - Multi-threaded: Scale up to 16K+ envs with rayon
   - Rule: `batch_size >= GPU_cores × 32` for full occupancy

5. **Compiler Optimizations**:
   - **Always use `--release`** for benchmarks (10-20x faster than debug)
   - Enable LTO: `lto = "fat"` in Cargo.toml for 5-10% gain
   - Profile-Guided Optimization (PGO): 10-20% additional speedup
   - Consider `codegen-units = 1` for maximum runtime performance

6. **Decoupled Monitoring** (Zero-Overhead TUI):
   - TUI runs in separate process via memory-mapped IPC
   - Training writes atomics, TUI reads independently
   - No performance impact, crash-isolated
   - See `crates/operant-python/src/tui.rs` for atomic metrics

### Performance Targets by Phase
- Phase 0 (Current): 40K SPS
- Phase 1 (GPU Envs): 500K SPS
- Phase 2 (Zero-Copy): 750K SPS
- Phase 3 (Kernel Fusion): 1.1M SPS
- Phase 4 (Multi-Stream): 1.5M SPS
- Phase 5 (FP16 + Tensor Cores): 2M+ SPS

**Competitive Benchmark**: PufferLib achieves ~1M SPS. Operant targets 1.5-2M SPS.

## Testing Philosophy

- **Integration tests** (`python/tests/test_integration.py`): End-to-end environment behavior
- **Rust tests**: Unit tests in respective crate files (`#[cfg(test)] mod tests`)
- **Benchmarks**: Separate scripts in `benches/` directory (not Criterion due to PyO3)

## Version Synchronization

Version is synchronized across:
- `Cargo.toml` workspace version: `0.3.3`
- `pyproject.toml` project version: `0.3.3`
- `python/operant/__init__.py` `__version__`: `"0.3.3"`

When releasing, update all three locations.
