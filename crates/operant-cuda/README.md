# operant-cuda

GPU-accelerated RL environments using CUDA for maximum performance.

## Overview

This crate implements **Phase 1** of the Operant optimization roadmap: **GPU-Resident Environments**.

By moving environment physics entirely to the GPU, we eliminate the CPU↔GPU data transfer bottleneck that accounts for 57.4% of training time, achieving an estimated **12.5x speedup** (75K SPS → 500K+ SPS).

## Architecture

### Key Innovation: Zero-Copy GPU-Resident State

Traditional RL frameworks:
```
CPU Env → CPU→GPU (obs) → GPU Model → GPU→CPU (actions) → CPU Env
         ⬆︎ 168 μs                    ⬆︎ 21 μs
         57.4% of step time!
```

Operant GPU environments:
```
GPU Env ←→ GPU Model (all data stays on GPU)
Zero transfer overhead!
```

### Implementation

**CUDA Kernels** ([kernels/cartpole.cu](kernels/cartpole.cu)):
- `cartpole_reset`: Initialize environment states
- `cartpole_step`: Physics simulation (Euler integration)
- Optimized for parallel execution (1 thread = 1 environment)
- Uses fast math and constexpr for maximum performance

**Rust Wrapper** ([src/cartpole.rs](src/cartpole.rs)):
- `CartPoleGpu`: GPU-resident environment manager
- Uses `cudarc` for CUDA integration
- Provides zero-copy buffers for PyTorch integration
- Batched operations for thousands of environments

## Requirements

### CUDA Toolkit

You need the NVIDIA CUDA Toolkit installed to compile kernels:

```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Or download from NVIDIA
# https://developer.nvidia.com/cuda-downloads
```

Verify installation:
```bash
nvcc --version
# Should show CUDA 12.x
```

### Hardware

- NVIDIA GPU with CUDA support (Compute Capability 8.6+ recommended)
- RTX 3090 (Ampere) or newer for best performance
- At least 8GB VRAM for 8192+ environments

## Building

```bash
cargo build --release
```

The build script (`build.rs`) automatically:
1. Locates CUDA toolkit (`$CUDA_PATH` or `/usr/local/cuda`)
2. Compiles `.cu` kernels to PTX using `nvcc`
3. Embeds PTX in the Rust binary

## Usage

```rust
use operant_cuda::CartPoleGpu;

// Create 8192 GPU-resident environments
let mut env = CartPoleGpu::new(8192, 0)?;

// Reset (all on GPU)
env.reset()?;

// Prepare actions on GPU
let actions = device.htod_copy(vec![1; 8192])?;

// Step (all on GPU, zero CPU↔GPU transfer)
env.step(&actions)?;

// Access buffers for PyTorch (zero-copy)
let buffers = env.gpu_buffers();
// buffers.states, buffers.rewards, etc. are GPU pointers
```

## Performance

**Expected Performance** (based on profiling data):

| Metric | CPU (Current) | GPU (Target) | Speedup |
|--------|--------------|--------------|---------|
| env.step() | 330 μs | ~30 μs | 11x faster |
| Transfer overhead | 189.8 μs (57.4%) | 0 μs (0%) | Eliminated |
| **Total SPS** | **75K** | **500K+** | **6.7x** |

With 8192 environments:
- CPU: 75,136 SPS
- **GPU (estimated): 500,000+ SPS**

## Implementation Status

**Phase 1A: CartPole GPU** ✅ Code Complete
- [x] CUDA kernel implementation
- [x] Rust wrapper with cudarc
- [x] Zero-copy buffer API
- [ ] CUDA toolkit installation (required)
- [ ] Build and benchmark validation

**Phase 1B: All Environments** ⏳ Planned
- [ ] MountainCar GPU kernel
- [ ] Pendulum GPU kernel
- [ ] Unified GPU environment API

**Phase 2: Zero-Copy Integration** ⏳ Planned
- [ ] Direct PyTorch tensor integration
- [ ] Eliminate get_obs() CPU transfers
- [ ] DLPack support for frameworks

## Testing

```bash
# Run tests (requires CUDA toolkit + GPU)
cargo test --release

# Run performance benchmark
cargo test --release -- --ignored test_cartpole_gpu_performance
```

Expected output:
```
GPU CartPole: 10,000,000+ SPS (8192 envs, 1000 steps)
```

## Technical Details

### CUDA Kernel Optimizations

1. **Thread-per-Environment**: Each CUDA thread handles one environment
   - Maximizes parallelism (8192 threads = 8192 envs)
   - No synchronization needed between environments

2. **Fast Math**: `--use_fast_math` for FP32 operations
   - Approximate trigonometry (sin, cos)
   - Acceptable for RL (small accuracy loss)
   - 2-3x faster than IEEE 754 strict

3. **Constexpr**: Constants evaluated at compile time
   - No runtime overhead for physics parameters
   - Better register allocation

4. **Euler Integration**: Simple, fast integration
   - Single-step: `x = x + τ * x_dot`
   - Matches CPU implementation for consistency

### Memory Layout

All buffers are **contiguous** and **aligned** for optimal GPU access:

```
states:      [num_envs, 4]  - (x, x_dot, theta, theta_dot)
rewards:     [num_envs]     - float32
terminals:   [num_envs]     - float32 (1.0 = done)
truncations: [num_envs]     - float32 (always 0.0 for CartPole)
actions:     [num_envs]     - int32 (0=left, 1=right)
```

### RNG

Uses **LCG (Linear Congruential Generator)** for initialization:
- Fast on GPU (no global state)
- Good enough for environment resets
- Each environment has independent seed

## Next Steps

1. **Install CUDA Toolkit** (see Requirements above)
2. **Build & Test**: `cargo test --release`
3. **Benchmark**: Validate 10M+ SPS on pure env stepping
4. **Integrate with PPO**: Phase 2 zero-copy tensors
5. **Expand to all envs**: MountainCar, Pendulum

## References

- [cudarc](https://github.com/coreylowman/cudarc) - Rust CUDA bindings
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [OpenAI Gym CartPole](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)
