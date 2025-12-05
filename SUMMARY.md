# Operant Performance Optimization Summary

## 🎯 Mission: Beat PufferLib Performance

**Current**: 40K SPS (3090 GPU + 5950X CPU)
**Target**: 1.5-2M SPS (50x improvement)
**PufferLib Benchmark**: ~1M SPS

---

## ✅ Work Completed Today

### 1. Fixed Critical NaN Bug
- **Root Cause**: Buffer.rs used `Arc::as_ptr()` (pointed to Arc metadata) instead of `.as_ptr()` (Vec data)
- **Fix**: Corrected pointer calculation in Buffer1D/2D/3D
- **Result**: ✅ All benchmarks pass, no NaN errors with 8192 envs

### 2. Simplified Benchmark Output
- Removed repetitive 15-line headers
- Single-line progress with `\r` overwrite
- Clean summary: `✓ Complete | 10,000k steps in 247.4321s | 40,445 SPS`

### 3. Fixed Default Parameters
- Changed from 8192 envs + 1M steps → 8192 envs + 10M steps
- Prevents 0-update benchmarks (1M steps < 1.05M steps/update)

---

## 🚀 Optimization Roadmap (6 Phases to 2M SPS)

### Phase 1: GPU-Resident Environments (Target: 500K SPS)
**Impact**: 12.5x speedup
**Status**: 🎯 Highest Priority

**Why**: Eliminates the PRIMARY bottleneck - CPU↔GPU data transfer
- Current: 256 transfers per update (128 steps × 2 directions)
- Solution: Run environments as CUDA kernels on GPU
- Tech: `cudarc` crate or raw CUDA + bindgen

**Files to Create**:
```
crates/operant-cuda/
├── src/
│   ├── lib.rs (Rust API)
│   ├── kernels/
│   │   ├── cartpole.cu
│   │   ├── mountain_car.cu
│   │   └── pendulum.cu
│   └── buffer.rs (CUDA memory management)
└── build.rs (NVCC compilation)
```

**Implementation Strategy**:
1. Start with simple CartPole kernel
2. Maintain SoA layout on GPU (coalesced memory access)
3. Launch 1 kernel for all 8192 envs (massive parallelism)
4. Expose via PyO3 as `CudaCartPole(num_envs, device_id)`

---

### Phase 2: Zero-Copy GPU Tensors (Target: 750K SPS)
**Impact**: 1.5x speedup on Phase 1
**Status**: ⏳ Depends on Phase 1

**Why**: Eliminates `obs.numpy()` → `torch.from_numpy()` conversions
- Current: CPU buffer intermediate
- Solution: Share CUDA device pointers directly

**Tech**: PyTorch's DLPack or raw CUDA IPC
```python
# Current (slow)
obs = env.reset()
obs_torch = torch.from_numpy(obs.numpy()).to('cuda')

# Target (zero-copy)
obs_cuda_ptr = env.reset_cuda()  # Returns GPU pointer
obs_torch = torch.cuda.from_dlpack(obs_cuda_ptr)  # Wraps existing memory
```

---

### Phase 3: Decoupled TUI Monitor (Target: 0 SPS cost + UX win)
**Impact**: Crash isolation, remote monitoring
**Status**: 🎯 Low-hanging fruit, immediate value

**Why**: Current TUI runs in-process (crash risk, no detach)
- Solution: Separate `operant-monitor` binary reads mmap metrics

**User Experience**:
```bash
# Start training with auto-spawned monitor
operant train --config ppo.toml --monitor

# Headless on server
operant train --config ppo.toml --detach

# Attach to running training
operant monitor --latest
operant monitor abc123-run-id
```

**Files to Create**:
```
crates/operant-monitor/
├── src/
│   ├── main.rs (TUI binary)
│   └── reader.rs (mmap reader)
└── Cargo.toml

crates/operant-python/src/
└── metrics_writer.rs (replaces tui.rs for IPC)
```

---

### Phase 4: Kernel Fusion & Batching (Target: 1.1M SPS)
**Impact**: 1.5x speedup on Phase 2
**Status**: 🚀 Advanced optimization

**Why**: Reduce CUDA kernel launch overhead
- Current: Separate kernels for `policy.forward()`, `env.step()`, `buffer.write()`
- Solution: Mega-kernel fuses entire rollout collection

**Tech**: Persistent thread blocks, shared memory for intermediate values

---

### Phase 5: Multi-Stream Async Execution (Target: 1.5M SPS)
**Impact**: 1.4x speedup on Phase 4
**Status**: 🚀 Advanced optimization

**Why**: Overlap computation with data transfer
- Stream 1: Collect rollout N
- Stream 2: Train policy on rollout N-1
- Hide latency with double-buffering

---

### Phase 6: FP16 + Tensor Cores (Target: 2M+ SPS)
**Impact**: 1.3x speedup on Phase 5
**Status**: 🔥 Final polish

**Why**: Leverage 3090 tensor cores for 2x FP16 throughput
- Policy network in FP16 (training only)
- Environments stay FP32 (physics precision)

**Tech**: PyTorch AMP (Automatic Mixed Precision)

---

## 📊 Performance Projections

| Phase | SPS | Cumulative Gain | vs PufferLib |
|-------|-----|----------------|--------------|
| Current | 40K | 1x | 0.04x |
| Phase 1 | 500K | 12.5x | 0.5x |
| Phase 2 | 750K | 18.75x | 0.75x |
| Phase 3 | 750K | 18.75x | 0.75x |
| Phase 4 | 1.1M | 27.5x | **1.1x** ✅ |
| Phase 5 | 1.5M | 37.5x | **1.5x** 🚀 |
| Phase 6 | 2M+ | 50x+ | **2x** 🔥 |

---

## 🎨 Why Operant Will Win

### Technical Advantages
1. **Hybrid CPU+GPU** - SIMD environments fallback for CPU-only systems
2. **Zero Dependencies** - Pure Rust + PyTorch (no complex toolchains)
3. **Rust Safety** - No segfaults, no race conditions
4. **SoA Memory Layout** - Cache-optimal on both CPU and GPU

### UX Advantages
1. **Integrated TUI** - No external services (WandB/Tensorboard)
2. **"It Just Works"** - Auto-detects GPU, sane defaults
3. **Detachable Monitoring** - SSH-friendly, crash-isolated
4. **Clean API** - Gymnasium-compatible, Pythonic

### Performance Advantages
1. **Kernel Fusion** - Fewer kernel launches than competitors
2. **Zero-Copy Everywhere** - Buffer protocol + CUDA IPC
3. **Profile-Guided Optimization** - 10-20% free gains
4. **Compile-Time Specialization** - Rust generics + LLVM

---

## 📝 Implementation Timeline

### Weeks 1-2: Foundation
- [x] Fix NaN bugs (DONE!)
- [x] Benchmark cleanup (DONE!)
- [ ] Profile current bottlenecks (perf, nvidia-nsight)
- [ ] Prototype GPU CartPole kernel (validate 10x gain)

### Weeks 3-4: Phase 1 (GPU Envs)
- [ ] `operant-cuda` crate setup
- [ ] CartPole CUDA kernel
- [ ] MountainCar CUDA kernel
- [ ] Pendulum CUDA kernel
- [ ] PyO3 bindings for GPU envs

### Weeks 5-6: Phase 2 (Zero-Copy)
- [ ] CUDA IPC buffer integration
- [ ] PyTorch DLPack interface
- [ ] Benchmark validation vs PufferLib

### Week 7: Phase 3 (Decoupled TUI)
- [ ] `operant-monitor` binary
- [ ] Memory-mapped metrics writer
- [ ] CLI subcommands (`monitor`, `train`)

### Weeks 8-10: Phases 4-6 (Advanced)
- [ ] Kernel fusion
- [ ] Multi-stream execution
- [ ] FP16 tensor cores

### Week 11-12: Polish & Release
- [ ] Comprehensive benchmarks
- [ ] Documentation & examples
- [ ] Blog post: "How We Beat PufferLib"

---

## 🔧 Quick Wins (Do These Now)

### 1. Enable LTO in Cargo.toml
```toml
[profile.release]
lto = "fat"              # +5-10% speedup
codegen-units = 1        # +2-5% speedup
panic = "abort"          # +1-2% speedup
```

### 2. Profile-Guided Optimization
```bash
# 1. Build with instrumentation
RUSTFLAGS="-Cprofile-generate=/tmp/pgo" poetry run maturin develop --release

# 2. Run representative workload
poetry run python benches/bench.py --num-envs 8192 --total-timesteps 10000000

# 3. Rebuild with PGO
RUSTFLAGS="-Cprofile-use=/tmp/pgo" poetry run maturin develop --release
```
**Expected**: +10-20% speedup

### 3. AVX-512 SIMD (5950X supports it)
```rust
// Change from f32x8 to f32x16
use std::simd::f32x16;
const LANES: usize = 16;  // Was: 8
```
**Expected**: +30% speedup on environment simulation

---

## 📚 Resources

### Documentation
- [OPTIMIZATION_PLAN.md](./OPTIMIZATION_PLAN.md) - Detailed technical plan
- [CLAUDE.md](./CLAUDE.md) - Updated with performance targets
- [README.md](./README.md) - User-facing documentation

### Benchmarking
- [benches/bench.py](./benches/bench.py) - Current benchmark script
- Run: `poetry run python benches/bench.py` (defaults to 8192 envs, 10M steps)

### Key Files
- `crates/operant-envs/src/gymnasium/*.rs` - CPU SIMD environments
- `crates/operant-python/src/tui.rs` - Current TUI (atomic metrics)
- `python/operant/models/ppo.py` - PPO training loop

---

## 🎯 Next Actions

1. **Immediate**:
   - [ ] Apply LTO + codegen-units = 1 to Cargo.toml
   - [ ] Run PGO rebuild
   - [ ] Profile with `perf` to confirm bottleneck locations

2. **This Week**:
   - [ ] Start `operant-cuda` crate
   - [ ] Simple CartPole CUDA kernel prototype
   - [ ] Validate 10x speedup claim

3. **This Month**:
   - [ ] Complete Phase 1 (all envs on GPU)
   - [ ] Start Phase 2 (zero-copy tensors)
   - [ ] Publish initial benchmarks

---

**Remember**: The goal isn't just to match PufferLib - it's to **dominate** with 2x better performance + superior UX. Every optimization matters. Every cycle counts. 🚀
