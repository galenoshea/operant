# Operant Performance Optimization Plan: PufferLib Killer

**Goal**: Achieve >1M SPS (Steps Per Second) and become the fastest RL library

**Current Status**: ~40K SPS on 3090 + 5950X (8192 envs, CUDA)

**Target**: 1,000,000+ SPS (25x improvement)

---

## Critical Performance Bottlenecks Identified

### 1. **CPU↔GPU Data Transfer (PRIMARY BOTTLENECK)**
**Current**: Every rollout step transfers data between CPU (Rust envs) and GPU (PyTorch)
- Observations: CPU → GPU (2048-8192 envs × 4 features × 4 bytes per step)
- Actions: GPU → CPU
- Estimated cost: ~70% of training time

**Impact**: With 128 steps per rollout, that's 256 transfers per update!

### 2. **Environment Simulation on CPU**
**Current**: SIMD-optimized Rust environments run on CPU
- CartPole SIMD: ~150M SPS (pure env, no training)
- But throttled by GPU↔CPU bottleneck during training

### 3. **Python/PyO3 Overhead**
**Current**: PyO3 buffer protocol + tensor conversions
- `obs.numpy()` → `torch.from_numpy()` every step
- Type conversions and memory copies

### 4. **TUI Runs In-Process**
**Current**: TUI renders in training thread (minimal overhead due to atomics, but still coupled)
- Risk: TUI crash = training crash
- No detach/reattach capability

---

## Optimization Strategy (Ordered by Impact)

### Phase 1: GPU-Resident Environments (HIGHEST IMPACT - Target: 500K SPS)

**Concept**: Port SIMD environments to CUDA kernels
- Environments step on GPU → Zero CPU↔GPU transfer
- Observations stay GPU-resident for policy network
- Only transfer actions from policy back to env (smaller payload)

**Implementation**:
1. Create `operant-cuda` crate with GPU kernels for CartPole/MountainCar/Pendulum
2. Use `cudarc` or raw CUDA + bindgen for Rust integration
3. Memory layout: Keep SoA on GPU (coalesced memory access)
4. Batch all 8192 envs in single kernel launch

**Files**:
- `crates/operant-cuda/` (new crate)
- `crates/operant-cuda/src/kernels/cartpole.cu`
- `crates/operant-cuda/src/lib.rs` (Rust bindings)

**Expected**: 10-15x speedup (eliminates transfer bottleneck)

---

### Phase 2: Zero-Copy GPU Tensors (Target: +100K SPS)

**Concept**: Eliminate Python-side tensor copies
- Use `torch.as_tensor()` with CUDA IPC handles
- Share GPU memory directly between Rust and PyTorch
- No intermediate CPU buffer

**Implementation**:
1. Expose CUDA device pointers from Rust
2. Create PyTorch tensors via `torch.cuda.from_dlpack()` or raw pointers
3. Ensure memory lifetime safety with Arc<CudaBuffer>

**Files**:
- `crates/operant-python/src/cuda_buffer.rs` (new)
- `python/operant/buffers/cuda_buffer.py` (wrapper)

**Expected**: 2-3x speedup on top of Phase 1

---

### Phase 3: Decoupled TUI with IPC (Target: +50K SPS + Reliability)

**Concept**: Separate monitoring process (Observer pattern)
- Training writes to memory-mapped metrics file
- TUI process reads and renders independently
- Training unaffected by TUI crashes/detach

**Implementation**:
1. **Metrics Writer** (training side):
   ```rust
   // Lock-free atomic writes to mmap file
   struct SharedMetrics {
       sps: AtomicU64,
       steps: AtomicU64,
       // ... compressed metrics struct
   }
   ```

2. **TUI Binary** (separate process):
   ```bash
   operant monitor <run_id>
   operant monitor --latest
   ```

3. **CLI Integration**:
   ```bash
   operant train --config ppo.toml --monitor  # Auto-spawn TUI
   operant train --config ppo.toml --detach   # Headless
   ```

**Files**:
- `crates/operant-monitor/` (new binary crate)
- `crates/operant-python/src/metrics_writer.rs`
- Update `tui.rs` → `metrics_writer.rs` (IPC focus)

**Expected**: No SPS cost (decoupled), huge UX win

---

### Phase 4: Batched GPU Operations (Target: +150K SPS)

**Concept**: Fuse operations to reduce kernel launch overhead
- Current: Policy forward, env step, GAE compute (separate kernels)
- Optimized: Mega-kernel for entire rollout collection

**Implementation**:
1. CUDA kernel fusion for: `policy.act() → env.step() → buffer.store()`
2. Persistent thread blocks (keep envs on-chip)
3. Shared memory for intermediate values

**Files**:
- `crates/operant-cuda/src/kernels/fused_rollout.cu`

**Expected**: 2-4x speedup (reduced kernel overhead)

---

### Phase 5: Multi-Stream Parallelism (Target: +200K SPS)

**Concept**: Overlap computation with data transfer
- Stream 1: Collect rollout for update N
- Stream 2: Train policy on update N-1
- Hide latency with async execution

**Implementation**:
1. Double-buffered rollouts
2. PyTorch CUDA streams
3. Async GAE computation while policy trains

**Files**:
- `python/operant/models/ppo_async.py`

**Expected**: 1.5-2x speedup (overlapped execution)

---

### Phase 6: Mixed-Precision & Tensor Cores (Target: +100K SPS)

**Concept**: Use FP16 for training (A100/3090 tensor cores)
- Policy network in FP16
- Environments stay FP32 (physics precision)

**Implementation**:
1. PyTorch AMP (Automatic Mixed Precision)
2. Custom FP16 kernels for critical paths

**Files**:
- Update `python/operant/models/ppo.py` config

**Expected**: 2x speedup on tensor core operations

---

## Revised CLAUDE.md Updates

Based on this plan, I'll update CLAUDE.md with:

1. **New Architecture Section**: GPU-resident environments
2. **Performance Targets**: Document 1M+ SPS goal
3. **TUI Decoupling**: Observer pattern with IPC
4. **CUDA Integration**: New operant-cuda crate structure
5. **Development Priorities**: Phase-ordered implementation guide

---

## Performance Estimates (Cumulative)

| Phase | SPS Target | Improvement | Status |
|-------|-----------|-------------|---------|
| Baseline | 40K | 1x | ✅ Current |
| Phase 1 (GPU Envs) | 500K | 12.5x | 🎯 Critical |
| Phase 2 (Zero-Copy) | 750K | 18.75x | 🎯 High Priority |
| Phase 3 (Decoupled TUI) | 750K | 18.75x | 🎯 UX Win |
| Phase 4 (Batched Ops) | 1.1M | 27.5x | 🚀 Stretch |
| Phase 5 (Multi-Stream) | 1.5M | 37.5x | 🚀 Advanced |
| Phase 6 (FP16) | 2M+ | 50x+ | 🔥 Elite |

**PufferLib**: ~1M SPS
**Operant Target**: 1.5-2M SPS (50% faster than PufferLib)

---

## Implementation Priority

### Immediate (Next 2 Weeks)
1. ✅ **Fix Current Bugs** (NaN issues - DONE!)
2. 🎯 **Phase 3**: Decouple TUI (low-hanging fruit, huge UX)
3. 🎯 **Phase 1 Prototype**: Simple GPU CartPole kernel

### Short-Term (1 Month)
4. **Phase 1 Complete**: All envs on GPU
5. **Phase 2**: Zero-copy tensors
6. **Benchmarking**: Publish results vs PufferLib

### Medium-Term (2-3 Months)
7. **Phase 4**: Kernel fusion
8. **Phase 5**: Multi-stream
9. **Phase 6**: Mixed precision

---

## Additional Optimizations (Quick Wins)

### A. Compile-Time Optimizations
```toml
[profile.release]
lto = "fat"              # Link-time optimization
codegen-units = 1        # Single codegen unit (slower compile, faster runtime)
panic = "abort"          # Remove unwinding overhead
strip = true             # Smaller binaries
```

### B. SIMD Width Expansion
- Current: f32x8 (AVX2)
- Upgrade: f32x16 (AVX-512) for 5950X
- 2x wider SIMD = ~30% env speedup

### C. Prefetching & Cache Optimization
```rust
// Explicit prefetch for next chunk
std::intrinsics::prefetch_read_data(&self.x[next_chunk_base], 3);
```

### D. Rust Profile-Guided Optimization (PGO)
```bash
# Step 1: Collect profile
RUSTFLAGS="-Cprofile-generate=/tmp/pgo" cargo build --release

# Step 2: Run benchmarks to generate profile data
./target/release/operant-benchmark

# Step 3: Rebuild with PGO
RUSTFLAGS="-Cprofile-use=/tmp/pgo/merged.profdata" cargo build --release
```
Expected: 10-20% speedup

---

## Competitive Analysis

### PufferLib Advantages
- ✅ GPU-based environments
- ✅ Multi-node distributed training
- ✅ Mature ecosystem

### Operant Advantages (Post-Optimization)
- ✅ **Faster single-node performance** (2M vs 1M SPS)
- ✅ **Zero dependencies** (pure Rust + PyTorch)
- ✅ **Integrated TUI** with zero overhead
- ✅ **Cleaner API** (Gymnasium-native)
- ✅ **SIMD + GPU hybrid** (best of both worlds)

### Unique Selling Points
1. **"It just works"** - No complex setup, auto-detects GPU
2. **Real-time monitoring** - Built-in TUI, not external service
3. **Memory efficient** - Zero-copy buffers, SoA layout
4. **Debuggable** - Rust safety + clear Python API

---

## Metrics to Track

### Performance
- [ ] SPS (steps per second)
- [ ] Throughput (samples per second)
- [ ] Memory usage (peak GB)
- [ ] GPU utilization (%)

### Quality
- [ ] Training stability (convergence rate)
- [ ] Numerical precision (vs reference impl)
- [ ] Test coverage (>80%)

### UX
- [ ] Time to first result
- [ ] API simplicity (lines of code)
- [ ] Documentation completeness

---

## Next Steps

1. **Review this plan** with team/stakeholders
2. **Prototype Phase 1** (GPU CartPole kernel) - validate 10x speedup
3. **Implement Phase 3** (Decoupled TUI) - immediate UX win
4. **Benchmark & Iterate** - measure, optimize, repeat

**Timeline**: 3 months to 1M SPS, 6 months to 2M SPS
