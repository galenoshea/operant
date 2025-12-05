# Custom Environment Integration Guide

This guide shows how to integrate a custom Rust environment (like GridWalk from `custom_environment.rs`) into Operant's Python API.

## Overview

Creating a custom environment involves 4 steps:
1. Implement the `Environment` trait in Rust
2. Create a PyO3 wrapper
3. Register in the Python module
4. Use in training

## Step 1: Implement Environment Trait

See [`custom_environment.rs`](./custom_environment.rs) for a complete example.

Key points:
- Use **Struct-of-Arrays (SoA)** layout for SIMD optimization
- Pre-allocate all buffers in constructor
- Implement proper error handling with `Result` types
- Include episode statistics in `get_logs()`

```rust
use operant_core::{Environment, Result};

pub struct MyCustomEnv {
    num_envs: usize,
    // SoA buffers
    states: Vec<f32>,
    // ... other fields
}

impl Environment for MyCustomEnv {
    fn reset(&mut self, seed: u64) { /* ... */ }
    fn step(&mut self, actions: &[f32]) { /* ... */ }
    fn write_observations(&self, buffer: &mut [f32]) { /* ... */ }
    // ... other required methods
}
```

## Step 2: Create PyO3 Wrapper

Add to `crates/operant-python/src/lib.rs`:

```rust
use pyo3::prelude::*;
use operant_envs::gymnasium::gridwalk::GridWalk;  // Your env
use crate::buffer::{Buffer1D, Buffer2D};
use std::sync::Arc;

#[pyclass]
pub struct PyGridWalk {
    inner: GridWalk,
    num_envs: usize,
    // Pre-allocated Arc buffers for zero-copy
    obs_buffer: Arc<Vec<f32>>,
    reward_buffer: Arc<Vec<f32>>,
    terminal_buffer: Arc<Vec<f32>>,
    truncation_buffer: Arc<Vec<f32>>,
    action_buffer: Vec<f32>,
}

#[pymethods]
impl PyGridWalk {
    #[new]
    fn new(num_envs: usize, grid_size: i32) -> PyResult<Self> {
        let inner = GridWalk::new(num_envs, grid_size)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let obs_size = inner.observation_dim();
        Ok(Self {
            inner,
            num_envs,
            obs_buffer: Arc::new(vec![0.0; num_envs * obs_size]),
            reward_buffer: Arc::new(vec![0.0; num_envs]),
            terminal_buffer: Arc::new(vec![0.0; num_envs]),
            truncation_buffer: Arc::new(vec![0.0; num_envs]),
            action_buffer: vec![0.0; num_envs],
        })
    }

    fn reset(&mut self, seed: Option<u64>) -> PyResult<Buffer2D> {
        self.inner.reset(seed.unwrap_or(0));

        // Write to buffer (no allocation)
        let obs_buf = Arc::get_mut(&mut self.obs_buffer)
            .ok_or_else(|| PyValueError::new_err("Buffer still referenced"))?;
        self.inner.write_observations(obs_buf);

        // Return zero-copy buffer
        Buffer2D::from_flat_arc(
            Arc::clone(&self.obs_buffer),
            self.num_envs,
            self.inner.observation_dim(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn step(&mut self, actions: &PyAny) -> PyResult<(Buffer2D, Buffer1D, Buffer1D, Buffer1D)> {
        // Convert actions to Vec<f32>
        if let Ok(arr) = actions.downcast::<PyArray1<f32>>() {
            self.action_buffer.copy_from_slice(arr.readonly().as_slice()?);
        } else if let Ok(arr) = actions.downcast::<PyArray1<i32>>() {
            for (i, &val) in arr.readonly().as_slice()?.iter().enumerate() {
                self.action_buffer[i] = val as f32;
            }
        } else {
            return Err(PyTypeError::new_err("actions must be numpy array"));
        }

        // Step environment
        self.inner.step(&self.action_buffer);

        // Write to buffers (no allocation)
        let obs_buf = Arc::get_mut(&mut self.obs_buffer).unwrap();
        let rew_buf = Arc::get_mut(&mut self.reward_buffer).unwrap();
        let term_buf = Arc::get_mut(&mut self.terminal_buffer).unwrap();
        let trunc_buf = Arc::get_mut(&mut self.truncation_buffer).unwrap();

        self.inner.write_observations(obs_buf);
        self.inner.write_rewards(rew_buf);
        self.inner.write_terminals(term_buf);
        self.inner.write_truncations(trunc_buf);

        // Return zero-copy buffers
        Ok((
            Buffer2D::from_flat_arc(
                Arc::clone(&self.obs_buffer),
                self.num_envs,
                self.inner.observation_dim(),
            )
            .unwrap(),
            Buffer1D::from_vec_arc(Arc::clone(&self.reward_buffer)),
            Buffer1D::from_vec_arc(Arc::clone(&self.terminal_buffer)),
            Buffer1D::from_vec_arc(Arc::clone(&self.truncation_buffer)),
        ))
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.num_envs
    }

    #[getter]
    fn observation_space(&self) -> HashMap<String, Vec<f32>> {
        self.inner.observation_space()
    }

    #[getter]
    fn action_space(&self) -> HashMap<String, Vec<f32>> {
        self.inner.action_space()
    }

    fn get_logs(&self) -> HashMap<String, f32> {
        self.inner.get_logs()
    }
}

// Register in module
#[pymodule]
fn operant(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // ... existing registrations ...

    // Add custom environment
    let envs_module = PyModule::new(py, "envs")?;
    envs_module.add_class::<PyGridWalk>()?;
    m.add_submodule(envs_module)?;

    Ok(())
}
```

## Step 3: Add Python Wrapper

Update `python/operant/__init__.py`:

```python
def _create_gridwalk_vec_env(num_envs: int = 1, grid_size: int = 10) -> _VecEnvWrapper:
    """Create a vectorized GridWalk environment."""
    return _VecEnvWrapper(_rust_envs.PyGridWalk(num_envs, grid_size))

class _EnvsModule:
    @staticmethod
    def GridWalk(num_envs: int = 1, grid_size: int = 10) -> _VecEnvWrapper:
        """Create a vectorized GridWalk environment.

        Args:
            num_envs: Number of parallel environments
            grid_size: Size of grid (grid_size × grid_size)

        Returns:
            Wrapped GridWalk environment
        """
        return _create_gridwalk_vec_env(num_envs, grid_size)

    def __dir__(self):
        return [
            "CartPole", "MountainCar", "Pendulum",
            "GridWalk",  # Add custom environment
        ]
```

## Step 4: Use in Training

```python
from operant.envs import GridWalk
from operant.models import PPO, PPOConfig
from pathlib import Path

# Create custom environment
env = GridWalk(num_envs=4096, grid_size=10)

# Configure training
config = PPOConfig(
    lr=3e-4,
    n_steps=128,
    batch_size=256,
    checkpoint_dir=Path("./checkpoints"),
    checkpoint_interval=10,
)

# Train
model = PPO(env, config=config)
model.learn(total_timesteps=1_000_000)
```

## Best Practices

### Memory Layout
- **Always use SoA (Struct-of-Arrays)** for vectorized data
- Keep separate arrays for each component (x, y, z, etc.)
- This enables SIMD auto-vectorization by the compiler

```rust
// GOOD: SoA layout
struct MyEnv {
    pos_x: Vec<f32>,  // All X positions
    pos_y: Vec<f32>,  // All Y positions
}

// BAD: AoS (Array-of-Structs) layout - prevents SIMD
struct Position { x: f32, y: f32 }
struct MyEnv {
    positions: Vec<Position>,  // Interleaved data
}
```

### Zero-Copy Buffers
- Use `Arc<Vec<f32>>` for all observation/reward/done buffers
- Never clone buffers in `step()` or `reset()`
- PyTorch can access Arc'd memory directly via buffer protocol

### Error Handling
- Return `Result` types from constructors
- Validate configuration parameters
- Use descriptive error messages

```rust
pub fn new(num_envs: usize) -> Result<Self> {
    if num_envs == 0 {
        return Err(OperantError::InvalidConfig {
            param: "num_envs".to_string(),
            message: "must be at least 1".to_string(),
        });
    }
    Ok(Self { /* ... */ })
}
```

### Episode Statistics
- Track episode count, mean reward, mean length
- Reset trackers on episode boundaries
- Return from `get_logs()` for training monitoring

### Auto-Reset
- Always auto-reset environments on terminal/truncation
- This maintains constant batch size for neural networks
- Users never see terminal states (absorbed by reset)

## Performance Tips

1. **Pre-allocate everything** - No allocations in `step()`
2. **Use SIMD-friendly operations** - Simple loops over SoA arrays
3. **Batch operations** - Process all environments in single loop
4. **Avoid branches** - Use branchless math where possible
5. **Profile** - Use `cargo bench` to measure performance

## Testing

Add tests to verify correctness:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_creation() {
        let env = MyEnv::new(4).unwrap();
        assert_eq!(env.num_envs(), 4);
    }

    #[test]
    fn test_invalid_config() {
        assert!(MyEnv::new(0).is_err());
    }

    #[test]
    fn test_reset_observations() {
        let mut env = MyEnv::new(4).unwrap();
        env.reset(42);

        let mut obs = vec![0.0; 4 * env.observation_dim()];
        env.write_observations(&mut obs);

        // Verify observations are valid
        for &val in &obs {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_step_rewards() {
        let mut env = MyEnv::new(4).unwrap();
        env.reset(42);

        let actions = vec![0.0; 4];
        env.step(&actions);

        let mut rewards = vec![0.0; 4];
        env.write_rewards(&mut rewards);

        // Verify rewards are computed
        assert!(rewards.iter().any(|&r| r != 0.0));
    }
}
```

## Example Environments in Operant

Study these for reference:
- `crates/operant-envs/src/gymnasium/cartpole.rs` - Discrete actions
- `crates/operant-envs/src/gymnasium/pendulum.rs` - Continuous actions
- `crates/operant-envs/src/gymnasium/mountain_car.rs` - Simple physics

## Resources

- [Environment Trait Documentation](../crates/operant-core/src/env.rs)
- [Buffer Protocol Implementation](../crates/operant-python/src/buffer.rs)
- [PyO3 Documentation](https://pyo3.rs/)
