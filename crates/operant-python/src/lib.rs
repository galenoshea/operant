//! Python bindings for Operant RL environments.
//!
//! Provides PyO3-based bindings with zero-copy buffer protocol support for PyTorch.

#![cfg_attr(feature = "simd", feature(portable_simd))]

mod buffer;
use buffer::{Buffer1D, Buffer2D, Buffer3D};

use operant_core::Environment;
use operant_envs::{CartPole, MountainCar, Pendulum};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use operant_cuda::CartPoleGpu;

#[cfg(feature = "tui")]
mod tui;
#[cfg(feature = "tui")]
use tui::TUILogger;

mod rl;
use rl::{AsyncEnvPool, RolloutBuffer, RunningNormalizer};

/// High-performance vectorized CartPole environment.
///
/// Uses SIMD-optimized Rust implementation with zero-copy numpy arrays.
#[pyclass]
pub struct PyCartPoleVecEnv {
    inner: CartPole,
    num_envs: usize,
    obs_buffer: Arc<Vec<f32>>,
    reward_buffer: Arc<Vec<f32>>,
    terminal_buffer: Arc<Vec<f32>>,  // Changed to f32 for zero-copy
    truncation_buffer: Arc<Vec<f32>>,  // Changed to f32 for zero-copy
    action_buffer: Vec<f32>,  // Not shared, keep as Vec
}

#[pymethods]
impl PyCartPoleVecEnv {
    /// Create a new vectorized CartPole environment.
    ///
    /// # Arguments
    ///
    /// * `num_envs` - Number of parallel environment instances
    /// * `workers` - Number of worker threads (1 = single-threaded, >1 = parallel with rayon).
    ///               Requires the `parallel` feature to be enabled.
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if `num_envs` is 0.
    #[new]
    #[pyo3(signature = (num_envs, workers=None))]
    pub fn new(num_envs: usize, workers: Option<usize>) -> PyResult<Self> {
        let workers = workers.unwrap_or(1);
        let inner = CartPole::with_workers(num_envs, workers)?;
        Ok(Self {
            inner,
            num_envs,
            obs_buffer: Arc::new(vec![0.0; num_envs * 4]),
            reward_buffer: Arc::new(vec![0.0; num_envs]),
            terminal_buffer: Arc::new(vec![0.0; num_envs]),  // Now f32
            truncation_buffer: Arc::new(vec![0.0; num_envs]),  // Now f32
            action_buffer: vec![0.0; num_envs],
        })
    }

    /// Reset all environments.
    ///
    /// Returns observations as a Buffer2D of shape (num_envs, 4).
    /// Compatible with PyTorch via buffer protocol (use `torch.as_tensor()`).
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if buffer creation fails.
    pub fn reset(
        &mut self,
        seed: Option<u64>,
    ) -> PyResult<Buffer2D> {
        let seed = seed.unwrap_or(0);
        self.inner.reset(seed);

        // Allocate new buffer for reset observations
        let mut obs_buf = vec![0.0f32; self.num_envs * 4];
        self.inner.write_observations(&mut obs_buf);

        // Zero-copy: Wrap in Arc and share with Python
        Buffer2D::from_flat_arc(Arc::new(obs_buf), self.num_envs, 4)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Step all environments with the given actions.
    ///
    /// # Arguments
    ///
    /// * `actions` - Any array-like object (PyTorch tensor, list, etc.) with shape (num_envs,)
    ///
    /// # Returns
    ///
    /// Tuple of (observations, rewards, terminals, truncations) where:
    /// - observations: Buffer2D shape (num_envs, 4), dtype float32
    /// - rewards: Buffer1D shape (num_envs,), dtype float32
    /// - terminals: Buffer1D shape (num_envs,), dtype float32 (0.0 or 1.0)
    /// - truncations: Buffer1D shape (num_envs,), dtype float32 (0.0 or 1.0)
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if actions cannot be converted or have wrong size.
    pub fn step(
        &mut self,
        actions: &Bound<'_, PyAny>,
    ) -> PyResult<(Buffer2D, Buffer1D, Buffer1D, Buffer1D)> {
        // Extract actions from any array-like object (tensor, list, etc.)
        // Try to get as i32 slice first (for discrete actions)
        let actions_vec: Vec<i32> = actions.extract()?;

        if actions_vec.len() != self.num_envs {
            return Err(PyValueError::new_err(format!(
                "Expected {} actions, got {}",
                self.num_envs,
                actions_vec.len()
            )));
        }

        // Reuse action buffer to eliminate allocation in hot loop
        self.action_buffer.clear();
        self.action_buffer
            .extend(actions_vec.iter().map(|&a| a as f32));

        self.inner.step_auto_reset(&self.action_buffer);

        // Allocate new buffers for this step's outputs
        // Python will hold references to these via Arc, preventing reuse
        // This is the correct "zero-copy" design: Python accesses Rust memory directly,
        // but we allocate fresh buffers each step to avoid reference conflicts
        let mut obs_buf = vec![0.0f32; self.num_envs * 4];
        let mut rew_buf = vec![0.0f32; self.num_envs];
        let mut term_buf = vec![0.0f32; self.num_envs];
        let mut trunc_buf = vec![0.0f32; self.num_envs];

        self.inner.write_observations(&mut obs_buf);
        self.inner.write_rewards(&mut rew_buf);

        // Write terminals/truncations as u8, then convert to f32
        let mut temp_u8 = vec![0u8; self.num_envs];
        self.inner.write_terminals(&mut temp_u8);
        for (i, &val) in temp_u8.iter().enumerate() {
            term_buf[i] = val as f32;
        }

        self.inner.write_truncations(&mut temp_u8);
        for (i, &val) in temp_u8.iter().enumerate() {
            trunc_buf[i] = val as f32;
        }

        // Zero-copy: Wrap in Arc and share with Python
        // Python can access this memory directly via buffer protocol
        let obs = Buffer2D::from_flat_arc(Arc::new(obs_buf), self.num_envs, 4)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rewards = Buffer1D::from_vec_arc(Arc::new(rew_buf));
        let terminals = Buffer1D::from_vec_arc(Arc::new(term_buf));
        let truncations = Buffer1D::from_vec_arc(Arc::new(trunc_buf));

        Ok((obs, rewards, terminals, truncations))
    }

    /// Get the number of environments.
    #[getter]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Get the observation shape per environment.
    #[getter]
    pub fn observation_size(&self) -> usize {
        4
    }

    /// Gymnasium-compatible observation space specification.
    ///
    /// Returns a dict matching gymnasium.spaces.Box format:
    /// - shape: (4,) for [cart_pos, cart_vel, pole_angle, pole_vel]
    /// - low: lower bounds (unbounded for velocities)
    /// - high: upper bounds (unbounded for velocities)
    /// - dtype: "float32"
    #[getter]
    pub fn observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("shape", (4,))?;
        dict.set_item("dtype", "float32")?;
        // CartPole bounds: cart_position, cart_velocity, pole_angle, pole_angular_velocity
        dict.set_item("low", [-4.8_f32, f32::NEG_INFINITY, -0.418, f32::NEG_INFINITY])?;
        dict.set_item("high", [4.8_f32, f32::INFINITY, 0.418, f32::INFINITY])?;
        Ok(dict)
    }

    /// Gymnasium-compatible observation space for a single environment.
    #[getter]
    pub fn single_observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.observation_space(py)
    }

    /// Gymnasium-compatible action space specification.
    ///
    /// Returns a dict matching gymnasium.spaces.Discrete format:
    /// - n: 2 (push left or push right)
    /// - dtype: "int32"
    #[getter]
    pub fn action_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("n", 2)?;
        dict.set_item("dtype", "int32")?;
        Ok(dict)
    }

    /// Gymnasium-compatible action space for a single environment.
    #[getter]
    pub fn single_action_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.action_space(py)
    }

    /// Get episode statistics since last clear.
    ///
    /// Returns dict with:
    /// - episode_count: number of completed episodes
    /// - total_reward: sum of rewards across all completed episodes
    /// - total_steps: sum of steps across all completed episodes
    /// - mean_reward: average episode reward (or 0 if no episodes)
    /// - mean_length: average episode length (or 0 if no episodes)
    pub fn get_logs(&self) -> std::collections::HashMap<&str, f32> {
        let log = self.inner.get_log();
        let mut result = std::collections::HashMap::new();
        result.insert("episode_count", log.episode_count as f32);
        result.insert("total_reward", log.total_reward);
        result.insert("total_steps", log.total_steps as f32);

        let mean_reward = if log.episode_count > 0 {
            log.total_reward / log.episode_count as f32
        } else {
            0.0
        };
        let mean_length = if log.episode_count > 0 {
            log.total_steps as f32 / log.episode_count as f32
        } else {
            0.0
        };
        result.insert("mean_reward", mean_reward);
        result.insert("mean_length", mean_length);
        result
    }

    /// Clear episode statistics.
    pub fn clear_logs(&mut self) {
        self.inner.clear_log();
    }
}

/// High-performance vectorized MountainCar environment.
///
/// Uses SIMD-optimized Rust implementation with zero-copy numpy arrays.
#[pyclass]
pub struct PyMountainCarVecEnv {
    inner: MountainCar,
    num_envs: usize,
    obs_buffer: Arc<Vec<f32>>,
    reward_buffer: Arc<Vec<f32>>,
    terminal_buffer: Arc<Vec<f32>>,  // Changed to f32 for zero-copy
    truncation_buffer: Arc<Vec<f32>>,  // Changed to f32 for zero-copy
    action_buffer: Vec<f32>,  // Not shared, keep as Vec
}

#[pymethods]
impl PyMountainCarVecEnv {
    /// Create a new vectorized MountainCar environment.
    ///
    /// # Arguments
    ///
    /// * `num_envs` - Number of parallel environment instances
    /// * `workers` - Number of worker threads (1 = single-threaded, >1 = parallel with rayon).
    ///               Requires the `parallel` feature to be enabled.
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if `num_envs` is 0.
    #[new]
    #[pyo3(signature = (num_envs, workers=None))]
    pub fn new(num_envs: usize, workers: Option<usize>) -> PyResult<Self> {
        let workers = workers.unwrap_or(1);
        let inner = MountainCar::with_workers(num_envs, workers)?;
        Ok(Self {
            inner,
            num_envs,
            obs_buffer: Arc::new(vec![0.0; num_envs * 2]),
            reward_buffer: Arc::new(vec![0.0; num_envs]),
            terminal_buffer: Arc::new(vec![0.0; num_envs]),  // Now f32
            truncation_buffer: Arc::new(vec![0.0; num_envs]),  // Now f32
            action_buffer: vec![0.0; num_envs],
        })
    }

    /// Reset all environments.
    ///
    /// Returns observations as a numpy array of shape (num_envs, 2).
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if buffer creation fails.
    pub fn reset(
        &mut self,
        seed: Option<u64>,
    ) -> PyResult<Buffer2D> {
        let seed = seed.unwrap_or(0);
        self.inner.reset(seed);

        // Allocate new buffer for reset observations
        let mut obs_buf = vec![0.0f32; self.num_envs * 2];
        self.inner.write_observations(&mut obs_buf);

        // Zero-copy: Wrap in Arc and share with Python
        Buffer2D::from_flat_arc(Arc::new(obs_buf), self.num_envs, 2)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Step all environments with the given actions.
    ///
    /// # Arguments
    ///
    /// * `actions` - Any array-like object with shape (num_envs,), values 0, 1, or 2
    ///
    /// # Returns
    ///
    /// Tuple of (observations, rewards, terminals, truncations) where:
    /// - observations: Buffer2D shape (num_envs, 2), dtype float32
    /// - rewards: Buffer1D shape (num_envs,), dtype float32
    /// - terminals: Buffer1D shape (num_envs,), dtype float32
    /// - truncations: Buffer1D shape (num_envs,), dtype float32
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if actions cannot be converted or have wrong size.
    pub fn step(
        &mut self,
        actions: &Bound<'_, PyAny>,
    ) -> PyResult<(Buffer2D, Buffer1D, Buffer1D, Buffer1D)> {
        // Extract actions from any array-like object
        let actions_vec: Vec<i32> = actions.extract()?;

        if actions_vec.len() != self.num_envs {
            return Err(PyValueError::new_err(format!(
                "Expected {} actions, got {}",
                self.num_envs,
                actions_vec.len()
            )));
        }

        // Reuse action buffer to eliminate allocation in hot loop
        self.action_buffer.clear();
        self.action_buffer
            .extend(actions_vec.iter().map(|&a| a as f32));

        self.inner.step_auto_reset(&self.action_buffer);

        // Allocate new buffers for this step's outputs
        let mut obs_buf = vec![0.0f32; self.num_envs * 2];
        let mut rew_buf = vec![0.0f32; self.num_envs];
        let mut term_buf = vec![0.0f32; self.num_envs];
        let mut trunc_buf = vec![0.0f32; self.num_envs];

        self.inner.write_observations(&mut obs_buf);
        self.inner.write_rewards(&mut rew_buf);

        // Write terminals/truncations as u8, then convert to f32
        let mut temp_u8 = vec![0u8; self.num_envs];
        self.inner.write_terminals(&mut temp_u8);
        for (i, &val) in temp_u8.iter().enumerate() {
            term_buf[i] = val as f32;
        }

        self.inner.write_truncations(&mut temp_u8);
        for (i, &val) in temp_u8.iter().enumerate() {
            trunc_buf[i] = val as f32;
        }

        // Zero-copy: Wrap in Arc and share with Python
        let obs = Buffer2D::from_flat_arc(Arc::new(obs_buf), self.num_envs, 2)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rewards = Buffer1D::from_vec_arc(Arc::new(rew_buf));
        let terminals = Buffer1D::from_vec_arc(Arc::new(term_buf));
        let truncations = Buffer1D::from_vec_arc(Arc::new(trunc_buf));

        Ok((obs, rewards, terminals, truncations))
    }

    /// Get the number of environments.
    #[getter]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Get the observation shape per environment.
    #[getter]
    pub fn observation_size(&self) -> usize {
        2
    }

    /// Gymnasium-compatible observation space specification.
    ///
    /// Returns a dict matching gymnasium.spaces.Box format:
    /// - shape: (2,) for [position, velocity]
    /// - low: lower bounds
    /// - high: upper bounds
    /// - dtype: "float32"
    #[getter]
    pub fn observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("shape", (2,))?;
        dict.set_item("dtype", "float32")?;
        // MountainCar bounds: position [-1.2, 0.6], velocity [-0.07, 0.07]
        dict.set_item("low", [-1.2_f32, -0.07])?;
        dict.set_item("high", [0.6_f32, 0.07])?;
        Ok(dict)
    }

    /// Gymnasium-compatible observation space for a single environment.
    #[getter]
    pub fn single_observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.observation_space(py)
    }

    /// Gymnasium-compatible action space specification.
    ///
    /// Returns a dict matching gymnasium.spaces.Discrete format:
    /// - n: 3 (push left, no push, push right)
    /// - dtype: "int32"
    #[getter]
    pub fn action_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("n", 3)?;
        dict.set_item("dtype", "int32")?;
        Ok(dict)
    }

    /// Gymnasium-compatible action space for a single environment.
    #[getter]
    pub fn single_action_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.action_space(py)
    }

    /// Get episode statistics since last clear.
    ///
    /// Returns dict with:
    /// - episode_count: number of completed episodes
    /// - total_reward: sum of rewards across all completed episodes
    /// - total_steps: sum of steps across all completed episodes
    /// - mean_reward: average episode reward (or 0 if no episodes)
    /// - mean_length: average episode length (or 0 if no episodes)
    pub fn get_logs(&self) -> std::collections::HashMap<&str, f32> {
        let log = self.inner.get_log();
        let mut result = std::collections::HashMap::new();
        result.insert("episode_count", log.episode_count as f32);
        result.insert("total_reward", log.total_reward);
        result.insert("total_steps", log.total_steps as f32);

        let mean_reward = if log.episode_count > 0 {
            log.total_reward / log.episode_count as f32
        } else {
            0.0
        };
        let mean_length = if log.episode_count > 0 {
            log.total_steps as f32 / log.episode_count as f32
        } else {
            0.0
        };
        result.insert("mean_reward", mean_reward);
        result.insert("mean_length", mean_length);
        result
    }

    /// Clear episode statistics.
    pub fn clear_logs(&mut self) {
        self.inner.clear_log();
    }
}

/// High-performance vectorized Pendulum environment.
///
/// Uses SIMD-optimized Rust implementation with zero-copy numpy arrays.
#[pyclass]
pub struct PyPendulumVecEnv {
    inner: Pendulum,
    num_envs: usize,
    obs_buffer: Arc<Vec<f32>>,
    reward_buffer: Arc<Vec<f32>>,
    terminal_buffer: Arc<Vec<f32>>,  // Changed to f32 for zero-copy
    truncation_buffer: Arc<Vec<f32>>,  // Changed to f32 for zero-copy
}

#[pymethods]
impl PyPendulumVecEnv {
    /// Create a new vectorized Pendulum environment.
    ///
    /// # Arguments
    ///
    /// * `num_envs` - Number of parallel environment instances
    /// * `workers` - Number of worker threads (1 = single-threaded, >1 = parallel with rayon).
    ///               Requires the `parallel` feature to be enabled.
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if `num_envs` is 0.
    #[new]
    #[pyo3(signature = (num_envs, workers=None))]
    pub fn new(num_envs: usize, workers: Option<usize>) -> PyResult<Self> {
        let workers = workers.unwrap_or(1);
        let inner = Pendulum::with_workers(num_envs, workers)?;
        Ok(Self {
            inner,
            num_envs,
            obs_buffer: Arc::new(vec![0.0; num_envs * 3]),
            reward_buffer: Arc::new(vec![0.0; num_envs]),
            terminal_buffer: Arc::new(vec![0.0; num_envs]),  // Now f32
            truncation_buffer: Arc::new(vec![0.0; num_envs]),  // Now f32
        })
    }

    /// Reset all environments.
    ///
    /// Returns observations as a numpy array of shape (num_envs, 3).
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if buffer creation fails.
    pub fn reset(
        &mut self,
        seed: Option<u64>,
    ) -> PyResult<Buffer2D> {
        let seed = seed.unwrap_or(0);
        self.inner.reset(seed);

        // Allocate new buffer for reset observations
        let mut obs_buf = vec![0.0f32; self.num_envs * 3];
        self.inner.write_observations(&mut obs_buf);

        // Zero-copy: Wrap in Arc and share with Python
        Buffer2D::from_flat_arc(Arc::new(obs_buf), self.num_envs, 3)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Step all environments with the given actions.
    ///
    /// # Arguments
    ///
    /// * `actions` - Any array-like object with shape (num_envs,), continuous torque values [-2.0, 2.0]
    ///
    /// # Returns
    ///
    /// Tuple of (observations, rewards, terminals, truncations) where:
    /// - observations: Buffer2D shape (num_envs, 3), dtype float32 (cos(theta), sin(theta), theta_dot)
    /// - rewards: Buffer1D shape (num_envs,), dtype float32
    /// - terminals: Buffer1D shape (num_envs,), dtype float32 (always 0.0)
    /// - truncations: Buffer1D shape (num_envs,), dtype float32
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if actions cannot be converted or have wrong size.
    pub fn step(
        &mut self,
        actions: &Bound<'_, PyAny>,
    ) -> PyResult<(Buffer2D, Buffer1D, Buffer1D, Buffer1D)> {
        // Extract actions from any array-like object
        let actions_vec: Vec<f32> = actions.extract()?;

        if actions_vec.len() != self.num_envs {
            return Err(PyValueError::new_err(format!(
                "Expected {} actions, got {}",
                self.num_envs,
                actions_vec.len()
            )));
        }

        self.inner.step_auto_reset(&actions_vec);

        // Allocate new buffers for this step's outputs
        let mut obs_buf = vec![0.0f32; self.num_envs * 3];
        let mut rew_buf = vec![0.0f32; self.num_envs];
        let mut term_buf = vec![0.0f32; self.num_envs];
        let mut trunc_buf = vec![0.0f32; self.num_envs];

        self.inner.write_observations(&mut obs_buf);
        self.inner.write_rewards(&mut rew_buf);

        // Write terminals/truncations as u8, then convert to f32
        let mut temp_u8 = vec![0u8; self.num_envs];
        self.inner.write_terminals(&mut temp_u8);
        for (i, &val) in temp_u8.iter().enumerate() {
            term_buf[i] = val as f32;
        }

        self.inner.write_truncations(&mut temp_u8);
        for (i, &val) in temp_u8.iter().enumerate() {
            trunc_buf[i] = val as f32;
        }

        // Zero-copy: Wrap in Arc and share with Python
        let obs = Buffer2D::from_flat_arc(Arc::new(obs_buf), self.num_envs, 3)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rewards = Buffer1D::from_vec_arc(Arc::new(rew_buf));
        let terminals = Buffer1D::from_vec_arc(Arc::new(term_buf));
        let truncations = Buffer1D::from_vec_arc(Arc::new(trunc_buf));

        Ok((obs, rewards, terminals, truncations))
    }

    /// Get the number of environments.
    #[getter]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Get the observation shape per environment.
    #[getter]
    pub fn observation_size(&self) -> usize {
        3
    }

    /// Gymnasium-compatible observation space specification.
    ///
    /// Returns a dict matching gymnasium.spaces.Box format:
    /// - shape: (3,) for [cos(theta), sin(theta), theta_dot]
    /// - low: lower bounds
    /// - high: upper bounds
    /// - dtype: "float32"
    #[getter]
    pub fn observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("shape", (3,))?;
        dict.set_item("dtype", "float32")?;
        // Pendulum bounds: cos(theta), sin(theta), angular velocity
        dict.set_item("low", [-1.0_f32, -1.0, -8.0])?;
        dict.set_item("high", [1.0_f32, 1.0, 8.0])?;
        Ok(dict)
    }

    /// Gymnasium-compatible observation space for a single environment.
    #[getter]
    pub fn single_observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.observation_space(py)
    }

    /// Gymnasium-compatible action space specification.
    ///
    /// Returns a dict matching gymnasium.spaces.Box format:
    /// - shape: (1,) for torque
    /// - low: -2.0
    /// - high: 2.0
    /// - dtype: "float32"
    #[getter]
    pub fn action_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("shape", (1,))?;
        dict.set_item("dtype", "float32")?;
        dict.set_item("low", [-2.0_f32])?;
        dict.set_item("high", [2.0_f32])?;
        Ok(dict)
    }

    /// Gymnasium-compatible action space for a single environment.
    #[getter]
    pub fn single_action_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.action_space(py)
    }

    /// Get episode statistics since last clear.
    ///
    /// Returns dict with:
    /// - episode_count: number of completed episodes
    /// - total_reward: sum of rewards across all completed episodes
    /// - total_steps: sum of steps across all completed episodes
    /// - mean_reward: average episode reward (or 0 if no episodes)
    /// - mean_length: average episode length (or 0 if no episodes)
    pub fn get_logs(&self) -> std::collections::HashMap<&str, f32> {
        let log = self.inner.get_log();
        let mut result = std::collections::HashMap::new();
        result.insert("episode_count", log.episode_count as f32);
        result.insert("total_reward", log.total_reward);
        result.insert("total_steps", log.total_steps as f32);

        let mean_reward = if log.episode_count > 0 {
            log.total_reward / log.episode_count as f32
        } else {
            0.0
        };
        let mean_length = if log.episode_count > 0 {
            log.total_steps as f32 / log.episode_count as f32
        } else {
            0.0
        };
        result.insert("mean_reward", mean_reward);
        result.insert("mean_length", mean_length);
        result
    }

    /// Clear episode statistics.
    pub fn clear_logs(&mut self) {
        self.inner.clear_log();
    }
}

/// GPU-accelerated vectorized CartPole environment.
///
/// Uses CUDA kernels for massively parallel environment stepping.
/// All physics computation happens on GPU with zero internal transfers.
#[cfg(feature = "cuda")]
#[pyclass]
pub struct PyCartPoleGpuEnv {
    inner: CartPoleGpu,
    num_envs: usize,
    device_id: usize,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl PyCartPoleGpuEnv {
    /// Create a new GPU-accelerated vectorized CartPole environment.
    ///
    /// # Arguments
    ///
    /// * `num_envs` - Number of parallel environment instances
    /// * `device_id` - CUDA device ID (default: 0)
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if GPU initialization fails or CUDA is not available.
    #[new]
    #[pyo3(signature = (num_envs, device_id=None))]
    pub fn new(num_envs: usize, device_id: Option<usize>) -> PyResult<Self> {
        let device_id = device_id.unwrap_or(0);
        let inner = CartPoleGpu::new(num_envs, device_id)
            .map_err(|e| PyValueError::new_err(format!("Failed to create GPU environment: {}", e)))?;
        Ok(Self {
            inner,
            num_envs,
            device_id,
        })
    }

    /// Reset all environments.
    ///
    /// Returns observations as a Buffer2D of shape (num_envs, 4).
    /// Compatible with PyTorch via buffer protocol (use `torch.as_tensor()`).
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if GPU reset or transfer fails.
    pub fn reset(&mut self, seed: Option<u64>) -> PyResult<Buffer2D> {
        // Reset on GPU
        self.inner
            .reset()
            .map_err(|e| PyValueError::new_err(format!("GPU reset failed: {}", e)))?;

        // Transfer observations from GPU to CPU
        let obs_cpu = self
            .inner
            .get_obs()
            .map_err(|e| PyValueError::new_err(format!("GPU→CPU transfer failed: {}", e)))?;

        // Wrap in zero-copy buffer for Python
        Buffer2D::from_flat_arc(Arc::new(obs_cpu), self.num_envs, 4)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Step all environments with the given actions.
    ///
    /// # Arguments
    ///
    /// * `actions` - Any array-like object (PyTorch tensor, list, etc.) with shape (num_envs,)
    ///
    /// # Returns
    ///
    /// Tuple of (observations, rewards, terminals, truncations) where:
    /// - observations: Buffer2D shape (num_envs, 4), dtype float32
    /// - rewards: Buffer1D shape (num_envs,), dtype float32
    /// - terminals: Buffer1D shape (num_envs,), dtype float32 (0.0 or 1.0)
    /// - truncations: Buffer1D shape (num_envs,), dtype float32 (0.0 or 1.0)
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if actions cannot be converted, have wrong size, or GPU step fails.
    pub fn step(
        &mut self,
        actions: &Bound<'_, PyAny>,
    ) -> PyResult<(Buffer2D, Buffer1D, Buffer1D, Buffer1D)> {
        // 1. Extract actions from Python (CPU)
        let actions_vec: Vec<i32> = actions.extract()?;

        if actions_vec.len() != self.num_envs {
            return Err(PyValueError::new_err(format!(
                "Expected {} actions, got {}",
                self.num_envs,
                actions_vec.len()
            )));
        }

        // 2. Transfer actions to GPU
        let actions_gpu = self
            .inner
            .device()
            .htod_copy(actions_vec)
            .map_err(|e| PyValueError::new_err(format!("CPU→GPU transfer failed: {}", e)))?;

        // 3. Step on GPU (all physics on GPU, zero internal transfer)
        self.inner
            .step(&actions_gpu)
            .map_err(|e| PyValueError::new_err(format!("GPU step failed: {}", e)))?;

        // 4. Transfer results from GPU to CPU
        let obs_cpu = self
            .inner
            .get_obs()
            .map_err(|e| PyValueError::new_err(format!("GPU→CPU obs transfer failed: {}", e)))?;
        let rewards_cpu = self
            .inner
            .get_rewards()
            .map_err(|e| PyValueError::new_err(format!("GPU→CPU rewards transfer failed: {}", e)))?;
        let terminals_cpu = self
            .inner
            .get_terminals()
            .map_err(|e| PyValueError::new_err(format!("GPU→CPU terminals transfer failed: {}", e)))?;
        let truncations_cpu = self
            .inner
            .get_truncations()
            .map_err(|e| PyValueError::new_err(format!("GPU→CPU truncations transfer failed: {}", e)))?;

        // 5. Wrap in Buffer types for zero-copy to PyTorch
        let obs = Buffer2D::from_flat_arc(Arc::new(obs_cpu), self.num_envs, 4)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rewards = Buffer1D::from_vec_arc(Arc::new(rewards_cpu));
        let terminals = Buffer1D::from_vec_arc(Arc::new(terminals_cpu));
        let truncations = Buffer1D::from_vec_arc(Arc::new(truncations_cpu));

        Ok((obs, rewards, terminals, truncations))
    }

    /// Get the number of environments.
    #[getter]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Get the observation shape per environment.
    #[getter]
    pub fn observation_size(&self) -> usize {
        4
    }

    /// Gymnasium-compatible observation space specification.
    ///
    /// Returns a dict matching gymnasium.spaces.Box format:
    /// - shape: (4,) for [cart_pos, cart_vel, pole_angle, pole_vel]
    /// - low: lower bounds (unbounded for velocities)
    /// - high: upper bounds (unbounded for velocities)
    /// - dtype: "float32"
    #[getter]
    pub fn observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("shape", (4,))?;
        dict.set_item("dtype", "float32")?;
        // CartPole bounds: cart_position, cart_velocity, pole_angle, pole_angular_velocity
        dict.set_item("low", [-4.8_f32, f32::NEG_INFINITY, -0.418, f32::NEG_INFINITY])?;
        dict.set_item("high", [4.8_f32, f32::INFINITY, 0.418, f32::INFINITY])?;
        Ok(dict)
    }

    /// Gymnasium-compatible observation space for a single environment.
    #[getter]
    pub fn single_observation_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.observation_space(py)
    }

    /// Gymnasium-compatible action space specification.
    ///
    /// Returns a dict matching gymnasium.spaces.Discrete format:
    /// - n: 2 (push left or push right)
    /// - dtype: "int32"
    #[getter]
    pub fn action_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("n", 2)?;
        dict.set_item("dtype", "int32")?;
        Ok(dict)
    }

    /// Gymnasium-compatible action space for a single environment.
    #[getter]
    pub fn single_action_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.action_space(py)
    }

    /// Get GPU device ID.
    #[getter]
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Get episode statistics since last clear.
    ///
    /// Returns dict with:
    /// - episode_count: number of completed episodes
    /// - total_reward: sum of rewards across all completed episodes
    /// - total_steps: sum of steps across all completed episodes
    /// - mean_reward: average episode reward (or 0 if no episodes)
    /// - mean_length: average episode length (or 0 if no episodes)
    pub fn get_logs(&self) -> std::collections::HashMap<&str, f32> {
        // GPU environment doesn't track logs yet (would need additional GPU→CPU transfers)
        // Return empty logs for now - PPO will work without them
        let mut result = std::collections::HashMap::new();
        result.insert("episode_count", 0.0);
        result.insert("total_reward", 0.0);
        result.insert("total_steps", 0.0);
        result.insert("mean_reward", 0.0);
        result.insert("mean_length", 0.0);
        result
    }

    /// Clear episode statistics.
    pub fn clear_logs(&mut self) {
        // No-op for now since we don't track logs
    }
}

/// Register operant.envs submodule with environment classes.
fn register_envs_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let envs_mod = PyModule::new(py, "envs")?;

    // Add environment classes to envs submodule
    envs_mod.add_class::<PyCartPoleVecEnv>()?;
    envs_mod.add_class::<PyMountainCarVecEnv>()?;
    envs_mod.add_class::<PyPendulumVecEnv>()?;

    // Add GPU environment if CUDA feature is enabled
    #[cfg(feature = "cuda")]
    envs_mod.add_class::<PyCartPoleGpuEnv>()?;

    // Register submodule with parent
    parent.add_submodule(&envs_mod)?;

    // Add to sys.modules for proper Python import
    py.import("sys")?
        .getattr("modules")?
        .set_item("operant.envs", envs_mod)?;

    Ok(())
}

/// Register TUI module if feature is enabled.
#[cfg(feature = "tui")]
fn register_tui_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<TUILogger>()?;
    // Alias for backwards compatibility
    parent.add("Logger", parent.getattr("TUILogger")?)?;
    Ok(())
}

/// Register RL utilities module with RolloutBuffer, RunningNormalizer, and AsyncEnvPool.
fn register_rl_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let rl_mod = PyModule::new(py, "_rl")?;

    // Add RL classes to _rl submodule
    rl_mod.add_class::<RolloutBuffer>()?;
    rl_mod.add_class::<RunningNormalizer>()?;
    rl_mod.add_class::<AsyncEnvPool>()?;

    // Register submodule with parent
    parent.add_submodule(&rl_mod)?;

    // Add to sys.modules for proper Python import
    py.import("sys")?
        .getattr("modules")?
        .set_item("operant._rl", rl_mod)?;

    Ok(())
}

/// Operant Python module.
#[pymodule]
fn operant(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register buffer types (for buffer protocol support)
    m.add_class::<Buffer1D>()?;
    m.add_class::<Buffer2D>()?;
    m.add_class::<Buffer3D>()?;

    // Register envs submodule
    register_envs_module(m)?;

    // Register RL utilities submodule
    register_rl_module(m)?;

    // Register TUI logger if feature enabled
    #[cfg(feature = "tui")]
    register_tui_module(m)?;

    // Backwards compatibility - deprecated imports at root level
    m.add_class::<PyCartPoleVecEnv>()?;
    m.add_class::<PyMountainCarVecEnv>()?;
    m.add_class::<PyPendulumVecEnv>()?;

    #[cfg(feature = "cuda")]
    m.add_class::<PyCartPoleGpuEnv>()?;

    Ok(())
}
