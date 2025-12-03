//! Python bindings for Operant RL environments.
//!
//! Provides PyO3-based bindings with zero-copy numpy array support.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use operant_core::VecEnvironment;
use operant_envs::{CartPole, MountainCar, Pendulum};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[cfg(feature = "tui")]
mod tui;
#[cfg(feature = "tui")]
use tui::TUILogger;

mod rl;
use rl::RolloutBuffer;

/// High-performance vectorized CartPole environment.
///
/// Uses SIMD-optimized Rust implementation with zero-copy numpy arrays.
#[pyclass]
pub struct PyCartPoleVecEnv {
    inner: CartPole,
    num_envs: usize,
    obs_buffer: Vec<f32>,
    reward_buffer: Vec<f32>,
    terminal_buffer: Vec<u8>,
    truncation_buffer: Vec<u8>,
    action_buffer: Vec<f32>,
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
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be greater than 0"));
        }
        let workers = workers.unwrap_or(1);
        let inner = CartPole::with_workers(num_envs, workers);
        Ok(Self {
            inner,
            num_envs,
            obs_buffer: vec![0.0; num_envs * 4],
            reward_buffer: vec![0.0; num_envs],
            terminal_buffer: vec![0; num_envs],
            truncation_buffer: vec![0; num_envs],
            action_buffer: vec![0.0; num_envs],
        })
    }

    /// Reset all environments.
    ///
    /// Returns observations as a numpy array of shape (num_envs, 4).
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if observation array reshape fails.
    pub fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let seed = seed.unwrap_or(0);
        self.inner.reset(seed);
        self.inner.write_observations(&mut self.obs_buffer);

        // Reshape from flat (num_envs * 4) to 2D (num_envs, 4)
        let arr = PyArray1::from_slice(py, &self.obs_buffer);
        arr.reshape([self.num_envs, 4])
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape observations: {}", e)))
    }

    /// Step all environments with the given actions.
    ///
    /// # Arguments
    ///
    /// * `actions` - numpy array of shape (num_envs,) with dtype int32, values 0 or 1
    ///
    /// # Returns
    ///
    /// Tuple of (observations, rewards, terminals, truncations) where:
    /// - observations: shape (num_envs, 4), dtype float32
    /// - rewards: shape (num_envs,), dtype float32
    /// - terminals: shape (num_envs,), dtype uint8
    /// - truncations: shape (num_envs,), dtype uint8
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if actions array is not contiguous or has wrong size.
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: &Bound<'py, PyArray1<i32>>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u8>>,
    )> {
        // SAFETY: We verify the array is contiguous and correctly sized before use.
        // The array remains valid for the duration of this call.
        let actions_slice = unsafe {
            actions
                .as_slice()
                .map_err(|_| PyValueError::new_err("Actions must be a contiguous C-order array"))?
        };

        if actions_slice.len() != self.num_envs {
            return Err(PyValueError::new_err(format!(
                "Expected {} actions, got {}",
                self.num_envs,
                actions_slice.len()
            )));
        }

        // Reuse action buffer to eliminate allocation in hot loop
        self.action_buffer.clear();
        self.action_buffer
            .extend(actions_slice.iter().map(|&a| a as f32));

        self.inner.step_auto_reset(&self.action_buffer);

        self.inner.write_observations(&mut self.obs_buffer);
        self.inner.write_rewards(&mut self.reward_buffer);
        self.inner.write_terminals(&mut self.terminal_buffer);
        self.inner.write_truncations(&mut self.truncation_buffer);

        // Reshape observations from flat to 2D
        let obs_arr = PyArray1::from_slice(py, &self.obs_buffer);
        let obs_2d = obs_arr
            .reshape([self.num_envs, 4])
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape observations: {}", e)))?;

        Ok((
            obs_2d,
            PyArray1::from_slice(py, &self.reward_buffer),
            PyArray1::from_slice(py, &self.terminal_buffer),
            PyArray1::from_slice(py, &self.truncation_buffer),
        ))
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
    obs_buffer: Vec<f32>,
    reward_buffer: Vec<f32>,
    terminal_buffer: Vec<u8>,
    truncation_buffer: Vec<u8>,
    action_buffer: Vec<f32>,
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
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be greater than 0"));
        }
        let workers = workers.unwrap_or(1);
        let inner = MountainCar::with_workers(num_envs, workers);
        Ok(Self {
            inner,
            num_envs,
            obs_buffer: vec![0.0; num_envs * 2],
            reward_buffer: vec![0.0; num_envs],
            terminal_buffer: vec![0; num_envs],
            truncation_buffer: vec![0; num_envs],
            action_buffer: vec![0.0; num_envs],
        })
    }

    /// Reset all environments.
    ///
    /// Returns observations as a numpy array of shape (num_envs, 2).
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if observation array reshape fails.
    pub fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let seed = seed.unwrap_or(0);
        self.inner.reset(seed);
        self.inner.write_observations(&mut self.obs_buffer);

        // Reshape from flat (num_envs * 2) to 2D (num_envs, 2)
        let arr = PyArray1::from_slice(py, &self.obs_buffer);
        arr.reshape([self.num_envs, 2])
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape observations: {}", e)))
    }

    /// Step all environments with the given actions.
    ///
    /// # Arguments
    ///
    /// * `actions` - numpy array of shape (num_envs,) with dtype int32, values 0, 1, or 2
    ///
    /// # Returns
    ///
    /// Tuple of (observations, rewards, terminals, truncations) where:
    /// - observations: shape (num_envs, 2), dtype float32
    /// - rewards: shape (num_envs,), dtype float32
    /// - terminals: shape (num_envs,), dtype uint8
    /// - truncations: shape (num_envs,), dtype uint8
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if actions array is not contiguous or has wrong size.
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: &Bound<'py, PyArray1<i32>>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u8>>,
    )> {
        // SAFETY: We verify the array is contiguous and correctly sized before use.
        // The array remains valid for the duration of this call.
        let actions_slice = unsafe {
            actions
                .as_slice()
                .map_err(|_| PyValueError::new_err("Actions must be a contiguous C-order array"))?
        };

        if actions_slice.len() != self.num_envs {
            return Err(PyValueError::new_err(format!(
                "Expected {} actions, got {}",
                self.num_envs,
                actions_slice.len()
            )));
        }

        // Reuse action buffer to eliminate allocation in hot loop
        self.action_buffer.clear();
        self.action_buffer
            .extend(actions_slice.iter().map(|&a| a as f32));

        self.inner.step_auto_reset(&self.action_buffer);

        self.inner.write_observations(&mut self.obs_buffer);
        self.inner.write_rewards(&mut self.reward_buffer);
        self.inner.write_terminals(&mut self.terminal_buffer);
        self.inner.write_truncations(&mut self.truncation_buffer);

        // Reshape observations from flat to 2D
        let obs_arr = PyArray1::from_slice(py, &self.obs_buffer);
        let obs_2d = obs_arr
            .reshape([self.num_envs, 2])
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape observations: {}", e)))?;

        Ok((
            obs_2d,
            PyArray1::from_slice(py, &self.reward_buffer),
            PyArray1::from_slice(py, &self.terminal_buffer),
            PyArray1::from_slice(py, &self.truncation_buffer),
        ))
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
    obs_buffer: Vec<f32>,
    reward_buffer: Vec<f32>,
    terminal_buffer: Vec<u8>,
    truncation_buffer: Vec<u8>,
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
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be greater than 0"));
        }
        let workers = workers.unwrap_or(1);
        let inner = Pendulum::with_workers(num_envs, workers);
        Ok(Self {
            inner,
            num_envs,
            obs_buffer: vec![0.0; num_envs * 3],
            reward_buffer: vec![0.0; num_envs],
            terminal_buffer: vec![0; num_envs],
            truncation_buffer: vec![0; num_envs],
        })
    }

    /// Reset all environments.
    ///
    /// Returns observations as a numpy array of shape (num_envs, 3).
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if observation array reshape fails.
    pub fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let seed = seed.unwrap_or(0);
        self.inner.reset(seed);
        self.inner.write_observations(&mut self.obs_buffer);

        // Reshape from flat (num_envs * 3) to 2D (num_envs, 3)
        let arr = PyArray1::from_slice(py, &self.obs_buffer);
        arr.reshape([self.num_envs, 3])
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape observations: {}", e)))
    }

    /// Step all environments with the given actions.
    ///
    /// # Arguments
    ///
    /// * `actions` - numpy array of shape (num_envs,) with dtype float32, continuous torque values [-2.0, 2.0]
    ///
    /// # Returns
    ///
    /// Tuple of (observations, rewards, terminals, truncations) where:
    /// - observations: shape (num_envs, 3), dtype float32 (cos(theta), sin(theta), theta_dot)
    /// - rewards: shape (num_envs,), dtype float32
    /// - terminals: shape (num_envs,), dtype uint8 (always 0, Pendulum has no terminal state)
    /// - truncations: shape (num_envs,), dtype uint8
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if actions array is not contiguous or has wrong size.
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u8>>,
    )> {
        // SAFETY: We verify the array is contiguous and correctly sized before use.
        // The array remains valid for the duration of this call.
        let actions_slice = unsafe {
            actions
                .as_slice()
                .map_err(|_| PyValueError::new_err("Actions must be a contiguous C-order array"))?
        };

        if actions_slice.len() != self.num_envs {
            return Err(PyValueError::new_err(format!(
                "Expected {} actions, got {}",
                self.num_envs,
                actions_slice.len()
            )));
        }

        self.inner.step_auto_reset(actions_slice);

        self.inner.write_observations(&mut self.obs_buffer);
        self.inner.write_rewards(&mut self.reward_buffer);
        self.inner.write_terminals(&mut self.terminal_buffer);
        self.inner.write_truncations(&mut self.truncation_buffer);

        // Reshape observations from flat to 2D
        let obs_arr = PyArray1::from_slice(py, &self.obs_buffer);
        let obs_2d = obs_arr
            .reshape([self.num_envs, 3])
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape observations: {}", e)))?;

        Ok((
            obs_2d,
            PyArray1::from_slice(py, &self.reward_buffer),
            PyArray1::from_slice(py, &self.terminal_buffer),
            PyArray1::from_slice(py, &self.truncation_buffer),
        ))
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

/// Register operant.envs submodule with environment classes.
fn register_envs_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let envs_mod = PyModule::new(py, "envs")?;

    // Add environment classes to envs submodule
    envs_mod.add_class::<PyCartPoleVecEnv>()?;
    envs_mod.add_class::<PyMountainCarVecEnv>()?;
    envs_mod.add_class::<PyPendulumVecEnv>()?;

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

/// Register RL utilities module with RolloutBuffer.
fn register_rl_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let rl_mod = PyModule::new(py, "_rl")?;

    // Add RL classes to _rl submodule
    rl_mod.add_class::<RolloutBuffer>()?;

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

    Ok(())
}
