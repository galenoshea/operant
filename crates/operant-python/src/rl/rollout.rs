//! High-performance rollout buffer with GAE computation.

use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// High-performance rollout buffer for PPO-style algorithms.
///
/// Stores transitions in a Structure-of-Arrays (SoA) layout for cache efficiency.
/// Computes Generalized Advantage Estimation (GAE) in optimized Rust.
#[pyclass]
pub struct RolloutBuffer {
    num_envs: usize,
    num_steps: usize,
    obs_dim: usize,
    act_dim: usize,
    is_continuous: bool,

    // Preallocated storage [num_steps * num_envs * dim]
    observations: Vec<f32>,
    actions: Vec<f32>,
    rewards: Vec<f32>,
    dones: Vec<f32>,
    values: Vec<f32>,
    log_probs: Vec<f32>,

    // Computed after rollout collection
    advantages: Vec<f32>,
    returns: Vec<f32>,

    // Current insertion position
    step: usize,
    full: bool,
}

#[pymethods]
impl RolloutBuffer {
    /// Create a new rollout buffer.
    ///
    /// # Arguments
    /// * `num_envs` - Number of parallel environments
    /// * `num_steps` - Number of steps per rollout
    /// * `obs_dim` - Observation dimension
    /// * `act_dim` - Action dimension (1 for discrete, action_size for continuous)
    /// * `is_continuous` - Whether action space is continuous
    #[new]
    #[pyo3(signature = (num_envs, num_steps, obs_dim, act_dim=1, is_continuous=false))]
    pub fn new(
        num_envs: usize,
        num_steps: usize,
        obs_dim: usize,
        act_dim: usize,
        is_continuous: bool,
    ) -> PyResult<Self> {
        if num_envs == 0 || num_steps == 0 || obs_dim == 0 {
            return Err(PyValueError::new_err("Dimensions must be > 0"));
        }

        let total_steps = num_steps * num_envs;
        let obs_size = total_steps * obs_dim;
        let act_size = if is_continuous {
            total_steps * act_dim
        } else {
            total_steps
        };

        Ok(Self {
            num_envs,
            num_steps,
            obs_dim,
            act_dim,
            is_continuous,
            observations: vec![0.0; obs_size],
            actions: vec![0.0; act_size],
            rewards: vec![0.0; total_steps],
            dones: vec![0.0; total_steps],
            values: vec![0.0; total_steps],
            log_probs: vec![0.0; total_steps],
            advantages: vec![0.0; total_steps],
            returns: vec![0.0; total_steps],
            step: 0,
            full: false,
        })
    }

    /// Add a single step of transitions for all environments.
    ///
    /// # Arguments
    /// * `observations` - Shape (num_envs, obs_dim)
    /// * `actions` - Shape (num_envs,) for discrete, (num_envs, act_dim) for continuous
    /// * `rewards` - Shape (num_envs,)
    /// * `dones` - Shape (num_envs,)
    /// * `values` - Shape (num_envs,)
    /// * `log_probs` - Shape (num_envs,)
    pub fn add<'py>(
        &mut self,
        observations: &Bound<'py, PyArray2<f32>>,
        actions: &Bound<'py, PyArray1<f32>>,
        rewards: &Bound<'py, PyArray1<f32>>,
        dones: &Bound<'py, PyArray1<f32>>,
        values: &Bound<'py, PyArray1<f32>>,
        log_probs: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<()> {
        if self.step >= self.num_steps {
            return Err(PyValueError::new_err("Buffer is full, call reset() first"));
        }

        // Copy observations
        let obs_slice = unsafe { observations.as_slice()? };
        let obs_start = self.step * self.num_envs * self.obs_dim;
        self.observations[obs_start..obs_start + obs_slice.len()].copy_from_slice(obs_slice);

        // Copy actions
        let act_slice = unsafe { actions.as_slice()? };
        let act_start = if self.is_continuous {
            self.step * self.num_envs * self.act_dim
        } else {
            self.step * self.num_envs
        };
        self.actions[act_start..act_start + act_slice.len()].copy_from_slice(act_slice);

        // Copy scalars
        let step_start = self.step * self.num_envs;
        let rew_slice = unsafe { rewards.as_slice()? };
        let done_slice = unsafe { dones.as_slice()? };
        let val_slice = unsafe { values.as_slice()? };
        let logp_slice = unsafe { log_probs.as_slice()? };

        self.rewards[step_start..step_start + self.num_envs].copy_from_slice(rew_slice);
        self.dones[step_start..step_start + self.num_envs].copy_from_slice(done_slice);
        self.values[step_start..step_start + self.num_envs].copy_from_slice(val_slice);
        self.log_probs[step_start..step_start + self.num_envs].copy_from_slice(logp_slice);

        self.step += 1;
        if self.step >= self.num_steps {
            self.full = true;
        }

        Ok(())
    }

    /// Compute GAE advantages and returns.
    ///
    /// # Arguments
    /// * `last_values` - Value estimates for final observations, shape (num_envs,)
    /// * `gamma` - Discount factor
    /// * `gae_lambda` - GAE lambda parameter
    pub fn compute_gae<'py>(
        &mut self,
        last_values: &Bound<'py, PyArray1<f32>>,
        gamma: f32,
        gae_lambda: f32,
    ) -> PyResult<()> {
        let last_vals = unsafe { last_values.as_slice()? };

        // Initialize last advantage per environment
        let mut last_gae = vec![0.0f32; self.num_envs];

        // Iterate backwards through steps
        for t in (0..self.num_steps).rev() {
            let step_idx = t * self.num_envs;

            for e in 0..self.num_envs {
                let idx = step_idx + e;

                // Get next value and non-terminal mask
                let (next_value, next_non_terminal) = if t == self.num_steps - 1 {
                    (last_vals[e], 1.0 - self.dones[idx])
                } else {
                    let next_idx = (t + 1) * self.num_envs + e;
                    (self.values[next_idx], 1.0 - self.dones[next_idx])
                };

                // TD error
                let delta =
                    self.rewards[idx] + gamma * next_value * next_non_terminal - self.values[idx];

                // GAE
                last_gae[e] = delta + gamma * gae_lambda * next_non_terminal * last_gae[e];
                self.advantages[idx] = last_gae[e];
            }
        }

        // Compute returns = advantages + values
        for i in 0..self.advantages.len() {
            self.returns[i] = self.advantages[i] + self.values[i];
        }

        Ok(())
    }

    /// Get flattened data arrays for training.
    ///
    /// Returns (observations, actions, log_probs, advantages, returns) as numpy arrays.
    pub fn get_all<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        if !self.full {
            return Err(PyValueError::new_err("Buffer not full"));
        }

        let total = self.num_steps * self.num_envs;

        // Create numpy arrays from Rust vectors
        let obs_array = self.observations.to_pyarray(py);
        let obs_2d = obs_array.reshape([total, self.obs_dim])?;

        Ok((
            obs_2d,
            self.actions.to_pyarray(py),
            self.log_probs.to_pyarray(py),
            self.advantages.to_pyarray(py),
            self.returns.to_pyarray(py),
        ))
    }

    /// Reset the buffer for new rollout collection.
    pub fn reset(&mut self) {
        self.step = 0;
        self.full = false;
    }

    /// Check if buffer is full.
    #[getter]
    pub fn is_full(&self) -> bool {
        self.full
    }

    /// Get current step count.
    #[getter]
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Get total samples in buffer when full.
    #[getter]
    pub fn total_samples(&self) -> usize {
        self.num_steps * self.num_envs
    }

    /// Get number of environments.
    #[getter]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Get number of steps per rollout.
    #[getter]
    pub fn num_steps(&self) -> usize {
        self.num_steps
    }

    /// Get observation dimension.
    #[getter]
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }
}
