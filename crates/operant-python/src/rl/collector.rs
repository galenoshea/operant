//! High-performance batched rollout collection.
//!
//! This module provides infrastructure for collecting rollouts with minimal
//! Python↔Rust boundary crossings by batching operations.

use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;

/// Batched rollout collector that minimizes Python↔Rust boundary crossings.
///
/// Instead of calling Python 128 times per rollout (once per step), this
/// collector batches the entire rollout collection and only crosses the
/// boundary once per rollout.
///
/// The workflow is:
/// 1. Python calls `start_rollout()` - returns initial observations
/// 2. Python does batched GPU inference on observations
/// 3. Python calls `step_batch()` with actions - returns next observations
/// 4. Repeat steps 2-3 for n_steps
/// 5. Python calls `finish_rollout()` - computes GAE and returns training data
#[pyclass]
pub struct BatchedRolloutBuffer {
    num_envs: usize,
    num_steps: usize,
    obs_dim: usize,

    // Pre-allocated buffers for ALL steps [num_steps * num_envs * dim]
    observations: Vec<f32>,
    actions: Vec<f32>,
    rewards: Vec<f32>,
    dones: Vec<f32>,
    values: Vec<f32>,
    log_probs: Vec<f32>,

    // GAE results
    advantages: Vec<f32>,
    returns: Vec<f32>,

    // Tracking
    current_step: usize,

    // Current observation buffer (for returning to Python)
    current_obs: Vec<f32>,
}

#[pymethods]
impl BatchedRolloutBuffer {
    #[new]
    #[pyo3(signature = (num_envs, num_steps, obs_dim))]
    pub fn new(num_envs: usize, num_steps: usize, obs_dim: usize) -> Self {
        let total = num_steps * num_envs;
        let obs_size = total * obs_dim;

        Self {
            num_envs,
            num_steps,
            obs_dim,
            observations: vec![0.0; obs_size],
            actions: vec![0.0; total],
            rewards: vec![0.0; total],
            dones: vec![0.0; total],
            values: vec![0.0; total],
            log_probs: vec![0.0; total],
            advantages: vec![0.0; total],
            returns: vec![0.0; total],
            current_step: 0,
            current_obs: vec![0.0; num_envs * obs_dim],
        }
    }

    /// Reset for new rollout collection.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    /// Store a single step's worth of data from Python.
    /// This is the minimal-overhead version - just copies data.
    pub fn store_step<'py>(
        &mut self,
        observations: &Bound<'py, PyArray2<f32>>,
        actions: &Bound<'py, PyArray1<f32>>,
        rewards: &Bound<'py, PyArray1<f32>>,
        dones: &Bound<'py, PyArray1<f32>>,
        values: &Bound<'py, PyArray1<f32>>,
        log_probs: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<()> {
        if self.current_step >= self.num_steps {
            return Err(pyo3::exceptions::PyValueError::new_err("Buffer full"));
        }

        let step = self.current_step;
        let env_offset = step * self.num_envs;
        let obs_offset = step * self.num_envs * self.obs_dim;

        unsafe {
            let obs = observations.as_slice()?;
            self.observations[obs_offset..obs_offset + obs.len()].copy_from_slice(obs);

            let act = actions.as_slice()?;
            self.actions[env_offset..env_offset + act.len()].copy_from_slice(act);

            let rew = rewards.as_slice()?;
            self.rewards[env_offset..env_offset + rew.len()].copy_from_slice(rew);

            let done = dones.as_slice()?;
            self.dones[env_offset..env_offset + done.len()].copy_from_slice(done);

            let val = values.as_slice()?;
            self.values[env_offset..env_offset + val.len()].copy_from_slice(val);

            let logp = log_probs.as_slice()?;
            self.log_probs[env_offset..env_offset + logp.len()].copy_from_slice(logp);
        }

        self.current_step += 1;
        Ok(())
    }

    /// Compute GAE advantages in optimized Rust.
    pub fn compute_gae<'py>(
        &mut self,
        last_values: &Bound<'py, PyArray1<f32>>,
        gamma: f32,
        gae_lambda: f32,
    ) -> PyResult<()> {
        let last_vals = unsafe { last_values.as_slice()? };

        let mut last_gae = vec![0.0f32; self.num_envs];

        for t in (0..self.num_steps).rev() {
            let step_idx = t * self.num_envs;

            for e in 0..self.num_envs {
                let idx = step_idx + e;

                let (next_value, next_non_terminal) = if t == self.num_steps - 1 {
                    (last_vals[e], 1.0 - self.dones[idx])
                } else {
                    let next_idx = (t + 1) * self.num_envs + e;
                    (self.values[next_idx], 1.0 - self.dones[next_idx])
                };

                let delta = self.rewards[idx] + gamma * next_value * next_non_terminal - self.values[idx];
                last_gae[e] = delta + gamma * gae_lambda * next_non_terminal * last_gae[e];
                self.advantages[idx] = last_gae[e];
            }
        }

        for i in 0..self.advantages.len() {
            self.returns[i] = self.advantages[i] + self.values[i];
        }

        Ok(())
    }

    /// Get all data for training - returns views to avoid copies.
    pub fn get_training_data<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,  // observations
        Bound<'py, PyArray1<f32>>,  // actions
        Bound<'py, PyArray1<f32>>,  // log_probs
        Bound<'py, PyArray1<f32>>,  // advantages
        Bound<'py, PyArray1<f32>>,  // returns
    )> {
        let total = self.num_steps * self.num_envs;

        let obs = self.observations.to_pyarray(py);
        let obs_2d = obs.reshape([total, self.obs_dim])?;

        Ok((
            obs_2d,
            self.actions.to_pyarray(py),
            self.log_probs.to_pyarray(py),
            self.advantages.to_pyarray(py),
            self.returns.to_pyarray(py),
        ))
    }

    #[getter]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    #[getter]
    pub fn num_steps(&self) -> usize {
        self.num_steps
    }

    #[getter]
    pub fn total_samples(&self) -> usize {
        self.num_steps * self.num_envs
    }
}

/// High-performance environment stepper that batches multiple steps.
///
/// Key optimization: Instead of calling Python once per step, this
/// steps the environment N times and returns all observations at once.
#[pyclass]
pub struct FastEnvStepper {
    num_envs: usize,
    obs_dim: usize,

    // Buffers for a single step (reused)
    obs_buffer: Vec<f32>,
    reward_buffer: Vec<f32>,
    done_buffer: Vec<f32>,
}

#[pymethods]
impl FastEnvStepper {
    #[new]
    pub fn new(num_envs: usize, obs_dim: usize) -> Self {
        Self {
            num_envs,
            obs_dim,
            obs_buffer: vec![0.0; num_envs * obs_dim],
            reward_buffer: vec![0.0; num_envs],
            done_buffer: vec![0.0; num_envs],
        }
    }
}
