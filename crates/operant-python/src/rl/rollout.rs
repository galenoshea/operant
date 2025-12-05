//! High-performance rollout buffer with GAE computation.

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use crate::buffer::{Buffer1D, Buffer2D};

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

    /// Add a batch of transitions (entire rollout at once).
    ///
    /// This is more efficient than calling add() repeatedly as it minimizes FFI overhead.
    ///
    /// # Arguments
    /// * `observations` - Shape (num_steps, num_envs, obs_dim)
    /// * `actions` - Shape (num_steps, num_envs) for discrete, (num_steps, num_envs, act_dim) for continuous
    /// * `rewards` - Shape (num_steps, num_envs)
    /// * `dones` - Shape (num_steps, num_envs)
    /// * `values` - Shape (num_steps, num_envs)
    /// * `log_probs` - Shape (num_steps, num_envs)
    pub fn add_batch<'py>(
        &mut self,
        observations: &Bound<'py, PyArray3<f32>>,
        actions: &Bound<'py, PyArray2<f32>>,
        rewards: &Bound<'py, PyArray2<f32>>,
        dones: &Bound<'py, PyArray2<f32>>,
        values: &Bound<'py, PyArray2<f32>>,
        log_probs: &Bound<'py, PyArray2<f32>>,
    ) -> PyResult<()> {
        // Validate shapes
        let obs_dims = observations.dims();
        if obs_dims[0] != self.num_steps || obs_dims[1] != self.num_envs || obs_dims[2] != self.obs_dim {
            return Err(PyValueError::new_err(format!(
                "Expected observations shape ({}, {}, {}), got ({}, {}, {})",
                self.num_steps, self.num_envs, self.obs_dim,
                obs_dims[0], obs_dims[1], obs_dims[2]
            )));
        }

        let act_dims = actions.dims();
        let expected_act_shape = if self.is_continuous {
            (self.num_steps, self.num_envs * self.act_dim)
        } else {
            (self.num_steps, self.num_envs)
        };
        if act_dims[0] != expected_act_shape.0 || act_dims[1] != expected_act_shape.1 {
            return Err(PyValueError::new_err(format!(
                "Expected actions shape {:?}, got ({}, {})",
                expected_act_shape, act_dims[0], act_dims[1]
            )));
        }

        // Copy all data at once (contiguous memory operations)
        unsafe {
            let obs_slice = observations.as_slice()?;
            self.observations.copy_from_slice(obs_slice);

            let act_slice = actions.as_slice()?;
            self.actions[..act_slice.len()].copy_from_slice(act_slice);

            let rew_slice = rewards.as_slice()?;
            self.rewards.copy_from_slice(rew_slice);

            let done_slice = dones.as_slice()?;
            self.dones.copy_from_slice(done_slice);

            let val_slice = values.as_slice()?;
            self.values.copy_from_slice(val_slice);

            let logp_slice = log_probs.as_slice()?;
            self.log_probs.copy_from_slice(logp_slice);
        }

        self.step = self.num_steps;
        self.full = true;

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

            #[cfg(feature = "simd")]
            {
                self.compute_gae_step_simd(step_idx, t, last_vals, &mut last_gae, gamma, gae_lambda);
            }

            #[cfg(not(feature = "simd"))]
            {
                self.compute_gae_step_scalar(step_idx, t, last_vals, &mut last_gae, gamma, gae_lambda);
            }
        }

        #[cfg(feature = "simd")]
        {
            self.compute_returns_simd();
        }

        #[cfg(not(feature = "simd"))]
        {
            for i in 0..self.advantages.len() {
                self.returns[i] = self.advantages[i] + self.values[i];
            }
        }

        Ok(())
    }

    /// Get flattened data arrays for training.
    ///
    /// Returns (observations, actions, log_probs, advantages, returns) as numpy arrays.
    pub fn get_all(
        &self,
    ) -> PyResult<(Buffer2D, Buffer1D, Buffer1D, Buffer1D, Buffer1D)> {
        if !self.full {
            return Err(PyValueError::new_err("Buffer not full"));
        }

        let total = self.num_steps * self.num_envs;

        let obs_buffer = Buffer2D::from_flat(
            self.observations.clone(),
            total,
            self.obs_dim,
        )?;
        let actions_buffer = Buffer1D::from_vec(self.actions.clone());
        let log_probs_buffer = Buffer1D::from_vec(self.log_probs.clone());
        let advantages_buffer = Buffer1D::from_vec(self.advantages.clone());
        let returns_buffer = Buffer1D::from_vec(self.returns.clone());

        Ok((
            obs_buffer,
            actions_buffer,
            log_probs_buffer,
            advantages_buffer,
            returns_buffer,
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

// Helper methods for GAE computation (outside #[pymethods] block)
impl RolloutBuffer {
    /// Scalar GAE computation for one step (fallback or non-SIMD builds).
    #[inline(always)]
    fn compute_gae_step_scalar(
        &mut self,
        step_idx: usize,
        t: usize,
        last_vals: &[f32],
        last_gae: &mut [f32],
        gamma: f32,
        gae_lambda: f32,
    ) {
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

    /// SIMD GAE computation for one step (processes 8 envs at a time).
    #[cfg(feature = "simd")]
    #[inline(always)]
    fn compute_gae_step_simd(
        &mut self,
        step_idx: usize,
        t: usize,
        last_vals: &[f32],
        last_gae: &mut [f32],
        gamma: f32,
        gae_lambda: f32,
    ) {
        use std::simd::{f32x8, num::SimdFloat};

        const LANES: usize = 8;
        let gamma_vec = f32x8::splat(gamma);
        let gae_lambda_vec = f32x8::splat(gae_lambda);
        let one_vec = f32x8::splat(1.0);

        // Process chunks of 8 environments
        for chunk in (0..self.num_envs).step_by(LANES) {
            if chunk + LANES <= self.num_envs {
                let idx = step_idx + chunk;

                let rewards_vec = f32x8::from_slice(&self.rewards[idx..]);
                let values_vec = f32x8::from_slice(&self.values[idx..]);
                let last_gae_vec = f32x8::from_slice(&last_gae[chunk..]);

                let (next_value, next_non_terminal) = if t == self.num_steps - 1 {
                    let next_val = f32x8::from_slice(&last_vals[chunk..]);
                    let dones = f32x8::from_slice(&self.dones[idx..]);
                    let non_term = one_vec - dones;
                    (next_val, non_term)
                } else {
                    let next_idx = (t + 1) * self.num_envs + chunk;
                    let next_val = f32x8::from_slice(&self.values[next_idx..]);
                    let dones = f32x8::from_slice(&self.dones[next_idx..]);
                    let non_term = one_vec - dones;
                    (next_val, non_term)
                };

                let delta = rewards_vec + gamma_vec * next_value * next_non_terminal - values_vec;

                let new_gae = delta + gamma_vec * gae_lambda_vec * next_non_terminal * last_gae_vec;

                new_gae.copy_to_slice(&mut last_gae[chunk..]);
                new_gae.copy_to_slice(&mut self.advantages[idx..]);
            } else {
                for e in chunk..self.num_envs {
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
        }
    }

    /// SIMD returns computation: returns = advantages + values
    #[cfg(feature = "simd")]
    #[inline(always)]
    fn compute_returns_simd(&mut self) {
        use std::simd::f32x8;

        const LANES: usize = 8;
        let total = self.advantages.len();

        for i in (0..total).step_by(LANES) {
            if i + LANES <= total {
                let adv = f32x8::from_slice(&self.advantages[i..]);
                let val = f32x8::from_slice(&self.values[i..]);
                let ret = adv + val;
                ret.copy_to_slice(&mut self.returns[i..]);
            } else {
                for j in i..total {
                    self.returns[j] = self.advantages[j] + self.values[j];
                }
            }
        }
    }
}
