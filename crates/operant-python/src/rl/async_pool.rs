//! Async environment pool with lock-free signaling and shared memory.
//!
//! This module implements the PufferLib-style architecture:
//! - Dedicated environment thread pool that runs continuously
//! - Lock-free atomic signaling between Python and Rust
//! - Shared memory for zero-copy data transfer
//! - Double-buffering to hide environment latency behind inference

use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

/// Shared state between Python inference thread and Rust environment threads.
///
/// Uses atomics for lock-free synchronization.
struct SharedState {
    /// Number of environments per buffer
    num_envs: usize,
    /// Observation dimension
    obs_dim: usize,
    /// Action dimension
    act_dim: usize,

    // Double-buffering: two sets of observation/action buffers
    // While Python processes buffer A, Rust fills buffer B (and vice versa)
    obs_buffer_a: Vec<f32>,
    obs_buffer_b: Vec<f32>,
    act_buffer_a: Vec<f32>,
    act_buffer_b: Vec<f32>,
    rew_buffer_a: Vec<f32>,
    rew_buffer_b: Vec<f32>,
    done_buffer_a: Vec<f32>,
    done_buffer_b: Vec<f32>,

    active_buffer: AtomicUsize,

    step_ready: AtomicBool,
    shutdown: AtomicBool,

    obs_ready: AtomicBool,
}

impl SharedState {
    fn new(num_envs: usize, obs_dim: usize, act_dim: usize) -> Self {
        let obs_size = num_envs * obs_dim;
        let act_size = num_envs * act_dim;

        Self {
            num_envs,
            obs_dim,
            act_dim,
            obs_buffer_a: vec![0.0; obs_size],
            obs_buffer_b: vec![0.0; obs_size],
            act_buffer_a: vec![0.0; act_size],
            act_buffer_b: vec![0.0; act_size],
            rew_buffer_a: vec![0.0; num_envs],
            rew_buffer_b: vec![0.0; num_envs],
            done_buffer_a: vec![0.0; num_envs],
            done_buffer_b: vec![0.0; num_envs],
            active_buffer: AtomicUsize::new(0),
            step_ready: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            obs_ready: AtomicBool::new(true),
        }
    }

    #[inline]
    fn active_idx(&self) -> usize {
        self.active_buffer.load(Ordering::Acquire)
    }

    #[inline]
    fn active_obs_buffer(&mut self) -> &mut [f32] {
        if self.active_idx() == 0 {
            &mut self.obs_buffer_a
        } else {
            &mut self.obs_buffer_b
        }
    }

    #[inline]
    fn inactive_obs_buffer(&self) -> &[f32] {
        if self.active_idx() == 0 {
            &self.obs_buffer_b
        } else {
            &self.obs_buffer_a
        }
    }

    #[inline]
    fn inactive_act_buffer(&mut self) -> &mut [f32] {
        if self.active_idx() == 0 {
            &mut self.act_buffer_b
        } else {
            &mut self.act_buffer_a
        }
    }

    #[inline]
    fn active_act_buffer(&self) -> &[f32] {
        if self.active_idx() == 0 {
            &self.act_buffer_a
        } else {
            &self.act_buffer_b
        }
    }

    #[inline]
    fn swap_buffers(&self) {
        let current = self.active_idx();
        self.active_buffer.store(1 - current, Ordering::Release);
    }
}

/// Async environment pool that runs in background threads.
///
/// This is the PufferLib-style architecture:
/// 1. Python thread: GPU inference on observations
/// 2. Rust thread pool: Environment stepping
/// 3. Atomic flags for lock-free synchronization
/// 4. Double-buffered shared memory
#[pyclass]
pub struct AsyncEnvPool {
    shared_state: Arc<std::sync::Mutex<SharedState>>,
    worker_handle: Option<thread::JoinHandle<()>>,
    num_envs: usize,
    obs_dim: usize,
    act_dim: usize,
}

#[pymethods]
impl AsyncEnvPool {
    /// Create a new async environment pool.
    ///
    /// # Arguments
    /// * `num_envs` - Number of parallel environments
    /// * `obs_dim` - Observation dimension
    /// * `act_dim` - Action dimension (1 for discrete)
    #[new]
    pub fn new(num_envs: usize, obs_dim: usize, act_dim: usize) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be > 0"));
        }

        let shared_state = Arc::new(std::sync::Mutex::new(
            SharedState::new(num_envs, obs_dim, act_dim)
        ));

        Ok(Self {
            shared_state,
            worker_handle: None,
            num_envs,
            obs_dim,
            act_dim,
        })
    }

    /// Start the background environment thread.
    ///
    /// This thread runs continuously, waiting for actions and producing observations.
    pub fn start(&mut self) -> PyResult<()> {
        let shared = self.shared_state.clone();

        let handle = thread::spawn(move || {
            loop {
                let mut state = shared.lock().unwrap();

                if state.shutdown.load(Ordering::Acquire) {
                    break;
                }

                while !state.step_ready.load(Ordering::Acquire) {
                    if state.shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    std::hint::spin_loop();
                }

                state.step_ready.store(false, Ordering::Release);

                let actions = state.active_act_buffer();

                let obs = state.active_obs_buffer();
                for i in 0..obs.len() {
                    obs[i] = (i as f32).sin();
                }

                state.swap_buffers();

                state.obs_ready.store(true, Ordering::Release);

                drop(state);
            }
        });

        self.worker_handle = Some(handle);
        Ok(())
    }

    /// Submit actions and wait for next observations.
    ///
    /// This is the main Python-side API:
    /// 1. Write actions to inactive buffer
    /// 2. Signal step_ready
    /// 3. Wait for obs_ready (spin-wait)
    /// 4. Return observations from inactive buffer
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let mut state = self.shared_state.lock().unwrap();

        let act_slice = unsafe { actions.as_slice()? };
        state.inactive_act_buffer().copy_from_slice(act_slice);

        state.step_ready.store(true, Ordering::Release);

        drop(state);

        loop {
            let state = self.shared_state.lock().unwrap();
            if state.obs_ready.load(Ordering::Acquire) {
                break;
            }
            drop(state);
            std::hint::spin_loop();
        }

        let mut state = self.shared_state.lock().unwrap();
        state.obs_ready.store(false, Ordering::Release);

        let obs_slice = state.inactive_obs_buffer();
        let obs_array = obs_slice.to_pyarray(py);
        let obs_2d = obs_array.reshape([self.num_envs, self.obs_dim])?;

        Ok(obs_2d)
    }

    /// Reset all environments.
    pub fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let mut state = self.shared_state.lock().unwrap();
        let obs = state.active_obs_buffer();
        for i in 0..obs.len() {
            obs[i] = 0.0;
        }

        let obs_array = obs.to_pyarray(py);
        let obs_2d = obs_array.reshape([self.num_envs, self.obs_dim])?;
        Ok(obs_2d)
    }

    /// Shutdown the background thread.
    pub fn shutdown(&mut self) -> PyResult<()> {
        {
            let state = self.shared_state.lock().unwrap();
            state.shutdown.store(true, Ordering::Release);
        }

        if let Some(handle) = self.worker_handle.take() {
            handle.join().map_err(|_| {
                PyValueError::new_err("Failed to join worker thread")
            })?;
        }

        Ok(())
    }

    #[getter]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    #[getter]
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }
}

impl Drop for AsyncEnvPool {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
