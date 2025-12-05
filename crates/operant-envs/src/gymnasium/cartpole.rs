//! SIMD-optimized CartPole environment with Struct-of-Arrays memory layout.
//!
//! This module provides a high-performance vectorized CartPole environment
//! that uses SIMD instructions to process multiple environments in parallel.
//!
//! Key optimizations:
//! - Struct-of-Arrays (SoA) memory layout for cache efficiency
//! - SIMD physics using f32x8 (AVX2) or f32x4 (SSE2/NEON)
//! - Branchless termination checks
//! - Optimized auto-reset with mask-based operations

const GRAVITY: f32 = 9.8;
const CART_MASS: f32 = 1.0;
const POLE_MASS: f32 = 0.1;
const POLE_LENGTH: f32 = 0.5;
const FORCE_MAG: f32 = 10.0;
const DT: f32 = 0.02;
const X_THRESHOLD: f32 = 2.4;
const THETA_THRESHOLD: f32 = 12.0 * std::f32::consts::PI / 180.0;
const MAX_STEPS: u32 = 500;

use operant_core::LogData;
use crate::shared::rng::*;
use rand::SeedableRng;

#[cfg(feature = "simd")]
use std::simd::{f32x8, cmp::SimdPartialOrd, num::SimdFloat};
#[cfg(feature = "simd")]
use crate::shared::simd::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use crate::shared::parallel::SyncPtr;

/// Log data for CartPole metrics tracking.
#[derive(Clone, Debug, Default)]
pub struct CartPoleLog {
    /// Total reward accumulated across completed episodes.
    pub total_reward: f32,
    /// Number of completed episodes.
    pub episode_count: u32,
    /// Total steps across completed episodes.
    pub total_steps: u32,
}

impl LogData for CartPoleLog {
    fn merge(&mut self, other: &Self) {
        self.total_reward += other.total_reward;
        self.episode_count += other.episode_count;
        self.total_steps += other.total_steps;
    }

    fn clear(&mut self) {
        self.total_reward = 0.0;
        self.episode_count = 0;
        self.total_steps = 0;
    }

    fn episode_count(&self) -> f32 {
        self.episode_count as f32
    }
}

/// SIMD-optimized CartPole with Struct-of-Arrays memory layout.
///
/// All environment states are stored in contiguous arrays for optimal
/// cache performance and SIMD vectorization.
pub struct CartPole {
    x: Vec<f32>,
    x_dot: Vec<f32>,
    theta: Vec<f32>,
    theta_dot: Vec<f32>,
    rewards: Vec<f32>,
    terminals: Vec<u8>,
    truncations: Vec<u8>,
    ticks: Vec<u32>,
    episode_rewards: Vec<f32>,
    num_envs: usize,
    max_steps: u32,
    init_range: f32,
    base_seed: u64,
    rng_seeds: Vec<u64>,
    log: CartPoleLog,
    /// Number of worker threads (1 = single-threaded, >1 = parallel).
    workers: usize,
}

impl CartPole {
    /// Create a new SIMD-optimized CartPole vectorized environment.
    ///
    /// # Arguments
    ///
    /// * `num_envs` - Number of parallel environment instances
    /// * `max_steps` - Maximum episode length before truncation
    /// * `init_range` - Range for random initial state values
    /// * `workers` - Number of worker threads (1 = single-threaded, >1 = parallel)
    pub fn new(num_envs: usize, max_steps: u32, init_range: f32, workers: usize) -> operant_core::Result<Self> {
        if num_envs == 0 {
            return Err(operant_core::OperantError::InvalidConfig {
                param: "num_envs".to_string(),
                message: "must be at least 1".to_string(),
            });
        }
        let workers = workers.max(1);

        Ok(Self {
            x: vec![0.0; num_envs],
            x_dot: vec![0.0; num_envs],
            theta: vec![0.0; num_envs],
            theta_dot: vec![0.0; num_envs],
            rewards: vec![0.0; num_envs],
            terminals: vec![0; num_envs],
            truncations: vec![0; num_envs],
            ticks: vec![0; num_envs],
            episode_rewards: vec![0.0; num_envs],
            num_envs,
            max_steps,
            init_range,
            base_seed: 0,
            rng_seeds: (0..num_envs as u64).collect(),
            log: CartPoleLog::default(),
            workers,
        })
    }

    /// Create with default CartPole configuration (single-threaded).
    pub fn with_defaults(num_envs: usize) -> operant_core::Result<Self> {
        Self::new(num_envs, MAX_STEPS, 0.05, 1)
    }

    /// Create with default CartPole configuration and specified worker count.
    pub fn with_workers(num_envs: usize, workers: usize) -> operant_core::Result<Self> {
        Self::new(num_envs, MAX_STEPS, 0.05, workers)
    }

    /// Get the number of workers.
    #[inline]
    pub fn workers(&self) -> usize {
        self.workers
    }

    /// Get the number of environments.
    #[inline]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Get the observation size per environment.
    #[inline]
    pub fn observation_size(&self) -> usize {
        4
    }

    /// Reset all environments with deterministic seeding.
    pub fn reset(&mut self, base_seed: u64) {
        self.base_seed = base_seed;

        for i in 0..self.num_envs {
            self.reset_single(i, base_seed + i as u64);
            // For initial reset, also clear the step outputs
            self.rewards[i] = 0.0;
            self.terminals[i] = 0;
            self.truncations[i] = 0;
        }
    }

    /// Reset a single environment.
    ///
    /// NOTE: This intentionally does NOT reset rewards/terminals/truncations
    /// because those contain the result of the step that triggered the reset.
    /// Those values should be read by the agent before the next step clears them.
    #[inline]
    fn reset_single(&mut self, idx: usize, seed: u64) {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        self.rng_seeds[idx] = seed;

        self.x[idx] = random_uniform(&mut rng, -self.init_range, self.init_range);
        self.x_dot[idx] = random_uniform(&mut rng, -self.init_range, self.init_range);
        self.theta[idx] = random_uniform(&mut rng, -self.init_range, self.init_range);
        self.theta_dot[idx] = random_uniform(&mut rng, -self.init_range, self.init_range);

        self.ticks[idx] = 0;
        self.episode_rewards[idx] = 0.0;
    }

    /// Step all environments (scalar implementation).
    ///
    /// This is the baseline implementation used when SIMD is not available
    /// or for environments that don't align to SIMD lane count.
    pub fn step_scalar(&mut self, actions: &[f32]) {
        debug_assert_eq!(actions.len(), self.num_envs);

        for i in 0..self.num_envs {
            self.step_single_env(i, actions[i]);
        }
    }

    #[inline(always)]
    fn step_single_env(&mut self, idx: usize, action: f32) {
        let force = if action as i32 == 1 { FORCE_MAG } else { -FORCE_MAG };

        let x = self.x[idx];
        let x_dot = self.x_dot[idx];
        let theta = self.theta[idx];
        let theta_dot = self.theta_dot[idx];

        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let total_mass = CART_MASS + POLE_MASS;
        let pole_mass_length = POLE_MASS * POLE_LENGTH;

        let temp = (force + pole_mass_length * theta_dot * theta_dot * sin_theta) / total_mass;

        // Prevent division by zero when cos_theta approximation error causes denom ≤ 0
        let denom = POLE_LENGTH * (4.0 / 3.0 - POLE_MASS * cos_theta * cos_theta / total_mass);
        let safe_denom = denom.max(1e-6);
        let theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / safe_denom;

        let x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass;

        let new_x = x + DT * x_dot;
        let new_x_dot = x_dot + DT * x_acc;
        let new_theta = theta + DT * theta_dot;
        let new_theta_dot = theta_dot + DT * theta_acc;

        self.x[idx] = new_x;
        self.x_dot[idx] = new_x_dot;
        self.theta[idx] = new_theta;
        self.theta_dot[idx] = new_theta_dot;
        self.ticks[idx] += 1;

        let terminal = (new_x.abs() > X_THRESHOLD) | (new_theta.abs() > THETA_THRESHOLD);
        let truncated = self.ticks[idx] >= self.max_steps;

        self.terminals[idx] = terminal as u8;
        self.truncations[idx] = truncated as u8;

        let reward = if terminal { 0.0 } else { 1.0 };
        self.rewards[idx] = reward;
        self.episode_rewards[idx] += reward;

        if terminal || truncated {
            self.log.total_reward += self.episode_rewards[idx];
            self.log.episode_count += 1;
            self.log.total_steps += self.ticks[idx];
        }
    }

    /// Step all environments with auto-reset (scalar implementation).
    pub fn step_auto_reset_scalar(&mut self, actions: &[f32]) {
        self.step_scalar(actions);

        for i in 0..self.num_envs {
            if self.terminals[i] != 0 || self.truncations[i] != 0 {
                let new_seed = self.rng_seeds[i].wrapping_add(self.num_envs as u64);
                self.reset_single(i, new_seed);
            }
        }
    }

    /// Step all environments using SIMD (when feature enabled).
    #[cfg(feature = "simd")]
    pub fn step_simd(&mut self, actions: &[f32]) {
        debug_assert_eq!(actions.len(), self.num_envs);

        const LANES: usize = 8;
        let num_chunks = self.num_envs / LANES;

        for chunk in 0..num_chunks {
            let base = chunk * LANES;
            self.step_simd_chunk(base, &actions[base..base + LANES]);
        }

        let remainder_start = num_chunks * LANES;
        for i in remainder_start..self.num_envs {
            self.step_single_env(i, actions[i]);
        }
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    fn step_simd_chunk(&mut self, base: usize, actions: &[f32]) {
        let x = f32x8::from_slice(&self.x[base..]);
        let x_dot = f32x8::from_slice(&self.x_dot[base..]);
        let theta = f32x8::from_slice(&self.theta[base..]);
        let theta_dot = f32x8::from_slice(&self.theta_dot[base..]);

        let actions_simd = f32x8::from_slice(actions);
        let force = actions_simd * f32x8::splat(2.0 * FORCE_MAG) - f32x8::splat(FORCE_MAG);

        let cos_theta = simd_cos(theta);
        let sin_theta = simd_sin(theta);

        let total_mass = f32x8::splat(CART_MASS + POLE_MASS);
        let pole_ml = f32x8::splat(POLE_MASS * POLE_LENGTH);
        let gravity = f32x8::splat(GRAVITY);
        let pole_length = f32x8::splat(POLE_LENGTH);
        let pole_mass = f32x8::splat(POLE_MASS);
        let four_thirds = f32x8::splat(4.0 / 3.0);

        let temp = (force + pole_ml * theta_dot * theta_dot * sin_theta) / total_mass;

        // Prevent division by zero when cos_theta approximation error causes denom ≤ 0
        let denom = pole_length * (four_thirds - pole_mass * cos_theta * cos_theta / total_mass);
        let safe_denom = denom.simd_max(f32x8::splat(1e-6));
        let theta_acc = (gravity * sin_theta - cos_theta * temp) / safe_denom;

        let x_acc = temp - pole_ml * theta_acc * cos_theta / total_mass;

        let dt = f32x8::splat(DT);
        let new_x = x + dt * x_dot;
        let new_x_dot = x_dot + dt * x_acc;
        let new_theta = theta + dt * theta_dot;
        let new_theta_dot = theta_dot + dt * theta_acc;

        new_x.copy_to_slice(&mut self.x[base..base + 8]);
        new_x_dot.copy_to_slice(&mut self.x_dot[base..base + 8]);
        new_theta.copy_to_slice(&mut self.theta[base..base + 8]);
        new_theta_dot.copy_to_slice(&mut self.theta_dot[base..base + 8]);

        let x_threshold = f32x8::splat(X_THRESHOLD);
        let theta_threshold = f32x8::splat(THETA_THRESHOLD);

        let x_out = simd_abs(new_x).simd_gt(x_threshold);
        let theta_out = simd_abs(new_theta).simd_gt(theta_threshold);
        let terminal_mask = x_out | theta_out;
        let terminal_bits = terminal_mask.to_bitmask() as u8;

        let ticks_vec = f32x8::from_array([
            self.ticks[base] as f32,
            self.ticks[base + 1] as f32,
            self.ticks[base + 2] as f32,
            self.ticks[base + 3] as f32,
            self.ticks[base + 4] as f32,
            self.ticks[base + 5] as f32,
            self.ticks[base + 6] as f32,
            self.ticks[base + 7] as f32,
        ]);
        let new_ticks = ticks_vec + f32x8::splat(1.0);

        for lane in 0..8 {
            self.ticks[base + lane] = new_ticks.to_array()[lane] as u32;
        }

        let max_steps_vec = f32x8::splat(self.max_steps as f32);
        let truncation_mask = new_ticks.simd_ge(max_steps_vec);
        let truncation_bits = truncation_mask.to_bitmask() as u8;

        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);
        let reward_vec = terminal_mask.select(zero, one);

        reward_vec.copy_to_slice(&mut self.rewards[base..base + 8]);

        let terminal_bytes: [u8; 8] = [
            terminal_bits & 1,
            (terminal_bits >> 1) & 1,
            (terminal_bits >> 2) & 1,
            (terminal_bits >> 3) & 1,
            (terminal_bits >> 4) & 1,
            (terminal_bits >> 5) & 1,
            (terminal_bits >> 6) & 1,
            (terminal_bits >> 7) & 1,
        ];
        let truncation_bytes: [u8; 8] = [
            truncation_bits & 1,
            (truncation_bits >> 1) & 1,
            (truncation_bits >> 2) & 1,
            (truncation_bits >> 3) & 1,
            (truncation_bits >> 4) & 1,
            (truncation_bits >> 5) & 1,
            (truncation_bits >> 6) & 1,
            (truncation_bits >> 7) & 1,
        ];
        self.terminals[base..base + 8].copy_from_slice(&terminal_bytes);
        self.truncations[base..base + 8].copy_from_slice(&truncation_bytes);

        let episode_rewards_vec = f32x8::from_slice(&self.episode_rewards[base..base + 8]);
        let new_episode_rewards = episode_rewards_vec + reward_vec;
        new_episode_rewards.copy_to_slice(&mut self.episode_rewards[base..base + 8]);

        let mut chunk_total_reward = 0.0;
        let mut chunk_episode_count = 0;
        let mut chunk_total_steps = 0;

        let combined_mask = terminal_bits | truncation_bits;
        for lane in 0..8 {
            if (combined_mask >> lane) & 1 != 0 {
                chunk_total_reward += self.episode_rewards[base + lane];
                chunk_episode_count += 1;
                chunk_total_steps += self.ticks[base + lane];
            }
        }

        self.log.total_reward += chunk_total_reward;
        self.log.episode_count += chunk_episode_count;
        self.log.total_steps += chunk_total_steps;
    }

    /// Step all environments with auto-reset using SIMD.
    #[cfg(feature = "simd")]
    pub fn step_auto_reset_simd(&mut self, actions: &[f32]) {
        self.step_simd(actions);

        for i in 0..self.num_envs {
            if self.terminals[i] != 0 || self.truncations[i] != 0 {
                let new_seed = self.rng_seeds[i].wrapping_add(self.num_envs as u64);
                self.reset_single(i, new_seed);
            }
        }
    }

    /// Step all environments in parallel across worker threads.
    ///
    /// Each worker processes a chunk of environments using SIMD (if enabled).
    /// Thread-local logs are merged after parallel processing.
    #[cfg(feature = "parallel")]
    pub fn step_parallel(&mut self, actions: &[f32]) {
        debug_assert_eq!(actions.len(), self.num_envs);

        let num_envs = self.num_envs;
        let workers = self.workers;
        let max_steps = self.max_steps;

        let base_chunk_size = num_envs / workers;
        let chunk_size = (base_chunk_size / 8) * 8;
        let chunk_size = chunk_size.max(8);

        let x_ptr = unsafe { SyncPtr::new(self.x.as_mut_ptr()) };
        let x_dot_ptr = unsafe { SyncPtr::new(self.x_dot.as_mut_ptr()) };
        let theta_ptr = unsafe { SyncPtr::new(self.theta.as_mut_ptr()) };
        let theta_dot_ptr = unsafe { SyncPtr::new(self.theta_dot.as_mut_ptr()) };
        let rewards_ptr = unsafe { SyncPtr::new(self.rewards.as_mut_ptr()) };
        let terminals_ptr = unsafe { SyncPtr::new(self.terminals.as_mut_ptr()) };
        let truncations_ptr = unsafe { SyncPtr::new(self.truncations.as_mut_ptr()) };
        let ticks_ptr = unsafe { SyncPtr::new(self.ticks.as_mut_ptr()) };
        let episode_rewards_ptr = unsafe { SyncPtr::new(self.episode_rewards.as_mut_ptr()) };

        let chunk_logs: Vec<CartPoleLog> = (0..workers)
            .into_par_iter()
            .map(|worker_idx| {
                let start = worker_idx * chunk_size;
                let end = if worker_idx == workers - 1 {
                    num_envs
                } else {
                    (start + chunk_size).min(num_envs)
                };

                if start >= num_envs {
                    return CartPoleLog::default();
                }

                let mut local_log = CartPoleLog::default();

                for i in start..end {
                    unsafe {
                        let action = *actions.get_unchecked(i);
                        let force = if action as i32 == 1 { FORCE_MAG } else { -FORCE_MAG };

                        let x = *x_ptr.add(i);
                        let x_dot = *x_dot_ptr.add(i);
                        let theta = *theta_ptr.add(i);
                        let theta_dot = *theta_dot_ptr.add(i);

                        let cos_theta = theta.cos();
                        let sin_theta = theta.sin();

                        let total_mass = CART_MASS + POLE_MASS;
                        let pole_mass_length = POLE_MASS * POLE_LENGTH;

                        let temp = (force + pole_mass_length * theta_dot * theta_dot * sin_theta)
                            / total_mass;

                        let theta_acc = (GRAVITY * sin_theta - cos_theta * temp)
                            / (POLE_LENGTH
                                * (4.0 / 3.0 - POLE_MASS * cos_theta * cos_theta / total_mass));

                        let x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass;

                        let new_x = x + DT * x_dot;
                        let new_x_dot = x_dot + DT * x_acc;
                        let new_theta = theta + DT * theta_dot;
                        let new_theta_dot = theta_dot + DT * theta_acc;

                        *x_ptr.add(i) = new_x;
                        *x_dot_ptr.add(i) = new_x_dot;
                        *theta_ptr.add(i) = new_theta;
                        *theta_dot_ptr.add(i) = new_theta_dot;

                        let tick = *ticks_ptr.add(i) + 1;
                        *ticks_ptr.add(i) = tick;

                        let terminal =
                            (new_x.abs() > X_THRESHOLD) | (new_theta.abs() > THETA_THRESHOLD);
                        let truncated = tick >= max_steps;

                        *terminals_ptr.add(i) = terminal as u8;
                        *truncations_ptr.add(i) = truncated as u8;

                        let reward = if terminal { 0.0 } else { 1.0 };
                        *rewards_ptr.add(i) = reward;

                        let episode_reward = *episode_rewards_ptr.add(i) + reward;
                        *episode_rewards_ptr.add(i) = episode_reward;

                        if terminal || truncated {
                            local_log.total_reward += episode_reward;
                            local_log.episode_count += 1;
                            local_log.total_steps += tick;
                        }
                    }
                }

                local_log
            })
            .collect();

        for log in chunk_logs {
            self.log.merge(&log);
        }
    }

    /// Step all environments with auto-reset using parallel workers.
    #[cfg(feature = "parallel")]
    pub fn step_auto_reset_parallel(&mut self, actions: &[f32]) {
        self.step_parallel(actions);

        for i in 0..self.num_envs {
            if self.terminals[i] != 0 || self.truncations[i] != 0 {
                let new_seed = self.rng_seeds[i].wrapping_add(self.num_envs as u64);
                self.reset_single(i, new_seed);
            }
        }
    }

    /// Step using the best available method based on configuration.
    ///
    /// Priority: parallel (if workers > 1) > SIMD > scalar
    #[inline]
    pub fn step(&mut self, actions: &[f32]) {
        #[cfg(feature = "parallel")]
        if self.workers > 1 {
            self.step_parallel(actions);
            return;
        }

        #[cfg(feature = "simd")]
        {
            self.step_simd(actions);
            return;
        }

        #[cfg(not(feature = "simd"))]
        {
            self.step_scalar(actions);
        }
    }

    /// Step with auto-reset using the best available method.
    ///
    /// Priority: parallel (if workers > 1) > SIMD > scalar
    #[inline]
    pub fn step_auto_reset(&mut self, actions: &[f32]) {
        #[cfg(feature = "parallel")]
        if self.workers > 1 {
            self.step_auto_reset_parallel(actions);
            return;
        }

        #[cfg(feature = "simd")]
        {
            self.step_auto_reset_simd(actions);
            return;
        }

        #[cfg(not(feature = "simd"))]
        {
            self.step_auto_reset_scalar(actions);
        }
    }

    /// Write observations to a flat buffer (for zero-copy numpy integration).
    ///
    /// Buffer layout: [x0, x_dot0, theta0, theta_dot0, x1, x_dot1, ...]
    pub fn write_observations(&self, buffer: &mut [f32]) {
        debug_assert!(buffer.len() >= self.num_envs * 4);

        #[cfg(feature = "simd")]
        {
            self.write_observations_simd(buffer);
        }

        #[cfg(not(feature = "simd"))]
        {
            self.write_observations_scalar(buffer);
        }
    }

    #[cfg(not(feature = "simd"))]
    fn write_observations_scalar(&self, buffer: &mut [f32]) {
        for i in 0..self.num_envs {
            let base = i * 4;
            buffer[base] = self.x[i];
            buffer[base + 1] = self.x_dot[i];
            buffer[base + 2] = self.theta[i];
            buffer[base + 3] = self.theta_dot[i];
        }
    }

    #[cfg(feature = "simd")]
    fn write_observations_simd(&self, buffer: &mut [f32]) {
        const LANES: usize = 8;
        const OBS_DIM: usize = 4;
        let num_chunks = self.num_envs / LANES;

        for chunk in 0..num_chunks {
            let in_base = chunk * LANES;
            let out_base = chunk * LANES * OBS_DIM;

            let x_arr = &self.x[in_base..in_base + LANES];
            let x_dot_arr = &self.x_dot[in_base..in_base + LANES];
            let theta_arr = &self.theta[in_base..in_base + LANES];
            let theta_dot_arr = &self.theta_dot[in_base..in_base + LANES];

            for pair in 0..4 {
                let i = pair * 2;
                let out_idx = out_base + pair * 8;

                buffer[out_idx] = x_arr[i];
                buffer[out_idx + 1] = x_dot_arr[i];
                buffer[out_idx + 2] = theta_arr[i];
                buffer[out_idx + 3] = theta_dot_arr[i];

                buffer[out_idx + 4] = x_arr[i + 1];
                buffer[out_idx + 5] = x_dot_arr[i + 1];
                buffer[out_idx + 6] = theta_arr[i + 1];
                buffer[out_idx + 7] = theta_dot_arr[i + 1];
            }
        }

        let remainder_start = num_chunks * LANES;
        for i in remainder_start..self.num_envs {
            let base = i * OBS_DIM;
            buffer[base] = self.x[i];
            buffer[base + 1] = self.x_dot[i];
            buffer[base + 2] = self.theta[i];
            buffer[base + 3] = self.theta_dot[i];
        }
    }

    /// Write rewards to buffer.
    pub fn write_rewards(&self, buffer: &mut [f32]) {
        buffer[..self.num_envs].copy_from_slice(&self.rewards);
    }

    /// Write terminal flags to buffer.
    pub fn write_terminals(&self, buffer: &mut [u8]) {
        buffer[..self.num_envs].copy_from_slice(&self.terminals);
    }

    /// Write truncation flags to buffer.
    pub fn write_truncations(&self, buffer: &mut [u8]) {
        buffer[..self.num_envs].copy_from_slice(&self.truncations);
    }

    /// Get aggregated log data.
    pub fn get_log(&self) -> CartPoleLog {
        self.log.clone()
    }

    /// Clear log data.
    pub fn clear_log(&mut self) {
        self.log.clear();
    }

    /// Check if done (any environment).
    pub fn any_done(&self) -> bool {
        self.terminals.iter().any(|&t| t != 0) || self.truncations.iter().any(|&t| t != 0)
    }
}

impl operant_core::Environment for CartPole {
    #[inline]
    fn num_envs(&self) -> usize {
        self.num_envs
    }

    #[inline]
    fn observation_size(&self) -> usize {
        4
    }

    #[inline]
    fn num_actions(&self) -> Option<usize> {
        Some(2)
    }

    fn reset(&mut self, seed: u64) {
        CartPole::reset(self, seed);
    }

    fn step(&mut self, actions: &[f32]) {
        self.step_auto_reset(actions);
    }

    fn write_observations(&self, buffer: &mut [f32]) {
        CartPole::write_observations(self, buffer);
    }

    fn write_rewards(&self, buffer: &mut [f32]) {
        CartPole::write_rewards(self, buffer);
    }

    fn write_terminals(&self, buffer: &mut [u8]) {
        CartPole::write_terminals(self, buffer);
    }

    fn write_truncations(&self, buffer: &mut [u8]) {
        CartPole::write_truncations(self, buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartpole_creation() {
        let env = CartPole::with_defaults(1024);
        assert_eq!(env.num_envs(), 1024);
        assert_eq!(env.observation_size(), 4);
    }

    #[test]
    fn test_cartpole_reset() {
        let mut env = CartPole::with_defaults(8);
        env.reset(42);

        for i in 0..8 {
            assert!(env.x[i].abs() <= 0.05);
            assert!(env.x_dot[i].abs() <= 0.05);
            assert!(env.theta[i].abs() <= 0.05);
            assert!(env.theta_dot[i].abs() <= 0.05);
        }
    }

    #[test]
    fn test_cartpole_step_scalar() {
        let mut env = CartPole::with_defaults(4);
        env.reset(0);

        let actions = vec![1.0, 0.0, 1.0, 0.0];
        env.step_scalar(&actions);

        let has_motion = env.x_dot.iter().any(|&v| v.abs() > 0.0);
        assert!(has_motion);
    }

    #[test]
    fn test_cartpole_auto_reset() {
        let mut env = CartPole::new(2, 10, 0.05, 1);
        env.reset(0);

        let actions = vec![1.0, 1.0];

        for _ in 0..100 {
            env.step_auto_reset_scalar(&actions);
        }

        assert!(env.log.episode_count > 0);
    }

    #[test]
    fn test_cartpole_write_observations() {
        let mut env = CartPole::with_defaults(2);
        env.reset(42);

        let mut buffer = vec![0.0f32; 8];
        env.write_observations(&mut buffer);

        assert_eq!(buffer[0], env.x[0]);
        assert_eq!(buffer[1], env.x_dot[0]);
        assert_eq!(buffer[2], env.theta[0]);
        assert_eq!(buffer[3], env.theta_dot[0]);
        assert_eq!(buffer[4], env.x[1]);
    }

    #[test]
    fn test_cartpole_physics_matches_original() {
        let mut soa = CartPole::with_defaults(1);
        soa.reset(42);

        let initial_x = soa.x[0];
        let initial_theta = soa.theta[0];

        soa.step_scalar(&[1.0]);

        assert_ne!(soa.x[0], initial_x);
        assert_ne!(soa.theta[0], initial_theta);

        let mut soa2 = CartPole::with_defaults(1);
        soa2.reset(42);
        soa2.step_scalar(&[1.0]);

        assert_eq!(soa.x[0], soa2.x[0]);
        assert_eq!(soa.theta[0], soa2.theta[0]);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_cartpole_simd_matches_scalar() {
        let mut scalar_env = CartPole::with_defaults(16);
        let mut simd_env = CartPole::with_defaults(16);

        scalar_env.reset(123);
        simd_env.reset(123);

        let actions: Vec<f32> = (0..16).map(|i| (i % 2) as f32).collect();

        scalar_env.step_scalar(&actions);
        simd_env.step_simd(&actions);

        for i in 0..16 {
            assert!(
                (scalar_env.x[i] - simd_env.x[i]).abs() < 1e-5,
                "x mismatch at {}: {} vs {}",
                i,
                scalar_env.x[i],
                simd_env.x[i]
            );
            assert!(
                (scalar_env.theta[i] - simd_env.theta[i]).abs() < 1e-5,
                "theta mismatch at {}: {} vs {}",
                i,
                scalar_env.theta[i],
                simd_env.theta[i]
            );
        }
    }

    #[test]
    fn test_initialization() {
        let env = CartPole::with_defaults(128);
        assert_eq!(env.num_envs(), 128);

        // Check initial state bounds
        for i in 0..128 {
            assert!(env.x[i].abs() <= 0.05);
            assert!(env.x_dot[i].abs() <= 0.05);
            assert!(env.theta[i].abs() <= 0.05);
            assert!(env.theta_dot[i].abs() <= 0.05);
        }
    }

    #[test]
    fn test_reset_deterministic() {
        let mut env1 = CartPole::with_defaults(64);
        let mut env2 = CartPole::with_defaults(64);

        env1.reset(12345);
        env2.reset(12345);

        // Same seed should produce identical initial states
        for i in 0..64 {
            assert_eq!(env1.x[i], env2.x[i]);
            assert_eq!(env1.theta[i], env2.theta[i]);
            assert_eq!(env1.x_dot[i], env2.x_dot[i]);
            assert_eq!(env1.theta_dot[i], env2.theta_dot[i]);
        }
    }

    #[test]
    fn test_terminal_detection() {
        let mut env = CartPole::with_defaults(1);
        env.reset(0);

        env.x[0] = 10.0;

        let actions = vec![0.0];
        env.step_auto_reset(&actions);

        assert!(env.x[0].abs() < 1.0, "Should have reset after terminal");
    }

    #[test]
    fn test_episode_logging() {
        let mut env = CartPole::with_defaults(32);
        env.reset(0);

        let actions: Vec<f32> = vec![0.0; 32];
        for _ in 0..100 {
            env.step_auto_reset(&actions);
        }

        let log = env.get_log();
        assert!(log.episode_count > 0, "Should complete some episodes");
        assert!(log.total_reward > 0.0, "Should accumulate some reward");
        assert!(log.total_steps > 0, "Should count steps");
    }

    #[test]
    fn test_clear_log() {
        let mut env = CartPole::with_defaults(16);
        env.reset(0);

        let actions = vec![0.0; 16];
        for _ in 0..50 {
            env.step_auto_reset(&actions);
        }

        let log_before = env.get_log();
        assert!(log_before.episode_count > 0);

        env.clear_log();
        let log_after = env.get_log();
        assert_eq!(log_after.episode_count, 0);
        assert_eq!(log_after.total_reward, 0.0);
        assert_eq!(log_after.total_steps, 0);
    }

    #[test]
    fn test_observation_write() {
        let mut env = CartPole::with_defaults(8);
        env.reset(42);

        let mut buffer = vec![0.0f32; 8 * 4];
        env.write_observations(&mut buffer);

        for i in 0..8 {
            let base = i * 4;
            assert_eq!(buffer[base], env.x[i]);
            assert_eq!(buffer[base + 1], env.x_dot[i]);
            assert_eq!(buffer[base + 2], env.theta[i]);
            assert_eq!(buffer[base + 3], env.theta_dot[i]);
        }
    }

    #[test]
    fn test_action_effects() {
        let mut env1 = CartPole::with_defaults(1);
        let mut env2 = CartPole::with_defaults(1);

        env1.reset(100);
        env2.reset(100);

        // Apply different actions
        let left = vec![0.0];  // Push left
        let right = vec![1.0];  // Push right

        env1.step_auto_reset(&left);
        env2.step_auto_reset(&right);

        assert_ne!(env1.x_dot[0], env2.x_dot[0], "Different actions should change velocity");
    }

    #[test]
    fn test_batch_consistency() {
        let mut env = CartPole::with_defaults(64);
        env.reset(999);

        let actions = vec![0.0; 64];
        env.step_auto_reset(&actions);

        assert_eq!(env.x.len(), 64);
        assert_eq!(env.rewards.len(), 64);
        assert_eq!(env.terminals.len(), 64);
        assert_eq!(env.truncations.len(), 64);
    }

    #[test]
    fn test_with_workers() {
        let env = CartPole::with_workers(1024, 4);
        assert_eq!(env.num_envs(), 1024);
        assert_eq!(env.workers(), 4);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_matches_scalar() {
        let mut scalar_env = CartPole::with_defaults(64);
        let mut parallel_env = CartPole::with_workers(64, 4);

        scalar_env.reset(123);
        parallel_env.reset(123);

        let actions: Vec<f32> = (0..64).map(|i| (i % 2) as f32).collect();

        scalar_env.step_scalar(&actions);
        parallel_env.step_parallel(&actions);

        for i in 0..64 {
            assert!(
                (scalar_env.x[i] - parallel_env.x[i]).abs() < 1e-5,
                "x mismatch at {}: {} vs {}",
                i,
                scalar_env.x[i],
                parallel_env.x[i]
            );
            assert!(
                (scalar_env.theta[i] - parallel_env.theta[i]).abs() < 1e-5,
                "theta mismatch at {}: {} vs {}",
                i,
                scalar_env.theta[i],
                parallel_env.theta[i]
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_auto_reset() {
        let mut env = CartPole::with_workers(32, 4);
        env.reset(0);

        let actions: Vec<f32> = vec![0.0; 32];
        for _ in 0..100 {
            env.step_auto_reset(&actions);
        }

        let log = env.get_log();
        assert!(log.episode_count > 0, "Should complete some episodes");
        assert!(log.total_reward > 0.0, "Should accumulate some reward");
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_deterministic() {
        let mut env1 = CartPole::with_workers(128, 4);
        let mut env2 = CartPole::with_workers(128, 4);

        env1.reset(42);
        env2.reset(42);

        let actions: Vec<f32> = (0..128).map(|i| (i % 2) as f32).collect();

        for _ in 0..10 {
            env1.step_auto_reset(&actions);
            env2.step_auto_reset(&actions);
        }

        // Same seed + same workers should produce identical results
        for i in 0..128 {
            assert_eq!(env1.x[i], env2.x[i], "x should match at index {}", i);
            assert_eq!(env1.theta[i], env2.theta[i], "theta should match at index {}", i);
        }
    }
}
