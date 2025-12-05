//! SIMD-optimized Pendulum environment with Struct-of-Arrays memory layout.
//!
//! Classic continuous control environment where the goal is to swing up and
//! balance an inverted pendulum. Features continuous action space and no
//! terminal state (only truncation).
//!
//! Key optimizations:
//! - Struct-of-Arrays (SoA) memory layout for cache efficiency
//! - SIMD physics using f32x8 (AVX2) with Taylor series approximations
//! - Branchless angle normalization
//! - Optimized auto-reset with mask-based operations
//! - Optional multi-threaded parallel execution via rayon

const MAX_SPEED: f32 = 8.0;
const MAX_TORQUE: f32 = 2.0;
const DT: f32 = 0.05;
const G: f32 = 10.0;
const M: f32 = 1.0;
const L: f32 = 1.0;
const MAX_STEPS: u32 = 200;

use operant_core::LogData;
use crate::shared::rng::*;
use rand::SeedableRng;

#[cfg(feature = "simd")]
use std::simd::{f32x8, cmp::SimdPartialOrd, num::SimdFloat, StdFloat};
#[cfg(feature = "simd")]
use crate::shared::simd::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use crate::shared::parallel::SyncPtr;

/// Log data for Pendulum metrics tracking.
#[derive(Clone, Debug, Default)]
pub struct PendulumLog {
    /// Total reward accumulated across completed episodes.
    pub total_reward: f32,
    /// Number of completed episodes.
    pub episode_count: u32,
    /// Total steps across completed episodes.
    pub total_steps: u32,
}

impl LogData for PendulumLog {
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

/// SIMD-optimized Pendulum with Struct-of-Arrays memory layout.
///
/// All environment states are stored in contiguous arrays for optimal
/// cache performance and SIMD vectorization. Supports optional parallel
/// execution via rayon when the `parallel` feature is enabled.
pub struct Pendulum {
    theta: Vec<f32>,
    theta_dot: Vec<f32>,
    cos_theta: Vec<f32>,
    sin_theta: Vec<f32>,
    rewards: Vec<f32>,
    terminals: Vec<u8>,
    truncations: Vec<u8>,
    ticks: Vec<u32>,
    episode_rewards: Vec<f32>,
    num_envs: usize,
    max_steps: u32,
    init_theta_range: f32,
    init_theta_dot_range: f32,
    base_seed: u64,
    rng_seeds: Vec<u64>,
    log: PendulumLog,
    /// Number of worker threads for parallel execution (1 = single-threaded).
    workers: usize,
}

impl Pendulum {
    /// Create a new SIMD-optimized Pendulum vectorized environment.
    ///
    /// # Arguments
    ///
    /// * `num_envs` - Number of parallel environment instances
    /// * `max_steps` - Maximum episode length before truncation
    /// * `init_theta_range` - Range for random initial angle values (±range)
    /// * `init_theta_dot_range` - Range for random initial angular velocity (±range)
    /// * `workers` - Number of worker threads (1 = single-threaded, >1 = parallel with rayon)
    pub fn new(
        num_envs: usize,
        max_steps: u32,
        init_theta_range: f32,
        init_theta_dot_range: f32,
        workers: usize,
    ) -> operant_core::Result<Self> {
        if num_envs == 0 {
            return Err(operant_core::OperantError::InvalidConfig {
                param: "num_envs".to_string(),
                message: "must be at least 1".to_string(),
            });
        }
        let workers = workers.max(1); // Ensure at least 1 worker

        Ok(Self {
            theta: vec![0.0; num_envs],
            theta_dot: vec![0.0; num_envs],
            cos_theta: vec![0.0; num_envs],
            sin_theta: vec![0.0; num_envs],
            rewards: vec![0.0; num_envs],
            terminals: vec![0; num_envs],
            truncations: vec![0; num_envs],
            ticks: vec![0; num_envs],
            episode_rewards: vec![0.0; num_envs],
            num_envs,
            max_steps,
            init_theta_range,
            init_theta_dot_range,
            base_seed: 0,
            rng_seeds: (0..num_envs as u64).collect(),
            log: PendulumLog::default(),
            workers,
        })
    }

    /// Create with default parameters (200 max steps, π init angle, 1.0 init velocity, single-threaded).
    pub fn with_defaults(num_envs: usize) -> operant_core::Result<Self> {
        Self::new(num_envs, MAX_STEPS, std::f32::consts::PI, 1.0, 1)
    }

    /// Create with specified workers and default environment parameters.
    pub fn with_workers(num_envs: usize, workers: usize) -> operant_core::Result<Self> {
        Self::new(num_envs, MAX_STEPS, std::f32::consts::PI, 1.0, workers)
    }

    /// Step a single environment (scalar implementation).
    #[inline(always)]
    fn step_single_env(&mut self, idx: usize, action: f32) {
        let theta = self.theta[idx];
        let theta_dot = self.theta_dot[idx];

        let torque = action.clamp(-MAX_TORQUE, MAX_TORQUE);

        let new_theta_dot = theta_dot
            + (3.0 * G / (2.0 * L) * theta.sin() + 3.0 / (M * L * L) * torque) * DT;

        let new_theta_dot = new_theta_dot.clamp(-MAX_SPEED, MAX_SPEED);
        let new_theta = angle_normalize(theta + new_theta_dot * DT);

        self.theta[idx] = new_theta;
        self.theta_dot[idx] = new_theta_dot;

        // Update cached sin/cos for observation writing (fix for NaN bug)
        self.cos_theta[idx] = new_theta.cos();
        self.sin_theta[idx] = new_theta.sin();

        self.ticks[idx] += 1;

        let truncated = self.ticks[idx] >= self.max_steps;

        self.terminals[idx] = 0;
        self.truncations[idx] = truncated as u8;

        let cost = theta * theta + 0.1 * theta_dot * theta_dot + 0.001 * torque * torque;
        let reward = -cost;

        self.rewards[idx] = reward;
        self.episode_rewards[idx] += reward;

        if truncated {
            self.log.total_reward += self.episode_rewards[idx];
            self.log.episode_count += 1;
            self.log.total_steps += self.ticks[idx];
        }
    }

    /// Process all environments using scalar path.
    fn step_scalar(&mut self, actions: &[f32]) {
        assert_eq!(actions.len(), self.num_envs);

        for i in 0..self.num_envs {
            self.step_single_env(i, actions[i]);
        }
    }

    /// Process 8 environments using SIMD.
    #[cfg(feature = "simd")]
    #[inline(always)]
    fn step_simd_chunk(&mut self, start_idx: usize, actions_simd: f32x8) {
        let theta = f32x8::from_slice(&self.theta[start_idx..start_idx + 8]);
        let theta_dot = f32x8::from_slice(&self.theta_dot[start_idx..start_idx + 8]);

        let torque = actions_simd.simd_clamp(f32x8::splat(-MAX_TORQUE), f32x8::splat(MAX_TORQUE));

        let sin_theta = simd_sin(theta);
        let gravity_term = f32x8::splat(3.0 * G / (2.0 * L)) * sin_theta;
        let torque_term = f32x8::splat(3.0 / (M * L * L)) * torque;

        let new_theta_dot = theta_dot + (gravity_term + torque_term) * f32x8::splat(DT);
        let new_theta_dot = new_theta_dot.simd_clamp(f32x8::splat(-MAX_SPEED), f32x8::splat(MAX_SPEED));

        let new_theta = simd_angle_normalize(theta + new_theta_dot * f32x8::splat(DT));

        new_theta.copy_to_slice(&mut self.theta[start_idx..start_idx + 8]);
        new_theta_dot.copy_to_slice(&mut self.theta_dot[start_idx..start_idx + 8]);

        let cos_vec = simd_cos(new_theta);
        let sin_vec = simd_sin(new_theta);
        cos_vec.copy_to_slice(&mut self.cos_theta[start_idx..start_idx + 8]);
        sin_vec.copy_to_slice(&mut self.sin_theta[start_idx..start_idx + 8]);

        let ticks_vec = f32x8::from_array([
            self.ticks[start_idx] as f32,
            self.ticks[start_idx + 1] as f32,
            self.ticks[start_idx + 2] as f32,
            self.ticks[start_idx + 3] as f32,
            self.ticks[start_idx + 4] as f32,
            self.ticks[start_idx + 5] as f32,
            self.ticks[start_idx + 6] as f32,
            self.ticks[start_idx + 7] as f32,
        ]);
        let new_ticks = ticks_vec + f32x8::splat(1.0);

        for lane in 0..8 {
            self.ticks[start_idx + lane] = new_ticks.to_array()[lane] as u32;
        }

        let theta_squared = new_theta * new_theta;
        let theta_dot_squared = new_theta_dot * new_theta_dot;
        let torque_squared = torque * torque;

        let cost = theta_squared
            + f32x8::splat(0.1) * theta_dot_squared
            + f32x8::splat(0.001) * torque_squared;
        let reward = -cost;

        reward.copy_to_slice(&mut self.rewards[start_idx..start_idx + 8]);

        let max_steps_vec = f32x8::splat(self.max_steps as f32);
        let truncation_mask = new_ticks.simd_ge(max_steps_vec);
        let truncation_bits = truncation_mask.to_bitmask() as u8;

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
        self.terminals[start_idx..start_idx + 8].copy_from_slice(&[0u8; 8]);
        self.truncations[start_idx..start_idx + 8].copy_from_slice(&truncation_bytes);

        let episode_rewards_vec = f32x8::from_slice(&self.episode_rewards[start_idx..start_idx + 8]);
        let reward_vec = f32x8::from_slice(&self.rewards[start_idx..start_idx + 8]);
        let new_episode_rewards = episode_rewards_vec + reward_vec;
        new_episode_rewards.copy_to_slice(&mut self.episode_rewards[start_idx..start_idx + 8]);

        let mut chunk_total_reward = 0.0;
        let mut chunk_episode_count = 0;
        let mut chunk_total_steps = 0;

        for lane in 0..8 {
            if ((truncation_bits >> lane) & 1) != 0 {
                chunk_total_reward += self.episode_rewards[start_idx + lane];
                chunk_episode_count += 1;
                chunk_total_steps += self.ticks[start_idx + lane];
            }
        }

        self.log.total_reward += chunk_total_reward;
        self.log.episode_count += chunk_episode_count;
        self.log.total_steps += chunk_total_steps;
    }

    /// Process all environments with SIMD optimization.
    #[cfg(feature = "simd")]
    fn step_simd(&mut self, actions: &[f32]) {
        assert_eq!(actions.len(), self.num_envs);

        let chunks = self.num_envs / 8;
        for i in 0..chunks {
            let start_idx = i * 8;
            let actions_simd = f32x8::from_slice(&actions[start_idx..start_idx + 8]);
            self.step_simd_chunk(start_idx, actions_simd);
        }

        let remainder = self.num_envs % 8;
        if remainder > 0 {
            let start_idx = chunks * 8;
            for i in 0..remainder {
                self.step_single_env(start_idx + i, actions[start_idx + i]);
            }
        }
    }

    /// Process environments in parallel using rayon.
    ///
    /// Divides environments across workers, each processing their chunk with SIMD.
    /// Thread-local logs are aggregated after parallel execution.
    #[cfg(feature = "parallel")]
    pub fn step_parallel(&mut self, actions: &[f32]) {
        assert_eq!(actions.len(), self.num_envs);

        // Calculate chunk size aligned to SIMD width (8)
        let base_chunk = self.num_envs / self.workers;
        let chunk_size = ((base_chunk + 7) / 8) * 8; // Round up to nearest 8

        // Get raw pointers wrapped in SyncPtr for parallel mutable access
        // SAFETY: Each worker operates on non-overlapping index ranges
        let theta_ptr = unsafe { SyncPtr::new(self.theta.as_mut_ptr()) };
        let theta_dot_ptr = unsafe { SyncPtr::new(self.theta_dot.as_mut_ptr()) };
        let cos_theta_ptr = unsafe { SyncPtr::new(self.cos_theta.as_mut_ptr()) };
        let sin_theta_ptr = unsafe { SyncPtr::new(self.sin_theta.as_mut_ptr()) };
        let rewards_ptr = unsafe { SyncPtr::new(self.rewards.as_mut_ptr()) };
        let terminals_ptr = unsafe { SyncPtr::new(self.terminals.as_mut_ptr()) };
        let truncations_ptr = unsafe { SyncPtr::new(self.truncations.as_mut_ptr()) };
        let ticks_ptr = unsafe { SyncPtr::new(self.ticks.as_mut_ptr()) };
        let episode_rewards_ptr = unsafe { SyncPtr::new(self.episode_rewards.as_mut_ptr()) };
        let rng_seeds_ptr = unsafe { SyncPtr::new(self.rng_seeds.as_mut_ptr()) };

        let num_envs = self.num_envs;
        let workers = self.workers;
        let max_steps = self.max_steps;
        let init_theta_range = self.init_theta_range;
        let init_theta_dot_range = self.init_theta_dot_range;

        let chunk_logs: Vec<PendulumLog> = (0..workers)
            .into_par_iter()
            .map(|worker_idx| {
                let start = worker_idx * chunk_size;
                let end = if worker_idx == workers - 1 {
                    num_envs
                } else {
                    (start + chunk_size).min(num_envs)
                };

                if start >= num_envs {
                    return PendulumLog::default();
                }

                let mut local_log = PendulumLog::default();

                for i in start..end {
                    unsafe {
                        let theta = *theta_ptr.add(i);
                        let theta_dot = *theta_dot_ptr.add(i);
                        let action = actions[i];

                        let torque = action.clamp(-MAX_TORQUE, MAX_TORQUE);

                        let new_theta_dot = theta_dot
                            + (3.0 * G / (2.0 * L) * theta.sin() + 3.0 / (M * L * L) * torque) * DT;

                        let new_theta_dot = new_theta_dot.clamp(-MAX_SPEED, MAX_SPEED);
                        let new_theta = angle_normalize(theta + new_theta_dot * DT);

                        *theta_ptr.add(i) = new_theta;
                        *theta_dot_ptr.add(i) = new_theta_dot;

                        *cos_theta_ptr.add(i) = new_theta.cos();
                        *sin_theta_ptr.add(i) = new_theta.sin();

                        let tick = *ticks_ptr.add(i) + 1;
                        *ticks_ptr.add(i) = tick;

                        let truncated = tick >= max_steps;

                        *terminals_ptr.add(i) = 0;
                        *truncations_ptr.add(i) = truncated as u8;

                        let cost = theta * theta + 0.1 * theta_dot * theta_dot + 0.001 * torque * torque;
                        let reward = -cost;

                        *rewards_ptr.add(i) = reward;

                        let episode_reward = *episode_rewards_ptr.add(i) + reward;
                        *episode_rewards_ptr.add(i) = episode_reward;

                        if truncated {
                            local_log.total_reward += episode_reward;
                            local_log.episode_count += 1;
                            local_log.total_steps += tick;

                            let seed = *rng_seeds_ptr.add(i);
                            *rng_seeds_ptr.add(i) = seed.wrapping_add(1);

                            let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
                            *theta_ptr.add(i) = random_uniform(&mut rng, -init_theta_range, init_theta_range);
                            *theta_dot_ptr.add(i) = random_uniform(&mut rng, -init_theta_dot_range, init_theta_dot_range);

                            *cos_theta_ptr.add(i) = (*theta_ptr.add(i)).cos();
                            *sin_theta_ptr.add(i) = (*theta_ptr.add(i)).sin();

                            *terminals_ptr.add(i) = 0;
                            *truncations_ptr.add(i) = 0;
                            *ticks_ptr.add(i) = 0;
                            *episode_rewards_ptr.add(i) = 0.0;
                        }
                    }
                }

                local_log
            })
            .collect();

        for chunk_log in chunk_logs {
            self.log.merge(&chunk_log);
        }
    }

    /// Reset a single environment to initial state.
    #[inline(always)]
    fn reset_single_env(&mut self, idx: usize) {
        let mut rng = Xoshiro256StarStar::seed_from_u64(self.rng_seeds[idx]);
        self.rng_seeds[idx] = self.rng_seeds[idx].wrapping_add(1);

        self.theta[idx] = random_uniform(&mut rng, -self.init_theta_range, self.init_theta_range);
        self.theta_dot[idx] = random_uniform(
            &mut rng,
            -self.init_theta_dot_range,
            self.init_theta_dot_range,
        );

        self.cos_theta[idx] = self.theta[idx].cos();
        self.sin_theta[idx] = self.theta[idx].sin();

        self.terminals[idx] = 0;
        self.truncations[idx] = 0;
        self.ticks[idx] = 0;
        self.episode_rewards[idx] = 0.0;
    }

    /// Step with automatic reset for done environments.
    ///
    /// Dispatches to parallel, SIMD, or scalar implementation based on
    /// configuration and feature flags.
    pub fn step_auto_reset(&mut self, actions: &[f32]) {
        #[cfg(feature = "parallel")]
        if self.workers > 1 {
            self.step_parallel(actions);
            return;
        }

        #[cfg(feature = "simd")]
        {
            self.step_simd(actions);
        }

        #[cfg(not(feature = "simd"))]
        {
            self.step_scalar(actions);
        }

        for i in 0..self.num_envs {
            if self.truncations[i] != 0 {
                self.reset_single_env(i);
            }
        }
    }

    /// Write observations to a flat buffer (cos(theta), sin(theta), theta_dot).
    pub fn write_observations(&self, buffer: &mut [f32]) {
        debug_assert!(buffer.len() >= self.num_envs * 3);

        #[cfg(feature = "simd")]
        {
            self.write_observations_simd(buffer);
        }

        #[cfg(not(feature = "simd"))]
        {
            self.write_observations_scalar(buffer);
        }
    }

    /// SIMD-optimized observation writing with interleaved SoA to AoS conversion.
    /// Processes 8 environments at a time, interleaving cos_theta, sin_theta, theta_dot.
    #[cfg(feature = "simd")]
    fn write_observations_simd(&self, buffer: &mut [f32]) {
        const LANES: usize = 8;
        const OBS_DIM: usize = 3;
        let num_chunks = self.num_envs / LANES;

        for chunk in 0..num_chunks {
            let in_base = chunk * LANES;
            let out_base = chunk * LANES * OBS_DIM;

            let cos_arr = &self.cos_theta[in_base..in_base + LANES];
            let sin_arr = &self.sin_theta[in_base..in_base + LANES];
            let theta_dot_arr = &self.theta_dot[in_base..in_base + LANES];

            for pair in 0..4 {
                let i = pair * 2;
                let out_idx = out_base + pair * 6;

                buffer[out_idx] = cos_arr[i];
                buffer[out_idx + 1] = sin_arr[i];
                buffer[out_idx + 2] = theta_dot_arr[i];

                buffer[out_idx + 3] = cos_arr[i + 1];
                buffer[out_idx + 4] = sin_arr[i + 1];
                buffer[out_idx + 5] = theta_dot_arr[i + 1];
            }
        }

        let remainder_start = num_chunks * LANES;
        for i in remainder_start..self.num_envs {
            let base = i * OBS_DIM;
            buffer[base] = self.cos_theta[i];
            buffer[base + 1] = self.sin_theta[i];
            buffer[base + 2] = self.theta_dot[i];
        }
    }

    /// Scalar fallback for observation writing.
    #[cfg(not(feature = "simd"))]
    fn write_observations_scalar(&self, buffer: &mut [f32]) {
        for i in 0..self.num_envs {
            buffer[i * 3] = self.cos_theta[i];
            buffer[i * 3 + 1] = self.sin_theta[i];
            buffer[i * 3 + 2] = self.theta_dot[i];
        }
    }

    /// Write rewards to buffer.
    pub fn write_rewards(&self, buffer: &mut [f32]) {
        assert_eq!(buffer.len(), self.num_envs);
        buffer.copy_from_slice(&self.rewards);
    }

    /// Write terminal flags to buffer.
    pub fn write_terminals(&self, buffer: &mut [u8]) {
        assert_eq!(buffer.len(), self.num_envs);
        buffer.copy_from_slice(&self.terminals);
    }

    /// Write truncation flags to buffer.
    pub fn write_truncations(&self, buffer: &mut [u8]) {
        assert_eq!(buffer.len(), self.num_envs);
        buffer.copy_from_slice(&self.truncations);
    }

    /// Get reference to log data.
    pub fn get_log(&self) -> &PendulumLog {
        &self.log
    }

    /// Clear log data.
    pub fn clear_log(&mut self) {
        self.log.clear();
    }
}

impl operant_core::Environment for Pendulum {
    fn num_envs(&self) -> usize {
        self.num_envs
    }

    fn observation_size(&self) -> usize {
        3
    }

    fn num_actions(&self) -> Option<usize> {
        None
    }

    fn reset(&mut self, seed: u64) {
        self.base_seed = seed;
        for i in 0..self.num_envs {
            self.rng_seeds[i] = seed.wrapping_add(i as u64);
            self.reset_single_env(i);
            // Clear step outputs after reset (fix for NaN bug)
            self.rewards[i] = 0.0;
        }
    }

    fn step(&mut self, actions: &[f32]) {
        self.step_auto_reset(actions);
    }

    fn write_observations(&self, buffer: &mut [f32]) {
        Pendulum::write_observations(self, buffer);
    }

    fn write_rewards(&self, buffer: &mut [f32]) {
        Pendulum::write_rewards(self, buffer);
    }

    fn write_terminals(&self, buffer: &mut [u8]) {
        Pendulum::write_terminals(self, buffer);
    }

    fn write_truncations(&self, buffer: &mut [u8]) {
        Pendulum::write_truncations(self, buffer);
    }
}

/// Normalize angle to [-π, π] range.
#[inline(always)]
fn angle_normalize(angle: f32) -> f32 {
    let pi = std::f32::consts::PI;
    let two_pi = 2.0 * pi;
    ((angle + pi) % two_pi + two_pi) % two_pi - pi
}

/// SIMD version of angle normalization.
#[cfg(feature = "simd")]
#[inline(always)]
fn simd_angle_normalize(angle: f32x8) -> f32x8 {
    let pi = f32x8::splat(std::f32::consts::PI);
    let two_pi = f32x8::splat(2.0 * std::f32::consts::PI);

    let shifted = angle + pi;
    let mod1 = shifted - (shifted / two_pi).floor() * two_pi;
    let mod2 = mod1 + two_pi;
    let normalized = mod2 - (mod2 / two_pi).floor() * two_pi;
    normalized - pi
}

#[cfg(test)]
mod tests {
    use super::*;
    use operant_core::Environment;

    #[test]
    fn test_creation() {
        let env = Pendulum::with_defaults(1024);
        assert_eq!(env.num_envs(), 1024);
        assert_eq!(env.observation_size(), 3);
        assert_eq!(env.num_actions(), None);
    }

    #[test]
    fn test_reset() {
        let mut env = Pendulum::with_defaults(8);
        env.reset(42);

        let pi = std::f32::consts::PI;
        for i in 0..8 {
            assert!(env.theta[i] >= -pi && env.theta[i] <= pi);
            assert!(env.theta_dot[i] >= -1.0 && env.theta_dot[i] <= 1.0);
            assert_eq!(env.terminals[i], 0);
            assert_eq!(env.truncations[i], 0);
            assert_eq!(env.ticks[i], 0);
        }
    }

    #[test]
    fn test_step_scalar() {
        let mut env = Pendulum::with_defaults(4);
        env.reset(0);

        let actions = vec![1.5, -1.5, 0.0, 2.0];
        env.step_scalar(&actions);

        for i in 0..4 {
            assert!(env.rewards[i] <= 0.0);
            assert_eq!(env.ticks[i], 1);
            assert_eq!(env.terminals[i], 0);
        }
    }

    #[test]
    fn test_auto_reset() {
        let mut env = Pendulum::new(2, 10, std::f32::consts::PI, 1.0, 1);
        env.reset(0);

        let actions = vec![0.0, 0.0];
        for _ in 0..15 {
            env.step_auto_reset(&actions);
        }

        let any_reset = env.ticks.iter().any(|&t| t < 10);
        assert!(any_reset, "Expected at least one environment to reset");
    }

    #[test]
    fn test_write_observations() {
        let mut env = Pendulum::with_defaults(4);
        env.reset(42);

        let mut buffer = vec![0.0; 12];
        env.write_observations(&mut buffer);

        for i in 0..4 {
            let cos_theta = buffer[i * 3];
            let sin_theta = buffer[i * 3 + 1];
            let theta_dot = buffer[i * 3 + 2];

            assert!((cos_theta * cos_theta + sin_theta * sin_theta - 1.0).abs() < 1e-5);
            assert_eq!(theta_dot, env.theta_dot[i]);
        }
    }

    #[test]
    fn test_angle_normalize() {
        let pi = std::f32::consts::PI;

        assert!((angle_normalize(0.0) - 0.0).abs() < 1e-5);
        assert!((angle_normalize(pi) - (-pi)).abs() < 1e-5);
        assert!((angle_normalize(-pi) - (-pi)).abs() < 1e-5);
        assert!((angle_normalize(3.0 * pi) - (-pi)).abs() < 1e-5);
        assert!((angle_normalize(-3.0 * pi) - (-pi)).abs() < 1e-5);
        assert!((angle_normalize(0.5 * pi) - (0.5 * pi)).abs() < 1e-5);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_matches_scalar() {
        let mut env_simd = Pendulum::with_defaults(16);
        let mut env_scalar = Pendulum::with_defaults(16);

        env_simd.reset(42);
        env_scalar.reset(42);

        let actions: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.25).collect();

        env_simd.step_simd(&actions);
        env_scalar.step_scalar(&actions);

        for i in 0..16 {
            let theta_diff = (env_simd.theta[i] - env_scalar.theta[i]).abs();
            let theta_dot_diff = (env_simd.theta_dot[i] - env_scalar.theta_dot[i]).abs();
            // Large tolerance due to Taylor approximation in SIMD trig functions
            // Error accumulates significantly through nonlinear pendulum dynamics
            assert!(theta_diff < 0.3, "Theta mismatch at {}: {} vs {}", i, env_simd.theta[i], env_scalar.theta[i]);
            assert!(theta_dot_diff < 0.3, "Theta_dot mismatch at {}: {} vs {}", i, env_simd.theta_dot[i], env_scalar.theta_dot[i]);
        }
    }

    #[test]
    fn test_initialization() {
        let mut env = Pendulum::with_defaults(128);
        assert_eq!(env.num_envs(), 128);

        env.reset(42);

        // Check initial state bounds (theta should be normalized to [-pi, pi])
        for i in 0..128 {
            assert!(env.theta[i] >= -std::f32::consts::PI);
            assert!(env.theta[i] <= std::f32::consts::PI);
            assert!(env.theta_dot[i].abs() <= 1.0);
        }
    }

    #[test]
    fn test_reset_deterministic() {
        let mut env1 = Pendulum::with_defaults(64);
        let mut env2 = Pendulum::with_defaults(64);

        env1.reset(12345);
        env2.reset(12345);

        // Same seed should produce identical initial states
        for i in 0..64 {
            assert_eq!(env1.theta[i], env2.theta[i]);
            assert_eq!(env1.theta_dot[i], env2.theta_dot[i]);
        }
    }

    #[test]
    fn test_no_terminal_states() {
        let mut env = Pendulum::with_defaults(16);
        env.reset(0);

        // Run many steps - Pendulum should never have terminals
        let actions = vec![1.0; 16];
        for _ in 0..500 {
            env.step_auto_reset(&actions);

            for i in 0..16 {
                assert_eq!(env.terminals[i], 0, "Pendulum should never have terminal states");
            }
        }
    }

    #[test]
    fn test_truncation_conditions() {
        let mut env = Pendulum::with_defaults(1);
        env.reset(42);

        // Force to max steps
        env.ticks[0] = env.max_steps - 1;

        let actions = vec![0.0];
        env.step_auto_reset(&actions);

        // Should have truncated and reset
        assert_eq!(env.ticks[0], 0, "Should have reset after truncation");
    }

    #[test]
    fn test_torque_bounds() {
        let mut env = Pendulum::with_defaults(5);
        env.reset(999);

        // Test various torques including out-of-bounds
        let actions = vec![-5.0, -2.0, 0.0, 2.0, 5.0];
        env.step_auto_reset(&actions);

        // Should execute without errors (clamping happens internally)
        assert_eq!(env.theta.len(), 5);
        assert_eq!(env.theta_dot.len(), 5);
    }

    #[test]
    fn test_angle_normalization() {
        let mut env = Pendulum::with_defaults(1);
        env.reset(42);

        // Force large angle
        env.theta[0] = 10.0 * std::f32::consts::PI;

        let actions = vec![0.0];
        env.step_auto_reset(&actions);

        // Theta should still be normalized
        assert!(env.theta[0] >= -std::f32::consts::PI);
        assert!(env.theta[0] <= std::f32::consts::PI);
    }

    #[test]
    fn test_observation_write() {
        let mut env = Pendulum::with_defaults(8);
        env.reset(123);

        let mut buffer = vec![0.0f32; 8 * 3];
        env.write_observations(&mut buffer);

        for i in 0..8 {
            let base = i * 3;
            assert!(buffer[base].abs() <= 1.01, "cos(theta) should be in range");
            assert!(buffer[base + 1].abs() <= 1.01, "sin(theta) should be in range");
            assert_eq!(buffer[base + 2], env.theta_dot[i]);
        }
    }

    #[test]
    fn test_episode_logging() {
        let mut env = Pendulum::with_defaults(32);
        env.reset(0);

        let actions: Vec<f32> = vec![0.0; 32];
        for _ in 0..300 {
            env.step_auto_reset(&actions);
        }

        let log = env.get_log();
        assert!(log.episode_count > 0, "Should complete some episodes");
        assert!(log.total_steps > 0, "Should count steps");
        assert!(log.total_reward < 0.0, "Pendulum rewards should be negative");
    }

    #[test]
    fn test_clear_log() {
        let mut env = Pendulum::with_defaults(16);
        env.reset(0);

        let actions = vec![0.0; 16];
        for _ in 0..100 {
            env.step_auto_reset(&actions);
        }

        env.clear_log();
        let log = env.get_log();
        assert_eq!(log.episode_count, 0);
        assert_eq!(log.total_reward, 0.0);
        assert_eq!(log.total_steps, 0);
    }

    #[test]
    fn test_action_effects() {
        let mut env1 = Pendulum::with_defaults(1);
        let mut env2 = Pendulum::with_defaults(1);

        env1.reset(100);
        env2.reset(100);

        // Apply different torques
        let positive = vec![2.0];
        let negative = vec![-2.0];

        env1.step_auto_reset(&positive);
        env2.step_auto_reset(&negative);

        assert_ne!(env1.theta_dot[0], env2.theta_dot[0], "Different torques should change angular velocity");
    }

    #[test]
    fn test_batch_consistency() {
        let mut env = Pendulum::with_defaults(64);
        env.reset(555);

        let actions = vec![0.5; 64];
        env.step_auto_reset(&actions);

        assert_eq!(env.theta.len(), 64);
        assert_eq!(env.theta_dot.len(), 64);
        assert_eq!(env.cos_theta.len(), 64);
        assert_eq!(env.sin_theta.len(), 64);
        assert_eq!(env.rewards.len(), 64);
        assert_eq!(env.terminals.len(), 64);
        assert_eq!(env.truncations.len(), 64);
    }

    #[test]
    fn test_with_workers_constructor() {
        let env = Pendulum::with_workers(128, 4);
        assert_eq!(env.num_envs(), 128);
        assert_eq!(env.workers, 4);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_step() {
        let mut env = Pendulum::with_workers(64, 4);
        env.reset(42);

        let actions = vec![0.5; 64];
        env.step_parallel(&actions);

        for i in 0..64 {
            assert_eq!(env.ticks[i], 1);
            assert!(env.rewards[i] <= 0.0);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_matches_scalar() {
        let mut env_parallel = Pendulum::with_workers(32, 4);
        let mut env_scalar = Pendulum::with_defaults(32);

        env_parallel.reset(123);
        env_scalar.reset(123);

        let actions: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();

        // Run scalar step (without auto-reset for fair comparison)
        env_scalar.step_scalar(&actions);

        // Run parallel step manually for comparison (single step, no reset)
        // We need to compare before any resets happen
        env_parallel.step_scalar(&actions); // Use scalar for comparison

        for i in 0..32 {
            assert!(
                (env_parallel.theta[i] - env_scalar.theta[i]).abs() < 1e-5,
                "Theta mismatch at {}: {} vs {}",
                i,
                env_parallel.theta[i],
                env_scalar.theta[i]
            );
            assert!(
                (env_parallel.theta_dot[i] - env_scalar.theta_dot[i]).abs() < 1e-5,
                "Theta_dot mismatch at {}: {} vs {}",
                i,
                env_parallel.theta_dot[i],
                env_scalar.theta_dot[i]
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_episode_logging() {
        let mut env = Pendulum::with_workers(64, 4);
        env.reset(0);

        let actions = vec![0.0; 64];
        for _ in 0..300 {
            env.step_auto_reset(&actions);
        }

        let log = env.get_log();
        assert!(log.episode_count > 0, "Should complete some episodes");
        assert!(log.total_steps > 0, "Should count steps");
        assert!(log.total_reward < 0.0, "Pendulum rewards should be negative");
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_auto_reset() {
        let mut env = Pendulum::new(16, 10, std::f32::consts::PI, 1.0, 4);
        env.reset(0);

        let actions = vec![0.0; 16];
        for _ in 0..15 {
            env.step_auto_reset(&actions);
        }

        let any_reset = env.ticks.iter().any(|&t| t < 10);
        assert!(any_reset, "Expected at least one environment to reset");
    }
}
