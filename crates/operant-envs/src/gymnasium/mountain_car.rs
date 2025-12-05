//! SIMD-optimized MountainCar environment with Struct-of-Arrays memory layout.
//!
//! Classic reinforcement learning environment where an underpowered car must
//! build momentum to reach the top of a hill. Features sparse rewards and
//! challenging exploration.
//!
//! Key optimizations:
//! - Struct-of-Arrays (SoA) memory layout for cache efficiency
//! - SIMD physics using f32x8 (AVX2) or f32x4 (SSE2/NEON)
//! - Branchless termination checks
//! - Optimized auto-reset with mask-based operations

const GRAVITY: f32 = 0.0025;
const FORCE: f32 = 0.001;
const MIN_POSITION: f32 = -1.2;
const MAX_POSITION: f32 = 0.6;
const GOAL_POSITION: f32 = 0.5;
const MAX_SPEED: f32 = 0.07;
const MAX_STEPS: u32 = 200;

use operant_core::LogData;
use crate::shared::rng::*;
use rand::SeedableRng;

#[cfg(feature = "simd")]
use std::simd::{f32x8, cmp::{SimdPartialOrd, SimdPartialEq}, num::SimdFloat, StdFloat};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use crate::shared::parallel::SyncPtr;

/// Log data for MountainCar metrics tracking.
#[derive(Clone, Debug, Default)]
pub struct MountainCarLog {
    /// Total reward accumulated across completed episodes.
    pub total_reward: f32,
    /// Number of completed episodes.
    pub episode_count: u32,
    /// Total steps across completed episodes.
    pub total_steps: u32,
}

impl LogData for MountainCarLog {
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

/// SIMD-optimized MountainCar with Struct-of-Arrays memory layout.
///
/// All environment states are stored in contiguous arrays for optimal
/// cache performance and SIMD vectorization.
pub struct MountainCar {
    position: Vec<f32>,
    velocity: Vec<f32>,
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
    log: MountainCarLog,
    /// Number of worker threads (1 = single-threaded, >1 = parallel).
    workers: usize,
}

impl MountainCar {
    /// Create a new SIMD-optimized MountainCar vectorized environment.
    ///
    /// # Arguments
    ///
    /// * `num_envs` - Number of parallel environment instances
    /// * `max_steps` - Maximum episode length before truncation
    /// * `init_range` - Range for random initial position values
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
            position: vec![0.0; num_envs],
            velocity: vec![0.0; num_envs],
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
            log: MountainCarLog::default(),
            workers,
        })
    }

    /// Create with default parameters (200 max steps, 0.6 init range, single-threaded).
    pub fn with_defaults(num_envs: usize) -> operant_core::Result<Self> {
        Self::new(num_envs, MAX_STEPS, 0.6, 1)
    }

    /// Create with default parameters and specified worker count.
    pub fn with_workers(num_envs: usize, workers: usize) -> operant_core::Result<Self> {
        Self::new(num_envs, MAX_STEPS, 0.6, workers)
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
        2
    }

    /// Step a single environment (scalar implementation).
    #[inline(always)]
    fn step_single_env(&mut self, idx: usize, action: f32) {
        let force_direction = action - 1.0;
        let mut velocity = self.velocity[idx];
        let mut position = self.position[idx];

        velocity += force_direction * FORCE + (position * 3.0).cos() * (-GRAVITY);
        velocity = velocity.clamp(-MAX_SPEED, MAX_SPEED);
        position += velocity;
        position = position.clamp(MIN_POSITION, MAX_POSITION);

        if position == MIN_POSITION && velocity < 0.0 {
            velocity = 0.0;
        }

        self.position[idx] = position;
        self.velocity[idx] = velocity;
        self.ticks[idx] += 1;

        let terminal = position >= GOAL_POSITION;
        let truncated = self.ticks[idx] >= self.max_steps;

        self.terminals[idx] = terminal as u8;
        self.truncations[idx] = truncated as u8;

        let reward = -1.0;
        self.rewards[idx] = reward;
        self.episode_rewards[idx] += reward;

        if terminal || truncated {
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
        let force_direction = actions_simd - f32x8::splat(1.0);

        let mut velocity = f32x8::from_slice(&self.velocity[start_idx..start_idx + 8]);
        let mut position = f32x8::from_slice(&self.position[start_idx..start_idx + 8]);

        let gravity_effect = (position * f32x8::splat(3.0)).cos() * f32x8::splat(-GRAVITY);
        velocity += force_direction * f32x8::splat(FORCE) + gravity_effect;
        velocity = velocity.simd_clamp(f32x8::splat(-MAX_SPEED), f32x8::splat(MAX_SPEED));
        position += velocity;
        position = position.simd_clamp(f32x8::splat(MIN_POSITION), f32x8::splat(MAX_POSITION));

        let at_left_bound = position.simd_eq(f32x8::splat(MIN_POSITION));
        let velocity_negative = velocity.simd_lt(f32x8::splat(0.0));
        let reset_velocity_mask = at_left_bound & velocity_negative;
        velocity = reset_velocity_mask.select(f32x8::splat(0.0), velocity);

        position.copy_to_slice(&mut self.position[start_idx..start_idx + 8]);
        velocity.copy_to_slice(&mut self.velocity[start_idx..start_idx + 8]);

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

        let max_steps_vec = f32x8::splat(self.max_steps as f32);
        let truncation_mask = new_ticks.simd_ge(max_steps_vec);
        let truncation_bits = truncation_mask.to_bitmask() as u8;

        let goal_threshold = f32x8::splat(GOAL_POSITION);
        let terminal_mask = position.simd_ge(goal_threshold);
        let terminal_bits = terminal_mask.to_bitmask() as u8;

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
        self.terminals[start_idx..start_idx + 8].copy_from_slice(&terminal_bytes);
        self.truncations[start_idx..start_idx + 8].copy_from_slice(&truncation_bytes);

        let reward_vec = f32x8::splat(-1.0);
        reward_vec.copy_to_slice(&mut self.rewards[start_idx..start_idx + 8]);

        let episode_rewards_vec = f32x8::from_slice(&self.episode_rewards[start_idx..start_idx + 8]);
        let new_episode_rewards = episode_rewards_vec + reward_vec;
        new_episode_rewards.copy_to_slice(&mut self.episode_rewards[start_idx..start_idx + 8]);

        let mut chunk_total_reward = 0.0;
        let mut chunk_episode_count = 0;
        let mut chunk_total_steps = 0;

        let combined_mask = terminal_bits | truncation_bits;
        for lane in 0..8 {
            if ((combined_mask >> lane) & 1) != 0 {
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

    /// Reset a single environment to initial state.
    #[inline(always)]
    fn reset_single_env(&mut self, idx: usize) {
        let mut rng = Xoshiro256StarStar::seed_from_u64(self.rng_seeds[idx]);
        self.rng_seeds[idx] = self.rng_seeds[idx].wrapping_add(1);

        self.position[idx] = random_uniform(&mut rng, -0.6, -0.4);
        self.velocity[idx] = 0.0;
        self.terminals[idx] = 0;
        self.truncations[idx] = 0;
        self.ticks[idx] = 0;
        self.episode_rewards[idx] = 0.0;
    }

    /// Step all environments in parallel across worker threads.
    #[cfg(feature = "parallel")]
    pub fn step_parallel(&mut self, actions: &[f32]) {
        assert_eq!(actions.len(), self.num_envs);

        let num_envs = self.num_envs;
        let workers = self.workers;
        let max_steps = self.max_steps;

        let base_chunk_size = num_envs / workers;
        let chunk_size = (base_chunk_size / 8) * 8;
        let chunk_size = chunk_size.max(8);

        let position_ptr = unsafe { SyncPtr::new(self.position.as_mut_ptr()) };
        let velocity_ptr = unsafe { SyncPtr::new(self.velocity.as_mut_ptr()) };
        let rewards_ptr = unsafe { SyncPtr::new(self.rewards.as_mut_ptr()) };
        let terminals_ptr = unsafe { SyncPtr::new(self.terminals.as_mut_ptr()) };
        let truncations_ptr = unsafe { SyncPtr::new(self.truncations.as_mut_ptr()) };
        let ticks_ptr = unsafe { SyncPtr::new(self.ticks.as_mut_ptr()) };
        let episode_rewards_ptr = unsafe { SyncPtr::new(self.episode_rewards.as_mut_ptr()) };

        let chunk_logs: Vec<MountainCarLog> = (0..workers)
            .into_par_iter()
            .map(|worker_idx| {
                let start = worker_idx * chunk_size;
                let end = if worker_idx == workers - 1 {
                    num_envs
                } else {
                    (start + chunk_size).min(num_envs)
                };

                if start >= num_envs {
                    return MountainCarLog::default();
                }

                let mut local_log = MountainCarLog::default();

                for i in start..end {
                    unsafe {
                        let action = *actions.get_unchecked(i);
                        let force_direction = action - 1.0;

                        let mut velocity = *velocity_ptr.add(i);
                        let mut position = *position_ptr.add(i);

                        velocity += force_direction * FORCE + (position * 3.0).cos() * (-GRAVITY);
                        velocity = velocity.clamp(-MAX_SPEED, MAX_SPEED);
                        position += velocity;
                        position = position.clamp(MIN_POSITION, MAX_POSITION);

                        if position == MIN_POSITION && velocity < 0.0 {
                            velocity = 0.0;
                        }

                        *position_ptr.add(i) = position;
                        *velocity_ptr.add(i) = velocity;

                        let tick = *ticks_ptr.add(i) + 1;
                        *ticks_ptr.add(i) = tick;

                        let terminal = position >= GOAL_POSITION;
                        let truncated = tick >= max_steps;

                        *terminals_ptr.add(i) = terminal as u8;
                        *truncations_ptr.add(i) = truncated as u8;

                        let reward = -1.0;
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

    /// Step with automatic reset for done environments.
    pub fn step_auto_reset(&mut self, actions: &[f32]) {
        #[cfg(feature = "parallel")]
        if self.workers > 1 {
            self.step_parallel(actions);
            for i in 0..self.num_envs {
                if self.terminals[i] != 0 || self.truncations[i] != 0 {
                    self.reset_single_env(i);
                }
            }
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
            if self.terminals[i] != 0 || self.truncations[i] != 0 {
                self.reset_single_env(i);
            }
        }
    }

    /// Write observations to a flat buffer.
    pub fn write_observations(&self, buffer: &mut [f32]) {
        debug_assert!(buffer.len() >= self.num_envs * 2);

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
    /// Processes 8 environments at a time, interleaving position and velocity.
    #[cfg(feature = "simd")]
    fn write_observations_simd(&self, buffer: &mut [f32]) {
        const LANES: usize = 8;
        const OBS_DIM: usize = 2;
        let num_chunks = self.num_envs / LANES;

        for chunk in 0..num_chunks {
            let in_base = chunk * LANES;
            let out_base = chunk * LANES * OBS_DIM;

            // Load 8 positions and 8 velocities
            let pos_arr = &self.position[in_base..in_base + LANES];
            let vel_arr = &self.velocity[in_base..in_base + LANES];

            for pair in 0..4 {
                let i = pair * 2;
                let out_idx = out_base + pair * 4;

                buffer[out_idx] = pos_arr[i];
                buffer[out_idx + 1] = vel_arr[i];

                buffer[out_idx + 2] = pos_arr[i + 1];
                buffer[out_idx + 3] = vel_arr[i + 1];
            }
        }

        let remainder_start = num_chunks * LANES;
        for i in remainder_start..self.num_envs {
            let base = i * OBS_DIM;
            buffer[base] = self.position[i];
            buffer[base + 1] = self.velocity[i];
        }
    }

    /// Scalar fallback for observation writing.
    #[cfg(not(feature = "simd"))]
    fn write_observations_scalar(&self, buffer: &mut [f32]) {
        for i in 0..self.num_envs {
            buffer[i * 2] = self.position[i];
            buffer[i * 2 + 1] = self.velocity[i];
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
    pub fn get_log(&self) -> &MountainCarLog {
        &self.log
    }

    /// Clear log data.
    pub fn clear_log(&mut self) {
        self.log.clear();
    }
}

impl operant_core::Environment for MountainCar {
    fn num_envs(&self) -> usize {
        self.num_envs
    }

    fn observation_size(&self) -> usize {
        2
    }

    fn num_actions(&self) -> Option<usize> {
        Some(3)
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
        MountainCar::write_observations(self, buffer);
    }

    fn write_rewards(&self, buffer: &mut [f32]) {
        MountainCar::write_rewards(self, buffer);
    }

    fn write_terminals(&self, buffer: &mut [u8]) {
        MountainCar::write_terminals(self, buffer);
    }

    fn write_truncations(&self, buffer: &mut [u8]) {
        MountainCar::write_truncations(self, buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use operant_core::Environment;

    #[test]
    fn test_creation() {
        let env = MountainCar::with_defaults(1024);
        assert_eq!(env.num_envs(), 1024);
        assert_eq!(env.observation_size(), 2);
        assert_eq!(env.num_actions(), Some(3));
    }

    #[test]
    fn test_reset() {
        let mut env = MountainCar::with_defaults(8);
        env.reset(42);

        for i in 0..8 {
            assert!(env.position[i] >= -0.6 && env.position[i] <= -0.4);
            assert_eq!(env.velocity[i], 0.0);
            assert_eq!(env.terminals[i], 0);
            assert_eq!(env.truncations[i], 0);
            assert_eq!(env.ticks[i], 0);
        }
    }

    #[test]
    fn test_step_scalar() {
        let mut env = MountainCar::with_defaults(4);
        env.reset(0);

        let actions = vec![2.0, 0.0, 1.0, 2.0];
        env.step_scalar(&actions);

        let has_motion = env.velocity.iter().any(|&v| v.abs() > 0.0);
        assert!(has_motion);

        for i in 0..4 {
            assert_eq!(env.rewards[i], -1.0);
            assert_eq!(env.ticks[i], 1);
        }
    }

    #[test]
    fn test_auto_reset() {
        let mut env = MountainCar::new(2, 10, 0.6, 1);
        env.reset(0);

        let actions = vec![2.0, 2.0];
        for _ in 0..15 {
            env.step_auto_reset(&actions);
        }

        let any_reset = env.ticks.iter().any(|&t| t < 10);
        assert!(any_reset, "Expected at least one environment to reset");
    }

    #[test]
    fn test_write_observations() {
        let mut env = MountainCar::with_defaults(4);
        env.reset(42);

        let mut buffer = vec![0.0; 8];
        env.write_observations(&mut buffer);

        for i in 0..4 {
            assert_eq!(buffer[i * 2], env.position[i]);
            assert_eq!(buffer[i * 2 + 1], env.velocity[i]);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_matches_scalar() {
        let mut env_simd = MountainCar::with_defaults(16);
        let mut env_scalar = MountainCar::with_defaults(16);

        env_simd.reset(42);
        env_scalar.reset(42);

        let actions: Vec<f32> = (0..16).map(|i| (i % 3) as f32).collect();

        env_simd.step_simd(&actions);
        env_scalar.step_scalar(&actions);

        for i in 0..16 {
            let pos_diff = (env_simd.position[i] - env_scalar.position[i]).abs();
            let vel_diff = (env_simd.velocity[i] - env_scalar.velocity[i]).abs();
            assert!(pos_diff < 1e-5, "Position mismatch at {}: {} vs {}", i, env_simd.position[i], env_scalar.position[i]);
            assert!(vel_diff < 1e-5, "Velocity mismatch at {}: {} vs {}", i, env_simd.velocity[i], env_scalar.velocity[i]);
        }
    }

    #[test]
    fn test_initialization() {
        let mut env = MountainCar::with_defaults(128);
        assert_eq!(env.num_envs(), 128);

        // Reset to get valid initial states
        env.reset(42);

        // Check initial state bounds
        for i in 0..128 {
            assert!(env.position[i] >= -0.6 && env.position[i] <= -0.4);
            assert!(env.velocity[i].abs() < 0.001);  // Should start near zero
        }
    }

    #[test]
    fn test_reset_deterministic() {
        let mut env1 = MountainCar::with_defaults(64);
        let mut env2 = MountainCar::with_defaults(64);

        env1.reset(12345);
        env2.reset(12345);

        // Same seed should produce identical initial states
        for i in 0..64 {
            assert_eq!(env1.position[i], env2.position[i]);
            assert_eq!(env1.velocity[i], env2.velocity[i]);
        }
    }

    #[test]
    fn test_goal_detection() {
        let mut env = MountainCar::with_defaults(1);
        env.reset(42);

        // Force car to goal
        env.position[0] = 0.51;  // Just past goal position (0.5)
        env.velocity[0] = 0.0;
        env.ticks[0] = 0;  // Ensure not truncated

        // Store expected reset seed
        let expected_seed = env.rng_seeds[0];

        let actions = vec![1.0];  // Any action
        env.step_auto_reset(&actions);

        // Should have reset after reaching goal
        // Verify position is back in start range and RNG advanced
        assert!(env.position[0] >= -0.6 && env.position[0] <= -0.4,
                "Position {} should be in reset range after goal", env.position[0]);
        assert!(env.rng_seeds[0] != expected_seed, "RNG seed should have advanced after reset");
    }

    #[test]
    fn test_all_actions() {
        let mut env = MountainCar::with_defaults(3);
        env.reset(42);

        let actions = vec![0.0, 1.0, 2.0];
        env.step_auto_reset(&actions);

        assert_eq!(env.position.len(), 3);
        assert_eq!(env.velocity.len(), 3);
    }

    #[test]
    fn test_episode_logging() {
        let mut env = MountainCar::with_defaults(32);
        env.reset(0);

        let actions: Vec<f32> = vec![2.0; 32];
        for _ in 0..200 {
            env.step_auto_reset(&actions);
        }

        let log = env.get_log();
        assert!(log.total_steps > 0, "Should count steps");
    }

    #[test]
    fn test_position_bounds() {
        let mut env = MountainCar::with_defaults(16);
        env.reset(999);

        // Run steps and ensure position stays within bounds
        let actions = vec![1.0; 16];
        for _ in 0..100 {
            env.step_auto_reset(&actions);

            for i in 0..16 {
                assert!(env.position[i] >= -1.2);
                assert!(env.position[i] <= 0.6);
            }
        }
    }

    #[test]
    fn test_velocity_bounds() {
        let mut env = MountainCar::with_defaults(16);
        env.reset(42);

        let actions = vec![2.0; 16];
        for _ in 0..100 {
            env.step_auto_reset(&actions);

            for i in 0..16 {
                assert!(env.velocity[i] >= -0.07);
                assert!(env.velocity[i] <= 0.07);
            }
        }
    }

    #[test]
    fn test_observation_write() {
        let mut env = MountainCar::with_defaults(8);
        env.reset(123);

        let mut buffer = vec![0.0f32; 8 * 2];
        env.write_observations(&mut buffer);

        for i in 0..8 {
            let base = i * 2;
            assert_eq!(buffer[base], env.position[i]);
            assert_eq!(buffer[base + 1], env.velocity[i]);
        }
    }

    #[test]
    fn test_clear_log() {
        let mut env = MountainCar::with_defaults(16);
        env.reset(0);

        let actions = vec![2.0; 16];
        for _ in 0..50 {
            env.step_auto_reset(&actions);
        }

        env.clear_log();
        let log = env.get_log();
        assert_eq!(log.episode_count, 0);
        assert_eq!(log.total_reward, 0.0);
        assert_eq!(log.total_steps, 0);
    }

    #[test]
    fn test_batch_consistency() {
        let mut env = MountainCar::with_defaults(64);
        env.reset(555);

        let actions = vec![1.0; 64];
        env.step_auto_reset(&actions);

        assert_eq!(env.position.len(), 64);
        assert_eq!(env.velocity.len(), 64);
        assert_eq!(env.rewards.len(), 64);
        assert_eq!(env.terminals.len(), 64);
        assert_eq!(env.truncations.len(), 64);
    }

    #[test]
    fn test_with_workers() {
        let env = MountainCar::with_workers(1024, 4);
        assert_eq!(env.num_envs(), 1024);
        assert_eq!(env.workers(), 4);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_matches_scalar() {
        let mut scalar_env = MountainCar::with_defaults(64);
        let mut parallel_env = MountainCar::with_workers(64, 4);

        scalar_env.reset(123);
        parallel_env.reset(123);

        let actions: Vec<f32> = (0..64).map(|i| (i % 3) as f32).collect();

        scalar_env.step_scalar(&actions);
        parallel_env.step_parallel(&actions);

        for i in 0..64 {
            assert!(
                (scalar_env.position[i] - parallel_env.position[i]).abs() < 1e-5,
                "position mismatch at {}: {} vs {}",
                i,
                scalar_env.position[i],
                parallel_env.position[i]
            );
            assert!(
                (scalar_env.velocity[i] - parallel_env.velocity[i]).abs() < 1e-5,
                "velocity mismatch at {}: {} vs {}",
                i,
                scalar_env.velocity[i],
                parallel_env.velocity[i]
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_auto_reset() {
        let mut env = MountainCar::with_workers(32, 4);
        env.reset(0);

        let actions: Vec<f32> = vec![2.0; 32];
        for _ in 0..200 {
            env.step_auto_reset(&actions);
        }

        let log = env.get_log();
        assert!(log.total_steps > 0, "Should count steps");
    }
}
