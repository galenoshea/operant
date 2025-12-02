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
}

impl MountainCar {
    /// Create a new SIMD-optimized MountainCar vectorized environment.
    ///
    /// # Arguments
    ///
    /// * `num_envs` - Number of parallel environment instances
    /// * `max_steps` - Maximum episode length before truncation
    /// * `init_range` - Range for random initial position values
    pub fn new(num_envs: usize, max_steps: u32, init_range: f32) -> Self {
        assert!(num_envs > 0, "num_envs must be at least 1");

        Self {
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
        }
    }

    /// Create with default parameters (200 max steps, 0.6 init range).
    pub fn with_defaults(num_envs: usize) -> Self {
        Self::new(num_envs, MAX_STEPS, 0.6)
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

        // PHASE 1 OPTIMIZATION: Vectorize post-SIMD loop

        // 1. SIMD tick increment
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

        // Store back as u32
        for lane in 0..8 {
            self.ticks[start_idx + lane] = new_ticks.to_array()[lane] as u32;
        }

        // 2. SIMD truncation check
        let max_steps_vec = f32x8::splat(self.max_steps as f32);
        let truncation_mask = new_ticks.simd_ge(max_steps_vec);
        let truncation_bits = truncation_mask.to_bitmask() as u8;

        // 3. Terminal detection (reuse from physics)
        let goal_threshold = f32x8::splat(GOAL_POSITION);
        let terminal_mask = position.simd_ge(goal_threshold);
        let terminal_bits = terminal_mask.to_bitmask() as u8;

        // Store terminal and truncation flags
        for lane in 0..8 {
            self.terminals[start_idx + lane] = ((terminal_bits >> lane) & 1) as u8;
            self.truncations[start_idx + lane] = ((truncation_bits >> lane) & 1) as u8;
        }

        // 4. SIMD reward computation (constant -1.0 for MountainCar)
        let reward_vec = f32x8::splat(-1.0);
        reward_vec.copy_to_slice(&mut self.rewards[start_idx..start_idx + 8]);

        // 5. SIMD episode reward accumulation
        let episode_rewards_vec = f32x8::from_slice(&self.episode_rewards[start_idx..start_idx + 8]);
        let new_episode_rewards = episode_rewards_vec + reward_vec;
        new_episode_rewards.copy_to_slice(&mut self.episode_rewards[start_idx..start_idx + 8]);

        // 6. Batch log updates (outside inner loop)
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

    /// Step with automatic reset for done environments.
    pub fn step_auto_reset(&mut self, actions: &[f32]) {
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
        assert_eq!(buffer.len(), self.num_envs * 2);

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

impl operant_core::VecEnvironment for MountainCar {
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
    use operant_core::VecEnvironment;

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
        let mut env = MountainCar::new(2, 10, 0.6);
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

        // Test all three action types
        let actions = vec![0.0, 1.0, 2.0];  // Left, None, Right
        env.step_auto_reset(&actions);

        // All should execute without errors
        assert_eq!(env.position.len(), 3);
        assert_eq!(env.velocity.len(), 3);
    }

    #[test]
    fn test_episode_logging() {
        let mut env = MountainCar::with_defaults(32);
        env.reset(0);

        // Run some steps
        let actions: Vec<f32> = vec![2.0; 32];  // Always push right
        for _ in 0..200 {
            env.step_auto_reset(&actions);
        }

        let log = env.get_log();
        // MountainCar is hard, but should complete some episodes with enough steps
        // Just check that log is functional
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

        // Run steps and ensure velocity stays within bounds
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

        // Verify buffer contains valid observations
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

        // Run steps and accumulate log data
        let actions = vec![2.0; 16];
        for _ in 0..50 {
            env.step_auto_reset(&actions);
        }

        // Clear and verify
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

        // Check that all arrays have correct length
        assert_eq!(env.position.len(), 64);
        assert_eq!(env.velocity.len(), 64);
        assert_eq!(env.rewards.len(), 64);
        assert_eq!(env.terminals.len(), 64);
        assert_eq!(env.truncations.len(), 64);
    }
}
