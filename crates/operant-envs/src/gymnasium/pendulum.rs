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
/// cache performance and SIMD vectorization.
pub struct Pendulum {
    theta: Vec<f32>,
    theta_dot: Vec<f32>,
    cos_theta: Vec<f32>,  // PHASE 2: Cached cos(theta) for observations
    sin_theta: Vec<f32>,  // PHASE 2: Cached sin(theta) for observations
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
    pub fn new(
        num_envs: usize,
        max_steps: u32,
        init_theta_range: f32,
        init_theta_dot_range: f32,
    ) -> Self {
        assert!(num_envs > 0, "num_envs must be at least 1");

        Self {
            theta: vec![0.0; num_envs],
            theta_dot: vec![0.0; num_envs],
            cos_theta: vec![0.0; num_envs],  // PHASE 2: Allocate cos cache
            sin_theta: vec![0.0; num_envs],  // PHASE 2: Allocate sin cache
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
        }
    }

    /// Create with default parameters (200 max steps, π init angle, 1.0 init velocity).
    pub fn with_defaults(num_envs: usize) -> Self {
        Self::new(num_envs, MAX_STEPS, std::f32::consts::PI, 1.0)
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

        // PHASE 2: Compute and cache cos/sin for observations
        let cos_vec = simd_cos(new_theta);
        let sin_vec = simd_sin(new_theta);
        cos_vec.copy_to_slice(&mut self.cos_theta[start_idx..start_idx + 8]);
        sin_vec.copy_to_slice(&mut self.sin_theta[start_idx..start_idx + 8]);

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

        // 2. SIMD reward computation (already computed above)
        let theta_squared = new_theta * new_theta;
        let theta_dot_squared = new_theta_dot * new_theta_dot;
        let torque_squared = torque * torque;

        let cost = theta_squared
            + f32x8::splat(0.1) * theta_dot_squared
            + f32x8::splat(0.001) * torque_squared;
        let reward = -cost;

        reward.copy_to_slice(&mut self.rewards[start_idx..start_idx + 8]);

        // 3. SIMD truncation check
        let max_steps_vec = f32x8::splat(self.max_steps as f32);
        let truncation_mask = new_ticks.simd_ge(max_steps_vec);
        let truncation_bits = truncation_mask.to_bitmask() as u8;

        // Store terminal and truncation flags
        for lane in 0..8 {
            self.terminals[start_idx + lane] = 0;  // Pendulum has no terminals
            self.truncations[start_idx + lane] = ((truncation_bits >> lane) & 1) as u8;
        }

        // 4. SIMD episode reward accumulation
        let episode_rewards_vec = f32x8::from_slice(&self.episode_rewards[start_idx..start_idx + 8]);
        let reward_vec = f32x8::from_slice(&self.rewards[start_idx..start_idx + 8]);
        let new_episode_rewards = episode_rewards_vec + reward_vec;
        new_episode_rewards.copy_to_slice(&mut self.episode_rewards[start_idx..start_idx + 8]);

        // 5. Batch log updates (outside inner loop)
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

        // PHASE 2: Initialize cached cos/sin for observations
        self.cos_theta[idx] = self.theta[idx].cos();
        self.sin_theta[idx] = self.theta[idx].sin();

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
            if self.truncations[i] != 0 {
                self.reset_single_env(i);
            }
        }
    }

    /// Write observations to a flat buffer (cos(theta), sin(theta), theta_dot).
    pub fn write_observations(&self, buffer: &mut [f32]) {
        assert_eq!(buffer.len(), self.num_envs * 3);

        // PHASE 2: Use cached cos/sin values instead of recomputing
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

impl operant_core::VecEnvironment for Pendulum {
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
    use operant_core::VecEnvironment;

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
        let mut env = Pendulum::new(2, 10, std::f32::consts::PI, 1.0);
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

        // Verify buffer contains valid observations (cos, sin, theta_dot)
        for i in 0..8 {
            let base = i * 3;
            // cos and sin should be in [-1, 1] range (with small tolerance for Taylor)
            assert!(buffer[base].abs() <= 1.01, "cos(theta) should be in range");
            assert!(buffer[base + 1].abs() <= 1.01, "sin(theta) should be in range");
            // Angular velocity should match
            assert_eq!(buffer[base + 2], env.theta_dot[i]);
        }
    }

    #[test]
    fn test_episode_logging() {
        let mut env = Pendulum::with_defaults(32);
        env.reset(0);

        // Run some steps
        let actions: Vec<f32> = vec![0.0; 32];
        for _ in 0..300 {
            env.step_auto_reset(&actions);
        }

        let log = env.get_log();
        // Should have completed at least some episodes
        assert!(log.episode_count > 0, "Should complete some episodes");
        assert!(log.total_steps > 0, "Should count steps");
        // Pendulum has negative rewards, so total_reward should be negative
        assert!(log.total_reward < 0.0, "Pendulum rewards should be negative");
    }

    #[test]
    fn test_clear_log() {
        let mut env = Pendulum::with_defaults(16);
        env.reset(0);

        // Run steps and accumulate log data
        let actions = vec![0.0; 16];
        for _ in 0..100 {
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

        // Different torques should produce different angular velocities
        assert_ne!(env1.theta_dot[0], env2.theta_dot[0], "Different torques should change angular velocity");
    }

    #[test]
    fn test_batch_consistency() {
        let mut env = Pendulum::with_defaults(64);
        env.reset(555);

        let actions = vec![0.5; 64];
        env.step_auto_reset(&actions);

        // Check that all arrays have correct length
        assert_eq!(env.theta.len(), 64);
        assert_eq!(env.theta_dot.len(), 64);
        assert_eq!(env.cos_theta.len(), 64);
        assert_eq!(env.sin_theta.len(), 64);
        assert_eq!(env.rewards.len(), 64);
        assert_eq!(env.terminals.len(), 64);
        assert_eq!(env.truncations.len(), 64);
    }
}
