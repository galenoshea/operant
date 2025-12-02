//! Vectorized environment traits for high-performance RL.

use std::fmt::Debug;

/// Log data that environments can return for tracking metrics.
pub trait LogData: Clone + Debug + Default {
    /// Merge another log into this one (for aggregation).
    fn merge(&mut self, other: &Self);

    /// Clear/reset the log counters.
    fn clear(&mut self);

    /// Get the number of episodes recorded.
    fn episode_count(&self) -> f32;
}

/// Trait for vectorized environments with SoA memory layout.
pub trait VecEnvironment {
    /// Returns the number of parallel environments.
    fn num_envs(&self) -> usize;

    /// Returns the observation size per environment.
    fn observation_size(&self) -> usize;

    /// Returns the number of discrete actions, or None for continuous.
    fn num_actions(&self) -> Option<usize>;

    /// Reset all environments with deterministic seeding.
    fn reset(&mut self, seed: u64);

    /// Step all environments (includes auto-reset for done envs).
    fn step(&mut self, actions: &[f32]);

    /// Write observations to buffer.
    fn write_observations(&self, buffer: &mut [f32]);

    /// Write rewards to buffer.
    fn write_rewards(&self, buffer: &mut [f32]);

    /// Write terminal flags to buffer.
    fn write_terminals(&self, buffer: &mut [u8]);

    /// Write truncation flags to buffer.
    fn write_truncations(&self, buffer: &mut [u8]);
}
