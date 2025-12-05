//! Custom Environment Example - GridWalk
//!
//! This example demonstrates how to implement a custom RL environment using
//! the Operant framework. GridWalk is a simple 2D navigation task where an
//! agent must reach a goal position in a grid world.
//!
//! ## Task Description
//! - **State**: (x, y) position in grid (values normalized to [-1, 1])
//! - **Actions**: 4 discrete actions (up, down, left, right)
//! - **Reward**: +1 for reaching goal, -0.01 per step (encourage efficiency)
//! - **Episode**: Terminates when goal is reached or 100 steps elapse

use operant_core::{Environment, LogData, OperantError, Result};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// GridWalk environment - navigate to goal in 2D grid.
///
/// Demonstrates Struct-of-Arrays (SoA) memory layout for SIMD optimization.
pub struct GridWalk {
    // Environment configuration
    num_envs: usize,
    grid_size: i32,
    max_steps: usize,

    // SoA buffers - CRITICAL for performance!
    // Separate arrays for each component enable SIMD vectorization
    pos_x: Vec<i32>,        // X positions
    pos_y: Vec<i32>,        // Y positions
    goal_x: Vec<i32>,       // Goal X positions
    goal_y: Vec<i32>,       // Goal Y positions
    step_count: Vec<usize>, // Steps taken this episode

    // Pre-allocated buffers for observations (reused every step)
    observations: Vec<f32>, // Flat array: [env0_x, env0_y, env1_x, env1_y, ...]
    rewards: Vec<f32>,
    terminals: Vec<f32>,
    truncations: Vec<f32>,

    // RNG for random resets
    rng: Xoshiro256PlusPlus,

    // Episode statistics for logging
    episode_count: usize,
    episode_rewards: Vec<f32>,
    episode_lengths: Vec<usize>,
    completed_episodes: usize,
    total_reward_sum: f32,
    total_length_sum: usize,
}

impl GridWalk {
    /// Create new GridWalk environment.
    ///
    /// # Arguments
    /// * `num_envs` - Number of parallel environments
    /// * `grid_size` - Size of grid (grid_size × grid_size)
    ///
    /// # Returns
    /// `Result<Self>` - Environment or error if configuration is invalid
    pub fn new(num_envs: usize, grid_size: i32) -> Result<Self> {
        // Validate configuration
        if num_envs == 0 {
            return Err(OperantError::InvalidConfig {
                param: "num_envs".to_string(),
                message: "must be at least 1".to_string(),
            });
        }
        if grid_size < 2 {
            return Err(OperantError::InvalidConfig {
                param: "grid_size".to_string(),
                message: "must be at least 2".to_string(),
            });
        }

        Ok(Self {
            num_envs,
            grid_size,
            max_steps: 100,
            // SoA buffers
            pos_x: vec![0; num_envs],
            pos_y: vec![0; num_envs],
            goal_x: vec![0; num_envs],
            goal_y: vec![0; num_envs],
            step_count: vec![0; num_envs],
            // Pre-allocated observation buffers
            observations: vec![0.0; num_envs * 4], // 4 features per env
            rewards: vec![0.0; num_envs],
            terminals: vec![0.0; num_envs],
            truncations: vec![0.0; num_envs],
            // RNG
            rng: Xoshiro256PlusPlus::seed_from_u64(0),
            // Statistics
            episode_count: 0,
            episode_rewards: vec![0.0; num_envs],
            episode_lengths: vec![0; num_envs],
            completed_episodes: 0,
            total_reward_sum: 0.0,
            total_length_sum: 0,
        })
    }

    /// Reset a single environment.
    fn reset_single(&mut self, env_idx: usize) {
        // Random start position
        self.pos_x[env_idx] = self.rng.gen_range(0..self.grid_size);
        self.pos_y[env_idx] = self.rng.gen_range(0..self.grid_size);

        // Random goal position (different from start)
        loop {
            let gx = self.rng.gen_range(0..self.grid_size);
            let gy = self.rng.gen_range(0..self.grid_size);
            if gx != self.pos_x[env_idx] || gy != self.pos_y[env_idx] {
                self.goal_x[env_idx] = gx;
                self.goal_y[env_idx] = gy;
                break;
            }
        }

        self.step_count[env_idx] = 0;
    }

    /// Normalize position to [-1, 1] range.
    #[inline]
    fn normalize(&self, pos: i32) -> f32 {
        (pos as f32 / (self.grid_size - 1) as f32) * 2.0 - 1.0
    }
}

impl Environment for GridWalk {
    fn reset(&mut self, seed: u64) {
        // Reseed RNG
        self.rng = Xoshiro256PlusPlus::seed_from_u64(seed);

        // Reset all environments
        for i in 0..self.num_envs {
            self.reset_single(i);
        }

        // Write initial observations
        self.write_observations(&mut self.observations);
    }

    fn step(&mut self, actions: &[f32]) {
        // Clear buffers
        self.rewards.fill(0.0);
        self.terminals.fill(0.0);
        self.truncations.fill(0.0);

        // Process each environment
        for i in 0..self.num_envs {
            // Decode action: 0=up, 1=down, 2=left, 3=right
            let action = actions[i] as i32;
            match action {
                0 => self.pos_y[i] = (self.pos_y[i] - 1).max(0),
                1 => self.pos_y[i] = (self.pos_y[i] + 1).min(self.grid_size - 1),
                2 => self.pos_x[i] = (self.pos_x[i] - 1).max(0),
                3 => self.pos_x[i] = (self.pos_x[i] + 1).min(self.grid_size - 1),
                _ => {} // Invalid action, no movement
            }

            self.step_count[i] += 1;

            // Check if goal reached
            let reached_goal = self.pos_x[i] == self.goal_x[i] && self.pos_y[i] == self.goal_y[i];

            // Compute reward
            if reached_goal {
                self.rewards[i] = 1.0;
                self.terminals[i] = 1.0;
            } else {
                self.rewards[i] = -0.01; // Small penalty per step
            }

            // Check truncation (max steps)
            if self.step_count[i] >= self.max_steps {
                self.truncations[i] = 1.0;
            }

            // Track episode statistics
            self.episode_rewards[i] += self.rewards[i];
            self.episode_lengths[i] += 1;

            // Auto-reset on episode end
            if self.terminals[i] > 0.0 || self.truncations[i] > 0.0 {
                // Record statistics
                self.completed_episodes += 1;
                self.total_reward_sum += self.episode_rewards[i];
                self.total_length_sum += self.episode_lengths[i];

                // Reset episode trackers
                self.episode_rewards[i] = 0.0;
                self.episode_lengths[i] = 0;

                // Reset environment
                self.reset_single(i);
            }
        }

        // Write observations for next step
        self.write_observations(&mut self.observations);
    }

    fn write_observations(&self, buffer: &mut [f32]) {
        // SoA → interleaved format for neural network
        // Output: [env0_x, env0_y, env0_goal_x, env0_goal_y, env1_x, ...]
        for i in 0..self.num_envs {
            let offset = i * 4;
            buffer[offset] = self.normalize(self.pos_x[i]);
            buffer[offset + 1] = self.normalize(self.pos_y[i]);
            buffer[offset + 2] = self.normalize(self.goal_x[i]);
            buffer[offset + 3] = self.normalize(self.goal_y[i]);
        }
    }

    fn write_rewards(&self, buffer: &mut [f32]) {
        buffer.copy_from_slice(&self.rewards);
    }

    fn write_terminals(&self, buffer: &mut [f32]) {
        buffer.copy_from_slice(&self.terminals);
    }

    fn write_truncations(&self, buffer: &mut [f32]) {
        buffer.copy_from_slice(&self.truncations);
    }

    fn num_envs(&self) -> usize {
        self.num_envs
    }

    fn observation_dim(&self) -> usize {
        4 // (x, y, goal_x, goal_y)
    }

    fn action_dim(&self) -> usize {
        4 // 4 discrete actions
    }

    fn is_continuous(&self) -> bool {
        false
    }

    fn observation_space(&self) -> std::collections::HashMap<String, Vec<f32>> {
        let mut space = std::collections::HashMap::new();
        space.insert("shape".to_string(), vec![4.0]);
        space.insert("low".to_string(), vec![-1.0; 4]);
        space.insert("high".to_string(), vec![1.0; 4]);
        space
    }

    fn action_space(&self) -> std::collections::HashMap<String, Vec<f32>> {
        let mut space = std::collections::HashMap::new();
        space.insert("n".to_string(), vec![4.0]);
        space
    }

    fn get_logs(&self) -> LogData {
        let mut logs = std::collections::HashMap::new();

        logs.insert("episode_count".to_string(), self.completed_episodes as f32);

        if self.completed_episodes > 0 {
            let mean_reward = self.total_reward_sum / self.completed_episodes as f32;
            let mean_length = self.total_length_sum as f32 / self.completed_episodes as f32;
            logs.insert("mean_reward".to_string(), mean_reward);
            logs.insert("mean_length".to_string(), mean_length);
        } else {
            logs.insert("mean_reward".to_string(), 0.0);
            logs.insert("mean_length".to_string(), 0.0);
        }

        logs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gridwalk_creation() {
        let env = GridWalk::new(4, 10).unwrap();
        assert_eq!(env.num_envs(), 4);
        assert_eq!(env.observation_dim(), 4);
        assert_eq!(env.action_dim(), 4);
        assert!(!env.is_continuous());
    }

    #[test]
    fn test_gridwalk_invalid_config() {
        // Zero envs should fail
        assert!(GridWalk::new(0, 10).is_err());

        // Grid too small should fail
        assert!(GridWalk::new(4, 1).is_err());
    }

    #[test]
    fn test_gridwalk_reset() {
        let mut env = GridWalk::new(4, 10).unwrap();
        env.reset(42);

        let mut obs = vec![0.0; 16]; // 4 envs × 4 features
        env.write_observations(&mut obs);

        // Check observations are in valid range [-1, 1]
        for &val in &obs {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_gridwalk_step() {
        let mut env = GridWalk::new(2, 10).unwrap();
        env.reset(42);

        // Take random actions
        let actions = vec![0.0, 2.0]; // up, left
        env.step(&actions);

        let mut rewards = vec![0.0; 2];
        env.write_rewards(&mut rewards);

        // Should receive small negative reward (step penalty)
        assert!(rewards[0] < 0.0);
        assert!(rewards[1] < 0.0);
    }
}
