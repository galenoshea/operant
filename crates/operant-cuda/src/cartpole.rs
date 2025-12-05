//! GPU-accelerated CartPole environment
//!
//! All state and computation remains on GPU, eliminating CPU↔GPU transfer overhead.

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// GPU-resident CartPole environment
///
/// Keeps all state (observations, rewards, terminals) on GPU memory.
/// Physics simulation runs entirely on GPU using CUDA kernels.
pub struct CartPoleGpu {
    device: Arc<CudaDevice>,
    num_envs: usize,

    // CUDA kernels
    reset_kernel: CudaFunction,
    step_kernel: CudaFunction,

    // GPU memory buffers
    states: CudaSlice<f32>,        // [num_envs, 4] - (x, x_dot, theta, theta_dot)
    rng_states: CudaSlice<u32>,    // [num_envs] - RNG state per environment
    rewards: CudaSlice<f32>,       // [num_envs]
    terminals: CudaSlice<f32>,     // [num_envs]
    truncations: CudaSlice<f32>,   // [num_envs]
}

impl CartPoleGpu {
    /// Create a new GPU-resident CartPole environment
    ///
    /// # Arguments
    /// * `num_envs` - Number of parallel environments
    /// * `device_id` - CUDA device ID (default: 0)
    pub fn new(num_envs: usize, device_id: usize) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize CUDA device
        let device = CudaDevice::new(device_id)?;

        // Load compiled PTX from build
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/cartpole.ptx"));
        device.load_ptx(ptx.into(), "cartpole", &["cartpole_reset", "cartpole_step"])?;

        let reset_kernel = device.get_func("cartpole", "cartpole_reset").unwrap();
        let step_kernel = device.get_func("cartpole", "cartpole_step").unwrap();

        // Allocate GPU memory
        let states = device.alloc_zeros::<f32>(num_envs * 4)?;
        let rewards = device.alloc_zeros::<f32>(num_envs)?;
        let terminals = device.alloc_zeros::<f32>(num_envs)?;
        let truncations = device.alloc_zeros::<f32>(num_envs)?;

        // Initialize RNG states with random seeds
        let mut rng_seeds = vec![0u32; num_envs];
        for (i, seed) in rng_seeds.iter_mut().enumerate() {
            *seed = (i as u32).wrapping_mul(1103515245).wrapping_add(12345);
        }
        let rng_states = device.htod_copy(rng_seeds)?;

        Ok(Self {
            device,
            num_envs,
            reset_kernel,
            step_kernel,
            states,
            rng_states,
            rewards,
            terminals,
            truncations,
        })
    }

    /// Reset all environments
    ///
    /// Initializes state to random values in [-0.05, 0.05]
    /// All computation happens on GPU.
    pub fn reset(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = LaunchConfig::for_num_elems(self.num_envs as u32);

        let params = (
            &self.states,
            &mut self.rng_states,
            self.num_envs as i32,
        );

        unsafe {
            self.reset_kernel.clone().launch(cfg, params)?;
        }

        self.device.synchronize()?;
        Ok(())
    }

    /// Step all environments with given actions
    ///
    /// # Arguments
    /// * `actions` - GPU buffer of discrete actions (0=left, 1=right)
    ///
    /// All computation happens on GPU. No CPU↔GPU transfer.
    pub fn step(&mut self, actions: &CudaSlice<i32>) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(actions.len(), self.num_envs, "Actions length must match num_envs");

        let cfg = LaunchConfig::for_num_elems(self.num_envs as u32);

        let params = (
            &mut self.states,
            actions,
            &mut self.rewards,
            &mut self.terminals,
            &mut self.truncations,
            self.num_envs as i32,
        );

        unsafe {
            self.step_kernel.clone().launch(cfg, params)?;
        }

        self.device.synchronize()?;
        Ok(())
    }

    /// Get current observations (requires GPU→CPU transfer)
    ///
    /// This should only be called when needed for training.
    /// For pure environment stepping, keep everything on GPU.
    pub fn get_obs(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(self.device.dtoh_sync_copy(&self.states)?)
    }

    /// Get rewards (requires GPU→CPU transfer)
    pub fn get_rewards(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(self.device.dtoh_sync_copy(&self.rewards)?)
    }

    /// Get terminals (requires GPU→CPU transfer)
    pub fn get_terminals(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(self.device.dtoh_sync_copy(&self.terminals)?)
    }

    /// Get truncations (requires GPU→CPU transfer)
    pub fn get_truncations(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(self.device.dtoh_sync_copy(&self.truncations)?)
    }

    /// Get GPU device reference (for zero-copy integration with PyTorch)
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get GPU buffer references (for zero-copy operations)
    pub fn gpu_buffers(&self) -> GpuBuffers {
        GpuBuffers {
            states: &self.states,
            rewards: &self.rewards,
            terminals: &self.terminals,
            truncations: &self.truncations,
        }
    }
}

/// GPU buffer references for zero-copy operations
pub struct GpuBuffers<'a> {
    pub states: &'a CudaSlice<f32>,
    pub rewards: &'a CudaSlice<f32>,
    pub terminals: &'a CudaSlice<f32>,
    pub truncations: &'a CudaSlice<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Only run if CUDA is available
    fn test_cartpole_gpu_basic() {
        let num_envs = 128;
        let mut env = CartPoleGpu::new(num_envs, 0).unwrap();

        // Reset
        env.reset().unwrap();
        let obs = env.get_obs().unwrap();
        assert_eq!(obs.len(), num_envs * 4);

        // Check states are in valid range [-0.05, 0.05]
        for &val in obs.iter() {
            assert!(val >= -0.05 && val <= 0.05, "Initial state out of range: {}", val);
        }

        // Step with random actions
        let actions: Vec<i32> = (0..num_envs).map(|i| (i % 2) as i32).collect();
        let actions_gpu = env.device.htod_copy(actions).unwrap();

        env.step(&actions_gpu).unwrap();

        let rewards = env.get_rewards().unwrap();
        assert_eq!(rewards.len(), num_envs);

        // Initial rewards should all be 1.0 (not done yet)
        for &r in rewards.iter() {
            assert!(r == 0.0 || r == 1.0, "Invalid reward: {}", r);
        }
    }

    #[test]
    #[ignore]
    fn test_cartpole_gpu_performance() {
        use std::time::Instant;

        let num_envs = 8192;
        let num_steps = 1000;

        let mut env = CartPoleGpu::new(num_envs, 0).unwrap();
        env.reset().unwrap();

        // Prepare actions on GPU
        let actions: Vec<i32> = vec![1; num_envs];
        let actions_gpu = env.device.htod_copy(actions).unwrap();

        // Benchmark
        let start = Instant::now();
        for _ in 0..num_steps {
            env.step(&actions_gpu).unwrap();
        }
        let elapsed = start.elapsed();

        let total_steps = num_envs * num_steps;
        let sps = total_steps as f64 / elapsed.as_secs_f64();

        println!("GPU CartPole: {:.0} SPS ({} envs, {} steps)", sps, num_envs, num_steps);

        // Should be much faster than CPU (target: 10M+ SPS)
        assert!(sps > 1_000_000.0, "GPU performance too low: {:.0} SPS", sps);
    }
}
