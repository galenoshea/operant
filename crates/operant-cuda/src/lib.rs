//! GPU-accelerated RL environments using CUDA
//!
//! This crate provides GPU-resident environment implementations that eliminate
//! CPU↔GPU data transfer overhead, achieving 10-100x speedup over CPU environments.

pub mod cartpole;

pub use cartpole::CartPoleGpu;
