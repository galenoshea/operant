//! Pavlov Environments - High-performance RL environment implementations
//!
//! This crate provides SIMD-optimized Rust implementations of Gymnasium-compatible
//! reinforcement learning environments with SoA memory layout.

#![cfg_attr(feature = "simd", feature(portable_simd))]

pub mod shared;
pub mod gymnasium;

pub use gymnasium::{
    CartPole, CartPoleLog, MountainCar, MountainCarLog, Pendulum, PendulumLog,
};
