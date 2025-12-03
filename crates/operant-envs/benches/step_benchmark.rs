//! Benchmark for environment step performance.
//!
//! Run with: cargo +nightly bench --package operant-envs --features simd,parallel

#![feature(test)]

extern crate test;

use operant_envs::gymnasium::{CartPole, MountainCar, Pendulum};
use operant_core::VecEnvironment;
use test::Bencher;

const NUM_ENVS: usize = 1024;

#[bench]
fn bench_cartpole_step_1024(b: &mut Bencher) {
    let mut env = CartPole::with_defaults(NUM_ENVS);
    env.reset(42);
    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i % 2) as f32).collect();

    b.iter(|| {
        env.step_auto_reset(&actions);
    });
}

#[bench]
fn bench_cartpole_write_observations_1024(b: &mut Bencher) {
    let mut env = CartPole::with_defaults(NUM_ENVS);
    env.reset(42);
    let mut buffer = vec![0.0f32; NUM_ENVS * 4];

    b.iter(|| {
        env.write_observations(&mut buffer);
    });
}

#[bench]
fn bench_mountain_car_step_1024(b: &mut Bencher) {
    let mut env = MountainCar::with_defaults(NUM_ENVS);
    env.reset(42);
    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i % 3) as f32).collect();

    b.iter(|| {
        env.step_auto_reset(&actions);
    });
}

#[bench]
fn bench_mountain_car_write_observations_1024(b: &mut Bencher) {
    let mut env = MountainCar::with_defaults(NUM_ENVS);
    env.reset(42);
    let mut buffer = vec![0.0f32; NUM_ENVS * 2];

    b.iter(|| {
        env.write_observations(&mut buffer);
    });
}

#[bench]
fn bench_pendulum_step_1024(b: &mut Bencher) {
    let mut env = Pendulum::with_defaults(NUM_ENVS);
    env.reset(42);
    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i as f32 - 512.0) * 0.004).collect();

    b.iter(|| {
        env.step_auto_reset(&actions);
    });
}

#[bench]
fn bench_pendulum_write_observations_1024(b: &mut Bencher) {
    let mut env = Pendulum::with_defaults(NUM_ENVS);
    env.reset(42);
    let mut buffer = vec![0.0f32; NUM_ENVS * 3];

    b.iter(|| {
        env.write_observations(&mut buffer);
    });
}

// Large batch benchmarks
const LARGE_NUM_ENVS: usize = 8192;

#[bench]
fn bench_cartpole_step_8192(b: &mut Bencher) {
    let mut env = CartPole::with_defaults(LARGE_NUM_ENVS);
    env.reset(42);
    let actions: Vec<f32> = (0..LARGE_NUM_ENVS).map(|i| (i % 2) as f32).collect();

    b.iter(|| {
        env.step_auto_reset(&actions);
    });
}

#[bench]
fn bench_cartpole_write_observations_8192(b: &mut Bencher) {
    let mut env = CartPole::with_defaults(LARGE_NUM_ENVS);
    env.reset(42);
    let mut buffer = vec![0.0f32; LARGE_NUM_ENVS * 4];

    b.iter(|| {
        env.write_observations(&mut buffer);
    });
}
