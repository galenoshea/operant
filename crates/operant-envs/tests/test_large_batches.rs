use operant_envs::gymnasium::CartPole;

#[test]
fn test_state_initialization_512() {
    // Create environment with default parameters
    let env = CartPole::new(512, 500, 0.05, 1).unwrap();

    // Write observations to buffer - if state is properly initialized to 0.0,
    // observations should all be 0.0 before reset()
    let mut buffer = vec![999.9; 512 * 4];  // Sentinel value
    env.write_observations(&mut buffer);

    // All observations should be 0.0 (initialized state)
    for (i, &val) in buffer.iter().enumerate() {
        assert_eq!(val, 0.0, "Observation at index {} not initialized to 0.0", i);
    }
}

#[test]
fn test_state_initialization_2048() {
    // Create environment with default parameters
    let env = CartPole::new(2048, 500, 0.05, 1).unwrap();

    // Write observations - check ALL 2048 environments
    let mut buffer = vec![999.9; 2048 * 4];  // Sentinel value
    env.write_observations(&mut buffer);

    // All observations should be 0.0 (initialized state)
    for (i, &val) in buffer.iter().enumerate() {
        assert_eq!(val, 0.0,
            "Observation at index {} (env {}, feature {}) not initialized to 0.0",
            i, i / 4, i % 4);
    }
}

#[test]
fn test_reset_randomizes_all_envs_2048() {
    let mut env = CartPole::new(2048, 500, 0.05, 1).unwrap();
    env.reset(42);

    // Write observations after reset
    let mut buffer = vec![0.0; 2048 * 4];
    env.write_observations(&mut buffer);

    // After reset, check each environment's observations
    for env_idx in 0..2048 {
        let base = env_idx * 4;
        let x = buffer[base];
        let x_dot = buffer[base + 1];
        let theta = buffer[base + 2];
        let theta_dot = buffer[base + 3];

        // At least one value should be non-zero (randomized)
        let state_sum = x.abs() + x_dot.abs() + theta.abs() + theta_dot.abs();
        assert!(state_sum > 0.0,
            "Environment {} not randomized (all observations zero)", env_idx);

        // Verify no NaN values
        assert!(!x.is_nan(), "x is NaN at env {}", env_idx);
        assert!(!x_dot.is_nan(), "x_dot is NaN at env {}", env_idx);
        assert!(!theta.is_nan(), "theta is NaN at env {}", env_idx);
        assert!(!theta_dot.is_nan(), "theta_dot is NaN at env {}", env_idx);

        // Verify no infinite values
        assert!(!x.is_infinite(), "x is Inf at env {}", env_idx);
        assert!(!x_dot.is_infinite(), "x_dot is Inf at env {}", env_idx);
        assert!(!theta.is_infinite(), "theta is Inf at env {}", env_idx);
        assert!(!theta_dot.is_infinite(), "theta_dot is Inf at env {}", env_idx);
    }
}

#[test]
fn test_write_observations_no_nan_2048() {
    let mut env = CartPole::new(2048, 500, 0.05, 1).unwrap();
    env.reset(42);

    let mut buffer = vec![0.0; 2048 * 4];
    env.write_observations(&mut buffer);

    for (i, &val) in buffer.iter().enumerate() {
        assert!(!val.is_nan(), "NaN at buffer index {} (env {}, feature {})",
            i, i / 4, i % 4);
        assert!(!val.is_infinite(), "Inf at buffer index {}", i);
    }
}

#[test]
fn test_step_produces_valid_observations_2048() {
    let mut env = CartPole::new(2048, 500, 0.05, 1).unwrap();
    env.reset(42);

    // Take a step
    let actions = vec![1.0; 2048];
    env.step(&actions);

    // Check observations are still valid
    let mut buffer = vec![0.0; 2048 * 4];
    env.write_observations(&mut buffer);

    for (i, &val) in buffer.iter().enumerate() {
        assert!(!val.is_nan(), "NaN after step at buffer index {} (env {}, feature {})",
            i, i / 4, i % 4);
        assert!(!val.is_infinite(), "Inf after step at buffer index {}", i);
    }
}
