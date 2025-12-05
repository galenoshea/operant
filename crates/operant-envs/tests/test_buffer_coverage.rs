use operant_envs::gymnasium::CartPole;

#[test]
fn test_write_observations_covers_all_buffer_positions_512() {
    let mut env = CartPole::new(512, 500, 0.05, 1).unwrap();
    env.reset(42);

    // Initialize buffer with sentinel value
    let mut buffer = vec![999.9f32; 512 * 4];
    env.write_observations(&mut buffer);

    // Check if ANY positions still have sentinel value
    let sentinel_count = buffer.iter().filter(|&&v| v == 999.9).count();

    assert_eq!(sentinel_count, 0,
        "write_observations didn't write to {} positions out of {}!",
        sentinel_count, buffer.len());

    // Also check for NaN
    let nan_count = buffer.iter().filter(|&&v| v.is_nan()).count();
    assert_eq!(nan_count, 0, "Found {} NaN values", nan_count);
}

#[test]
fn test_write_observations_covers_all_buffer_positions_2048() {
    let mut env = CartPole::new(2048, 500, 0.05, 1).unwrap();
    env.reset(42);

    // Initialize buffer with sentinel value
    let mut buffer = vec![999.9f32; 2048 * 4];
    env.write_observations(&mut buffer);

    // Check if ANY positions still have sentinel value
    let sentinel_count = buffer.iter().filter(|&&v| v == 999.9).count();

    assert_eq!(sentinel_count, 0,
        "write_observations didn't write to {} positions out of {}!",
        sentinel_count, buffer.len());

    // Also check for NaN
    let nan_count = buffer.iter().filter(|&&v| v.is_nan()).count();
    assert_eq!(nan_count, 0, "Found {} NaN values", nan_count);
}
