"""Integration tests for RolloutBuffer and GAE computation."""

import numpy as np
import pytest
import torch
from operant._rl import RolloutBuffer


def test_rollout_buffer_basic_usage():
    """Test basic RolloutBuffer creation and usage."""
    num_envs, num_steps = 4, 8
    obs_dim, action_dim = 4, 1

    # Create buffer
    buffer = RolloutBuffer(num_envs, num_steps, obs_dim, action_dim, False)

    # Add data
    for _ in range(num_steps):
        for _ in range(num_envs):
            obs = np.zeros(obs_dim, dtype=np.float32)
            action = np.zeros(action_dim, dtype=np.float32)
            reward = 1.0
            value = 0.5
            done = 0.0
            buffer.add(obs, action, reward, value, done)

    # Compute GAE
    last_vals = np.zeros(num_envs, dtype=np.float32)
    buffer.compute_gae(last_vals, gamma=0.99, gae_lambda=0.95)

    # Retrieve results
    advantages = buffer.get_advantages()
    returns = buffer.get_returns()

    assert advantages.shape == (num_envs * num_steps,)
    assert returns.shape == (num_envs * num_steps,)
    assert advantages.dtype == np.float32
    assert returns.dtype == np.float32


def test_gae_correctness_with_termination():
    """Test GAE computation with episode termination against manual numpy calculation."""
    num_envs, num_steps = 2, 3
    obs_dim, action_dim = 4, 1

    buffer = RolloutBuffer(num_envs, num_steps, obs_dim, action_dim, False)

    # Create test scenario: env 0 terminates at t=1
    rewards = []
    values = []
    dones = []

    for t in range(num_steps):
        for e in range(num_envs):
            obs = np.zeros(obs_dim, dtype=np.float32)
            action = np.zeros(action_dim, dtype=np.float32)
            reward = 1.0
            value = 0.5
            done = 1.0 if (e == 0 and t == 1) else 0.0

            buffer.add(obs, action, reward, value, done)

            rewards.append(reward)
            values.append(value)
            dones.append(done)

    # Compute GAE
    last_vals = np.array([0.5, 0.5], dtype=np.float32)
    gamma = 0.99
    gae_lambda = 0.95

    buffer.compute_gae(last_vals, gamma, gae_lambda)
    advantages = buffer.get_advantages()

    # Manual numpy calculation for verification
    rewards = np.array(rewards, dtype=np.float32).reshape(num_steps, num_envs)
    values = np.array(values, dtype=np.float32).reshape(num_steps, num_envs)
    dones = np.array(dones, dtype=np.float32).reshape(num_steps, num_envs)

    # Compute GAE manually
    manual_advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
    last_gae = np.zeros(num_envs, dtype=np.float32)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_value = last_vals
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        manual_advantages[t] = last_gae

    # Compare
    manual_advantages_flat = manual_advantages.T.flatten()  # Flatten in env-major order
    np.testing.assert_allclose(
        advantages,
        manual_advantages_flat,
        rtol=1e-4,
        err_msg="GAE computation doesn't match manual calculation"
    )


def test_batch_vs_incremental_add():
    """Test that add() and add_batch() produce identical results."""
    num_envs, num_steps = 4, 8
    obs_dim, action_dim = 4, 2

    buffer_incremental = RolloutBuffer(num_envs, num_steps, obs_dim, action_dim, False)
    buffer_batch = RolloutBuffer(num_envs, num_steps, obs_dim, action_dim, False)

    # Generate random test data
    np.random.seed(42)
    obs_data = np.random.randn(num_steps, num_envs, obs_dim).astype(np.float32)
    action_data = np.random.randn(num_steps, num_envs, action_dim).astype(np.float32)
    reward_data = np.random.randn(num_steps, num_envs).astype(np.float32)
    value_data = np.random.randn(num_steps, num_envs).astype(np.float32)
    done_data = np.random.randint(0, 2, (num_steps, num_envs)).astype(np.float32)

    # Add incrementally
    for t in range(num_steps):
        for e in range(num_envs):
            buffer_incremental.add(
                obs_data[t, e],
                action_data[t, e],
                reward_data[t, e],
                value_data[t, e],
                done_data[t, e]
            )

    # Add in batch
    for t in range(num_steps):
        buffer_batch.add_batch(
            obs_data[t],
            action_data[t],
            reward_data[t],
            value_data[t],
            done_data[t]
        )

    # Compute GAE on both
    last_vals = np.random.randn(num_envs).astype(np.float32)
    buffer_incremental.compute_gae(last_vals, gamma=0.99, gae_lambda=0.95)
    buffer_batch.compute_gae(last_vals, gamma=0.99, gae_lambda=0.95)

    # Compare results
    adv_incremental = buffer_incremental.get_advantages()
    adv_batch = buffer_batch.get_advantages()

    np.testing.assert_array_equal(
        adv_incremental,
        adv_batch,
        err_msg="Incremental and batch add methods produce different results"
    )


def test_ppo_integration_after_fix():
    """Integration test: verify PPO training shows learning after GAE fix."""
    from operant.envs import CartPoleVecEnv
    from operant.models import PPO

    # Small-scale test
    num_envs = 128
    total_timesteps = 50_000

    env = CartPoleVecEnv(num_envs=num_envs)

    model = PPO(
        env,
        lr=2.5e-4,
        batch_size=512,
        device="cpu",  # Use CPU for test reliability
        use_amp=False,
    )

    # Track rewards
    rewards = []

    def callback(metrics):
        rewards.append(metrics.get('mean_reward', 0))
        # Continue training
        return True

    # Train
    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=1)

    env.close()

    # Verify learning occurred
    # With correct GAE, CartPole should show improvement even in 50K steps
    # We expect final reward > initial reward (learning is happening)
    initial_avg = np.mean(rewards[:3]) if len(rewards) >= 3 else rewards[0]
    final_avg = np.mean(rewards[-3:]) if len(rewards) >= 3 else rewards[-1]

    assert final_avg > initial_avg, \
        f"No learning detected: initial={initial_avg:.2f}, final={final_avg:.2f}. GAE fix may not be working."

    # Additionally, we expect some meaningful reward (> 30) by the end
    assert final_avg > 30, \
        f"Final reward too low: {final_avg:.2f}. Expected > 30 with correct GAE."
