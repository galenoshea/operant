"""Operant Models: RL algorithms with Rust-accelerated components.

This module provides high-performance reinforcement learning algorithms
that leverage Rust for performance-critical operations like rollout
storage and GAE computation.

Requires PyTorch: pip install torch

Example:
    >>> from operant.envs import CartPoleVecEnv
    >>> from operant.models import PPO
    >>>
    >>> env = CartPoleVecEnv(num_envs=8)
    >>> model = PPO(env, lr=3e-4)
    >>> model.learn(total_timesteps=100000)
    >>>
    >>> # Save and load
    >>> model.save("ppo_cartpole.pt")
    >>> loaded = PPO.load("ppo_cartpole.pt", env)
"""

from .base import Algorithm
from .exploration import ICM, RND
from .networks import (
    ActorCritic,
    ContinuousActorCritic,
    DiscreteActorCritic,
    PopArtValueHead,
)
from .ppo import PPO, PPOConfig

__all__ = [
    "Algorithm",
    "PPO",
    "PPOConfig",
    "ActorCritic",
    "DiscreteActorCritic",
    "ContinuousActorCritic",
    "PopArtValueHead",
    "RND",
    "ICM",
]
