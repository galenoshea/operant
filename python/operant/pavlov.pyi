"""Type stubs for pavlov module.

This file provides type hints for the Rust-implemented Python extension module.
"""

from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt

class PyCartPoleVecEnv:
    """High-performance vectorized CartPole environment.

    Uses SIMD-optimized Rust implementation with zero-copy numpy arrays.

    Args:
        num_envs: Number of parallel environment instances

    Attributes:
        num_envs: Number of parallel environments
        observation_size: Size of observation vector per environment (4)
    """

    def __init__(self, num_envs: int) -> None:
        """Create a new vectorized CartPole environment.

        Args:
            num_envs: Number of parallel environment instances
        """
        ...

    def reset(self, seed: Optional[int] = None) -> npt.NDArray[np.float32]:
        """Reset all environments.

        Args:
            seed: Random seed for reproducibility (default: 0)

        Returns:
            Observations as numpy array of shape (num_envs, 4) with dtype float32.
            Each observation contains [position, velocity, angle, angular_velocity].
        """
        ...

    def step(
        self,
        actions: npt.NDArray[np.int32]
    ) -> Tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.uint8],
        npt.NDArray[np.uint8]
    ]:
        """Step all environments with the given actions.

        Args:
            actions: numpy array of shape (num_envs,) with dtype int32, values 0 or 1
                    0 = push left, 1 = push right

        Returns:
            Tuple of (observations, rewards, terminals, truncations) where:
            - observations: shape (num_envs, 4), dtype float32
            - rewards: shape (num_envs,), dtype float32
            - terminals: shape (num_envs,), dtype uint8 (1 if done, 0 otherwise)
            - truncations: shape (num_envs,), dtype uint8 (1 if truncated, 0 otherwise)
        """
        ...

    @property
    def num_envs(self) -> int:
        """Get the number of parallel environments."""
        ...

    @property
    def observation_size(self) -> int:
        """Get the observation shape per environment (always 4 for CartPole)."""
        ...

    def get_logs(self) -> dict[str, float]:
        """Get episode statistics since last clear.

        Returns:
            Dictionary with:
            - episode_count: number of completed episodes
            - total_reward: sum of rewards across all completed episodes
            - total_steps: sum of steps across all completed episodes
            - mean_reward: average episode reward (or 0 if no episodes)
            - mean_length: average episode length (or 0 if no episodes)
        """
        ...

    def clear_logs(self) -> None:
        """Clear episode statistics."""
        ...

class PyMountainCarVecEnv:
    """High-performance vectorized MountainCar environment.

    Uses SIMD-optimized Rust implementation with zero-copy numpy arrays.

    Args:
        num_envs: Number of parallel environment instances

    Attributes:
        num_envs: Number of parallel environments
        observation_size: Size of observation vector per environment (2)
    """

    def __init__(self, num_envs: int) -> None:
        """Create a new vectorized MountainCar environment.

        Args:
            num_envs: Number of parallel environment instances
        """
        ...

    def reset(self, seed: Optional[int] = None) -> npt.NDArray[np.float32]:
        """Reset all environments.

        Args:
            seed: Random seed for reproducibility (default: 0)

        Returns:
            Observations as numpy array of shape (num_envs, 2) with dtype float32.
            Each observation contains [position, velocity].
        """
        ...

    def step(
        self,
        actions: npt.NDArray[np.int32]
    ) -> Tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.uint8],
        npt.NDArray[np.uint8]
    ]:
        """Step all environments with the given actions.

        Args:
            actions: numpy array of shape (num_envs,) with dtype int32, values 0, 1, or 2
                    0 = push left, 1 = no push, 2 = push right

        Returns:
            Tuple of (observations, rewards, terminals, truncations) where:
            - observations: shape (num_envs, 2), dtype float32
            - rewards: shape (num_envs,), dtype float32
            - terminals: shape (num_envs,), dtype uint8 (1 if done, 0 otherwise)
            - truncations: shape (num_envs,), dtype uint8 (1 if truncated, 0 otherwise)
        """
        ...

    @property
    def num_envs(self) -> int:
        """Get the number of parallel environments."""
        ...

    @property
    def observation_size(self) -> int:
        """Get the observation shape per environment (always 2 for MountainCar)."""
        ...

    def get_logs(self) -> dict[str, float]:
        """Get episode statistics since last clear.

        Returns:
            Dictionary with:
            - episode_count: number of completed episodes
            - total_reward: sum of rewards across all completed episodes
            - total_steps: sum of steps across all completed episodes
            - mean_reward: average episode reward (or 0 if no episodes)
            - mean_length: average episode length (or 0 if no episodes)
        """
        ...

    def clear_logs(self) -> None:
        """Clear episode statistics."""
        ...

class PyPendulumVecEnv:
    """High-performance vectorized Pendulum environment.

    Uses SIMD-optimized Rust implementation with zero-copy numpy arrays.

    Args:
        num_envs: Number of parallel environment instances

    Attributes:
        num_envs: Number of parallel environments
        observation_size: Size of observation vector per environment (3)
    """

    def __init__(self, num_envs: int) -> None:
        """Create a new vectorized Pendulum environment.

        Args:
            num_envs: Number of parallel environment instances
        """
        ...

    def reset(self, seed: Optional[int] = None) -> npt.NDArray[np.float32]:
        """Reset all environments.

        Args:
            seed: Random seed for reproducibility (default: 0)

        Returns:
            Observations as numpy array of shape (num_envs, 3) with dtype float32.
            Each observation contains [cos(theta), sin(theta), angular_velocity].
        """
        ...

    def step(
        self,
        actions: npt.NDArray[np.float32]
    ) -> Tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.uint8],
        npt.NDArray[np.uint8]
    ]:
        """Step all environments with the given actions.

        Args:
            actions: numpy array of shape (num_envs,) with dtype float32
                    Continuous torque values in range [-2.0, 2.0]

        Returns:
            Tuple of (observations, rewards, terminals, truncations) where:
            - observations: shape (num_envs, 3), dtype float32 (cos(theta), sin(theta), theta_dot)
            - rewards: shape (num_envs,), dtype float32
            - terminals: shape (num_envs,), dtype uint8 (always 0, Pendulum has no terminal state)
            - truncations: shape (num_envs,), dtype uint8 (1 if truncated, 0 otherwise)
        """
        ...

    @property
    def num_envs(self) -> int:
        """Get the number of parallel environments."""
        ...

    @property
    def observation_size(self) -> int:
        """Get the observation shape per environment (always 3 for Pendulum)."""
        ...

    def get_logs(self) -> dict[str, float]:
        """Get episode statistics since last clear.

        Returns:
            Dictionary with:
            - episode_count: number of completed episodes
            - total_reward: sum of rewards across all completed episodes
            - total_steps: sum of steps across all completed episodes
            - mean_reward: average episode reward (or 0 if no episodes)
            - mean_length: average episode length (or 0 if no episodes)
        """
        ...

    def clear_logs(self) -> None:
        """Clear episode statistics."""
        ...
