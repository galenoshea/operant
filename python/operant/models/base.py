"""Base classes for RL algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

EnvType = TypeVar("EnvType")


class Algorithm(ABC, Generic[EnvType]):
    """Abstract base class for reinforcement learning algorithms.

    This class defines the interface that all RL algorithms must implement,
    providing a consistent API for training, prediction, and model persistence.

    Attributes:
        env: The vectorized environment instance.
        device: PyTorch device for computation ("cpu" or "cuda").
        num_envs: Number of parallel environments.
        obs_dim: Observation space dimension.
        act_dim: Action space dimension.
        is_continuous: Whether the action space is continuous.
    """

    def __init__(self, env: EnvType, device: str = "cpu"):
        """Initialize the algorithm.

        Args:
            env: Vectorized environment with Gymnasium-compatible interface.
            device: PyTorch device for computation.
        """
        self.env = env
        self.device = device
        self.num_envs = env.num_envs

        # Extract observation dimension
        obs_space = env.single_observation_space
        self.obs_dim = obs_space.shape[0]

        # Detect action space type
        act_space = env.single_action_space
        if hasattr(act_space, "n"):
            # Discrete action space
            self.is_continuous = False
            self.act_dim = act_space.n
        else:
            # Continuous action space (Box)
            self.is_continuous = True
            self.act_dim = act_space.shape[0]
            self.action_low = act_space.low
            self.action_high = act_space.high

    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        callback: Callable[[dict[str, Any]], bool] | None = None,
        log_interval: int = 1,
    ) -> "Algorithm":
        """Train the algorithm.

        Args:
            total_timesteps: Total number of environment steps to collect.
            callback: Optional callback called after each update with metrics dict.
                     Return False to stop training early.
            log_interval: Number of updates between logging.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def predict(
        self,
        observation: Any,
        deterministic: bool = False,
    ) -> tuple[Any, Any]:
        """Predict actions for given observations.

        Args:
            observation: Observation array, shape (num_envs, obs_dim) or (obs_dim,).
            deterministic: If True, return mean action (no sampling).

        Returns:
            Tuple of (actions, None). Second element reserved for future use.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model parameters to file.

        Args:
            path: File path for saving (should end in .pt).
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, env: EnvType, **kwargs: Any) -> "Algorithm":
        """Load model parameters from file.

        Args:
            path: File path to load from.
            env: Environment instance (required for proper initialization).
            **kwargs: Additional arguments passed to constructor.

        Returns:
            Loaded algorithm instance.
        """
        pass
