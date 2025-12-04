"""Neural network architectures for RL algorithms."""

from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def layer_init(
    layer: nn.Linear, std: float = 1.0, bias_const: float = 0.0
) -> nn.Linear:
    """Initialize a linear layer with orthogonal weights."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PopArtValueHead(nn.Module):
    """PopArt: Preserving Outputs Precisely, while Adaptively Rescaling Targets.

    Adaptive normalization for value function that handles non-stationary reward scales.
    The key insight is to normalize the value target while preserving the unnormalized
    output by rescaling the linear layer weights.

    Paper: https://arxiv.org/abs/1602.07714
    """

    def __init__(self, input_dim: int, beta: float = 0.0001):
        """Initialize PopArt value head.

        Args:
            input_dim: Input feature dimension.
            beta: Exponential moving average coefficient for statistics updates.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.beta = beta

        # Running statistics (unnormalized)
        self.register_buffer('mu', torch.zeros(1))
        self.register_buffer('sigma', torch.ones(1))
        self.register_buffer('nu', torch.zeros(1))  # Second moment for variance

        # Initialize with orthogonal weights
        nn.init.orthogonal_(self.linear.weight, 1.0)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features, shape (batch, input_dim).

        Returns:
            Unnormalized value estimates, shape (batch, 1).
        """
        # Compute normalized output
        normalized_value = self.linear(x)
        # Denormalize using current statistics
        return normalized_value * self.sigma + self.mu

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize value targets using current statistics.

        Args:
            targets: Unnormalized targets, shape (batch,) or (batch, 1).

        Returns:
            Normalized targets, shape matching input.
        """
        return (targets - self.mu) / (self.sigma + 1e-8)

    @torch.no_grad()
    def update_stats(self, targets: torch.Tensor) -> None:
        """Update running statistics and rescale weights to preserve outputs.

        Args:
            targets: Batch of unnormalized value targets, shape (batch,) or (batch, 1).
        """
        targets = targets.flatten()

        # Store old statistics
        old_mu = self.mu.clone()
        old_sigma = self.sigma.clone()

        # Update first moment (mean)
        batch_mean = targets.mean()
        self.mu = (1 - self.beta) * self.mu + self.beta * batch_mean

        # Update second moment
        batch_nu = (targets ** 2).mean()
        self.nu = (1 - self.beta) * self.nu + self.beta * batch_nu

        # Compute variance and standard deviation
        variance = self.nu - self.mu ** 2
        self.sigma = torch.sqrt(torch.clamp(variance, min=1e-4))

        # Rescale weights and bias to preserve outputs
        # New output = W * (sigma_old / sigma_new) * x + (mu_old - mu_new * sigma_old / sigma_new)
        scale = old_sigma / (self.sigma + 1e-8)
        self.linear.weight.data *= scale
        self.linear.bias.data = self.linear.bias.data * scale + (old_mu - self.mu * scale)


class ActorCritic(nn.Module):
    """Base class for actor-critic networks.

    Use ActorCritic.for_env(env) to automatically create the appropriate
    network type based on the environment's action space.
    """

    @staticmethod
    def for_env(
        env: object, hidden_sizes: Sequence[int] = (64, 64)
    ) -> "ActorCritic":
        """Factory method to create appropriate network for environment.

        Args:
            env: Environment with single_observation_space and single_action_space.
            hidden_sizes: Tuple of hidden layer sizes.

        Returns:
            DiscreteActorCritic or ContinuousActorCritic based on action space.
        """
        obs_dim = env.single_observation_space.shape[0]
        act_space = env.single_action_space

        if hasattr(act_space, "n"):
            return DiscreteActorCritic(obs_dim, act_space.n, hidden_sizes)
        else:
            return ContinuousActorCritic(
                obs_dim,
                act_space.shape[0],
                hidden_sizes,
                action_low=torch.tensor(act_space.low),
                action_high=torch.tensor(act_space.high),
            )


class DiscreteActorCritic(ActorCritic):
    """Actor-critic network for discrete action spaces.

    Uses Categorical distribution for action sampling.
    Suitable for CartPole, MountainCar, and similar discrete environments.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
    ):
        """Initialize the network.

        Args:
            obs_dim: Observation space dimension.
            act_dim: Number of discrete actions.
            hidden_sizes: Sizes of hidden layers in shared network.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Build shared feature network
        layers: list[nn.Module] = []
        in_size = obs_dim
        for hidden in hidden_sizes:
            layers.append(layer_init(nn.Linear(in_size, hidden)))
            layers.append(nn.ReLU(inplace=True))  # Inplace for 5-10% speedup
            in_size = hidden
        self.shared = nn.Sequential(*layers)

        # Actor head: outputs logits for categorical distribution
        self.actor = layer_init(nn.Linear(in_size, act_dim), std=0.01)

        # Critic head: outputs single value estimate
        self.critic = layer_init(nn.Linear(in_size, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value.

        Args:
            x: Observations, shape (batch, obs_dim).

        Returns:
            Tuple of (action_logits, value) with shapes (batch, act_dim) and (batch, 1).
        """
        features = self.shared(x)
        return self.actor(features), self.critic(features)

    def act(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return policy outputs.

        Args:
            x: Observations, shape (batch, obs_dim).
            action: Optional actions for computing log_prob (for PPO update).

        Returns:
            Tuple of (action, log_prob, entropy, value).
        """
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value estimate for observations.

        Args:
            x: Observations, shape (batch, obs_dim).

        Returns:
            Value estimates, shape (batch, 1).
        """
        features = self.shared(x)
        return self.critic(features)


class ContinuousActorCritic(ActorCritic):
    """Actor-critic network for continuous action spaces.

    Uses Normal distribution with learnable log_std for action sampling.
    Suitable for Pendulum and similar continuous control environments.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        action_low: torch.Tensor | None = None,
        action_high: torch.Tensor | None = None,
        log_std_init: float = 0.0,
    ):
        """Initialize the network.

        Args:
            obs_dim: Observation space dimension.
            act_dim: Action space dimension.
            hidden_sizes: Sizes of hidden layers in shared network.
            action_low: Lower bounds for actions (for clamping).
            action_high: Upper bounds for actions (for clamping).
            log_std_init: Initial value for log standard deviation.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)

        # Build shared feature network
        layers: list[nn.Module] = []
        in_size = obs_dim
        for hidden in hidden_sizes:
            layers.append(layer_init(nn.Linear(in_size, hidden)))
            layers.append(nn.ReLU(inplace=True))  # Inplace for 5-10% speedup
            in_size = hidden
        self.shared = nn.Sequential(*layers)

        # Actor head: outputs mean of Normal distribution
        self.actor_mean = layer_init(nn.Linear(in_size, act_dim), std=0.01)

        # Learnable log standard deviation
        self.actor_log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

        # Critic head: outputs single value estimate
        self.critic = layer_init(nn.Linear(in_size, 1), std=1.0)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean, log_std, and value.

        Args:
            x: Observations, shape (batch, obs_dim).

        Returns:
            Tuple of (action_mean, action_log_std, value).
        """
        features = self.shared(x)
        return self.actor_mean(features), self.actor_log_std, self.critic(features)

    def act(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return policy outputs.

        Args:
            x: Observations, shape (batch, obs_dim).
            action: Optional actions for computing log_prob (for PPO update).

        Returns:
            Tuple of (action, log_prob, entropy, value).
        """
        mean, log_std, value = self.forward(x)
        std = log_std.exp()
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()
            # Clamp to action bounds
            if self.action_low is not None:
                action = torch.clamp(action, self.action_low, self.action_high)

        # Sum log_prob across action dimensions
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value estimate for observations.

        Args:
            x: Observations, shape (batch, obs_dim).

        Returns:
            Value estimates, shape (batch, 1).
        """
        features = self.shared(x)
        return self.critic(features)
