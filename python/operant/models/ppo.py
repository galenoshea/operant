"""Proximal Policy Optimization (PPO) algorithm."""

import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from operant._rl import RolloutBuffer

from .base import Algorithm
from .networks import ActorCritic


class PPO(Algorithm):
    """Proximal Policy Optimization with clipped objective.

    This implementation uses a Rust-backed rollout buffer for efficient
    storage and GAE computation, with PyTorch for neural network training.

    Supports both discrete (CartPole, MountainCar) and continuous (Pendulum)
    action spaces, automatically detecting the appropriate network architecture.

    Example:
        >>> from operant.envs import CartPoleVecEnv
        >>> from operant.models import PPO
        >>>
        >>> env = CartPoleVecEnv(num_envs=8)
        >>> model = PPO(env, lr=3e-4, n_steps=128)
        >>> model.learn(total_timesteps=100000)
    """

    def __init__(
        self,
        env: Any,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        n_steps: int = 128,
        n_epochs: int = 4,
        batch_size: int = 256,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        network_class: type[ActorCritic] | None = None,
        network_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize PPO.

        Args:
            env: Vectorized environment with Gymnasium-compatible interface.
            lr: Learning rate for optimizer.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
            clip_eps: PPO clipping parameter.
            n_steps: Number of steps per rollout.
            n_epochs: Number of optimization epochs per update.
            batch_size: Mini-batch size for optimization.
            vf_coef: Value function loss coefficient.
            ent_coef: Entropy bonus coefficient.
            max_grad_norm: Maximum gradient norm for clipping.
            device: PyTorch device ("cpu" or "cuda").
            network_class: Optional custom network class (must have act/get_value).
            network_kwargs: Additional kwargs for network constructor.
        """
        super().__init__(env, device)

        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        # Create network
        if network_class is not None:
            kwargs = network_kwargs or {}
            self.policy = network_class(self.obs_dim, self.act_dim, **kwargs)
        else:
            self.policy = ActorCritic.for_env(env)
        self.policy = self.policy.to(device)

        # Optimizer
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr)

        # Create Rust rollout buffer
        act_dim_for_buffer = self.act_dim if self.is_continuous else 1
        self.buffer = RolloutBuffer(
            num_envs=self.num_envs,
            num_steps=n_steps,
            obs_dim=self.obs_dim,
            act_dim=act_dim_for_buffer,
            is_continuous=self.is_continuous,
        )

        # Training state
        self.total_timesteps = 0
        self._last_obs: torch.Tensor | None = None
        self._start_time: float | None = None

    def learn(
        self,
        total_timesteps: int,
        callback: Callable[[dict[str, Any]], bool] | None = None,
        log_interval: int = 1,
    ) -> "PPO":
        """Train PPO for specified timesteps.

        Args:
            total_timesteps: Total environment steps to collect.
            callback: Optional callback(metrics) -> bool, return False to stop.
            log_interval: Updates between logging (used with Logger).

        Returns:
            Self for method chaining.
        """
        self._start_time = time.time()

        # Initialize environment
        obs, _ = self.env.reset()
        self._last_obs = torch.from_numpy(np.asarray(obs)).float().to(self.device)

        num_updates = total_timesteps // (self.n_steps * self.num_envs)

        for update in range(num_updates):
            # Collect rollouts
            self._collect_rollouts()

            # Update policy
            metrics = self._update()

            # Compute timing
            self.total_timesteps = (update + 1) * self.n_steps * self.num_envs
            elapsed = time.time() - self._start_time
            sps = self.total_timesteps / elapsed if elapsed > 0 else 0

            # Get environment logs
            env_logs = self.env.get_logs()

            # Combine metrics
            metrics.update(
                {
                    "timesteps": self.total_timesteps,
                    "sps": sps,
                    "episodes": int(env_logs.get("episode_count", 0)),
                    "mean_reward": env_logs.get("mean_reward", 0),
                    "mean_length": env_logs.get("mean_length", 0),
                }
            )

            # Callback
            if callback is not None:
                if callback(metrics) is False:
                    break

        return self

    def _collect_rollouts(self) -> None:
        """Collect n_steps of experience and store in buffer."""
        self.buffer.reset()

        with torch.no_grad():
            for _ in range(self.n_steps):
                # Get action from policy
                action, log_prob, _, value = self.policy.act(self._last_obs)

                # Convert action to numpy for environment
                if self.is_continuous:
                    action_np = action.cpu().numpy()
                else:
                    action_np = action.cpu().numpy().astype(np.int32)

                # Step environment
                next_obs, reward, term, trunc, _ = self.env.step(action_np)
                done = term | trunc

                # Prepare data for Rust buffer
                obs_np = (
                    self._last_obs.cpu()
                    .numpy()
                    .reshape(self.num_envs, -1)
                    .astype(np.float32)
                )
                act_np = action.cpu().numpy().astype(np.float32)
                if not self.is_continuous:
                    act_np = act_np.astype(np.float32)
                rew_np = np.asarray(reward, dtype=np.float32)
                done_np = np.asarray(done, dtype=np.float32)
                val_np = value.squeeze(-1).cpu().numpy().astype(np.float32)
                logp_np = log_prob.cpu().numpy().astype(np.float32)

                # Store transition in Rust buffer
                self.buffer.add(obs_np, act_np, rew_np, done_np, val_np, logp_np)

                self._last_obs = (
                    torch.from_numpy(np.asarray(next_obs)).float().to(self.device)
                )

        # Compute GAE using Rust
        with torch.no_grad():
            last_value = self.policy.get_value(self._last_obs)
            last_value_np = last_value.squeeze(-1).cpu().numpy().astype(np.float32)

        self.buffer.compute_gae(last_value_np, self.gamma, self.gae_lambda)

    def _update(self) -> dict[str, float]:
        """Perform PPO update using collected rollouts."""
        # Get data from Rust buffer
        b_obs, b_actions, b_log_probs, b_advantages, b_returns = self.buffer.get_all()

        # Convert to torch tensors
        b_obs = torch.from_numpy(np.asarray(b_obs)).to(self.device)
        b_actions = torch.from_numpy(np.asarray(b_actions)).to(self.device)
        b_log_probs = torch.from_numpy(np.asarray(b_log_probs)).to(self.device)
        b_advantages = torch.from_numpy(np.asarray(b_advantages)).to(self.device)
        b_returns = torch.from_numpy(np.asarray(b_returns)).to(self.device)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # For discrete actions, convert to long
        if not self.is_continuous:
            b_actions = b_actions.long()

        total_samples = self.n_steps * self.num_envs
        indices = np.arange(total_samples)

        # Metrics accumulators
        pg_losses: list[float] = []
        v_losses: list[float] = []
        entropies: list[float] = []

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, total_samples, self.batch_size):
                idx = indices[start : start + self.batch_size]

                # Get new policy outputs
                _, new_log_prob, entropy, new_value = self.policy.act(
                    b_obs[idx], b_actions[idx]
                )

                # Compute ratio
                ratio = (new_log_prob - b_log_probs[idx]).exp()

                # Clipped surrogate objective
                pg_loss1 = -b_advantages[idx] * ratio
                pg_loss2 = -b_advantages[idx] * torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = ((new_value.squeeze() - b_returns[idx]) ** 2).mean()

                # Entropy loss
                ent_loss = entropy.mean()

                # Total loss
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(ent_loss.item())

        return {
            "policy_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def predict(
        self,
        observation: Any,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, None]:
        """Predict action for observation.

        Args:
            observation: Observation array.
            deterministic: If True, return mean action.

        Returns:
            Tuple of (action, None).
        """
        obs = torch.from_numpy(np.asarray(observation)).float().to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                if self.is_continuous:
                    mean, _, _ = self.policy.forward(obs)
                    action = mean
                else:
                    logits, _ = self.policy.forward(obs)
                    action = logits.argmax(dim=-1)
            else:
                action, _, _, _ = self.policy.act(obs)

        return action.cpu().numpy(), None

    def save(self, path: str) -> None:
        """Save model to file.

        Args:
            path: File path (should end in .pt).
        """
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_timesteps": self.total_timesteps,
                "config": {
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "clip_eps": self.clip_eps,
                    "n_steps": self.n_steps,
                    "n_epochs": self.n_epochs,
                    "batch_size": self.batch_size,
                    "vf_coef": self.vf_coef,
                    "ent_coef": self.ent_coef,
                    "max_grad_norm": self.max_grad_norm,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env: Any, **kwargs: Any) -> "PPO":
        """Load model from file.

        Args:
            path: File path to load.
            env: Environment instance.
            **kwargs: Override config values.

        Returns:
            Loaded PPO instance.
        """
        checkpoint = torch.load(path)
        config = checkpoint["config"]
        config.update(kwargs)

        model = cls(env, **config)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.total_timesteps = checkpoint["total_timesteps"]

        return model
