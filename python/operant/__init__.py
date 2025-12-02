"""Operant: High-performance RL environments with Rust backend.

This package provides fast, SIMD-optimized vectorized reinforcement learning
environments implemented in Rust with Python bindings.

## Quick Start

```python
import numpy as np
from operant.envs import CartPoleVecEnv

# Create 4096 parallel environments
env = CartPoleVecEnv(num_envs=4096)
obs, info = env.reset(seed=42)

# Run training loop
for step in range(1000):
    actions = np.random.randint(0, 2, size=4096, dtype=np.int32)
    obs, rewards, terminals, truncations, info = env.step(actions)
```

## Modules

- `operant.envs`: High-performance Rust-backed environments
- `operant.utils`: Training utilities (Logger, etc.)
"""

import sys
import warnings
from typing import Any

import numpy as np

# Import Rust extension to register operant.envs in sys.modules
from . import operant as _operant_ext

# IMPORTANT: Save reference to raw Rust envs module BEFORE we override sys.modules
_rust_envs = sys.modules['operant.envs']


# =============================================================================
# Gymnasium-compatible Space Classes
# =============================================================================

class BoxSpace:
    """Gymnasium-compatible Box space wrapper.

    Provides attribute-based access to space properties for compatibility
    with standard RL libraries that expect Gymnasium Space objects.
    """

    def __init__(self, space_dict: dict):
        self.shape = tuple(space_dict["shape"])
        self.dtype = np.dtype(space_dict["dtype"])
        self.low = np.array(space_dict["low"], dtype=self.dtype)
        self.high = np.array(space_dict["high"], dtype=self.dtype)
        self._space_dict = space_dict

    def sample(self) -> np.ndarray:
        """Sample a random value from the space."""
        return np.random.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x: np.ndarray) -> bool:
        """Check if x is within the space bounds."""
        return bool(np.all(x >= self.low) and np.all(x <= self.high))

    def __repr__(self) -> str:
        return f"BoxSpace(shape={self.shape}, dtype={self.dtype})"


class DiscreteSpace:
    """Gymnasium-compatible Discrete space wrapper.

    Provides attribute-based access to space properties for compatibility
    with standard RL libraries that expect Gymnasium Space objects.
    """

    def __init__(self, space_dict: dict):
        self.n = space_dict["n"]
        self.dtype = np.dtype(space_dict["dtype"])
        self._space_dict = space_dict

    def sample(self) -> int:
        """Sample a random action from the space."""
        return int(np.random.randint(0, self.n))

    def contains(self, x: int) -> bool:
        """Check if x is a valid action."""
        return 0 <= x < self.n

    def __repr__(self) -> str:
        return f"DiscreteSpace(n={self.n})"


# =============================================================================
# Environment Wrapper
# =============================================================================

class _VecEnvWrapper:
    """Wrapper that provides Gymnasium-compatible space objects.

    This wrapper converts the dict-based space representations from
    the Rust backend into proper Space objects with .shape, .n, etc.
    attributes that RL libraries expect.
    """

    def __init__(self, rust_env):
        self._env = rust_env
        self._obs_space = None
        self._act_space = None

    @property
    def observation_space(self) -> BoxSpace:
        """Get the observation space as a BoxSpace object."""
        if self._obs_space is None:
            self._obs_space = BoxSpace(self._env.observation_space)
        return self._obs_space

    @property
    def single_observation_space(self) -> BoxSpace:
        """Alias for observation_space (Gymnasium VecEnv compatibility)."""
        return self.observation_space

    @property
    def action_space(self):
        """Get the action space as a DiscreteSpace or BoxSpace object."""
        if self._act_space is None:
            space_dict = self._env.action_space
            if "n" in space_dict:
                self._act_space = DiscreteSpace(space_dict)
            else:
                self._act_space = BoxSpace(space_dict)
        return self._act_space

    @property
    def single_action_space(self):
        """Alias for action_space (Gymnasium VecEnv compatibility)."""
        return self.action_space

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._env.num_envs

    def reset(self, seed=None):
        """Reset all environments."""
        # Rust backend requires seed, default to 0 if not provided
        if seed is None:
            seed = 0
        return self._env.reset(seed=seed), {}

    def step(self, actions):
        """Step all environments."""
        obs, rewards, terminals, truncations = self._env.step(actions)
        return obs, rewards, terminals, truncations, {}

    def close(self):
        """Close the environments."""
        if hasattr(self._env, 'close'):
            self._env.close()

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying Rust environment."""
        return getattr(self._env, name)

    def __repr__(self) -> str:
        return f"{self._env.__class__.__name__}(num_envs={self.num_envs})"


# =============================================================================
# Environment Factory Functions
# =============================================================================

def _create_cartpole_vec_env(num_envs: int = 1, workers: int = 1) -> _VecEnvWrapper:
    """Create a vectorized CartPole environment."""
    return _VecEnvWrapper(_rust_envs.PyCartPoleVecEnv(num_envs, workers=workers))


def _create_mountaincar_vec_env(num_envs: int = 1, workers: int = 1) -> _VecEnvWrapper:
    """Create a vectorized MountainCar environment."""
    return _VecEnvWrapper(_rust_envs.PyMountainCarVecEnv(num_envs, workers=workers))


def _create_pendulum_vec_env(num_envs: int = 1, workers: int = 1) -> _VecEnvWrapper:
    """Create a vectorized Pendulum environment."""
    return _VecEnvWrapper(_rust_envs.PyPendulumVecEnv(num_envs, workers=workers))


# =============================================================================
# Module Facades
# =============================================================================

class _EnvsModule:
    """Environment submodule with clean class names and Gymnasium compatibility."""

    def __init__(self):
        pass  # Lazy initialization

    @staticmethod
    def CartPoleVecEnv(num_envs: int = 1, workers: int = 1) -> _VecEnvWrapper:
        """Create a vectorized CartPole environment."""
        return _create_cartpole_vec_env(num_envs, workers=workers)

    @staticmethod
    def MountainCarVecEnv(num_envs: int = 1, workers: int = 1) -> _VecEnvWrapper:
        """Create a vectorized MountainCar environment."""
        return _create_mountaincar_vec_env(num_envs, workers=workers)

    @staticmethod
    def PendulumVecEnv(num_envs: int = 1, workers: int = 1) -> _VecEnvWrapper:
        """Create a vectorized Pendulum environment."""
        return _create_pendulum_vec_env(num_envs, workers=workers)

    def __dir__(self):
        return ["CartPoleVecEnv", "MountainCarVecEnv", "PendulumVecEnv"]


class _UtilsModule:
    """Utilities submodule."""

    def __init__(self):
        # Try to import TUILogger from Rust (requires tui feature)
        try:
            from .operant import TUILogger
            self.TUILogger = TUILogger
            self.Logger = TUILogger  # Backwards compatibility alias
        except ImportError:
            # Fall back to Python logger if tui feature not compiled
            from .logger import Logger
            self.Logger = Logger
            self.TUILogger = None

    def __dir__(self):
        items = ["Logger"]
        if self.TUILogger is not None:
            items.append("TUILogger")
        return items


# Create submodule instances and register in sys.modules for proper import support
envs = _EnvsModule()
utils = _UtilsModule()

# Override the Rust-created operant.envs with our facade that has clean names
sys.modules['operant.envs'] = envs
# Register utils as a proper module
sys.modules['operant.utils'] = utils

# Backwards compatibility - deprecated root-level imports
from .operant import PyCartPoleVecEnv, PyMountainCarVecEnv, PyPendulumVecEnv

# Import Logger - prefer TUILogger if available
try:
    from .operant import TUILogger
    Logger = TUILogger
except ImportError:
    from .logger import Logger
    TUILogger = None


def _deprecated_import_warning(old_name: str, new_import: str) -> None:
    """Emit deprecation warning for old import patterns."""
    warnings.warn(
        f"Importing '{old_name}' from 'operant' is deprecated. "
        f"Use '{new_import}' instead. "
        f"Old imports will be removed in v0.4.0.",
        DeprecationWarning,
        stacklevel=3,
    )


# Override __getattr__ to warn on old usage patterns
def __getattr__(name: str) -> Any:
    if name == "PyCartPoleVecEnv":
        _deprecated_import_warning(name, "from operant.envs import CartPoleVecEnv")
        return PyCartPoleVecEnv
    elif name == "PyMountainCarVecEnv":
        _deprecated_import_warning(name, "from operant.envs import MountainCarVecEnv")
        return PyMountainCarVecEnv
    elif name == "PyPendulumVecEnv":
        _deprecated_import_warning(name, "from operant.envs import PendulumVecEnv")
        return PyPendulumVecEnv
    elif name == "Logger":
        _deprecated_import_warning(name, "from operant.utils import Logger")
        return Logger
    raise AttributeError(f"module 'operant' has no attribute '{name}'")


__all__ = [
    "envs",
    "utils",
    # Space classes for type hints
    "BoxSpace",
    "DiscreteSpace",
    # TUI Logger (primary)
    "TUILogger",
    "Logger",  # Alias for TUILogger
    # Deprecated - for backwards compatibility only
    "PyCartPoleVecEnv",
    "PyMountainCarVecEnv",
    "PyPendulumVecEnv",
]
__version__ = "0.3.3"
