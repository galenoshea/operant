"""Pavlov: High-performance RL environments with Rust backend.

This package provides fast, SIMD-optimized vectorized reinforcement learning
environments implemented in Rust with Python bindings.

## Quick Start

```python
import numpy as np
from pavlov.envs import CartPoleVecEnv

# Create 4096 parallel environments
env = CartPoleVecEnv(num_envs=4096)
obs = env.reset(seed=42)

# Run training loop
for step in range(1000):
    actions = np.random.randint(0, 2, size=4096, dtype=np.int32)
    obs, rewards, terminals, truncations = env.step(actions)
```

## Modules

- `pavlov.envs`: High-performance Rust-backed environments
- `pavlov.utils`: Training utilities (Logger, etc.)
"""

import sys
import warnings
from typing import Any

# Import Rust extension to register pavlov.envs in sys.modules
from . import pavlov as _pavlov_ext

# Create clean envs module facade that removes Py prefix
class _EnvsModule:
    """Environment submodule with clean class names (no Py prefix)."""

    def __init__(self):
        # Access the Rust-created pavlov.envs module
        _rust_envs = sys.modules['pavlov.envs']
        # Remove Py prefix for clean API
        self.CartPoleVecEnv = _rust_envs.PyCartPoleVecEnv
        self.MountainCarVecEnv = _rust_envs.PyMountainCarVecEnv
        self.PendulumVecEnv = _rust_envs.PyPendulumVecEnv

    def __dir__(self):
        return ["CartPoleVecEnv", "MountainCarVecEnv", "PendulumVecEnv"]

# Create utils module facade
class _UtilsModule:
    """Utilities submodule."""

    def __init__(self):
        from .logger import Logger
        self.Logger = Logger

    def __dir__(self):
        return ["Logger"]

# Create submodule instances and register in sys.modules for proper import support
envs = _EnvsModule()
utils = _UtilsModule()

# Override the Rust-created pavlov.envs with our facade that has clean names
sys.modules['pavlov.envs'] = envs
# Register utils as a proper module
sys.modules['pavlov.utils'] = utils

# Backwards compatibility - deprecated root-level imports
from .pavlov import PyCartPoleVecEnv, PyMountainCarVecEnv, PyPendulumVecEnv
from .logger import Logger

def _deprecated_import_warning(old_name: str, new_import: str) -> None:
    """Emit deprecation warning for old import patterns."""
    warnings.warn(
        f"Importing '{old_name}' from 'pavlov' is deprecated. "
        f"Use '{new_import}' instead. "
        f"Old imports will be removed in v0.4.0.",
        DeprecationWarning,
        stacklevel=3,
    )

# Override __getattr__ to warn on old usage patterns
def __getattr__(name: str) -> Any:
    if name == "PyCartPoleVecEnv":
        _deprecated_import_warning(name, "from pavlov.envs import CartPoleVecEnv")
        return PyCartPoleVecEnv
    elif name == "PyMountainCarVecEnv":
        _deprecated_import_warning(name, "from pavlov.envs import MountainCarVecEnv")
        return PyMountainCarVecEnv
    elif name == "PyPendulumVecEnv":
        _deprecated_import_warning(name, "from pavlov.envs import PendulumVecEnv")
        return PyPendulumVecEnv
    elif name == "Logger":
        _deprecated_import_warning(name, "from pavlov.utils import Logger")
        return Logger
    raise AttributeError(f"module 'pavlov' has no attribute '{name}'")

__all__ = [
    "envs",
    "utils",
    # Deprecated - for backwards compatibility only
    "PyCartPoleVecEnv",
    "PyMountainCarVecEnv",
    "PyPendulumVecEnv",
    "Logger",
]
__version__ = "0.3.0"
