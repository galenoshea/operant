# Changelog

All notable changes to Operant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-12-01

### Added
- Gymnasium-compatible `observation_space` and `action_space` properties
- Context manager support for Logger (`with Logger() as log:`)
- Comprehensive input validation for all environment methods
- Error handling tests and Logger tests

### Changed
- All environment constructors now validate `num_envs > 0`
- All `step()` methods validate action array shape and dtype
- All methods return proper Python exceptions instead of panicking
- Logger validates `print_interval` and `newline_interval` parameters

### Fixed
- Removed unsafe `.unwrap()` calls that could panic on invalid input
- Added proper error propagation with descriptive messages
- Logger file cleanup on exceptions
- API consistency across all three environments

### Security
- Replaced panic-prone unwrap() with proper error handling
- Added bounds checking for all array operations
- Improved safety documentation for remaining unsafe blocks

## [0.2.0] - 2025-12-01

### Breaking Changes

- **Package restructuring**: Environments moved to `pavlov.envs` submodule
- **Class renaming**: Removed `Py` prefix (e.g., `PyCartPoleVecEnv` → `CartPoleVecEnv`)
- Old import patterns deprecated (backwards compatible with warnings until v0.4.0)

### Added

- `operant.envs` submodule for environment classes (CartPoleVecEnv, MountainCarVecEnv, PendulumVecEnv)
- `operant.utils` submodule for utilities (Logger)
- Migration guide in README.md
- Comprehensive CHANGELOG.md

### Changed

- **Recommended import pattern**: `from operant.envs import CartPoleVecEnv` (was: `from operant import PyCartPoleVecEnv`)
- Cleaner API without `Py` prefix for better Python ergonomics
- Updated all examples and documentation to use new import patterns

### Deprecated

- Root-level imports: `from operant import PyCartPoleVecEnv`
  - Use instead: `from operant.envs import CartPoleVecEnv`
  - Old imports will be removed in v0.4.0

### Technical Details

- Rust PyO3 submodule registration for `operant.envs`
- Python facade layer for clean class name exports (removes `Py` prefix)
- All tests and benchmarks updated to new API
- Backwards compatibility maintained for smooth migration

## [0.1.0] - Previous Release

### Added

- Initial release with three Gymnasium-compatible environments:
  - CartPole-v1 (discrete actions)
  - MountainCar-v0 (discrete actions)
  - Pendulum-v1 (continuous actions)
- SIMD-optimized Rust implementations using AVX2
- Zero-copy numpy array integration via PyO3
- ~600x faster than Gymnasium for vectorized environments
- Auto-reset functionality for seamless episode transitions
- Episode logging and statistics tracking
- Struct-of-Arrays (SoA) memory layout for cache efficiency
- Python bindings with complete type annotations
