//! Error types for Operant RL library.

use std::fmt;

/// Result type for Operant operations.
pub type Result<T> = std::result::Result<T, OperantError>;

/// Error types that can occur in Operant.
#[derive(Debug, Clone)]
pub enum OperantError {
    /// Invalid configuration (num_envs = 0, invalid dimensions, etc.)
    InvalidConfig {
        param: String,
        message: String,
    },
    /// Buffer size mismatch
    BufferSizeMismatch {
        expected: usize,
        actual: usize,
    },
    /// Action dimension mismatch
    ActionDimensionMismatch {
        expected: usize,
        actual: usize,
    },
    /// Internal error (should not happen in correct usage)
    Internal(String),
}

impl fmt::Display for OperantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { param, message } => {
                write!(f, "Invalid configuration for '{}': {}", param, message)
            }
            Self::BufferSizeMismatch { expected, actual } => {
                write!(f, "Buffer size mismatch: expected {}, got {}", expected, actual)
            }
            Self::ActionDimensionMismatch { expected, actual } => {
                write!(f, "Action dimension mismatch: expected {}, got {}", expected, actual)
            }
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for OperantError {}

/// PyO3 integration - convert OperantError to Python exceptions
#[cfg(feature = "pyo3")]
impl From<OperantError> for pyo3::PyErr {
    fn from(err: OperantError) -> Self {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}
