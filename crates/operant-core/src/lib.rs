//! Operant Core - Traits for high-performance parallel RL environments.

pub mod env;
pub mod error;

pub use env::{Environment, LogData};
pub use error::{OperantError, Result};
