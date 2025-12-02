"""Inline console and CSV logging for training metrics."""

import csv
import time
from pathlib import Path
from typing import Any, Optional, TextIO


class Logger:
    """Lightweight logger with inline console updates and optional CSV output.

    Prints training metrics on a single updating line (like a progress bar),
    with periodic newlines for history. Optionally writes all metrics to CSV.

    Supports context manager protocol for safe resource cleanup.

    Example:
        >>> from pavlov.envs import CartPoleVecEnv
        >>> from pavlov.utils import Logger
        >>>
        >>> env = CartPoleVecEnv(4096)
        >>>
        >>> # Using context manager (recommended)
        >>> with Logger(csv_path="training.csv") as logger:
        ...     for step in range(10000):
        ...         obs, rewards, dones, truncated = env.step(actions)
        ...         logs = env.get_logs()
        ...         logger.log(
        ...             steps=4096,
        ...             reward=logs['mean_reward'],
        ...             length=logs['mean_length'],
        ...         )
        ...         env.clear_logs()
        >>> # CSV file automatically closed
        >>>
        >>> # Or manual close
        >>> logger = Logger()
        >>> # ... training loop ...
        >>> logger.close()
    """

    def __init__(
        self,
        print_interval: float = 0.5,
        newline_interval: int = 10,
        csv_path: Optional[str | Path] = None,
    ) -> None:
        """Initialize the logger.

        Args:
            print_interval: Seconds between console updates (default 0.5).
                Must be positive.
            newline_interval: Number of prints before inserting a newline (default 10).
                Must be positive.
            csv_path: Optional path for CSV output file.

        Raises:
            ValueError: If print_interval or newline_interval is not positive.
        """
        if print_interval <= 0:
            raise ValueError("print_interval must be positive")
        if newline_interval <= 0:
            raise ValueError("newline_interval must be positive")

        self.print_interval: float = print_interval
        self.newline_interval: int = newline_interval
        self.last_print: float = 0.0
        self.print_count: int = 0
        self.step_count: int = 0
        self.last_step_time: float = time.time()
        self.last_step_count: int = 0

        # CSV setup
        self.csv_file: Optional[TextIO] = None
        self.csv_writer: Optional[csv.DictWriter[str, Any]] = None
        self.csv_columns: Optional[list[str]] = None
        if csv_path:
            try:
                self.csv_file = open(csv_path, "w", newline="")
            except Exception:
                # Ensure no partial state if file open fails
                self.csv_file = None
                raise

    def __enter__(self) -> "Logger":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Exit context manager and close resources."""
        self.close()
        return False  # Don't suppress exceptions

    def log(self, steps: int = 0, **metrics: Any) -> None:
        """Log metrics to console and optionally CSV.

        Args:
            steps: Number of environment steps taken since last log call.
            **metrics: Key-value pairs of metrics to log (e.g., reward=195.3).
        """
        self.step_count += steps
        now = time.time()

        if now - self.last_print < self.print_interval:
            return

        # Calculate SPS
        elapsed = now - self.last_step_time
        sps = (self.step_count - self.last_step_count) / elapsed if elapsed > 0 else 0
        self.last_step_time = now
        self.last_step_count = self.step_count

        # Format console output
        parts = [f"SPS: {sps/1e6:.1f}M"]
        for key, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{key}: {value:.2f}")
            else:
                parts.append(f"{key}: {value}")

        line = " | ".join(parts)

        # Print inline or with newline
        self.print_count += 1
        if self.print_count % self.newline_interval == 0:
            print(f"\r{line}")
        else:
            print(f"\r{line}", end="", flush=True)

        # Write to CSV
        if self.csv_file:
            row = {"timestamp": now, "sps": sps, **metrics}
            if self.csv_writer is None:
                self.csv_columns = list(row.keys())
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_columns)
                self.csv_writer.writeheader()
            self.csv_writer.writerow(row)

        self.last_print = now

    def close(self) -> None:
        """Close the CSV file if open. Call this when training is complete."""
        print()  # Final newline
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
