"""Type stubs for logger module.

This file provides type hints for the Logger class.
"""

from typing import Any, Optional
from pathlib import Path

class Logger:
    """Lightweight logger with inline console updates and optional CSV output.

    Prints training metrics on a single updating line (like a progress bar),
    with periodic newlines for history. Optionally writes all metrics to CSV.

    Attributes:
        print_interval: Seconds between console updates
        newline_interval: Number of prints before inserting a newline
        last_print: Timestamp of last print
        print_count: Number of times printed
        step_count: Total number of steps logged
        last_step_time: Timestamp when steps were last counted
        last_step_count: Step count at last timing measurement
        csv_file: Open file handle for CSV output (if enabled)
        csv_writer: CSV DictWriter instance (if enabled)
        csv_columns: Column names for CSV (if enabled)
    """

    print_interval: float
    newline_interval: int
    last_print: float
    print_count: int
    step_count: int
    last_step_time: float
    last_step_count: int
    csv_file: Optional[Any]  # TextIOWrapper
    csv_writer: Optional[Any]  # csv.DictWriter
    csv_columns: Optional[list[str]]

    def __init__(
        self,
        print_interval: float = 0.5,
        newline_interval: int = 10,
        csv_path: Optional[str | Path] = None,
    ) -> None:
        """Initialize the logger.

        Args:
            print_interval: Seconds between console updates (default 0.5)
            newline_interval: Number of prints before inserting a newline (default 10)
            csv_path: Optional path for CSV output file
        """
        ...

    def log(self, steps: int = 0, **metrics: Any) -> None:
        """Log metrics to console and optionally CSV.

        Args:
            steps: Number of environment steps taken since last log call
            **metrics: Key-value pairs of metrics to log (e.g., reward=195.3)
        """
        ...

    def close(self) -> None:
        """Close the CSV file if open. Call this when training is complete."""
        ...
