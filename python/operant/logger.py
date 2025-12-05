"""Fallback logger when TUI feature is not available."""


class Logger:
    """Simple fallback logger for when TUI is not compiled."""

    def __init__(self, *args, **kwargs):
        """Initialize logger (no-op)."""
        pass

    def log(self, *args, **kwargs):
        """Log message (no-op)."""
        pass

    def update(self, *args, **kwargs):
        """Update display (no-op)."""
        pass

    def close(self):
        """Close logger (no-op)."""
        pass
