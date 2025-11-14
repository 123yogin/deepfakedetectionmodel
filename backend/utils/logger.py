"""
Logging utility with configurable log levels.
"""
import os
from backend.config.model_config import DEBUG_MODE, LOG_LEVEL


def debug_log(message: str):
    """Print debug message only if DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")


def info_log(message: str):
    """Print info message."""
    if LOG_LEVEL in ["DEBUG", "INFO"]:
        print(f"[INFO] {message}")


def warning_log(message: str):
    """Print warning message."""
    if LOG_LEVEL in ["DEBUG", "INFO", "WARNING"]:
        print(f"[WARNING] {message}")


def error_log(message: str):
    """Print error message (always shown)."""
    print(f"[ERROR] {message}")

