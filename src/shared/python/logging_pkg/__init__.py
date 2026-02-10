"""Logging configuration and utilities.

Source of Truth: UpstreamDrift (this repository)
Consumers: Tools (utils/logging_utils.py), Gasification_Model (internal)
Cross-repo install: pip install upstream-drift-shared

This package provides:
- logging_config: Centralized setup_logging(), get_logger(), log format constants
- logger_utils: Fallback-safe logging for standalone engine usage
"""

__all__: list[str] = []
