"""Datetime compatibility utilities.

This module centralizes datetime handling to ensure consistent
timezone-aware datetime usage across the API, addressing DRY violations.

Python 3.11+ uses datetime.UTC directly, while earlier versions need timezone.utc.
This module provides a unified interface.

Usage:
    from src.api.utils.datetime_compat import utc_now, UTC

    # Get current UTC time (always timezone-aware)
    now = utc_now()

    # Use UTC constant for explicit timezone
    dt = datetime(2026, 1, 30, tzinfo=UTC)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Final

# Python 3.11+ compatible UTC timezone
try:
    from datetime import UTC as _UTC
    UTC: Final = _UTC
except ImportError:
    # Fallback for Python < 3.11
    UTC: Final = timezone.utc  # type: ignore[misc]


def utc_now() -> datetime:
    """Get current datetime in UTC (timezone-aware).

    Returns:
        Current UTC datetime with timezone info.

    Example:
        >>> now = utc_now()
        >>> now.tzinfo is not None
        True
    """
    return datetime.now(UTC)


def utc_timestamp() -> float:
    """Get current UTC timestamp as float.

    Returns:
        Unix timestamp in seconds.
    """
    return utc_now().timestamp()


def iso_format(dt: datetime | None = None) -> str:
    """Format datetime as ISO 8601 string.

    Args:
        dt: Datetime to format. Defaults to current UTC time.

    Returns:
        ISO 8601 formatted string with Z suffix.
    """
    if dt is None:
        dt = utc_now()
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def from_timestamp(timestamp: float) -> datetime:
    """Create UTC datetime from Unix timestamp.

    Args:
        timestamp: Unix timestamp in seconds.

    Returns:
        UTC datetime with timezone info.
    """
    return datetime.fromtimestamp(timestamp, tz=UTC)


def add_minutes(dt: datetime | None, minutes: int) -> datetime:
    """Add minutes to a datetime.

    Args:
        dt: Base datetime. Defaults to current UTC time.
        minutes: Number of minutes to add.

    Returns:
        New datetime with minutes added.
    """
    if dt is None:
        dt = utc_now()
    return dt + timedelta(minutes=minutes)


def add_days(dt: datetime | None, days: int) -> datetime:
    """Add days to a datetime.

    Args:
        dt: Base datetime. Defaults to current UTC time.
        days: Number of days to add.

    Returns:
        New datetime with days added.
    """
    if dt is None:
        dt = utc_now()
    return dt + timedelta(days=days)


__all__ = [
    "UTC",
    "utc_now",
    "utc_timestamp",
    "iso_format",
    "from_timestamp",
    "add_minutes",
    "add_days",
]
