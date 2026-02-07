"""Datetime utilities with Python 3.10/3.11+ compatibility."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Final

# Python 3.11+ compatible UTC timezone
try:
    from datetime import UTC as _UTC

    UTC: Final = _UTC
except ImportError:
    # Fallback for Python < 3.11

    UTC: Final = _UTC  # type: ignore[misc]


def utc_now() -> datetime:
    """Get current datetime in UTC (timezone-aware).

    Returns:
        Current datetime in UTC timezone.
    """
    return datetime.now(UTC)


def add_minutes(dt: datetime, minutes: int) -> datetime:
    """Add minutes to a datetime.

    Args:
        dt: Base datetime
        minutes: Number of minutes to add

    Returns:
        New datetime with minutes added.
    """
    return dt + timedelta(minutes=minutes)


def add_days(dt: datetime, days: int) -> datetime:
    """Add days to a datetime.

    Args:
        dt: Base datetime
        days: Number of days to add

    Returns:
        New datetime with days added.
    """
    return dt + timedelta(days=days)


def iso_format(dt: datetime) -> str:
    """Format datetime as ISO 8601 string.

    Args:
        dt: Datetime to format

    Returns:
        ISO 8601 formatted string.
    """
    return dt.isoformat()


__all__ = [
    "UTC",
    "utc_now",
    "add_minutes",
    "add_days",
    "iso_format",
]
