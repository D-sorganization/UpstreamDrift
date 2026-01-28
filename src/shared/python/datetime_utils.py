"""Centralized datetime utilities for the Golf Modeling Suite.

This module consolidates common datetime/timestamp patterns across the codebase,
addressing DRY violations identified in Pragmatic Programmer reviews.

Usage:
    from src.shared.python.datetime_utils import (
        now_utc,
        now_local,
        timestamp_iso,
        timestamp_filename,
        parse_timestamp,
        format_duration,
    )

    # Get current time
    utc_time = now_utc()
    local_time = now_local()

    # Get formatted timestamps
    iso = timestamp_iso()  # "2024-01-15T10:30:00.123456+00:00"
    filename = timestamp_filename()  # "20240115_103000"

    # Parse timestamps
    dt = parse_timestamp("2024-01-15T10:30:00Z")

    # Format durations
    duration = format_duration(3661.5)  # "1h 1m 1.5s"
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

# ============================================================================
# Current Time Functions
# ============================================================================


def now_utc() -> datetime:
    """Get current time in UTC.

    Returns:
        Current datetime in UTC timezone.

    Example:
        >>> dt = now_utc()
        >>> dt.tzinfo
        datetime.timezone.utc
    """
    return datetime.now(UTC)


def now_local() -> datetime:
    """Get current time in local timezone.

    Returns:
        Current datetime in local timezone.

    Example:
        >>> dt = now_local()
        >>> dt.tzinfo is not None
        True
    """
    return datetime.now().astimezone()


def now() -> datetime:
    """Get current time (alias for now_utc).

    Returns:
        Current datetime in UTC timezone.
    """
    return now_utc()


# ============================================================================
# Timestamp Formatting
# ============================================================================


TimestampFormat = Literal["iso", "filename", "display", "date", "time", "compact"]


def timestamp(fmt: TimestampFormat = "iso", *, utc: bool = True) -> str:
    """Get current timestamp in specified format.

    Args:
        fmt: Timestamp format:
            - "iso": ISO 8601 format (2024-01-15T10:30:00.123456+00:00)
            - "filename": Safe for filenames (20240115_103000)
            - "display": Human readable (2024-01-15 10:30:00)
            - "date": Date only (2024-01-15)
            - "time": Time only (10:30:00)
            - "compact": Compact format (20240115103000)
        utc: Use UTC time (default) or local time.

    Returns:
        Formatted timestamp string.

    Example:
        >>> timestamp("filename")
        "20240115_103000"
    """
    dt = now_utc() if utc else now_local()
    return format_datetime(dt, fmt)


def timestamp_iso(*, utc: bool = True) -> str:
    """Get current timestamp in ISO 8601 format.

    Args:
        utc: Use UTC time (default) or local time.

    Returns:
        ISO 8601 formatted timestamp.

    Example:
        >>> timestamp_iso()
        "2024-01-15T10:30:00.123456+00:00"
    """
    return timestamp("iso", utc=utc)


def timestamp_filename(*, utc: bool = True) -> str:
    """Get current timestamp safe for filenames.

    Args:
        utc: Use UTC time (default) or local time.

    Returns:
        Filename-safe timestamp (YYYYMMDD_HHMMSS).

    Example:
        >>> timestamp_filename()
        "20240115_103000"
    """
    return timestamp("filename", utc=utc)


def timestamp_display(*, utc: bool = True) -> str:
    """Get current timestamp for display.

    Args:
        utc: Use UTC time (default) or local time.

    Returns:
        Human-readable timestamp.

    Example:
        >>> timestamp_display()
        "2024-01-15 10:30:00"
    """
    return timestamp("display", utc=utc)


def format_datetime(dt: datetime, fmt: TimestampFormat = "iso") -> str:
    """Format a datetime object to string.

    Args:
        dt: Datetime object to format.
        fmt: Output format (see timestamp() for options).

    Returns:
        Formatted datetime string.

    Example:
        >>> dt = datetime(2024, 1, 15, 10, 30, 0)
        >>> format_datetime(dt, "filename")
        "20240115_103000"
    """
    formats = {
        "iso": lambda d: d.isoformat(),
        "filename": lambda d: d.strftime("%Y%m%d_%H%M%S"),
        "display": lambda d: d.strftime("%Y-%m-%d %H:%M:%S"),
        "date": lambda d: d.strftime("%Y-%m-%d"),
        "time": lambda d: d.strftime("%H:%M:%S"),
        "compact": lambda d: d.strftime("%Y%m%d%H%M%S"),
    }

    formatter = formats.get(fmt)
    if formatter is None:
        raise ValueError(f"Unknown format: {fmt!r}. Valid: {list(formats.keys())}")

    return str(formatter(dt))


# ============================================================================
# Timestamp Parsing
# ============================================================================


def parse_timestamp(
    value: str,
    *,
    formats: list[str] | None = None,
    default: datetime | None = None,
) -> datetime | None:
    """Parse a timestamp string to datetime.

    Args:
        value: Timestamp string to parse.
        formats: List of formats to try (uses common formats if None).
        default: Default value if parsing fails.

    Returns:
        Parsed datetime or default.

    Example:
        >>> parse_timestamp("2024-01-15T10:30:00Z")
        datetime.datetime(2024, 1, 15, 10, 30, tzinfo=datetime.timezone.utc)
        >>> parse_timestamp("2024-01-15")
        datetime.datetime(2024, 1, 15, 0, 0)
    """
    if not value:
        return default

    # Try ISO format first (most common)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        pass

    # Try common formats
    common_formats = formats or [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%Y%m%d_%H%M%S",
        "%Y%m%d%H%M%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ]

    for fmt in common_formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return default


def parse_date(value: str, default: datetime | None = None) -> datetime | None:
    """Parse a date string to datetime.

    Args:
        value: Date string to parse.
        default: Default value if parsing fails.

    Returns:
        Parsed datetime (time set to midnight) or default.

    Example:
        >>> parse_date("2024-01-15")
        datetime.datetime(2024, 1, 15, 0, 0)
    """
    return parse_timestamp(
        value,
        formats=[
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y%m%d",
        ],
        default=default,
    )


# ============================================================================
# Duration Formatting
# ============================================================================


def format_duration(
    seconds: float,
    *,
    precision: int = 1,
    short: bool = False,
) -> str:
    """Format a duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds.
        precision: Decimal places for sub-second precision.
        short: Use short format (h/m/s vs hours/minutes/seconds).

    Returns:
        Formatted duration string.

    Example:
        >>> format_duration(3661.5)
        "1h 1m 1.5s"
        >>> format_duration(65, short=False)
        "1 minute 5 seconds"
    """
    if seconds < 0:
        return f"-{format_duration(-seconds, precision=precision, short=short)}"

    if seconds < 1:
        ms = seconds * 1000
        if short:
            return f"{ms:.{precision}f}ms"
        return f"{ms:.{precision}f} milliseconds"

    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []

    if hours >= 1:
        h = int(hours)
        if short:
            parts.append(f"{h}h")
        else:
            parts.append(f"{h} hour{'s' if h != 1 else ''}")

    if minutes >= 1:
        m = int(minutes)
        if short:
            parts.append(f"{m}m")
        else:
            parts.append(f"{m} minute{'s' if m != 1 else ''}")

    if secs > 0 or not parts:
        if secs == int(secs):
            s = int(secs)
            if short:
                parts.append(f"{s}s")
            else:
                parts.append(f"{s} second{'s' if s != 1 else ''}")
        else:
            if short:
                parts.append(f"{secs:.{precision}f}s")
            else:
                parts.append(f"{secs:.{precision}f} seconds")

    return " ".join(parts)


def format_timedelta(
    td: timedelta,
    *,
    precision: int = 1,
    short: bool = False,
) -> str:
    """Format a timedelta to human-readable string.

    Args:
        td: Timedelta to format.
        precision: Decimal places for sub-second precision.
        short: Use short format (h/m/s vs hours/minutes/seconds).

    Returns:
        Formatted duration string.

    Example:
        >>> format_timedelta(timedelta(hours=1, minutes=30))
        "1h 30m"
    """
    return format_duration(td.total_seconds(), precision=precision, short=short)


# ============================================================================
# Time Calculations
# ============================================================================


def time_ago(dt: datetime) -> str:
    """Format a datetime as relative time from now.

    Args:
        dt: Datetime to format.

    Returns:
        Relative time string (e.g., "5 minutes ago").

    Example:
        >>> time_ago(now_utc() - timedelta(minutes=5))
        "5 minutes ago"
    """
    now = now_utc()

    # Ensure both are timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    diff = now - dt

    if diff < timedelta(seconds=0):
        return "in the future"

    if diff < timedelta(seconds=60):
        secs = int(diff.total_seconds())
        return f"{secs} second{'s' if secs != 1 else ''} ago"

    if diff < timedelta(hours=1):
        mins = int(diff.total_seconds() / 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"

    if diff < timedelta(days=1):
        hours = int(diff.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"

    if diff < timedelta(days=30):
        days = diff.days
        return f"{days} day{'s' if days != 1 else ''} ago"

    if diff < timedelta(days=365):
        months = diff.days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"

    years = diff.days // 365
    return f"{years} year{'s' if years != 1 else ''} ago"


def add_days(dt: datetime | None = None, days: int = 0) -> datetime:
    """Add days to a datetime.

    Args:
        dt: Base datetime (default: now_utc()).
        days: Number of days to add (can be negative).

    Returns:
        New datetime with days added.
    """
    if dt is None:
        dt = now_utc()
    return dt + timedelta(days=days)


def start_of_day(dt: datetime | None = None) -> datetime:
    """Get the start of day (midnight) for a datetime.

    Args:
        dt: Datetime (default: now_utc()).

    Returns:
        Datetime at midnight (00:00:00).
    """
    if dt is None:
        dt = now_utc()
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def end_of_day(dt: datetime | None = None) -> datetime:
    """Get the end of day (23:59:59.999999) for a datetime.

    Args:
        dt: Datetime (default: now_utc()).

    Returns:
        Datetime at end of day.
    """
    if dt is None:
        dt = now_utc()
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def is_expired(dt: datetime, ttl_seconds: float) -> bool:
    """Check if a datetime has expired based on TTL.

    Args:
        dt: Datetime to check.
        ttl_seconds: Time-to-live in seconds.

    Returns:
        True if expired.

    Example:
        >>> created = now_utc() - timedelta(hours=1)
        >>> is_expired(created, ttl_seconds=1800)  # 30 min TTL
        True
    """
    now = now_utc()

    # Ensure both are timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    return (now - dt).total_seconds() > ttl_seconds
