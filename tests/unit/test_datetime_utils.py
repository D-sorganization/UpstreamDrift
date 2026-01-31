"""Unit tests for datetime_utils module.

Tests the centralized datetime utilities for timestamp formatting,
parsing, duration formatting, and time calculations.
"""
# ruff: noqa: UP017  # Use timezone.utc for Python 3.10 compatibility

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.shared.python.datetime_utils import (
    add_days,
    end_of_day,
    format_datetime,
    format_duration,
    format_timedelta,
    is_expired,
    now,
    now_local,
    now_utc,
    parse_date,
    parse_timestamp,
    start_of_day,
    time_ago,
    timestamp,
    timestamp_display,
    timestamp_filename,
    timestamp_iso,
)


class TestNowFunctions:
    """Tests for current time functions."""

    def test_now_utc_returns_utc_timezone(self) -> None:
        """now_utc should return datetime in UTC timezone."""
        dt = now_utc()
        assert dt.tzinfo is not None
        assert dt.tzinfo == timezone.utc

    def test_now_local_returns_timezone_aware(self) -> None:
        """now_local should return timezone-aware datetime."""
        dt = now_local()
        assert dt.tzinfo is not None

    def test_now_is_alias_for_now_utc(self) -> None:
        """now() should be alias for now_utc()."""
        dt1 = now()
        dt2 = now_utc()
        # Allow small time difference for execution
        assert abs((dt1 - dt2).total_seconds()) < 1


class TestTimestampFormatting:
    """Tests for timestamp formatting functions."""

    def test_timestamp_iso_format(self) -> None:
        """timestamp with iso format should return ISO 8601 string."""
        ts = timestamp("iso")
        # Should contain T separator and timezone info
        assert "T" in ts
        assert "+" in ts or "Z" in ts

    def test_timestamp_filename_format(self) -> None:
        """timestamp with filename format should be safe for filenames."""
        ts = timestamp("filename")
        # Format: YYYYMMDD_HHMMSS
        assert "_" in ts
        assert len(ts) == 15  # 8 + 1 + 6

    def test_timestamp_display_format(self) -> None:
        """timestamp with display format should be human readable."""
        ts = timestamp("display")
        # Format: YYYY-MM-DD HH:MM:SS
        assert " " in ts
        assert "-" in ts
        assert ":" in ts

    def test_timestamp_date_format(self) -> None:
        """timestamp with date format should return date only."""
        ts = timestamp("date")
        # Format: YYYY-MM-DD
        assert len(ts) == 10
        assert ts.count("-") == 2

    def test_timestamp_time_format(self) -> None:
        """timestamp with time format should return time only."""
        ts = timestamp("time")
        # Format: HH:MM:SS
        assert len(ts) == 8
        assert ts.count(":") == 2

    def test_timestamp_compact_format(self) -> None:
        """timestamp with compact format should be compact."""
        ts = timestamp("compact")
        # Format: YYYYMMDDHHMMSS
        assert len(ts) == 14
        assert ts.isdigit()

    def test_timestamp_iso_convenience(self) -> None:
        """timestamp_iso should return ISO format."""
        ts = timestamp_iso()
        assert "T" in ts

    def test_timestamp_filename_convenience(self) -> None:
        """timestamp_filename should return filename format."""
        ts = timestamp_filename()
        assert "_" in ts

    def test_timestamp_display_convenience(self) -> None:
        """timestamp_display should return display format."""
        ts = timestamp_display()
        assert " " in ts


class TestFormatDatetime:
    """Tests for format_datetime function."""

    def test_format_datetime_iso(self) -> None:
        """format_datetime with iso format should work correctly."""
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = format_datetime(dt, "iso")
        assert result == "2024-01-15T10:30:00+00:00"

    def test_format_datetime_filename(self) -> None:
        """format_datetime with filename format should work correctly."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = format_datetime(dt, "filename")
        assert result == "20240115_103000"

    def test_format_datetime_display(self) -> None:
        """format_datetime with display format should work correctly."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = format_datetime(dt, "display")
        assert result == "2024-01-15 10:30:00"

    def test_format_datetime_date(self) -> None:
        """format_datetime with date format should work correctly."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = format_datetime(dt, "date")
        assert result == "2024-01-15"

    def test_format_datetime_time(self) -> None:
        """format_datetime with time format should work correctly."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = format_datetime(dt, "time")
        assert result == "10:30:00"

    def test_format_datetime_compact(self) -> None:
        """format_datetime with compact format should work correctly."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = format_datetime(dt, "compact")
        assert result == "20240115103000"

    def test_format_datetime_invalid_format_raises(self) -> None:
        """format_datetime with invalid format should raise ValueError."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        with pytest.raises(ValueError, match="Unknown format"):
            format_datetime(dt, "invalid")  # type: ignore[arg-type]


class TestParseTimestamp:
    """Tests for parse_timestamp function."""

    def test_parse_iso_with_z(self) -> None:
        """parse_timestamp should handle ISO format with Z suffix."""
        result = parse_timestamp("2024-01-15T10:30:00Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        assert result.second == 0
        assert result.tzinfo == timezone.utc

    def test_parse_iso_with_offset(self) -> None:
        """parse_timestamp should handle ISO format with offset."""
        result = parse_timestamp("2024-01-15T10:30:00+05:30")
        assert result is not None
        assert result.year == 2024
        assert result.tzinfo is not None

    def test_parse_common_format_with_space(self) -> None:
        """parse_timestamp should handle YYYY-MM-DD HH:MM:SS format."""
        result = parse_timestamp("2024-01-15 10:30:00")
        assert result is not None
        assert result.year == 2024
        assert result.hour == 10

    def test_parse_date_only(self) -> None:
        """parse_timestamp should handle date-only format."""
        result = parse_timestamp("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 0

    def test_parse_filename_format(self) -> None:
        """parse_timestamp should handle filename format."""
        result = parse_timestamp("20240115_103000")
        assert result is not None
        assert result.year == 2024
        assert result.hour == 10

    def test_parse_empty_returns_default(self) -> None:
        """parse_timestamp with empty string should return default."""
        default = datetime(2020, 1, 1)
        result = parse_timestamp("", default=default)
        assert result == default

    def test_parse_invalid_returns_default(self) -> None:
        """parse_timestamp with invalid string should return default."""
        default = datetime(2020, 1, 1)
        result = parse_timestamp("not a date", default=default)
        assert result == default

    def test_parse_invalid_returns_none_without_default(self) -> None:
        """parse_timestamp with invalid string and no default returns None."""
        result = parse_timestamp("not a date")
        assert result is None


class TestParseDate:
    """Tests for parse_date function."""

    def test_parse_date_yyyy_mm_dd(self) -> None:
        """parse_date should handle YYYY-MM-DD format."""
        result = parse_date("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_date_with_slash(self) -> None:
        """parse_date should handle YYYY/MM/DD format."""
        result = parse_date("2024/01/15")
        assert result is not None
        assert result.year == 2024

    def test_parse_date_compact(self) -> None:
        """parse_date should handle YYYYMMDD format."""
        result = parse_date("20240115")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_format_milliseconds(self) -> None:
        """format_duration should handle sub-second durations."""
        result = format_duration(0.5, short=True)
        assert "ms" in result

    def test_format_seconds_only(self) -> None:
        """format_duration should handle seconds only."""
        result = format_duration(45, short=True)
        assert result == "45s"

    def test_format_minutes_and_seconds(self) -> None:
        """format_duration should handle minutes and seconds."""
        result = format_duration(125, short=True)
        assert "2m" in result
        assert "5s" in result

    def test_format_hours_minutes_seconds(self) -> None:
        """format_duration should handle hours, minutes, and seconds."""
        result = format_duration(3661.5, short=True)
        assert "1h" in result
        assert "1m" in result
        assert "1.5s" in result

    def test_format_long_form(self) -> None:
        """format_duration with short=False should use long form."""
        result = format_duration(65, short=False)
        assert "minute" in result
        assert "second" in result

    def test_format_negative_duration(self) -> None:
        """format_duration should handle negative durations."""
        result = format_duration(-60, short=True)
        assert result.startswith("-")


class TestFormatTimedelta:
    """Tests for format_timedelta function."""

    def test_format_timedelta_basic(self) -> None:
        """format_timedelta should format timedelta correctly."""
        td = timedelta(hours=1, minutes=30)
        result = format_timedelta(td, short=True)
        assert "1h" in result
        assert "30m" in result


class TestTimeAgo:
    """Tests for time_ago function."""

    def test_time_ago_seconds(self) -> None:
        """time_ago should handle seconds."""
        dt = now_utc() - timedelta(seconds=30)
        result = time_ago(dt)
        assert "second" in result
        assert "ago" in result

    def test_time_ago_minutes(self) -> None:
        """time_ago should handle minutes."""
        dt = now_utc() - timedelta(minutes=5)
        result = time_ago(dt)
        assert "minute" in result
        assert "ago" in result

    def test_time_ago_hours(self) -> None:
        """time_ago should handle hours."""
        dt = now_utc() - timedelta(hours=3)
        result = time_ago(dt)
        assert "hour" in result
        assert "ago" in result

    def test_time_ago_days(self) -> None:
        """time_ago should handle days."""
        dt = now_utc() - timedelta(days=5)
        result = time_ago(dt)
        assert "day" in result
        assert "ago" in result

    def test_time_ago_future(self) -> None:
        """time_ago should handle future dates."""
        dt = now_utc() + timedelta(days=1)
        result = time_ago(dt)
        assert "future" in result


class TestTimeCalculations:
    """Tests for time calculation functions."""

    def test_add_days_positive(self) -> None:
        """add_days should add positive days."""
        dt = datetime(2024, 1, 15, tzinfo=timezone.utc)
        result = add_days(dt, 5)
        assert result.day == 20

    def test_add_days_negative(self) -> None:
        """add_days should subtract negative days."""
        dt = datetime(2024, 1, 15, tzinfo=timezone.utc)
        result = add_days(dt, -5)
        assert result.day == 10

    def test_add_days_default_now(self) -> None:
        """add_days with no dt should use now_utc."""
        result = add_days(days=0)
        assert result.tzinfo == timezone.utc

    def test_start_of_day(self) -> None:
        """start_of_day should return midnight."""
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        result = start_of_day(dt)
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0
        assert result.microsecond == 0

    def test_end_of_day(self) -> None:
        """end_of_day should return 23:59:59.999999."""
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        result = end_of_day(dt)
        assert result.hour == 23
        assert result.minute == 59
        assert result.second == 59
        assert result.microsecond == 999999


class TestIsExpired:
    """Tests for is_expired function."""

    def test_is_expired_true(self) -> None:
        """is_expired should return True for expired datetime."""
        dt = now_utc() - timedelta(hours=1)
        assert is_expired(dt, ttl_seconds=1800) is True  # 30 min TTL

    def test_is_expired_false(self) -> None:
        """is_expired should return False for non-expired datetime."""
        dt = now_utc() - timedelta(minutes=5)
        assert is_expired(dt, ttl_seconds=1800) is False  # 30 min TTL

    def test_is_expired_naive_datetime(self) -> None:
        """is_expired should handle naive datetimes."""
        dt = datetime.now() - timedelta(hours=1)  # Naive datetime
        assert is_expired(dt, ttl_seconds=1800) is True
