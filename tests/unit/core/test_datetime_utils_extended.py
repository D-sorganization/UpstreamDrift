"""Extended tests for src.shared.python.core.datetime_utils module.

Covers time calculations (time_ago, add_days, start_of_day, end_of_day, is_expired),
additional parsing formats, format_duration edge cases, and format_datetime error paths.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.shared.python.core.datetime_utils import (
    add_days,
    end_of_day,
    format_datetime,
    format_duration,
    is_expired,
    now_utc,
    parse_date,
    parse_timestamp,
    start_of_day,
    time_ago,
    timestamp,
)

# ============================================================================
# Tests for time_ago
# ============================================================================


class TestTimeAgo:
    """Tests for time_ago relative time formatting."""

    @pytest.mark.parametrize(
        "delta, expected_exact",
        [
            (timedelta(seconds=1), "1 second ago"),
            (timedelta(minutes=1), "1 minute ago"),
            (timedelta(minutes=5), "5 minutes ago"),
            (timedelta(hours=1), "1 hour ago"),
            (timedelta(hours=3), "3 hours ago"),
            (timedelta(days=1), "1 day ago"),
            (timedelta(days=7), "7 days ago"),
        ],
        ids=["1s", "1m", "5m", "1h", "3h", "1d", "7d"],
    )
    def test_exact_time_ago(self, delta: timedelta, expected_exact: str) -> None:
        """time_ago should return exact string for known deltas."""
        dt = now_utc() - delta
        assert time_ago(dt) == expected_exact

    @pytest.mark.parametrize(
        "delta, expected_unit",
        [
            (timedelta(seconds=30), "second"),
            (timedelta(days=90), "month"),
            (timedelta(days=400), "year"),
        ],
        ids=["30s", "90d-months", "400d-years"],
    )
    def test_unit_in_time_ago(self, delta: timedelta, expected_unit: str) -> None:
        """time_ago should contain the correct unit for various deltas."""
        dt = now_utc() - delta
        result = time_ago(dt)
        assert expected_unit in result
        assert "ago" in result

    def test_future_time(self) -> None:
        dt = now_utc() + timedelta(hours=1)
        result = time_ago(dt)
        assert "future" in result

    def test_naive_datetime_treated_as_utc(self) -> None:
        """Naive datetime (no tzinfo) should be treated as UTC."""
        # Use utcnow-like naive datetime to avoid local timezone offset
        dt = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)
        result = time_ago(dt)
        assert "minute" in result


# ============================================================================
# Tests for add_days
# ============================================================================


class TestAddDays:
    """Tests for add_days utility."""

    def test_add_positive_days(self) -> None:
        base = datetime(2024, 1, 15, tzinfo=timezone.utc)
        result = add_days(base, days=5)
        assert result.day == 20

    def test_add_negative_days(self) -> None:
        base = datetime(2024, 1, 15, tzinfo=timezone.utc)
        result = add_days(base, days=-5)
        assert result.day == 10

    def test_default_base_is_now(self) -> None:
        result = add_days(days=0)
        diff = abs((now_utc() - result).total_seconds())
        assert diff < 2

    def test_add_zero_days(self) -> None:
        base = datetime(2024, 6, 15, tzinfo=timezone.utc)
        result = add_days(base, days=0)
        assert result == base

    def test_month_boundary(self) -> None:
        base = datetime(2024, 1, 30, tzinfo=timezone.utc)
        result = add_days(base, days=3)
        assert result.month == 2
        assert result.day == 2


# ============================================================================
# Tests for start_of_day
# ============================================================================


class TestStartOfDay:
    """Tests for start_of_day utility."""

    def test_sets_midnight(self) -> None:
        dt = datetime(2024, 6, 15, 14, 30, 45, 123456, tzinfo=timezone.utc)
        result = start_of_day(dt)
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0
        assert result.microsecond == 0

    def test_preserves_date(self) -> None:
        dt = datetime(2024, 6, 15, 14, 30, 45, tzinfo=timezone.utc)
        result = start_of_day(dt)
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15

    def test_default_uses_now(self) -> None:
        result = start_of_day()
        assert result.hour == 0
        assert result.minute == 0

    def test_already_midnight(self) -> None:
        dt = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        result = start_of_day(dt)
        assert result == dt


# ============================================================================
# Tests for end_of_day
# ============================================================================


class TestEndOfDay:
    """Tests for end_of_day utility."""

    def test_sets_end_of_day(self) -> None:
        dt = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = end_of_day(dt)
        assert result.hour == 23
        assert result.minute == 59
        assert result.second == 59
        assert result.microsecond == 999999

    def test_preserves_date(self) -> None:
        dt = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = end_of_day(dt)
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15

    def test_default_uses_now(self) -> None:
        result = end_of_day()
        assert result.hour == 23
        assert result.minute == 59


# ============================================================================
# Tests for is_expired
# ============================================================================


class TestIsExpired:
    """Tests for is_expired utility."""

    def test_expired(self) -> None:
        dt = now_utc() - timedelta(hours=1)
        assert is_expired(dt, ttl_seconds=1800) is True  # 30 min TTL

    def test_not_expired(self) -> None:
        dt = now_utc() - timedelta(seconds=10)
        assert is_expired(dt, ttl_seconds=3600) is False  # 1 hour TTL

    def test_just_expired(self) -> None:
        dt = now_utc() - timedelta(seconds=61)
        assert is_expired(dt, ttl_seconds=60) is True

    def test_naive_datetime(self) -> None:
        """Naive datetime should be handled gracefully."""
        dt = datetime.now() - timedelta(hours=2)  # noqa: DTZ005
        assert is_expired(dt, ttl_seconds=3600) is True

    def test_zero_ttl(self) -> None:
        dt = now_utc() - timedelta(seconds=1)
        assert is_expired(dt, ttl_seconds=0) is True


# ============================================================================
# Extended format_duration tests
# ============================================================================


class TestFormatDurationExtended:
    """Extended tests for format_duration edge cases."""

    def test_negative_duration(self) -> None:
        result = format_duration(-30)
        assert result.startswith("-")

    def test_sub_second_short(self) -> None:
        result = format_duration(0.5, short=True)
        assert "ms" in result

    def test_sub_second_long(self) -> None:
        result = format_duration(0.25, short=False)
        assert "milliseconds" in result

    def test_fractional_seconds_short(self) -> None:
        result = format_duration(3661.5, short=True)
        assert "h" in result
        assert "m" in result

    @pytest.mark.parametrize(
        "seconds, expected_unit",
        [
            (1, "second"),
            (60, "minute"),
            (3600, "hour"),
        ],
        ids=["1s-singular", "1m-singular", "1h-singular"],
    )
    def test_singular_units(self, seconds: int, expected_unit: str) -> None:
        """Long-form format should use singular unit names."""
        result = format_duration(seconds, short=False)
        assert expected_unit in result

    def test_precision_control(self) -> None:
        result = format_duration(1.123456, precision=3)
        assert "1.123" in result


# ============================================================================
# Extended parse_timestamp tests
# ============================================================================


class TestParseTimestampExtended:
    """Extended tests for parse_timestamp edge cases."""

    def test_empty_string(self) -> None:
        assert parse_timestamp("") is None

    def test_empty_string_with_default(self) -> None:
        default = datetime(2000, 1, 1)
        assert parse_timestamp("", default=default) == default

    def test_iso_with_z_suffix(self) -> None:
        result = parse_timestamp("2024-01-15T10:30:00Z")
        assert result is not None
        assert result.tzinfo is not None

    def test_compact_format(self) -> None:
        result = parse_timestamp("20240115_103000")
        assert result is not None
        assert result.year == 2024

    def test_slash_date_format(self) -> None:
        result = parse_timestamp("2024/01/15")
        assert result is not None
        assert result.year == 2024

    def test_custom_formats(self) -> None:
        result = parse_timestamp("15-Jan-2024", formats=["%d-%b-%Y"])
        assert result is not None
        assert result.year == 2024

    def test_parse_date_compact(self) -> None:
        result = parse_date("20240115")
        assert result is not None
        assert result.year == 2024

    def test_parse_date_slash(self) -> None:
        result = parse_date("2024/01/15")
        assert result is not None
        assert result.month == 1


# ============================================================================
# Extended format_datetime tests
# ============================================================================


class TestFormatDatetimeExtended:
    """Extended tests for format_datetime."""

    @pytest.mark.parametrize(
        "fmt, expected",
        [
            ("display", "2024-06-15 14:30:00"),
            ("date", "2024-06-15"),
            ("time", "14:30:00"),
            ("compact", "20240615143000"),
        ],
        ids=["display", "date", "time", "compact"],
    )
    def test_format_datetime_formats(self, fmt: str, expected: str) -> None:
        """format_datetime should produce the correct string for each format."""
        dt = datetime(2024, 6, 15, 14, 30, 0)
        assert format_datetime(dt, fmt) == expected

    def test_invalid_format_raises(self) -> None:
        dt = datetime(2024, 1, 15)
        with pytest.raises(ValueError, match="Unknown format"):
            format_datetime(dt, "invalid_format")  # type: ignore[arg-type]


# ============================================================================
# Extended timestamp tests
# ============================================================================


class TestTimestampExtended:
    """Extended tests for timestamp convenience functions."""

    def test_local_time_iso(self) -> None:
        ts = timestamp("iso", utc=False)
        assert "T" in ts

    def test_local_time_filename(self) -> None:
        ts = timestamp("filename", utc=False)
        assert "_" in ts
