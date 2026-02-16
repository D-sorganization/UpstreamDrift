"""Tests for src.shared.python.core.datetime_utils module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.shared.python.core.datetime_utils import (
    format_datetime,
    format_duration,
    format_timedelta,
    now,
    now_local,
    now_utc,
    parse_date,
    parse_timestamp,
    timestamp,
    timestamp_display,
    timestamp_filename,
    timestamp_iso,
)


class TestNowFunctions:
    """Tests for now_utc, now_local, now."""

    def test_now_utc_has_timezone(self) -> None:
        dt = now_utc()
        assert dt.tzinfo is not None
        assert dt.tzinfo == timezone.utc

    def test_now_local_has_timezone(self) -> None:
        dt = now_local()
        assert dt.tzinfo is not None

    def test_now_is_utc(self) -> None:
        dt = now()
        assert dt.tzinfo == timezone.utc

    def test_now_utc_is_recent(self) -> None:
        dt = now_utc()
        diff = datetime.now(timezone.utc) - dt
        assert abs(diff.total_seconds()) < 2


class TestTimestamp:
    """Tests for timestamp formatting functions."""

    def test_iso_format(self) -> None:
        ts = timestamp("iso")
        assert "T" in ts  # ISO 8601 has T separator

    def test_filename_format(self) -> None:
        ts = timestamp("filename")
        # Should be YYYYMMDD_HHMMSS pattern
        assert "_" in ts
        assert len(ts) == 15  # 8 + 1 + 6

    def test_display_format(self) -> None:
        ts = timestamp("display")
        assert " " in ts  # "YYYY-MM-DD HH:MM:SS"

    def test_date_format(self) -> None:
        ts = timestamp("date")
        assert len(ts) == 10  # YYYY-MM-DD

    def test_time_format(self) -> None:
        ts = timestamp("time")
        assert ":" in ts

    def test_compact_format(self) -> None:
        ts = timestamp("compact")
        # No separators
        assert "-" not in ts
        assert ":" not in ts

    def test_timestamp_iso_function(self) -> None:
        ts = timestamp_iso()
        assert "T" in ts

    def test_timestamp_filename_function(self) -> None:
        ts = timestamp_filename()
        assert "_" in ts

    def test_timestamp_display_function(self) -> None:
        ts = timestamp_display()
        assert " " in ts


class TestFormatDatetime:
    """Tests for format_datetime."""

    def test_format_iso(self) -> None:
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = format_datetime(dt, "iso")
        assert "2024" in result

    def test_format_filename(self) -> None:
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = format_datetime(dt, "filename")
        assert "20240115" in result


class TestParseTimestamp:
    """Tests for parse_timestamp and parse_date."""

    def test_parse_iso(self) -> None:
        result = parse_timestamp("2024-01-15T10:30:00")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_date_format(self) -> None:
        result = parse_timestamp("2024-01-15")
        assert result is not None
        assert result.year == 2024

    def test_parse_invalid_returns_default(self) -> None:
        default = datetime(2000, 1, 1)
        result = parse_timestamp("not_a_date", default=default)
        assert result == default

    def test_parse_date_function(self) -> None:
        result = parse_date("2024-01-15")
        assert result is not None
        assert result.year == 2024

    def test_parse_date_invalid(self) -> None:
        result = parse_date("abc")
        assert result is None


class TestFormatDuration:
    """Tests for format_duration and format_timedelta."""

    def test_seconds_only(self) -> None:
        result = format_duration(30.5)
        assert "30" in result

    def test_minutes_and_seconds(self) -> None:
        result = format_duration(65)
        assert "1" in result  # 1 minute

    def test_hours(self) -> None:
        result = format_duration(3661.5)
        assert "1" in result  # 1 hour

    def test_zero_duration(self) -> None:
        result = format_duration(0)
        assert "0" in result

    def test_timedelta_format(self) -> None:
        td = timedelta(hours=1, minutes=30)
        result = format_timedelta(td)
        assert "1" in result
        assert "30" in result

    def test_short_format_flag(self) -> None:
        result = format_duration(65, short=False)
        # Long format should use words
        assert isinstance(result, str)
