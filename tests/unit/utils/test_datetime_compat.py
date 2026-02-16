"""Tests for src.api.utils.datetime_compat module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.api.utils.datetime_compat import UTC, add_days, add_minutes, iso_format, utc_now


class TestUTC:
    """Tests for UTC constant."""

    def test_utc_is_timezone(self) -> None:
        assert UTC is not None

    def test_utc_offset_is_zero(self) -> None:
        dt = datetime.now(UTC)
        assert dt.utcoffset() == timedelta(0)


class TestUtcNow:
    """Tests for utc_now function."""

    def test_returns_datetime(self) -> None:
        result = utc_now()
        assert isinstance(result, datetime)

    def test_is_timezone_aware(self) -> None:
        result = utc_now()
        assert result.tzinfo is not None

    def test_utc_offset_is_zero(self) -> None:
        result = utc_now()
        assert result.utcoffset() == timedelta(0)


class TestAddMinutes:
    """Tests for add_minutes function."""

    def test_add_positive_minutes(self) -> None:
        base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = add_minutes(base, 30)
        assert result.minute == 30

    def test_add_negative_minutes(self) -> None:
        base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = add_minutes(base, -30)
        assert result.minute == 30
        assert result.hour == 11

    def test_add_zero_minutes(self) -> None:
        base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = add_minutes(base, 0)
        assert result == base


class TestAddDays:
    """Tests for add_days function."""

    def test_add_positive_days(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        result = add_days(base, 5)
        assert result.day == 6

    def test_add_negative_days(self) -> None:
        base = datetime(2026, 1, 10, tzinfo=UTC)
        result = add_days(base, -5)
        assert result.day == 5

    def test_crosses_month_boundary(self) -> None:
        base = datetime(2026, 1, 30, tzinfo=UTC)
        result = add_days(base, 5)
        assert result.month == 2
        assert result.day == 4


class TestIsoFormat:
    """Tests for iso_format function."""

    def test_returns_string(self) -> None:
        dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = iso_format(dt)
        assert isinstance(result, str)

    def test_iso_format_content(self) -> None:
        dt = datetime(2026, 1, 15, 14, 30, 0, tzinfo=UTC)
        result = iso_format(dt)
        assert "2026-01-15" in result
        assert "14:30:00" in result

    def test_round_trip(self) -> None:
        """ISO format should be parseable back."""
        dt = utc_now()
        formatted = iso_format(dt)
        parsed = datetime.fromisoformat(formatted)
        assert parsed.year == dt.year
        assert parsed.month == dt.month
        assert parsed.day == dt.day
