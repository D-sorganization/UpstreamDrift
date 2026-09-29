"""Tests for datetime compatibility utilities."""

from __future__ import annotations

from datetime import datetime, timezone

from src.api.utils import datetime_compat


def test_utc_now() -> None:
    now = datetime_compat.utc_now()
    assert isinstance(now, datetime)
    assert now.tzinfo is not None
    assert now.tzinfo.utcoffset(now) == timezone.utc.utcoffset(now)


def test_add_minutes() -> None:
    dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    added = datetime_compat.add_minutes(dt, 30)
    assert added == datetime(2026, 1, 1, 12, 30, 0, tzinfo=timezone.utc)

    subtracted = datetime_compat.add_minutes(dt, -15)
    assert subtracted == datetime(2026, 1, 1, 11, 45, 0, tzinfo=timezone.utc)


def test_add_days() -> None:
    dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    added = datetime_compat.add_days(dt, 5)
    assert added == datetime(2026, 1, 6, 12, 0, 0, tzinfo=timezone.utc)

    subtracted = datetime_compat.add_days(dt, -2)
    assert subtracted == datetime(2025, 12, 30, 12, 0, 0, tzinfo=timezone.utc)


def test_iso_format() -> None:
    dt = datetime(2026, 1, 1, 12, 30, 45, tzinfo=timezone.utc)
    formatted = datetime_compat.iso_format(dt)
    assert formatted == "2026-01-01T12:30:45+00:00"
