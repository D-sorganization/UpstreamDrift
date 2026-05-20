"""Tests for src/api/utils/datetime_compat.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.api.utils import datetime_compat as dc

pytestmark = pytest.mark.unit


def test_utc_now_is_timezone_aware() -> None:
    now = dc.utc_now()
    assert now.tzinfo is not None
    # Should be very close to "now"
    delta = abs((datetime.now(timezone.utc) - now).total_seconds())
    assert delta < 5.0


def test_add_minutes_positive_and_negative() -> None:
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=dc.UTC)
    assert dc.add_minutes(base, 30) == base + timedelta(minutes=30)
    assert dc.add_minutes(base, -15) == base - timedelta(minutes=15)
    assert dc.add_minutes(base, 0) == base


def test_add_days_positive_and_negative() -> None:
    base = datetime(2024, 1, 1, tzinfo=dc.UTC)
    assert dc.add_days(base, 7) == base + timedelta(days=7)
    assert dc.add_days(base, -1) == base - timedelta(days=1)


def test_iso_format_round_trip() -> None:
    base = datetime(2024, 6, 15, 8, 30, 45, tzinfo=dc.UTC)
    iso = dc.iso_format(base)
    assert isinstance(iso, str)
    assert "2024-06-15" in iso
    parsed = datetime.fromisoformat(iso)
    assert parsed == base


def test_utc_constant_is_utc() -> None:
    now = datetime.now(dc.UTC)
    assert now.utcoffset() == timedelta(0)
