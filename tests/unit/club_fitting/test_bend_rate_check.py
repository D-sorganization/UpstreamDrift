"""Single-mode bend-rate check on synthetic traces (#15150 C18)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.club_fitting.bend_rate_check import (
    SETTINGS_PATH,
    BendRateSettings,
    bend_rate_check,
    check_bend_trace,
    load_bend_rate_settings,
)
from src.shared.python.club_fitting.bend_trace import BendTrace

pytestmark = pytest.mark.unit
F_HZ = 4.3
TIMES = np.linspace(-0.2, 0.0, 2001)


def _ellipse(f_hz: float, a: float, b: float) -> np.ndarray:
    """2-D free vibration of one mode: an ellipse traced at ``f_hz``."""
    w = 2.0 * np.pi * f_hz
    return np.column_stack([a * np.cos(w * TIMES), b * np.sin(w * TIMES)])


def test_free_vibration_of_one_mode_stays_inside_the_bound() -> None:
    # Energy: |v|^2 + w^2 |d|^2 is constant, so |v| <= w max|d|.
    check = bend_rate_check(TIMES, _ellipse(F_HZ, 0.2, 0.08), F_HZ, 1.0)
    assert check.single_mode_bound_mps == pytest.approx(2 * np.pi * F_HZ * 0.2)
    assert check.peak_rate_mps == pytest.approx(2 * np.pi * F_HZ * 0.2, rel=1e-3)
    assert check.ratio <= 1.0 + 1e-3
    assert not check.flagged


def test_a_faster_bend_than_the_mode_allows_is_flagged() -> None:
    check = bend_rate_check(TIMES, _ellipse(3.0 * F_HZ, 0.05, 0.05), F_HZ, 1.0)
    assert check.ratio == pytest.approx(3.0, rel=1e-3)
    assert check.flagged


def test_the_vector_rate_is_used_not_the_rate_of_the_norm() -> None:
    # A bend vector turning through zero (C18): the norm dips to its minimum and
    # grows again, while the vector itself moves at constant speed.
    d = np.column_stack([np.full(TIMES.size, 0.02), 5.0 * (TIMES + 0.1)])
    check = bend_rate_check(TIMES, d, F_HZ, 1.0)
    assert check.peak_rate_mps == pytest.approx(5.0, rel=1e-6)
    assert check.peak_bend_m == pytest.approx(np.hypot(0.02, 0.5), rel=1e-9)


def test_bend_trace_in_mm_is_checked_in_metres() -> None:
    d = _ellipse(F_HZ, 0.2, 0.08) * 1e3
    trace = BendTrace(TIMES, d[:, 0], d[:, 1])
    settings = BendRateSettings(ratio_limit=1.0, mm_per_m=1e3, seconds_per_minute=60.0)
    check = check_bend_trace(trace, 60.0 * F_HZ, settings)
    assert check.peak_bend_m == pytest.approx(0.2)
    assert not check.flagged


def test_contracts() -> None:
    good = _ellipse(F_HZ, 0.1, 0.1)
    with pytest.raises(ValueError, match="increasing"):
        bend_rate_check(TIMES[::-1], good, F_HZ, 1.0)
    with pytest.raises(ValueError, match="three"):
        bend_rate_check(TIMES[:2], good[:2], F_HZ, 1.0)
    with pytest.raises(ValueError, match="rows"):
        bend_rate_check(TIMES, good[:-1], F_HZ, 1.0)
    with pytest.raises(ValueError, match="frequency_hz"):
        bend_rate_check(TIMES, good, 0.0, 1.0)
    with pytest.raises(ValueError, match="ratio_limit"):
        bend_rate_check(TIMES, good, F_HZ, -1.0)
    with pytest.raises(ValueError, match="no bend"):
        bend_rate_check(TIMES, np.zeros_like(good), F_HZ, 1.0)


def test_settings_load_and_validate(tmp_path: Path) -> None:
    settings = load_bend_rate_settings()
    assert settings.ratio_limit > 0.0
    data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    bad = tmp_path / "s.json"
    bad.write_text(json.dumps({**data, "ratio_limit": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="ratio_limit"):
        load_bend_rate_settings(bad)
    extra = tmp_path / "x.json"
    extra.write_text(json.dumps({**data, "unknown": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="keys"):
        load_bend_rate_settings(extra)
