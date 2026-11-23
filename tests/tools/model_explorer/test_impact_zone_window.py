"""Contact-window -> impact-zone clock adapter (#15524, epic #15150)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from src.shared.python.physics.impact_interval_handoff import UD_IMPACT_PLAYBACK_SCHEMA
from src.tools.model_explorer.impact_zone_clock import ImpactZoneClock
from src.tools.model_explorer.impact_zone_window import (
    ImpactZoneWindow,
    build_impact_zone_clock,
    impact_zone_window,
)

pytestmark = pytest.mark.unit

TIME = np.linspace(0.0, 2.0, 201)


def _summary(window: Any = (0.99, 1.01), qualified: bool = True) -> dict[str, Any]:
    return {
        "format": UD_IMPACT_PLAYBACK_SCHEMA,
        "qualified": qualified,
        "contact_window_s": None if window is None else list(window),
    }


def test_window_is_centred_on_contact_and_padded() -> None:
    win = impact_zone_window(_summary(), TIME, pad_s=0.05)
    assert win is not None
    assert win.impact_s == pytest.approx(1.0)
    assert win.half_width_s == pytest.approx(0.01 + 0.05)


def test_half_width_never_drops_below_the_minimum() -> None:
    win = impact_zone_window(
        _summary((1.0, 1.0 + 1e-6)), TIME, pad_s=0.0, min_half_width_s=0.02
    )
    assert win is not None
    assert win.half_width_s == pytest.approx(0.02)


def test_unqualified_summary_gives_no_window() -> None:
    assert impact_zone_window(_summary(qualified=False), TIME) is None
    assert impact_zone_window(_summary(window=None, qualified=False), TIME) is None


def test_qualified_flag_must_be_literally_true() -> None:
    summary = _summary()
    summary["qualified"] = "yes"
    assert impact_zone_window(summary, TIME) is None


def test_wrong_schema_or_non_mapping_gives_no_window() -> None:
    summary = _summary()
    summary["format"] = "something.else/1"
    assert impact_zone_window(summary, TIME) is None
    assert impact_zone_window(None, TIME) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "window",
    [
        None,
        (1.0,),
        (1.0, 1.1, 1.2),
        (1.1, 1.0),
        (1.0, 1.0),
        (math.nan, 1.0),
        (1.0, math.inf),
        ("a", "b"),
        (True, 2.0),
    ],
)
def test_malformed_contact_window_gives_no_window(window: Any) -> None:
    assert impact_zone_window(_summary(window), TIME) is None


def test_contact_outside_the_run_gives_no_window() -> None:
    # Contact centred at 5 s on a 0..2 s run: simulation time origin mismatch.
    assert impact_zone_window(_summary((4.99, 5.01)), TIME) is None


def test_contact_straddling_the_run_end_is_kept_when_centre_is_inside() -> None:
    win = impact_zone_window(_summary((1.95, 2.01)), TIME, pad_s=0.0)
    assert win is not None
    assert win.impact_s == pytest.approx(1.98)


def test_invalid_tuning_is_a_contract_violation() -> None:
    with pytest.raises(ValueError):
        impact_zone_window(_summary(), TIME, pad_s=-0.1)
    with pytest.raises(ValueError):
        impact_zone_window(_summary(), TIME, min_half_width_s=0.0)


def test_build_clock_slows_the_contact_interval() -> None:
    clock = build_impact_zone_clock(_summary(), TIME, slow_factor=0.1)
    assert isinstance(clock, ImpactZoneClock)
    clock.start(0, now_s=0.0)
    # Zone is 0.99 +/- (0.01 + default pad); media time at zone entry is exact.
    lead = 1.0 - 0.01 - clock_pad()
    assert clock.media_time_at(lead) == pytest.approx(lead)
    assert clock.media_time_at(lead + 0.1) == pytest.approx(lead + 0.01)


def clock_pad() -> float:
    from src.tools.model_explorer.impact_zone_window import DEFAULT_PAD_S

    return DEFAULT_PAD_S


def test_build_clock_returns_none_without_a_window() -> None:
    assert build_impact_zone_clock(_summary(qualified=False), TIME, 0.1) is None


def test_build_clock_forwards_speed_and_validates_slow_factor() -> None:
    clock = build_impact_zone_clock(_summary(), TIME, slow_factor=0.5, speed=0.5)
    assert clock is not None and clock.speed == 0.5
    with pytest.raises(ValueError):
        build_impact_zone_clock(_summary(), TIME, slow_factor=0.0)


def test_window_is_a_frozen_value() -> None:
    win = ImpactZoneWindow(impact_s=1.0, half_width_s=0.05)
    with pytest.raises(AttributeError):
        win.impact_s = 2.0  # type: ignore[misc]
