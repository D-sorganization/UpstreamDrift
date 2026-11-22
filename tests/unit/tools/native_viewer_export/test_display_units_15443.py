"""Yards by default and the model smash factor in native exports (#15443)."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from src.shared.python.impact_parameters.flight_report import FlightSummary
from src.shared.python.impact_parameters.panel_model import CardRow, ImpactCard
from src.shared.python.impact_parameters.target_frame import TargetFrame
from src.tools.native_viewer_export import (
    compositor,
    display_units as du,
    flight_inset as fi,
    flight_tracer as tracer,
    impact_card as ic,
    impact_factors as fac,
    landing_marker as lm,
)
from src.tools.native_viewer_export.ball import BallPlacement

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

YARDS, METRES = du.load_display_units(), du.METRIC
SMASH_REASON = "needs an impact model ball speed and its calibration status"


def _flight() -> FlightSummary:
    t = np.linspace(0.0, 4.0, 41)
    local = np.column_stack([50.0 * t, -2.0 * t, 5.0 * t * (4.0 - t)])
    return FlightSummary(
        carry_m=207.0,
        lateral_m=15.0,
        max_height_m=18.0,
        flight_time_s=4.0,
        ball_speed_mph=170.0,
        launch_deg=12.0,
        spin_rpm=2600.0,
        down_range_m=207.0,
        trajectory_m=local,
        times_s=t,
    )


def _playback(settings: tracer.TracerSettings):
    ball = BallPlacement((1.0, -0.2, 0.03), 0.021335, "test")
    frame = TargetFrame(target_dir=(0.0, -1.0, 0.0))
    return tracer.playback_from_flight(_flight(), frame, 1.0, ball, settings)


def test_default_is_yards_and_speed_mph() -> None:
    assert (YARDS.distance_unit, YARDS.speed_unit) == ("yd", "mph")
    assert du.distance_text(207.0, YARDS) == "226 yd"
    assert du.distance_text(207.0, METRES) == "207 m"
    assert du.METRES_PER_YARD == 0.9144
    assert du.convert_distance(0.9144, YARDS) == pytest.approx(1.0, abs=1e-12)


def test_speed_text_and_invalid_units() -> None:
    assert du.speed_text(170.0, YARDS) == "170 mph"
    assert du.speed_text(170.0, replace(YARDS, speed_unit="m/s")) == "76 m/s"
    with pytest.raises(ValueError, match="distance_unit"):
        du.DisplayUnits(distance_unit="ft")
    with pytest.raises(ValueError, match="speed_unit"):
        du.DisplayUnits(speed_unit="kph")
    with pytest.raises(ValueError, match="finite"):
        du.distance_text(float("nan"), YARDS)


def test_settings_file_is_validated(tmp_path) -> None:
    path = tmp_path / "units.json"
    path.write_text(json.dumps({"distance_unit": "m"}), encoding="utf-8")
    assert du.load_display_units(path).distance_unit == "m"
    path.write_text(json.dumps({"distance_unit": "ft"}), encoding="utf-8")
    with pytest.raises(ValueError, match="distance_unit"):
        du.load_display_units(path)
    path.write_text(json.dumps({"bogus": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="bogus"):
        du.load_display_units(path)


def test_hud_lines_in_yards_and_in_metres() -> None:
    text = " ".join(_playback(tracer.load_tracer_settings()).hud_lines(9.0))
    assert "Carry 226 yd" in text and "Apex 20 yd" in text
    assert "Side 16 yd right" in text and "Ball 170 mph" in text
    metric = replace(tracer.load_tracer_settings(), display_units=METRES)
    text = " ".join(_playback(metric).hud_lines(9.0))
    assert "Carry 207 m   Apex 18 m   Side 15 m right" in text


def test_inset_and_landing_label_follow_the_units() -> None:
    settings = tracer.load_tracer_settings()
    inset, marker = settings.inset, settings.landing_marker
    assert inset is not None and marker is not None
    assert (inset.unit, marker.carry_label_unit) == ("yd", "yd")
    assert lm.carry_text(marker, 207.0) == "Carry 226 yd"
    label = fi._fmt(inset, "apex_label_format", apex=18.0 * inset.unit_scale)
    assert label == "Apex 20 yd"
    metric = tracer.load_tracer_settings(units=METRES)
    assert metric.inset is not None and metric.landing_marker is not None
    assert lm.carry_text(metric.landing_marker, 207.0) == "Carry 207 m"
    assert metric.inset.unit == "m" and metric.inset.unit_scale == 1.0


def test_playback_keeps_si_and_names_the_display_unit() -> None:
    flight = _playback(tracer.load_tracer_settings())
    assert flight.carry_m == 207.0 and flight.lateral_m == 15.0
    assert flight.display_unit_names() == {"distance": "yd", "speed": "mph"}


def test_smash_factor_is_ball_over_clubhead_speed() -> None:
    assert round(ic.smash_factor(170.0, 114.7), 2) == 1.48
    for bad in ((0.0, 100.0), (170.0, 0.0), (float("nan"), 100.0), (-1.0, 100.0)):
        with pytest.raises(ValueError):
            ic.smash_factor(*bad)


def _card(units: str = "mph", speed: float = 114.7) -> ImpactCard:
    rows = (
        CardRow("clubhead_speed", "Clubhead Speed", units, speed),
        CardRow("smash_factor", "Smash Factor", "", None, SMASH_REASON),
    )
    return ImpactCard(units, 1.0, "explicit", {}, rows, {})


def test_card_shows_model_smash_and_keeps_unavailable_without_it() -> None:
    plain = compositor._card_lines(_card())
    assert ("Smash Factor", "unavailable", True) in plain
    assert SMASH_REASON in " ".join(line[0].strip() for line in plain)
    shown = compositor._card_lines(ic.with_model_smash(_card(), 170.0))
    assert ("Smash Factor", "1.48 (model)", False) in shown
    assert "needs an impact" not in " ".join(line[0] for line in shown)
    assert ic.with_model_smash(_card(), None) == _card()


def test_card_smash_in_metres_per_second_units() -> None:
    card = _card("m/s", 114.7 / 2.2369362920544)
    row = ic.with_model_smash(card, 170.0).rows[1]
    assert row.value == pytest.approx(170.0 / 114.7)


def test_factors_panel_shares_the_smash_row() -> None:
    settings = replace(
        fac.load_factor_settings(), rows=("clubhead_speed", "smash_factor")
    )
    card = _card()
    panel = fac.FactorsPanel(
        settings, 1.0, fac.AT_IMPACT, fac._rows(card, settings), card=card
    )
    assert dict(panel.lines_at(2.0))["Smash Factor"] == "unavailable"
    done = panel.with_model_smash(170.0)
    assert dict(done.lines_at(2.0))["Smash Factor"] == "1.48 (model)"


def test_overlays_take_the_flight_ball_speed_for_card_and_panel() -> None:
    from src.tools.native_viewer_export.core import ClipOverlays

    settings = replace(
        fac.load_factor_settings(), rows=("clubhead_speed", "smash_factor")
    )
    card = _card()
    panel = fac.FactorsPanel(
        settings, 1.0, fac.AT_IMPACT, fac._rows(card, settings), card=card
    )
    flight = _playback(tracer.load_tracer_settings())
    assert flight.ball_speed_mph == 170.0
    both = tracer.with_model_smash(ClipOverlays(flight, card, None, panel))
    assert both.card.rows[1].value == pytest.approx(170.0 / 114.7)
    assert dict(both.factors.lines_at(2.0))["Smash Factor"] == "1.48 (model)"
    bare = ClipOverlays(replace(flight, ball_speed_mph=None), card, None, panel)
    assert tracer.with_model_smash(bare) is bare
    assert tracer.with_model_smash(None) is None
