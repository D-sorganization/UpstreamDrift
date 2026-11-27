"""Smash factor and unit flags reach the exports (#15150, #15443)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.tools.native_viewer_export import (
    cli,
    display_units as du,
    flight_tracer as tracer,
    impact_card as ic,
    impact_factors as fac,
)
from src.tools.native_viewer_export.core import ClipOverlays
from tests.unit.tools.native_viewer_export import (
    test_display_units_15443 as units_test,
    test_impact_factors as ft,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ARGS = ["--bundle", "b.npz", "--out", "out", "--swing", "driver"]


def _settings(*flags: str):
    return cli.build_settings(cli.build_parser().parse_args([*ARGS, *flags]))


def _panel(speed_unit: str | None = None):
    return fac.panel_from_series(
        ft._series(), ft.IMPACT_S, ft.BALL, ft.DEFAULTS, speed_unit=speed_unit
    )


def test_default_factor_rows_include_smash() -> None:
    assert "smash_factor" in fac.load_factor_settings().rows
    assert dict(_panel().lines_at(ft.IMPACT_S + 0.1))["Smash Factor"] == "unavailable"


def test_flight_hook_fills_smash_once(monkeypatch) -> None:
    calls: list[int] = []
    flight = units_test._playback(tracer.load_tracer_settings())

    def fake(*_args):
        calls.append(1)
        return flight

    monkeypatch.setattr(tracer, "flight_for_export", fake)
    overlays = ClipOverlays(factors=_panel())
    done = tracer.attach_flight(overlays, None, _settings("--ball-flight"))
    assert len(calls) == 1 and done.flight is flight
    shown = dict(done.factors.lines_at(ft.IMPACT_S + 0.1))["Smash Factor"]
    assert shown.endswith("(model)")
    plain = tracer.attach_flight(overlays, None, _settings())
    assert plain is overlays and len(calls) == 1


def test_hook_accepts_no_overlays(monkeypatch) -> None:
    flight = units_test._playback(tracer.load_tracer_settings())
    monkeypatch.setattr(tracer, "flight_for_export", lambda *_: flight)
    done = tracer.attach_flight(None, None, _settings("--ball-flight"))
    assert done.flight is flight and done.factors is None


def test_unit_flags_flow_into_settings() -> None:
    default = _settings("--ball-flight")
    assert default.speed_unit is None
    units = default.ball_flight.display_units
    assert (units.distance_unit, units.speed_unit) == ("yd", "mph")
    metric = _settings("--ball-flight", "--distance-unit", "m", "--speed-unit", "m/s")
    units = metric.ball_flight.display_units
    assert (units.distance_unit, units.speed_unit) == ("m", "m/s")
    assert metric.speed_unit == "m/s"
    assert _settings("--speed-unit", "m/s").speed_unit == "m/s"


def test_unit_flag_choices_come_from_the_settings() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([*ARGS, "--distance-unit", "ft"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([*ARGS, "--speed-unit", "kph"])


def test_unit_flags_change_the_hud_text() -> None:
    metric = _settings("--ball-flight", "--distance-unit", "m", "--speed-unit", "m/s")
    text = " ".join(units_test._playback(metric.ball_flight).hud_lines(9.0))
    assert "Carry 207 m" in text and "Ball 76 m/s" in text
    default = _settings("--ball-flight")
    text = " ".join(units_test._playback(default.ball_flight).hud_lines(9.0))
    assert "Carry 226 yd" in text and "Ball 170 mph" in text


def test_card_and_panel_follow_the_speed_unit() -> None:
    card = ic.card_from_series(ft._series(), ft.IMPACT_S, ft.BALL, speed_unit="m/s")
    assert card.units == "m/s"
    assert ic.card_from_series(ft._series(), ft.IMPACT_S, ft.BALL).units == "mph"
    assert "m/s" in dict(_panel("m/s").lines_at(ft.IMPACT_S + 0.1))["Clubhead Speed"]
    with pytest.raises(ValueError, match="units"):
        _panel("kph")


def test_manifest_record_has_si_values_and_unit_names() -> None:
    flight = units_test._playback(tracer.load_tracer_settings())
    record = tracer.manifest_record(ClipOverlays(flight=flight))
    assert record["units"] == {"distance": "yd", "speed": "mph"}
    assert record["carry_m"] == 207.0 and record["apex_m"] == 18.0
    assert record["ball_speed_mps"] == pytest.approx(170.0 / du.MPS_TO_MPH)
    assert tracer.manifest_record(None) is None
    bare = tracer.FlightPlayback(tracer.load_tracer_settings(), "no ball placed")
    assert "unavailable" in tracer.manifest_record(ClipOverlays(flight=bare))
