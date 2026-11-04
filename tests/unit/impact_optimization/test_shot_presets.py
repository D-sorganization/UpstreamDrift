"""SHOW-C6 (#14562): shot-shape ImpactTarget presets (right-handed golfer).

Convention: +face = open (right), +path = in-to-out.  Curve = face - path:
negative curves left (draw), positive curves right (fade).  Start line is the
D-plane blend ``W*face + (1-W)*path`` and every preset aims it at the target.
"""

from __future__ import annotations

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.shot_presets import (
    START_LINE_FACE_WEIGHT,
    ShotPreset,
    all_presets,
    preset,
)

pytestmark = pytest.mark.unit


def _curve(p: ShotPreset) -> float:
    return p.target.face_deg - p.target.path_deg


def _start(p: ShotPreset) -> float:
    w = START_LINE_FACE_WEIGHT
    return w * p.target.face_deg + (1.0 - w) * p.target.path_deg


def test_at_least_four_presets_with_unique_names() -> None:
    names = [p.name for p in all_presets()]
    assert len(names) >= 4 and len(set(names)) == len(names)
    assert {"draw", "fade", "high", "low"} <= set(names)


def test_draw_curves_left_and_fade_curves_right() -> None:
    assert _curve(preset("draw")) < 0.0 < _curve(preset("fade"))


def test_draw_and_fade_are_mirror_images() -> None:
    d, f = preset("draw"), preset("fade")
    assert f.target.face_deg == pytest.approx(-d.target.face_deg)
    assert f.target.path_deg == pytest.approx(-d.target.path_deg)


@pytest.mark.parametrize("name", ["straight", "draw", "fade", "high", "low"])
def test_every_preset_starts_on_the_target_line(name: str) -> None:
    assert _start(preset(name)) == pytest.approx(0.0, abs=1e-9)


def test_curve_magnitude_is_configurable_and_scales_the_angles() -> None:
    a, b = preset("draw", curve_deg=2.0), preset("draw", curve_deg=4.0)
    assert _curve(b) == pytest.approx(2.0 * _curve(a))
    assert _curve(a) == pytest.approx(-2.0)


def test_high_and_low_bracket_straight_by_angle_of_attack() -> None:
    assert preset("low").target.aoa_deg < preset("straight").target.aoa_deg
    assert preset("straight").target.aoa_deg < preset("high").target.aoa_deg
    assert _curve(preset("high")) == 0.0 == _curve(preset("low"))


def test_assumptions_are_declared_not_silent() -> None:
    for p in all_presets():
        assert p.assumptions, p.name


def test_unknown_name_and_bad_magnitude_rejected() -> None:
    with pytest.raises(ContractViolationError):
        preset("slice")
    with pytest.raises(ContractViolationError):
        preset("draw", curve_deg=-1.0)
