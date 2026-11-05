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
    shape_label,
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


# --- ball-flight curvature sign (issue #14562 acceptance) -------------------

import numpy as np  # noqa: E402

from src.shared.python.impact_parameters import TargetFrame  # noqa: E402
from src.shared.python.impact_parameters.tools_gateway import (  # noqa: E402
    load_tools_delivery_gateway,
)

_LOFT_DEG = 12.0
_SPEED_MPS = 45.0


def _delivery(name: str):
    """Tools D-plane estimate for a preset's face/path, aoa and a fixed loft."""
    # Fail closed: the pinned Tools tree is a hard repo requirement (conftest
    # raises without it), so a missing gateway must fail, never skip.
    gw = load_tools_delivery_gateway()
    t = preset(name).target
    frame = TargetFrame()

    def world(a: float, b: float, c: float) -> np.ndarray:
        return a * frame.x_t + b * frame.y_t + c * np.asarray(frame.up)

    path, face, aoa, loft = (
        np.radians(d) for d in (t.path_deg, t.face_deg, t.aoa_deg, _LOFT_DEG)
    )
    # +path / +face = rightward (in-to-out / open), y_t is left of target.
    v = _SPEED_MPS * world(
        np.cos(aoa) * np.cos(path), -np.cos(aoa) * np.sin(path), np.sin(aoa)
    )
    n = world(np.cos(loft) * np.cos(face), -np.cos(loft) * np.sin(face), np.sin(loft))
    return gw.estimate(v, n, frame)


def test_draw_and_fade_curve_with_the_documented_tools_signs() -> None:
    """Tools convention: spin_axis_tilt + = fade side (right curve for a RH golfer),
    face_to_path + = open to path (fade). A draw is therefore negative on both."""
    draw, fade, straight = (_delivery(n) for n in ("draw", "fade", "straight"))
    assert draw.face_to_path_deg < 0.0 and draw.spin_axis_tilt_deg < 0.0
    assert fade.face_to_path_deg > 0.0 and fade.spin_axis_tilt_deg > 0.0
    assert draw.face_to_path_deg == pytest.approx(-4.0)
    assert fade.face_to_path_deg == pytest.approx(4.0)
    assert abs(straight.face_to_path_deg) < 1e-6
    assert abs(straight.spin_axis_tilt_deg) < 1e-6


def test_high_and_low_differ_in_attack_angle_only() -> None:
    high, low = _delivery("high"), _delivery("low")
    assert high.attack_angle_deg > 0.0 > low.attack_angle_deg
    assert high.face_to_path_deg == pytest.approx(low.face_to_path_deg)


# D30: shot shape is named by face-to-path, never by one angle.
def test_label_names_the_face_to_path_value_of_every_preset() -> None:
    assert preset("draw").label == "draw (face-to-path -4.0 deg)"
    assert preset("fade").label == "fade (face-to-path +4.0 deg)"
    assert preset("straight").label == "straight (face-to-path +0.0 deg)"


def test_issue_example_face_minus2_path_plus4_is_a_strong_draw_not_2_deg() -> None:
    label = shape_label(face_deg=-2.0, path_deg=4.0)
    assert label == "strong draw (face-to-path -6.0 deg)"
    assert "2 deg" not in label.replace("-6.0 deg", "")


@pytest.mark.parametrize(
    ("face", "path", "word"),
    [
        (0.1, 0.0, "straight"),
        (-1.0, 2.0, "draw"),
        (1.0, -2.0, "fade"),
        (3.0, -4.0, "strong fade"),
    ],
)
def test_shape_label_classifies_by_face_to_path(face, path, word) -> None:
    assert shape_label(face_deg=face, path_deg=path).startswith(word + " (")


def test_high_and_low_labels_name_their_attack_angle() -> None:
    assert preset("high").label == "high (aoa +3.0 deg)"
    assert preset("low").label == "low (aoa -3.0 deg)"
