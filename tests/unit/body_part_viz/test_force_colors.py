"""Behavioral contracts for signed axial-load display."""

import math
import json
from pathlib import Path

import pytest

from src.shared.python.body_part_viz.force_colors import ForceColorScale

pytestmark = pytest.mark.unit


def test_cross_runtime_conformance_examples():
    path = Path(__file__).resolve().parents[3] / "schemas/force-color-examples.json"
    for case in json.loads(path.read_text())["cases"]:
        scale = ForceColorScale.from_dict(case["settings"])
        assert scale.color(case["force_n"], "#123456") == case["expected"]


def test_disabled_preserves_base_color():
    scale = ForceColorScale()
    assert scale.color(100.0, "#abcdef") == "#abcdef"


def test_sign_zero_clipping_and_missing_are_distinct():
    scale = ForceColorScale(enabled=True, tension_limit_n=100, compression_limit_n=200)
    assert scale.color(100, "#abcdef") == "#0000ff"
    assert scale.color(-200, "#abcdef") == "#ff0000"
    assert scale.color(10000, "#abcdef") == "#0000ff"
    assert scale.color(-10000, "#abcdef") == "#ff0000"
    assert scale.color(0, "#abcdef") == "#ffffff"
    assert scale.color(None, "#abcdef") == "#abcdef"
    assert scale.color(math.nan, "#abcdef") == "#abcdef"
    assert scale.color(math.inf, "#abcdef") == "#abcdef"


def test_custom_colors_deadband_and_asymmetric_limits():
    scale = ForceColorScale(
        enabled=True,
        tension_limit_n=110,
        compression_limit_n=210,
        deadband_n=10,
        tension_color="#00ff00",
        compression_color="#ff00ff",
        neutral_color="#000000",
    )
    assert scale.color(10, "#abcdef") == "#000000"
    assert scale.color(-10, "#abcdef") == "#000000"
    assert scale.color(60, "#abcdef") == "#008000"
    assert scale.color(-110, "#abcdef") == "#800080"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tension_limit_n": 0},
        {"compression_limit_n": -1},
        {"deadband_n": -1},
        {"deadband_n": 1000},
        {"tension_limit_n": math.inf},
        {"compression_limit_n": math.nan},
        {"tension_color": "blue"},
        {"neutral_color": "#ffffff00"},
    ],
)
def test_invalid_configuration_fails_closed(kwargs):
    with pytest.raises(ValueError):
        ForceColorScale(**kwargs)


@pytest.mark.parametrize("kwargs", [{"enabled": 1}, {"deadband_n": True}])
def test_wrong_configuration_types_fail(kwargs):
    with pytest.raises(TypeError):
        ForceColorScale(**kwargs)


def test_settings_round_trip_and_unknown_fields_rejected():
    scale = ForceColorScale(enabled=True, tension_limit_n=42)
    assert ForceColorScale.from_dict(scale.to_dict()) == scale
    with pytest.raises(ValueError):
        ForceColorScale.from_dict({"typo": 42})


def test_force_requires_numeric_or_missing():
    scale = ForceColorScale(enabled=True)
    with pytest.raises(TypeError):
        scale.color("123", "#abcdef")
    with pytest.raises(TypeError):
        scale.color(True, "#abcdef")


# --- Translucent overlay policy (opacity + blend mode) -----------------------


def _scale(**kw):
    return ForceColorScale(enabled=True, **kw)


def test_overlay_defaults_are_backward_compatible():
    from src.shared.python.body_part_viz.force_colors import BLEND_MODES

    scale = ForceColorScale()
    assert (scale.opacity, scale.blend_mode) == (1.0, "replace")
    assert BLEND_MODES == ("replace", "tint")
    legacy = {"enabled": True, "tension_limit_n": 50.0}
    assert ForceColorScale.from_dict(legacy).opacity == 1.0
    assert ForceColorScale.from_dict(scale.to_dict()) == scale
    assert scale.to_dict()["opacity"] == 1.0


@pytest.mark.parametrize("bad", [-0.1, 1.1, math.nan, math.inf])
def test_opacity_range_validated(bad):
    with pytest.raises(ValueError):
        ForceColorScale(opacity=bad)


@pytest.mark.parametrize("bad", ["0.5", None, True])
def test_opacity_type_validated(bad):
    with pytest.raises(TypeError):
        ForceColorScale(opacity=bad)


def test_blend_mode_validated():
    with pytest.raises(ValueError):
        ForceColorScale(blend_mode="multiply")
    with pytest.raises(TypeError):
        ForceColorScale(blend_mode=1)


def test_shade_rgb_returns_none_when_disabled_or_unavailable():
    base = (0.2, 0.4, 0.6)
    assert ForceColorScale().shade_rgb(500, base) is None
    for force in (None, math.nan, math.inf):
        assert _scale().shade_rgb(force, base) is None


def test_shade_rgb_validates_inputs():
    scale = _scale()
    for bad in ((0.1, 0.2), (0.1, 0.2, 1.5), (0.1, 0.2, math.nan), "abc"):
        with pytest.raises((TypeError, ValueError)):
            scale.shade_rgb(10, bad)
    with pytest.raises(TypeError):
        scale.shade_rgb("x", (0, 0, 0))


def test_replace_full_opacity_matches_color():
    scale = _scale()
    for force in (-2000, -500, 0, 250, 1000, 5000):
        rgb = scale.shade_rgb(force, (0.2, 0.4, 0.6))
        expected = scale.color(force, "#000000")
        got = "#" + "".join(f"{round(c * 255):02x}" for c in rgb)
        assert got == expected


def test_replace_half_opacity_is_midpoint_of_base_and_color():
    scale = _scale(opacity=0.5)
    rgb = scale.shade_rgb(1000, (1.0, 1.0, 0.0))  # target blue
    assert rgb == pytest.approx((0.5, 0.5, 0.5))
    assert scale.shade_hex(1000, "#ffff00") == "#808080"


def test_tint_keeps_base_in_deadband_and_for_zero_load():
    scale = _scale(blend_mode="tint", deadband_n=100.0)
    base = (0.3, 0.5, 0.7)
    assert scale.shade_rgb(0, base) == pytest.approx(base)
    assert scale.shade_rgb(-100, base) == pytest.approx(base)


def test_tint_saturates_to_lerp_of_base_and_endpoint():
    scale = _scale(blend_mode="tint", opacity=0.4)
    base = (0.5, 0.5, 0.5)
    tension = scale.shade_rgb(10000, base)
    assert tension == pytest.approx((0.5 * 0.6, 0.5 * 0.6, 0.5 * 0.6 + 0.4))
    compression = scale.shade_rgb(-10000, base)
    assert compression == pytest.approx((0.5 * 0.6 + 0.4, 0.5 * 0.6, 0.5 * 0.6))
    half = scale.shade_rgb(500, base)  # fraction 0.5 -> weight 0.2
    assert half == pytest.approx((0.4, 0.4, 0.6))


def test_zero_opacity_keeps_base_in_every_mode():
    base = (0.1, 0.2, 0.3)
    for mode in ("replace", "tint"):
        scale = _scale(opacity=0.0, blend_mode=mode)
        assert scale.shade_rgb(1000, base) == pytest.approx(base)


def test_shade_hex_none_when_disabled_and_validates_base():
    assert ForceColorScale().shade_hex(10, "#123456") is None
    assert _scale().shade_hex(None, "#123456") is None
    with pytest.raises(ValueError):
        _scale(opacity=0.5).shade_hex(1000, "not-a-color")
    # Opaque replace never needs the base, so legacy blank bases keep working.
    assert _scale().shade_hex(1000, "") == "#0000ff"


def test_cross_runtime_shade_examples():
    path = Path(__file__).resolve().parents[3] / "schemas/force-color-examples.json"
    for case in json.loads(path.read_text())["shade_cases"]:
        scale = ForceColorScale.from_dict(case["settings"])
        assert scale.shade_hex(case["force_n"], case["base_color"]) == case["expected"]
