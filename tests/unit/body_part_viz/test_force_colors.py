"""Behavioral contracts for signed axial-load display."""

import math

import pytest

from src.shared.python.body_part_viz.force_colors import ForceColorScale


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
