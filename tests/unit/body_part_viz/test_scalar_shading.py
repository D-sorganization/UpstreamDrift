"""Quantity-agnostic segment scalar shading (#15395)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.body_part_viz.scalar_shading import (
    DISPLAY_TRANSFORMS,
    SCALE_KINDS,
    ScalarColorLegend,
    ScalarColorScale,
    SegmentScalarSeries,
    shade_outcome_hex,
)

pytestmark = pytest.mark.unit


def test_segment_scalar_series_validates_units_and_sign():
    series = SegmentScalarSeries(
        time_s=[0.0, 1.0],
        values={"seg": (10.0, None)},
        source="test",
        units="J",
        sign_convention="unsigned",
    )
    assert series.units == "J"
    assert series.value_at("seg", 0) == 10.0
    assert series.value_at("seg", 1) is None
    with pytest.raises(ValueError, match="units"):
        SegmentScalarSeries(
            time_s=[0.0],
            values={"seg": (1.0,)},
            source="x",
            units="",
            sign_convention="unsigned",
        )
    with pytest.raises(ValueError, match="sign_convention"):
        SegmentScalarSeries(
            time_s=[0.0],
            values={"seg": (1.0,)},
            source="x",
            units="J",
            sign_convention="not-a-convention",
        )


def test_scalar_scale_invalid_limits_units_sign():
    with pytest.raises(ValueError, match="scale_kind"):
        ScalarColorScale(scale_kind="rainbow")
    with pytest.raises(ValueError, match="units"):
        ScalarColorScale(units="")
    with pytest.raises(ValueError, match="sign_convention"):
        ScalarColorScale(sign_convention="bogus")
    with pytest.raises(ValueError, match="display_transform"):
        ScalarColorScale(display_transform="log")
    with pytest.raises(ValueError):
        ScalarColorScale(scale_kind="sequential", low_limit=10, high_limit=10)
    with pytest.raises(ValueError, match="low_limit must be nonnegative"):
        ScalarColorScale(scale_kind="sequential", low_limit=-1, high_limit=10)


def test_diverging_endpoint_maps_zero_to_neutral():
    scale = ScalarColorScale(
        enabled=True,
        scale_kind="diverging",
        sign_convention="signed-zero-center",
        low_limit=-100.0,
        high_limit=100.0,
        low_color="#ff0000",
        high_color="#0000ff",
        neutral_color="#ffffff",
    )
    assert scale.color(0.0, "#123456") == "#ffffff"
    assert shade_outcome_hex(scale, 0.0, "#123456").saturated is False


def test_diverging_sqrt_changes_mapped_colour():
    linear = ScalarColorScale(
        enabled=True,
        scale_kind="diverging",
        sign_convention="signed-zero-center",
        low_limit=-100.0,
        high_limit=100.0,
        low_color="#ff0000",
        high_color="#0000ff",
        neutral_color="#ffffff",
        display_transform="linear",
    )
    sqrt_scale = ScalarColorScale(
        enabled=True,
        scale_kind="diverging",
        sign_convention="signed-zero-center",
        low_limit=-100.0,
        high_limit=100.0,
        low_color="#ff0000",
        high_color="#0000ff",
        neutral_color="#ffffff",
        display_transform="sqrt",
    )
    assert linear.color(25.0, "#000000") == "#bfbfff"
    assert sqrt_scale.color(25.0, "#000000") == "#8080ff"
    assert linear.color(50.0, "#000000") == "#8080ff"
    assert sqrt_scale.color(50.0, "#000000") == "#4b4bff"


def test_diverging_saturation_flag_beyond_limits():
    scale = ScalarColorScale(
        enabled=True,
        scale_kind="diverging",
        sign_convention="signed-zero-center",
        low_limit=-100.0,
        high_limit=100.0,
        low_color="#ff0000",
        high_color="#0000ff",
        neutral_color="#ffffff",
    )
    out = shade_outcome_hex(scale, 500.0, "#123456")
    assert out.saturated is True
    assert out.color == "#0000ff"
    out_neg = shade_outcome_hex(scale, -500.0, "#123456")
    assert out_neg.saturated is True


def test_missing_and_nonfinite_return_base_and_not_saturated():
    scale = ScalarColorScale(
        enabled=True, scale_kind="sequential", low_limit=0, high_limit=10
    )
    base = "#abcdef"
    for sample in (None, math.nan, math.inf):
        assert scale.color(sample, base) == base
        outcome = shade_outcome_hex(scale, sample, base)
        assert outcome.color == base
        assert outcome.saturated is False


def test_sequential_viridis_golden():
    scale = ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        colormap_id="viridis",
        low_limit=0.0,
        high_limit=100.0,
        units="J",
        sign_convention="unsigned",
    )
    assert scale.color(0.0, "#000000") == "#440154"
    assert scale.color(50.0, "#000000") == "#21918c"
    assert scale.color(100.0, "#000000") == "#fde725"
    clipped = shade_outcome_hex(scale, 200.0, "#000000")
    assert clipped.saturated is True
    assert clipped.color == "#fde725"


def test_sqrt_transform_ticks_in_physical_units():
    linear = ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        colormap_id="cividis",
        low_limit=0.0,
        high_limit=100.0,
        display_transform="linear",
        units="J",
    )
    sqrt_scale = ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        colormap_id="cividis",
        low_limit=0.0,
        high_limit=100.0,
        display_transform="sqrt",
        units="J",
    )
    legend = sqrt_scale.legend_descriptor()
    assert legend.units == "J"
    assert legend.display_transform == "sqrt"
    assert legend.tick_values[-1] == pytest.approx(100.0)
    raw_mid = 25.0
    assert sqrt_scale.color(raw_mid, "#000000") != linear.color(raw_mid, "#000000")


def test_sqrt_transform_does_not_mutate_series_values():
    series = SegmentScalarSeries(
        time_s=[0.0, 1.0],
        values={"seg": (25.0, 49.0)},
        source="test",
        units="J",
        sign_convention="unsigned",
    )
    assert series.value_at("seg", 0) == 25.0
    assert series.value_at("seg", 1) == 49.0


def test_scalar_settings_round_trip():
    scale = ScalarColorScale(
        enabled=True,
        scale_kind="diverging",
        sign_convention="signed-zero-center",
        display_transform="sqrt",
        low_limit=-50,
        high_limit=80,
    )
    assert ScalarColorScale.from_dict(scale.to_dict()) == scale
    with pytest.raises(ValueError):
        ScalarColorScale.from_dict({"typo": 1})


def test_legend_descriptor_carries_saturation_marker():
    scale = ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        low_limit=0,
        high_limit=1,
        units="N",
    )
    legend = scale.legend_descriptor()
    assert isinstance(legend, ScalarColorLegend)
    assert legend.saturation_note


def test_force_matches_main_regression_goldens():
    path = Path(__file__).resolve().parents[3] / "schemas/force-color-examples.json"
    for case in json.loads(path.read_text())["regression_cases"]:
        scale = ForceColorScale.from_dict(case["settings"])
        method = case["method"]
        base = case["base_color"]
        if method == "color":
            assert scale.color(case["force_n"], base) == case["expected"]
        elif method == "shade_hex":
            assert scale.shade_hex(case["force_n"], base) == case["expected"]
        else:
            raise AssertionError(f"unknown regression method {method!r}")


def test_force_to_dict_unchanged_after_wrap():
    legacy = ForceColorScale(enabled=True, tension_limit_n=42.0)
    assert ForceColorScale.from_dict(legacy.to_dict()) == legacy


def test_cross_runtime_force_examples_still_pass():
    path = Path(__file__).resolve().parents[3] / "schemas/force-color-examples.json"
    data = json.loads(path.read_text())
    for case in data["cases"]:
        scale = ForceColorScale.from_dict(case["settings"])
        assert scale.color(case["force_n"], "#123456") == case["expected"]
    for case in data["shade_cases"]:
        scale = ForceColorScale.from_dict(case["settings"])
        assert scale.shade_hex(case["force_n"], case["base_color"]) == case["expected"]


def test_scale_kinds_and_transforms_frozen():
    assert "sequential" in SCALE_KINDS and "diverging" in SCALE_KINDS
    assert "linear" in DISPLAY_TRANSFORMS and "sqrt" in DISPLAY_TRANSFORMS
