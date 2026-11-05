"""Green specs: one description of a green's shape for every route (#14463)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.putting_dynamics import (
    GREEN_PRESETS,
    GreenSpec,
    HeightField,
    SurfaceSpec,
    bumpy_height_field,
    green_preset,
)

pytestmark = pytest.mark.unit


def test_the_default_spec_is_the_flat_uniform_green() -> None:
    surface = GreenSpec().surface(stimp_ft=11.0, extent_m=10.0)
    flat = SurfaceSpec.flat_uniform(stimp_ft=11.0, extent_m=10.0)
    np.testing.assert_array_equal(surface.height.heights_m, flat.height.heights_m)
    assert surface.friction == flat.friction


def test_a_sloped_spec_is_the_core_plane() -> None:
    surface = GreenSpec(grade_percent=2.0, aspect_deg=90.0).surface(10.0, 10.0)
    plane = HeightField.planar(2.0, 90.0, extent_m=10.0)
    np.testing.assert_array_equal(surface.height.heights_m, plane.heights_m)


def test_undulation_is_the_seeded_core_bump_field() -> None:
    spec = GreenSpec(grade_percent=1.0, bump_height_m=0.01, bump_length_m=2.0, seed=7)
    want = bumpy_height_field(
        seed=7,
        amplitude_m=0.01,
        correlation_length_m=2.0,
        base=HeightField.planar(1.0, 0.0, extent_m=12.0),
    )
    got = spec.surface(10.0, 12.0).height
    np.testing.assert_array_equal(got.heights_m, want.heights_m)
    other = GreenSpec(bump_height_m=0.01, bump_length_m=2.0, seed=8)
    assert not np.array_equal(other.surface(10.0, 12.0).height.heights_m, got.heights_m)


def test_a_tier_steps_up_on_the_uphill_side() -> None:
    # The green falls toward +x, so the upper tier lies toward -x.
    spec = GreenSpec(aspect_deg=0.0, tier_height_m=0.08, tier_width_m=1.0)
    height = spec.surface(10.0, 20.0).height
    upper, lower = height.elevation(-8.0, 0.0), height.elevation(8.0, 0.0)
    assert upper - lower == pytest.approx(0.08, abs=1e-4)
    assert height.elevation(0.0, 3.0) == pytest.approx(0.04, abs=1e-9)
    assert height.gradient(-8.0, 0.0) == pytest.approx((0.0, 0.0), abs=1e-4)
    assert height.gradient(0.0, 0.0)[0] < -0.03  # the step faces downhill


def test_presets_are_named_valid_specs() -> None:
    assert set(GREEN_PRESETS) == {
        "flat_practice",
        "undulating_championship",
        "severe_slopes",
        "tiered",
    }
    assert green_preset("flat_practice") == GreenSpec()
    assert green_preset("undulating_championship").bump_height_m > 0.0
    assert green_preset("severe_slopes").grade_percent == pytest.approx(5.0)
    assert green_preset("tiered").tier_height_m > 0.0
    for spec in GREEN_PRESETS.values():
        spec.surface(10.0, 20.0)
    with pytest.raises(ValueError, match="preset"):
        green_preset("augusta")


@pytest.mark.parametrize(
    "fields",
    [
        {"grade_percent": -1.0},
        {"grade_percent": 11.0},
        {"aspect_deg": float("nan")},
        {"bump_height_m": 0.06},
        {"bump_length_m": 0.0},
        {"tier_height_m": 0.2},
        {"tier_width_m": 0.0},
        {"seed": -1},
    ],
)
def test_out_of_range_specs_are_refused(fields: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        GreenSpec(**fields)
