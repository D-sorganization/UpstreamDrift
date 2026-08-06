"""Surface-field tests for putting_dynamics (#8345 P2).

Analytic limits, seeded reproducibility, and the keyed-stream
discipline reused from the Tools variation engine.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.putting_dynamics import (
    FrictionField,
    HeightField,
    bumpy_friction_field,
    bumpy_height_field,
)

pytestmark = pytest.mark.unit


class TestHeightField:
    def test_flat_elevation_and_gradient_are_zero(self) -> None:
        field = HeightField.flat(extent_m=10.0, spacing_m=0.5)
        assert field.elevation(1.234, -2.345) == 0.0
        assert field.gradient(1.234, -2.345) == (0.0, 0.0)

    def test_bilinear_is_exact_on_grid_nodes(self) -> None:
        rng = np.random.default_rng(7)
        heights = rng.normal(0.0, 0.01, size=(9, 9))
        field = HeightField(heights_m=heights, spacing_m=1.0, origin_m=(-4.0, -4.0))
        assert field.elevation(-4.0 + 3.0, -4.0 + 5.0) == pytest.approx(
            heights[5, 3], abs=1e-15
        )

    def test_planar_gradient_matches_grade_and_aspect(self) -> None:
        grade, aspect_deg = 2.0, 30.0
        field = HeightField.planar(grade, aspect_deg, extent_m=20.0)
        gx, gy = field.gradient(1.3, -2.7)
        aspect = math.radians(aspect_deg)
        assert gx == pytest.approx(-0.02 * math.cos(aspect), rel=1e-9)
        assert gy == pytest.approx(-0.02 * math.sin(aspect), rel=1e-9)
        # Downhill direction: elevation drops along the aspect.
        step = 1.0
        drop = field.elevation(
            step * math.cos(aspect), step * math.sin(aspect)
        ) - field.elevation(0.0, 0.0)
        assert drop == pytest.approx(-0.02 * step, rel=1e-9)

    def test_out_of_grid_queries_clamp(self) -> None:
        field = HeightField.planar(2.0, 0.0, extent_m=10.0)
        inside = field.elevation(-5.0, 0.0)
        outside = field.elevation(-50.0, 0.0)
        assert outside == pytest.approx(inside, abs=0.02 * 0.5)

    def test_rejects_tiny_grids(self) -> None:
        with pytest.raises(ValueError, match="2x2"):
            HeightField(heights_m=np.zeros((1, 5)))

    def test_factory_rejects_invalid_spacing_with_contract_error(self) -> None:
        with pytest.raises(ValueError, match="spacing"):
            HeightField.flat(spacing_m=0.0)

    def test_field_owns_an_immutable_copy_of_input_data(self) -> None:
        source = np.zeros((3, 3))
        field = HeightField(source)
        source[1, 1] = 1.0
        assert field.heights_m[1, 1] == 0.0
        assert not field.heights_m.flags.writeable


class TestFrictionField:
    def test_uniform_multipliers_are_one(self) -> None:
        field = FrictionField.uniform(extent_m=10.0)
        roll, slide = field.multipliers(0.7, -3.1)
        assert roll == pytest.approx(1.0, rel=1e-12)
        assert slide == pytest.approx(1.0, rel=1e-12)

    def test_rejects_non_positive_multipliers(self) -> None:
        bad = np.ones((4, 4))
        bad[2, 2] = 0.0
        with pytest.raises(ValueError, match="positive"):
            FrictionField(roll_multiplier=bad, slide_multiplier=np.ones((4, 4)))


class TestBumpyFields:
    def test_zero_amplitude_is_identical_to_base(self) -> None:
        base = HeightField.flat(extent_m=10.0)
        assert bumpy_height_field(3, 0.0, 1.0, base) is base
        fbase = FrictionField.uniform(extent_m=10.0)
        assert bumpy_friction_field(3, 0.0, 1.0, fbase) is fbase

    def test_same_seed_reproduces_identical_field(self) -> None:
        a = bumpy_height_field(42, 0.005, 1.0, HeightField.flat(extent_m=10.0))
        b = bumpy_height_field(42, 0.005, 1.0, HeightField.flat(extent_m=10.0))
        assert np.array_equal(a.heights_m, b.heights_m)

    def test_different_seeds_differ(self) -> None:
        a = bumpy_height_field(1, 0.005, 1.0, HeightField.flat(extent_m=10.0))
        b = bumpy_height_field(2, 0.005, 1.0, HeightField.flat(extent_m=10.0))
        assert not np.array_equal(a.heights_m, b.heights_m)

    def test_amplitude_sets_bump_scale(self) -> None:
        field = bumpy_height_field(5, 0.01, 1.0, HeightField.flat(extent_m=20.0))
        assert field.heights_m.std() == pytest.approx(0.01, rel=0.05)

    def test_height_and_friction_streams_are_independent(self) -> None:
        # Keyed per-field streams (Tools variation-engine discipline):
        # the height draw is the same whether or not friction is drawn.
        before = bumpy_height_field(9, 0.005, 1.0, HeightField.flat(extent_m=10.0))
        bumpy_friction_field(9, 0.2, 1.0, FrictionField.uniform(extent_m=10.0))
        after = bumpy_height_field(9, 0.005, 1.0, HeightField.flat(extent_m=10.0))
        assert np.array_equal(before.heights_m, after.heights_m)

    def test_negative_seed_is_rejected_even_when_amplitude_is_zero(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            bumpy_height_field(-1, 0.0, 1.0, HeightField.flat(extent_m=10.0))
        with pytest.raises(ValueError, match="seed"):
            bumpy_friction_field(-1, 0.0, 1.0, FrictionField.uniform(extent_m=10.0))

    def test_friction_field_reproducible_and_positive(self) -> None:
        a = bumpy_friction_field(11, 0.3, 2.0, FrictionField.uniform(extent_m=10.0))
        b = bumpy_friction_field(11, 0.3, 2.0, FrictionField.uniform(extent_m=10.0))
        assert np.array_equal(a.roll_multiplier, b.roll_multiplier)
        assert np.array_equal(a.slide_multiplier, b.slide_multiplier)
        assert np.all(a.roll_multiplier > 0.0)
        # Roll and slide use distinct keyed streams.
        assert not np.array_equal(a.roll_multiplier, a.slide_multiplier)
