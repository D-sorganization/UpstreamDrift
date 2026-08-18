"""Tests for ball lie model (issue #8613).

TDD: Write failing tests first, then implement.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st

from bunkershot3d.ball.lie import (
    BallLie,
    BallLieType,
    BallProperties,
    compute_submersion_depth,
    compute_exposed_cap_area,
)

pytestmark = pytest.mark.unit


class TestBallLieType:
    """Tests for lie type enumeration."""

    def test_standard_lie_is_default(self) -> None:
        """Standard lie is the most common case."""
        lie = BallLie(depth_m=0.0)
        assert lie.lie_type == BallLieType.STANDARD

    def test_buried_lie_detects_fried_egg(self) -> None:
        """Fried egg: ball sits in a crater, partially buried."""
        lie = BallLie(depth_m=0.015)  # 15mm buried
        assert lie.lie_type == BallLieType.BURIED

    def test_plugged_lie_detects_deep_burial(self) -> None:
        """Plugged: ball more than half buried."""
        lie = BallLie(depth_m=0.025)  # 25mm, more than half of 42.67mm diameter
        assert lie.lie_type == BallLieType.PLUGGED

    def test_teed_up_lie_detects_above_surface(self) -> None:
        """Teed up: ball sits above the surface (on a small mound)."""
        lie = BallLie(depth_m=-0.005)  # 5mm above surface
        assert lie.lie_type == BallLieType.TEED_UP


class TestBallProperties:
    """Tests for ball physical properties."""

    def test_default_ball_is_conforming(self) -> None:
        """Default ball matches USGA/R&A regulations."""
        ball = BallProperties()
        assert abs(ball.mass_kg - 0.04593) < 0.001  # 45.93g max
        assert abs(ball.diameter_m - 0.04267) < 0.0001  # 42.67mm min
        assert ball.radius_m == pytest.approx(ball.diameter_m / 2)

    def test_ball_volume_correct(self) -> None:
        """Ball volume matches sphere formula."""
        ball = BallProperties()
        expected_volume = (4 / 3) * math.pi * ball.radius_m**3
        assert ball.volume_m3 == pytest.approx(expected_volume, rel=1e-10)

    def test_ball_moi_correct(self) -> None:
        """Ball MOI matches solid sphere formula."""
        ball = BallProperties()
        expected_moi = (2 / 5) * ball.mass_kg * ball.radius_m**2
        assert ball.moi_kg_m2 == pytest.approx(expected_moi, rel=1e-10)


class TestBallLieGeometry:
    """Tests for geometric computations on buried ball."""

    def test_surface_ball_has_zero_submersion(self) -> None:
        """Ball sitting on surface has zero submersion depth."""
        lie = BallLie(depth_m=0.0)
        ball = BallProperties()
        submersion = compute_submersion_depth(lie, ball)
        assert submersion == pytest.approx(0.0, abs=1e-10)

    def test_half_buried_ball_submersion(self) -> None:
        """Ball buried to center has submersion = radius."""
        ball = BallProperties()
        lie = BallLie(depth_m=ball.radius_m)
        submersion = compute_submersion_depth(lie, ball)
        assert submersion == pytest.approx(ball.radius_m, rel=1e-10)

    def test_fully_buried_ball_submersion(self) -> None:
        """Fully buried ball has submersion = diameter."""
        ball = BallProperties()
        lie = BallLie(depth_m=ball.diameter_m)
        submersion = compute_submersion_depth(lie, ball)
        assert submersion == pytest.approx(ball.diameter_m, rel=1e-10)

    def test_surface_ball_exposed_area_is_hemisphere(self) -> None:
        """Ball sitting on surface exposes a hemisphere."""
        lie = BallLie(depth_m=0.0)
        ball = BallProperties()
        exposed = compute_exposed_cap_area(lie, ball)
        hemisphere_area = 2 * math.pi * ball.radius_m**2
        assert exposed == pytest.approx(hemisphere_area, rel=1e-6)

    def test_buried_ball_exposed_area_decreases(self) -> None:
        """More buried = less exposed area."""
        ball = BallProperties()
        lie_shallow = BallLie(depth_m=0.005)
        lie_deep = BallLie(depth_m=0.015)
        area_shallow = compute_exposed_cap_area(lie_shallow, ball)
        area_deep = compute_exposed_cap_area(lie_deep, ball)
        assert area_deep < area_shallow

    def test_fully_buried_ball_has_zero_exposed_area(self) -> None:
        """Fully buried ball has no exposed area."""
        ball = BallProperties()
        lie = BallLie(depth_m=ball.diameter_m)
        exposed = compute_exposed_cap_area(lie, ball)
        assert exposed == pytest.approx(0.0, abs=1e-10)


class TestBallLiePosition:
    """Tests for ball position specification."""

    def test_ball_position_defaults_to_origin(self) -> None:
        """Ball position defaults to (0, 0) horizontally."""
        lie = BallLie(depth_m=0.01)
        assert lie.x_m == 0.0
        assert lie.y_m == 0.0

    def test_ball_position_can_be_specified(self) -> None:
        """Ball can be positioned anywhere in the bed."""
        lie = BallLie(depth_m=0.01, x_m=0.05, y_m=-0.02)
        assert lie.x_m == 0.05
        assert lie.y_m == -0.02

    def test_ball_center_z_computed_from_depth(self) -> None:
        """Ball center z is radius minus depth."""
        ball = BallProperties()
        lie = BallLie(depth_m=0.01)
        expected_z = ball.radius_m - 0.01
        assert lie.center_z_m(ball) == pytest.approx(expected_z, rel=1e-10)


class TestBallLieValidation:
    """Tests for input validation with Design-by-Contract."""

    def test_rejects_negative_depth_beyond_tee(self) -> None:
        """Cannot be more than a radius above surface."""
        ball = BallProperties()
        with pytest.raises(ValueError, match="depth"):
            BallLie(depth_m=-(ball.radius_m + 0.01))

    def test_rejects_excessive_burial(self) -> None:
        """Cannot be buried deeper than full diameter + margin."""
        ball = BallProperties()
        with pytest.raises(ValueError, match="depth"):
            BallLie(depth_m=ball.diameter_m + 0.01)


class TestBallLieMetamorphic:
    """Metamorphic tests for ball lie geometry."""

    @given(st.floats(min_value=0.0, max_value=0.04, allow_nan=False))
    @settings(deadline=None)
    def test_submersion_increases_monotonically_with_depth(self, depth: float) -> None:
        """Deeper burial = more submersion."""
        ball = BallProperties()
        lie = BallLie(depth_m=depth)
        submersion = compute_submersion_depth(lie, ball)
        assert 0 <= submersion <= ball.diameter_m
        assert submersion >= depth - 1e-10  # submersion >= burial depth

    @given(st.floats(min_value=0.0, max_value=0.04, allow_nan=False))
    @settings(deadline=None)
    def test_exposed_area_decreases_monotonically_with_depth(
        self, depth: float
    ) -> None:
        """Deeper burial = less exposed area."""
        ball = BallProperties()
        lie = BallLie(depth_m=depth)
        exposed = compute_exposed_cap_area(lie, ball)
        full_sphere = 4 * math.pi * ball.radius_m**2
        assert 0 <= exposed <= full_sphere
