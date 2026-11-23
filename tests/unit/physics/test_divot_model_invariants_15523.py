"""Closed-form and boundary invariants of the iron divot model (Refs #15523).

The merged behavioural tests check directions (fatter => more loss).  These pin
the *quantities*: the pre-ball turf work against its analytic integral, the
clean/fat boundary, linear scaling in width and hardness, and the case where
the whole divot lies in front of the ball.
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.physics.divot_model import (
    DEFAULT_ARC_RADIUS_M,
    DEFAULT_DIVOT_WIDTH_M,
    DEFAULT_EFFECTIVE_MASS_KG,
    TURF_SHEAR_STRENGTH_PA_AT_FULL_HARDNESS,
    SwingArc,
    compute_divot,
)
from src.shared.python.physics.terrain.materials import MATERIALS

pytestmark = pytest.mark.unit

FAIRWAY = MATERIALS["fairway"]
SPEED = 40.0
R = DEFAULT_ARC_RADIUS_M


def _arc(low_x: float, low_z: float) -> SwingArc:
    return SwingArc(low_point_offset_m=low_x, low_point_height_m=low_z)


def _closed_form_loss_j(
    low_x: float, low_z: float, hardness: float, width: float = DEFAULT_DIVOT_WIDTH_M
) -> float:
    """Integral of depth over [entry, min(0, exit)] times shear * width."""
    half = math.sqrt(-2.0 * R * low_z)
    entry, exit_ = low_x - half, low_x + half
    hi = min(0.0, exit_)

    def antiderivative(x: float) -> float:
        u = x - low_x
        return -low_z * x - u**3 / (6.0 * R)

    area = antiderivative(hi) - antiderivative(entry)
    return TURF_SHEAR_STRENGTH_PA_AT_FULL_HARDNESS * hardness * width * area


class TestClosedForm:
    @pytest.mark.parametrize(
        ("low_x", "low_z"),
        [(-0.02, -0.02), (-0.05, -0.03), (0.0, -0.025), (-0.10, -0.04)],
    )
    def test_energy_loss_matches_analytic_integral(
        self, low_x: float, low_z: float
    ) -> None:
        r = compute_divot(_arc(low_x, low_z), SPEED, FAIRWAY)
        expected = _closed_form_loss_j(low_x, low_z, FAIRWAY.hardness)
        assert r.energy_loss_j == pytest.approx(expected, rel=1e-4)

    def test_retention_follows_energy_balance(self) -> None:
        r = compute_divot(_arc(-0.05, -0.03), SPEED, FAIRWAY)
        kinetic = 0.5 * DEFAULT_EFFECTIVE_MASS_KG * SPEED**2
        assert r.speed_retention == pytest.approx(
            math.sqrt(1.0 - r.energy_loss_j / kinetic)
        )
        assert r.spin_retention == pytest.approx(r.speed_retention**2)

    def test_volume_is_two_thirds_length_depth_width(self) -> None:
        r = compute_divot(_arc(0.08, -0.02), SPEED, FAIRWAY)
        assert r.volume_m3 == pytest.approx(
            (2.0 / 3.0) * r.length_m * r.max_depth_m * DEFAULT_DIVOT_WIDTH_M
        )


class TestBoundary:
    def test_divot_entering_exactly_at_the_ball_is_clean(self) -> None:
        half = math.sqrt(-2.0 * R * -0.02)
        r = compute_divot(_arc(half, -0.02), SPEED, FAIRWAY)
        assert r.entry_offset_m == pytest.approx(0.0, abs=1e-12)
        assert not r.is_fat_shot
        assert r.energy_loss_j == 0.0
        assert r.speed_retention == 1.0
        assert r.spin_retention == 1.0

    def test_loss_is_continuous_across_the_clean_fat_boundary(self) -> None:
        half = math.sqrt(-2.0 * R * -0.02)
        just_fat = compute_divot(_arc(half - 1e-4, -0.02), SPEED, FAIRWAY)
        assert just_fat.is_fat_shot
        assert 0.0 < just_fat.energy_loss_j < 1e-3
        assert just_fat.speed_retention == pytest.approx(1.0, abs=1e-4)


class TestScaling:
    def test_loss_is_linear_in_width(self) -> None:
        narrow = compute_divot(_arc(-0.05, -0.03), SPEED, FAIRWAY, divot_width_m=0.05)
        wide = compute_divot(_arc(-0.05, -0.03), SPEED, FAIRWAY, divot_width_m=0.10)
        assert wide.energy_loss_j == pytest.approx(2.0 * narrow.energy_loss_j)

    def test_loss_is_linear_in_hardness(self) -> None:
        soft = MATERIALS["rough"]
        a = compute_divot(_arc(-0.05, -0.03), SPEED, soft)
        b = compute_divot(_arc(-0.05, -0.03), SPEED, FAIRWAY)
        assert a.energy_loss_j / soft.hardness == pytest.approx(
            b.energy_loss_j / FAIRWAY.hardness
        )

    def test_volume_ignores_hardness(self) -> None:
        a = compute_divot(_arc(-0.05, -0.03), SPEED, MATERIALS["rough"])
        b = compute_divot(_arc(-0.05, -0.03), SPEED, FAIRWAY)
        assert a.volume_m3 == b.volume_m3


class TestDivotEndingBeforeBall:
    """Sole leaves the turf before the ball: only the cut span costs energy."""

    def test_only_the_in_turf_span_before_the_ball_is_charged(self) -> None:
        low_x, low_z = -0.30, -0.01  # half length ~0.173 -> exit ~ -0.127 < 0
        r = compute_divot(_arc(low_x, low_z), SPEED, FAIRWAY)
        assert r.exit_offset_m < 0.0
        assert r.is_fat_shot
        # Whole divot is charged, and nothing beyond its exit point.
        full = _closed_form_loss_j(low_x, low_z, FAIRWAY.hardness)
        assert r.energy_loss_j == pytest.approx(full, rel=1e-4)
        # pre_ball_length_m is entry-to-ball distance (pinned by the merged
        # tests), which exceeds the contact length here.
        assert r.pre_ball_length_m == pytest.approx(-r.entry_offset_m)
        assert r.pre_ball_length_m > r.length_m
