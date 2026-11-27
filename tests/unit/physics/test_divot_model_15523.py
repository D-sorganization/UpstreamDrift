"""Divot modelling for iron shots and fat-shot detection (#15523).

The sole path is a parabolic arc about its low point. When the low point is
ahead of the ball the divot starts after the ball (a clean, ball-first iron
strike). When the arc reaches the turf *before* the ball the strike is fat:
the pre-ball turf work bleeds clubhead speed and spin.
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.core.contracts import PreconditionError
from src.shared.python.physics.divot_model import (
    DivotResult,
    SwingArc,
    compute_divot,
)
from src.shared.python.physics.terrain.materials import MATERIALS

FAIRWAY = MATERIALS["fairway"]
SPEED = 40.0


def _arc(low_x: float, low_z: float) -> SwingArc:
    return SwingArc(low_point_offset_m=low_x, low_point_height_m=low_z)


@pytest.mark.unit
class TestNoContact:
    def test_arc_above_turf_makes_no_divot(self) -> None:
        r = compute_divot(_arc(0.05, 0.01), SPEED, FAIRWAY)
        assert isinstance(r, DivotResult)
        assert r.has_divot is False
        assert r.is_fat_shot is False
        assert r.volume_m3 == 0.0
        assert r.speed_retention == 1.0
        assert r.spin_retention == 1.0


@pytest.mark.unit
class TestCleanIronStrike:
    def test_divot_starts_after_ball_and_is_not_fat(self) -> None:
        r = compute_divot(_arc(0.30, -0.01), SPEED, FAIRWAY)
        assert r.has_divot
        assert r.entry_offset_m > 0.0
        assert r.is_fat_shot is False
        assert r.pre_ball_length_m == 0.0
        assert r.speed_retention == 1.0
        assert r.spin_retention == 1.0
        assert r.max_depth_m == pytest.approx(0.01)
        assert r.volume_m3 > 0.0

    def test_geometry_is_symmetric_about_low_point(self) -> None:
        r = compute_divot(_arc(0.12, -0.015), SPEED, FAIRWAY, arc_radius_m=1.5)
        half = math.sqrt(2 * 1.5 * 0.015)
        assert r.entry_offset_m == pytest.approx(0.12 - half)
        assert r.exit_offset_m == pytest.approx(0.12 + half)
        assert r.length_m == pytest.approx(2 * half)


@pytest.mark.unit
class TestFatShot:
    def test_pre_ball_contact_is_fat_and_loses_speed_and_spin(self) -> None:
        r = compute_divot(_arc(-0.05, -0.02), SPEED, FAIRWAY)
        assert r.entry_offset_m < 0.0
        assert r.is_fat_shot is True
        assert r.pre_ball_length_m == pytest.approx(-r.entry_offset_m)
        assert r.energy_loss_j > 0.0
        assert 0.0 < r.speed_retention < 1.0
        assert 0.0 < r.spin_retention <= r.speed_retention

    def test_fatter_means_more_loss(self) -> None:
        slightly = compute_divot(_arc(0.0, -0.02), SPEED, FAIRWAY)
        very = compute_divot(_arc(-0.10, -0.02), SPEED, FAIRWAY)
        assert very.speed_retention < slightly.speed_retention
        assert very.energy_loss_j > slightly.energy_loss_j

    def test_deeper_cut_means_more_loss(self) -> None:
        shallow = compute_divot(_arc(-0.05, -0.01), SPEED, FAIRWAY)
        deep = compute_divot(_arc(-0.05, -0.03), SPEED, FAIRWAY)
        assert deep.energy_loss_j > shallow.energy_loss_j

    def test_harder_turf_costs_more_energy(self) -> None:
        soft = MATERIALS["rough"]
        hard = MATERIALS["green"]
        a = compute_divot(_arc(-0.05, -0.02), SPEED, soft)
        b = compute_divot(_arc(-0.05, -0.02), SPEED, hard)
        assert (a.energy_loss_j > b.energy_loss_j) == (soft.hardness > hard.hardness)

    def test_stalls_instead_of_negative_speed(self) -> None:
        r = compute_divot(_arc(-0.4, -0.08), 2.0, FAIRWAY)
        assert r.speed_retention == 0.0
        assert r.spin_retention == 0.0


@pytest.mark.unit
class TestContracts:
    @pytest.mark.parametrize("speed", [0.0, -1.0, float("nan")])
    def test_rejects_bad_speed(self, speed: float) -> None:
        with pytest.raises(PreconditionError):
            compute_divot(_arc(0.0, -0.01), speed, FAIRWAY)

    def test_rejects_bad_geometry(self) -> None:
        with pytest.raises(PreconditionError):
            compute_divot(_arc(0.0, -0.01), SPEED, FAIRWAY, arc_radius_m=0.0)
        with pytest.raises(PreconditionError):
            compute_divot(_arc(0.0, -0.01), SPEED, FAIRWAY, divot_width_m=-1.0)
