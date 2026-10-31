"""Turf-model tests (#14141 step 4b): closed forms and invariants only."""

from __future__ import annotations

import pathlib

import pytest

from src.shared.python.core.physics_constants import GOLF_BALL_RADIUS_M, GRAVITY_M_S2
from src.shared.python.putting_dynamics import (
    BallState,
    GrassType,
    TurfCondition,
    TurfSpec,
    simulate_ball,
    stimp_to_rolling_mu,
)
from src.shared.python.putting_dynamics.friction import STIMP_RELEASE_SPEED_MPS

pytestmark = pytest.mark.unit

_G = float(GRAVITY_M_S2)
_R = float(GOLF_BALL_RADIUS_M)
_FOOT_M = 0.3048
_PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[3] / (
    "src/shared/python/putting_dynamics"
)


class TestLegacyLawNotPorted:
    def test_0_196_does_not_appear_in_the_shared_package(self) -> None:
        """ud-legacy-roll/1 stays in the engine (ADR-0045)."""
        offenders = [
            path
            for path in _PACKAGE_DIR.glob("*.py")
            if "0.196" in path.read_text(encoding="utf-8")
        ]
        assert offenders == []


class TestSingleLaw:
    def test_normal_condition_uses_the_usga_law_exactly(self) -> None:
        spec = TurfSpec(stimp_ft=10.0, condition=TurfCondition.NORMAL)
        assert spec.to_friction_params().mu_roll0 == pytest.approx(
            stimp_to_rolling_mu(10.0), rel=1e-12
        )

    def test_usga_stimpmeter_definition_closed_form(self) -> None:
        """USGA definition restated: v^2 / (2 g mu S) == 1 at NORMAL."""
        for stimp in (4.0, 8.0, 10.0, 13.0, 16.0):
            mu = (
                TurfSpec(stimp_ft=stimp, condition=TurfCondition.NORMAL)
                .to_friction_params()
                .mu_roll0
            )
            ratio = STIMP_RELEASE_SPEED_MPS**2 / (2.0 * _G * mu * stimp * _FOOT_M)
            assert ratio == pytest.approx(1.0, rel=1e-9)

    def test_condition_multiplier_applied_exactly_once(self) -> None:
        base = stimp_to_rolling_mu(10.0)
        spec = TurfSpec(stimp_ft=10.0, condition=TurfCondition.WET)
        assert spec.to_friction_params().mu_roll0 == pytest.approx(
            base * TurfCondition.WET.friction_multiplier, rel=1e-12
        )


class TestConditionOrdering:
    def test_wet_is_slower_than_dry(self) -> None:
        dry = TurfSpec(stimp_ft=10.0, condition=TurfCondition.DRY)
        wet = TurfSpec(stimp_ft=10.0, condition=TurfCondition.WET)
        assert wet.to_friction_params().mu_roll0 > dry.to_friction_params().mu_roll0

    def test_frosted_is_the_slowest_condition(self) -> None:
        frosted_mu = (
            TurfSpec(stimp_ft=10.0, condition=TurfCondition.FROSTED)
            .to_friction_params()
            .mu_roll0
        )
        for condition in TurfCondition:
            if condition is TurfCondition.FROSTED:
                continue
            other_mu = (
                TurfSpec(stimp_ft=10.0, condition=condition)
                .to_friction_params()
                .mu_roll0
            )
            assert frosted_mu >= other_mu


class TestGrainMapping:
    def test_grain_angle_passes_through_unchanged(self) -> None:
        spec = TurfSpec(stimp_ft=10.0, grain_strength=0.2, grain_direction_rad=0.3)
        params = spec.to_friction_params()
        assert (params.grain_strength, params.grain_direction_rad) == (0.2, 0.3)

    def test_grass_type_supplies_the_grain_default(self) -> None:
        spec = TurfSpec(stimp_ft=10.0, grass_type=GrassType.BERMUDA)
        assert spec.grain_strength == GrassType.BERMUDA.default_grain_strength


class TestPreconditions:
    def test_stimp_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="stimp_ft"):
            TurfSpec(stimp_ft=2.0)

    def test_grain_strength_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="grain_strength"):
            TurfSpec(stimp_ft=10.0, grain_strength=0.95)

    def test_compaction_factor_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="compaction_factor"):
            TurfSpec(stimp_ft=10.0, compaction_factor=0.0)

    def test_unknown_preset_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            TurfSpec.preset("not_a_real_preset")

    def test_known_presets_construct(self) -> None:
        for name in ("tournament_fast", "municipal_slow"):
            spec = TurfSpec.preset(name)
            assert spec.to_friction_params().mu_roll0 > 0.0


class TestSurfaceSpecDrivesTheSolver:
    def test_faster_green_rolls_farther(self) -> None:
        slow = TurfSpec(stimp_ft=8.0, condition=TurfCondition.NORMAL)
        fast = TurfSpec(stimp_ft=13.0, condition=TurfCondition.NORMAL)
        v0 = 1.5
        ball = BallState(x_m=0.0, y_m=0.0, vx_mps=v0, spin_rad_s=v0 / _R)
        slow_result = simulate_ball(ball, slow.to_surface_spec())
        fast_result = simulate_ball(ball, fast.to_surface_spec())
        assert fast_result.total_distance_m > slow_result.total_distance_m

    def test_to_surface_spec_respects_extent_override(self) -> None:
        spec = TurfSpec(stimp_ft=10.0)
        default_surface = spec.to_surface_spec()
        narrow_surface = spec.to_surface_spec(extent_m=20.0)
        assert (
            narrow_surface.height.heights_m.shape
            != default_surface.height.heights_m.shape
        )
        assert narrow_surface.height.heights_m.shape == (81, 81)
