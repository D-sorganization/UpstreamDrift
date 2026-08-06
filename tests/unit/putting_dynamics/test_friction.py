"""Friction-law tests for putting_dynamics (#8345 P2)."""

from __future__ import annotations

import math

import pytest

from src.shared.python.putting_dynamics import (
    FrictionParams,
    grain_factor,
    is_static_hold,
    rolling_mu,
    rolling_mu_to_stimp,
    sliding_mu,
    stimp_to_rolling_mu,
)

pytestmark = pytest.mark.unit


def _params(**overrides: float) -> FrictionParams:
    defaults: dict[str, float] = {"mu_roll0": stimp_to_rolling_mu(10.0)}
    defaults.update(overrides)
    return FrictionParams(**defaults)


class TestStimpConversion:
    def test_stimp_10_is_in_published_band(self) -> None:
        # 0.05-0.07 for tournament greens (Tools swing_sim.putting.roll
        # derivation, restated).
        assert 0.05 <= stimp_to_rolling_mu(10.0) <= 0.07

    def test_round_trip_is_exact(self) -> None:
        for stimp in (4.0, 8.0, 10.0, 13.0):
            assert rolling_mu_to_stimp(stimp_to_rolling_mu(stimp)) == pytest.approx(
                stimp, rel=1e-12
            )

    def test_faster_green_has_lower_mu(self) -> None:
        assert stimp_to_rolling_mu(13.0) < stimp_to_rolling_mu(8.0)

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="stimp"):
            stimp_to_rolling_mu(2.0)


class TestVelocityDependentRolling:
    def test_k_v_zero_recovers_constant_law(self) -> None:
        params = _params()
        assert rolling_mu(params, 3.0) == pytest.approx(params.mu_roll0, rel=1e-12)

    def test_linear_velocity_scaling(self) -> None:
        params = _params(k_v_per_mps=0.1)
        assert rolling_mu(params, 2.0) == pytest.approx(
            params.mu_roll0 * 1.2, rel=1e-12
        )

    def test_spatial_multiplier_scales(self) -> None:
        params = _params()
        assert rolling_mu(params, 1.0, spatial_multiplier=1.5) == pytest.approx(
            params.mu_roll0 * 1.5, rel=1e-12
        )


class TestGrainAnisotropy:
    def test_disabled_grain_is_unity(self) -> None:
        assert grain_factor(_params(), 1.234) == 1.0

    def test_extremes_at_with_and_against_grain(self) -> None:
        params = _params(grain_strength=0.2, grain_direction_rad=0.5)
        assert grain_factor(params, 0.5) == pytest.approx(0.8, rel=1e-12)
        assert grain_factor(params, 0.5 + math.pi) == pytest.approx(1.2, rel=1e-12)

    def test_sliding_mu_sees_grain(self) -> None:
        params = _params(grain_strength=0.2, grain_direction_rad=0.0)
        assert sliding_mu(params, 1.0, 0.0) == pytest.approx(
            params.mu_slide * 0.8, rel=1e-12
        )


class TestStaticHold:
    def test_shallow_slope_holds(self) -> None:
        params = _params()
        assert is_static_hold(params, 0.02)

    def test_steep_slope_releases(self) -> None:
        params = _params()
        assert not is_static_hold(params, params.mu_static + 0.01)

    def test_multiplier_shifts_the_bound(self) -> None:
        params = _params()
        slope = params.mu_static * 1.1
        assert not is_static_hold(params, slope)
        assert is_static_hold(params, slope, spatial_multiplier=1.5)

    def test_mu_static_must_cover_rolling(self) -> None:
        with pytest.raises(ValueError, match="mu_static"):
            FrictionParams(mu_roll0=0.06, mu_static=0.05)
