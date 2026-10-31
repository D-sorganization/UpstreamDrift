"""Tests for the virtual USGA Stimpmeter (epic #14141, putting v4 step 4a).

No snapshotted model outputs: every assertion is either a closed-form
check against an externally sourced constant, a round-trip consistency
check against the existing stimp-to-friction law, or a qualitative
ordering property of the USGA two-direction procedure.
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.putting_dynamics.friction import (
    STIMP_RELEASE_SPEED_MPS,
    FrictionParams,
    stimp_to_rolling_mu,
)
from src.shared.python.putting_dynamics.stimpmeter import (
    STIMPMETER_NOTCH_DISTANCE_M,
    STIMPMETER_RELEASE_ANGLE_DEG,
    measure_stimp_ft,
    ramp_release_speed_mps,
    stimpmeter_rollout_ft,
)
from src.shared.python.putting_dynamics.surfaces import (
    FrictionField,
    HeightField,
    SurfaceSpec,
)

pytestmark = pytest.mark.unit

#: Measurement tolerance a USGA release-speed derivation should land
#: within (module docstring finding); see the strict xfail below.
_RELEASE_SPEED_TOL_MPS = 0.05


def _planar_surface(grade_percent: float = 2.0, stimp_ft: float = 10.0) -> SurfaceSpec:
    """A uniform planar green, downhill along +x (test helper)."""
    return SurfaceSpec(
        height=HeightField.planar(grade_percent=grade_percent, aspect_deg=0.0),
        friction_field=FrictionField.uniform(),
        friction=FrictionParams(mu_roll0=stimp_to_rolling_mu(stimp_ft)),
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ramp_release_speed_mps() at the only uncited-but-honest default "
        "(groove_contact_half_width_m=0, the flat-track limit) gives "
        "~1.911 m/s for the USGA 30in/20deg geometry, ~0.08 m/s above the "
        "measured 1.83 m/s -- outside the +-0.05 m/s band. A nonzero "
        "groove width can close the gap, but no independently sourced "
        "USGA groove width was found (see stimpmeter.py module docstring); "
        "tuning the width to hit 1.83 m/s would be fitting, not deriving, "
        "so this is left as a documented open finding rather than patched "
        "to pass."
    ),
)
def test_ramp_release_speed_reproduces_usga_measurement() -> None:
    v_mps = ramp_release_speed_mps(
        notch_distance_m=STIMPMETER_NOTCH_DISTANCE_M,
        angle_deg=STIMPMETER_RELEASE_ANGLE_DEG,
    )
    assert abs(v_mps - STIMP_RELEASE_SPEED_MPS) <= _RELEASE_SPEED_TOL_MPS


def test_ramp_release_speed_default_is_finite_and_positive() -> None:
    v_mps = ramp_release_speed_mps()
    assert math.isfinite(v_mps)
    assert v_mps > 0.0


def test_ramp_release_speed_narrower_groove_lowers_speed() -> None:
    # A larger groove contact half-width raises the effective rotational
    # inertia (narrower r_eff), so the release speed must decrease
    # monotonically (module docstring derivation).
    wide = ramp_release_speed_mps(groove_contact_half_width_m=0.0)
    narrow = ramp_release_speed_mps(groove_contact_half_width_m=0.4 * 0.02133)
    assert narrow < wide


@pytest.mark.parametrize("stimp_ft", [8.0, 10.0, 12.0, 13.0])
def test_measure_stimp_ft_definition_consistency(stimp_ft: float) -> None:
    """measure_stimp_ft() on a flat_uniform(stimp_ft=S) surface is ~ S.

    Consistency check, not a physics claim: the USGA stimp reading is
    this model's *input* to :func:`~.friction.stimp_to_rolling_mu`, so
    rolling a calibration ball back out on that same surface should
    recover the input -- the small residual is the v_stop truncation
    and skid-to-roll discretization in :func:`~.solver.simulate_ball`.
    """
    surface = SurfaceSpec.flat_uniform(stimp_ft=stimp_ft)
    measured_ft = measure_stimp_ft(surface)
    assert math.isclose(measured_ft, stimp_ft, rel_tol=1e-3)


def test_two_direction_average_differs_from_single_direction_on_slope() -> None:
    surface = _planar_surface(grade_percent=2.0, stimp_ft=10.0)
    downhill_ft = stimpmeter_rollout_ft(surface, heading_rad=0.0)
    uphill_ft = stimpmeter_rollout_ft(surface, heading_rad=math.pi)
    averaged_ft = measure_stimp_ft(surface, heading_rad=0.0)

    assert downhill_ft != pytest.approx(uphill_ft)
    assert averaged_ft != pytest.approx(downhill_ft)
    assert averaged_ft != pytest.approx(uphill_ft)


def test_two_direction_average_lies_between_the_single_directions() -> None:
    surface = _planar_surface(grade_percent=2.0, stimp_ft=10.0)
    downhill_ft = stimpmeter_rollout_ft(surface, heading_rad=0.0)
    uphill_ft = stimpmeter_rollout_ft(surface, heading_rad=math.pi)
    averaged_ft = measure_stimp_ft(surface, heading_rad=0.0)

    low, high = sorted((downhill_ft, uphill_ft))
    assert low < averaged_ft < high


class TestRampReleaseSpeedPreconditions:
    def test_rejects_non_positive_notch_distance(self) -> None:
        with pytest.raises(ValueError):
            ramp_release_speed_mps(notch_distance_m=0.0)

    def test_rejects_angle_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            ramp_release_speed_mps(angle_deg=90.0)
        with pytest.raises(ValueError):
            ramp_release_speed_mps(angle_deg=0.0)

    def test_rejects_groove_width_at_or_beyond_ball_radius(self) -> None:
        with pytest.raises(ValueError):
            ramp_release_speed_mps(groove_contact_half_width_m=0.03)

    def test_rejects_negative_groove_width(self) -> None:
        with pytest.raises(ValueError):
            ramp_release_speed_mps(groove_contact_half_width_m=-0.001)


class TestRolloutAndMeasurePreconditions:
    def test_rollout_rejects_non_positive_release_speed(self) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        with pytest.raises(ValueError):
            stimpmeter_rollout_ft(surface, release_speed_mps=0.0)

    def test_measure_rejects_non_finite_heading(self) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        with pytest.raises(ValueError):
            measure_stimp_ft(surface, heading_rad=math.nan)
