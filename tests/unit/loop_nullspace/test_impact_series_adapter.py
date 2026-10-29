"""Series -> rigid club state adapter and event location for the NSX-83 map."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from src.shared.python.impact_parameters import (
    ClubheadSeries,
    extract_impact_parameters,
)
from src.shared.python.impact_parameters.adapters.club_face import (
    NATIVE_CLUB_FACE,
    rigid_body_series,
)
from src.shared.python.loop_nullspace.evidence import AllocationDependence
from src.shared.python.loop_nullspace.impact_outcome_map import (
    FACE_OUTCOMES,
    OUTCOME_NAMES,
    RigidClubState,
    advance_constant_twist,
    constant_twist_flow,
    exp_so3,
    impact_outcome_map,
)
from src.shared.python.loop_nullspace.impact_series_adapter import (
    advance_to_event,
    club_state_from_series,
)
from tests.unit.loop_nullspace.test_impact_outcome_map import (
    EVENT,
    FRAME,
    SPEC,
    make_state,
)

pytestmark = pytest.mark.unit

DT = 1e-3
EVENT_INDEX = 2


def _series_around(state: RigidClubState, n: int = 5) -> ClubheadSeries:
    """Constant-twist samples; sample ``EVENT_INDEX`` is ``state`` itself."""
    steps = [advance_constant_twist(state, (k - EVENT_INDEX) * DT) for k in range(n)]
    return rigid_body_series(
        [k * DT for k in range(n)],
        np.stack([s.origin_m for s in steps]),
        np.stack([s.rotation for s in steps]),
        np.stack([s.velocity_mps for s in steps]),
        np.stack([s.angular_velocity_rps for s in steps]),
        state.face_spec,
    )


@pytest.mark.parametrize("index", [0, 1, EVENT_INDEX, 3, 4])
def test_recovers_the_constant_twist_state(index: int) -> None:
    truth = advance_constant_twist(make_state(), (index - EVENT_INDEX) * DT)
    got = club_state_from_series(_series_around(make_state()), index, SPEC)
    assert np.allclose(got.rotation, truth.rotation, atol=1e-12)
    assert np.allclose(got.origin_m, truth.origin_m, atol=1e-12)
    assert np.allclose(got.angular_velocity_rps, truth.angular_velocity_rps, atol=1e-9)
    assert np.allclose(got.velocity_mps, truth.velocity_mps, atol=1e-9)
    assert got.face_unobservable_reason is None


def test_default_spec_is_the_native_club_face() -> None:
    state = dataclasses.replace(make_state(), face_spec=NATIVE_CLUB_FACE)
    got = club_state_from_series(_series_around(state), EVENT_INDEX)
    assert got.face_spec == NATIVE_CLUB_FACE
    assert np.allclose(got.origin_m, state.origin_m, atol=1e-12)


def test_series_state_gives_the_same_outcomes_and_jacobian_as_the_source_state() -> (
    None
):
    truth = make_state()
    series = _series_around(truth)
    got = club_state_from_series(series, EVENT_INDEX, SPEC)
    ours = impact_outcome_map(got, EVENT, FRAME, flow=constant_twist_flow)
    ref = impact_outcome_map(truth, EVENT, FRAME, flow=constant_twist_flow)
    extracted = extract_impact_parameters(
        series, FRAME, impact_index=EVENT_INDEX, use_tools=False
    )
    for name, value in zip(OUTCOME_NAMES, ours.values, strict=True):
        assert value == pytest.approx(getattr(extracted, name), abs=1e-9)
    assert np.allclose(ours.values, ref.values, atol=1e-9)
    assert np.allclose(ours.jacobian, ref.jacobian, atol=1e-6)


def test_unobserved_rotation_makes_face_rows_unavailable_and_keeps_chs_and_aoa() -> (
    None
):
    reason = "mocap-matched swing: club axial rotation unobserved"
    base = _series_around(dataclasses.replace(make_state(), face_spec=NATIVE_CLUB_FACE))
    blind = ClubheadSeries(
        times_s=base.times_s,
        face_center_m=base.face_center_m,
        velocity_mps=base.velocity_mps,
        face_unobservable_reason=reason,
    )
    state = club_state_from_series(blind, EVENT_INDEX)
    assert state.face_unobservable_reason == reason
    assert np.allclose(state.origin_m, blind.face_center_m[EVENT_INDEX])
    assert np.allclose(state.velocity_mps, blind.velocity_mps[EVENT_INDEX])
    out = impact_outcome_map(
        state, EVENT, FRAME, flow=constant_twist_flow, check_on_event=False
    )
    for name in FACE_OUTCOMES:
        assert reason in out.unavailable[name]
        assert (
            out.records[name].allocation_dependence is AllocationDependence.UNAVAILABLE
        )
    for name in ("attack_angle_deg", "clubhead_speed_mps"):
        assert name not in out.unavailable
        assert np.isfinite(out.values[OUTCOME_NAMES.index(name)])


def test_rejects_wrong_types_and_bad_indices() -> None:
    series = _series_around(make_state())
    with pytest.raises(TypeError, match="ClubheadSeries"):
        club_state_from_series("series", 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ClubFaceSpec"):
        club_state_from_series(series, 1, "spec")  # type: ignore[arg-type]
    for bad in (-1, len(series)):
        with pytest.raises(ValueError, match="out of range"):
            club_state_from_series(series, bad, SPEC)


def test_partial_axes_are_rejected_not_guessed() -> None:
    series = _series_around(make_state())
    partial = dataclasses.replace(series, toe_axis=None, grip_axis=None)
    with pytest.raises(ValueError, match="toe_axis and grip_axis"):
        club_state_from_series(partial, 1, SPEC)


def test_degenerate_axes_are_rejected() -> None:
    series = _series_around(make_state())
    bad = dataclasses.replace(series, toe_axis=np.array(series.face_normal))
    with pytest.raises(ValueError, match="orthonormal"):
        club_state_from_series(bad, 1, SPEC)


def test_half_turn_between_samples_is_rejected() -> None:
    state = make_state()
    flipped = dataclasses.replace(
        state, rotation=exp_so3([0.0, 0.0, math.pi]) @ state.rotation
    )
    steps = [state, state, flipped, flipped]
    series = rigid_body_series(
        [k * DT for k in range(4)],
        np.stack([s.origin_m for s in steps]),
        np.stack([s.rotation for s in steps]),
        np.stack([s.velocity_mps for s in steps]),
        np.stack([s.angular_velocity_rps for s in steps]),
        state.face_spec,
    )
    with pytest.raises(ValueError, match="half turn"):
        club_state_from_series(series, 1, SPEC)


def test_advance_to_event_lands_on_the_plane_at_the_analytic_time() -> None:
    on_event = make_state()
    gap = 0.004
    crossing = advance_to_event(advance_constant_twist(on_event, -gap), EVENT)
    assert abs(EVENT.value(crossing.state)) < 1e-12
    assert crossing.time_s == pytest.approx(gap, abs=1e-9)
    assert np.allclose(crossing.state.origin_m, on_event.origin_m, atol=1e-9)
    assert np.allclose(crossing.state.rotation, on_event.rotation, atol=1e-9)


def test_advance_to_event_keeps_an_on_event_state_unchanged() -> None:
    state = make_state()
    crossing = advance_to_event(state, EVENT)
    assert crossing.time_s == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(crossing.state.origin_m, state.origin_m, atol=1e-12)


def test_advance_to_event_fails_when_the_face_moves_away() -> None:
    receding = dataclasses.replace(
        make_state(), velocity_mps=-np.asarray(make_state().velocity_mps)
    )
    late = advance_constant_twist(receding, 0.002)
    with pytest.raises(ValueError, match="does not reach"):
        advance_to_event(late, EVENT)


def test_advance_to_event_rejects_a_crossing_beyond_the_horizon() -> None:
    far = advance_constant_twist(make_state(), -0.5)
    with pytest.raises(ValueError, match="does not reach"):
        advance_to_event(far, EVENT, max_horizon_s=0.05)
    with pytest.raises(ValueError, match="max_horizon_s"):
        advance_to_event(far, EVENT, max_horizon_s=0.0)


def test_advance_to_event_requires_a_ball_plane_event() -> None:
    with pytest.raises(TypeError, match="BallPlaneEvent"):
        advance_to_event(make_state(), "event")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RigidClubState"):
        advance_to_event("state", EVENT)  # type: ignore[arg-type]
