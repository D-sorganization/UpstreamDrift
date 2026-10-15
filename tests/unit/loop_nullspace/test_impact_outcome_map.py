"""Event-triggered impact outcome map on a synthetic rigid clubhead (NSX-83)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.impact_parameters import TargetFrame
from src.shared.python.impact_parameters.adapters.club_face import ClubFaceSpec
from src.shared.python.impact_parameters.tools_gateway import (
    ToolsDeliveryGateway,
    ToolsDeliveryUnavailableError,
)
from src.shared.python.loop_nullspace.evidence import AllocationDependence
from src.shared.python.loop_nullspace.impact_outcome_map import (
    FACE_OUTCOMES,
    OUTCOME_NAMES,
    BallPlaneEvent,
    RigidClubState,
    advance_constant_twist,
    constant_twist_flow,
    impact_outcome_map,
    rigid_kinematic_jacobian,
    tools_ball_flight_map,
)

pytestmark = pytest.mark.unit

FRAME = TargetFrame()
SPEC = ClubFaceSpec(face_center_body_m=(0.01, 0.04, -0.02))
EVENT = BallPlaneEvent(
    point_m=tuple(FRAME.ball_m), normal=tuple(float(x) for x in FRAME.x_t)
)


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def make_state(
    *, v_scale: float = 1.0, reason: str | None = None, graze: bool = False
) -> RigidClubState:
    """Club at the event: face centre exactly on the ball plane."""
    rot = (
        _rot(np.array([0.0, 0.0, 1.0]), -math.pi / 2 + 0.06)
        @ _rot(np.array([1.0, 0.2, 0.0]), 0.12)
        @ _rot(np.array([0.0, 1.0, 0.3]), -0.05)
    )
    centre = rot @ np.asarray(SPEC.face_center_body_m)
    target = np.array([0.0, 0.0, 0.01]) + 0.013 * FRAME.y_t
    if graze:
        vel = 40.0 * FRAME.y_t + 1.0 * FRAME.x_t - 3.0 * FRAME.z_t
    else:
        vel = v_scale * (
            38.0 * FRAME.x_t + 6.0 * FRAME.y_t - 4.0 * FRAME.z_t
        )
    return RigidClubState(
        origin_m=target - centre,
        rotation=rot,
        velocity_mps=vel,
        angular_velocity_rps=np.array([2.0, 5.0, 31.0]),
        face_spec=SPEC,
        face_unobservable_reason=reason,
    )


def test_state_is_on_the_event_and_flow_is_transversal() -> None:
    state = make_state()
    assert abs(EVENT.value(state)) < 1e-12
    assert EVENT.gradient(state) @ constant_twist_flow(state) > 10.0


def test_event_gradient_matches_central_difference() -> None:
    state = make_state()
    grad = EVENT.gradient(state)
    h = 1e-6
    for i in range(12):
        e = np.zeros(12)
        e[i] = h
        fd = (EVENT.value(state.retract(e)) - EVENT.value(state.retract(-e))) / (
            2 * h
        )
        assert grad[i] == pytest.approx(fd, abs=1e-8)


def test_analytic_jacobian_matches_central_fd_to_1e8() -> None:
    state = make_state()
    fd = impact_outcome_map(state, EVENT, FRAME, flow=constant_twist_flow)
    an = impact_outcome_map(
        state,
        EVENT,
        FRAME,
        flow=constant_twist_flow,
        kinematic_jacobian=rigid_kinematic_jacobian,
    )
    assert fd.jacobian_method == "central_fd" and an.jacobian_method == "analytic"
    assert fd.names == OUTCOME_NAMES == an.names
    assert not fd.unavailable
    err = np.max(np.abs(fd.jacobian_fixed_time - an.jacobian_fixed_time))
    assert err < 1e-8, err
    err_ev = np.max(np.abs(fd.jacobian - an.jacobian))
    assert err_ev < 1e-8, err_ev
    np.testing.assert_allclose(fd.values, an.values, atol=1e-12)


def test_pure_time_shift_gives_zero_event_change_but_fixed_time_error() -> None:
    state = make_state()
    omap = impact_outcome_map(
        state, EVENT, FRAME, flow=constant_twist_flow,
        kinematic_jacobian=rigid_kinematic_jacobian,
    )
    flow = constant_twist_flow(state)
    tau = 1e-3
    dx = flow * tau  # a pure time shift along the flow
    assert np.max(np.abs(omap.jacobian @ dx)) < 1e-9
    fixed = omap.jacobian_fixed_time @ dx
    # Independent check: d(face angle)/dt by central difference of the exact flow.
    h = 1e-6
    rates = (
        impact_outcome_map(
            advance_constant_twist(state, h), EVENT, FRAME, flow=constant_twist_flow,
            check_on_event=False,
        ).values
        - impact_outcome_map(
            advance_constant_twist(state, -h), EVENT, FRAME, flow=constant_twist_flow,
            check_on_event=False,
        ).values
    ) / (2 * h)
    np.testing.assert_allclose(fixed, rates * tau, rtol=1e-6, atol=1e-9)
    psi_dot = rates[OUTCOME_NAMES.index("face_angle_deg")]
    assert abs(psi_dot) > 500.0  # deg/s: face-closure scale
    assert fixed[OUTCOME_NAMES.index("face_angle_deg")] == pytest.approx(
        psi_dot * tau, rel=1e-6
    )
    assert abs(psi_dot * tau) > 0.5  # a 1 ms error is a fraction of a degree+


def _event_outcomes(state: RigidClubState, dx: np.ndarray) -> np.ndarray:
    """Outcomes at the true event of the perturbed state (root find on t)."""
    pert = state.retract(dx)
    s = 0.0
    for _ in range(30):
        moved = advance_constant_twist(pert, s)
        g = EVENT.value(moved)
        rate = EVENT.gradient(moved) @ constant_twist_flow(moved)
        step = g / rate
        s -= step
        if abs(step) < 1e-15:
            break
    moved = advance_constant_twist(pert, s)
    return impact_outcome_map(
        moved, EVENT, FRAME, flow=constant_twist_flow, check_on_event=False
    ).values


def test_moving_event_linearization_needs_the_saltation_term() -> None:
    state = make_state()
    omap = impact_outcome_map(
        state, EVENT, FRAME, flow=constant_twist_flow,
        kinematic_jacobian=rigid_kinematic_jacobian,
    )
    rng = np.random.default_rng(7)
    dx = 1e-4 * rng.standard_normal(12) * np.array([1] * 3 + [1] * 3 + [30] * 3 + [30] * 3)
    truth = _event_outcomes(state, dx) - omap.values
    with_term = omap.jacobian @ dx
    without_term = omap.jacobian_fixed_time @ dx  # timing term removed
    scale = np.max(np.abs(truth))
    assert np.max(np.abs(with_term - truth)) < 1e-3 * scale
    assert np.max(np.abs(without_term - truth)) > 1e-2 * scale


def test_unobserved_rotation_makes_face_rows_unavailable() -> None:
    reason = "axial club rotation unobserved (mocap-matched swing)"
    omap = impact_outcome_map(
        make_state(reason=reason), EVENT, FRAME, flow=constant_twist_flow
    )
    for name in FACE_OUTCOMES:
        assert name in omap.unavailable
        assert reason in omap.unavailable[name]
        rec = omap.records[name]
        assert rec.allocation_dependence is AllocationDependence.UNAVAILABLE
        assert reason in str(rec.reason)
        i = OUTCOME_NAMES.index(name)
        assert np.all(np.isnan(omap.jacobian[i]))
    for name in ("club_path_deg", "attack_angle_deg", "clubhead_speed_mps"):
        assert name not in omap.unavailable
        i = OUTCOME_NAMES.index(name)
        assert np.all(np.isfinite(omap.jacobian[i]))
        assert omap.records[name].allocation_dependence is (
            AllocationDependence.INVARIANT
        )
    an = impact_outcome_map(
        make_state(reason=reason), EVENT, FRAME, flow=constant_twist_flow,
        kinematic_jacobian=rigid_kinematic_jacobian,
    )
    assert set(an.unavailable) == set(omap.unavailable)


def test_off_event_state_and_bad_inputs_are_rejected() -> None:
    state = make_state()
    off = state.retract(np.array([0.05, 0, 0, 0, 0, 0] + [0.0] * 6))
    with pytest.raises(ValueError, match="event"):
        impact_outcome_map(off, EVENT, FRAME, flow=constant_twist_flow)
    with pytest.raises(TypeError):
        impact_outcome_map(state, None, FRAME, flow=constant_twist_flow)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="step"):
        impact_outcome_map(state, EVENT, FRAME, flow=constant_twist_flow, step=0.0)


def test_grazing_contact_is_flagged_unreliable() -> None:
    omap = impact_outcome_map(
        make_state(graze=True), EVENT, FRAME, flow=constant_twist_flow
    )
    assert omap.reliability == "UNRELIABLE_LINEAR_ESTIMATE"
    assert "Monte Carlo" in omap.reliability_note
    ok = impact_outcome_map(make_state(), EVENT, FRAME, flow=constant_twist_flow)
    assert ok.reliability == "LINEAR_OK"


def test_section_and_flow_are_labelled() -> None:
    omap = impact_outcome_map(
        make_state(), EVENT, FRAME, flow=constant_twist_flow
    )
    assert omap.section == "event"
    assert "diagnostic" in omap.fixed_time_label.lower()


# ---- Tools ball-flight bridge (stub gateway; Tools is never imported) ----


def _stub_gateway() -> ToolsDeliveryGateway:
    def derive(params):
        return SimpleNamespace(
            spin_loft_deg=abs(params.dynamic_loft_deg - params.attack_angle_deg)
        )

    def dplane(v_app, n_unit):
        return SimpleNamespace(
            club_path_deg=0.0,
            attack_angle_deg=0.0,
            face_angle_deg=0.0,
            dynamic_loft_deg=1.0,
            face_to_path_deg=math.degrees(math.atan2(n_unit[2], n_unit[0]))
            - math.degrees(math.atan2(v_app[2], v_app[0])),
            planar_spin_loft_deg=None,
            dplane_tilt_deg=float(np.linalg.norm(v_app)),
            status="ok",
        )

    module = SimpleNamespace(
        derive_delivery=derive,
        DeliveryParameters=lambda **kw: SimpleNamespace(**kw),
        analyze_dplane=dplane,
    )
    return ToolsDeliveryGateway(module)


def test_tools_ball_flight_uses_black_box_fd_and_event_projection() -> None:
    state = make_state()
    omap = impact_outcome_map(state, EVENT, FRAME, flow=constant_twist_flow)
    bf = tools_ball_flight_map(
        state, omap, FRAME, gateway_loader=_stub_gateway, flow=constant_twist_flow
    )
    assert "spin_loft_3d_deg" in bf.names
    assert bf.unavailable.keys() == {"planar_spin_loft_deg"}
    i = bf.names.index("spin_loft_3d_deg")
    # event-section Jacobian annihilates a pure time shift
    dx = constant_twist_flow(state) * 1e-3
    assert abs(bf.jacobian[i] @ dx) < 1e-7
    assert abs(bf.jacobian_fixed_time[i] @ dx) > 1e-4
    assert "model estimate" in bf.label


def test_tools_unavailable_is_reported_not_raised() -> None:
    def missing() -> ToolsDeliveryGateway:
        raise ToolsDeliveryUnavailableError("Tools not installed")

    state = make_state()
    omap = impact_outcome_map(state, EVENT, FRAME, flow=constant_twist_flow)
    bf = tools_ball_flight_map(
        state, omap, FRAME, gateway_loader=missing, flow=constant_twist_flow
    )
    assert bf.available is False
    assert "Tools not installed" in bf.unavailable_reason
    for rec in bf.records.values():
        assert rec.allocation_dependence is AllocationDependence.UNAVAILABLE


def test_tools_ball_flight_is_unavailable_when_face_unobserved() -> None:
    state = make_state(reason="axial rotation unobserved")
    omap = impact_outcome_map(state, EVENT, FRAME, flow=constant_twist_flow)
    bf = tools_ball_flight_map(
        state, omap, FRAME, gateway_loader=_stub_gateway, flow=constant_twist_flow
    )
    assert bf.available is False
    assert "axial rotation unobserved" in bf.unavailable_reason
