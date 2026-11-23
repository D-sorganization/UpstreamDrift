"""GCV-20 (#11767): the shared ball-impact force hook used by every replay.

Toy plant: a carriage at ``x`` (mass ``M``) carrying an arm of length ``L``
at angle ``theta`` (inertia ``I`` about the pivot). The face frame sits at
the arm tip, its normal along the arm's tangent. Everything has a closed
form: the point Jacobian, the effective mass along the normal and the
post-collision velocity.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.impact_parameters import ball_impact as bi
from src.shared.python.model_appearance.ball import BALL_RADIUS_M as GOLF_BALL_RADIUS_M
from src.shared.python.motion_matching.contact_event import ContactRoot
from src.shared.python.motion_matching import impact_force as imf

pytestmark = pytest.mark.unit

M, INERTIA, L = 2.0, 0.05, 1.0


def poses(q: np.ndarray) -> dict[str, np.ndarray]:
    x, th = q
    c, s = math.cos(th), math.sin(th)
    pose = np.eye(4)
    pose[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    pose[:3, 3] = [x + L * c, L * s, 0.0]
    return {"Clubhead": pose}


def accel(q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return np.asarray(tau, dtype=float) / np.array([M, INERTIA])


def plan(t_start: float = 0.5, duration: float = 4.5e-4) -> imf.ImpactForce:
    return imf.ImpactForce(
        frame="Clubhead",
        point_in_frame=np.zeros(3),
        axis_in_frame=np.array([0.0, 1.0, 0.0]),  # tangent to the arm
        t_start_s=t_start,
        duration_s=duration,
        swing_span_s=(0.0, 1.0),
    )


def test_generalized_force_is_the_jacobian_transpose() -> None:
    q = np.array([0.3, 0.4])
    force = np.array([1.0, -2.0, 0.5])
    impact = plan().with_force(force)
    jac = np.array([[1.0, -L * math.sin(0.4)], [0.0, L * math.cos(0.4)], [0, 0]])
    np.testing.assert_allclose(
        impact.generalized_force(q, poses), jac.T @ force, atol=1e-6
    )


def test_unlatched_plan_has_no_force() -> None:
    impact = plan()
    assert impact.contact_search_status is None
    with pytest.raises(ValueError, match="latched"):
        impact.generalized_force(np.zeros(2), poses)


def test_latch_uses_the_plant_effective_mass() -> None:
    th = 0.0
    q, v = np.array([0.0, th]), np.array([0.0, 40.0])  # tip moves +y at 40 m/s
    impact = plan().latch(q, v, poses, accel)
    hit = impact.collision
    assert hit is not None
    # Normal +y at theta = 0: J_n = [0, L]; m_eff = I / L^2.
    assert hit.effective_mass_kg == pytest.approx(INERTIA / L**2, rel=1e-6)
    assert hit.approach_speed_mps == pytest.approx(40.0, rel=1e-6)
    np.testing.assert_allclose(
        impact.force_world * impact.duration_s, hit.impulse_on_club_n_s
    )


def test_integrating_through_the_window_gives_the_analytic_rebound() -> None:
    q, v = np.array([0.0, 0.0]), np.array([0.0, 40.0])
    dt, t = 1e-3, 0.4995  # the window starts mid-step
    impact = plan()

    def rk4(t_a, h, qq, vv, external):  # noqa: ANN001, ANN202
        def a(q_, v_):  # noqa: ANN001, ANN202
            ext = np.zeros(2) if external is None else external(q_)
            return accel(q_, v_, ext)

        k1q, k1v = vv, a(qq, vv)
        k2q, k2v = vv + h / 2 * k1v, a(qq + h / 2 * k1q, vv + h / 2 * k1v)
        k3q, k3v = vv + h / 2 * k2v, a(qq + h / 2 * k2q, vv + h / 2 * k2v)
        k4q, k4v = vv + h * k3v, a(qq + h * k3q, vv + h * k3v)
        return (
            qq + h / 6 * (k1q + 2 * k2q + 2 * k3q + k4q),
            vv + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v),
            None,
        )

    q1, v1, latched, _ = imf.step_through_impact(
        impact, t, dt, q, v, rk4, poses=poses, accel=accel
    )
    assert latched.is_latched
    m_eff, mb, e = INERTIA / L**2, bi.BALL_MASS_KG, bi.COR_LIMIT
    expected_dv = (1 + e) * mb * 40.0 / (m_eff + mb)
    tip_speed = L * v1[1]  # the arm turns ~0.02 rad in 0.45 ms: ~2e-4 error
    assert 40.0 - tip_speed == pytest.approx(expected_dv, rel=2e-3)


def test_steps_outside_the_window_are_untouched() -> None:
    calls = []

    def step(t_a, h, qq, vv, external):  # noqa: ANN001, ANN202
        calls.append((t_a, h, external))
        return qq, vv, "tau"

    impact = plan()
    _, _, out, first = imf.step_through_impact(
        impact, 0.1, 1e-3, np.zeros(2), np.ones(2), step, poses=poses, accel=accel
    )
    assert calls == [(0.1, 1e-3, None)] and first == "tau"
    assert not out.is_latched
    _, _, _, _ = imf.step_through_impact(
        None, 0.1, 1e-3, np.zeros(2), np.ones(2), step, poses=poses, accel=accel
    )
    assert len(calls) == 2


def test_sub_intervals_split_exactly_at_the_window_edges() -> None:
    impact = plan(t_start=0.5002, duration=4.5e-4)
    parts = impact.sub_intervals(0.5, 1e-3)
    edges = [p[0] for p in parts] + [parts[-1][1]]
    np.testing.assert_allclose(edges, [0.5, 0.5002, 0.50065, 0.501])
    assert [p[2] for p in parts] == [False, True, False]
    assert impact.sub_intervals(0.2, 1e-3) == [(0.2, 0.201, False)]
    inside = plan(t_start=0.5, duration=4e-3).sub_intervals(0.501, 1e-3)
    assert inside == [(0.501, 0.502, True)]


def test_record_round_trip_keeps_the_latched_force() -> None:
    impact = plan().latch(np.zeros(2), np.array([0.0, 40.0]), poses, accel)
    again = imf.ImpactForce.from_record(json.loads(json.dumps(impact.to_record())))
    np.testing.assert_allclose(again.force_world, impact.force_world)
    assert again.is_latched and again.t_start_s == impact.t_start_s
    shifted = again.shifted(-0.25)
    assert shifted.t_start_s == pytest.approx(0.25)
    assert shifted.swing_span_s == pytest.approx((-0.25, 0.75))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"duration_s": 0.0}, "duration_s"),
        ({"t_start_s": 1.5}, "inside the swing"),
        ({"axis_in_frame": np.zeros(3)}, "axis_in_frame"),
        ({"point_in_frame": np.array([np.nan, 0, 0])}, "point_in_frame"),
        ({"frame": ""}, "frame"),
        ({"cor": 0.0}, "cor"),
        ({"ball_mass_kg": 0.0}, "ball_mass_kg"),
    ],
)
def test_plan_contracts(change: dict, message: str) -> None:
    fields = {
        "frame": "Clubhead",
        "point_in_frame": np.zeros(3),
        "axis_in_frame": np.array([0.0, 1.0, 0.0]),
        "t_start_s": 0.5,
        "duration_s": 4.5e-4,
        "swing_span_s": (0.0, 1.0),
    }
    fields.update(change)
    with pytest.raises(ValueError, match=message):
        imf.ImpactForce(**fields)


def test_window_must_fit_in_the_swing() -> None:
    with pytest.raises(ValueError, match="inside the swing"):
        plan(t_start=0.9999, duration=1e-3)


def test_spec_plan_does_not_load_finite_geometry_unless_opted_in(monkeypatch) -> None:
    from src.shared.python.motion_matching import club_face_target as cft

    monkeypatch.setattr(cft, "face_centre_in_frame", lambda _spec: np.zeros(3))
    monkeypatch.setattr(
        cft, "face_axis_in_frame", lambda _spec: np.array([1.0, 0.0, 0.0])
    )

    def unexpected(_spec):
        raise AssertionError("legacy plan must not load face triangles")

    monkeypatch.setattr(cft, "face_triangles_in_frame", unexpected)
    impact = imf.ImpactForce.for_spec(
        {}, t_start_s=0.5, swing_span_s=(0.0, 1.0), ball_centre_m=np.ones(3)
    )
    assert impact.face_triangles_in_frame is None


def test_reference_rates_do_not_differentiate_across_impact() -> None:
    t = np.linspace(0.0, 0.1, 37)
    t_hit = 0.0505
    q = np.where(t <= t_hit, 40.0 * t, 40.0 * t_hit + 25.0 * (t - t_hit))[:, None]
    vel, acc, last = imf.reference_rates(t, q, t_hit)
    assert t[last] <= t_hit < t[last + 1]
    split = (t_hit, last)
    np.testing.assert_allclose(
        imf.sample_rate_table(t, vel, t_hit, split), [40.0], rtol=1e-9
    )
    np.testing.assert_allclose(
        imf.sample_rate_table(t, vel, t_hit + 1e-4, split), [25.0], rtol=1e-9
    )
    assert np.abs(acc).max() < 1e-6  # piecewise linear: no spike
    joined_vel, joined_acc, none = imf.reference_rates(t, q, None)
    assert none is None and np.abs(joined_acc).max() > 1e3
    np.testing.assert_array_equal(
        imf.sample_rate_table(t, joined_vel, 0.02, None),
        [np.interp(0.02, t, joined_vel[:, 0])],
    )
    with pytest.raises(ValueError, match="two samples"):
        imf.reference_rates(t, q, t[-2])


def test_passage_trigger_fires_when_the_face_reaches_the_ball() -> None:
    """Contact starts at the simulated face centre's closest approach to the
    ball point, not at the capture's impact time (the replay lags it)."""
    ball = np.array([L, 0.0, 0.0])  # tip at theta = 0
    armed = plan(t_start=0.40).with_passage(ball)
    assert not armed.is_scheduled and armed.trigger == "passage"
    th0, rate = -0.4, 40.0  # tip reaches the ball at t = 0.41 + 0.01
    q, v = np.array([0.0, th0]), np.array([0.0, rate])
    t, dt = 0.41, 1e-3
    calls = []

    def drift(t_a, h, qq, vv, external):  # noqa: ANN001, ANN202
        calls.append((t_a, h, external is not None))
        return qq + h * vv, vv, None

    impact = armed
    for _ in range(20):
        q, v, impact, _ = imf.step_through_impact(
            impact, t, dt, q, v, drift, poses=poses, accel=accel
        )
        t += dt
    assert impact.is_scheduled and impact.is_latched
    assert impact.t_start_s == pytest.approx(0.41 + 0.4 / rate, abs=2e-4)
    forced = [c for c in calls if c[2]]
    assert sum(h for _, h, _ in forced) == pytest.approx(impact.duration_s)
    record = imf.ImpactForce.from_record(impact.to_record())
    assert record.is_scheduled and record.trigger == "passage"


def test_passage_trigger_needs_the_head_near_the_ball() -> None:
    far = plan(t_start=0.40).with_passage(np.array([5.0, 0.0, 0.0]))
    q, v = np.array([0.0, 0.0]), np.array([0.0, 40.0])
    out = far.schedule(0.41, 1e-3, q, v, poses)
    assert not out.is_scheduled
    early = plan(t_start=0.40).with_passage(np.array([L, 0.0, 0.0]))
    assert not early.schedule(0.30, 1e-3, q, v, poses).is_scheduled  # not armed


def _surface_plan(ball: np.ndarray, *, t_start: float = 0.0) -> imf.ImpactForce:
    triangles = np.array(
        [
            [[0.0, -0.2, -0.2], [0.0, 0.2, -0.2], [0.0, 0.0, 0.2]],
            [[0.0, -0.2, -0.2], [0.0, 0.0, 0.2], [0.0, -0.2, 0.2]],
        ]
    )
    return replace(
        plan(t_start=t_start),
        axis_in_frame=np.array([1.0, 0.0, 0.0]),
        ball_centre_m=ball,
        face_triangles_in_frame=triangles,
    ).with_surface_contact()


def rotating_poses(q: np.ndarray) -> dict[str, np.ndarray]:
    angle = float(q[1])
    c, s = math.cos(angle), math.sin(angle)
    pose = np.eye(4)
    pose[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    pose[:3, 3] = [q[0], 0.0, 0.0]
    return {"Clubhead": pose}


def _ball_on_rotated_face() -> np.ndarray:
    angle = 0.5
    return np.array(
        [
            1.0 + GOLF_BALL_RADIUS_M * math.cos(angle),
            GOLF_BALL_RADIUS_M * math.sin(angle),
            0.0,
        ]
    )


def test_surface_trigger_uses_integrated_root_and_moving_face_pose() -> None:
    q = np.zeros(2)
    v = np.array([100.0, 50.0])
    ball = _ball_on_rotated_face()
    impact = _surface_plan(ball)
    active = []
    latched_states = []

    def drift(t_a, h, qq, vv, external):  # noqa: ANN001, ANN202
        active.append(external is not None)
        acceleration = np.zeros(2) if external is None else external(qq) / [M, INERTIA]
        return qq + h * vv + 0.5 * h * h * acceleration, vv + h * acceleration, None

    def record_accel(qq, vv, tau):  # noqa: ANN001, ANN202
        latched_states.append((qq.copy(), vv.copy()))
        return accel(qq, vv, tau)

    q1, v1, triggered, _ = imf.step_through_impact(
        impact, 0.0, 0.02, q, v, drift, poses=rotating_poses, accel=record_accel
    )
    assert triggered is not None and triggered.trigger == "surface"
    assert triggered.is_scheduled and triggered.is_latched
    assert triggered.contact_event is not None
    assert triggered.contact_search_status == "sampled_bracket_candidate"
    assert not triggered.contact_search_has_prior_unresolved_intervals
    event = triggered.contact_event
    assert event.time_s == pytest.approx(0.01, abs=5e-10)
    np.testing.assert_allclose(event.q, [1.0, 0.5], atol=1e-8)
    np.testing.assert_allclose(event.qd, v)
    np.testing.assert_allclose(event.contact_point_m, [1.0, 0.0, 0.0], atol=2e-8)
    np.testing.assert_allclose(
        event.contact_normal,
        [math.cos(0.5), math.sin(0.5), 0.0],
        atol=1e-14,
    )
    assert event.approach_speed_mps == pytest.approx(100.0 * math.cos(0.5), rel=1e-6)
    np.testing.assert_allclose(latched_states[0][0], event.q)
    np.testing.assert_allclose(latched_states[0][1], event.qd)
    assert any(active)
    assert not np.array_equal(q1, q) and not np.array_equal(v1, v)


def test_surface_hit_commits_only_real_zoh_stages_and_event() -> None:
    from src.shared.python.motion_matching.same_input.integrator import (
        ROOT_COORDINATES,
        StepPolicy,
        integrate,
    )
    from src.shared.python.motion_matching.same_input.stage_trace import StageRecorder

    class Plant:
        coordinate_order = (*ROOT_COORDINATES, "x", "angle")

        def acceleration(self, q, v, tau):
            return np.asarray(tau) / np.array([1] * 6 + [M, INERTIA])

        def kinematic_frames(self, q):
            return rotating_poses(q[6:])

        def closure_pose_residual(self, q):
            return np.zeros(1)

    q, v = np.zeros(8), np.zeros(8)
    v[6:] = [100, 50]
    recorder = StageRecorder()
    run = integrate(
        Plant(),
        q,
        v,
        lambda *_: np.zeros(8),
        steps=1,
        dt_s=0.02,
        policy=StepPolicy(substeps=1, project=False),
        impact=_surface_plan(_ball_on_rotated_face()),
        recorder=recorder,
    )
    assert run.ball_impact is not None
    event = run.ball_impact["contact_event"]
    assert event["time_s"] == pytest.approx(0.01, abs=5e-10)
    stages = recorder.steps[0].stages
    assert len(stages) == 12
    assert {s.segment for s in stages} == {0, 1, 2}
    assert [segment.force_active for segment in recorder.steps[0].segments] == [
        False,
        True,
        False,
    ]
    expected = recorder.expected_stages()
    actual = [(s.step, s.hold, s.segment, s.substep, s.stage, s.time_s) for s in stages]
    assert expected == actual
    assert recorder.outcome.status == "complete"
    assert all(np.array_equal(s.tau, np.zeros(8)) for s in stages)
    np.testing.assert_allclose(stages[4].q, event["q"], atol=1e-10)
    np.testing.assert_allclose(stages[4].v, event["qd"], atol=1e-10)
    changed = list(stages)
    changed[4] = replace(changed[4], tau=np.ones(8))
    recorder.steps[0] = replace(recorder.steps[0], stages=tuple(changed))
    recorder.finish()
    assert recorder.outcome.status == "partial"


def test_surface_trigger_fails_closed_and_records_a_finite_miss() -> None:
    q, v = np.zeros(2), np.array([100.0, 0.0])
    impact = _surface_plan(np.array([1.0, 2.0, 0.0]))
    calls = []

    def drift(t_a, h, qq, vv, external):  # noqa: ANN001, ANN202
        calls.append(external)
        return qq + h * vv, vv, None

    q1, v1, result, _ = imf.step_through_impact(
        impact, 0.0, 0.02, q, v, drift, poses=rotating_poses, accel=accel
    )
    assert result is not None and not result.is_scheduled and not result.is_latched
    assert result.contact_event is None
    assert np.isfinite(result.contact_miss_penalty_m)
    assert result.contact_search_status == "unresolved_no_accepted_candidate"
    restored = imf.ImpactForce.from_record(result.to_record())
    assert restored.contact_search_status == "unresolved_no_accepted_candidate"
    assert restored.contact_search_has_prior_unresolved_intervals
    assert all(external is None for external in calls)
    np.testing.assert_allclose(q1, q + 0.02 * v)
    np.testing.assert_array_equal(v1, v)


def test_later_sampled_bracket_preserves_prior_unresolved_interval() -> None:
    impact = _surface_plan(np.array([0.1, 0.0, 0.0]))
    q, v = np.zeros(2), np.array([10.0, 0.0])

    def drift(_t, h, qq, vv, _external):  # noqa: ANN001, ANN202
        return qq + h * vv, vv, None

    q, v, missed, _ = imf.step_through_impact(
        impact, 0.0, 0.005, q, v, drift, poses=rotating_poses, accel=accel
    )
    assert missed is not None and not missed.is_scheduled
    assert missed.contact_search_status == "unresolved_no_accepted_candidate"
    assert missed.contact_search_has_prior_unresolved_intervals

    _, _, hit, _ = imf.step_through_impact(
        missed, 0.005, 0.01, q, v, drift, poses=rotating_poses, accel=accel
    )
    assert hit is not None and hit.is_scheduled
    assert hit.contact_event is not None
    assert hit.contact_search_status == "sampled_bracket_candidate"
    assert hit.contact_search_has_prior_unresolved_intervals
    restored = imf.ImpactForce.from_record(json.loads(json.dumps(hit.to_record())))
    assert restored.contact_search_status == "sampled_bracket_candidate"
    assert restored.contact_search_has_prior_unresolved_intervals


def test_surface_search_separates_trial_and_accepted_advances() -> None:
    q, v = np.zeros(2), np.array([100.0, 0.0])
    impact = _surface_plan(np.array([1.0, 2.0, 0.0]))
    trials, accepted = [], []

    def trial(t, h, qq, vv, external):
        trials.append((t, h))
        return qq + h * vv, vv, None

    def step(t, h, qq, vv, external):
        accepted.append((t, h))
        return qq + h * vv, vv, None

    q1, _, result, _ = imf.step_through_impact(
        impact,
        0.0,
        0.02,
        q,
        v,
        step,
        poses=rotating_poses,
        accel=accel,
        trial_step=trial,
    )
    assert trials
    assert accepted == [(0.0, 0.02)]
    assert not result.is_scheduled
    np.testing.assert_allclose(q1, q + 0.02 * v)


def test_surface_latch_uses_the_off_centre_root_for_impulse_and_force_jacobian() -> (
    None
):
    angle, local_y = 0.5, 0.05
    normal = np.array([math.cos(angle), math.sin(angle), 0.0])
    face_point = np.array(
        [1.0 - local_y * math.sin(angle), local_y * math.cos(angle), 0.0]
    )
    ball = face_point + GOLF_BALL_RADIUS_M * normal
    impact = _surface_plan(ball)
    q, v = np.zeros(2), np.array([100.0, 50.0])

    def drift(_t, h, qq, vv, external):  # noqa: ANN001, ANN202
        return qq + h * vv, vv, None

    _, _, latched, _ = imf.step_through_impact(
        impact, 0.0, 0.02, q, v, drift, poses=rotating_poses, accel=accel
    )
    assert latched.collision is not None and latched.contact_event is not None
    assert latched.contact_point_in_frame is not None
    np.testing.assert_allclose(
        latched.contact_point_in_frame, [0.0, local_y, 0.0], atol=2e-8
    )
    expected_jac = latched.point_jacobian(
        latched.contact_event.q, rotating_poses, latched.contact_point_in_frame
    )
    expected_hit_velocity = expected_jac @ latched.contact_event.qd
    assert latched.collision.approach_speed_mps == pytest.approx(
        float(latched.contact_event.contact_normal @ expected_hit_velocity), rel=1e-6
    )
    np.testing.assert_allclose(
        latched.generalized_force(latched.contact_event.q, rotating_poses),
        expected_jac.T @ latched.force_world,
    )


def test_unscheduled_surface_plan_cannot_accept_a_replay_force() -> None:
    impact = _surface_plan(np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="surface.*force"):
        impact.with_force(np.array([1.0, 0.0, 0.0]))


def test_surface_search_requires_an_integrator_advance() -> None:
    impact = _surface_plan(np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="advance"):
        impact.schedule(0.0, 0.01, np.zeros(2), np.zeros(2), rotating_poses)


def test_surface_arming_cannot_be_after_its_scheduled_root() -> None:
    plan = _surface_plan(np.array([1.0, 0.0, 0.0]), t_start=0.1)
    with pytest.raises(ValueError, match="armed_at_s"):
        replace(plan, armed_at_s=0.2)


def test_cross_step_miss_penalty_uses_smooth_weighted_aggregation() -> None:
    q, v = np.zeros(2), np.array([100.0, 0.0])
    impact = _surface_plan(np.array([1.0, 2.0, 0.0]))

    def drift(_t, h, qq, vv, _external):  # noqa: ANN001, ANN202
        return qq + h * vv, vv, None

    _, _, first, _ = imf.step_through_impact(
        impact, 0.0, 0.01, q, v, drift, poses=rotating_poses, accel=accel
    )
    _, _, second, _ = imf.step_through_impact(
        first,
        0.01,
        0.01,
        np.array([1.0, 0.0]),
        v,
        drift,
        poses=rotating_poses,
        accel=accel,
    )
    assert second.contact_miss_penalty_weight_s == pytest.approx(0.02)
    assert second.contact_miss_penalty_logsum is not None
    assert np.isfinite(second.contact_miss_penalty_m)
    restored = imf.ImpactForce.from_record(json.loads(json.dumps(second.to_record())))
    assert restored.contact_miss_penalty_m == pytest.approx(
        second.contact_miss_penalty_m
    )


def test_surface_trigger_honors_arm_time_inside_the_step() -> None:
    impact = _surface_plan(_ball_on_rotated_face(), t_start=0.005)
    q, v = np.zeros(2), np.array([100.0, 50.0])
    calls = []

    def drift(t_a, h, qq, vv, external):  # noqa: ANN001, ANN202
        calls.append((t_a, h, external is not None))
        return qq + h * vv, vv, None

    _, _, result, _ = imf.step_through_impact(
        impact, 0.0, 0.02, q, v, drift, poses=rotating_poses, accel=accel
    )
    assert result is not None and result.is_latched
    assert result.contact_event is not None
    assert result.contact_event.time_s >= impact.armed_at_s
    assert any(t0 == pytest.approx(0.005) and not active for t0, _, active in calls)


def test_surface_sub_intervals_are_chronological_when_event_follows_arming() -> None:
    impact = replace(
        _surface_plan(np.array([1.0, 0.0, 0.0]), t_start=0.005),
        scheduled=True,
        t_start_s=0.007,
        contact_search_status="sampled_bracket_candidate",
        contact_event=ContactRoot(
            time_s=0.007,
            q=np.zeros(2),
            qd=np.array([100.0, 0.0]),
            contact_point_m=np.zeros(3),
            contact_normal=np.array([1.0, 0.0, 0.0]),
            approach_speed_mps=100.0,
            triangle_index=0,
        ),
    )
    intervals = impact.sub_intervals(0.0, 0.02)
    assert intervals == [
        (0.0, 0.005, False),
        (0.005, 0.007, False),
        (0.007, 0.007 + impact.duration_s, True),
        (0.007 + impact.duration_s, 0.02, False),
    ]
    assert all(a < b for a, b, _ in intervals)


def test_shifted_unscheduled_surface_plan_moves_arming_clock() -> None:
    plan = _surface_plan(np.array([1.0, 0.0, 0.0]), t_start=0.1)
    shifted = plan.shifted(0.3)
    assert shifted.t_start_s == pytest.approx(0.4)
    assert shifted.armed_at_s == pytest.approx(0.4)
    assert shifted.swing_span_s == pytest.approx((0.3, 1.3))


def test_shifted_scheduled_surface_plan_moves_root_and_arming_clocks() -> None:
    plan = _surface_plan(np.array([1.0, 0.0, 0.0]), t_start=0.1)
    root = ContactRoot(
        time_s=0.1,
        q=np.zeros(2),
        qd=np.array([100.0, 0.0]),
        contact_point_m=np.zeros(3),
        contact_normal=np.array([1.0, 0.0, 0.0]),
        approach_speed_mps=100.0,
        triangle_index=0,
    )
    scheduled = replace(
        plan,
        scheduled=True,
        armed_at_s=0.05,
        contact_event=root,
        contact_search_status="sampled_bracket_candidate",
    )
    shifted = scheduled.shifted(0.3)
    assert shifted.t_start_s == pytest.approx(0.4)
    assert shifted.armed_at_s == pytest.approx(0.35)
    assert shifted.contact_event is not None
    assert shifted.contact_event.time_s == pytest.approx(0.4)
    restored = imf.ImpactForce.from_record(shifted.to_record())
    assert restored.contact_event is not None
    assert restored.contact_event.time_s == pytest.approx(restored.t_start_s)


def test_shifted_latched_surface_record_moves_all_event_clocks() -> None:
    plan = _surface_plan(np.array([1.0, 0.0, 0.0]), t_start=0.1)
    root = ContactRoot(
        time_s=0.1,
        q=np.zeros(2),
        qd=np.array([100.0, 0.0]),
        contact_point_m=np.zeros(3),
        contact_normal=np.array([1.0, 0.0, 0.0]),
        approach_speed_mps=100.0,
        triangle_index=0,
    )
    latched = replace(
        plan,
        scheduled=True,
        armed_at_s=0.05,
        contact_event=root,
        contact_point_in_frame=np.zeros(3),
        contact_search_status="sampled_bracket_candidate",
    ).with_force(np.array([1.0, 2.0, 3.0]))
    restored = imf.ImpactForce.from_record(json.loads(json.dumps(latched.to_record())))
    shifted = restored.shifted(-0.02)
    assert shifted.is_latched
    assert shifted.t_start_s == pytest.approx(0.08)
    assert shifted.armed_at_s == pytest.approx(0.03)
    assert shifted.contact_event is not None
    assert shifted.contact_event.time_s == pytest.approx(0.08)
    assert imf.ImpactForce.from_record(
        shifted.to_record()
    ).contact_event.time_s == pytest.approx(0.08)


def test_surface_geometry_and_root_round_trip_and_latched_replay_do_not_reschedule() -> (
    None
):
    impact = _surface_plan(np.array([1.0, 0.0, 0.0]))
    state = np.zeros(2), np.array([100.0, 50.0])

    def drift(_t, h, qq, vv, external):  # noqa: ANN001, ANN202
        return qq + h * vv, vv, None

    _, _, triggered, _ = imf.step_through_impact(
        impact, 0.0, 0.02, *state, drift, poses=rotating_poses, accel=accel
    )
    assert triggered is not None and triggered.is_latched
    assert triggered.contact_search_status == "sampled_bracket_candidate"
    assert not triggered.contact_search_has_prior_unresolved_intervals
    record = json.loads(json.dumps(triggered.to_record()))
    restored = imf.ImpactForce.from_record(record)
    assert restored.contact_event is not None
    assert restored.contact_search_status == "sampled_bracket_candidate"
    assert not restored.contact_search_has_prior_unresolved_intervals
    np.testing.assert_allclose(
        restored.face_triangles_in_frame, impact.face_triangles_in_frame
    )
    assert restored.contact_event.time_s == pytest.approx(
        triggered.contact_event.time_s
    )
    replay_calls = []

    def replay_step(t_a, h, qq, vv, external):  # noqa: ANN001, ANN202
        replay_calls.append(external is not None)
        return drift(t_a, h, qq, vv, external)

    q2, v2, replayed, _ = imf.step_through_impact(
        restored,
        0.0,
        0.02,
        np.array([-10.0, 0.0]),
        np.zeros(2),
        replay_step,
        poses=rotating_poses,
        accel=accel,
    )
    assert replayed is restored
    assert replayed.contact_event.time_s == pytest.approx(
        triggered.contact_event.time_s
    )
    assert any(replay_calls)
    assert np.isfinite(q2).all() and np.isfinite(v2).all()


def test_surface_record_fails_closed_without_geometry_or_with_invalid_event() -> None:
    record = _surface_plan(np.array([1.0, 0.0, 0.0])).to_record()
    missing_geometry = dict(record)
    missing_geometry.pop("face_triangles_in_frame")
    with pytest.raises(ValueError, match="surface contact needs"):
        imf.ImpactForce.from_record(missing_geometry)

    bad_event = dict(record)
    bad_event["scheduled"] = True
    bad_event["contact_search_status"] = "sampled_bracket_candidate"
    bad_event["contact_event"] = {
        "time_s": 0.5,
        "q": [0.5],
        "qd": [1.0],
        "contact_point_m": [0.5, 0.0, 0.0],
        "contact_normal": [0.0, 0.0, 0.0],
        "approach_speed_mps": 1.0,
        "triangle_index": 0,
    }
    with pytest.raises(ValueError, match="contact_event fields"):
        imf.ImpactForce.from_record(bad_event)


def test_rearming_surface_trigger_clears_event_and_latched_force() -> None:
    impact = _surface_plan(np.array([1.0, 0.0, 0.0]))
    q, v = np.zeros(2), np.array([100.0, 50.0])

    def drift(_t, h, qq, vv, external):  # noqa: ANN001, ANN202
        return qq + h * vv, vv, None

    _, _, triggered, _ = imf.step_through_impact(
        impact, 0.0, 0.02, q, v, drift, poses=rotating_poses, accel=accel
    )
    assert triggered is not None and triggered.is_latched
    rearmed = triggered.rearmed(lead_s=0.002)
    assert not rearmed.is_scheduled and not rearmed.is_latched
    assert rearmed.contact_event is None
    assert rearmed.contact_search_status == "unsearched"
    assert rearmed.armed_at_s == pytest.approx(triggered.t_start_s - 0.002)
