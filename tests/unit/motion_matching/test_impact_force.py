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

import numpy as np
import pytest

from src.shared.python.impact_parameters import ball_impact as bi
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
    with pytest.raises(ValueError, match="latched"):
        plan().generalized_force(np.zeros(2), poses)


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
