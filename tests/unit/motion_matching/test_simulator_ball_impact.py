"""GCV-20 (#11767): the pipeline simulator applies the ball impulse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching import full_body_forward_dynamics as fs
from src.shared.python.motion_matching.impact_force import ImpactForce

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
DT = 1e-3


@pytest.fixture(scope="module")
def setup() -> tuple[fs.FullBodySimulator, ImpactForce, np.ndarray, np.ndarray]:
    spec_bytes = SPEC.read_bytes()
    sim = fs.FullBodySimulator(NativeMujocoFullBodyModel(spec_bytes))
    impact = ImpactForce.for_spec(
        json.loads(spec_bytes), t_start_s=1.2e-3, swing_span_s=(0.0, 1.0)
    )
    q = np.zeros(sim.nv)
    q[2] = 2.0  # airborne: no ground contact
    jac = impact.point_jacobian(q, sim.frame_poses)
    _, normal = impact._point_and_normal(q, sim.frame_poses)
    direction = jac.T @ normal
    v = sim.consistent_velocity(q, 30.0 * direction / (direction @ direction))
    return sim, impact, q, v


def _face_normal_speed(sim, impact, q, v) -> float:  # noqa: ANN001
    _, normal = impact._point_and_normal(q, sim.frame_poses)
    return float(normal @ (impact.point_jacobian(q, sim.frame_poses) @ v))


def test_run_delivers_the_collision_speed_change(setup) -> None:  # noqa: ANN001
    sim, impact, q0, v0 = setup
    assert _face_normal_speed(sim, impact, q0, v0) > 5.0

    def idle(t, q, v):  # noqa: ANN001, ANN202
        return np.zeros(sim.nv)

    free = sim.run(q0, v0, idle, duration_s=3 * DT, dt_s=DT)
    hit = sim.run(q0, v0, idle, duration_s=3 * DT, dt_s=DT, impact=impact)
    assert free.ball_impact is None and hit.ball_impact is not None
    collision = hit.ball_impact["collision"]
    drop = _face_normal_speed(sim, impact, free.q[-1], free.v[-1]) - (
        _face_normal_speed(sim, impact, hit.q[-1], hit.v[-1])
    )
    assert drop == pytest.approx(collision["club_normal_speed_change_mps"], rel=0.05)
    # Before the window the runs are bit-identical.
    np.testing.assert_array_equal(free.q[1], hit.q[1])
    np.testing.assert_allclose(
        np.asarray(hit.ball_impact["force_world_n"]) * impact.duration_s,
        collision["impulse_on_club_n_s"],
    )


def test_split_tracking_controller_has_no_feedforward_spike(setup) -> None:  # noqa: ANN001
    sim, _, q0, _ = setup
    t = np.linspace(0.0, 0.1, 37)
    t_hit = 0.0505
    ramp = np.where(t <= t_hit, 40.0 * t, 40.0 * t_hit + 25.0 * (t - t_hit))
    reference = np.repeat(q0[None, :], t.size, axis=0)
    reference[:, 10] += 0.01 * ramp
    kwargs = {"omega_rad_s": 30.0, "acceleration_feedforward": 1.0}
    joined = fs.tracking_controller(sim, t, reference, **kwargs)
    split = fs.tracking_controller(sim, t, reference, split_time_s=t_hit, **kwargs)
    k = int(np.searchsorted(t, t_hit))
    state_q = reference[k]
    state_v = np.zeros(sim.nv)
    spike = np.abs(joined(t[k], state_q, state_v) - split(t[k], state_q, state_v))
    assert spike.max() > 0.0
    far = t[3]
    np.testing.assert_allclose(
        joined(far, reference[3], state_v), split(far, reference[3], state_v)
    )
