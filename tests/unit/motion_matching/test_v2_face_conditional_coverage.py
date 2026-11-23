"""Synthetic driver face contact, not a matched-swing or physical validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import least_squares

from src.shared.python.motion_matching.club_face_target import face_triangles_in_frame
from src.shared.python.motion_matching.contact_event import find_contact_event
from src.shared.python.motion_matching.same_input.integrator import zoh_rk4_step
from src.shared.python.motion_matching.same_input.plant import VectorPlant
from src.shared.python.motion_matching.same_input.stage_trace import fixed_substep_sizes

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]


def test_synthetic_driver_face_root_halves_fixed_integration_steps() -> None:
    pytest.importorskip("mujoco")
    spec = (
        Path(__file__).resolve().parents[3]
        / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
    ).read_bytes()
    plant = VectorPlant("mujoco", spec)
    # Solve only closure from a synthetic zero state; this is not a fit, a
    # physiological posture, or a captured swing.
    closure = least_squares(
        plant.closure_pose_residual,
        np.zeros(plant.nv),
        max_nfev=100,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    assert closure.success and np.max(np.abs(closure.fun)) < 1e-8
    q0, v0, effort = closure.x, np.zeros(plant.nv), np.zeros(plant.nv)
    face = face_triangles_in_frame(json.loads(spec))

    def face_world(q: np.ndarray) -> np.ndarray:
        pose = plant.kinematic_frames(q)["Clubhead"]
        return face @ pose[:3, :3].T + pose[:3, 3]

    triangle = face_world(q0)[300]
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    v0[:3] = normal
    v0[3:6] = (0.2, -0.3, 0.1)
    radius = 0.02135
    # Synthetic test ball, explicitly unrelated to the matched-swing address.
    ball = triangle.mean(axis=0) + normal * (radius + 5e-7)

    def run(step: float, centre: np.ndarray):
        def advance(t: float) -> tuple[np.ndarray, np.ndarray]:
            q, v = q0.copy(), v0.copy()
            if t > 0:
                for h in fixed_substep_sizes(t, step, 0.0):
                    q, v = zoh_rk4_step(plant, q, v, effort, h)
            return q, v

        def velocity(q: np.ndarray, v: np.ndarray, point: np.ndarray) -> np.ndarray:
            return plant.frame_point_velocity(q, v, "Clubhead", point)

        return find_contact_event(
            advance,
            face_world,
            centre,
            radius,
            duration_s=1e-6,
            time_tolerance_s=1e-11,
            distance_tolerance_m=1e-10,
            face_point_velocity_world=velocity,
            max_clearance_rate_mps=2.0,  # synthetic test assumption, not validated
            coverage_max_evaluations=2000,
        )

    coarse, fine = run(2.5e-7, ball), run(1.25e-7, ball)
    assert coarse.event is not None and fine.event is not None
    assert coarse.coverage_status == "conditional_bracket_candidate"
    assert coarse.event.approach_speed_mps > 0
    assert abs(np.linalg.norm(ball - coarse.event.contact_point_m) - radius) < 1e-10
    assert coarse.event.time_s == pytest.approx(fine.event.time_s, abs=1e-11)
    np.testing.assert_allclose(coarse.event.q, fine.event.q, atol=1e-10, rtol=0)
    np.testing.assert_allclose(coarse.event.qd, fine.event.qd, atol=1e-5, rtol=0)
    v0 *= -1
    receding = run(1.25e-7, ball)
    assert receding.event is None
    assert np.isfinite(receding.closest_approach_penalty_m)
    v0 *= -1
    tangent = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    tangent /= np.linalg.norm(tangent)
    off_face = run(1.25e-7, ball + 0.2 * tangent)
    assert off_face.event is None
    assert np.isfinite(off_face.closest_approach_penalty_m)
