"""A sampled candidate is never a first-contact certification."""

import numpy as np
import pytest

from src.shared.python.motion_matching.face_trajectory_diagnostic import diagnose

pytestmark = pytest.mark.unit
FACE = np.array([[[0.0, -1, -1], [0.0, 1, -1], [0.0, 0, 1]]])


def advance(t):
    return np.array([t]), np.array([1.0])


def triangles(q):
    return FACE + [q[0], 0, 0]


def velocity(q, qd, point):
    return np.array([qd[0], 0.0, 0.0])


def test_approaching_finite_face_candidate_is_unqualified():
    result = diagnose(advance, triangles, velocity, [0.7, 0, 0], 0.2, 0.8)
    assert result.status == "unqualified"
    assert result.coverage == "uncertified"
    assert result.candidate is not None
    assert result.candidate.time_s == pytest.approx(0.5, abs=1e-8)
    assert result.candidate.approach_mps == pytest.approx(1.0)
    np.testing.assert_allclose(result.candidate.q, [0.5], atol=1e-8)


def test_endpoint_root_never_evaluates_outside_interval():
    def bounded(t):
        assert 0 <= t <= 0.5
        return advance(t)

    result = diagnose(bounded, triangles, velocity, [0.7, 0, 0], 0.2, 0.5)
    assert result.candidate is not None
    assert result.candidate.time_s == pytest.approx(0.5, abs=1e-8)
    assert result.coverage == "uncertified"


def test_initial_overlap_and_unapproaching_root_fail_closed():
    with pytest.raises(ValueError, match="initial"):
        diagnose(advance, triangles, velocity, [0.1, 0, 0], 0.2, 0.8)

    def reverse(q, qd, point):
        return np.array([-1.0, 0, 0])

    result = diagnose(advance, triangles, reverse, [0.7, 0, 0], 0.2, 0.8)
    assert result.candidate is None and result.status == "unqualified"


def test_sampled_miss_never_certifies_absence_of_earlier_contact():
    # At both endpoints the face is separated, but it crosses in between.
    def oscillating(t):
        return np.array([0.6 * np.sin(np.pi * t)]), np.array([0.0])

    result = diagnose(
        oscillating, triangles, velocity, [0.7, 0, 0], 0.2, 1.0, samples=2
    )
    assert result.candidate is None
    assert result.coverage == "uncertified" and result.status == "unqualified"


def test_backside_and_triangle_edge_are_not_accepted_as_candidates():
    def backwards(t):
        return np.array([-t]), np.array([-1.0])

    backside = diagnose(backwards, triangles, velocity, [-0.7, 0, 0], 0.2, 0.8)
    assert backside.candidate is None
    edge = diagnose(advance, triangles, velocity, [0.7, 0, -1], 0.2, 0.8)
    assert edge.candidate is None


def test_invalid_trajectory_state_fails_closed():
    def invalid(t):
        return np.array([np.nan]), np.array([1.0])

    with pytest.raises(ValueError, match="finite"):
        diagnose(invalid, triangles, velocity, [0.7, 0, 0], 0.2, 0.8)


def test_discontinuous_gap_cannot_become_a_candidate():
    def jump(t):
        return np.array([0.0 if t < 0.5 else 0.6]), np.array([1.0])

    result = diagnose(jump, triangles, velocity, [0.7, 0, 0], 0.2, 0.8)
    assert result.candidate is None
    assert result.coverage == "uncertified"


@pytest.mark.requires_mujoco
def test_synthetic_v2_mujoco_candidate_converges_under_step_halving():
    import json
    from pathlib import Path

    pytest.importorskip("mujoco")
    from scipy.optimize import least_squares

    from src.shared.python.motion_matching.club_face_target import (
        face_triangles_in_frame,
    )
    from src.shared.python.motion_matching.same_input import VectorPlant, zoh_rk4_step

    spec = Path(__file__).resolve().parents[3] / (
        "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
    )
    spec_bytes = spec.read_bytes()
    plant = VectorPlant("mujoco", spec_bytes)
    fit = least_squares(plant.closure_pose_residual, np.zeros(plant.nv), max_nfev=100)
    assert fit.success and np.max(np.abs(fit.fun)) < 1e-7
    q0, v0, effort = fit.x, np.zeros(plant.nv), np.zeros(plant.nv)
    local_face = face_triangles_in_frame(json.loads(spec_bytes))

    def pose(q):
        return plant.kinematic_frames(q)["Clubhead"]

    def world_faces(q):
        transform = pose(q)
        return local_face @ transform[:3, :3].T + transform[:3, 3]

    triangle = world_faces(q0)[300]
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    v0[:3] = normal
    v0[3:6] = [0.2, -0.3, 0.1]
    ball = triangle.mean(axis=0) + normal * (0.02135 + 5e-7)

    def material_velocity(q, qd, point):
        frame = pose(q)
        local = frame[:3, :3].T @ (point - frame[:3, 3])
        eps = 1e-7
        ahead, behind = pose(q + eps * qd), pose(q - eps * qd)
        return (
            (ahead[:3, :3] @ local + ahead[:3, 3])
            - (behind[:3, :3] @ local + behind[:3, 3])
        ) / (2 * eps)

    def run(step):
        def advance_state(t):
            q, v = q0.copy(), v0.copy()
            count = max(1, int(np.ceil(t / step)))
            if t > 0:
                for _ in range(count):
                    q, v = zoh_rk4_step(plant, q, v, effort, t / count)
            return q, v

        return diagnose(
            advance_state,
            world_faces,
            material_velocity,
            ball,
            0.02135,
            1e-6,
            time_tolerance_s=1e-11,
            distance_tolerance_m=1e-10,
        )

    coarse, fine = run(2.5e-7), run(1.25e-7)
    assert coarse.status == fine.status == "unqualified"
    assert coarse.coverage == fine.coverage == "uncertified"
    assert coarse.candidate is not None and fine.candidate is not None
    assert coarse.candidate.time_s == pytest.approx(fine.candidate.time_s, abs=1e-11)
