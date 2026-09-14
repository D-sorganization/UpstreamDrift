"""Unit tests for shared alternating marker calibration (FB-4)."""

import numpy as np
import pytest

from src.shared.python.motion_matching import marker_calibration as module
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

pytestmark = pytest.mark.unit


def _rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_rigid_pose_recovers_rotation_and_translation() -> None:
    body = np.array(
        [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3], [0.1, 0.1, 0.1]]
    )
    rot, trans = _rotation(0.7), np.array([1.0, -2.0, 0.5])
    world = body @ rot.T + trans
    r, t = module.rigid_pose_from_markers(body, world)
    np.testing.assert_allclose(r, rot, atol=1e-12)
    np.testing.assert_allclose(t, trans, atol=1e-12)
    np.testing.assert_allclose(module.express_in_body(world, r, t), body, atol=1e-12)
    with pytest.raises(ValueError):
        module.rigid_pose_from_markers(body[:2], world[:2])
    with pytest.raises(ValueError):
        module.rigid_pose_from_markers(body, world[:3])


def test_alternating_calibration_converges_on_a_two_body_rig() -> None:
    true_offsets = {
        "A1": (0.1, 0.0, 0.0),
        "A2": (0.0, 0.1, 0.0),
        "A3": (0.0, 0.0, 0.1),
        "B1": (0.2, 0.0, 0.0),
        "B2": (0.0, 0.2, 0.0),
        "B3": (0.0, 0.0, 0.2),
    }
    bodies = {k: ("A" if k.startswith("A") else "B") for k in true_offsets}
    frames = 5
    q_true = np.column_stack((np.linspace(0, 1, frames), np.linspace(2, 3, frames)))

    def poses(q: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            "A": (np.eye(3), np.array([q[0], 0.0, 0.0])),
            "B": (np.eye(3), np.array([0.0, q[1], 0.0])),
        }

    labels = tuple(true_offsets)
    points = np.zeros((frames, len(labels), 3))
    for f in range(frames):
        pose = poses(q_true[f])
        for i, label in enumerate(labels):
            r, t = pose[bodies[label]]
            points[f, i] = r @ np.array(true_offsets[label]) + t
    capture = TourCapture(
        np.arange(frames) / 360.0, labels, points, np.ones((frames, len(labels)), bool)
    )

    def ik(
        offsets: dict[str, tuple[str, tuple[float, float, float]]], cap: TourCapture
    ) -> np.ndarray:
        out = np.zeros((cap.frames, 2))
        for f in range(cap.frames):
            for body, col in (("A", 0), ("B", 1)):
                idx = [i for i, lb in enumerate(cap.labels) if offsets[lb][0] == body]
                diffs = cap.points_m[f, idx] - np.array(
                    [offsets[lb][1] for lb in np.array(cap.labels)[idx]]
                )
                out[f, col] = float(np.mean(diffs[:, col]))
        return out

    result = module.calibrate_marker_offsets(
        capture, bodies, poses, ik, initial_q=np.zeros(2), iterations=4
    )
    assert result.rms_per_iteration_m[-1] < 1e-9
    for f in range(frames):
        pose = poses(result.q[f])
        for i, label in enumerate(labels):
            body, offset = result.offsets[label]
            r, t = pose[body]
            np.testing.assert_allclose(
                r @ np.array(offset) + t, points[f, i], atol=1e-9
            )
    assert result.iterations == 4 and result.q.shape == (frames, 2)
    assert 1 <= result.best_iteration <= 4
    assert result.rms_per_iteration_m[result.best_iteration - 1] < 1e-9
    assert set(result.per_marker_rms_m.keys()) == set(labels)
    for label in labels:
        assert result.per_marker_rms_m[label] < 1e-9

    with pytest.raises(ValueError):
        module.calibrate_marker_offsets(
            capture, bodies, poses, ik, initial_q=np.zeros(2), iterations=0
        )
    with pytest.raises(ValueError):
        module.calibrate_marker_offsets(
            capture, {"A1": "A"}, poses, ik, initial_q=np.zeros(2), iterations=1
        )


def test_alternating_calibration_selects_best_iteration_on_oscillation() -> None:
    labels = ("M1", "M2", "M3")
    bodies = {"M1": "B", "M2": "B", "M3": "B"}
    frames = 3
    true_offsets = {"M1": (0.1, 0.0, 0.0), "M2": (0.0, 0.1, 0.0), "M3": (0.0, 0.0, 0.1)}

    def poses(q: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {"B": (np.eye(3), np.array([q[0], 0.0, 0.0]))}

    points = np.zeros((frames, len(labels), 3))
    for f in range(frames):
        for i, label in enumerate(labels):
            points[f, i] = np.array(true_offsets[label]) + np.array(
                [float(f), 0.0, 0.0]
            )
    capture = TourCapture(
        np.arange(frames) / 360.0, labels, points, np.ones((frames, len(labels)), bool)
    )

    call_count = [0]

    def oscillating_ik(
        offsets: dict[str, tuple[str, tuple[float, float, float]]], cap: TourCapture
    ) -> np.ndarray:
        call_count[0] += 1
        shift = {1: 0.20, 2: 0.05, 3: 0.40}.get(call_count[0], 0.50)
        q = np.zeros((cap.frames, 1))
        for f in range(cap.frames):
            q[f, 0] = float(f) + shift
        return q

    result = module.calibrate_marker_offsets(
        capture, bodies, poses, oscillating_ik, initial_q=np.zeros(1), iterations=3
    )
    assert result.iterations == 3
    assert len(result.rms_per_iteration_m) == 3
    assert result.best_iteration == 2
    assert result.rms_per_iteration_m[1] < result.rms_per_iteration_m[0]
    assert result.rms_per_iteration_m[1] < result.rms_per_iteration_m[2]

    for f in range(frames):
        assert result.q[f, 0] == pytest.approx(float(f) + 0.05, abs=1e-5)

    assert set(result.per_marker_rms_m.keys()) == set(labels)
    for label in labels:
        assert result.per_marker_rms_m[label] == pytest.approx(
            result.rms_per_iteration_m[1], rel=1e-5
        )
