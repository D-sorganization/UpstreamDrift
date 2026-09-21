"""Alternating marker-offset calibration with injected FK and IK (OS-3 core)."""

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching import (
    marker_calibration as module,
)
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
    # Two bodies translate with q (no rotation); true offsets are fixed.
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
        np.arange(frames) / 360, labels, points, np.ones((frames, len(labels)), bool)
    )

    def ik(
        offsets: dict[str, tuple[str, tuple[float, float, float]]], cap: TourCapture
    ) -> np.ndarray:
        # Exact least squares for the translation-only rig.
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
    # Offsets and coordinates share a translation gauge on this rig, so the
    # verifiable contract is the fit: placements plus IK reproduce the capture.
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
    # Rig where IK degrades on later iterations (e.g. iteration 1 ok, iteration 2 optimal, iteration 3 degraded)
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
        iter_idx = call_count[0]
        # Iteration 1: shift by +0.20
        # Iteration 2: shift by +0.05 -> lowest error
        # Iteration 3: degrade by +0.40 -> higher error
        shift = {1: 0.20, 2: 0.05, 3: 0.40}.get(iter_idx, 0.50)
        q = np.zeros((cap.frames, 1))
        for f in range(cap.frames):
            q[f, 0] = float(f) + shift
        return q

    result = module.calibrate_marker_offsets(
        capture, bodies, poses, oscillating_ik, initial_q=np.zeros(1), iterations=3
    )
    assert result.iterations == 3
    assert len(result.rms_per_iteration_m) == 3
    # Iteration 2 has the lowest RMS
    assert result.best_iteration == 2
    assert result.rms_per_iteration_m[1] < result.rms_per_iteration_m[0]
    assert result.rms_per_iteration_m[1] < result.rms_per_iteration_m[2]

    # Returned q must correspond to the best iteration (iteration 2), not iteration 3
    for f in range(frames):
        assert result.q[f, 0] == pytest.approx(float(f) + 0.05, abs=1e-5)

    assert set(result.per_marker_rms_m.keys()) == set(labels)
    for label in labels:
        assert result.per_marker_rms_m[label] == pytest.approx(
            result.rms_per_iteration_m[1], rel=1e-5
        )


def test_static_offsets_recover_body_frame_positions_over_valid_frames() -> None:
    rot, trans = _rotation(0.3), np.array([0.5, -1.0, 2.0])
    offsets = {"A": (0.1, 0.0, 0.2), "B": (-0.1, 0.3, 0.0)}
    world = np.array(
        [[rot @ np.array(o) + trans for o in offsets.values()] for _ in range(3)]
    )
    world[1, 0] += 5.0  # invalid frame carries garbage
    capture = TourCapture(
        time_s=np.array([0.0, 0.1, 0.2]),
        labels=("A", "B"),
        points_m=world,
        valid=np.array([[True, True], [False, True], [True, True]]),
    )
    bodies = {"A": "trunk", "B": "trunk"}
    poses = [{"trunk": (rot, trans)}] * 3
    result = module.static_marker_offsets(capture, bodies, poses)
    for label, offset in offsets.items():
        assert result[label][0] == "trunk"
        np.testing.assert_allclose(result[label][1], offset, atol=1e-12)
    with pytest.raises(ValueError):
        module.static_marker_offsets(capture, bodies, poses[:2])
    with pytest.raises(ValueError):
        module.static_marker_offsets(capture, {"A": "arm", "B": "trunk"}, poses)


def test_bound_marker_offsets_clamps_radius_and_prior_deviation() -> None:
    """Verify that bound_marker_offsets restricts offsets within anatomical bounds."""
    raw_offsets = {
        "M_Normal": ("torso", (0.1, 0.0, 0.0)),
        "M_Far": ("torso", (1.0, 0.0, 0.0)),  # Exceeds max radius 0.5m
        "M_PriorDeviated": ("torso", (0.2, 0.0, 0.0)),
    }
    priors = {"M_PriorDeviated": (0.0, 0.0, 0.0)}  # Deviated by 0.2m > 0.08m

    bounded = module.bound_marker_offsets(
        raw_offsets,
        max_offset_radius_m=0.50,
        prior_offsets=priors,
        max_deviation_from_prior_m=0.08,
    )

    # Normal remains untouched
    np.testing.assert_allclose(bounded["M_Normal"][1], (0.1, 0.0, 0.0))
    # M_Far is clamped to radius 0.50m
    np.testing.assert_allclose(bounded["M_Far"][1], (0.5, 0.0, 0.0))
    # M_PriorDeviated is clamped to distance 0.08m from prior
    np.testing.assert_allclose(bounded["M_PriorDeviated"][1], (0.08, 0.0, 0.0))


def test_calibrate_marker_offsets_with_holdout() -> None:
    """Verify holdout evaluation during marker calibration."""
    labels = ("M1", "M2", "M3", "M_Holdout")
    bodies = dict.fromkeys(labels, "body")
    frames = 4

    def poses(q: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {"body": (np.eye(3), np.array([q[0], 0.0, 0.0]))}

    true_offsets = {
        "M1": (0.1, 0.0, 0.0),
        "M2": (0.0, 0.1, 0.0),
        "M3": (0.0, 0.0, 0.1),
        "M_Holdout": (0.2, 0.2, 0.0),
    }
    points = np.zeros((frames, len(labels), 3))
    for f in range(frames):
        for i, lb in enumerate(labels):
            points[f, i] = np.array(true_offsets[lb]) + np.array([float(f), 0.0, 0.0])

    capture = TourCapture(
        np.arange(frames) / 360.0, labels, points, np.ones((frames, len(labels)), bool)
    )

    def ik(offsets: dict, cap: TourCapture) -> np.ndarray:
        q = np.zeros((cap.frames, 1))
        for f in range(cap.frames):
            q[f, 0] = float(f)
        return q

    result, holdout_rms = module.calibrate_marker_offsets_with_holdout(
        capture,
        bodies,
        poses,
        ik,
        initial_q=np.zeros(1),
        iterations=2,
        holdout_labels=["M_Holdout"],
    )
    assert holdout_rms < 1e-9
    assert result.rms_per_iteration_m[-1] < 1e-9
    assert "M_Holdout" not in result.offsets  # Holdout not included in train offsets
