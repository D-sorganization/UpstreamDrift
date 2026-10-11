"""Sagittal 3-link to full-body landmark retargeting (LIFT-5, #11745)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.lifting.sagittal_retarget import (
    LANDMARK_LABELS,
    SagittalFrame,
    SagittalSolution,
    retarget_landmarks,
    rotate_sagittal,
    segment_angles,
)

pytestmark = pytest.mark.unit

FRAME = SagittalFrame(forward=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))


def _standing() -> dict[str, np.ndarray]:
    """Upright landmarks: ankles at the origin plane, left is +y."""
    points: dict[str, np.ndarray] = {}
    for side, y in (("l", 0.1), ("r", -0.1)):
        points[f"ankle_{side}"] = np.array([0.0, y, 0.08])
        points[f"knee_{side}"] = np.array([0.0, y, 0.52])
        points[f"hip_{side}"] = np.array([0.0, y, 0.91])
        points[f"shoulder_{side}"] = np.array([0.0, 2 * y, 1.50])
    points["bar"] = np.array([-0.02, 0.0, 1.55])
    return points


def _solution(angles: np.ndarray) -> SagittalSolution:
    n = angles.shape[0]
    return SagittalSolution(
        time_s=np.linspace(0.0, 1.0, n),
        segment_angles_rad=angles,
        bar_m=np.zeros((n, 2)),
        source={"optimizer": "synthetic"},
    )


def test_rotation_moves_vertical_forward_for_positive_angle() -> None:
    rotated = rotate_sagittal(np.array([0.0, 0.3, 1.0]), np.pi / 2, FRAME)
    np.testing.assert_allclose(rotated, [1.0, 0.3, 0.0], atol=1e-15)


def test_segment_angles_of_upright_landmarks_are_zero() -> None:
    np.testing.assert_allclose(segment_angles(_standing(), FRAME), 0.0, atol=1e-15)


def test_retarget_at_the_standing_angles_reproduces_the_landmarks() -> None:
    standing = _standing()
    targets = retarget_landmarks(_solution(np.zeros((3, 3))), standing, FRAME)
    assert targets.shape == (3, len(LANDMARK_LABELS), 3)
    expected = np.array([standing[label] for label in LANDMARK_LABELS])
    for frame in targets:
        np.testing.assert_allclose(frame, expected, atol=1e-15)


def test_retarget_preserves_segment_lengths_and_fixes_the_feet() -> None:
    standing = _standing()
    angles = np.array([[0.4, -1.5, 1.0], [0.2, -0.7, 0.5]])
    targets = retarget_landmarks(_solution(angles), standing, FRAME)
    index = {label: k for k, label in enumerate(LANDMARK_LABELS)}
    for frame, wanted in zip(targets, angles, strict=True):
        shown = {label: frame[index[label]] for label in LANDMARK_LABELS}
        np.testing.assert_allclose(segment_angles(shown, FRAME), wanted, atol=1e-12)
        for side in "lr":
            ankle = f"ankle_{side}"
            np.testing.assert_array_equal(shown[ankle], standing[ankle])
            shank = np.linalg.norm(shown[f"knee_{side}"] - shown[ankle])
            assert shank == pytest.approx(0.44, abs=1e-12)
        bar_to_hip = shown["bar"] - 0.5 * (shown["hip_l"] + shown["hip_r"])
        assert np.linalg.norm(bar_to_hip) == pytest.approx(
            np.linalg.norm(standing["bar"] - np.array([0.0, 0.0, 0.91])), abs=1e-12
        )


def test_solution_rejects_bad_shapes_and_times() -> None:
    with pytest.raises(ValueError, match="segment_angles_rad"):
        SagittalSolution(np.arange(3.0), np.zeros((3, 2)), np.zeros((3, 2)), {})
    with pytest.raises(ValueError, match="bar_m"):
        SagittalSolution(np.arange(3.0), np.zeros((3, 3)), np.zeros((2, 2)), {})
    with pytest.raises(ValueError, match="increasing"):
        SagittalSolution(np.zeros(3), np.zeros((3, 3)), np.zeros((3, 2)), {})
    with pytest.raises(ValueError, match="finite"):
        SagittalSolution(np.arange(3.0), np.full((3, 3), np.nan), np.zeros((3, 2)), {})
    with pytest.raises(TypeError, match="source"):
        SagittalSolution(np.arange(3.0), np.zeros((3, 3)), np.zeros((3, 2)), [])


def test_solution_round_trips_through_its_record() -> None:
    solution = _solution(np.array([[0.1, -0.2, 0.3], [0.2, -0.4, 0.5]]))
    again = SagittalSolution.from_record(solution.to_record())
    np.testing.assert_array_equal(again.time_s, solution.time_s)
    np.testing.assert_array_equal(again.segment_angles_rad, solution.segment_angles_rad)
    np.testing.assert_array_equal(again.bar_m, solution.bar_m)
    assert again.source == solution.source


def test_frame_requires_orthonormal_axes() -> None:
    with pytest.raises(ValueError, match="orthogonal"):
        SagittalFrame(forward=(1.0, 0.0, 0.5), up=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        SagittalFrame(forward=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))


def test_retarget_rejects_missing_landmarks() -> None:
    standing = _standing()
    del standing["bar"]
    with pytest.raises(ValueError, match="bar"):
        retarget_landmarks(_solution(np.zeros((2, 3))), standing, FRAME)
