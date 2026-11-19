"""Markers re-placed on their segments from a whole solved swing (#15267).

A marker placed from one posture carries that posture's error for the whole
swing. ``place_markers_over_swing`` moves each marker to its mean position in
its segment's frame over every frame it was seen in.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.swing_marker_placement import (
    PlacementSettings,
    place_markers_over_swing,
)

pytestmark = pytest.mark.unit
FRAMES = 60
TRUE = {"A": (0.03, -0.02, 0.10), "B": (-0.05, 0.04, 0.00), "C": (0.00, 0.06, -0.08)}
BODY = {"A": "thorax", "B": "thorax", "C": "pelvis"}


def _swing(noise_m: float = 0.0):
    """Two segments tumbling through space with markers fixed on them."""
    rng = np.random.default_rng(7)
    rotations, origins = {}, {}
    for body in ("thorax", "pelvis"):
        rotations[body] = np.array(
            [np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(FRAMES)]
        )
        origins[body] = rng.normal(size=(FRAMES, 3))
    labels = tuple(TRUE)
    points = np.zeros((FRAMES, len(labels), 3))
    for k, label in enumerate(labels):
        body = BODY[label]
        points[:, k] = origins[body] + rotations[body] @ np.array(TRUE[label])
    points += rng.normal(scale=noise_m, size=points.shape)
    return (
        points,
        np.ones((FRAMES, len(labels)), dtype=bool),
        labels,
        rotations,
        origins,
    )


def _start(shift=(0.02, 0.0, -0.01)):
    return {k: (BODY[k], tuple(np.add(TRUE[k], shift))) for k in TRUE}


def test_misplaced_markers_return_to_where_they_are() -> None:
    points, valid, labels, rotations, origins = _swing()
    placed = place_markers_over_swing(
        points, valid, labels, _start(), rotations, origins
    )
    for label, true in TRUE.items():
        np.testing.assert_allclose(placed[label].offset_m, true, atol=1e-9)
        assert placed[label].body == BODY[label]
        assert placed[label].rms_before_m == pytest.approx(np.hypot(0.02, 0.01))
        assert placed[label].rms_after_m <= 1e-9
        assert placed[label].moved_m == pytest.approx(np.hypot(0.02, 0.01))
        assert placed[label].frames_used == FRAMES


def test_the_fit_never_gets_worse_with_noise() -> None:
    points, valid, labels, rotations, origins = _swing(noise_m=0.004)
    placed = place_markers_over_swing(
        points, valid, labels, _start(), rotations, origins
    )
    for label, true in TRUE.items():
        assert placed[label].rms_after_m <= placed[label].rms_before_m
        assert np.linalg.norm(np.subtract(placed[label].offset_m, true)) < 0.003


def test_unseen_frames_are_left_out_and_a_rarely_seen_marker_stays() -> None:
    points, valid, labels, rotations, origins = _swing()
    points[:20, 0] = 9.0  # rubbish where marker A was not seen
    valid[:20, 0] = False
    valid[5:, 2] = False  # marker C seen in five frames only
    start = _start()
    placed = place_markers_over_swing(points, valid, labels, start, rotations, origins)
    np.testing.assert_allclose(placed["A"].offset_m, TRUE["A"], atol=1e-9)
    assert placed["A"].frames_used == FRAMES - 20
    assert placed["C"].offset_m == start["C"][1]
    assert placed["C"].moved_m == 0.0
    assert not placed["C"].replaced


def test_a_move_is_limited_to_the_setting() -> None:
    points, valid, labels, rotations, origins = _swing()
    placed = place_markers_over_swing(
        points,
        valid,
        labels,
        _start(shift=(0.10, 0.0, 0.0)),
        rotations,
        origins,
        PlacementSettings(max_move_m=0.03),
    )
    for label, true in TRUE.items():
        assert placed[label].moved_m == pytest.approx(0.03)
        assert placed[label].limited
        # Moved straight toward the true place, 30 mm of the 100 mm.
        np.testing.assert_allclose(
            placed[label].offset_m, np.add(true, (0.07, 0.0, 0.0)), atol=1e-9
        )


def test_bad_input_is_refused() -> None:
    points, valid, labels, rotations, origins = _swing()
    start = _start()
    with pytest.raises(ValueError, match="min_frames"):
        PlacementSettings(min_frames=0)
    with pytest.raises(ValueError, match="max_move_m"):
        PlacementSettings(max_move_m=-0.01)
    with pytest.raises(ValueError, match="shape"):
        place_markers_over_swing(
            points[:, :2], valid, labels, start, rotations, origins
        )
    with pytest.raises(ValueError, match="no placement"):
        place_markers_over_swing(
            points, valid, labels, {"A": start["A"]}, rotations, origins
        )
    with pytest.raises(ValueError, match="pelvis"):
        place_markers_over_swing(
            points, valid, labels, start, {"thorax": rotations["thorax"]}, origins
        )
    with pytest.raises(ValueError, match="frames"):
        place_markers_over_swing(
            points,
            valid,
            labels,
            start,
            {k: v[:10] for k, v in rotations.items()},
            origins,
        )
