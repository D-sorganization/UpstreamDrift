"""Club face-angle observability audit (#10382, MS-108).

Synthetic observable and unobservable club marker configurations, plus
missing and gapped club markers. The audit reuses the triad thresholds of
:mod:`club_face_target`; it adds no tolerance of its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching import club_face_target as cft
from src.shared.python.motion_matching.club_observability import (
    REASON_DEGENERATE_TRIAD,
    REASON_MARKERS_NOT_CAPTURED,
    REASON_NO_OBSERVED_FRAME,
    REASON_OBSERVABLE,
    REASON_SHAPE_MISMATCH,
    REASON_TOO_FEW_MARKERS,
    FaceObservability,
    audit_face_observability,
)

pytestmark = pytest.mark.unit

TRIAD = cft.HEAD_TRIAD_LABELS
OFFSETS = {
    TRIAD[0]: np.array([-0.005, -0.047, -0.020]),
    TRIAD[1]: np.array([0.040, 0.034, -0.022]),
    TRIAD[2]: np.array([0.052, 0.015, 0.025]),
}
LABELS = (*TRIAD, "HeadTop")
FRAMES = 8


def _capture(
    offsets: dict[str, np.ndarray] = OFFSETS, frames: int = FRAMES
) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly translated triad plus one body marker, every frame valid."""
    local = np.array([offsets.get(label, np.zeros(3)) for label in TRIAD])
    points = np.zeros((frames, len(LABELS), 3))
    for k in range(frames):
        points[k, :3] = local + np.array([0.1 * k, 1.0, 0.5])
        points[k, 3] = [0.0, 0.0, 1.7]
    return points, np.ones(points.shape[:2], dtype=bool)


class TestObservableConfiguration:
    def test_full_rigid_triad_is_observable_with_full_coverage(self) -> None:
        points, valid = _capture()
        audit = audit_face_observability(points, valid, LABELS, OFFSETS)
        assert audit == FaceObservability(
            observable=True,
            reason=REASON_OBSERVABLE,
            observed_frames=FRAMES,
            total_frames=FRAMES,
            gaps=(),
        )
        assert audit.coverage == pytest.approx(1.0)

    def test_gapped_marker_reports_half_open_gap_runs_and_coverage(self) -> None:
        points, valid = _capture()
        valid[2:4, 1] = False  # one head marker occluded for two frames
        points[6, 0] = np.nan  # a non-finite sample counts as a gap
        audit = audit_face_observability(points, valid, LABELS, OFFSETS)
        assert audit.observable
        assert audit.gaps == ((2, 4), (6, 7))
        assert audit.observed_frames == FRAMES - 3
        assert audit.coverage == pytest.approx((FRAMES - 3) / FRAMES)

    def test_leading_and_trailing_gaps_are_reported(self) -> None:
        points, valid = _capture()
        valid[0, 2] = False
        valid[-2:, 0] = False
        audit = audit_face_observability(points, valid, LABELS, OFFSETS)
        assert audit.gaps == ((0, 1), (FRAMES - 2, FRAMES))

    def test_body_marker_gaps_do_not_affect_club_observability(self) -> None:
        points, valid = _capture()
        valid[:, 3] = False
        audit = audit_face_observability(points, valid, LABELS, OFFSETS)
        assert audit.observable
        assert audit.gaps == ()


class TestUnobservableConfiguration:
    def test_two_markers_cannot_observe_axial_rotation(self) -> None:
        points, valid = _capture()
        sparse = {label: OFFSETS[label] for label in TRIAD[:2]}
        audit = audit_face_observability(points, valid, LABELS, sparse)
        assert not audit.observable
        assert audit.reason == REASON_TOO_FEW_MARKERS
        assert audit.observed_frames == 0
        assert audit.coverage == 0.0

    def test_collinear_markers_cannot_observe_axial_rotation(self) -> None:
        collinear = {
            TRIAD[0]: np.array([0.0, 0.0, 0.0]),
            TRIAD[1]: np.array([0.0, 0.0, 0.05]),
            TRIAD[2]: np.array([0.0, 0.0, 0.10]),
        }
        points, valid = _capture(collinear)
        audit = audit_face_observability(points, valid, LABELS, collinear)
        assert not audit.observable
        assert audit.reason == REASON_DEGENERATE_TRIAD

    def test_triad_marker_absent_from_capture_is_unobservable(self) -> None:
        points, valid = _capture()
        labels = (TRIAD[0], TRIAD[1], "NotAClubMarker", "HeadTop")
        audit = audit_face_observability(points, valid, labels, OFFSETS)
        assert not audit.observable
        assert audit.reason == REASON_MARKERS_NOT_CAPTURED

    def test_marker_missing_in_every_frame_is_unobservable(self) -> None:
        points, valid = _capture()
        valid[:, 1] = False
        audit = audit_face_observability(points, valid, LABELS, OFFSETS)
        assert not audit.observable
        assert audit.reason == REASON_NO_OBSERVED_FRAME
        assert audit.gaps == ((0, FRAMES),)

    def test_offsets_of_another_club_are_unobservable(self) -> None:
        points, valid = _capture()
        other = {label: 2.0 * offset for label, offset in OFFSETS.items()}
        audit = audit_face_observability(points, valid, LABELS, other)
        assert not audit.observable
        assert audit.reason == REASON_SHAPE_MISMATCH


class TestContract:
    def test_mismatched_mask_shape_is_a_caller_error(self) -> None:
        points, valid = _capture()
        with pytest.raises(ValueError, match="matching mask"):
            audit_face_observability(points, valid[:, :2], LABELS, OFFSETS)

    def test_empty_capture_is_a_caller_error(self) -> None:
        points, valid = _capture(frames=0)
        with pytest.raises(ValueError, match="at least one frame"):
            audit_face_observability(points, valid, LABELS, OFFSETS)

    def test_observable_result_requires_an_observed_frame(self) -> None:
        with pytest.raises(ValueError, match="observed frame"):
            FaceObservability(
                observable=True,
                reason=REASON_OBSERVABLE,
                observed_frames=0,
                total_frames=4,
                gaps=((0, 4),),
            )
