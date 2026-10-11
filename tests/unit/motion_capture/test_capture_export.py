"""Unit tests for capture export pure functions (Issue #11163).

Tests adhere strictly to Design by Contract (DbC), testing both valid behavior
and contract violations on synthetic arrays with no private data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from src.motion_capture.capture_export import (
    detect_impact_frame,
    fill_short_gaps,
    relabel_markers,
    subject_parameters,
    to_capture_frame,
)
from src.shared.python.contracts import PostconditionError, PreconditionError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. fill_short_gaps
# ---------------------------------------------------------------------------


class TestFillShortGaps:
    """Unit and contract tests for fill_short_gaps."""

    def test_precondition_non_array_raises(self) -> None:
        with pytest.raises((PreconditionError, TypeError)):
            fill_short_gaps([1.0, 2.0, 3.0], max_gap=2)  # type: ignore[arg-type]

    def test_precondition_invalid_ndim_raises(self) -> None:
        # 3D array is not supported
        arr_3d = np.zeros((5, 3, 3))
        with pytest.raises((PreconditionError, ValueError)):
            fill_short_gaps(arr_3d, max_gap=2)

    def test_precondition_invalid_max_gap_raises(self) -> None:
        arr = np.array([1.0, np.nan, 3.0])
        with pytest.raises((PreconditionError, ValueError)):
            fill_short_gaps(arr, max_gap=0)
        with pytest.raises((PreconditionError, ValueError)):
            fill_short_gaps(arr, max_gap=-2)

    def test_no_gaps_returns_identical_array(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        filled, count = fill_short_gaps(x, max_gap=2)
        assert count == 0
        np.testing.assert_array_equal(filled, x)
        assert filled is not x  # Pure function: returns a copy

    def test_1d_short_gap_filled_linearly(self) -> None:
        # 1-sample gap
        x = np.array([1.0, np.nan, 3.0])
        filled, count = fill_short_gaps(x, max_gap=1)
        assert count == 1
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(filled, expected)

        # 3-sample gap
        x3 = np.array([10.0, np.nan, np.nan, np.nan, 50.0])
        filled3, count3 = fill_short_gaps(x3, max_gap=3)
        assert count3 == 3
        expected3 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        np.testing.assert_allclose(filled3, expected3)

    def test_gap_longer_than_max_gap_not_filled(self) -> None:
        x = np.array([10.0, np.nan, np.nan, np.nan, np.nan, 50.0])
        filled, count = fill_short_gaps(x, max_gap=3)
        assert count == 0
        assert np.isnan(filled[1:5]).all()
        assert filled[0] == 10.0
        assert filled[5] == 50.0

    def test_leading_and_trailing_gaps_not_filled(self) -> None:
        x = np.array([np.nan, np.nan, 1.0, np.nan, 3.0, np.nan, np.nan])
        filled, count = fill_short_gaps(x, max_gap=2)
        # Leading (len 2) and trailing (len 2) remain NaN; middle 1-frame gap filled
        assert count == 1
        assert np.isnan(filled[0])
        assert np.isnan(filled[1])
        assert filled[2] == 1.0
        assert filled[3] == 2.0
        assert filled[4] == 3.0
        assert np.isnan(filled[5])
        assert np.isnan(filled[6])

    def test_input_array_not_mutated(self) -> None:
        x = np.array([1.0, np.nan, 3.0])
        _ = fill_short_gaps(x, max_gap=1)
        assert np.isnan(x[1])

    def test_empty_array_handling(self) -> None:
        empty_1d = np.array([], dtype=np.float64)
        filled_1d, count_1d = fill_short_gaps(empty_1d, max_gap=2)
        assert count_1d == 0
        assert len(filled_1d) == 0

        empty_2d = np.zeros((0, 3), dtype=np.float64)
        filled_2d, count_2d = fill_short_gaps(empty_2d, max_gap=2)
        assert count_2d == 0
        assert filled_2d.shape == (0, 3)

    def test_all_nan_array_remains_unfilled(self) -> None:
        all_nan = np.full(10, np.nan)
        filled, count = fill_short_gaps(all_nan, max_gap=5)
        assert count == 0
        assert np.isnan(filled).all()

    def test_2d_array_per_column_filling(self) -> None:
        # Array of shape (5, 3)
        x = np.array(
            [
                [1.0, 10.0, 100.0],
                [np.nan, 20.0, np.nan],
                [3.0, np.nan, np.nan],
                [4.0, 40.0, 400.0],
                [5.0, 50.0, 500.0],
            ]
        )
        filled, count = fill_short_gaps(x, max_gap=2)
        # Col 0: 1 gap at idx 1 (len 1 <= 2) -> filled (count=1)
        # Col 1: 1 gap at idx 2 (len 1 <= 2) -> filled (count=1)
        # Col 2: gap at idx 1..2 (len 2 <= 2) -> filled (count=2)
        assert count == 4
        assert filled.shape == (5, 3)
        assert filled[1, 0] == 2.0
        assert filled[2, 1] == 30.0
        np.testing.assert_allclose(filled[1:3, 2], [200.0, 300.0])


# ---------------------------------------------------------------------------
# 2. relabel_markers
# ---------------------------------------------------------------------------


class TestRelabelMarkers:
    """Unit and contract tests for relabel_markers."""

    def test_precondition_non_array_raises(self) -> None:
        with pytest.raises((PreconditionError, TypeError)):
            relabel_markers([[1, 2, 3]], ["M1"], ["M1"])  # type: ignore[arg-type]

    def test_precondition_invalid_ndim_raises(self) -> None:
        arr_2d = np.zeros((10, 3))
        with pytest.raises((PreconditionError, ValueError)):
            relabel_markers(arr_2d, ["M1"], ["M1"])

    def test_precondition_label_count_mismatch_raises(self) -> None:
        points = np.zeros((10, 2, 3))
        with pytest.raises((PreconditionError, ValueError)):
            relabel_markers(points, ["M1"], ["M1"])

    def test_precondition_invalid_layout_raises(self) -> None:
        points = np.zeros((4, 2, 3))
        with pytest.raises((PreconditionError, ValueError)):
            relabel_markers(points, ["A", "B"], ["A", "B"], layout="invalid_layout")

    def test_explicit_layout_parameter(self) -> None:
        # Both shapes equal 3 (coords and frames)
        points = np.zeros((3, 2, 3))
        points[:, 0, :] = 5.0
        points[:, 1, :] = 8.0
        reordered = relabel_markers(
            points, ["M1", "M2"], ["M2", "M1"], layout="(N, M, 3)"
        )
        assert reordered.shape == (3, 2, 3)
        np.testing.assert_allclose(reordered[:, 0, :], 8.0)
        np.testing.assert_allclose(reordered[:, 1, :], 5.0)

    def test_precondition_missing_required_target_label_raises(self) -> None:
        points = np.ones((5, 2, 3))
        labels = ["M1", "M2"]
        target_labels = ["M1", "M3"]  # M3 is missing from source and not optional
        with pytest.raises((PreconditionError, ValueError), match="M3"):
            relabel_markers(points, labels, target_labels)

    def test_reorder_frames_first_layout(self) -> None:
        # Layout: (N, M, 3) where N=4 frames, M=3 markers
        points = np.zeros((4, 3, 3))
        points[:, 0, :] = 10.0  # Marker A
        points[:, 1, :] = 20.0  # Marker B
        points[:, 2, :] = 30.0  # Marker C
        labels = ["A", "B", "C"]
        target_labels = ["C", "A", "B"]

        reordered = relabel_markers(points, labels, target_labels)
        assert reordered.shape == (4, 3, 3)
        np.testing.assert_allclose(reordered[:, 0, :], 30.0)
        np.testing.assert_allclose(reordered[:, 1, :], 10.0)
        np.testing.assert_allclose(reordered[:, 2, :], 20.0)

    def test_reorder_coords_first_layout(self) -> None:
        # Layout: (3, M, N) where 3 coords, M=3 markers, N=5 frames
        points = np.zeros((3, 3, 5))
        points[:, 0, :] = 1.0  # Marker X
        points[:, 1, :] = 2.0  # Marker Y
        points[:, 2, :] = 3.0  # Marker Z
        labels = ["X", "Y", "Z"]
        target_labels = ["Z", "X"]

        reordered = relabel_markers(points, labels, target_labels)
        assert reordered.shape == (3, 2, 5)
        np.testing.assert_allclose(reordered[:, 0, :], 3.0)
        np.testing.assert_allclose(reordered[:, 1, :], 1.0)

    def test_optional_target_labels_filled_with_nan(self) -> None:
        # Layout: (N, M, 3)
        points = np.ones((3, 2, 3))
        labels = ["M1", "M2"]
        target_labels = ["M1", "OPT_UNKNOWN", "M2"]
        optional = ["OPT_UNKNOWN"]

        reordered = relabel_markers(
            points, labels, target_labels, optional_labels=optional
        )
        assert reordered.shape == (3, 3, 3)
        np.testing.assert_allclose(reordered[:, 0, :], 1.0)
        assert np.isnan(reordered[:, 1, :]).all()
        np.testing.assert_allclose(reordered[:, 2, :], 1.0)


# ---------------------------------------------------------------------------
# 3. to_capture_frame
# ---------------------------------------------------------------------------


class TestToCaptureFrame:
    """Unit and contract tests for to_capture_frame."""

    def test_precondition_non_array_raises(self) -> None:
        with pytest.raises((PreconditionError, TypeError)):
            to_capture_frame(
                [1, 2, 3],  # type: ignore[arg-type]
                from_axes="x-forward,y-left,z-up",
                to_axes="x-forward,y-left,z-up",
            )

    def test_precondition_invalid_convention_syntax_raises(self) -> None:
        pts = np.array([1.0, 2.0, 3.0])
        with pytest.raises((PreconditionError, ValueError)):
            to_capture_frame(
                pts, from_axes="invalid_axes", to_axes="x-forward,y-left,z-up"
            )

    def test_precondition_non_orthogonal_axes_raises(self) -> None:
        pts = np.array([1.0, 2.0, 3.0])
        # x and y both mapped to forward -> non-orthogonal
        with pytest.raises((PreconditionError, ValueError)):
            to_capture_frame(
                pts,
                from_axes="x-forward,y-forward,z-up",
                to_axes="x-forward,y-left,z-up",
            )

    def test_handedness_inversion_violates_det_postcondition(self) -> None:
        pts = np.array([1.0, 2.0, 3.0])
        # from is right-handed (det = +1), to is left-handed (det = -1) -> R det = -1
        with pytest.raises((PostconditionError, PreconditionError, ValueError)):
            to_capture_frame(
                pts,
                from_axes="x-forward,y-left,z-up",
                to_axes="x-forward,y-right,z-up",
            )

    def test_single_vector_transformation(self) -> None:
        pt = np.array([1.0, 2.0, 3.0])
        from_axes = "x-forward,y-up,z-right"
        to_axes = "x-forward,y-left,z-up"
        res = to_capture_frame(pt, from_axes=from_axes, to_axes=to_axes)
        np.testing.assert_allclose(res, [1.0, -3.0, 2.0])

    def test_identity_transformation(self) -> None:
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        conv = "x-forward,y-left,z-up"
        res = to_capture_frame(pts, from_axes=conv, to_axes=conv)
        np.testing.assert_array_equal(res, pts)

    def test_y_up_to_z_up_capture_to_simscape_equivalence(self) -> None:
        # Y-up capture convention: x-forward, y-up, z-right (right-handed)
        # Simscape convention: x-forward, y-left, z-up (right-handed)
        # Formula: (x, y, z) -> (x, -z, y)
        from_axes = "x-forward,y-up,z-right"
        to_axes = "x-forward,y-left,z-up"

        pts = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        res = to_capture_frame(pts, from_axes=from_axes, to_axes=to_axes)

        expected = np.array([[1.0, -3.0, 2.0], [10.0, -30.0, 20.0]])
        np.testing.assert_allclose(res, expected)

    def test_coords_first_3d_array_transformation(self) -> None:
        # (3, M, N)
        pts = np.zeros((3, 2, 4))
        pts[0, :, :] = 1.0  # x
        pts[1, :, :] = 2.0  # y
        pts[2, :, :] = 3.0  # z

        from_axes = "x-forward,y-up,z-right"
        to_axes = "x-forward,y-left,z-up"
        res = to_capture_frame(pts, from_axes=from_axes, to_axes=to_axes)

        assert res.shape == (3, 2, 4)
        np.testing.assert_allclose(res[0, :, :], 1.0)
        np.testing.assert_allclose(res[1, :, :], -3.0)
        np.testing.assert_allclose(res[2, :, :], 2.0)


# ---------------------------------------------------------------------------
# 4. detect_impact_frame
# ---------------------------------------------------------------------------


class TestDetectImpactFrame:
    """Unit and contract tests for detect_impact_frame."""

    def _generate_synthetic_arc_swing(
        self, n_frames: int = 101, peak_frame: int = 60
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic swing whose club head follows a known arc.

        Speed increases along the arc to a maximum at peak_frame, then decreases.
        """
        t = np.linspace(0.0, 1.0, n_frames)
        # Parameterize angle along an arc: theta(t) with max angular velocity at peak_frame
        # Speed profile: Gaussian-shaped acceleration peaking at t[peak_frame]
        t_peak = t[peak_frame]
        sigma = 0.15
        speed_profile = 20.0 + 30.0 * np.exp(-0.5 * ((t - t_peak) / sigma) ** 2)

        # Integrate speed to get arc length s(t) using trapezoidal rule (unbiased centered integration)
        dt = t[1] - t[0]
        s = np.zeros_like(speed_profile)
        s[1:] = np.cumsum(0.5 * (speed_profile[:-1] + speed_profile[1:]) * dt)
        radius = 1.2  # club radius in metres
        theta = s / radius

        # Circular arc in X-Z plane
        club_head = np.column_stack(
            [radius * np.cos(theta), np.zeros_like(theta), radius * np.sin(theta)]
        )
        return club_head, t

    def test_precondition_invalid_shapes_raises(self) -> None:
        t = np.linspace(0, 1, 10)
        with pytest.raises((PreconditionError, ValueError)):
            detect_impact_frame(np.zeros((10, 2)), t)
        with pytest.raises((PreconditionError, ValueError)):
            detect_impact_frame(np.zeros((9, 3)), t)

    def test_precondition_non_monotonic_time_raises(self) -> None:
        club_head = np.zeros((5, 3))
        t = np.array([0.0, 0.1, 0.05, 0.3, 0.4])
        with pytest.raises((PreconditionError, ValueError)):
            detect_impact_frame(club_head, t)

    def test_precondition_insufficient_valid_frames_raises(self) -> None:
        t = np.linspace(0, 1, 10)
        club_head = np.full((10, 3), np.nan)
        club_head[5] = [1.0, 2.0, 3.0]  # Only 1 valid frame
        with pytest.raises((PreconditionError, ValueError)):
            detect_impact_frame(club_head, t)

    def test_downswing_window_constraint(self) -> None:
        club_head, t = self._generate_synthetic_arc_swing(n_frames=101, peak_frame=60)
        # Constrain search to frames [20, 50] (before peak at 60)
        impact = detect_impact_frame(club_head, t, downswing_window=(20, 50))
        # Within [20, 50), the highest speed should be at frame 49
        assert 20 <= impact < 50

    def test_known_arc_swing_detects_true_peak(self) -> None:
        # 1. Synthetic swing whose club head follows a known arc
        club_head, t = self._generate_synthetic_arc_swing(n_frames=101, peak_frame=60)
        impact = detect_impact_frame(club_head, t)
        assert impact == 60

    def test_swing_with_3_frame_gap_at_impact(self) -> None:
        # 2. That swing with a 3-frame gap at impact
        club_head, t = self._generate_synthetic_arc_swing(n_frames=101, peak_frame=60)
        # Drop out frames 59, 60, 61
        club_head[59:62, :] = np.nan

        impact = detect_impact_frame(club_head, t)
        # Must return the nearest valid frame (58 or 62) and never a gap frame
        assert impact in (58, 62)
        assert not np.isnan(club_head[impact]).any()
        assert impact not in (59, 60, 61)

    def test_swing_with_single_frame_dropout_at_impact_reports(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 3. A swing with a dropout exactly at the impact frame,
        # where it must return the nearest valid frame and report it.
        club_head, t = self._generate_synthetic_arc_swing(n_frames=101, peak_frame=60)
        club_head[60, :] = np.nan

        with caplog.at_level(logging.WARNING):
            impact = detect_impact_frame(club_head, t)

        assert impact in (59, 61)
        assert not np.isnan(club_head[impact]).any()
        assert impact != 60
        # Verification that the dropout was reported via logger
        assert any(
            "dropout" in rec.message.lower() or "gap" in rec.message.lower()
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# 5. subject_parameters
# ---------------------------------------------------------------------------


class TestSubjectParameters:
    """Unit and contract tests for subject_parameters."""

    def test_precondition_height_out_of_range_raises(self) -> None:
        with pytest.raises((PreconditionError, ValueError)):
            subject_parameters(height_m=1.19, mass_kg=75.0)
        with pytest.raises((PreconditionError, ValueError)):
            subject_parameters(height_m=2.31, mass_kg=75.0)

    def test_precondition_mass_out_of_range_raises(self) -> None:
        with pytest.raises((PreconditionError, ValueError)):
            subject_parameters(height_m=1.80, mass_kg=34.9)
        with pytest.raises((PreconditionError, ValueError)):
            subject_parameters(height_m=1.80, mass_kg=200.1)

    def test_precondition_invalid_types_raises(self) -> None:
        with pytest.raises((PreconditionError, TypeError)):
            subject_parameters(height_m="1.80", mass_kg=75.0)  # type: ignore[arg-type]
        with pytest.raises((PreconditionError, TypeError)):
            subject_parameters(height_m=1.80, mass_kg="75.0")  # type: ignore[arg-type]

    def test_valid_parameters_default_neutral_id(self) -> None:
        params = subject_parameters(height_m=1.82, mass_kg=78.5)
        assert isinstance(params, dict)
        assert params["HEIGHT_M"] == pytest.approx(1.82)
        assert params["MASS_KG"] == pytest.approx(78.5)
        assert params["ID"] == "capture-O"

    def test_custom_neutral_id_argument(self) -> None:
        params = subject_parameters(height_m=1.75, mass_kg=70.0, subject_id="capture-A")
        assert params["ID"] == "capture-A"
