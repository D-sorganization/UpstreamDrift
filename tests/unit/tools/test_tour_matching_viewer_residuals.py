"""Unit tests for best-candidate viewer with honest residuals (MMR-16, #11102).

Validates:
1. Raw observations cannot be overwritten by fitted positions (immutability).
2. Camera and visual appearance preset changes do not change physics score.
3. Worst marker / phase can be selected from the residual plot and jumps to the exact frame.
4. All captions match selected candidate and drive mode.
5. Numerical receipt and rendered frame indices agree.
6. Missing-engine recovery in viewer without unhandled exceptions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.engine_core.engine_availability import skip_if_unavailable
from src.tools.tour_matching_viewer.core import (
    ReplayData,
    ResidualSummary,
    compute_residual_summary,
    export_board_ready_still,
)

pytestmark = [
    skip_if_unavailable("pyqt6"),
    pytest.mark.unit,
]

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


def _get_spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _build_test_replay(
    *,
    n_frames: int = 10,
    planted_worst_frame: int = 4,
    planted_worst_marker: int = 2,
    base_error: float = 0.005,
    worst_error: float = 0.045,
    drive_mode: str = "torque_driven",
) -> ReplayData:
    spec = _get_spec()
    coord_names = tuple(spec["coordinate_order"])
    n_coords = len(coord_names)
    times_s = np.linspace(0.0, 1.8, n_frames)
    q = np.zeros((n_frames, n_coords), dtype=float)

    n_markers = 5
    target_markers = np.zeros((n_frames, n_markers, 3), dtype=float)
    model_markers = np.zeros((n_frames, n_markers, 3), dtype=float)
    model_markers += base_error  # base 5 mm error

    # Plant worst error on specified frame and marker
    model_markers[planted_worst_frame, planted_worst_marker, 0] += worst_error

    valid_mask = np.ones((n_frames, n_markers), dtype=bool)

    return ReplayData(
        time_s=times_s,
        coordinates=q,
        model_markers_m=model_markers,
        target_markers_m=target_markers,
        valid_mask=valid_mask,
        coordinate_names=coord_names,
        drive_mode=drive_mode,
    )


class TestReplayDataObservationImmutability:
    """Acceptance criterion 1: raw observations cannot be overwritten by fitted positions."""

    def test_raw_observations_cannot_be_overwritten(self) -> None:
        replay = _build_test_replay()
        assert replay.target_markers_m is not None
        assert not replay.target_markers_m.flags.writeable

        with pytest.raises((ValueError, RuntimeError), match="read-only"):
            replay.target_markers_m[0, 0, 0] = 999.0

    def test_fitted_positions_do_not_mutate_target_markers(self) -> None:
        replay = _build_test_replay()
        assert replay.target_markers_m is not None
        assert replay.model_markers_m is not None

        # Verify initial target marker values
        target_copy = replay.target_markers_m.copy()

        # Compute residuals or access model markers
        summary = compute_residual_summary(replay)
        assert summary.worst_frame_idx == 4

        # Target markers array must remain identical
        assert np.array_equal(replay.target_markers_m, target_copy)


class TestResidualSummaryAndWorstSelection:
    """Acceptance criterion 3: worst marker/phase can be selected from residual plot."""

    def test_compute_residual_summary_identifies_worst_frame_and_marker(self) -> None:
        replay = _build_test_replay(
            n_frames=10,
            planted_worst_frame=6,
            planted_worst_marker=3,
            worst_error=0.050,
        )
        marker_names = ("Head", "Pelvis", "LeftHand", "ClubHead", "RightHand")
        summary = compute_residual_summary(replay, marker_names=marker_names)

        assert isinstance(summary, ResidualSummary)
        assert summary.worst_frame_idx == 6
        assert summary.worst_marker_idx == 3
        assert summary.worst_marker_name == "ClubHead"
        assert summary.max_marker_error_m >= 0.050
        assert len(summary.frame_residuals) == 10
        # Frame 6 in a 10-frame swing (index 6/9 ~ 67%) falls in Impact phase
        assert "impact" in summary.worst_phase.lower()

    def test_select_worst_residual_jumps_to_exact_frame(self, qtbot) -> None:
        from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

        widget = TourMatchingViewerWidget(spec_path=SPEC_PATH)
        qtbot.addWidget(widget)

        replay = _build_test_replay(n_frames=10, planted_worst_frame=4)
        widget.load_replay_data(replay, candidate_hash="cand_test_123")

        # Initially at frame 0
        assert widget.current_frame == 0

        # Select worst residual
        widget.select_worst_residual()
        assert widget.current_frame == 4
        # Numerical receipt and rendered frame indices agree (Criterion 5)
        assert widget.current_rendered_frame_index == 4
        assert widget.current_frame_time_s == pytest.approx(float(replay.time_s[4]))


class TestCameraAndAppearanceIndependence:
    """Acceptance criterion 4: camera/appearance changes do not change physics score."""

    def test_camera_and_appearance_changes_preserve_physics_score(self, qtbot) -> None:
        from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

        widget = TourMatchingViewerWidget(spec_path=SPEC_PATH)
        qtbot.addWidget(widget)

        replay = _build_test_replay(n_frames=10, base_error=0.008)
        widget.load_replay_data(replay, candidate_hash="cand_physics_guard")

        initial_rms = widget.current_rms_error
        initial_score = widget.physics_score

        # 1. Change camera views
        for view_name in ("front", "side", "top", "isometric"):
            widget.set_camera_view(view_name)
            assert widget.current_rms_error == pytest.approx(initial_rms, abs=1e-12)
            assert widget.physics_score == pytest.approx(initial_score, abs=1e-12)

        # 2. Change appearance presets
        for preset in ("default", "high_contrast", "residual_vectors", "dots_and_mesh"):
            widget.set_appearance_preset(preset)
            assert widget.current_rms_error == pytest.approx(initial_rms, abs=1e-12)
            assert widget.physics_score == pytest.approx(initial_score, abs=1e-12)

        # Candidate hash and verdict remain identical
        assert widget.candidate_hash == "cand_physics_guard"


class TestCaptionsAndDriveMode:
    """Acceptance criterion 2: all captions match selected candidate and drive mode."""

    def test_captions_display_candidate_and_drive_mode(self, qtbot) -> None:
        from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

        widget = TourMatchingViewerWidget(spec_path=SPEC_PATH)
        qtbot.addWidget(widget)

        replay = _build_test_replay(drive_mode="torque_driven")
        widget.load_replay_data(
            replay,
            candidate_hash="cand_hash_987",
            engine_name="drake",
            drive_mode="torque_driven",
        )

        title = widget.title_caption
        assert "cand_hash_987" in title
        assert "Drake" in title
        assert "Torque Driven" in title

        sub_caption = widget.capabilities_caption
        assert "Torque Driven" in sub_caption

    def test_export_board_ready_still_carries_complete_captions(
        self, tmp_path: Path
    ) -> None:
        replay = _build_test_replay(drive_mode="torque_driven")
        spec = _get_spec()
        out_path = tmp_path / "board_ready_still.png"

        exported = export_board_ready_still(
            replay,
            spec,
            frame_idx=4,
            output_path=out_path,
            candidate_hash="cand_board_123",
            engine_name="drake",
            drive_mode="torque_driven",
            verdict="REJECTED (marker error 45 mm)",
        )

        assert exported.is_file()
        assert exported.stat().st_size > 0


class TestMissingEngineRecovery:
    """Acceptance criterion 6: graceful missing-engine recovery without crashing."""

    def test_missing_engine_shows_recovery_dialog(self, qtbot, monkeypatch) -> None:
        from src.shared.python.motion_matching.native_viewers import (
            ViewerUnavailableError,
        )
        from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

        widget = TourMatchingViewerWidget(spec_path=SPEC_PATH)
        qtbot.addWidget(widget)

        replay = _build_test_replay()
        widget.load_replay_data(replay)

        # Mock open_in_native_viewer to raise ViewerUnavailableError
        def mock_launch(*args, **kwargs):
            raise ViewerUnavailableError("Simscape runtime not found on host")

        monkeypatch.setattr(
            "src.shared.python.motion_matching.native_viewers.open_in_native_viewer",
            mock_launch,
        )

        # Simulating launcher call: must handle error gracefully
        with patch.object(widget, "_show_recovery_message") as mock_msg:
            widget.launch_native_backend("simscape")
            mock_msg.assert_called_once()
            assert "Simscape" in mock_msg.call_args[0][0]


class TestWholeMarkerRmsPerMarkerDistance:
    """Codex P1 @ core.py: frame RMS must pool XYZ per marker (3D distance),
    matching the canonical tour_metrics.compute_shared_metrics receipt metric."""

    @staticmethod
    def _frame_error_replay(frame_offsets: np.ndarray) -> ReplayData:
        """Replay with valid mask all-True and per-marker single-axis offsets."""
        spec = _get_spec()
        coord_names = tuple(spec["coordinate_order"])
        n_frames, n_markers = frame_offsets.shape[0], frame_offsets.shape[1]
        time_s = np.linspace(0.0, 0.1, n_frames)
        model_markers = frame_offsets.copy()
        target_markers = np.zeros_like(frame_offsets)
        return ReplayData(
            time_s=time_s,
            coordinates=np.zeros((n_frames, len(coord_names))),
            model_markers_m=model_markers,
            target_markers_m=target_markers,
            valid_mask=np.ones((n_frames, n_markers), dtype=bool),
            coordinate_names=coord_names,
            drive_mode="torque_driven",
        )

    def test_single_marker_single_axis_offset_reports_full_3d_distance(
        self,
    ) -> None:
        # 20 mm X-only offset on the single observed marker: canonical receipt
        # RMSE is 20 mm, not sqrt(mean of the flattened components) = 11.55 mm.
        offsets = np.zeros((3, 1, 3))
        offsets[:, 0, 0] = 0.020
        replay = self._frame_error_replay(offsets)

        summary = compute_residual_summary(replay)
        assert all(
            fr.rms_error_m == pytest.approx(0.020) for fr in summary.frame_residuals
        )
        assert summary.mean_rms_m == pytest.approx(0.020)

    def test_frame_rms_matches_canonical_shared_metrics(self) -> None:
        from src.shared.python.motion_matching.tour_capture_contract import (
            TourCapture,
        )
        from src.shared.python.motion_matching.tour_metrics import (
            compute_shared_metrics,
        )

        offsets = np.zeros((2, 4, 3))
        offsets[0, 0, 0] = 0.020
        offsets[0, 1, 1] = 0.005
        offsets[0, 1, 2] = -0.002
        offsets[1, 3, 2] = 0.011
        offset_all = offsets[0].copy()  # single-frame subset for the canonical call
        replay = self._frame_error_replay(offsets)

        labels = tuple(f"M{i}" for i in range(4))
        capture = TourCapture(
            time_s=np.array([0.0]),
            labels=labels,
            points_m=np.zeros((1, 4, 3)),
            valid=np.ones((1, 4), dtype=bool),
        )
        canonical = compute_shared_metrics(
            capture, np.zeros((1, 4, 3)) + offset_all[None]
        )
        summary = compute_residual_summary(replay)

        assert summary.frame_residuals[0].rms_error_m == pytest.approx(
            canonical.whole_marker_rmse_m
        )

    def test_worst_frame_rms_is_per_marker_pooled_distance(self) -> None:
        offsets = np.zeros((4, 3, 3))
        offsets[:, 1, 1] = 0.004  # every frame: 4 mm Y-only offset on marker 1
        replay = self._frame_error_replay(offsets)

        summary = compute_residual_summary(replay)
        # Pooled per-marker 3D distance: one marker carries 4 mm, others 0.
        assert summary.worst_frame_rms_m == pytest.approx(0.004 / np.sqrt(3))


class TestUnobservedFramesExcluded:
    """Codex P2 @ core.py: frames with no valid markers are never selected as
    the global worst and are excluded from mean calculations."""

    @staticmethod
    def _replay_with_unobserved_placeholder_frame() -> ReplayData:
        spec = _get_spec()
        coord_names = tuple(spec["coordinate_order"])
        n_frames, n_markers = 6, 5
        time_s = np.linspace(0.0, 0.3, n_frames)
        target_markers = np.zeros((n_frames, n_markers, 3), dtype=float)
        model_markers = np.zeros_like(target_markers) + 0.005  # 5 mm base error
        # Plant a genuine observed worst at frame 1 (X offset)
        model_markers[1, 2, 0] += 0.050
        # Frame 3: no valid markers, finite huge placeholder coordinates
        valid_mask = np.ones((n_frames, n_markers), dtype=bool)
        valid_mask[3, :] = False
        model_markers[3, :, :] = 10.0
        return ReplayData(
            time_s=time_s,
            coordinates=np.zeros((n_frames, len(coord_names))),
            model_markers_m=model_markers,
            target_markers_m=target_markers,
            valid_mask=valid_mask,
            coordinate_names=coord_names,
            drive_mode="torque_driven",
        )

    def test_all_invalid_frame_never_becomes_global_worst(self) -> None:
        replay = self._replay_with_unobserved_placeholder_frame()
        summary = compute_residual_summary(replay)

        assert summary.worst_frame_idx == 1
        # 5 mm base error on every axis plus the 50 mm planted X offset
        expected_max = float(np.sqrt((0.005 + 0.050) ** 2 + 0.005**2 + 0.005**2))
        assert summary.max_marker_error_m == pytest.approx(expected_max)
        # The unobserved frame carries no worst-marker claim at all
        assert summary.frame_residuals[3].worst_marker is None

    def test_mean_rms_excludes_unobserved_frames(self) -> None:
        replay = self._replay_with_unobserved_placeholder_frame()
        summary = compute_residual_summary(replay)

        observed = [
            fr.rms_error_m
            for fr in summary.frame_residuals
            if fr.worst_marker is not None
        ]
        assert len(observed) == 5  # unobserved frame 3 is out of the mean
        assert summary.mean_rms_m == pytest.approx(float(np.mean(observed)))


class TestSelectedReceiptProvenanceInViewer:
    """Codex P1 @ gui.py: load_file must honour receipt-derived provenance and
    the rejection verdict instead of filename placeholders."""

    def _widget(self, qtbot: Any) -> Any:
        from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

        widget = TourMatchingViewerWidget(spec_path=SPEC_PATH)
        qtbot.addWidget(widget)
        return widget

    def test_load_file_accepts_receipt_provenance_overrides(
        self, qtbot: Any, monkeypatch: Any, tmp_path: Path
    ) -> None:
        from src.tools.tour_matching_viewer import gui as viewer_gui

        widget = self._widget(qtbot)
        replay = _build_test_replay(drive_mode="torque_driven")
        monkeypatch.setattr(viewer_gui, "load_replay", lambda *a, **k: replay)

        widget.load_file(
            tmp_path / "wrong_name.npz",
            candidate_hash="rowsha000111",
            engine_name="drake",
            drive_mode="torque_driven",
            is_accepted=False,
            rejection_reason="whole_marker_rmse_m gate failed",
        )

        assert widget.candidate_hash == "rowsha000111"
        assert widget.drive_mode == "torque_driven"
        assert "drake" in widget.title_caption.lower()
        assert not widget._rejection_banner.isHidden()
        assert "whole_marker_rmse_m" in widget._rejection_banner.text()

    def test_load_file_without_provenance_keeps_legacy_fallback(
        self, qtbot: Any, monkeypatch: Any, tmp_path: Path
    ) -> None:
        from src.tools.tour_matching_viewer import gui as viewer_gui

        widget = self._widget(qtbot)
        replay = _build_test_replay(drive_mode="kinematic_prescribed")
        monkeypatch.setattr(viewer_gui, "load_replay", lambda *a, **k: replay)

        widget.load_file(tmp_path / "returned81_demo.npz")
        assert "returned81" in widget.title_caption
        assert widget._rejection_banner.isHidden()
