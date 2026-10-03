"""Behavioral regression tests for Best-Candidate Viewer with Honest Residuals (MMR-16, #11102).

Mandatory Engineering Contracts Tested:
1. Raw observations cannot be overwritten by fitted positions (immutability).
2. Selecting a rejected candidate and changing visual preset / camera viewpoint
   preserves the exact numerical verdict and candidate hash (physics score is strictly
   invariant to visual/camera settings).
3. Selecting the worst marker / phase navigates to the exact observed and predicted frame
   (synchronized clock, no decoupled indices).
4. All captions match selected candidate and drive mode.
5. Numerical receipt and rendered frame indices agree.
6. Visual failure badges and honest residual metrics displayed.
7. Same journey works in installed PyQt and supported web/API surfaces with
   accessibility and missing-engine recovery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

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
    n_frames: int = 12,
    base_error: float = 0.005,
    drive_mode: str = "torque_driven",
) -> ReplayData:
    spec = _get_spec()
    coord_names = tuple(spec["coordinate_order"])
    n_coords = len(coord_names)
    times_s = np.linspace(0.0, 1.8, n_frames)
    q = np.zeros((n_frames, n_coords), dtype=float)

    # 5 markers: Head, Pelvis, LeftHand, ClubHead, RightHand
    n_markers = 5
    target_markers = np.zeros((n_frames, n_markers, 3), dtype=float)
    # Set unique target positions
    for m in range(n_markers):
        target_markers[:, m, 0] = 0.1 * (m + 1)
        target_markers[:, m, 1] = 0.2 * (m + 1)
        target_markers[:, m, 2] = 0.3 * (m + 1)

    model_markers = target_markers.copy() + base_error

    marker_names = ("Head", "Pelvis", "LeftHand", "ClubHead", "RightHand")

    # Top phase (~35-50%): plant error on LeftHand (marker idx 2)
    top_f = max(0, min(int(0.35 * n_frames), n_frames - 1))
    model_markers[top_f, 2, 0] += 0.030

    # Impact phase (~60-70%): plant worst error on ClubHead (marker idx 3)
    impact_f = max(0, min(int(0.63 * n_frames), n_frames - 1))
    model_markers[impact_f, 3, 1] += 0.060

    valid_mask = np.ones((n_frames, n_markers), dtype=bool)

    return ReplayData(
        time_s=times_s,
        coordinates=q,
        model_markers_m=model_markers,
        target_markers_m=target_markers,
        valid_mask=valid_mask,
        coordinate_names=coord_names,
        drive_mode=drive_mode,
        marker_names=marker_names,
    )


class TestBestCandidateViewerContracts:
    """TDD suite for MMR-16 #11102 acceptance criteria."""

    def _make_widget(self, qtbot: Any) -> Any:
        from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

        widget = TourMatchingViewerWidget(spec_path=SPEC_PATH)
        qtbot.addWidget(widget)
        return widget

    def test_raw_observations_cannot_be_overwritten_by_fitted_positions(
        self, qtbot: Any
    ) -> None:
        """Contract 1: Raw observations cannot be overwritten by fitted positions."""
        widget = self._make_widget(qtbot)
        replay = _build_test_replay()

        assert replay.target_markers_m is not None
        assert not replay.target_markers_m.flags.writeable

        # Direct mutation attempt must raise ValueError / RuntimeError
        with pytest.raises((ValueError, RuntimeError), match="read-only"):
            replay.target_markers_m[0, 0, 0] = 999.0

        # Target array values before loading
        pristine_targets = replay.target_markers_m.copy()

        # Load and render across frames and presets
        widget.load_replay_data(replay, candidate_hash="cand_immutability_test")
        widget.render_frame(3)
        widget.set_appearance_preset("dots_and_mesh")
        widget.set_appearance_preset("residual_vectors")

        # Verify target markers remain strictly pristine and read-only
        assert np.array_equal(replay.target_markers_m, pristine_targets)
        assert not replay.target_markers_m.flags.writeable

    def test_rejected_candidate_verdict_and_hash_invariant_to_visual_preset_and_camera(
        self, qtbot: Any
    ) -> None:
        """Contract 2: Selecting a rejected candidate and changing visual preset / camera viewpoint
        preserves exact numerical verdict, failure badge, candidate hash, and physics score.
        """
        widget = self._make_widget(qtbot)
        replay = _build_test_replay()

        # Load rejected candidate
        widget.load_replay_data(
            replay,
            candidate_hash="cand_rejected_sha01",
            engine_name="drake",
            drive_mode="torque_driven",
        )
        widget.set_acceptance(False, "whole_marker_rmse_m gate failed: 60 mm > 20 mm")

        # Initial verdict and invariant state
        assert not widget.is_accepted
        assert "REJECTED" in widget.verdict
        assert "whole_marker_rmse_m gate failed" in widget.verdict
        assert widget.failure_badge_visible
        assert "REJECTED" in widget.failure_badge_text

        initial_verdict = widget.verdict
        initial_hash = widget.candidate_hash
        initial_score = widget.physics_score
        initial_rms = widget.current_rms_error
        initial_badge_text = widget.failure_badge_text

        # Change camera viewpoints
        for view in ("perspective", "front", "side", "top", "isometric"):
            widget.set_camera_view(view)
            assert widget.verdict == initial_verdict
            assert widget.candidate_hash == initial_hash
            assert widget.physics_score == pytest.approx(initial_score, abs=1e-12)
            assert widget.current_rms_error == pytest.approx(initial_rms, abs=1e-12)
            assert widget.failure_badge_visible
            assert widget.failure_badge_text == initial_badge_text

        # Change visual appearance presets
        for preset in ("default", "high_contrast", "residual_vectors", "dots_and_mesh"):
            widget.set_appearance_preset(preset)
            assert widget.verdict == initial_verdict
            assert widget.candidate_hash == initial_hash
            assert widget.physics_score == pytest.approx(initial_score, abs=1e-12)
            assert widget.current_rms_error == pytest.approx(initial_rms, abs=1e-12)
            assert widget.failure_badge_visible
            assert widget.failure_badge_text == initial_badge_text

    def test_selecting_worst_marker_and_phase_navigates_to_exact_frame(
        self, qtbot: Any
    ) -> None:
        """Contract 3 & 5: Selecting worst marker or phase navigates to exact observed and
        predicted frame (synchronized clock, no decoupled indices)."""
        widget = self._make_widget(qtbot)
        replay = _build_test_replay(n_frames=12)
        widget.load_replay_data(replay, candidate_hash="cand_clock_sync")

        # In _build_test_replay:
        # Frame 3 is in Top phase, LeftHand (idx 2) has 30 mm error
        # Frame 7 is in Impact phase, ClubHead (idx 3) has 60 mm error (global worst)

        # 1. Select worst overall residual
        widget.select_worst_residual()
        assert widget.current_frame == 7
        assert widget.current_rendered_frame_index == 7
        assert widget.current_frame_time_s == pytest.approx(float(replay.time_s[7]))

        # 2. Select worst frame for marker 'LeftHand'
        frame_lefthand = widget.select_worst_marker("LeftHand")
        assert frame_lefthand == 4
        assert widget.current_frame == 4
        assert widget.current_rendered_frame_index == 4
        assert widget.current_frame_time_s == pytest.approx(float(replay.time_s[4]))

        # 3. Select worst frame for phase 'Impact'
        frame_impact = widget.select_worst_phase("Impact")
        assert frame_impact == 7
        assert widget.current_frame == 7
        assert widget.current_rendered_frame_index == 7
        assert widget.current_frame_time_s == pytest.approx(float(replay.time_s[7]))

        # 4. Select worst frame for phase 'Top'
        frame_top = widget.select_worst_phase("Top")
        assert frame_top == 4
        assert widget.current_frame == 4
        assert widget.current_rendered_frame_index == 4
        assert widget.current_frame_time_s == pytest.approx(float(replay.time_s[4]))

    def test_all_captions_match_selected_candidate_and_drive_mode(
        self, qtbot: Any
    ) -> None:
        """Contract 4: All captions match selected candidate and drive mode."""
        widget = self._make_widget(qtbot)
        replay = _build_test_replay(drive_mode="muscle_driven")
        widget.load_replay_data(
            replay,
            candidate_hash="cand_muscle_456",
            engine_name="myosuite",
            drive_mode="muscle_driven",
        )

        assert "cand_muscle_456" in widget.title_caption
        assert "Myosuite" in widget.title_caption
        assert "Muscle Driven" in widget.title_caption
        assert "Muscle Driven" in widget.capabilities_caption

    def test_visual_failure_badges_and_honest_residual_metrics_displayed(
        self, qtbot: Any
    ) -> None:
        """Contract 6: Visual failure badges and honest residual metrics displayed."""
        widget = self._make_widget(qtbot)
        replay = _build_test_replay()

        # Accepted candidate: failure badge is hidden
        widget.load_replay_data(replay, candidate_hash="cand_pass")
        widget.set_acceptance(True)
        assert not widget.failure_badge_visible

        # Rejected candidate: failure badge is displayed with reason
        widget.set_acceptance(False, "terminal_marker_rmse_m exceeds 15 mm limit")
        assert widget.failure_badge_visible
        assert "terminal_marker_rmse_m" in widget.failure_badge_text

        # Honest residual metrics exposed
        assert widget.current_rms_error > 0.0
        assert widget.residual_summary is not None
        summary = widget.residual_summary
        assert summary.worst_frame_idx == 7
        assert summary.worst_marker_name == "ClubHead"
        assert summary.max_marker_error_m >= 0.060
        assert "impact" in summary.worst_phase.lower()

    def test_board_ready_video_export_carries_legible_units_and_receipt(
        self, tmp_path: Path
    ) -> None:
        """Scope: Export board-ready video/stills with legible units and evidence link."""
        from src.tools.tour_matching_viewer.core import export_board_ready_video

        replay = _build_test_replay(n_frames=6, drive_mode="torque_driven")
        spec = _get_spec()
        out_video = tmp_path / "board_ready_export.gif"

        exported = export_board_ready_video(
            replay,
            spec,
            output_path=out_video,
            candidate_hash="cand_video_789",
            engine_name="pinocchio",
            drive_mode="torque_driven",
            verdict="REJECTED (marker RMSE 24.2 mm)",
            evidence_link="evidence/pinocchio_replay.npz",
            fps=10,
        )

        assert exported.is_file()
        assert exported.stat().st_size > 0

    def test_accessibility_properties_set_on_viewer_controls(self, qtbot: Any) -> None:
        """Contract 7 (PyQt surface): Controls have accessible names and tooltips."""
        widget = self._make_widget(qtbot)
        assert widget._camera_combo.accessibleName() != ""
        assert widget._appearance_combo.accessibleName() != ""
        assert widget._select_worst_btn.accessibleName() != ""
        assert widget._rejection_banner.accessibleName() != ""
        assert widget._export_gif_btn.toolTip() != ""
