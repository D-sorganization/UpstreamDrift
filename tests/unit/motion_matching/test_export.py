"""Unit tests for video and fit-quality report export (MS-86, #10359).

Validates:
- export_video creates GIF and MP4 animations with expected frame counts.
- export_video fails closed when candidate or marker trajectories are missing.
- export_report produces Markdown containing standardized metrics, gates, and full provenance.
- export_report produces valid PDF files when requested.
- export_report faithfully preserves ledger/receipt verdicts without cosmetic masking.
- CLI subcommands dispatch export-video and export-report accurately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch
import sys

import numpy as np
import pytest

from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_io import save_candidate
from src.shared.python.motion_matching.export import (
    export_report,
    export_video,
)

pytestmark = [pytest.mark.unit]


@pytest.fixture
def mock_candidate() -> MatchedSwingCandidate:
    """Create a minimal valid MatchedSwingCandidate with marker trajectories."""
    n_frames = 10
    n_markers = 4
    time_s = np.linspace(0.0, 0.18, n_frames)
    q = np.zeros((n_frames, 3))
    v = np.zeros((n_frames, 3))
    tau = np.zeros((n_frames, 3))

    model_markers = np.zeros((n_frames, n_markers, 3))
    target_markers = np.zeros((n_frames, n_markers, 3))
    for i in range(n_frames):
        model_markers[i, :, 0] = i * 0.01
        target_markers[i, :, 0] = i * 0.012

    validity = np.ones((n_frames, n_markers), dtype=bool)

    meta = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.DYNAMIC,
        engine="mujoco",
        model_name="test_humanoid",
        coordinate_names=("q0", "q1", "q2"),
        velocity_names=("v0", "v1", "v2"),
        actuator_names=("tau0", "tau1", "tau2"),
        marker_names=("m0", "m1", "m2", "m3"),
    )
    markers = CandidateMarkers(
        model_markers_m=model_markers,
        target_markers_m=target_markers,
        marker_validity=validity,
    )
    return MatchedSwingCandidate(
        metadata=meta,
        time_s=time_s,
        q=q,
        v=v,
        tau=tau,
        markers=markers,
    )


@pytest.fixture
def candidate_path(tmp_path: Path, mock_candidate: MatchedSwingCandidate) -> Path:
    """Save candidate package to temporary NPZ archive."""
    npz_p = tmp_path / "test_candidate.npz"
    save_candidate(mock_candidate, npz_p)
    return npz_p


@pytest.fixture
def mock_receipt() -> dict[str, Any]:
    """Create a representative execution receipt matching GS / full-body schema."""
    return {
        "backend": "mujoco",
        "engine": "mujoco",
        "candidate_sha256": "3f94aa92f28acde61b6b2f8c871ad536f93eef62cbbae5d0ffe7c85e4b41a6ad",
        "capture": "driver",
        "capture_sha256": "545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba",
        "lane": "ground_support",
        "elapsed_s": 12.34,
        "qualification": "kinematic IK and computed-torque tracking milestone; unqualified until Simscape parity; not a fit, not acceptance",
        "acceptance": {
            "verdict": "REJECTED",
            "reason": "Terminal marker RMSE exceeds strict threshold",
            "metrics": {
                "whole_marker_rmse_m": 0.0419,
                "early_marker_rmse_m": 0.0210,
                "terminal_marker_rmse_m": 0.0895,
                "club_marker_rmse_m": 0.0382,
                "pelvis_yaw_rmse_rad": 0.0523,
            },
            "gates": [
                {
                    "name": "whole_marker_rmse",
                    "status": "PASS",
                    "threshold": 0.05,
                    "measured": 0.0419,
                    "unit": "m",
                },
                {
                    "name": "terminal_marker_rmse",
                    "status": "FAIL",
                    "threshold": 0.05,
                    "measured": 0.0895,
                    "unit": "m",
                },
                {
                    "name": "lowest_sphere_height",
                    "status": "PASS",
                    "threshold": -0.02,
                    "measured": -0.0114,
                    "unit": "m",
                },
            ],
        },
        "physical_constraints": {
            "lowest_sphere_height_min_m": -0.0114,
            "lowest_sphere_height_max_m": -0.0003,
            "inside_support_polygon_fraction": 0.958,
            "peak_joint_torque_n_m": 1448.7,
        },
    }


@pytest.fixture
def receipt_path(tmp_path: Path, mock_receipt: dict[str, Any]) -> Path:
    """Save mock receipt to temporary JSON file."""
    p = tmp_path / "test_receipt.json"
    p.write_text(json.dumps(mock_receipt, indent=2), encoding="utf-8")
    return p


class TestExportVideo:
    def test_export_video_gif_from_candidate_obj(
        self, tmp_path: Path, mock_candidate: MatchedSwingCandidate
    ) -> None:
        imageio = pytest.importorskip("imageio.v2")

        out_gif = tmp_path / "output.gif"
        res = export_video(mock_candidate, "mujoco", out_gif, stride=2, fps=10)
        assert res.is_file()
        assert res.stat().st_size > 0

        # Verify frame count matches ceil(10 / 2) = 5 frames
        frames = imageio.mimread(str(res))
        assert len(frames) == 5

    def test_export_video_gif_from_path(
        self, tmp_path: Path, candidate_path: Path
    ) -> None:
        imageio = pytest.importorskip("imageio.v2")

        out_gif = tmp_path / "output_from_path.gif"
        res = export_video(candidate_path, "mujoco", out_gif, stride=2)
        assert res.is_file()
        frames = imageio.mimread(str(res))
        assert len(frames) == 5

    def test_export_video_mp4(
        self, tmp_path: Path, mock_candidate: MatchedSwingCandidate
    ) -> None:
        import cv2

        out_mp4 = tmp_path / "output.mp4"
        res = export_video(mock_candidate, "drake", out_mp4, stride=2, fps=10)
        assert res.is_file()
        assert res.stat().st_size > 0

        cap = cv2.VideoCapture(str(res))
        assert cap.isOpened()
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert count == 5

    def test_export_video_fails_closed_on_missing_markers(self, tmp_path: Path) -> None:
        meta = CandidateMetadata(
            schema_version=CANDIDATE_SCHEMA_VERSION,
            profile=CandidateProfile.DYNAMIC,
            engine="mujoco",
            coordinate_names=("q0",),
            velocity_names=("v0",),
            actuator_names=("tau0",),
        )
        cand_no_markers = MatchedSwingCandidate(
            metadata=meta,
            time_s=np.array([0.0, 0.1]),
            q=np.zeros((2, 1)),
            v=np.zeros((2, 1)),
            tau=np.zeros((2, 1)),
            markers=None,
        )
        out_gif = tmp_path / "should_fail.gif"
        with pytest.raises(ValueError, match="marker trajectory"):
            export_video(cand_no_markers, "mujoco", out_gif)

    def test_export_video_unsupported_format(
        self, tmp_path: Path, mock_candidate: MatchedSwingCandidate
    ) -> None:
        out_avi = tmp_path / "output.unsupported"
        with pytest.raises((ValueError, AssertionError)):
            export_video(mock_candidate, "mujoco", out_avi)


class TestExportReport:
    def test_export_report_markdown_from_dict(
        self, tmp_path: Path, mock_receipt: dict[str, Any]
    ) -> None:
        out_md = tmp_path / "fit_report.md"
        res = export_report(mock_receipt, out_md)
        assert res.is_file()
        content = res.read_text(encoding="utf-8")

        # Provenance checks (#8820 / U3)
        assert "Cryptographic Provenance" in content
        assert "mujoco" in content.lower()
        assert mock_receipt["candidate_sha256"] in content
        assert "REJECTED" in content
        assert "Timestamp" in content or "UTC" in content

        # Metrics checks
        assert "Whole Marker RMSE" in content
        assert "Early Marker RMSE" in content
        assert "Terminal Marker RMSE" in content
        assert "Club Marker RMSE" in content
        assert "Pelvis Yaw RMSE" in content
        assert "0.0419" in content or "41.9" in content

        # Gates checks
        assert "whole_marker_rmse" in content
        assert "terminal_marker_rmse" in content
        assert "FAIL" in content

        # Qualification
        assert "unqualified until Simscape parity" in content

    def test_export_report_markdown_from_file(
        self, tmp_path: Path, receipt_path: Path
    ) -> None:
        out_md = tmp_path / "fit_report_file.md"
        res = export_report(receipt_path, out_md)
        assert res.is_file()
        content = res.read_text(encoding="utf-8")
        assert "mujoco" in content.lower()
        assert "REJECTED" in content

    def test_export_report_pdf(
        self, tmp_path: Path, mock_receipt: dict[str, Any]
    ) -> None:
        out_pdf = tmp_path / "fit_report.pdf"
        res = export_report(mock_receipt, out_pdf)
        assert res.is_file()
        assert res.stat().st_size > 0
        header = res.read_bytes()[:5]
        assert header == b"%PDF-"

    def test_export_report_preserves_ledger_verdict(
        self, tmp_path: Path, mock_receipt: dict[str, Any]
    ) -> None:
        out_md = tmp_path / "verdict_test.md"
        res = export_report(mock_receipt, out_md)
        content = res.read_text(encoding="utf-8")
        assert "REJECTED" in content
        lines = [line for line in content.splitlines() if "verdict" in line.lower()]
        assert any("rejected" in line_entry.lower() for line_entry in lines)
        assert not any("verdict: pass" in line_entry.lower() for line_entry in lines)


class TestExportCLI:
    def test_cli_export_report(self, tmp_path: Path, receipt_path: Path) -> None:
        from src.shared.python.motion_matching.__main__ import main

        out_md = tmp_path / "cli_report.md"
        test_args = [
            "python",
            "export-report",
            "--receipt",
            str(receipt_path),
            "--out",
            str(out_md),
        ]
        with patch.object(sys, "argv", test_args):
            ret = main()
            assert ret == 0
        assert out_md.is_file()
        content = out_md.read_text(encoding="utf-8")
        assert "REJECTED" in content

    def test_cli_export_video(self, tmp_path: Path, candidate_path: Path) -> None:
        pytest.importorskip("imageio.v2")
        from src.shared.python.motion_matching.__main__ import main

        out_gif = tmp_path / "cli_video.gif"
        test_args = [
            "python",
            "export-video",
            "--candidate",
            str(candidate_path),
            "--engine",
            "mujoco",
            "--out",
            str(out_gif),
            "--stride",
            "2",
        ]
        with patch.object(sys, "argv", test_args):
            ret = main()
            assert ret == 0
        assert out_gif.is_file()
        assert out_gif.stat().st_size > 0
