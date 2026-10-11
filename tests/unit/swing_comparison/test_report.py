"""Unit tests for swing comparison reporting (Issue #11164)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.shared.python.swing_comparison.metrics import (
    compare,
)
from src.shared.python.swing_comparison.motion import (
    SwingMotion,
)
from src.shared.python.swing_comparison.report import (
    comparison_to_dict,
    comparison_to_markdown,
)


def _make_test_motion() -> SwingMotion:
    n = 80
    t = np.arange(n, dtype=np.float64) * 0.01
    head = np.zeros((n, 3))
    grip = np.zeros((n, 3))
    theta = np.linspace(-np.pi * 0.7, np.pi * 0.4, n)
    head[:, 0] = np.sin(theta)
    head[:, 2] = -np.cos(theta) + 0.1
    grip[:, 0] = 0.5 * head[:, 0]
    grip[:, 2] = 0.5 * head[:, 2] + 0.5

    markers = {
        "WaistLeft": np.column_stack(
            [0.15 * np.cos(theta), 0.15 * np.sin(theta), np.full(n, 0.9)]
        ),
        "WaistRight": np.column_stack(
            [-0.15 * np.cos(theta), -0.15 * np.sin(theta), np.full(n, 0.9)]
        ),
        "LShoulderBack": np.column_stack(
            [0.2 * np.cos(theta), 0.2 * np.sin(theta), np.full(n, 1.4)]
        ),
        "RShoulderBack": np.column_stack(
            [-0.2 * np.cos(theta), -0.2 * np.sin(theta), np.full(n, 1.4)]
        ),
        "LShoulderTop": np.column_stack(
            [0.2 * np.cos(theta), 0.2 * np.sin(theta), np.full(n, 1.4)]
        ),
        "LElbowOut": np.zeros((n, 3)),
        "LWristTop": grip.copy(),
        "Marker_2:2:1": head.copy(),
        "Marker_3:3:1": grip.copy(),
    }
    return SwingMotion(t=t, markers=markers, club_head=head, grip=grip)


@pytest.mark.unit
class TestComparisonReport:
    """Test suite for Markdown and JSON rendering of ComparisonReport."""

    def test_markdown_rendering_title_case(self) -> None:
        """Verify report renders to Markdown with Title Case headings."""
        motion_a = _make_test_motion()
        motion_b = _make_test_motion()
        report = compare(motion_a, motion_b)

        md = comparison_to_markdown(report)
        assert isinstance(md, str)
        assert len(md) > 100

        # Check Title Case headings
        assert "# Swing Comparison Report" in md
        assert "## Key Swing Events and Tempo" in md
        assert "## Segment Rotations and X-Factor" in md
        assert "## Kinematic Sequence" in md
        assert "## Arm and Wrist Mechanics" in md
        assert "## Club Delivery and Hand Path" in md
        assert "## Trajectory Alignment and Marker RMS" in md

    def test_json_dict_serialization(self) -> None:
        """Verify report converts to a JSON-serializable dictionary."""
        motion_a = _make_test_motion()
        motion_b = _make_test_motion()
        report = compare(motion_a, motion_b)

        d = comparison_to_dict(report)
        assert isinstance(d, dict)

        # Must serialize cleanly with stdlib json
        json_str = json.dumps(d, indent=2)
        assert isinstance(json_str, str)
        loaded = json.loads(json_str)
        assert loaded["mean_marker_rms"] == 0.0
        assert "metrics_a" in loaded
        assert "metrics_b" in loaded
        assert "differences" in loaded

    def test_report_file_writing_to_tmp_path(self, tmp_path: Path) -> None:
        """Verify report files are written only to tmp_path."""
        motion_a = _make_test_motion()
        motion_b = _make_test_motion()
        report = compare(motion_a, motion_b)

        md_file = tmp_path / "comparison_report.md"
        json_file = tmp_path / "comparison_report.json"

        md_file.write_text(comparison_to_markdown(report), encoding="utf-8")
        json_file.write_text(json.dumps(comparison_to_dict(report)), encoding="utf-8")

        assert md_file.exists()
        assert json_file.exists()
        assert len(md_file.read_text(encoding="utf-8")) > 0
        assert len(json_file.read_text(encoding="utf-8")) > 0
