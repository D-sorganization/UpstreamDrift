"""Tests for the initial-state diff visualizer.

These never call MATLAB. We synthesize ``InitialStateDiffReport``
instances directly, plus a round-trip test that writes a fake MAT
file with ``scipy.io.savemat`` and reads it back through the loader.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

# Make the in-repo package importable.
_PKG_PARENT = Path(__file__).resolve().parents[3] / "src" / "shared" / "python"
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from src.shared.python.motion_matching.diagnostics.initial_state_diff import (  # noqa: E402
    InitialStateDiffReport,
    load_diff_report,
    plot_cartesian_delta_summary,
    plot_per_joint_delta_bars,
    plot_skeleton_overlay,
    summarize_for_pr_comment,
)


def _make_report(
    *,
    joint_delta_deg: np.ndarray | None = None,
    butt_delta_m: np.ndarray | None = None,
    head_delta_m: np.ndarray | None = None,
    significant: bool = True,
) -> InitialStateDiffReport:
    if joint_delta_deg is None:
        joint_delta_deg = np.array([0.1, 2.5, -3.7, 0.05, 1.2])
    n = joint_delta_deg.size
    spec_q = np.linspace(0, 1, n)
    actual_q = spec_q + np.deg2rad(joint_delta_deg)

    spec_butt = np.array([0.0, 0.0, 1.0])
    spec_head = np.array([1.0, 0.0, 0.5])
    if butt_delta_m is None:
        butt_delta_m = np.array([0.001, 0.0, 0.0])
    if head_delta_m is None:
        head_delta_m = np.array([0.006, -0.002, 0.001])

    return InitialStateDiffReport(
        specified={"q": spec_q, "r_butt": spec_butt, "r_clubhead": spec_head},
        actual={
            "q": actual_q,
            "r_butt": spec_butt + butt_delta_m,
            "r_clubhead": spec_head + head_delta_m,
        },
        delta={
            "q_per_joint_deg": joint_delta_deg,
            "q_max_deg": float(np.max(np.abs(joint_delta_deg))),
            "r_butt_mm": float(1000 * np.linalg.norm(butt_delta_m)),
            "r_clubhead_mm": float(1000 * np.linalg.norm(head_delta_m)),
            "is_significant": significant,
        },
        joint_names=[f"joint_{i}" for i in range(n)],
        input_file="/fake/3DModelInputs_Impact.mat",
        input_file_hash="abc123",
        model_name="GolfSwing3D_KineticallyDriven",
        timestamp="2026-05-05T12:00:00Z",
        thresholds={"joint_threshold_deg": 1.0, "pos_threshold_mm": 5.0},
    )


def test_load_diff_report_round_trip(tmp_path: Path) -> None:
    """Write a fake MAT in the layout produced by diagnose_initial_state.m
    and verify the loader rebuilds the schema."""
    pytest.importorskip("scipy")
    from scipy.io import savemat

    payload = {
        "specified": {
            "q": np.array([0.1, 0.2, 0.3]),
            "r_butt": np.array([0.0, 0.0, 1.0]),
            "r_clubhead": np.array([1.0, 0.0, 0.0]),
        },
        "actual": {
            "q": np.array([0.11, 0.21, 0.31]),
            "r_butt": np.array([0.001, 0.0, 1.0]),
            "r_clubhead": np.array([1.001, 0.0, 0.0]),
        },
        "delta": {
            "q_per_joint_deg": np.array([0.573, 0.573, 0.573]),
            "q_max_deg": 0.573,
            "r_butt_mm": 1.0,
            "r_clubhead_mm": 1.0,
            "is_significant": False,
        },
        "joint_names": np.array(["a", "b", "c"], dtype=object),
        "input_file": "/fake/path.mat",
        "input_file_hash": "deadbeef",
        "model_name": "TestModel",
        "timestamp": "2026-05-05T00:00:00Z",
        "thresholds": {"joint_threshold_deg": 1.0, "pos_threshold_mm": 5.0},
    }
    mat_path = tmp_path / "report.mat"
    savemat(str(mat_path), payload)

    report = load_diff_report(mat_path)
    assert isinstance(report, InitialStateDiffReport)
    assert report.specified["q"].shape == (3,)
    assert report.actual["q"].shape == (3,)
    assert report.joint_names == ["a", "b", "c"]
    assert report.input_file_hash == "deadbeef"
    assert report.is_significant is False
    assert report.delta["r_clubhead_mm"] == pytest.approx(1.0)


def test_skeleton_overlay_returns_figure_with_arrows() -> None:
    report = _make_report()
    fig = plot_skeleton_overlay(report)
    # Two skeletons (specified + actual) should each contribute a Line3D.
    assert fig is not None
    # Custom flag set by the renderer when arrows are drawn.
    assert getattr(fig, "_delta_arrow_count", 0) >= 1


def test_joint_delta_bars_sorted_by_magnitude() -> None:
    report = _make_report(joint_delta_deg=np.array([0.1, -5.0, 1.2, 3.0, -0.05]))
    fig = plot_per_joint_delta_bars(report)
    sorted_abs = getattr(fig, "_sorted_abs_deltas", None)
    assert sorted_abs is not None
    # Strictly non-increasing.
    diffs = np.diff(sorted_abs)
    assert np.all(diffs <= 1e-12), f"Expected sorted descending, got {sorted_abs}"


def test_summarize_for_pr_comment_flags_significant_deltas() -> None:
    report = _make_report(significant=True)
    md = summarize_for_pr_comment(report)
    assert "SIGNIFICANT" in md
    assert "max joint delta" in md
    assert "clubhead" in md
    assert "Top joint deltas" in md

    report_negligible = _make_report(
        joint_delta_deg=np.array([0.001, 0.002]),
        butt_delta_m=np.zeros(3),
        head_delta_m=np.zeros(3),
        significant=False,
    )
    md2 = summarize_for_pr_comment(report_negligible)
    assert "negligible" in md2
    assert "SIGNIFICANT" not in md2


def test_handles_missing_optional_fields() -> None:
    """Reports without Cartesian markers must still render plots and summary."""
    n = 3
    spec_q = np.array([0.1, 0.2, 0.3])
    actual_q = spec_q + np.deg2rad(np.array([0.5, 0.5, 0.5]))
    report = InitialStateDiffReport(
        specified={"q": spec_q},
        actual={"q": actual_q},
        delta={
            "q_per_joint_deg": np.array([0.5, 0.5, 0.5]),
            "q_max_deg": 0.5,
            "r_butt_mm": float("nan"),
            "r_clubhead_mm": float("nan"),
            "is_significant": False,
        },
        joint_names=[f"q{i}" for i in range(n)],
    )
    # All three plots must render without raising.
    plot_skeleton_overlay(report)
    plot_per_joint_delta_bars(report)
    plot_cartesian_delta_summary(report)
    md = summarize_for_pr_comment(report)
    assert "nan" in md.lower()


def test_input_validation_rejects_malformed_report() -> None:
    bad = InitialStateDiffReport(
        specified={"q": np.array([0.1, 0.2])},
        actual={"q": np.array([0.1, 0.2, 0.3])},  # mismatched length
        delta={
            "q_per_joint_deg": np.array([0, 0, 0]),
            "q_max_deg": 0.0,
            "r_butt_mm": 0.0,
            "r_clubhead_mm": 0.0,
            "is_significant": False,
        },
        joint_names=["a", "b"],
    )
    with pytest.raises(ValueError, match="joint counts differ"):
        plot_skeleton_overlay(bad)

    bad_names = InitialStateDiffReport(
        specified={"q": np.array([0.1, 0.2])},
        actual={"q": np.array([0.1, 0.2])},
        delta={
            "q_per_joint_deg": np.array([0.0, 0.0]),
            "q_max_deg": 0.0,
            "r_butt_mm": 0.0,
            "r_clubhead_mm": 0.0,
            "is_significant": False,
        },
        joint_names=["only_one"],
    )
    with pytest.raises(ValueError, match="joint_names length"):
        plot_per_joint_delta_bars(bad_names)
