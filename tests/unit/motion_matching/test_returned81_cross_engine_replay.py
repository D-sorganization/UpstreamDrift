"""Tests for Returned81 cross-engine replay format and 5-metric evaluation.

Step 4 of Visuals Handoff (VISUALS_HANDOFF.md):
- NPZ reader/writer roundtrip with exact key and shape validation.
- 5-metric function against the archived Pinocchio numbers:
  whole 26.366 mm, early 11.427 mm, terminal 46.305 mm, clubhead 15.955 mm, pelvis yaw error 13.92%.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.replay_metrics import (
    ReplayFiveMetrics,
    compute_replay_five_metrics,
    load_native_replay_npz,
    save_native_replay_npz,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
REPLAYS_DIR = ROOT / "docs/development/full_body_models/evidence/replays"
PINOCCHIO_NPZ = REPLAYS_DIR / "pinocchio_returned81_replay.npz"
CANDIDATE_JSON = REPLAYS_DIR / "returned-candidate.json"


def test_npz_roundtrip_validation(tmp_path: Path) -> None:
    """Test NPZ serialization roundtrip and shape/key validation."""
    n_frames = 15
    n_states = 54
    n_markers = 25

    time_s = np.linspace(0.0, 0.85, n_frames)
    native_state = np.ones((n_frames, n_states), dtype=float)
    markers_m = np.zeros((n_frames, n_markers, 3), dtype=float)
    target_m = np.zeros((n_frames, n_markers, 3), dtype=float)
    valid = np.ones((n_frames, n_markers), dtype=bool)

    npz_path = tmp_path / "roundtrip.npz"
    save_native_replay_npz(
        npz_path,
        time_s=time_s,
        native_state=native_state,
        markers_m=markers_m,
        target_m=target_m,
        valid=valid,
    )

    loaded = load_native_replay_npz(npz_path)
    np.testing.assert_allclose(loaded["time_s"], time_s)
    np.testing.assert_allclose(loaded["native_state"], native_state)
    np.testing.assert_allclose(loaded["markers_m"], markers_m)
    np.testing.assert_allclose(loaded["target_m"], target_m)
    assert np.array_equal(loaded["valid"], valid)

    # Validate rejection on bad shapes or non-monotone time
    bad_time = np.linspace(0.0, 0.85, n_frames - 1)
    with pytest.raises(ValueError, match="shape mismatch"):
        save_native_replay_npz(
            tmp_path / "bad_shape.npz",
            time_s=bad_time,
            native_state=native_state,
            markers_m=markers_m,
            target_m=target_m,
            valid=valid,
        )

    non_monotone_time = np.array([0.0, 0.2, 0.1, 0.3])
    with pytest.raises(ValueError, match="strictly monotonically increasing"):
        save_native_replay_npz(
            tmp_path / "bad_mono.npz",
            time_s=non_monotone_time,
            native_state=np.zeros((4, n_states)),
            markers_m=np.zeros((4, n_markers, 3)),
            target_m=np.zeros((4, n_markers, 3)),
            valid=np.ones((4, n_markers), dtype=bool),
        )


def test_five_metrics_against_archived_pinocchio_reference() -> None:
    """Validate 5-metric function against the Pinocchio returned81 reference numbers."""
    if not PINOCCHIO_NPZ.exists() or not CANDIDATE_JSON.exists():
        pytest.skip("Pinocchio reference files missing")

    replay = load_native_replay_npz(PINOCCHIO_NPZ)
    candidate = json.loads(CANDIDATE_JSON.read_text(encoding="utf-8"))
    labels = candidate["marker_labels"]

    metrics: ReplayFiveMetrics = compute_replay_five_metrics(
        time_s=replay["time_s"],
        pred_markers_m=replay["markers_m"],
        target_markers_m=replay["target_m"],
        valid=replay["valid"],
        marker_labels=labels,
    )

    # Check metrics against archived Pinocchio numbers per VISUALS_HANDOFF.md Step 4
    assert metrics.whole_rms_m == pytest.approx(0.026366, abs=1e-5)
    assert metrics.early_rms_m == pytest.approx(0.011427, abs=1e-5)
    assert metrics.terminal_rms_m == pytest.approx(0.046305, abs=1e-5)
    assert metrics.club_cluster_rms_m == pytest.approx(0.015955, abs=1e-5)
    assert metrics.pelvis_yaw_error_pct == pytest.approx(13.923, abs=1e-2)


def test_cross_engine_replay_parity_and_receipt() -> None:
    """Validate cross-engine replays (MuJoCo, Drake, Pinocchio) and receipt.json."""
    receipt_path = REPLAYS_DIR / "receipt.json"
    mj_npz = REPLAYS_DIR / "mujoco_returned81_replay.npz"
    drake_npz = REPLAYS_DIR / "drake_returned81_replay.npz"

    if not (receipt_path.exists() and mj_npz.exists() and drake_npz.exists()):
        pytest.skip("Cross-engine replays or receipt missing")

    pin_replay = load_native_replay_npz(PINOCCHIO_NPZ)
    mj_replay = load_native_replay_npz(mj_npz)
    drake_replay = load_native_replay_npz(drake_npz)

    # MuJoCo vs Pinocchio parity (< 1e-12 m)
    mj_diff = np.max(np.abs(mj_replay["markers_m"] - pin_replay["markers_m"]))
    assert mj_diff < 1e-12, f"MuJoCo vs Pinocchio marker diff exceeded: {mj_diff}"

    # Drake vs Pinocchio parity (< 2e-5 m)
    drake_diff = np.max(np.abs(drake_replay["markers_m"] - pin_replay["markers_m"]))
    assert drake_diff < 2e-5, f"Drake vs Pinocchio marker diff exceeded: {drake_diff}"

    # Check combined receipt
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert len(receipt["comparison_table"]) == 5
    for row in receipt["comparison_table"]:
        if "max_abs_diff_m" in row:
            assert row["max_abs_diff_m"] < 5e-5
        elif "max_abs_diff_pct" in row:
            assert row["max_abs_diff_pct"] < 0.01


def test_early_rms_nan_when_no_valid_sample_before_cutoff() -> None:
    """Issue #11573: early_rms_m must report NaN, not 0.0, when unmeasurable."""
    time_s = np.array([0.70, 0.85])
    pred = np.zeros((2, 2, 3))
    target = np.ones((2, 2, 3))
    valid = np.ones((2, 2), dtype=bool)
    labels = ["Marker_1", "Marker_2"]

    metrics = compute_replay_five_metrics(
        time_s=time_s,
        pred_markers_m=pred,
        target_markers_m=target,
        valid=valid,
        marker_labels=labels,
        early_cutoff_s=0.60,
    )
    assert np.isnan(metrics.early_rms_m), f"Expected NaN, got {metrics.early_rms_m}"


def test_terminal_rms_nan_when_last_frame_has_no_valid_markers() -> None:
    """Issue #11573: terminal_rms_m must report NaN, not 0.0, when last frame has no valid markers."""
    time_s = np.array([0.10, 0.85])
    pred = np.zeros((2, 2, 3))
    target = np.ones((2, 2, 3))
    valid = np.array([[True, True], [False, False]], dtype=bool)
    labels = ["Marker_1", "Marker_2"]

    metrics = compute_replay_five_metrics(
        time_s=time_s,
        pred_markers_m=pred,
        target_markers_m=target,
        valid=valid,
        marker_labels=labels,
    )
    assert np.isnan(metrics.terminal_rms_m), (
        f"Expected NaN, got {metrics.terminal_rms_m}"
    )


def test_club_cluster_rms_nan_when_no_club_markers_exist() -> None:
    """Issue #11573: club_cluster_rms_m must report NaN, not 0.0, when no club markers exist."""
    time_s = np.array([0.10, 0.85])
    pred = np.zeros((2, 2, 3))
    target = np.ones((2, 2, 3))
    valid = np.ones((2, 2), dtype=bool)
    labels = ["WaistLeft", "WaistRight"]

    metrics = compute_replay_five_metrics(
        time_s=time_s,
        pred_markers_m=pred,
        target_markers_m=target,
        valid=valid,
        marker_labels=labels,
    )
    assert np.isnan(metrics.club_cluster_rms_m), (
        f"Expected NaN, got {metrics.club_cluster_rms_m}"
    )


def test_club_cluster_rms_nan_when_club_markers_invalid_on_terminal_frame() -> None:
    """Issue #11573: club_cluster_rms_m must report NaN, not 0.0, when club markers are invalid on terminal frame."""
    time_s = np.array([0.10, 0.85])
    pred = np.zeros((2, 2, 3))
    target = np.ones((2, 2, 3))
    # Marker 0 is club, valid on frame 0, invalid on frame 1 (terminal)
    valid = np.array([[True, True], [False, True]], dtype=bool)
    labels = ["Marker_2", "Wrist"]

    metrics = compute_replay_five_metrics(
        time_s=time_s,
        pred_markers_m=pred,
        target_markers_m=target,
        valid=valid,
        marker_labels=labels,
    )
    assert np.isnan(metrics.club_cluster_rms_m), (
        f"Expected NaN, got {metrics.club_cluster_rms_m}"
    )


def test_pelvis_yaw_nan_when_waist_markers_missing() -> None:
    """Issue #11573: pelvis_yaw_error_pct must report NaN, not 0.0, when waist markers are absent."""
    time_s = np.array([0.10, 0.85])
    pred = np.zeros((2, 2, 3))
    target = np.ones((2, 2, 3))
    valid = np.ones((2, 2), dtype=bool)
    labels = ["Marker_1", "Marker_2"]

    metrics = compute_replay_five_metrics(
        time_s=time_s,
        pred_markers_m=pred,
        target_markers_m=target,
        valid=valid,
        marker_labels=labels,
    )
    assert np.isnan(metrics.pelvis_yaw_error_pct), (
        f"Expected NaN, got {metrics.pelvis_yaw_error_pct}"
    )


def test_pelvis_yaw_nan_when_waist_markers_invalid_on_terminal_frame() -> None:
    """Issue #11573: pelvis_yaw_error_pct must report NaN, not 0.0, when waist markers are invalid on terminal frame."""
    time_s = np.array([0.10, 0.85])
    pred = np.zeros((2, 2, 3))
    target = np.ones((2, 2, 3))
    # WaistLeft is invalid on frame 1
    valid = np.array([[True, True], [False, True]], dtype=bool)
    labels = ["WaistLeft", "WaistRight"]

    metrics = compute_replay_five_metrics(
        time_s=time_s,
        pred_markers_m=pred,
        target_markers_m=target,
        valid=valid,
        marker_labels=labels,
    )
    assert np.isnan(metrics.pelvis_yaw_error_pct), (
        f"Expected NaN, got {metrics.pelvis_yaw_error_pct}"
    )


def test_acceptance_evaluate_fails_on_nan_metric() -> None:
    """Issue #11573: evaluate() must fail a gate whose measured value is NaN, not skip it."""
    from src.shared.python.motion_matching.acceptance import (
        GateStatus,
        Horizon,
        evaluate,
    )

    acceptance_input = {
        "shared_metrics": {
            "whole_marker_rmse_m": 0.015,
            "early_marker_rmse_m": float("nan"),
            "terminal_marker_rmse_m": 0.020,
            "club_marker_rmse_m": float("nan"),
            "pelvis_yaw_rmse_rad": 0.02,
        },
        "contact_audit": {
            "max_normal_force_n": 100.0,
            "max_penetration_m": 0.001,
            "weight_fraction": 0.95,
        },
        "max_closure_residual_m": 0.005,
    }
    verdict = evaluate(acceptance_input, horizon=Horizon.G1)
    assert not verdict.is_physically_accepted, (
        "Replay with NaN gated metrics must not be accepted"
    )
    early_gate = next(g for g in verdict.gates if g.name == "early_marker_rmse_m")
    assert early_gate.status == GateStatus.FAILED
    club_gate = next(g for g in verdict.gates if g.name == "club_marker_rmse_m")
    assert club_gate.status == GateStatus.FAILED
