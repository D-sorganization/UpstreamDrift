"""Unit tests verifying MuJoCo calibration provenance and initial closure checks (#10167)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
UPPER_PATH = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
CALIB_DIR = ROOT / "docs/development/full_body_models/evidence/fb4_calibration/mujoco"
HISTORICAL_DIR = CALIB_DIR / "historical"


def _load_adapter() -> tuple[Any, dict[str, Any]]:
    from src.engines.physics_engines.mujoco.python.full_body_ik import MujocoFullBodyIK
    from src.shared.python.motion_matching.full_body_spec import load_full_body_spec

    spec = load_full_body_spec(
        SPEC_PATH, json.loads(UPPER_PATH.read_text(encoding="utf-8"))
    )
    return MujocoFullBodyIK(spec), spec


@pytest.mark.requires_mujoco
def test_historical_calibration_exhibits_scrambled_coordinate_error() -> None:
    """Historical stored IK generated before #10140 shows ~1.266 m grip closure."""
    pytest.importorskip("mujoco")
    hist_traj = HISTORICAL_DIR / "ik_trajectory.npz"
    if not hist_traj.exists():
        pytest.skip("Historical calibration artifact not found")

    adapter, _ = _load_adapter()
    with np.load(hist_traj) as archive:
        q0_hist = archive["q"][0].copy()

    # Under corrected coordinate-indexed mapping, historical q0 causes ~1.266 m error
    hist_closure = float(np.linalg.norm(adapter.closure_residuals(q0_hist)))
    assert hist_closure > 1.0, f"Expected legacy error > 1.0 m, got {hist_closure}"
    assert np.isclose(hist_closure, 1.265687, atol=1e-3)


@pytest.mark.requires_mujoco
def test_active_calibration_satisfies_initial_closure_check() -> None:
    """Active calibrated IK trajectory q0 satisfies initial weld loop closure (< 2 cm)."""
    pytest.importorskip("mujoco")
    active_traj = CALIB_DIR / "ik_trajectory.npz"
    assert active_traj.exists(), "Active ik_trajectory.npz must exist"

    adapter, _ = _load_adapter()
    with np.load(active_traj) as archive:
        q0_active = archive["q"][0].copy()

    closure_res = adapter.closure_residuals(q0_active)
    closure_norm = float(np.linalg.norm(closure_res))

    # Scientifically justified initial closure check:
    # Must be orders of magnitude better than the legacy 1.266 m scrambled error.
    # The active calibrated trajectory achieves ~1.37 cm (0.0137 m).
    assert closure_norm < 0.02, (
        f"Active initial closure residual {closure_norm:.6f} m exceeds 0.02 m threshold; "
        "indicates stale or improperly mapped calibration artifact."
    )
    # But it is not identically zero (real trajectory fit)
    assert closure_norm > 0.001


def test_active_calibration_coordinate_order_matches_spec() -> None:
    """Active ik_trajectory coordinate_order matches full_body_spec_v1.json exactly."""
    active_traj = CALIB_DIR / "ik_trajectory.npz"
    assert active_traj.exists(), "Active ik_trajectory.npz must exist"

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    expected_order = list(spec["coordinate_order"])

    with np.load(active_traj) as archive:
        stored_order = list(archive["coordinate_order"])

    assert stored_order == expected_order
    assert len(stored_order) == 41


def test_active_calibration_receipt_metadata_and_hashes() -> None:
    """Active receipt.json records valid artifact hashes and status."""
    receipt_path = CALIB_DIR / "receipt.json"
    assert receipt_path.exists(), "receipt.json must exist"

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["engine"] == "mujoco"
    # #10960 P0-9: the active fit misses its RMS and closure thresholds, so the
    # receipt is honestly REJECTED and must say which threshold failed.
    assert receipt["status"] == "REJECTED"
    assert "threshold" in receipt["note"]
    assert receipt["full_trajectory"]["num_frames"] == 654
    assert receipt["full_trajectory"]["closure_max_error_m"] < 0.02

    # Verify sha256 of artifacts match receipt
    calib_offsets_path = CALIB_DIR / "calibrated_offsets.json"
    ik_traj_path = CALIB_DIR / "ik_trajectory.npz"

    calib_offsets_sha = hashlib.sha256(calib_offsets_path.read_bytes()).hexdigest()
    ik_traj_sha = hashlib.sha256(ik_traj_path.read_bytes()).hexdigest()

    assert receipt["artifacts"]["calibrated_offsets_sha256"] == calib_offsets_sha
    assert receipt["artifacts"]["ik_trajectory_sha256"] == ik_traj_sha
