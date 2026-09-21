"""Native MyoSuite kinematic replay marker parity (MS-52, #10345)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_myosuite, pytest.mark.requires_mujoco]

ROOT = Path(__file__).resolve().parents[2]
SOURCE_CANDIDATE = ROOT / "evidence/matched/driver_full_pinocchio/candidate.npz"
MARKER_RMS_LIMIT_M = 0.015


def _source_markers() -> tuple[np.ndarray, np.ndarray]:
    with np.load(SOURCE_CANDIDATE, allow_pickle=False) as z:
        markers = np.asarray(z["markers_m"], dtype=np.float64)
        valid = np.asarray(z["valid"], dtype=bool)
    return markers, valid


@pytest.mark.requires_mujoco
def test_native_replay_produces_receipt_and_artifacts(tmp_path: Path) -> None:
    """Kinematic replay writes receipt, candidate, and GIF with honest parity fields."""
    pytest.importorskip("mujoco")
    if not SOURCE_CANDIDATE.is_file():
        pytest.skip("driver_full_pinocchio candidate missing")

    from src.engines.physics_engines.myosuite.python.replay import (
        ReplayConfig,
        run_kinematic_replay,
    )

    out = tmp_path / "driver_g1_myosuite"
    receipt = run_kinematic_replay(
        ReplayConfig(
            candidate=SOURCE_CANDIDATE,
            output_dir=out,
            source_engine="mujoco",
        )
    )
    assert receipt["stage"] == "replay"
    assert receipt["dynamics"]["status"] == "not_run"
    parity = receipt["parity"]
    assert parity["comparison"] == "native_vs_source_predicted_markers"
    assert np.isfinite(parity["marker_rms_m"])
    assert parity["marker_rms_limit_m"] == MARKER_RMS_LIMIT_M
    assert "topology_note" in parity
    assert (out / "receipt.json").is_file()
    assert (out / "candidate.npz").is_file()
    assert (out / "playback.gif").is_file()


@pytest.mark.requires_mujoco
def test_native_marker_parity_within_budget_when_scene_qualified(
    tmp_path: Path,
) -> None:
    """15 mm marker parity requires the MS-51 golfer scene, not the placeholder MJCF."""
    pytest.importorskip("mujoco")
    if not SOURCE_CANDIDATE.is_file():
        pytest.skip("driver_full_pinocchio candidate missing")

    from src.engines.physics_engines.myosuite.python.golfer_scene import (
        resolve_golfer_scene,
    )
    from src.engines.physics_engines.myosuite.python.replay import (
        ReplayConfig,
        run_kinematic_replay,
    )

    scene = resolve_golfer_scene()
    if scene.is_placeholder:
        pytest.skip("MS-51 pinned myo_sim golfer scene required for 15 mm parity")

    out = tmp_path / "driver_g1_myosuite"
    receipt = run_kinematic_replay(
        ReplayConfig(candidate=SOURCE_CANDIDATE, output_dir=out)
    )
    assert receipt["parity"]["marker_rms_m"] <= MARKER_RMS_LIMIT_M


@pytest.mark.requires_mujoco
def test_receipt_hashes_match_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    if not SOURCE_CANDIDATE.is_file():
        pytest.skip("driver_full_pinocchio candidate missing")

    from src.engines.physics_engines.myosuite.python.replay import (
        ReplayConfig,
        run_kinematic_replay,
    )

    out = tmp_path / "replay_out"
    receipt = run_kinematic_replay(
        ReplayConfig(candidate=SOURCE_CANDIDATE, output_dir=out)
    )
    for name, meta in receipt["artifacts"].items():
        path = out / meta["path"]
        assert path.is_file(), name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == meta["sha256"], name


@pytest.mark.unit
def test_receipt_schema_fields_without_native_run() -> None:
    """Receipt parity block documents topology-aware comparison contract."""
    sample = {
        "schema_version": "matched-swing-replay/1",
        "engine": "myosuite",
        "stage": "replay",
        "dynamics": {"status": "not_run", "reason": "kinematic milestone MS-52"},
        "parity": {
            "comparison": "native_vs_source_predicted_markers",
            "marker_rms_m": 0.01,
            "topology_note": "intermediate placeholder scene until MS-51 pins myo_sim",
        },
    }
    assert sample["dynamics"]["status"] == "not_run"
    assert sample["parity"]["marker_rms_m"] <= MARKER_RMS_LIMIT_M
