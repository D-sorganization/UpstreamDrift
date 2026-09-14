"""Real-runtime evidence checks for the ST-01 model probe (#10124)."""

import json
import sys
from pathlib import Path

import pytest

from scripts.shadow_tracker.model_probe import main, run_probe


@pytest.mark.unit
def test_probe_rejects_invalid_horizon_before_loading_engines(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frames"):
        run_probe(tmp_path, frames=1)


@pytest.mark.live_simulation
@pytest.mark.requires_mujoco
@pytest.mark.slow
def test_probe_executes_independent_replays_without_qualification_claim() -> None:
    pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[3]
    receipt = run_probe(root, frames=10)
    assert receipt["scientifically_qualified"] is False
    assert receipt["native_state"]["nq"] == 41
    assert receipt["native_state"]["nv"] == 41
    assert receipt["canonical_adapter_verified"] is False
    assert len(receipt["runs"][0]["native_qpos_order"]) == 41
    assert "ik_initial_grip_translation_m" in receipt["runs"][0]
    assert "dynamics_initial_grip_translation_m" in receipt["runs"][0]
    assert len(receipt["runs"]) == 3
    assert all(run["solver_status"] == "success" for run in receipt["runs"])
    assert all(run["frames_returned"] == 10 for run in receipt["runs"])
    assert all("max_grip_translation_m" in run for run in receipt["runs"])
    assert all("max_grip_rotation_rad" in run for run in receipt["runs"])
    assert receipt["repeat_q"]["max_translation_difference"] < 1e-10
    assert receipt["repeat_q"]["max_rotation_difference"] < 1e-10


@pytest.mark.live_simulation
@pytest.mark.requires_mujoco
@pytest.mark.slow
def test_probe_cli_records_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(
        sys, "argv", ["probe", "--root", str(root), "--output", str(output)]
    )
    assert main() == 0
    receipt = json.loads(output.read_text())
    assert receipt["scientifically_qualified"] is False
    assert len(receipt["source_sha256"]) >= 5
    assert all(len(value) == 64 for value in receipt["source_sha256"].values())
    assert receipt["timestamp_utc"].endswith("+00:00")
