"""Portable authored replay state receipts preserve vectors without simulation."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from src.shared.python.physics import SwingState
from src.shared.python.physics.flight_trajectory_export import FLIGHT_FRAME_ID
from src.shared.python.workspace import necromatcher_impact_receipt as owner
from src.shared.python.workspace.necromatcher_impact import (
    ReplayImpactGeometry,
    ReplayImpactSelection,
)
from src.shared.python.workspace.necromatcher_impact_receipt import (
    export_replay_impact_receipt,
    load_replay_impact_receipt,
)

pytestmark = pytest.mark.unit


def extracted_state() -> SwingState:
    """Independent asymmetric fixture with complete authored extraction metadata."""
    geometry = ReplayImpactGeometry(
        "club",
        (0.01, -0.02, 0.03),
        (0.6, 0.8, 0),
        (0, 0, 1),
        0.21,
        0.004,
        "Authored effective impact geometry and inertia",
    )
    selection = ReplayImpactSelection(
        7,
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        (2, -3, 4),
        "Authored sample and flight map; no contact detection",
    )
    metadata = {
        "frame_id": FLIGHT_FRAME_ID,
        "impact_extraction_schema": "necromatcher/replay-impact/1",
        "recorded_sample_index": 7,
        "recorded_time_s": 0.75,
        "replay_clock_policy": "authored_simulation_seconds",
        "physical_source_time_qualified": False,
        "scientific_qualified": False,
        "capture_pts_used_for_velocity": False,
        "geometry": geometry.to_record(),
        "selection": selection.to_record(),
        "head_point_flight_m": [2.01, -3.02, 4.03],
        "capture_initial_frame": {"frame_id": "source-17", "pts_numerator": 17},
        "capture_initial_frame_index": 17,
        "basis_probe_length_m": 1.0,
        "extra_authored_note": {"nothing_calibrated": True},
    }
    for index, name in enumerate(("replay", "profile", "fit", "model", "capture")):
        metadata[name + "_id"] = name + "-id"
        metadata[name + "_hash"] = "sha256:" + str(index + 1) * 64
    return SwingState(
        np.array([42.0, 4.0, -3.0]),
        np.array([1.0, -2.0, 3.0]),
        np.array([0.6, 0.8, 0]),
        clubhead_mass=0.21,
        clubhead_moi=0.004,
        clubhead_loft_deg=0,
        impact_offset=np.array([0.002, -0.003]),
        engine_name="synthetic",
        metadata=metadata,
    )


def trajectory_file(tmp_path: Path) -> Path:
    """Produce the independent canonical six-key viewer interchange fixture."""
    payload = {
        "format": "swing_sim.ball_flight_trajectory/1",
        "source_id": "independent-flight",
        "frame_id": FLIGHT_FRAME_ID,
        "channels": ["velocity_mps"],
        "provenance": {
            "model_family": "ud.flight_models",
            "model_name": "Waterloo/Penner",
            "parameter_digest": "a" * 64,
        },
        "samples": [
            {"time_s": 0, "position_m": [0, 0, 0], "velocity_mps": [40, 3, 10]},
            {"time_s": 0.1, "position_m": [4, 0.3, 0.95], "velocity_mps": [39, 3, 9]},
        ],
    }
    target = tmp_path / "trajectory.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def test_portable_exact_roundtrip_preserves_all_state_without_rerun(
    tmp_path: Path,
) -> None:
    trajectory = trajectory_file(tmp_path)
    original_bytes = trajectory.read_bytes()
    state = extracted_state()
    before = deepcopy(state.metadata)
    receipt = tmp_path / "receipt.json"
    assert export_replay_impact_receipt(state, trajectory, receipt) == receipt
    moved = tmp_path / "portable"
    moved.mkdir()
    relocated_trajectory = moved / "renamed-flight.json"
    relocated_trajectory.write_bytes(original_bytes)
    relocated_receipt = moved / "renamed-receipt.json"
    relocated_receipt.write_bytes(receipt.read_bytes())
    loaded = load_replay_impact_receipt(relocated_receipt, relocated_trajectory)
    for field in (
        "clubhead_velocity",
        "clubhead_angular_velocity",
        "clubhead_orientation",
        "impact_offset",
    ):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(state, field))
    for field in ("clubhead_mass", "clubhead_moi", "clubhead_loft_deg", "engine_name"):
        assert getattr(loaded, field) == getattr(state, field)
    assert loaded.metadata == before == state.metadata
    assert loaded.metadata is not state.metadata
    loaded.metadata["geometry"]["body"] = "mutated"
    loaded.clubhead_velocity[:] = 0
    assert load_replay_impact_receipt(receipt, trajectory).metadata == before
    assert trajectory.read_bytes() == original_bytes
    wire = json.loads(original_bytes)
    assert len(wire) == 6 and len(wire["provenance"]) == 3
    assert "path" not in json.loads(receipt.read_bytes())["trajectory"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("scientific_qualified", True),
        ("physical_source_time_qualified", 0),
        ("capture_pts_used_for_velocity", None),
        ("frame_id", "app_xtarget_yup_zright"),
        ("impact_extraction_schema", "unsupported"),
        ("replay_clock_policy", "capture_pts"),
        ("recorded_sample_index", True),
        ("recorded_time_s", float("nan")),
        ("model_hash", "wrong"),
        ("geometry", {}),
    ],
)
def test_export_rejects_malformed_extraction_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    state = extracted_state()
    state.metadata[field] = value
    destination = tmp_path / "refused.json"
    with pytest.raises((ValueError, TypeError)):
        export_replay_impact_receipt(state, trajectory_file(tmp_path), destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("clubhead_velocity", [True, 4, -3]),
        ("clubhead_angular_velocity", [1, 2]),
        ("clubhead_orientation", [1, 1, 1]),
        ("clubhead_mass", 0),
        ("clubhead_moi", float("inf")),
        ("impact_offset", [1, 2, 3]),
        ("engine_name", ""),
    ],
)
def test_export_rejects_invalid_state(
    tmp_path: Path, field: str, value: object
) -> None:
    state = extracted_state()
    setattr(state, field, value)
    with pytest.raises((ValueError, TypeError)):
        export_replay_impact_receipt(
            state, trajectory_file(tmp_path), tmp_path / "receipt.json"
        )


def test_overwrite_refused_without_modifying_any_bytes(tmp_path: Path) -> None:
    trajectory = trajectory_file(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b"owned-existing-receipt")
    with pytest.raises(FileExistsError):
        export_replay_impact_receipt(extracted_state(), trajectory, receipt)
    assert receipt.read_bytes() == b"owned-existing-receipt"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "fault", ["hash", "size", "frame", "wire", "flag", "nan", "schema"]
)
def test_loader_rejects_corruption_and_unsupported_wire(
    tmp_path: Path, fault: str
) -> None:
    trajectory = trajectory_file(tmp_path)
    receipt = tmp_path / "receipt.json"
    export_replay_impact_receipt(extracted_state(), trajectory, receipt)
    record = json.loads(receipt.read_bytes())
    if fault == "hash":
        trajectory.write_bytes(trajectory.read_bytes() + b" ")
    elif fault == "size":
        record["trajectory"]["size_bytes"] = True
    elif fault == "frame":
        record["state"]["metadata"]["frame_id"] = "app_xtarget_yup_zright"
    elif fault == "wire":
        trajectory.write_text('{"frame_id":"flight_xfwd_yleft_zup"}')
        import hashlib

        record["trajectory"] = {
            "size_bytes": trajectory.stat().st_size,
            "sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        }
    elif fault == "flag":
        record["state"]["metadata"]["scientific_qualified"] = 0
    elif fault == "nan":
        record["state"]["clubhead_velocity"][0] = float("nan")
    else:
        record["schema"] = "unsupported"
    receipt.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        load_replay_impact_receipt(receipt, trajectory)


def test_none_offset_preserved(tmp_path: Path) -> None:
    state = extracted_state()
    state.impact_offset = None
    trajectory = trajectory_file(tmp_path)
    receipt = export_replay_impact_receipt(state, trajectory, tmp_path / "receipt.json")
    assert load_replay_impact_receipt(receipt, trajectory).impact_offset is None


def test_failed_stage_write_preserves_primary_and_cleans_owned_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_fsync(_descriptor: int) -> None:
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(owner.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="synthetic fsync failure"):
        export_replay_impact_receipt(
            extracted_state(), trajectory_file(tmp_path), tmp_path / "receipt.json"
        )
    assert not (tmp_path / "receipt.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_link_preserves_primary_trajectory_and_cleans_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = trajectory_file(tmp_path)
    original = trajectory.read_bytes()

    def fail_link(_source: Path, _target: Path) -> None:
        raise OSError("synthetic link failure")

    monkeypatch.setattr(owner.os, "link", fail_link)
    with pytest.raises(OSError, match="synthetic link failure"):
        export_replay_impact_receipt(
            extracted_state(), trajectory, tmp_path / "receipt.json"
        )
    assert trajectory.read_bytes() == original
    assert not (tmp_path / "receipt.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_cleanup_fault_never_masks_primary_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = trajectory_file(tmp_path)
    with monkeypatch.context() as patch:

        def fail_link(_source: Path, _target: Path) -> None:
            raise OSError("primary link failure")

        def fail_cleanup(_path: Path, **_kwargs: object) -> None:
            raise OSError("synthetic cleanup failure")

        patch.setattr(owner.os, "link", fail_link)
        patch.setattr(Path, "unlink", fail_cleanup)
        with pytest.raises(OSError, match="primary link failure") as caught:
            export_replay_impact_receipt(
                extracted_state(), trajectory, tmp_path / "receipt.json"
            )
        assert "cleanup failed" in " ".join(caught.value.__notes__)
    for owned in tmp_path.glob("*.tmp"):
        owned.unlink()


def test_cold_load_blocks_native_sdks_and_gui(tmp_path: Path) -> None:
    trajectory = trajectory_file(tmp_path)
    receipt = export_replay_impact_receipt(
        extracted_state(), trajectory, tmp_path / "receipt.json"
    )
    code = """
import importlib.abc
import sys
from pathlib import Path
class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = {'mujoco', 'pydrake', 'pinocchio', 'opensim', 'PyQt6'}
        if fullname.split('.')[0] in forbidden:
            raise AssertionError('Unexpected SDK import: ' + fullname)
        return None
sys.meta_path.insert(0, NoSDK())
from src.shared.python.workspace.necromatcher_impact_receipt import (
    load_replay_impact_receipt,
)
state = load_replay_impact_receipt(Path(sys.argv[1]), Path(sys.argv[2]))
assert state.clubhead_velocity.tolist() == [42.0, 4.0, -3.0]
assert state.metadata['scientific_qualified'] is False
"""
    root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        str(path) for path in (root, root / "src", root / "src/shared/python")
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(receipt), str(trajectory)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
