"""Owned impact worker contracts; actual impact, optional test-only flight fixture."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from src.shared.python.workspace import necromatcher_impact_worker as worker
from src.shared.python.workspace.necromatcher_impact import (
    ReplayImpactGeometry,
    ReplayImpactSelection,
    extract_replay_impact_state,
)
from src.shared.python.workspace.necromatcher_impact_receipt import (
    export_replay_impact_receipt,
    load_replay_impact_receipt,
)
from src.shared.python.workspace.trajectory_handoff import (
    ShotTrajectoryHandoffCoordinator,
)
from tests.unit.workspace.impact_fixture import impact_case

pytestmark = pytest.mark.unit


@pytest.fixture
def case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[dict, object]:
    library = impact_case(monkeypatch)
    stamp = {
        "source_commit": "a" * 40,
        "source_sha256": "b" * 64,
        "runtime_sha256": "c" * 64,
        "started_at_utc": "earlier",
    }
    from src.shared.python.workspace.necromatcher_impact_jobs import impact_job_parents

    api = SimpleNamespace(
        library=lambda root: library,
        stamp=lambda: stamp.copy(),
        parents=impact_job_parents,
        extract=extract_replay_impact_state,
        coordinator=ShotTrajectoryHandoffCoordinator,
        receipt=export_replay_impact_receipt,
    )
    monkeypatch.setattr(worker, "_api", lambda: api)
    geometry = ReplayImpactGeometry(
        "club", (0.1, 0, 0), (0, -1, 0), (0, 0, 1), 0.2, 0.005, "authored test"
    )
    selection = ReplayImpactSelection(
        1, ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0, 0, 0), "recorded test"
    )
    request = {
        "library_root": str(tmp_path),
        "replay_id": "replay",
        "geometry": geometry.to_record(),
        "selection": selection.to_record(),
        "execution_stamp": stamp,
        "output_root": str(tmp_path / "output"),
    }
    return request, api


@pytest.fixture
def optional_flight() -> object:
    # Canonical integration fixture substitutes ONLY the unavailable Rust flight.
    from tests.integration.test_shot_trajectory_handoff import (
        _ensure_rust_available_or_mock,
    )

    generator = _ensure_rust_available_or_mock.__wrapped__()
    next(generator)
    yield
    try:
        next(generator)
    except StopIteration:
        pass


def test_actual_impact_and_receipt_preserve_screw_vectors(
    case, optional_flight
) -> None:
    request, api = case
    output = worker.compute_native_impact(request)
    root = Path(request["output_root"])
    assert {p.name for p in root.iterdir()} == {
        "trajectory.json",
        "impact-receipt.json",
        "result.json",
    }
    state = load_replay_impact_receipt(
        root / "impact-receipt.json", root / "trajectory.json"
    )
    np.testing.assert_allclose(
        state.clubhead_velocity, [1.7, 3 * 0.1 * np.cos(np.pi / 2), 0], rtol=0, atol=0
    )
    np.testing.assert_array_equal(state.clubhead_angular_velocity, [0, 0, 3])
    result = json.loads((root / "result.json").read_text())
    assert result["scientific_qualified"] is False
    assert result["physical_source_time_qualified"] is False
    assert result["environment"] == worker._json_value(
        asdict(api.coordinator().get_environment_conditions())
    )
    assert result["impact_params"]["cor"] > 0
    assert result["impact_state"]["ball_velocity"][0] > 0
    assert set(output["artifact_hashes"]) == {
        "trajectory.json",
        "impact-receipt.json",
        "result.json",
    }


@pytest.mark.parametrize("fault", ["stamp", "geometry", "extra", "existing"])
def test_admission_rejects_before_extraction(case, monkeypatch, fault: str) -> None:
    request, api = case
    calls = []
    api.extract = lambda *args: calls.append(args)
    if fault == "stamp":
        request["execution_stamp"] = {}
    elif fault == "geometry":
        request["geometry"]["mass_kg"] = True
    elif fault == "extra":
        request["unexpected"] = 1
    else:
        Path(request["output_root"]).mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        worker.compute_native_impact(request)
    assert not calls


@pytest.mark.parametrize("fault", ["stamp", "parents", "receipt"])
def test_after_fault_has_no_partial_success(case, optional_flight, fault: str) -> None:
    request, api = case
    original = api.receipt

    def receipt(*args):
        original(*args)
        if fault == "stamp":
            api.stamp = lambda: {"changed": True}
        elif fault == "parents":
            api.parents = lambda *args: {"changed": True}
        else:
            raise OSError("receipt fault")

    api.receipt = receipt
    with pytest.raises((ValueError, OSError)):
        worker.compute_native_impact(request)
    assert not Path(request["output_root"]).exists()
    assert not list(Path(request["output_root"]).parent.glob(".impact-*"))


def test_real_parent_tamper_rejected_before_pipeline(case) -> None:
    request, api = case
    library = api.library("")
    library.trace.meta["model_hash"] = "sha256:" + "a" * 64
    with pytest.raises(ValueError, match="parent"):
        worker.compute_native_impact(request)
    assert not Path(request["output_root"]).exists()


def test_stamp_observation_time_is_not_stable_identity(case, optional_flight) -> None:
    request, api = case
    current = dict(request["execution_stamp"], started_at_utc="later")
    api.stamp = lambda: current
    worker.compute_native_impact(request)
    record = json.loads((Path(request["output_root"]) / "result.json").read_text())
    assert record["execution_stamp"] == request["execution_stamp"]


@pytest.mark.parametrize("key", ["source_commit", "source_sha256", "runtime_sha256"])
def test_stable_stamp_change_rejects_before_extraction(case, key: str) -> None:
    request, api = case
    api.stamp = lambda: dict(request["execution_stamp"], **{key: "changed"})
    with pytest.raises(ValueError, match="stamp"):
        worker.compute_native_impact(request)
    assert not Path(request["output_root"]).exists()


def test_publish_race_preserves_foreign_output(case, optional_flight) -> None:
    request, api = case
    original = api.receipt
    root = Path(request["output_root"])

    def receipt(*args):
        original(*args)
        root.mkdir()
        (root / "foreign.txt").write_text("preserve")

    api.receipt = receipt
    with pytest.raises(FileExistsError):
        worker.compute_native_impact(request)
    assert (root / "foreign.txt").read_text() == "preserve"
    assert {p.name for p in root.iterdir()} == {"foreign.txt"}


def test_cold_import_and_main_sdk_order(tmp_path: Path) -> None:
    import subprocess
    import sys

    from src.shared.python.core import repo_python_environment
    from src.shared.python.version_info import get_repo_root

    code = """
import sys, types
from src.shared.python.workspace import necromatcher_impact_worker as worker
assert "mujoco" not in sys.modules
class Gate:
    def find_spec(self, name, path=None, target=None):
        if name.endswith(".necromatcher"):
            assert "mujoco" in sys.modules, "workspace loaded before SDK"
            raise RuntimeError("verified SDK first")
sys.modules["mujoco"] = types.ModuleType("mujoco")
sys.modules.pop("src.shared.python.workspace.necromatcher", None)
sys.meta_path.insert(0, Gate())
try:
    worker.main()
except RuntimeError as exc:
    assert str(exc) == "verified SDK first"
else:
    raise AssertionError("missing import gate")
"""
    root = get_repo_root()
    request_path = _main_envelope(tmp_path, "a" * 32)
    completed = subprocess.run(
        [sys.executable, "-c", code, str(request_path)],
        cwd=root,
        env=repo_python_environment(root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def _main_envelope(root: Path, run_id: str) -> Path:
    path = root / "impact-runs" / run_id / "request.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "kind": "necromatcher/impact-job/1",
                "run_id": run_id,
                "library_root": str(root),
                "replay_id": "unused",
                "parents": {},
                "geometry": {},
                "selection": {},
                "budget_wall_s": 10,
                "execution_stamp": {},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("fault", ["run_id", "linked_parent"])
def test_invalid_main_path_rejects_before_sdk(tmp_path: Path, fault: str) -> None:
    import subprocess
    import sys

    from src.shared.python.core import repo_python_environment
    from src.shared.python.version_info import get_repo_root

    path = _main_envelope(tmp_path, "not-uuid" if fault == "run_id" else "a" * 32)
    if fault == "linked_parent":
        alias = tmp_path.parent / (tmp_path.name + "-alias")
        try:
            alias.symlink_to(tmp_path, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"Directory symlink capability unavailable: {exc}")
        path = alias / "impact-runs" / ("a" * 32) / "request.json"
    code = """
import sys, importlib.abc
class NoSDK(importlib.abc.MetaPathFinder):
 def find_spec(self, name, path=None, target=None):
  if name.split('.')[0] == 'mujoco':
   raise AssertionError('SDK loaded before path rejection')
sys.meta_path.insert(0, NoSDK())
from src.shared.python.workspace.necromatcher_impact_worker import main
try:
 main()
except ValueError:
 pass
else:
 raise AssertionError('Malformed request admitted')
"""
    root = get_repo_root()
    completed = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=root,
        env=repo_python_environment(root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
