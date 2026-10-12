"""Authenticated saved research impact converts without rerunning physics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
from src.shared.python.golf_simulator.contracts import ShotMetadata
import json
import pytest
from tests.unit.workspace.test_necromatcher_impact_jobs import (
    fake_worker,
    wait,
)
from tests.unit.workspace import test_necromatcher_impact_jobs as job_fixtures
from tests.unit.workspace.test_necromatcher_impact_contracts import geometry, selection

pytestmark = pytest.mark.unit
setup = job_fixtures.setup


def _metadata() -> ShotMetadata:
    from src.shared.python.golf_simulator import AimContext, ShotMetadata, SourceKind

    return ShotMetadata(
        "shot",
        "local-session",
        AimContext(((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        "2026-10-04T10:00:00Z",
        source_kind=SourceKind.MODEL_CONTACT,
    )


def _worker(
    path: Path, budget: float, cancelled: Callable, *, operation: str
) -> dict[str, Any]:
    from src.shared.python.workspace.artifact_handoff import compute_file_sha256

    fake_worker(path, budget, cancelled, operation=operation)
    out = path.parent / "output"
    result = json.loads((out / "result.json").read_bytes())
    result["extraction_metadata"] = json.loads(
        (out / "impact-receipt.json").read_bytes()
    )["state"]["metadata"]
    from dataclasses import asdict
    from src.shared.python.physics.ball_properties import BallProperties
    from src.shared.python.physics.impact_model.types import ImpactParameters
    from src.shared.python.physics.ball_launch_conditions import EnvironmentalConditions

    result.update(
        environment={
            key: (value.tolist() if hasattr(value, "tolist") else value)
            for key, value in asdict(EnvironmentalConditions()).items()
        },
        ball_assumptions=asdict(BallProperties()),
        impact_params=asdict(ImpactParameters()),
        impact_method="RIGID_BODY",
        flight_settings={"dt_s": 0.01, "max_time_s": 10.0},
    )
    result["impact_state"] = {
        "ball_velocity": [42, 4, -3],
        "ball_angular_velocity": [1, -2, 3],
        "clubhead_velocity": [7, 8, 9],
        "clubhead_angular_velocity": [2, 3, 4],
        "contact_duration": 0.0005,
        "energy_transfer": 12.5,
        "impact_location": [0.002, -0.003],
    }
    (out / "result.json").write_text(json.dumps(result))
    return {"artifact_hashes": {p.name: compute_file_sha256(p) for p in out.iterdir()}}


@pytest.fixture
def completed(setup: Any, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, str]:
    jobs, library = setup
    monkeypatch.setattr(jobs, "execute_native_research_worker", _worker)
    session = jobs.NativeImpactSession(library)
    try:
        run = session.submit("replay", geometry(), selection(), 30)["run_id"]
        view = wait(session, run)
        assert view["execution_verified"], view
    finally:
        session.close()
    return library, run


def test_exact_asymmetric_vectors_and_unverified_context(completed: Any) -> None:
    from src.shared.python.workspace.necromatcher_golf import load_research_impact_shot

    library, run = completed
    value = load_research_impact_shot(library, "replay", run, _metadata())
    assert value.shot.ball_velocity_m_s == (42.0, 4.0, -3.0)
    assert value.shot.ball_angular_velocity_rad_s == (1.0, -2.0, 3.0)
    assert value.shot.model_run_id == run and value.shot.impact_id == run
    record = value.to_record()
    assert record["qualification"] == {
        "contact": "unverified",
        "numerical": "unverified",
        "scientific": "unverified",
    }
    assert record["recorded_time_s"] == value.shot.impact_time_s
    assert record["assumptions"]["environment"]["air_density"] == 1.225
    assert record["assumptions"] and "library_root" not in json.dumps(record)
    record["assumptions"].clear()
    assert value.to_record()["assumptions"]


@pytest.mark.parametrize("fault", ["source", "qualification", "caller_time"])
def test_requested_promotion_or_foreign_metadata_rejected(
    completed: Any, fault: str
) -> None:
    from src.shared.python.golf_simulator import (
        SourceKind,
        ShotQualification,
        ContactStatus,
        NumericalStatus,
        ScientificStatus,
    )
    from src.shared.python.workspace.necromatcher_golf import load_research_impact_shot

    library, run = completed
    metadata = _metadata()
    if fault == "source":
        metadata = replace(metadata, source_kind=SourceKind.IMPORTED)
    elif fault == "caller_time":
        metadata = replace(metadata, impact_time_s=999.0)
    else:
        metadata = replace(
            metadata,
            qualification=ShotQualification(
                ContactStatus.QUALIFIED,
                NumericalStatus.CONVERGED,
                ScientificStatus.BENCHMARKED,
            ),
        )
    with pytest.raises(ValueError):
        load_research_impact_shot(library, "replay", run, metadata)


@pytest.mark.parametrize("fault", ["parent", "trajectory", "result"])
def test_stale_bundle_refused_by_canonical_download(completed: Any, fault: str) -> None:
    from src.shared.python.workspace.necromatcher_golf import load_research_impact_shot

    library, run = completed
    path = (
        library.assets["model"].path
        if fault == "parent"
        else library.root / "impact-runs" / run / "output" / (fault + ".json")
    )
    path.write_bytes(b"tampered")
    with pytest.raises((ValueError, RuntimeError)):
        load_research_impact_shot(library, "replay", run, _metadata())


def test_foreign_run_rejected(completed: Any) -> None:
    from src.shared.python.workspace.necromatcher_golf import load_research_impact_shot

    library, _ = completed
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        load_research_impact_shot(library, "replay", "f" * 32, _metadata())


@pytest.mark.parametrize("fault", ["shape", "boolean", "nonfinite"])
def test_saved_post_impact_numeric_contract(
    setup: Any, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    from src.shared.python.workspace.artifact_handoff import compute_file_sha256
    from src.shared.python.workspace.necromatcher_golf import load_research_impact_shot

    jobs, library = setup

    def malformed(
        path: Path, budget: float, cancelled: Callable, *, operation: str
    ) -> dict[str, Any]:
        _worker(path, budget, cancelled, operation=operation)
        output = path.parent / "output"
        target = output / "result.json"
        result = json.loads(target.read_bytes())
        result["impact_state"]["ball_velocity"] = {
            "shape": [1, 2],
            "boolean": [True, 2, 3],
            "nonfinite": [float("nan"), 2, 3],
        }[fault]
        target.write_text(json.dumps(result))
        return {
            "artifact_hashes": {
                p.name: compute_file_sha256(p) for p in output.iterdir()
            }
        }

    monkeypatch.setattr(jobs, "execute_native_research_worker", malformed)
    session = jobs.NativeImpactSession(library)
    try:
        run = session.submit("replay", geometry(), selection(), 30)["run_id"]
        view = wait(session, run)
        assert view["execution_verified"] is (fault != "nonfinite"), view
    finally:
        session.close()
    with pytest.raises((ValueError, RuntimeError)):
        load_research_impact_shot(library, "replay", run, _metadata())


def test_download_snapshot_mutation_cannot_replace_saved_vectors(
    completed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zipfile import ZipFile
    from src.shared.python.workspace.necromatcher_golf import load_research_impact_shot
    from src.shared.python.workspace.necromatcher_impact_jobs import NativeImpactSession

    library, run = completed
    original = NativeImpactSession.download
    calls = []

    def altered(self: Any, replay_id: str, run_id: str) -> Path:
        path = original(self, replay_id, run_id)
        calls.append(path)
        if len(calls) == 1:
            with ZipFile(path) as archive:
                records = {name: archive.read(name) for name in archive.namelist()}
            result = json.loads(records["result.json"])
            result["impact_state"]["ball_velocity"] = [99, 4, -3]
            records["result.json"] = json.dumps(result).encode()
            with ZipFile(path, "w") as archive:
                for name, raw in records.items():
                    archive.writestr(name, raw)
        return path

    monkeypatch.setattr(NativeImpactSession, "download", altered)
    with pytest.raises(ValueError, match="fresh authenticated"):
        load_research_impact_shot(library, "replay", run, _metadata())


def test_cold_bridge_import_requires_no_native_sdk() -> None:
    import subprocess
    import sys
    from src.shared.python.core import repo_python_environment
    from src.shared.python.version_info import get_repo_root

    root = get_repo_root()
    script = """
import importlib.abc,sys
class Gate(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'mujoco','opensim','PyQt6','pinocchio','pydrake'}:
   raise AssertionError('Native import '+fullname)
sys.meta_path.insert(0,Gate())
from src.shared.python.workspace.necromatcher_golf import ResearchImpactShot,load_research_impact_shot
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=repo_python_environment(root),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "fault", ["id_override", "path", "string_source", "string_qualification"]
)
def test_public_contract_rejects_context_override_and_enum_coercion(
    completed: Any, fault: str
) -> None:
    from src.shared.python.workspace.necromatcher_golf import load_research_impact_shot

    library, run = completed
    value = load_research_impact_shot(library, "replay", run, _metadata())
    if fault in {"id_override", "path"}:
        context = json.loads(value.context_json)
        context["replay_id" if fault == "id_override" else "local_path"] = "foreign"
        with pytest.raises(ValueError):
            replace(
                value,
                context_json=json.dumps(
                    context, sort_keys=True, allow_nan=False, separators=(",", ":")
                ),
            )
    else:
        metadata = _metadata()
        if fault == "string_source":
            metadata = replace(metadata, source_kind="model_contact")
        else:
            from src.shared.python.golf_simulator import ShotQualification

            metadata = replace(
                metadata,
                qualification=ShotQualification(
                    "unverified", "unverified", "unverified"
                ),
            )
        with pytest.raises(ValueError):
            load_research_impact_shot(library, "replay", run, metadata)


def test_nonidentity_aim_rotates_saved_linear_and_axial_vectors(completed: Any) -> None:
    from src.shared.python.golf_simulator import AimContext
    from src.shared.python.workspace.necromatcher_golf import load_research_impact_shot

    library, run = completed
    metadata = replace(
        _metadata(), aim_context=AimContext(((0, -1, 0), (1, 0, 0), (0, 0, 1)))
    )
    value = load_research_impact_shot(library, "replay", run, metadata)
    assert value.shot.ball_velocity_m_s == (-4.0, 42.0, -3.0)
    assert value.shot.ball_angular_velocity_rad_s == (2.0, 1.0, 3.0)
    assert value.to_record()["source_to_target_rotation"] == [
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ]


def test_incompatible_retained_source_frame_refused(completed: Any) -> None:
    from src.shared.python.workspace.necromatcher_golf import load_research_impact_shot

    library, run = completed
    value = load_research_impact_shot(library, "replay", run, _metadata())
    context = json.loads(value.context_json)
    context["framepolicy"]["source_frame_id"] = "app_xforward_yleft_zup"
    with pytest.raises(ValueError, match="frame policy"):
        replace(
            value,
            context_json=json.dumps(
                context, sort_keys=True, allow_nan=False, separators=(",", ":")
            ),
        )
