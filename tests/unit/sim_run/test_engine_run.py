"""Other engines emit the shared run manifest (RUN-5, Issue #14435 / Epic #14436)."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.sim_run import engine_run
from src.sim_run.engine_run import EngineUnavailable, Rollout, run_engine
from src.sim_run.manifest_emitter import MANIFEST_FILENAME, sha256_file
from src.sim_run.reproduce import ReproduceError, reproduce_engine, reproduce_run
from src.sim_run.run_registry import RunRegistry
from src.sim_run.trajectory_run import TRAJECTORY_FILENAME

pytestmark = pytest.mark.unit

_COMMIT = "d" * 40
_CREATED = datetime(2026, 11, 4, 13, 0, tzinfo=timezone.utc)
_URDF = (
    '<robot name="pendulum"><link name="world"/><link name="arm"><inertial>'
    '<origin xyz="0 0 -0.25"/><mass value="1"/><inertia ixx="0.02" iyy="0.02" '
    'izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial></link>'
    '<joint name="hinge" type="revolute"><parent link="world"/><child link="arm"/>'
    '<axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="10"/>'
    "</joint></robot>"
)


def _osim_pendulum(tmp_path: Path) -> str:
    opensim = pytest.importorskip("opensim")
    model = opensim.Model()
    model.setName("pendulum")
    arm = opensim.Body(
        "arm", 1.0, opensim.Vec3(0, -0.25, 0), opensim.Inertia(0.02, 0.001, 0.02)
    )
    model.addBody(arm)
    zero = opensim.Vec3(0)
    model.addJoint(
        opensim.PinJoint("hinge", model.getGround(), zero, zero, arm, zero, zero)
    )
    model.finalizeConnections()
    path = tmp_path / "pendulum.osim"
    model.printToXML(str(path))
    return path.read_text(encoding="utf-8")


def _model(engine: str, tmp_path: Path) -> str:
    return _osim_pendulum(tmp_path) if engine == "opensim" else _URDF


def _call(engine: str, out: Path, model: str, **rollout: float) -> Any:
    return run_engine(
        engine,
        model,
        model_id="test/pendulum",
        output_dir=out,
        rollout=Rollout(duration=0.01, **rollout),
        timestep=0.002,
        ud_commit=_COMMIT,
        created_at=_CREATED,
    )


def _run(engine: str, out: Path, model: str, **rollout: float) -> Any:
    try:
        return _call(engine, out, model, **rollout)
    except EngineUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("engine", ["pinocchio", "drake", "opensim", "myosuite"])
def test_engine_run_emits_a_valid_manifest(engine: str, tmp_path: Path) -> None:
    model = _model(engine, tmp_path)
    manifest = _run(engine, tmp_path / "run", model, seed=3, qvel_noise=0.1)
    trajectory = tmp_path / "run" / TRAJECTORY_FILENAME
    assert manifest.engine.name == engine
    assert manifest.engine.version.strip()
    assert manifest.model.revision == (
        "sha256:" + hashlib.sha256(model.encode("utf-8")).hexdigest()
    )
    assert manifest.seed == 3
    assert manifest.outputs == {TRAJECTORY_FILENAME: sha256_file(trajectory)}
    rows = np.load(trajectory)
    assert rows.shape == (6, 3)
    assert rows[-1, 0] == pytest.approx(0.01)


def test_absent_engine_is_reported_with_a_reason(tmp_path, monkeypatch) -> None:
    adapter = engine_run.ENGINE_ADAPTERS["pinocchio"]
    absent = engine_run.EngineAdapter(adapter.engine_class, "no_such_xyz", "urdf")
    monkeypatch.setitem(engine_run.ENGINE_ADAPTERS, "pinocchio", absent)
    with pytest.raises(EngineUnavailable, match="pinocchio is not installed"):
        _call("pinocchio", tmp_path, _URDF)


def test_unknown_engine_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown engine 'bullet'"):
        _call("bullet", tmp_path, _URDF)


def test_engine_run_reproduces_through_the_store(tmp_path: Path) -> None:
    _run("pinocchio", tmp_path / "run", _URDF, seed=2, qvel_noise=0.2)
    registry = RunRegistry(tmp_path / "store")
    run_id = registry.add(tmp_path / "run")
    report = reproduce_run(registry, run_id, _URDF, work_dir=tmp_path / "rerun")
    assert report.reproduced, report.format()


def test_reproduce_reports_missing_parameters(tmp_path: Path) -> None:
    _run("pinocchio", tmp_path / "run", _URDF)
    registry = RunRegistry(tmp_path / "store")
    stored = registry.show(registry.add(tmp_path / "run"))
    parameters = {k: v for k, v in stored.parameters.items() if k != "timestep_s"}
    with pytest.raises(ReproduceError, match="timestep_s"):
        reproduce_engine(
            stored.model_copy(update={"parameters": parameters}), _URDF, tmp_path / "x"
        )


def test_git_revision_model_cannot_be_verified(tmp_path: Path) -> None:
    _run("pinocchio", tmp_path / "run", _URDF)
    path = tmp_path / "run" / MANIFEST_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["model"]["revision"] = "e" * 40
    path.write_text(json.dumps(data), encoding="utf-8")
    registry = RunRegistry(tmp_path / "store")
    run_id = registry.add(tmp_path / "run")
    with pytest.raises(ReproduceError, match="git revision"):
        reproduce_run(registry, run_id, _URDF, work_dir=tmp_path / "rerun")
