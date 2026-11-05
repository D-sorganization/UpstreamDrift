"""Reproduce a stored run and report drift (RUN-4, Issue #14434 / Epic #14436)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.sim_run import cli
from src.sim_run.manifest import RunManifest
from src.sim_run.manifest_emitter import RunProvenance, emit_run_manifest
from src.sim_run.reproduce import ReproduceError, reproduce_engine, reproduce_run
from src.sim_run.run_registry import RunRegistry, RunRegistryError

pytestmark = pytest.mark.unit

_COMMIT = "c" * 40
_CREATED = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
_MODEL = "<mujoco model='fake'/>"
_REVISION = "sha256:" + hashlib.sha256(_MODEL.encode("utf-8")).hexdigest()
_TRAJECTORY = np.arange(12, dtype=float).reshape(4, 3)


def _provenance(seed: int = 5, engine_version: str = "1.0") -> RunProvenance:
    return RunProvenance(
        model_id="test/fake",
        revision=_REVISION,
        parameters={"duration_s": 0.01},
        input_sha256="3" * 64,
        seed=seed,
        engine_name="fake",
        engine_version=engine_version,
        ud_commit=_COMMIT,
    )


def _write_run(out: Path, provenance: RunProvenance) -> RunManifest:
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "trajectory.npy", _TRAJECTORY)
    return emit_run_manifest(
        out,
        outputs=[out / "trajectory.npy"],
        provenance=provenance,
        created_at=_CREATED,
    )


class _FakeEngine:
    """Reruns by rewriting the fixture trajectory; records the seed it got."""

    def __init__(self, engine_version: str = "1.0") -> None:
        self.engine_version = engine_version
        self.seeds: list[int] = []

    def __call__(self, manifest: RunManifest, model: str, out: Path) -> RunManifest:
        self.seeds.append(manifest.seed)
        provenance = replace(
            _provenance(manifest.seed), engine_version=self.engine_version
        )
        return _write_run(out, provenance)


@pytest.fixture
def stored(tmp_path: Path) -> tuple[RunRegistry, str]:
    registry = RunRegistry(tmp_path / "store")
    _write_run(tmp_path / "run", _provenance())
    return registry, registry.add(tmp_path / "run")


def _reproduce(stored, tmp_path: Path, engine=None, **kwargs):
    registry, run_id = stored
    return reproduce_run(
        registry,
        run_id,
        _MODEL,
        work_dir=tmp_path / "rerun",
        reproducers={"fake": engine or _FakeEngine()},
        **kwargs,
    )


def test_identical_rerun_reproduces_with_zero_drift(stored, tmp_path) -> None:
    report = _reproduce(stored, tmp_path)
    assert report.reproduced
    assert not report.engine_mismatch
    assert [(d.name, d.max_abs_diff, d.within) for d in report.outputs] == [
        ("trajectory.npy", 0.0, True)
    ]


def test_drift_beyond_tolerance_is_reported(stored, tmp_path) -> None:
    registry, run_id = stored
    np.save(tmp_path / "drifted.npy", _TRAJECTORY + 0.25)
    registry.output_path(run_id, "trajectory.npy").write_bytes(
        (tmp_path / "drifted.npy").read_bytes()
    )
    strict = _reproduce(stored, tmp_path)
    assert not strict.reproduced
    assert strict.outputs[0].max_abs_diff == pytest.approx(0.25)
    assert _reproduce(stored, tmp_path, tolerance=0.5).reproduced


def test_shape_change_is_drift_without_a_distance(stored, tmp_path) -> None:
    registry, run_id = stored
    np.save(tmp_path / "short.npy", _TRAJECTORY[:2])
    registry.output_path(run_id, "trajectory.npy").write_bytes(
        (tmp_path / "short.npy").read_bytes()
    )
    report = _reproduce(stored, tmp_path, tolerance=1e6)
    assert not report.reproduced
    assert report.outputs[0].max_abs_diff is None


def test_engine_version_mismatch_is_reported_not_ignored(stored, tmp_path) -> None:
    report = _reproduce(stored, tmp_path, engine=_FakeEngine(engine_version="2.0"))
    assert report.engine_mismatch
    assert not report.reproduced
    assert report.outputs[0].within
    assert "engine mismatch: fake 1.0 -> fake 2.0" in report.format()


def test_rerun_uses_the_stored_seed(stored, tmp_path) -> None:
    engine = _FakeEngine()
    report = _reproduce(stored, tmp_path, engine=engine)
    assert engine.seeds == [5]
    assert report.seed == 5


def test_model_that_does_not_match_revision_is_rejected(stored, tmp_path) -> None:
    registry, run_id = stored
    with pytest.raises(ReproduceError, match="revision"):
        reproduce_run(
            registry,
            run_id,
            "<mujoco model='other'/>",
            work_dir=tmp_path / "rerun",
            reproducers={"fake": _FakeEngine()},
        )


def test_engine_without_reproducer_is_rejected(stored, tmp_path) -> None:
    registry, run_id = stored
    with pytest.raises(ReproduceError, match="no reproducer for engine 'fake'"):
        reproduce_run(
            registry, run_id, _MODEL, work_dir=tmp_path / "rerun", reproducers={}
        )


def test_negative_tolerance_is_rejected(stored, tmp_path) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        _reproduce(stored, tmp_path, tolerance=-1.0)


def test_corrupt_stored_manifest_is_a_registry_error(stored, tmp_path) -> None:
    registry, run_id = stored
    path = tmp_path / "store" / "runs" / f"{run_id}.json"
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(RunRegistryError, match=run_id):
        registry.show(run_id)


def test_cli_reproduce_exit_codes(stored, tmp_path, capsys, monkeypatch) -> None:
    registry, run_id = stored
    model = tmp_path / "model.xml"
    model.write_text(_MODEL, encoding="utf-8")
    monkeypatch.setattr("src.sim_run.reproduce.REPRODUCERS", {"fake": _FakeEngine()})
    argv = [
        "reproduce",
        run_id,
        "--model",
        str(model),
        "--store",
        str(tmp_path / "store"),
    ]
    assert cli.main(argv) == 0
    assert "reproduced: yes" in capsys.readouterr().out
    monkeypatch.setattr(
        "src.sim_run.reproduce.REPRODUCERS", {"fake": _FakeEngine("2.0")}
    )
    assert cli.main(argv) == 1
    assert "reproduced: no" in capsys.readouterr().out
    assert cli.main(["reproduce", "0" * 64, "--model", str(model)] + argv[4:]) == 2
    assert "ud-sim reproduce: unknown run" in capsys.readouterr().err


def test_mujoco_fixture_run_reproduces(tmp_path) -> None:
    pytest.importorskip("mujoco")
    from src.sim_run.mujoco_run import run_mujoco

    xml = (
        "<mujoco model='tiny'><option timestep='0.002'/><worldbody>"
        "<body pos='0 0 1'><joint type='hinge' axis='0 1 0'/>"
        "<geom type='capsule' fromto='0 0 0 0 0 -0.5' size='0.03' mass='1'/>"
        "</body></worldbody></mujoco>"
    )
    run_mujoco(
        xml,
        model_id="test/tiny",
        output_dir=tmp_path / "run",
        duration=0.02,
        seed=11,
        qvel_noise=0.2,
        ud_commit=_COMMIT,
        created_at=_CREATED,
    )
    registry = RunRegistry(tmp_path / "store")
    run_id = registry.add(tmp_path / "run")
    report = reproduce_run(registry, run_id, xml, work_dir=tmp_path / "rerun")
    assert report.reproduced, report.format()
    assert report.seed == 11


def test_git_revision_model_fails_closed(tmp_path) -> None:
    registry = RunRegistry(tmp_path / "store")
    _write_run(tmp_path / "run", replace(_provenance(), revision="e" * 40))
    run_id = registry.add(tmp_path / "run")
    with pytest.raises(ReproduceError, match="git revision"):
        reproduce_run(registry, run_id, _MODEL, work_dir=tmp_path / "rerun")


def test_missing_recorded_parameter_is_a_reproduce_error(tmp_path) -> None:
    manifest = _write_run(tmp_path / "run", _provenance())
    with pytest.raises(ReproduceError, match="qvel_noise"):
        reproduce_engine(manifest, _MODEL, tmp_path / "rerun")
