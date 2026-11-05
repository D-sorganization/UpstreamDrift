"""Content-addressed run store and lookup CLI (RUN-3, Issue #14433 / Epic #14436)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from src.sim_run import cli
from src.sim_run.manifest_emitter import (
    MANIFEST_FILENAME,
    RunProvenance,
    emit_run_manifest,
    sha256_json,
)
from src.sim_run.run_registry import (
    STORE_ENV,
    RunRegistry,
    RunRegistryError,
    default_store_root,
)

pytestmark = pytest.mark.unit

_COMMIT = "b" * 40


def _provenance(seed: int = 7) -> RunProvenance:
    return RunProvenance(
        model_id="test/tiny_pendulum",
        revision="sha256:" + "1" * 64,
        parameters={"duration": 0.05, "qvel_noise": 0.1},
        input_sha256="2" * 64,
        seed=seed,
        engine_name="mujoco",
        engine_version="3.2.0",
        ud_commit=_COMMIT,
    )


def _run_dir(tmp_path: Path, name: str = "run", seed: int = 7) -> Path:
    run_dir = tmp_path / name
    (run_dir / "plots").mkdir(parents=True)
    (run_dir / "trajectory.npy").write_bytes(b"trajectory-bytes")
    (run_dir / "plots" / "q.svg").write_bytes(b"<svg/>")
    emit_run_manifest(
        run_dir,
        outputs=[run_dir / "trajectory.npy", run_dir / "plots" / "q.svg"],
        provenance=_provenance(seed),
        created_at=datetime(2026, 11, 4, 3, seed, tzinfo=timezone.utc),
    )
    return run_dir


def _manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def test_run_id_is_the_canonical_manifest_hash(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    run_id = RunRegistry(tmp_path / "store").add(run_dir)
    assert run_id == sha256_json(_manifest(run_dir))


def test_add_is_idempotent_and_stable_across_stores(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    first = RunRegistry(tmp_path / "a").add(run_dir)
    assert RunRegistry(tmp_path / "a").add(run_dir) == first
    assert RunRegistry(tmp_path / "b").add(run_dir) == first


def test_show_returns_manifest_and_outputs_by_hash(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "store")
    run_id = registry.add(_run_dir(tmp_path))
    manifest = registry.show(run_id)
    assert manifest.seed == 7
    assert manifest.engine.version == "3.2.0"
    path = registry.output_path(run_id, "plots/q.svg")
    assert path.read_bytes() == b"<svg/>"
    assert path.name == manifest.outputs["plots/q.svg"]


def test_list_orders_runs_by_creation_time(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "store")
    late = registry.add(_run_dir(tmp_path, "late", seed=9))
    early = registry.add(_run_dir(tmp_path, "early", seed=3))
    assert [entry.run_id for entry in registry.list_runs()] == [early, late]


def test_identical_outputs_are_stored_once(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "store")
    registry.add(_run_dir(tmp_path, "one", seed=1))
    registry.add(_run_dir(tmp_path, "two", seed=2))
    assert len(list((tmp_path / "store" / "objects").iterdir())) == 2


def test_add_rejects_an_output_that_no_longer_matches(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    (run_dir / "trajectory.npy").write_bytes(b"edited")
    with pytest.raises(RunRegistryError, match="trajectory.npy"):
        RunRegistry(tmp_path / "store").add(run_dir)
    assert not (tmp_path / "store" / "runs").exists()


def test_add_rejects_a_directory_without_a_valid_manifest(tmp_path: Path) -> None:
    with pytest.raises(RunRegistryError, match=MANIFEST_FILENAME):
        RunRegistry(tmp_path / "store").add(tmp_path)


@pytest.mark.parametrize("run_id", ["abc", "../" + "a" * 61, "A" * 64])
def test_show_rejects_malformed_run_id(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        RunRegistry(tmp_path / "store").show(run_id)


def test_show_unknown_run_raises(tmp_path: Path) -> None:
    with pytest.raises(RunRegistryError, match="unknown run"):
        RunRegistry(tmp_path / "store").show("c" * 64)


def test_store_root_comes_from_env_not_a_committed_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "env-store"))
    assert default_store_root() == tmp_path / "env-store"
    monkeypatch.delenv(STORE_ENV)
    assert default_store_root().is_relative_to(Path.home())


def test_cli_add_list_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = ["--store", str(tmp_path / "store")]
    assert cli.main(["runs", *store, "add", str(_run_dir(tmp_path))]) == 0
    run_id = capsys.readouterr().out.strip()
    assert cli.main(["runs", *store, "list"]) == 0
    assert run_id in capsys.readouterr().out
    assert cli.main(["runs", *store, "show", run_id]) == 0
    shown = capsys.readouterr().out
    for text in ("test/tiny_pendulum", "sha256:" + "1" * 64, "qvel_noise", "seed: 7"):
        assert text in shown
    assert "mujoco 3.2.0" in shown


def test_cli_show_unknown_run_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["runs", "--store", str(tmp_path), "show", "d" * 64])
    assert code == 1
    assert "unknown run" in capsys.readouterr().err
