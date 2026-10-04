"""Impact fingerprints cover physics code and installed flight-kernel bytes."""

from copy import deepcopy
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def stamp_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.shared.python.workspace import necromatcher_impact_execution as owner

    physics = tmp_path / "src/shared/python/physics"
    physics.mkdir(parents=True)
    (physics / "solver.py").write_text("FIRST", encoding="utf-8")
    for relative in (
        "src/api/routes/_ball_flight_trajectory_import.py",
        "src/launchers/_shot_tracer_trajectory_import.py",
    ):
        consumer = tmp_path / relative
        consumer.parent.mkdir(parents=True, exist_ok=True)
        consumer.write_text("VALIDATOR", encoding="utf-8")
    base = {
        "source_commit": "retained-commit",
        "started_at_utc": "retained-start",
        "source_sha256": "old-source",
        "source_files": {"workspace.py": "sha256:" + "a" * 64},
        "runtime": {"python": "3.13", "tools_commit": "retained-tools"},
        "runtime_sha256": "old-runtime",
    }
    monkeypatch.setattr(owner, "fit_execution_stamp", lambda: deepcopy(base))
    monkeypatch.setattr(owner, "get_repo_root", lambda: tmp_path)

    def unavailable(_name: str):
        raise PackageNotFoundError

    monkeypatch.setattr(owner, "distribution", unavailable)
    return owner, physics, base


def test_physics_edit_changes_only_source_digest(stamp_case) -> None:
    owner, physics, base = stamp_case
    first = owner.impact_execution_stamp()
    (physics / "solver.py").write_text("SECOND", encoding="utf-8")
    second = owner.impact_execution_stamp()
    assert first["source_sha256"] != second["source_sha256"]
    assert first["runtime_sha256"] == second["runtime_sha256"]
    assert second["source_commit"] == base["source_commit"]
    assert (
        second["source_files"]["workspace.py"] == base["source_files"]["workspace.py"]
    )
    assert base["source_sha256"] == "old-source"
    assert "upstream_physics" not in base["runtime"]


def test_missing_kernel_is_recorded_without_import(stamp_case) -> None:
    owner, _, _ = stamp_case
    first = owner.impact_execution_stamp()
    assert first["runtime"]["upstream_physics"] == {
        "version": "not-installed",
        "files": {},
    }
    assert first == owner.impact_execution_stamp()


def test_same_version_changed_binary_invalidates_runtime(
    stamp_case, tmp_path, monkeypatch
) -> None:
    owner, _, _ = stamp_case
    kernel = tmp_path / "upstream_physics.pyd"
    kernel.write_bytes(b"original compiled kernel")
    installed = SimpleNamespace(
        version="2.1.3",
        files=[Path(kernel.name)],
        locate_file=lambda name: tmp_path / name,
    )
    monkeypatch.setattr(owner, "distribution", lambda _: installed)
    first = owner.impact_execution_stamp()
    kernel.write_bytes(b"different compiled kernel")
    second = owner.impact_execution_stamp()
    assert first["runtime_sha256"] != second["runtime_sha256"]
    assert first["source_sha256"] == second["source_sha256"]
    assert second["runtime"]["upstream_physics"]["version"] == "2.1.3"


def test_missing_installed_kernel_file_fails_closed(
    stamp_case, tmp_path, monkeypatch
) -> None:
    owner, _, _ = stamp_case
    installed = SimpleNamespace(
        version="2.1.3",
        files=[Path("missing.pyd")],
        locate_file=lambda name: tmp_path / name,
    )
    monkeypatch.setattr(owner, "distribution", lambda _: installed)
    with pytest.raises(FileNotFoundError):
        owner.impact_execution_stamp()
