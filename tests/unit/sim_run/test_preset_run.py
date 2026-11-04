"""Release presets run through the run store (RUN-6, Issue #14518 / Epic #14436)."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from src.sim_run import preset_run
from src.sim_run.manifest_emitter import sha256_file
from src.sim_run.preset_run import PresetRunError, PresetSpec, run_preset
from src.sim_run.reproduce import reproduce_run
from src.sim_run.run_registry import RunRegistry

pytestmark = pytest.mark.unit

_COMMIT = "a" * 40
_CREATED = datetime(2026, 11, 4, 14, 0, tzinfo=timezone.utc)
_STUDY = """
import json, pathlib, sys
out = pathlib.Path("release")
(out / "data").mkdir(parents=True, exist_ok=True)
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
(out / "data" / "study.json").write_text(json.dumps({"peak": 2.0 * scale}))
"""
_FIGURE = """
import json, pathlib
out = pathlib.Path("release")
peak = json.loads((out / "data" / "study.json").read_text())["peak"]
(out / "figures").mkdir(exist_ok=True)
(out / "figures" / "fig_study.svg").write_text(f"<svg data-peak='{peak}'/>")
"""


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repo root with a two-step preset (study, then its figure)."""
    package = tmp_path / "repo" / "fakepreset"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "study.py").write_text(_STUDY)
    (package / "figure.py").write_text(_FIGURE)
    (tmp_path / "repo" / "release").mkdir()
    (tmp_path / "repo" / "release" / "README.md").write_text("untouched input")
    monkeypatch.syspath_prepend(str(tmp_path / "repo"))
    return tmp_path / "repo"


def _spec(**overrides: object) -> PresetSpec:
    fields: dict[str, object] = {
        "name": "fake_study",
        "tier": "planar_test",
        "commands": (("fakepreset.study",), ("fakepreset.figure",)),
        "watch": ("release",),
    }
    fields.update(overrides)
    return PresetSpec(**fields)  # type: ignore[arg-type]


def _run(root: Path, out: Path, spec: PresetSpec | None = None):
    return run_preset(
        spec or _spec(),
        repo_root=root,
        output_dir=out,
        ud_commit=_COMMIT,
        created_at=_CREATED,
    )


def test_preset_run_records_the_files_it_wrote(root: Path, tmp_path: Path) -> None:
    manifest = _run(root, tmp_path / "run")
    names = {"release/data/study.json", "release/figures/fig_study.svg"}
    assert set(manifest.outputs) == names
    for name in names:
        assert manifest.outputs[name] == sha256_file(root / name)
        assert (tmp_path / "run" / name).read_bytes() == (root / name).read_bytes()
    assert manifest.model.model_id == "release-preset/fake_study"
    spec_text = _spec().to_text()
    digest = hashlib.sha256(spec_text.encode("utf-8")).hexdigest()
    assert manifest.model.revision == "sha256:" + digest
    assert manifest.engine.name == preset_run.PRESET_ENGINE
    assert manifest.parameters["tier"] == "planar_test"
    assert manifest.parameters["commands"] == [
        ["fakepreset.study"],
        ["fakepreset.figure"],
    ]


def test_input_digest_covers_unwritten_watched_files(root, tmp_path) -> None:
    first = _run(root, tmp_path / "a").input_data.sha256
    assert _run(root, tmp_path / "b").input_data.sha256 == first
    (root / "release" / "README.md").write_text("changed input")
    assert _run(root, tmp_path / "c").input_data.sha256 != first


def test_spec_text_round_trips() -> None:
    spec = _spec(commands=(("fakepreset.study", "write"),))
    assert PresetSpec.from_text(spec.to_text()) == spec


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": "bad name"}, "name"),
        ({"commands": ()}, "command"),
        ({"commands": (("os; rm",),)}, "module"),
        ({"watch": ("../outside",)}, "watch"),
        ({"watch": ()}, "watch"),
    ],
)
def test_malformed_spec_is_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _spec(**overrides)


def test_failing_command_is_reported(root: Path, tmp_path: Path) -> None:
    (root / "fakepreset" / "broken.py").write_text("raise SystemExit(3)")
    spec = _spec(commands=(("fakepreset.broken",),))
    with pytest.raises(PresetRunError, match="fakepreset.broken exited 3"):
        _run(root, tmp_path / "run", spec)


def test_preset_that_writes_nothing_is_rejected(root: Path, tmp_path: Path) -> None:
    (root / "fakepreset" / "noop.py").write_text("")
    spec = _spec(commands=(("fakepreset.noop",),))
    with pytest.raises(PresetRunError, match="wrote no watched files"):
        _run(root, tmp_path / "run", spec)


def test_preset_run_reproduces_through_the_store(root, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(preset_run, "REPO_ROOT", root)
    _run(root, tmp_path / "run")
    registry = RunRegistry(tmp_path / "store")
    run_id = registry.add(tmp_path / "run")
    report = reproduce_run(
        registry,
        run_id,
        _spec().to_text(),
        work_dir=tmp_path / "rerun",
        reproducers={preset_run.PRESET_ENGINE: preset_run.reproduce_preset},
    )
    assert report.reproduced, report.format()


def test_changed_preset_output_is_drift(root, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(preset_run, "REPO_ROOT", root)
    _run(root, tmp_path / "run")
    registry = RunRegistry(tmp_path / "store")
    run_id = registry.add(tmp_path / "run")
    study = root / "fakepreset" / "study.py"
    stat = study.stat()
    study.write_text(_STUDY.replace("2.0", "3.0"))
    # Same size and mtime: a cached bytecode file would hide the edit.
    os.utime(study, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    report = reproduce_run(
        registry,
        run_id,
        _spec().to_text(),
        work_dir=tmp_path / "rerun",
        reproducers={preset_run.PRESET_ENGINE: preset_run.reproduce_preset},
    )
    assert not report.reproduced
    assert "DRIFT" in report.format()
    assert json.loads((root / "release" / "data" / "study.json").read_text()) == {
        "peak": 3.0
    }


def test_preset_runs_with_this_interpreter(root, tmp_path, monkeypatch) -> None:
    seen: list[list[str]] = []
    real = preset_run.subprocess.run

    def spy(args, **kwargs):
        seen.append(list(args))
        return real(args, **kwargs)

    monkeypatch.setattr(preset_run.subprocess, "run", spy)
    _run(root, tmp_path / "run")
    assert seen[0][0] == sys.executable
    assert seen[0][1] == "-X" and seen[0][2].startswith("pycache_prefix=")
    assert seen[0][3:] == ["-m", "fakepreset.study"]
    assert not list(root.rglob("__pycache__"))


def test_command_timeout_is_reported(root, tmp_path, monkeypatch) -> None:
    (root / "fakepreset" / "slow.py").write_text("import time; time.sleep(5)")
    monkeypatch.setattr(preset_run, "COMMAND_TIMEOUT_S", 0.5)
    spec = _spec(commands=(("fakepreset.slow",),))
    with pytest.raises(PresetRunError, match="fakepreset.slow timed out"):
        _run(root, tmp_path / "run", spec)
