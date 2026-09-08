"""Unit tests for the declared-route-producer gate (issue #9484).

A launcher-manifest tile declaring ``web.mode: route`` advertises a web page.
Every such declaration must have a pipeline that produces what the route
serves: either the host UI build (React routes) or, for tiles embedding a
vendored Tools bundle, a workflow step that builds that bundle from the pinned
vendor tree with the mandated base path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from scripts import check_declared_route_producers as gate

# --- fixtures -----------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _manifest(tiles: list[dict]) -> str:
    import json

    return json.dumps(
        {"version": "1.0.0", "description": "test manifest", "tiles": tiles}
    )


RATE_OF_CLOSURE_TILE = {
    "id": "rate_of_closure",
    "name": "Rate of Closure Impact Explorer",
    "type": "special_app",
    "provider": "tools",
    "web": {"mode": "route", "route": "/tools/impact-explorer"},
}

MODEL_EXPLORER_TILE = {
    "id": "model_explorer",
    "name": "Model Explorer",
    "type": "special_app",
    "web": {"mode": "route", "route": "/tools/model-explorer"},
}

WORKFLOW = """\
name: Synthetic Workflow
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: {name}
        working-directory: {workdir}
        run: |
{run}
"""


def _workflow(name: str, workdir: str, run: str) -> str:
    indented = "\n".join("          " + line for line in run.splitlines())
    return WORKFLOW.format(name=name, workdir=workdir, run=indented)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    """Return a minimal synthetic repo root with manifest and workflows dirs."""
    _write(tmp_path / "src" / "config" / "launcher_manifest.json", _manifest([]))
    (tmp_path / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    return tmp_path


def _set_manifest(repo: Path, tiles: list[dict]) -> None:
    _write(repo / "src" / "config" / "launcher_manifest.json", _manifest(tiles))


# --- RED: the gate fails while no pipeline produces the bundle -----------------


def test_impact_explorer_route_without_producer_fails(repo: Path, capsys) -> None:
    _set_manifest(repo, [RATE_OF_CLOSURE_TILE])
    _write(
        repo / ".github/workflows/unrelated.yml",
        _workflow("unrelated", "ui", "npm ci"),
    )

    assert gate.main() == 1
    out = capsys.readouterr().out
    assert "/tools/impact-explorer" in out
    assert "vendor/ud-tools/src/rate_of_closure/web" in out


def test_vendored_build_without_base_flag_fails(repo: Path, capsys) -> None:
    _set_manifest(repo, [RATE_OF_CLOSURE_TILE])
    _write(
        repo / ".github/workflows/almost.yml",
        _workflow(
            "build bundle",
            "vendor/ud-tools/src/rate_of_closure/web",
            "npm ci\nnpm run build",
        ),
    )

    assert gate.main() == 1
    out = capsys.readouterr().out
    assert "--base=/impact-explorer-app/" in out


# --- GREEN: the declared routes have producers ---------------------------------


def test_vendored_producer_with_exact_authority_command_passes(repo: Path) -> None:
    _set_manifest(repo, [RATE_OF_CLOSURE_TILE])
    _write(
        repo / ".github/workflows/ci-standard.yml",
        _workflow(
            "build impact explorer",
            "vendor/ud-tools/src/rate_of_closure/web",
            "npm ci\nnpm run build -- --base=/impact-explorer-app/",
        ),
    )

    assert gate.main() == 0


def test_ui_route_produced_by_ui_build_passes(repo: Path) -> None:
    _set_manifest(repo, [MODEL_EXPLORER_TILE])
    _write(
        repo / ".github/workflows/frontend.yml",
        _workflow("build ui", "./ui", "npm ci\nnpm run build"),
    )

    assert gate.main() == 0


def test_ui_route_without_ui_build_fails(repo: Path, capsys) -> None:
    _set_manifest(repo, [MODEL_EXPLORER_TILE])
    _write(
        repo / ".github/workflows/frontend.yml",
        _workflow("test ui", "./ui", "npm run test"),
    )

    assert gate.main() == 1
    assert "/tools/model-explorer" in capsys.readouterr().out


# --- edge cases ----------------------------------------------------------------


def test_manifest_without_route_tiles_passes(repo: Path) -> None:
    native = {
        "id": "data_explorer",
        "name": "Data Explorer",
        "web": {"mode": "native-window"},
    }
    unavailable = {
        "id": "project_map",
        "name": "Project Map",
        "web": {"mode": "unavailable", "reason": "doc"},
    }
    _set_manifest(repo, [native, unavailable])
    # No workflows at all: nothing declared, nothing to produce.

    assert gate.main() == 0


def test_route_without_route_value_fails(repo: Path, capsys) -> None:
    malformed_mode = {
        "id": "broken_tile",
        "name": "Broken",
        "web": {"mode": "route"},
    }
    _set_manifest(repo, [malformed_mode])

    assert gate.main() == 1
    assert "broken_tile" in capsys.readouterr().out


def test_malformed_manifest_fails(repo: Path, capsys) -> None:
    _write(
        repo / "src" / "config" / "launcher_manifest.json",
        '{"description": "no tiles key"}',
    )

    assert gate.main() == 1
    assert "tiles" in capsys.readouterr().out
