"""Preserve Renderer Evidence Helpers Without Running Historical Publication Jobs."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pytest

from tests.research.test_historical_fixture_security import read_fixture, verify_source

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/research/shot_pattern_analysis/phase2/validation"
SOURCE = BASE / "selected_renderer_representative_v1/qualify.py"
pytestmark = pytest.mark.unit


def _namespace(source: str) -> dict[str, Any]:
    verify_source("renderer", source)
    nodes: list[ast.stmt] = [
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
    ]
    namespace: dict[str, Any] = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "np": np,
        "dataclasses": dataclasses,
        "matplotlib": matplotlib,
    }
    # Trusted fixture AST: fixed path/digests verified before extraction; no external code.
    # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    return namespace


def _archive() -> str:
    return read_fixture("renderer", archived=True)


def test_figure_capture_matches_archived_axis_and_text_evidence() -> None:
    old = _archive()
    after = _namespace(read_fixture("renderer"))
    fig = Figure(figsize=(4, 3))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot([1, 2], [3, 5])
    ax.scatter([1, 2], [2, 4], c=[0.1, 0.9])
    ax.set(xlabel="Lateral (m)", ylabel="Downrange (m)", title="Shot Pattern")
    namespace = _namespace(old)
    namespace.update(fig=fig, current={}, kind="density")
    save = next(
        n
        for n in ast.walk(ast.parse(old))
        if isinstance(n, ast.FunctionDef) and n.name == "save"
    )
    block = next(n for n in save.body if isinstance(n, ast.If))
    # The first statement updates the enclosing de-duplication set, then capture begins.
    # Trusted fixture AST: fixed path/digests verified before extraction; no external code.
    # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
    exec(
        compile(
            ast.Module(body=block.body[1:], type_ignores=[]), "archived-capture", "exec"
        ),
        namespace,
    )
    assert after["_capture_figure"](fig) == namespace["current"]["figures"]["density"]


@pytest.mark.parametrize("bad_hash", [False, True])
def test_source_snapshots_preserve_bound_hashes_and_failure(
    tmp_path: Path, bad_hash: bool
) -> None:
    root = tmp_path / "source"
    directory = root / "cell"
    directory.mkdir(parents=True)
    (root / "matrix_plan.json").write_text("{}")
    (directory / "summary.json").write_text(
        json.dumps({"status": "completed", "config": {"n_shots": 10}})
    )
    (directory / "shots.csv.gz").write_bytes(b"synthetic archived bytes")

    def digest(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    files = {
        name: digest(directory / name) for name in ["summary.json", "shots.csv.gz"]
    }
    (directory / "manifest.json").write_text(json.dumps({"files_sha256": files}))
    row = {
        "sources": [
            {
                "protocol_id": "p",
                "cell": "cell",
                "source_cell_manifest_sha256": digest(directory / "manifest.json"),
                "source_plan_sha256": digest(root / "matrix_plan.json"),
            }
        ]
    }
    if bad_hash:
        (directory / "shots.csv.gz").write_bytes(b"changed")
    old = _archive()
    namespace = _namespace(old)
    namespace.update(row=row, roots={"p": root}, inputs={}, snapshots={})
    prepare = next(
        n
        for n in ast.parse(old).body
        if isinstance(n, ast.FunctionDef) and n.name == "prepare"
    )
    loop = next(
        n
        for n in ast.walk(prepare)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id == "proof"
    )
    current = _namespace(read_fixture("renderer"))
    inputs: dict[str, str] = {}
    if bad_hash:
        with pytest.raises(AssertionError):
            # Trusted fixture AST: fixed path/digests verified before extraction; no external code.
            # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
            exec(
                compile(
                    ast.Module(body=[loop], type_ignores=[]),
                    "archived-snapshot",
                    "exec",
                ),
                namespace,
            )
        with pytest.raises(AssertionError):
            current["_source_snapshots"](row, {"p": root}, inputs)
    else:
        # Trusted fixture AST: fixed path/digests verified before extraction; no external code.
        # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
        exec(
            compile(
                ast.Module(body=[loop], type_ignores=[]), "archived-snapshot", "exec"
            ),
            namespace,
        )
        assert (
            current["_source_snapshots"](row, {"p": root}, inputs)
            == namespace["snapshots"]
        )
    assert inputs == namespace["inputs"]
