"""Preserve Component Review Evidence on Synthetic Saved Artifacts."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace
from typing import Any, NamedTuple

from PIL import Image, ImageDraw, ImageFont
import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/research/shot_pattern_analysis/phase2/validation"
SOURCE = BASE / "v6_output_review_harness_v1/review_batch.py"
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("boundary_ready", [False, True])
def test_missing_figure_provider_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary_ready: bool
) -> None:
    """The archived entry point resolved figure identities before any receipt."""
    import runpy
    from src.tools.shot_pattern_analysis import (
        matched_supplement_figures as figures,
        matched_supplement_pipeline as pipeline,
    )

    report = tmp_path / "report"
    report.mkdir()
    (report / "REPORT_SUPPLEMENT.json").write_text('{"unique_cells":[]}')
    output = tmp_path / "review"
    module = runpy.run_path(str(SOURCE))
    entry = module["main"]
    entry.__globals__["completion_boundary"] = lambda *_args: (
        {} if boundary_ready else None
    )
    monkeypatch.setattr(pipeline, "_report_source_hashes", dict)
    monkeypatch.setattr(
        pipeline,
        "_load_numeric_parity_reference",
        lambda _path: {"artifact_sha256": {}},
    )
    monkeypatch.setattr(
        pipeline, "_check_numeric_parity_reference", lambda *_args: None
    )
    monkeypatch.delattr(figures, "_slug")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SOURCE),
            "--repo",
            str(ROOT),
            "--report",
            str(report),
            "--completion-receipt",
            str(tmp_path / "completion.json"),
            "--start-receipt",
            str(tmp_path / "start.json"),
            "--start-sha256",
            "pinned",
            "--expected-numeric",
            str(tmp_path / "numeric.json"),
            "--output",
            str(output),
            "--group",
            "cells",
        ],
    )
    with pytest.raises(ImportError):
        entry()
    assert not output.exists()


def _load(source: str) -> dict[str, Any]:
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    ]
    namespace = {
        "Path": Path,
        "Any": Any,
        "Callable": Callable,
        "Mapping": Mapping,
        "NamedTuple": NamedTuple,
        "hashlib": hashlib,
        "html": html,
        "re": re,
        "json": json,
        "Image": Image,
        "ImageDraw": ImageDraw,
        "ImageFont": ImageFont,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    return namespace


@pytest.mark.parametrize("changed", [False, True])
def test_component_capture_preserves_artifact_hashes_contact_pixels_and_errors(
    tmp_path: Path, changed: bool
) -> None:
    from src.tools.shot_pattern_analysis.matched_supplement_figures import _slug

    args = SimpleNamespace(
        report=tmp_path / "report",
        output=tmp_path / "review",
        group="cells",
        offset=0,
        limit=1,
    )
    args.output.mkdir()
    directory = args.report / "figures/cells"
    directory.mkdir(parents=True)
    suffixes = ["overhead.png", "overhead.pdf", "overhead.svg"]
    stem = _slug("synthetic_cell")
    png = directory / f"{stem}_overhead.png"
    Image.new("RGB", (50, 80), "red").save(png)
    (directory / f"{stem}_overhead.pdf").write_bytes(b"synthetic PDF presence fixture")
    (directory / f"{stem}_overhead.svg").write_text(
        "<svg><!-- Synthetic &amp; Caption --></svg>"
    )
    bound = {
        str(p.relative_to(args.report)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.iterdir()
    }
    if changed:
        png.write_bytes(b"changed")
    values = {
        "args": args,
        "records": [{"cell": "synthetic_cell"}],
        "suffixes": suffixes,
        "boundary": {"asset_sha256": bound},
        "_slug": _slug,
        "reviewed": [],
    }
    archived = next(
        (BASE / "refinement_v3/historical_source_snapshots").glob(
            "*-review_batch.py.gz"
        )
    )
    old = gzip.decompress(archived.read_bytes()).decode()
    namespace = _load(old)
    namespace.update(values)
    main = next(
        n
        for n in ast.parse(old).body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    start = next(
        i
        for i, n in enumerate(main.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(x, ast.Name) and x.id == "font" for x in n.targets)
    )
    code = compile(
        ast.Module(body=main.body[start : start + 2], type_ignores=[]),
        "archived-review",
        "exec",
    )
    current = _load(SOURCE.read_text())
    identity = current["_figure_identity_provider"]()
    if changed:
        with pytest.raises(ValueError, match="completed artifact changed"):
            exec(code, namespace)
        with pytest.raises(ValueError, match="completed artifact changed"):
            current["_review_components"](
                args, values["records"], suffixes, values["boundary"], identity
            )
    else:
        exec(code, namespace)
        expected = namespace["reviewed"]
        before = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.output.iterdir()
        }
        shutil.rmtree(args.output)
        args.output.mkdir()
        assert (
            current["_review_components"](
                args, values["records"], suffixes, values["boundary"], identity
            )
            == expected
        )
        assert {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.output.iterdir()
        } == before


@pytest.mark.parametrize(
    "group,kind", [("cells", None), ("pairs", None), ("pairs", "curve_strength")]
)
def test_selection_order_and_export_kinds_match_archive(
    tmp_path: Path, group: str, kind: str | None
) -> None:
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        pair_comparison_identity,
    )

    records = [
        {"source_protocol": "p", "cell": "b"},
        {"source_protocol": "p", "cell": "a"},
    ]
    # Pair ordering uses the actual canonical identity provider.
    pairs = [
        {
            "comparison_kind": k,
            "target_x_m": 100.0,
            "source_protocols": ["p", "p"],
            "sources": [
                {"protocol_id": "p", "cell": c},
                {"protocol_id": "p", "cell": c + "2"},
            ],
        }
        for k, c in [("curve_strength", "b"), ("across_lean", "a")]
    ]
    (tmp_path / "REPORT_SUPPLEMENT.json").write_text(
        json.dumps({"unique_cells": records})
    )
    (tmp_path / "REPORT_SUPPLEMENT_COMPARISONS.json").write_text(
        json.dumps({"comparisons": pairs})
    )
    args = SimpleNamespace(report=tmp_path, group=group, kind=kind)
    archived = next(
        (BASE / "refinement_v3/historical_source_snapshots").glob(
            "*-review_batch.py.gz"
        )
    )
    old = gzip.decompress(archived.read_bytes()).decode()
    namespace = _load(old)
    namespace.update(args=args, pair_comparison_identity=pair_comparison_identity)
    main = next(
        n
        for n in ast.parse(old).body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    selection = next(
        n
        for n in main.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "args.group == 'pairs'"
    )
    exec(
        compile(
            ast.Module(body=[selection], type_ignores=[]), "archived-selection", "exec"
        ),
        namespace,
    )
    current = _load(SOURCE.read_text())
    assert current["_selected_records"](
        args, current["_figure_identity_provider"]()
    ) == (
        namespace["records"],
        namespace["suffixes"],
    )
