"""Function budgets for frozen-evidence graphics helper entry points."""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import runpy
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "docs/research/shot_pattern_analysis/phase2/validation"
SCRIPTS = (
    PHASE / "supplement_pilot_figures_v1/render_diagnostics.py",
    PHASE / "supplement_v6_final_graphics_review_v2/publish_density_layout_child.py",
    PHASE / "supplement_v6_final_graphics_review_v2/reconcile_layout_child.py",
    PHASE / "supplement_v6_final_graphics_review_v2/verify_final_png_coverage.py",
)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_historical_graphics_main_fits_function_budget(script: Path) -> None:
    main = next(
        node
        for node in ast.parse(script.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assert main.end_lineno - main.lineno + 1 <= 100


def test_final_png_coverage_receipt_matches_frozen_source(tmp_path: Path) -> None:
    """The 33 saved review batches produce the original receipt bytes."""
    source = SCRIPTS[3].parent
    copied = tmp_path / "saved_review_inputs"
    shutil.copytree(source, copied)
    module = runpy.run_path(str(SCRIPTS[3]))
    module["main"].__globals__["ROOT"] = copied
    module["main"]()
    receipt = copied / "final_png_coverage_receipt.json"
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == (
        "ad2567554e457228c79e53e9863350581a6803f1dc09725b1f0353b75f4da744"
    )


def test_pilot_diagnostic_receipt_matches_historical_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay both source versions with inert saved-data and plot adapters."""
    archived = (
        PHASE / "refinement_v3/historical_source_snapshots/11-render_diagnostics.py.gz"
    )
    old_path = tmp_path / "historical.py"
    old_path.write_bytes(gzip.decompress(archived.read_bytes()))
    output = tmp_path / "diagnostics"
    output.mkdir()
    fake_script = output / "render_diagnostics.py"
    fake_script.write_bytes(b"fixed script provenance")
    phase = tmp_path / "docs/research/shot_pattern_analysis/phase2"
    phase.mkdir(parents=True)
    execution = {"source": "saved"}
    (phase / "source_transition_supplement.json").write_text(
        json.dumps({"new_execution": execution})
    )
    names = ("pw_primary_face", "pw_control_face", "driver_control_speed")
    for name in names:
        root = phase / "supplement_pilot_v1" / name
        cell = f"{name}_rotation2_combined2"
        directory = root / cell
        directory.mkdir(parents=True)
        config = {
            "n_shots": 32,
            "match_tolerance_m": 1e-6,
            "club": {"club_id": "driver", "shaft_lean_deg": 0.0},
            "curve_scale": 1.0,
        }
        (root / "matrix_plan.json").write_text(
            json.dumps(
                {
                    "stage": "pilot",
                    "execution": execution,
                    "cells": {cell: config},
                    "supplement_protocol": "saved-pilot",
                }
            )
        )
        for filename in ("manifest.json", "summary.json", "shots.csv.gz"):
            (directory / filename).write_bytes(b"saved input")
        (directory / "flight_samples.json").write_text(
            json.dumps({"trajectories": {"Straight": [], "Draw": [], "Fade": []}})
        )

    def fake_plot(_points, stem, **_kwargs):
        path = stem.with_suffix(".png")
        path.write_bytes(b"frozen raster")
        return {"png": path}

    def run(script: Path) -> bytes:
        module = runpy.run_path(str(script))
        globals_ = module["main"].__globals__
        globals_["__file__"] = str(fake_script)
        globals_["matched_source_snapshot"] = lambda: execution
        globals_["load_validated_cell"] = lambda *_args: {
            "status": "completed",
            "target_x_m": 195.0,
            "nominals": {},
        }
        globals_["verify_nominal_deliveries_hit_target"] = lambda *_a, **_k: None
        globals_["load_individually_supported_endpoints"] = lambda *_a: (
            {
                pattern: [[195.0, 0.0], [195.1, 0.1]]
                for pattern in ("Straight", "Draw", "Fade")
            },
            dict.fromkeys(("Straight", "Draw", "Fade"), 32),
        )
        globals_["write_academic_dispersion_figure"] = fake_plot
        globals_["write_overhead_trajectory_figure"] = fake_plot
        globals_["main"]()
        receipt = (output / "render_receipt.json").read_bytes()
        (output / "render_receipt.json").unlink()
        for name in names:
            shutil.rmtree(output / name)
        return receipt

    monkeypatch.chdir(tmp_path)
    historical = run(old_path)
    assert run(SCRIPTS[0]) == historical


def test_pilot_diagnostic_uses_native_density_bins_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #12309: the saved pilot adapter reaches the current writer."""
    import src.tools.shot_pattern_analysis.academic_plots as plots

    phase = tmp_path / "phase2"
    name = "driver_control_speed"
    root = phase / "supplement_pilot_v1" / name
    cell = "driver_rotation2_combined2"
    directory = root / cell
    directory.mkdir(parents=True)
    config = {
        "n_shots": 32,
        "match_tolerance_m": 1e-6,
        "club": {"club_id": "driver", "shaft_lean_deg": 0.0},
        "curve_scale": 1.0,
    }
    (root / "matrix_plan.json").write_text(
        json.dumps(
            {
                "stage": "pilot",
                "execution": {"id": "saved"},
                "cells": {cell: config},
                "supplement_protocol": "saved-pilot",
            }
        )
    )
    for filename in ("manifest.json", "summary.json", "shots.csv.gz"):
        (directory / filename).write_bytes(b"saved input")
    (directory / "flight_samples.json").write_text(
        json.dumps({"trajectories": {"Straight": [], "Draw": [], "Fade": []}})
    )
    observed_bins = []
    prepare = plots._prepare_density_inputs

    def capture_prepare(*args, **kwargs):
        observed_bins.append(args[3])
        return prepare(*args, **kwargs)

    def capture_formats(_figure, stem, *, dpi):
        assert dpi == 300
        stem.parent.mkdir(parents=True, exist_ok=True)
        path = stem.with_suffix(".png")
        path.write_bytes(b"native density reached export")
        return {"png": path}

    monkeypatch.setattr(plots, "_prepare_density_inputs", capture_prepare)
    monkeypatch.setattr(plots, "_save_formats", capture_formats)
    module = runpy.run_path(str(SCRIPTS[0]))
    helper = module["_render_pilot_cell"]
    globals_ = helper.__globals__
    globals_["load_validated_cell"] = lambda *_args: {
        "status": "completed",
        "target_x_m": 195.0,
        "nominals": {},
    }
    globals_["verify_nominal_deliveries_hit_target"] = lambda *_a, **_k: None
    globals_["load_individually_supported_endpoints"] = lambda *_a: (
        {
            pattern: [
                [195.0 + index * 0.05, offset + index * 0.02] for index in range(20)
            ]
            for pattern, offset in (("Straight", 0.0), ("Draw", 2.0), ("Fade", -2.0))
        },
        dict.fromkeys(("Straight", "Draw", "Fade"), 32),
    )
    globals_["write_overhead_trajectory_figure"] = lambda _paths, stem, **_kwargs: (
        capture_formats(None, stem, dpi=300)
    )
    (tmp_path / "output").mkdir()
    summary = helper(name, phase, tmp_path / "output", {"id": "saved"}, {}, {})
    assert observed_bins == [48]
    assert summary["supported_counts"] == {"Straight": 20, "Draw": 20, "Fade": 20}


def _old_main(tmp_path: Path, archive_name: str):
    archive = PHASE / "refinement_v3/historical_source_snapshots" / archive_name
    directory = tmp_path / "a/b/c/d/e/f/g"
    directory.mkdir(parents=True)
    script = directory / f"{archive_name}.py"
    script.write_bytes(gzip.decompress(archive.read_bytes()))
    return runpy.run_path(str(script))["main"]


def test_density_child_publisher_retains_guard_order(tmp_path: Path) -> None:
    old = _old_main(tmp_path, "12-publish_density_layout_child.py.gz")
    new = runpy.run_path(str(SCRIPTS[1]))["main"]
    baseline, child, repair = (tmp_path / name for name in ("base", "child", "repair"))
    for function in (old, new):
        function.__globals__.update(
            BASELINE=baseline, CHILD=child, REPAIR_SOURCE=repair
        )
    for expected in (
        "baseline and copied child publication must exist",
        "isolated repair source is missing",
    ):
        for function in (old, new):
            with pytest.raises(FileNotFoundError, match=expected):
                function()
        baseline.mkdir(exist_ok=True)
        child.mkdir(exist_ok=True)


def test_layout_reconciliation_retains_guard_order(tmp_path: Path) -> None:
    old = _old_main(tmp_path, "13-reconcile_layout_child.py.gz")
    new = runpy.run_path(str(SCRIPTS[2]))["main"]
    for function in (old, new):
        function.__globals__.update(
            HERE=tmp_path, BASELINE=tmp_path / "base", CHILD=tmp_path / "child"
        )
    for function in (old, new):
        with pytest.raises(FileNotFoundError, match="layout_repair_child_publication"):
            function()
    (tmp_path / "layout_repair_child_publication_receipt.json").write_text("{}")
    for function in (old, new):
        with pytest.raises(FileNotFoundError, match="density_layout_repair_scope"):
            function()
    (tmp_path / "density_layout_repair_scope.json").write_text('{"affected":[]}')
    for function in (old, new):
        with pytest.raises(ValueError, match="scope has duplicate figure paths"):
            function()


def test_layout_reconciliation_receipt_matches_historical_source(
    tmp_path: Path,
) -> None:
    """A 6,456-entry synthetic inventory yields byte-identical lineage."""
    baseline, child = tmp_path / "base", tmp_path / "child"
    for root in (baseline, child):
        root.mkdir()
    affected = [{"svg_path": f"/saved/density_{index:02d}.svg"} for index in range(49)]
    scope_path = tmp_path / "density_layout_repair_scope.json"
    scope_path.write_text(json.dumps({"affected": affected}))

    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    base_entries = []
    child_entries = []
    replacements = []
    for index in range(6456):
        relative = (
            f"comparisons/density_{index // 3:02d}.{('png', 'pdf', 'svg')[index % 3]}"
            if index < 147
            else f"unchanged/item_{index:04d}.png"
        )
        if index < 147:
            base_bytes, child_bytes = (
                f"base-{index}".encode(),
                f"child-{index}".encode(),
            )
            for root, data in ((baseline, base_bytes), (child, child_bytes)):
                path = root / "figures" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            replacements.append(
                {
                    "relative_path": relative,
                    "baseline_sha256": digest(base_bytes),
                    "child_sha256": digest(child_bytes),
                }
            )
            base_hash, child_hash = digest(base_bytes), digest(child_bytes)
        else:
            base_hash = child_hash = digest(relative.encode())
        base_entries.append({"relative_path": relative, "sha256": base_hash})
        child_entries.append({"relative_path": relative, "sha256": child_hash})
    for root, entries in ((baseline, base_entries), (child, child_entries)):
        (root / "REPORT_SUPPLEMENT_FIGURES.json").write_text(
            json.dumps({"figures": entries})
        )
        (root / "report.json").write_bytes(b"identical numeric report")
    base_ledger = digest((baseline / "REPORT_SUPPLEMENT_FIGURES.json").read_bytes())
    child_ledger = digest((child / "REPORT_SUPPLEMENT_FIGURES.json").read_bytes())
    pipeline = {
        "artifact_paths": {
            "report": "report.json",
            "figure_ledger": "REPORT_SUPPLEMENT_FIGURES.json",
        },
        "artifact_sha256": {
            "report": digest(b"identical numeric report"),
            "figure_ledger": base_ledger,
        },
        "figure_ledger_sha256": base_ledger,
    }
    (baseline / "REPORT_SUPPLEMENT_PIPELINE.json").write_text(json.dumps(pipeline))
    child_pipeline = json.loads(json.dumps(pipeline))
    child_pipeline["artifact_sha256"]["figure_ledger"] = child_ledger
    child_pipeline["figure_ledger_sha256"] = child_ledger
    child_pipeline["graphics_layout_repair"] = {
        "scope_sha256": digest(scope_path.read_bytes())
    }
    (child / "REPORT_SUPPLEMENT_PIPELINE.json").write_text(json.dumps(child_pipeline))
    (tmp_path / "layout_repair_child_publication_receipt.json").write_text(
        json.dumps({"replacements": replacements})
    )
    old = _old_main(tmp_path, "13-reconcile_layout_child.py.gz")
    new = runpy.run_path(str(SCRIPTS[2]))["main"]
    for function in (old, new):
        function.__globals__.update(HERE=tmp_path, BASELINE=baseline, CHILD=child)
    old()
    receipt = tmp_path / "layout_child_reconciliation_receipt.json"
    historical = receipt.read_bytes()
    receipt.unlink()
    new()
    assert receipt.read_bytes() == historical


def test_density_child_publication_matches_historical_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publish the same 49 synthetic repair triples into isolated child copies."""
    from src.tools.shot_pattern_analysis import (
        matched_figure_identity,
        matched_supplement_figures,
    )

    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    base, child, evidence = (
        tmp_path / name for name in ("baseline", "child", "evidence")
    )
    base.mkdir()
    evidence.mkdir()
    repair = tmp_path / "repair"
    source_module = (
        repair / "src/tools/shot_pattern_analysis/matched_supplement_figures.py"
    )
    source_module.parent.mkdir(parents=True)
    source_module.write_bytes(b"isolated source identity")
    source_root = tmp_path / "source"
    (source_root / "c").mkdir(parents=True)
    (source_root / "matrix_plan.json").write_text(
        json.dumps({"cells": {"c": {"n_shots": 32}}})
    )
    (source_root / "c/summary.json").write_text('{"status":"completed"}')
    source_csv = source_root / "c/shots.csv.gz"
    source_csv.write_bytes(b"saved shot rows")
    source_ref = {
        "protocol_id": "original_primary/1",
        "cell": "c",
        "source_csv_sha256": digest(source_csv.read_bytes()),
    }
    comparisons = [
        {
            "name": f"pair_{index:02d}",
            "comparison_kind": f"pair_{index:02d}",
            "target_x_m": 180.0,
            "sources": [source_ref, source_ref],
        }
        for index in range(49)
    ]
    artifacts = {
        "report": "REPORT.md",
        "inventory": "inventory.json",
        "comparisons_json": "REPORT_SUPPLEMENT_COMPARISONS.json",
        "comparisons_csv": "comparisons.csv",
        "comparison_markdown": "comparisons.md",
        "original_source_inventory": "sources.json",
        "control_aliases": "aliases.json",
        "figure_ledger": "REPORT_SUPPLEMENT_FIGURES.json",
    }
    ledger_entries = []
    affected = []
    for comparison in comparisons:
        name = comparison["name"] + "_density"
        for extension in ("png", "pdf", "svg"):
            relative = f"comparisons/{name}.{extension}"
            path = base / "figures" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"baseline {relative}".encode())
            ledger_entries.append(
                {"relative_path": relative, "sha256": digest(path.read_bytes())}
            )
            if extension == "svg":
                affected.append(
                    {
                        "svg_path": str(path),
                        "baseline_svg_sha256": digest(path.read_bytes()),
                    }
                )
    for key, relative in artifacts.items():
        path = base / relative
        if key == "comparisons_json":
            path.write_text(json.dumps({"comparisons": comparisons}))
        elif key == "figure_ledger":
            path.write_text(json.dumps({"figures": ledger_entries}))
        else:
            path.write_bytes(f"baseline {key}".encode())
    pipeline = {
        "artifact_paths": artifacts,
        "artifact_sha256": {
            key: digest((base / relative).read_bytes())
            for key, relative in artifacts.items()
        },
        "original_sources": {
            "original_primary": {"root": str(source_root)},
            "original_control": {"root": str(source_root)},
        },
        "figure_ledger_sha256": digest(
            (base / artifacts["figure_ledger"]).read_bytes()
        ),
    }
    (base / "REPORT_SUPPLEMENT_PIPELINE.json").write_text(json.dumps(pipeline))
    (evidence / "density_layout_repair_scope.json").write_text(
        json.dumps({"affected": affected})
    )

    def render(pair, staging, *, figure_kinds):
        assert pair["target_x_m"] == 180.0
        assert figure_kinds == frozenset({"density_png", "density_pdf", "density_svg"})
        result = []
        for extension in ("png", "pdf", "svg"):
            relative = f"comparisons/{pair['comparison_kind']}_density.{extension}"
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"repaired {relative}".encode())
            result.append(
                {
                    "relative_path": relative,
                    "figure_kind": f"density_{extension}",
                    "sha256": digest(path.read_bytes()),
                }
            )
        return result

    monkeypatch.setattr(
        matched_figure_identity,
        "pair_comparison_slug",
        lambda comparison: comparison["comparison_kind"],
    )
    monkeypatch.setattr(
        matched_supplement_figures, "render_selected_pair_figures", render
    )
    old = _old_main(tmp_path, "12-publish_density_layout_child.py.gz")
    new = runpy.run_path(str(SCRIPTS[1]))["main"]
    staging = evidence / "layout_repair_render_staging"
    for function in (old, new):
        function.__globals__.update(
            BASELINE=base,
            CHILD=child,
            EVIDENCE=evidence,
            STAGING=staging,
            REPAIR_SOURCE=repair,
        )
    shutil.copytree(base, child)
    assert old() == 0
    receipt_path = evidence / "layout_repair_child_publication_receipt.json"
    historical = receipt_path.read_bytes()
    shutil.rmtree(child)
    shutil.copytree(base, child)
    shutil.rmtree(staging)
    receipt_path.unlink()
    assert new() == 0
    assert receipt_path.read_bytes() == historical
