"""Atomic, source-bound publication of matched delivery experiments."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import platform
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from src.shared.python.physics.incremental_contact import contact_backend_info

from .academic_plots import (
    write_academic_dispersion_figure,
    write_overhead_trajectory_figure,
)
from .core import AnalysisCancelled
from .matched_physics import UnsupportedShot
from .matched_runner import (
    ExperimentConfig,
    MatchedAnalysisResult,
    run_matched_analysis,
)
from .nominal_matching import NominalMatchError
from .provenance import assert_source_unchanged, source_snapshot
from .scoring import build_broadie_approx_baseline


def matched_source_snapshot() -> dict:
    """Extend the canonical native/flight/scoring snapshot with phase-two sources."""
    result = source_snapshot()
    root = Path(__file__).resolve().parents[3]
    package = Path(__file__).parent
    files = [
        package / name
        for name in (
            "club_geometry.py",
            "matched_config.py",
            "matched_physics.py",
            "matched_runner.py",
            "matched_export.py",
            "matched_diagnostics.py",
            "matched_report_plots.py",
            "noise.py",
            "nominal_matching.py",
            "matched_supplement.py",
            "density.py",
            "academic_plots.py",
        )
    ]
    files.extend(sorted((root / "src/shared/python/physics").glob("*contact*.py")))
    matrix = package / "matched_matrix.py"
    if matrix.is_file():
        files.append(matrix)
    result["source_sha256"].update(
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        }
    )
    result["contact_backend"] = contact_backend_info()
    result["numerical_runtime"] = {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
    }
    return result


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise AnalysisCancelled("matched export cancelled before publication")


def _write_manifest(output_dir: Path) -> None:
    hashes = {
        str(p.relative_to(output_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output_dir.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    _write_json(
        output_dir / "manifest.json",
        {"schema": "matched-shot-study/1", "files_sha256": hashes},
    )


def _contact_execution(config: ExperimentConfig) -> dict[str, str | int]:
    """Record the selected numerical law separately from available auto backend."""
    club = config.club
    if club.contact_law == "maximum_dissipation_newton":
        selected = "not_applicable"
    elif club.contact_backend == "auto":
        selected = str(contact_backend_info()["active_backend"])
    else:
        selected = club.contact_backend
    return {
        "contact_law": club.contact_law,
        "requested_backend": club.contact_backend,
        "selected_backend": selected,
        "steps_per_compression": club.steps_per_compression,
    }


def _write_figures(
    result: MatchedAnalysisResult,
    output_dir: Path,
    cancelled: Callable[[], bool] | None,
) -> dict:
    points = {
        name: [
            (r.carry_x_m, r.carry_y_m)
            for r in result.shots
            if r.pattern == name and r.status == "supported"
        ]
        for name in result.nominals
    }
    if any(not values for values in points.values()):
        return {
            "status": "unavailable",
            "reason": "one_or_more_patterns_have_no_supported_endpoints",
        }
    cfg = result.config
    sd = cfg.noise.standard_deviations
    metadata = {
        "static_loft_deg": cfg.club.loft_deg,
        "lean_deg": cfg.club.shaft_lean_deg,
        "rotation_SD_deg": sd[0],
        "path_SD_deg": sd[1],
        "speed_SD_mph": sd[2] / 0.44704,
        "strike_SD_mm": f"{sd[3] * 1000:g}/{sd[4] * 1000:g}",
        "max_match_error_m": max(n.residual_m for n in result.nominals.values()),
        "valid_shots_by_pattern": {n: len(v) for n, v in points.items()},
        "total_shots_by_pattern": dict.fromkeys(points, cfg.n_shots),
    }
    scenario = f"{cfg.club.club_id.replace('_', ' ').title()} | Curve ×{cfg.curve_scale:g} | Geometric Model"
    _check_cancelled(cancelled)
    density = write_academic_dispersion_figure(
        points,
        output_dir / "density",
        scenario=scenario,
        metadata=metadata,
        target_point_m=(result.target_x_m, 0),
        bins=48,
    )
    _check_cancelled(cancelled)
    overhead = write_overhead_trajectory_figure(
        result.flight_samples,
        output_dir / "overhead",
        scenario=scenario,
        metadata=metadata,
        target_x_m=result.target_x_m,
    )
    _check_cancelled(cancelled)
    return {
        "status": "available",
        "density": {k: p.name for k, p in density.items()},
        "overhead": {k: p.name for k, p in overhead.items()},
    }


def export_matched_analysis(
    result: MatchedAnalysisResult,
    output_dir: Path,
    *,
    execution: dict,
    figures: bool = True,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Serialize a completed result into staging; caller supplies pre-run evidence.

    For atomic public output use execute_and_export. This lower-level function
    refuses existing completed/data files and verifies source identity again.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(p.name != "run_start.json" for p in output_dir.iterdir()):
        raise FileExistsError("export destination already contains data")
    _check_cancelled(cancelled)
    assert_source_unchanged(execution, matched_source_snapshot())
    _write_json(
        output_dir / "run_start.json",
        {"config": asdict(result.config), "execution": execution},
    )
    # Fixed gzip timestamp preserves byte hashes across identical exports.
    with (output_dir / "shots.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                first = asdict(result.shots[0])
                writer = csv.DictWriter(text, fieldnames=list(first))
                writer.writeheader()
                for row in result.shots:
                    writer.writerow(asdict(row))
    summary = asdict(result)
    del summary["shots"]
    del summary["flight_samples"]
    summary["status"] = "completed"
    summary["population"] = (
        "Every sampled input retained; unsupported endpoints are null, never resampled."
    )
    summary["aim_adjustment_applied"] = False
    summary["figures"] = (
        _write_figures(result, output_dir, cancelled)
        if figures
        else {"status": "not_requested"}
    )
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "flight_samples.json",
        {"trajectories": result.flight_samples, "target_x_m": result.target_x_m},
    )
    baseline = build_broadie_approx_baseline()
    _write_json(output_dir / "scoring_baseline.json", baseline.model_dump(mode="json"))
    _write_json(
        output_dir / "receipt.json",
        {
            "execution": execution,
            "source_unchanged_during_run": True,
            "model_qualification": "conditional_geometric_research_no_empirical_accuracy_claim",
            "contact_execution": _contact_execution(result.config),
            "shot_rows": len(result.shots),
            "all_inputs_retained": True,
        },
    )
    _check_cancelled(cancelled)
    assert_source_unchanged(execution, matched_source_snapshot())
    _write_manifest(output_dir)
    return output_dir


def execute_and_export(
    request: ExperimentConfig,
    output_dir: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    figures: bool = True,
) -> Path:
    """Run from a pre-recorded source identity and publish only a complete bundle.

    Nominal-domain failures produce explicit unavailable cells. Numerical errors,
    cancellation, or source drift leave no final published directory.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("output destination already contains data")
    _check_cancelled(cancelled)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    execution = matched_source_snapshot()
    with TemporaryDirectory(
        prefix=f".{output_dir.name}.pending-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        _write_json(
            staging / "run_start.json",
            {"config": asdict(request), "execution": execution},
        )
        try:
            result = run_matched_analysis(
                request, progress=progress, cancelled=cancelled
            )
        except (UnsupportedShot, NominalMatchError) as exc:
            assert_source_unchanged(execution, matched_source_snapshot())
            _write_json(
                staging / "summary.json",
                {
                    "status": "unavailable",
                    "reason": str(exc),
                    "config": asdict(request),
                    "shot_rows": 0,
                    "execution": execution,
                    "interpretation": "No Monte Carlo sampling occurred; nominal setup is unavailable under the stated model/search bounds.",
                },
            )
            _write_manifest(staging)
        else:
            assert_source_unchanged(execution, matched_source_snapshot())
            export_matched_analysis(
                result,
                staging,
                execution=execution,
                figures=figures,
                cancelled=cancelled,
            )
        _check_cancelled(cancelled)
        os.replace(staging, output_dir)
    return output_dir
