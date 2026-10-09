"""Render integrity-checked matched-study figures without rerunning physics.

Final runs include per-cell endpoint-density and raw overhead figures for the
shaft-only and combined-noise profiles, shared-scale matrix heatmaps for every
noise profile, an explicit feasibility ledger, and a Facebook-ready PNG
selection. Pilot previews require an explicit flag and a preview-named output.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

PATTERNS = ("Straight", "Draw", "Fade")
_COMBINED_NOISE = (0.5, 0.89408, 0.00635, 0.00635, 0.0, 0.0)


def _detail_profile(standard_deviations: list[float]) -> str | None:
    """Name only the predeclared shaft-only and combined-noise profiles."""
    if len(standard_deviations) != 7:
        return None
    if standard_deviations[0] not in (1.0, 2.0):
        return None
    if all(math.isclose(value, 0.0, abs_tol=1e-10) for value in standard_deviations[1:]):
        return "shaft_only"
    if all(
        math.isclose(actual, expected, abs_tol=1e-8)
        for actual, expected in zip(standard_deviations[1:], _COMBINED_NOISE, strict=True)
    ):
        return "combined2"
    return None


def _club_config(cell: dict[str, Any]) -> dict[str, Any]:
    config = cell.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("club"), dict):
        raise ValueError(f"cell {cell.get('cell')} has no club configuration")
    return config["club"]


def _facebook_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    config = cell["config"]
    club = _club_config(cell)
    noise = config.get("noise", {})
    return (
        club.get("club_id"),
        tuple(noise.get("standard_deviations", ())),
        json.dumps(noise.get("correlation"), sort_keys=True),
        config.get("curve_scale"),
        config.get("delivery_mode"),
        club.get("contact_law"),
    )


def _facebook_selection(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select lean-0 social comparisons, using labeled PW lean-5 fallbacks."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        if _detail_profile(cell.get("config", {}).get("noise", {}).get("standard_deviations", [])) is None:
            continue
        groups[_facebook_key(cell)].append(cell)

    selected: list[dict[str, Any]] = []
    for group in groups.values():
        club_id = _club_config(group[0]).get("club_id")
        lean_zero = next(
            (cell for cell in group if math.isclose(_club_config(cell).get("shaft_lean_deg", math.nan), 0.0)),
            None,
        )
        if lean_zero is None:
            continue
        choice = lean_zero
        note = "nominal lean 0°"
        if lean_zero.get("status") != "completed" and club_id == "pitching_wedge":
            lean_five = next(
                (
                    cell
                    for cell in group
                    if math.isclose(_club_config(cell).get("shaft_lean_deg", math.nan), 5.0)
                    and cell.get("status") == "completed"
                ),
                None,
            )
            if lean_five is not None:
                choice = lean_five
                note = "lean 0° unavailable; showing lean 5°"
        selected.append({**choice, "selection_note": note})
    return sorted(selected, key=lambda cell: cell["cell"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _stratum_key(cell: dict[str, Any]) -> tuple[str, ...]:
    config = cell["config"]
    club = _club_config(cell)
    noise = config.get("noise", {})
    return (
        str(config.get("delivery_mode", "unspecified")),
        str(club.get("contact_law", "unspecified")),
        _json_hash(noise.get("standard_deviations", [])),
        _json_hash(noise.get("correlation")),
        str(config.get("n_shots", "unspecified")),
        str(config.get("seed", "unspecified")),
    )


def _stratum_name(group: list[dict[str, Any]]) -> str:
    cell = group[0]
    config = cell["config"]
    law = _club_config(cell).get("contact_law", "unknown")
    noise = config.get("noise", {}).get("standard_deviations", [])
    labels = ("rot", "path", "speed", "contactx", "contacty", "lean", "attack")
    noise_label = "_".join(
        f"{label}{float(value):g}"
        for label, value in zip(labels, noise, strict=False)
    )
    return (
        f"{config.get('delivery_mode', 'mode-unknown')}__{law}__"
        f"{noise_label}__corr-{_json_hash(config.get('noise', {}).get('correlation'))[:8]}__"
        f"n{config.get('n_shots', 'unknown')}"
    )


def _heatmap_cells(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt report cell identifiers to the heatmap consumer's scenario key."""
    return [{**cell, "scenario": cell["cell"]} for cell in group]


def _comparison_group_key(cell: dict[str, Any]) -> tuple[str, ...] | None:
    config = cell["config"]
    club = _club_config(cell)
    if not math.isclose(float(club.get("shaft_lean_deg", math.nan)), 0.0, abs_tol=1e-9):
        return None
    noise = config.get("noise", {})
    standard_deviations = noise.get("standard_deviations", [])
    profile = _detail_profile(standard_deviations)
    if profile is None:
        return None
    return (
        str(club.get("club_id")),
        profile,
        str(config.get("delivery_mode", "unspecified")),
        str(club.get("contact_law", "unspecified")),
        _json_hash(club),
        _json_hash(standard_deviations[1:]),
        _json_hash(noise.get("correlation")),
        str(config.get("n_shots", "unspecified")),
        str(config.get("seed", "unspecified")),
        str(config.get("target_radius_m", "unspecified")),
        str(config.get("match_tolerance_m", "unspecified")),
    )


def _comparison_pair_specs(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create lean-zero, shared-target SD and curve pairs per club/profile."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        if (key := _comparison_group_key(cell)) is not None:
            groups[key].append(cell)

    pairs: list[dict[str, Any]] = []
    for key, group in groups.items():
        by_controls: dict[tuple[float, float], dict[str, Any]] = {}
        for cell in group:
            config = cell["config"]
            sd = float(config["noise"]["standard_deviations"][0])
            curve = float(config["curve_scale"])
            by_controls[(sd, curve)] = cell
        club_id, profile = key[:2]
        controls = (
            ("shaft_rotation_sd", "SD 1°", "SD 2°", (1.0, 1.0), (2.0, 1.0)),
            ("curve_scale", "Curve ×1", "Curve ×2", (2.0, 1.0), (2.0, 2.0)),
        )
        for comparison, first_label, second_label, first_key, second_key in controls:
            for pattern in PATTERNS:
                first = by_controls.get(first_key)
                second = by_controls.get(second_key)
                pairs.append(
                    {
                        "comparison": comparison,
                        "pattern": pattern,
                        "club_id": club_id,
                        "profile": profile,
                        "first_label": first_label,
                        "second_label": second_label,
                        "cells": [first, second],
                        "group_key": key,
                    }
                )
    return pairs


def _supported_points(cell_dir: Path) -> tuple[dict[str, list[list[float]]], dict[str, int]]:
    points = {pattern: [] for pattern in PATTERNS}
    total = dict.fromkeys(PATTERNS, 0)
    with gzip.open(cell_dir / "shots.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            pattern = row.get("pattern")
            if pattern not in points:
                raise ValueError(f"unknown pattern in {cell_dir.name}: {pattern}")
            total[pattern] += 1
            if row.get("status") != "supported":
                continue
            x = float(row["carry_x_m"])
            y = float(row["carry_y_m"])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"non-finite supported endpoint in {cell_dir.name}")
            points[pattern].append([x, y])
    return points, total


def _cell_metadata(
    cell: dict[str, Any], valid_counts: dict[str, int], total_counts: dict[str, int]
) -> dict[str, Any]:
    config = cell["config"]
    club = _club_config(cell)
    noise = config.get("noise", {})
    standard_deviations = noise.get("standard_deviations", [])
    fields = (
        "shaft rotation",
        "path",
        "speed",
        "horizontal contact",
        "vertical contact",
        "shaft lean",
        "attack angle",
    )
    max_residual = max(
        (
            float(value.get("residual_m", 0.0))
            for value in cell.get("nominals", {}).values()
        ),
        default=0.0,
    )
    return {
        "club": club.get("club_id"),
        "delivery mode": config.get("delivery_mode", "unspecified"),
        "contact law": club.get("contact_law", "unspecified"),
        "reference club speed mps": club.get("club_speed_mps"),
        "static loft deg": club.get("loft_deg"),
        "nominal shaft lean deg": club.get("shaft_lean_deg"),
        "curve scale": config.get("curve_scale"),
        "random seed": config.get("seed"),
        "attempts per pattern": config.get("n_shots"),
        "target residual max m": f"{max_residual:.3g}",
        "noise SDs": ", ".join(
            f"{field}={float(value):g}"
            for field, value in zip(fields, standard_deviations, strict=False)
        ),
        "valid_shots_by_pattern": valid_counts,
        "total_shots_by_pattern": total_counts,
    }


def _render_detail(
    cell: dict[str, Any],
    source_root: Path,
    output_root: Path,
    *,
    dpi: int,
    preview: bool,
) -> list[Path]:
    from src.tools.shot_pattern_analysis.academic_plots import (
        write_academic_dispersion_figure,
        write_overhead_trajectory_figure,
    )

    cell_dir = source_root / cell["cell"]
    points, total_counts = _supported_points(cell_dir)
    valid_counts = {pattern: len(points[pattern]) for pattern in PATTERNS}
    target_x = float(cell["target_x_m"])
    config = cell["config"]
    club = _club_config(cell)
    mode = str(config.get("delivery_mode", "unspecified"))
    profile = _detail_profile(config.get("noise", {}).get("standard_deviations", []))
    scenario = (
        f"{'PILOT PREVIEW — ' if preview else ''}{cell['cell']} — {club.get('club_id')} — {profile}; "
        f"shaft rotation SD {config['noise']['standard_deviations'][0]:g}°; "
        f"curve ×{config.get('curve_scale')}; {mode}"
    )
    metadata = _cell_metadata(cell, valid_counts, total_counts)
    cell_output = output_root / "details" / cell["cell"]
    cell_output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    density_paths = write_academic_dispersion_figure(
        points,
        cell_output / "density",
        scenario=scenario,
        metadata=metadata,
        target_point_m=(target_x, 0.0),
        dpi=dpi,
    )
    paths.extend(density_paths.values())

    raw = json.loads((cell_dir / "flight_samples.json").read_text(encoding="utf-8"))
    if set(raw.get("trajectories", {})) != set(PATTERNS):
        raise ValueError(f"raw flight samples missing pattern data: {cell['cell']}")
    overhead_paths = write_overhead_trajectory_figure(
        raw["trajectories"],
        cell_output / "overhead",
        target_x_m=float(raw["target_x_m"]),
        scenario=scenario,
        metadata=metadata,
        dpi=dpi,
    )
    paths.extend(overhead_paths.values())
    return paths


def _render_density_comparison(
    pair: dict[str, Any],
    source_root: Path,
    output_root: Path,
    *,
    dpi: int,
    preview: bool,
) -> tuple[dict[str, Any], list[Path]]:
    from src.tools.shot_pattern_analysis.academic_plots import write_academic_dispersion_figure

    first, second = pair["cells"]
    row: dict[str, Any] = {
        "comparison": pair["comparison"],
        "pattern": pair["pattern"],
        "club": pair["club_id"],
        "profile": pair["profile"],
        "first_cell": first["cell"] if first else "",
        "second_cell": second["cell"] if second else "",
        "first_condition": pair["first_label"],
        "second_condition": pair["second_label"],
        "status": "unavailable",
        "reason": "",
        "target_x_m": "",
        "png": "",
        "pdf": "",
        "svg": "",
    }
    for candidate in (first, second):
        if candidate is None:
            row["reason"] = "comparison condition missing from planned matrix"
            return row, []
        if candidate.get("status") != "completed":
            row["reason"] = str(candidate.get("reason") or candidate.get("status"))
            return row, []
    target_x_values = [first.get("target_x_m"), second.get("target_x_m")]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in target_x_values):
        row["reason"] = "matched target unavailable"
        return row, []
    if not math.isclose(float(target_x_values[0]), float(target_x_values[1]), abs_tol=1e-9):
        row["reason"] = "comparison cells do not share a common target"
        return row, []

    labels = (
        f"{pair['pattern']} | {pair['first_label']}",
        f"{pair['pattern']} | {pair['second_label']}",
    )
    points_by_label: dict[str, list[list[float]]] = {}
    valid_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    for cell, label in zip((first, second), labels, strict=True):
        points, totals = _supported_points(source_root / cell["cell"])
        points_by_label[label] = points[pair["pattern"]]
        valid_counts[label] = len(points[pair["pattern"]])
        total_counts[label] = totals[pair["pattern"]]
    if any(not points_by_label[label] for label in labels):
        row["reason"] = "no supported airborne endpoints for comparison pattern"
        return row, []

    first_config = first["config"]
    club = _club_config(first)
    residuals = [
        max(
            (float(value.get("residual_m", 0.0)) for value in cell.get("nominals", {}).values()),
            default=0.0,
        )
        for cell in (first, second)
    ]
    metadata = {
        "club": club.get("club_id"),
        "pattern": pair["pattern"],
        "profile": pair["profile"],
        "comparison": pair["comparison"],
        "contact law": club.get("contact_law"),
        "nominal shaft lean deg": club.get("shaft_lean_deg"),
        "common target downrange m": float(target_x_values[0]),
        "first condition": pair["first_label"],
        "second condition": pair["second_label"],
        "target residuals m": f"{residuals[0]:.3g} vs {residuals[1]:.3g}",
        "valid_shots_by_pattern": valid_counts,
        "total_shots_by_pattern": total_counts,
    }
    scenario = (
        f"{'PILOT PREVIEW — ' if preview else ''}{pair['club_id']} {pair['pattern']}; "
        f"{pair['profile']}; {pair['first_label']} versus {pair['second_label']}; "
        f"matched target {float(target_x_values[0]):.2f} m; nominal lean 0°"
    )
    stem = (
        output_root
        / "comparisons"
        / str(pair["club_id"])
        / str(pair["profile"])
        / str(pair["pattern"])
        / str(pair["comparison"])
    )
    paths = write_academic_dispersion_figure(
        points_by_label,
        stem,
        scenario=scenario,
        metadata=metadata,
        target_point_m=(float(target_x_values[0]), 0.0),
        dpi=dpi,
    )
    row["status"] = "completed"
    row["target_x_m"] = float(target_x_values[0])
    row.update({extension: path.relative_to(output_root).as_posix() for extension, path in paths.items()})
    return row, list(paths.values())


def _write_ledgers(
    cells: list[dict[str, Any]],
    output_root: Path,
    rendered_cells: set[str],
    comparison_rows: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    availability = output_root / "availability.csv"
    with availability.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("cell", "status", "reason", "club", "lean_deg", "curve_scale", "shaft_rotation_sd_deg"),
        )
        writer.writeheader()
        for cell in cells:
            config = cell["config"]
            noise = config.get("noise", {}).get("standard_deviations", [])
            writer.writerow(
                {
                    "cell": cell["cell"],
                    "status": cell["status"],
                    "reason": cell.get("reason", ""),
                    "club": _club_config(cell).get("club_id"),
                    "lean_deg": _club_config(cell).get("shaft_lean_deg"),
                    "curve_scale": config.get("curve_scale"),
                    "shaft_rotation_sd_deg": noise[0] if noise else "",
                }
            )

    social_density = output_root / "facebook_ready.csv"
    with social_density.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "comparison", "pattern", "club", "profile", "first_cell", "second_cell",
                "first_condition", "second_condition", "status", "reason", "target_x_m", "nominal_lean_deg", "png",
            ),
        )
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(
                {
                    field: (
                        0
                        if field == "nominal_lean_deg"
                        else row.get("club", row.get("club_id", ""))
                        if field == "club"
                        else row.get(field, "")
                    )
                    for field in (
                        "comparison", "pattern", "club", "profile", "first_cell", "second_cell",
                        "first_condition", "second_condition", "status", "reason", "target_x_m",
                        "nominal_lean_deg", "png",
                    )
                }
            )

    social_overhead = output_root / "facebook_overhead.csv"
    with social_overhead.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("cell", "status", "reason", "club", "lean_deg", "curve_scale", "shaft_rotation_sd_deg", "selection_note", "png_overhead"),
        )
        writer.writeheader()
        for cell in _facebook_selection(cells):
            config = cell["config"]
            noise = config.get("noise", {}).get("standard_deviations", [])
            completed = cell.get("status") == "completed" and cell["cell"] in rendered_cells
            overhead_source = output_root / "details" / cell["cell"] / "overhead.png"
            completed = completed and overhead_source.is_file()
            writer.writerow(
                {
                    "cell": cell["cell"],
                    "status": cell["status"],
                    "reason": cell.get("reason", ""),
                    "club": _club_config(cell).get("club_id"),
                    "lean_deg": _club_config(cell).get("shaft_lean_deg"),
                    "curve_scale": config.get("curve_scale"),
                    "shaft_rotation_sd_deg": noise[0] if noise else "",
                    "selection_note": cell["selection_note"],
                    "png_overhead": overhead_source.relative_to(output_root).as_posix() if completed else "",
                }
            )
    return availability, social_density, social_overhead


def _write_comparison_ledger(rows: list[dict[str, Any]], output_root: Path) -> Path:
    path = output_root / "comparison_availability.csv"
    fields = (
        "comparison", "pattern", "club", "profile", "first_cell", "second_cell",
        "first_condition", "second_condition", "status", "reason", "target_x_m",
        "png", "pdf", "svg",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: row.get("club", row.get("club_id", ""))
                    if field == "club"
                    else row.get(field, "")
                    for field in fields
                }
            )
    return path


def _validate_final_protocol(report: dict[str, Any], output_name: str) -> None:
    """Require the complete predeclared final population and target protocol."""
    expected = 324 if output_name == "final_figures_v1" else 72
    cells = report["cells"]
    if report.get("stage") != "final" or report.get("planned_cells") != expected or len(cells) != expected:
        raise ValueError("final figures require the complete 324/72-cell final protocol")
    fixed_target = output_name == "control_final_figures_v1"
    if any(cell["config"]["n_shots"] != 10_000 for cell in cells):
        raise ValueError("final figures require 10,000 attempts per pattern")
    if any((cell["config"].get("target_x_m") is not None) != fixed_target for cell in cells):
        raise ValueError("final figure target protocol does not match output kind")


def _assert_inputs_unchanged(source_root: Path, report: dict[str, Any], *, preview: bool) -> None:
    """Revalidate every input byte after rendering, before publishing a manifest."""
    from src.tools.shot_pattern_analysis.matched_report import load_matrix_report

    current = load_matrix_report(source_root, require_complete=not preview)
    for key in ("plan_sha256", "cell_manifest_sha256", "report_source_sha256"):
        if current[key] != report[key]:
            raise RuntimeError(f"figure source changed during rendering: {key}")


def build_final_figures(
    source_root: Path,
    output_root: Path,
    *,
    preview: bool = False,
    selected_cells: tuple[str, ...] = (),
    dpi: int = 180,
) -> Path:
    """Render source-bound figures and a provenance manifest for one report."""
    from src.tools.shot_pattern_analysis.matched_report import load_matrix_report
    from src.tools.shot_pattern_analysis.matched_report_plots import export_matched_matrix_heatmaps

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("figure outputs must be outside the source report root")
    report = load_matrix_report(source_root, require_complete=not preview)
    stage = str(report.get("stage", "unknown"))
    if preview:
        if stage != "pilot" or "preview" not in output_root.name.lower():
            raise ValueError("pilot rendering requires --preview and a preview-named output directory")
    elif stage == "pilot" or output_root.name not in (
        "final_figures_v1",
        "control_final_figures_v1",
    ):
        raise ValueError(
            "final figures require a non-pilot report and final_figures_v1/control_final_figures_v1 sibling output root"
        )
    if not preview:
        _validate_final_protocol(report, output_root.name)
    if selected_cells and not preview:
        raise ValueError("selected_cells is allowed only for explicit pilot previews")
    if preview and not selected_cells:
        raise ValueError("pilot previews require at least one explicit --cell selection")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"figure output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    cells = report["cells"]
    if selected_cells:
        unknown = set(selected_cells) - {cell["cell"] for cell in cells}
        if unknown:
            raise ValueError(f"unknown selected cell(s): {', '.join(sorted(unknown))}")
        detail_candidates = [cell for cell in cells if cell["cell"] in selected_cells]
    else:
        detail_candidates = cells
    detail_cells = [
        cell
        for cell in detail_candidates
        if cell.get("status") == "completed"
        and _detail_profile(cell.get("config", {}).get("noise", {}).get("standard_deviations", [])) is not None
    ]
    detail_paths: list[Path] = []
    for cell in detail_cells:
        detail_paths.extend(
            _render_detail(cell, source_root, output_root, dpi=dpi, preview=preview)
        )

    comparison_pairs = _comparison_pair_specs(cells)
    if preview:
        comparison_keys = {_comparison_group_key(cell) for cell in detail_candidates}
        comparison_pairs = [
            pair for pair in comparison_pairs if pair["group_key"] in comparison_keys
        ]
    comparison_rows: list[dict[str, Any]] = []
    comparison_paths: list[Path] = []
    for pair in comparison_pairs:
        row, paths = _render_density_comparison(
            pair,
            source_root,
            output_root,
            dpi=dpi,
            preview=preview,
        )
        comparison_rows.append(row)
        comparison_paths.extend(paths)
    comparison_ledger = _write_comparison_ledger(comparison_rows, output_root)

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        groups[_stratum_key(cell)].append(cell)
    if preview:
        preview_keys = {_stratum_key(cell) for cell in detail_candidates}
        groups = {key: group for key, group in groups.items() if key in preview_keys}
    matrix_paths: list[Path] = []
    for group in groups.values():
        name = _stratum_name(group)
        paths = export_matched_matrix_heatmaps(
            _heatmap_cells(group),
            output_root / "matrix" / name,
            selection_label=(
                f"{'PILOT PREVIEW' if preview else 'FINAL'}; source={source_root.name}; "
                f"delivery={group[0]['config'].get('delivery_mode', 'unspecified')}; "
                f"contact={_club_config(group[0]).get('contact_law')}; "
                f"noise SDs={group[0]['config'].get('noise', {}).get('standard_deviations')}"
            ),
            dpi=dpi,
            formats=("png",),
        )
        matrix_paths.extend(paths.values())
    availability, social_density, social_overhead = _write_ledgers(
        cells,
        output_root,
        {cell["cell"] for cell in detail_cells},
        comparison_rows,
    )
    social_density_count = sum(
        row["status"] == "completed" and bool(row.get("png"))
        and (output_root / str(row["png"])).is_file()
        for row in comparison_rows
    )
    with social_overhead.open(encoding="utf-8", newline="") as stream:
        social_overhead_count = sum(
            bool(row.get("png_overhead"))
            and (output_root / str(row["png_overhead"])).is_file()
            for row in csv.DictReader(stream)
        )

    source_files = {
        "matched_report.py": _sha256(_repository_root() / "src/tools/shot_pattern_analysis/matched_report.py"),
        "academic_plots.py": _sha256(_repository_root() / "src/tools/shot_pattern_analysis/academic_plots.py"),
        "matched_report_plots.py": _sha256(_repository_root() / "src/tools/shot_pattern_analysis/matched_report_plots.py"),
        "build_final_figures.py": _sha256(Path(__file__)),
    }
    output_files = [
        *detail_paths,
        *comparison_paths,
        *matrix_paths,
        availability,
        social_density,
        social_overhead,
        comparison_ledger,
    ]
    manifest = {
        "schema": "matched-study-figure-production/1",
        "kind": "pilot_preview" if preview else "final",
        "source_root": str(source_root),
        "source_stage": stage,
        "planned_cells": report["planned_cells"],
        "cell_status_counts": report["counts"],
        "plan_sha256": report["plan_sha256"],
        "cell_manifest_sha256": report["cell_manifest_sha256"],
        "report_source_sha256": report["report_source_sha256"],
        "plot_source_sha256": source_files,
        "rendered_detail_cells": [cell["cell"] for cell in detail_cells],
        "matrix_strata_count": len(groups),
        "direct_comparison_count": len(comparison_rows),
        "direct_comparison_status_counts": {
            status: sum(row["status"] == status for row in comparison_rows)
            for status in ("completed", "unavailable")
        },
        "availability_ledger": availability.relative_to(output_root).as_posix(),
        "comparison_availability_ledger": comparison_ledger.relative_to(output_root).as_posix(),
        "facebook_ready_index": social_density.relative_to(output_root).as_posix(),
        "facebook_ready_image_count": social_density_count,
        "facebook_overhead_index": social_overhead.relative_to(output_root).as_posix(),
        "facebook_overhead_image_count": social_overhead_count,
        "outputs": {
            path.relative_to(output_root).as_posix(): _sha256(path)
            for path in output_files
        },
        "interpretation": (
            "Figures use saved supported simulation rows; unsupported/infeasible cells remain in availability.csv. "
            "This postprocessor performs no simulation or score calculation."
        ),
    }
    _assert_inputs_unchanged(source_root, report, preview=preview)
    manifest_path = output_root / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="integrity-checked matched-study report root")
    parser.add_argument("output_root", type=Path, help="new figure output root")
    parser.add_argument("--preview", action="store_true", help="explicitly label a pilot preview")
    parser.add_argument("--cell", action="append", default=[], help="pilot preview cell; repeat to select multiple")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    print(
        build_final_figures(
            args.source_root,
            args.output_root,
            preview=args.preview,
            selected_cells=tuple(args.cell),
            dpi=args.dpi,
        )
    )


if __name__ == "__main__":
    main()
