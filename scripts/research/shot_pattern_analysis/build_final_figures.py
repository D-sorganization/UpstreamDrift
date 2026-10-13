"""Render integrity-checked matched-study figures without rerunning physics.

Final runs include per-cell endpoint-density and raw overhead figures for the
shaft-only and combined-noise profiles, shared-scale matrix heatmaps for every
noise profile, an explicit feasibility ledger, and separate concise 16:9
Facebook-ready PNGs. The academic figures retain full provenance captions;
both modes use the same endpoint histogram/HDR and raw trajectory primitives.
Pilot previews require an explicit flag and a preview-named output.
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
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from src.tools.shot_pattern_analysis.academic_plots import FigureOptions

PATTERNS = ("Straight", "Draw", "Fade")
_COMBINED_NOISE = (0.5, 0.89408, 0.00635, 0.00635, 0.0, 0.0)
_MATCHED_RAW_LANDING = "Matched Raw Landing"
_CLUB_NAMES = {
    "driver": "Driver",
    "seven_iron": "7-Iron",
    "pitching_wedge": "Pitching Wedge",
}

from scripts.research.shot_pattern_analysis.final_figure_ledgers import (
    _club_config,
    _write_availability_ledger,
    _write_comparison_ledger,
    _write_facebook_comparison_ledger,
    _write_facebook_overhead_ledger,
    _write_facebook_pattern_density_ledger,
)

from src.tools.shot_pattern_analysis.provenance import sha256_file
from src.tools.shot_pattern_analysis.matched_report import (
    load_individually_supported_endpoints,
)


def _delivery_mode_label(config: dict[str, Any]) -> str:
    """Describe saved matched runs whose schema has no explicit mode field."""
    return str(config.get("delivery_mode") or _MATCHED_RAW_LANDING)


def _detail_profile(standard_deviations: list[float]) -> str | None:
    """Name only the predeclared shaft-only and combined-noise profiles."""
    if len(standard_deviations) != 7:
        return None
    if standard_deviations[0] not in (1.0, 2.0):
        return None
    if all(
        math.isclose(value, 0.0, abs_tol=1e-10) for value in standard_deviations[1:]
    ):
        return "shaft_only"
    if all(
        math.isclose(actual, expected, abs_tol=1e-8)
        for actual, expected in zip(
            standard_deviations[1:], _COMBINED_NOISE, strict=True
        )
    ):
        return "combined2"
    return None


def _facebook_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    config = cell["config"]
    club = _club_config(cell)
    noise = config.get("noise", {})
    return (
        club.get("club_id"),
        tuple(noise.get("standard_deviations", ())),
        json.dumps(noise.get("correlation"), sort_keys=True),
        config.get("curve_scale"),
        _delivery_mode_label(config),
        club.get("contact_law"),
    )


def _facebook_selection(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select lean-0 social comparisons, using labeled PW lean-5 fallbacks."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        if (
            _detail_profile(
                cell.get("config", {}).get("noise", {}).get("standard_deviations", [])
            )
            is None
        ):
            continue
        groups[_facebook_key(cell)].append(cell)

    selected: list[dict[str, Any]] = []
    for group in groups.values():
        club_id = _club_config(group[0]).get("club_id")
        lean_zero = next(
            (
                cell
                for cell in group
                if math.isclose(_club_config(cell).get("shaft_lean_deg", math.nan), 0.0)
            ),
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
                    if math.isclose(
                        _club_config(cell).get("shaft_lean_deg", math.nan), 5.0
                    )
                    and cell.get("status") == "completed"
                ),
                None,
            )
            if lean_five is not None:
                choice = lean_five
                note = "lean 0° unavailable; showing lean 5°"
        selected.append({**choice, "selection_note": note})
    return sorted(selected, key=lambda cell: cell["cell"])


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
        _delivery_mode_label(config),
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
        f"{label}{float(value):g}" for label, value in zip(labels, noise, strict=False)
    )
    return (
        f"{_delivery_mode_label(config).lower().replace(' ', '-')}__{law}__"
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
        _delivery_mode_label(config),
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


def _supported_points(
    cell_dir: Path,
) -> tuple[dict[str, list[list[float]]], dict[str, int]]:
    plan = json.loads((cell_dir.parent / "matrix_plan.json").read_text())
    config = plan.get("cells", {}).get(cell_dir.name)
    if not isinstance(config, dict):
        raise ValueError(f"cell is not in its matrix plan: {cell_dir.name}")
    return load_individually_supported_endpoints(
        cell_dir / "shots.csv.gz", int(config["n_shots"])
    )


def _cell_metadata(
    cell: dict[str, Any], valid_counts: dict[str, int], total_counts: dict[str, int]
) -> dict[str, Any]:
    config = cell["config"]
    club = _club_config(cell)
    noise = config.get("noise", {})
    standard_deviations = noise.get("standard_deviations", [])
    fields = (
        "shaft rotation (deg)",
        "path (deg)",
        "speed (m/s)",
        "horizontal contact (m)",
        "vertical contact (m)",
        "shaft lean (deg)",
        "attack angle (deg)",
    )
    max_residual = max(
        (
            float(value.get("residual_m", 0.0))
            for value in cell.get("nominals", {}).values()
        ),
        default=0.0,
    )
    nominal_solutions = "; ".join(
        f"{pattern}: face {float(values['face_deg']):+0.3f}°, "
        f"path {float(values['path_deg']):+0.3f}°, "
        f"speed {float(values['speed_mps']):0.3f} m/s"
        for pattern, values in cell.get("nominals", {}).items()
        if all(key in values for key in ("face_deg", "path_deg", "speed_mps"))
    )
    return {
        "club": club.get("club_id"),
        "delivery mode": _delivery_mode_label(config),
        "contact law": club.get("contact_law", "unspecified"),
        "reference club speed mps": club.get("club_speed_mps"),
        "static loft deg": club.get("loft_deg"),
        "nominal shaft lean deg": club.get("shaft_lean_deg"),
        "curve scale": config.get("curve_scale"),
        "random seed": config.get("seed"),
        "attempts per pattern": config.get("n_shots"),
        "target residual max m": f"{max_residual:.3g}",
        "nominal pattern solutions": nominal_solutions or "not recorded",
        "noise SDs": ", ".join(
            f"{field}={float(value):g}"
            for field, value in zip(fields, standard_deviations, strict=False)
        ),
        "valid_shots_by_pattern": valid_counts,
        "total_shots_by_pattern": total_counts,
    }


def _social_noise_summary(
    standard_deviations: list[float] | tuple[float, ...],
) -> str:
    """Format delivery noise components with readable units for social captions."""
    if (
        not isinstance(standard_deviations, (list, tuple))
        or len(standard_deviations) != 7
    ):
        raise ValueError("social figure requires all seven noise standard deviations")
    if not all(
        isinstance(v, (int, float)) and math.isfinite(float(v))
        for v in standard_deviations
    ):
        raise ValueError("noise standard deviations must be finite numbers")
    return (
        f"Path SD {float(standard_deviations[1]):g}° • "
        f"Speed SD {float(standard_deviations[2]) * 2.2369362921:.2g} mph • "
        f"Strike SD {float(standard_deviations[3]) * 1000:g}/{float(standard_deviations[4]) * 1000:g} mm "
        "horizontal/vertical"
    )


def _social_metadata(
    cell: dict[str, Any], valid_counts: dict[str, int], total_counts: dict[str, int]
) -> tuple[str, dict[str, Any]]:
    """Describe only the controls needed to interpret a shareable figure."""
    config = cell["config"]
    club = _club_config(cell)
    noise = config["noise"]["standard_deviations"]
    if len(noise) != 7:
        raise ValueError("social figure requires all seven noise standard deviations")
    club_id = str(club.get("club_id"))
    if club_id not in _CLUB_NAMES:
        raise ValueError(f"unrecognized club for social figure: {club_id}")
    subtitle = (
        f"Lean {float(club['shaft_lean_deg']):g}° • "
        f"Shaft Twist SD {float(noise[0]):g}° • Curve ×{float(config['curve_scale']):g}"
    )
    protocol = (
        cell.get("protocol_id")
        or config.get("supplement_protocol")
        or "Original Matched-Delivery Protocol"
    )
    return subtitle, {
        "social_club": _CLUB_NAMES[club_id],
        "Protocol": protocol,
        "Noise": _social_noise_summary(noise),
        "Matched Target": f"{float(cell['target_x_m']):.2f} m downrange",
        "valid_shots_by_pattern": valid_counts,
        "total_shots_by_pattern": total_counts,
    }


def _figure_options(
    presentation: Literal["academic", "social"], dpi: int
) -> FigureOptions:
    """Map the builder's presentation/DPI controls onto writer options.

    This is the single keyword-to-options mapping for every final writer call
    (#12309). Only ``presentation`` and ``dpi`` are set; histogram bins and
    HDR region masses keep the writer defaults, exactly as the historical
    keyword calls did.
    """
    from src.tools.shot_pattern_analysis.academic_plots import FigureOptions

    if presentation not in ("academic", "social"):
        raise ValueError("presentation must be academic or social")
    if isinstance(dpi, bool) or not isinstance(dpi, int):
        raise TypeError("dpi must be an integer")
    return FigureOptions(presentation=presentation, dpi=dpi)


def _render_cell_figures(
    cell: dict[str, Any],
    source_root: Path,
    output_root: Path,
    *,
    dpi: int,
    preview: bool,
    presentation: Literal["academic", "social"],
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
    mode = _delivery_mode_label(config)
    profile = _detail_profile(config.get("noise", {}).get("standard_deviations", []))
    if presentation == "social":
        scenario, metadata = _social_metadata(cell, valid_counts, total_counts)
        if preview:
            scenario = f"PILOT PREVIEW — {scenario}"
        cell_output = output_root / "social" / "details" / cell["cell"]
    else:
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
        options=_figure_options(presentation, dpi),
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
        options=_figure_options(presentation, dpi),
    )
    paths.extend(overhead_paths.values())
    return paths


def _render_detail(
    cell: dict[str, Any],
    source_root: Path,
    output_root: Path,
    *,
    dpi: int,
    preview: bool,
) -> list[Path]:
    return _render_cell_figures(
        cell,
        source_root,
        output_root,
        dpi=dpi,
        preview=preview,
        presentation="academic",
    )


def _render_social_detail(
    cell: dict[str, Any],
    source_root: Path,
    output_root: Path,
    *,
    dpi: int,
    preview: bool,
) -> list[Path]:
    return _render_cell_figures(
        cell, source_root, output_root, dpi=dpi, preview=preview, presentation="social"
    )


def _comparison_caption(
    pair: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any],
    target_x_values: list[Any],
    valid_counts: dict[str, int],
    total_counts: dict[str, int],
    preview: bool,
) -> tuple[dict[str, Any], str]:
    """Build the matched-control caption and source-bound plot metadata."""
    first_config = first["config"]
    club = _club_config(first)
    residuals = [
        max(
            (
                float(value.get("residual_m", 0.0))
                for value in cell.get("nominals", {}).values()
            ),
            default=0.0,
        )
        for cell in (first, second)
    ]
    comparison_name = str(pair.get("comparison"))
    noise_sds = first_config.get("noise", {}).get("standard_deviations", [])
    if comparison_name == "curve_scale":
        if not noise_sds:
            raise ValueError(
                "noise standard deviations missing for curve scale comparison"
            )
        fixed_sd = float(noise_sds[0])
        if not math.isfinite(fixed_sd):
            raise ValueError("shaft twist SD must be finite")
        fixed_control = f"Shaft Twist SD {fixed_sd:g}°"
        comparison_condition = f"{pair['first_label']} vs {pair['second_label']}"
    elif comparison_name == "shaft_rotation_sd":
        curve_scale = float(first_config.get("curve_scale", math.nan))
        if not math.isfinite(curve_scale):
            raise ValueError("curve scale must be finite")
        fixed_control = f"Curve ×{curve_scale:g}"
        comparison_condition = (
            f"Shaft Twist {pair['first_label']} vs {pair['second_label']}"
        )
    else:
        raise ValueError(f"unrecognized comparison type: {comparison_name}")

    profile = str(pair.get("profile"))
    if profile == "shaft_only":
        profile_label = "Shaft Twist Only"
    elif profile == "combined2":
        profile_label = "Combined Delivery Noise"
    else:
        profile_label = profile.replace("_", " ").title()

    metadata: dict[str, Any] = {
        "social_club": _CLUB_NAMES.get(str(pair["club_id"]), str(pair["club_id"])),
        "Protocol": first.get("protocol_id") or "Original Matched-Delivery Protocol",
        "Target": f"{float(target_x_values[0]):.2f} m downrange",
    }
    if club.get("loft_deg") is not None:
        loft_val = float(club["loft_deg"])
        if not math.isfinite(loft_val):
            raise ValueError("static loft must be finite")
        metadata["Static Loft"] = f"{loft_val:g}°"
    lean_val = float(club.get("shaft_lean_deg", 0.0))
    if not math.isfinite(lean_val):
        raise ValueError("nominal lean must be finite")
    metadata["Nominal Lean"] = f"{lean_val:g}°"
    metadata["Noise"] = _social_noise_summary(noise_sds)
    metadata["Match Residuals"] = f"{residuals[0]:.3g} vs {residuals[1]:.3g} m"
    metadata["valid_shots_by_pattern"] = valid_counts
    metadata["total_shots_by_pattern"] = total_counts

    scenario = (
        f"{'PILOT PREVIEW — ' if preview else ''}"
        f"{pair['pattern']} • {comparison_condition} • "
        f"{fixed_control} • {profile_label}"
    )
    return metadata, scenario


def _render_density_comparison(
    pair: dict[str, Any],
    source_root: Path,
    output_root: Path,
    *,
    dpi: int,
    preview: bool,
) -> tuple[dict[str, Any], list[Path]]:
    from src.tools.shot_pattern_analysis.academic_plots import (
        write_academic_dispersion_figure,
    )

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
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in target_x_values
    ):
        row["reason"] = "matched target unavailable"
        return row, []
    if not math.isclose(
        float(target_x_values[0]), float(target_x_values[1]), abs_tol=1e-9
    ):
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

    metadata, scenario = _comparison_caption(
        pair, first, second, target_x_values, valid_counts, total_counts, preview
    )
    stem = (
        output_root
        / "social"
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
        options=_figure_options("social", dpi),
    )
    row["status"] = "completed"
    row["target_x_m"] = float(target_x_values[0])
    row.update(
        {
            extension: path.relative_to(output_root).as_posix()
            for extension, path in paths.items()
        }
    )
    return row, list(paths.values())


def _write_ledgers(
    cells: list[dict[str, Any]],
    output_root: Path,
    rendered_cells: set[str],
    comparison_rows: list[dict[str, Any]],
) -> tuple[Path, Path, Path, Path]:
    availability = _write_availability_ledger(cells, output_root)
    social_density = _write_facebook_comparison_ledger(comparison_rows, output_root)
    selected_social = _facebook_selection(cells)
    social_pattern_density = _write_facebook_pattern_density_ledger(
        selected_social, output_root, rendered_cells
    )
    social_overhead = _write_facebook_overhead_ledger(
        selected_social, output_root, rendered_cells
    )
    return availability, social_density, social_overhead, social_pattern_density


def _validate_final_protocol(report: dict[str, Any], output_name: str) -> None:
    """Require the complete predeclared final population and target protocol."""
    expected = 324 if output_name == "final_figures_v1" else 72
    cells = report["cells"]
    if (
        report.get("stage") != "final"
        or report.get("planned_cells") != expected
        or len(cells) != expected
    ):
        raise ValueError(
            "final figures require the complete 324/72-cell final protocol"
        )
    fixed_target = output_name == "control_final_figures_v1"
    if any(cell["config"]["n_shots"] != 10_000 for cell in cells):
        raise ValueError("final figures require 10,000 attempts per pattern")
    if any(
        (cell["config"].get("target_x_m") is not None) != fixed_target for cell in cells
    ):
        raise ValueError("final figure target protocol does not match output kind")


def _assert_inputs_unchanged(
    source_root: Path, report: dict[str, Any], *, preview: bool
) -> None:
    """Revalidate every input byte after rendering, before publishing a manifest."""
    from src.tools.shot_pattern_analysis.matched_report import load_matrix_report

    current = load_matrix_report(source_root, require_complete=not preview)
    for key in ("plan_sha256", "cell_manifest_sha256", "report_source_sha256"):
        if current[key] != report[key]:
            raise RuntimeError(f"figure source changed during rendering: {key}")


def _prepare_output(
    source_root: Path,
    output_root: Path,
    preview: bool,
    selected_cells: tuple[str, ...],
) -> tuple[Path, Path, dict[str, Any], str]:
    """Validate the source and output contract before publishing any file."""
    from src.tools.shot_pattern_analysis.matched_report import load_matrix_report

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("figure outputs must be outside the source report root")
    report = load_matrix_report(source_root, require_complete=not preview)
    stage = str(report.get("stage", "unknown"))
    if preview:
        if stage != "pilot" or "preview" not in output_root.name.lower():
            raise ValueError(
                "pilot rendering requires --preview and a preview-named output directory"
            )
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
        raise ValueError(
            "pilot previews require at least one explicit --cell selection"
        )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"figure output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    return source_root, output_root, report, stage


def _render_detail_sets(
    cells: list[dict[str, Any]],
    selected_cells: tuple[str, ...],
    source_root: Path,
    output_root: Path,
    dpi: int,
    preview: bool,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], set[str], list[Path], list[Path]
]:
    """Render selected academic details and their social counterparts."""

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
        and _detail_profile(
            cell.get("config", {}).get("noise", {}).get("standard_deviations", [])
        )
        is not None
    ]
    detail_paths: list[Path] = []
    for cell in detail_cells:
        detail_paths.extend(
            _render_detail(cell, source_root, output_root, dpi=dpi, preview=preview)
        )
    detail_ids = {cell["cell"] for cell in detail_cells}
    social_detail_paths: list[Path] = []
    for cell in _facebook_selection(cells):
        if cell["cell"] in detail_ids and cell["status"] == "completed":
            social_detail_paths.extend(
                _render_social_detail(
                    cell, source_root, output_root, dpi=dpi, preview=preview
                )
            )
    return (
        detail_candidates,
        detail_cells,
        detail_ids,
        detail_paths,
        social_detail_paths,
    )


def _render_comparisons(
    cells: list[dict[str, Any]],
    detail_candidates: list[dict[str, Any]],
    source_root: Path,
    output_root: Path,
    dpi: int,
    preview: bool,
) -> tuple[list[dict[str, Any]], list[Path], Path]:
    """Render paired endpoint comparisons and their availability ledger."""
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
    return comparison_rows, comparison_paths, comparison_ledger


def _render_matrices(
    cells: list[dict[str, Any]],
    detail_candidates: list[dict[str, Any]],
    source_root: Path,
    output_root: Path,
    dpi: int,
    preview: bool,
) -> tuple[dict[tuple[str, ...], list[dict[str, Any]]], list[Path]]:
    """Render each saved stratum with the original source labels and order."""
    from src.tools.shot_pattern_analysis.matched_report_plots import (
        export_matched_matrix_heatmaps,
    )

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        groups[_stratum_key(cell)].append(cell)
    if preview:
        preview_keys = {_stratum_key(cell) for cell in detail_candidates}
        groups = {key: group for key, group in groups.items() if key in preview_keys}
    matrix_paths: list[Path] = []
    for group in groups.values():
        name = _stratum_name(group)
        matrix_exports = export_matched_matrix_heatmaps(
            _heatmap_cells(group),
            output_root / "matrix" / name,
            selection_label=(
                f"{'PILOT PREVIEW' if preview else 'FINAL'}; source={source_root.name}; "
                f"delivery={_delivery_mode_label(group[0]['config'])}; "
                f"contact={_club_config(group[0]).get('contact_law')}; "
                f"noise SDs={group[0]['config'].get('noise', {}).get('standard_deviations')}"
            ),
            dpi=dpi,
            formats=("png",),
        )
        matrix_paths.extend(matrix_exports.values())
    return groups, matrix_paths


def _social_export_counts(
    output_root: Path,
    comparison_rows: list[dict[str, Any]],
    social_overhead: Path,
    social_pattern_density: Path,
) -> tuple[int, int, int]:
    """Count only actual exported social images for the manifest."""
    social_density_count = sum(
        row["status"] == "completed"
        and bool(row.get("png"))
        and (output_root / str(row["png"])).is_file()
        for row in comparison_rows
    )
    with social_overhead.open(encoding="utf-8", newline="") as stream:
        social_overhead_count = sum(
            bool(row.get("png_overhead"))
            and (output_root / str(row["png_overhead"])).is_file()
            for row in csv.DictReader(stream)
        )
    with social_pattern_density.open(encoding="utf-8", newline="") as stream:
        social_pattern_density_count = sum(
            bool(row.get("png_density"))
            and (output_root / str(row["png_density"])).is_file()
            for row in csv.DictReader(stream)
        )
    return social_density_count, social_overhead_count, social_pattern_density_count


def _plot_source_hashes() -> dict[str, str]:
    """Bind the figure manifest to the exact producer source bytes."""
    root = _repository_root()
    return {
        "matched_report.py": sha256_file(
            root / "src/tools/shot_pattern_analysis/matched_report.py"
        ),
        "academic_plots.py": sha256_file(
            root / "src/tools/shot_pattern_analysis/academic_plots.py"
        ),
        "matched_report_plots.py": sha256_file(
            root / "src/tools/shot_pattern_analysis/matched_report_plots.py"
        ),
        "build_final_figures.py": sha256_file(Path(__file__)),
        "final_figure_ledgers.py": sha256_file(
            Path(__file__).with_name("final_figure_ledgers.py")
        ),
    }


def build_final_figures(
    source_root: Path,
    output_root: Path,
    *,
    preview: bool = False,
    selected_cells: tuple[str, ...] = (),
    dpi: int = 180,
) -> Path:
    """Render source-bound figures and a provenance manifest for one report."""
    source_root, output_root, report, stage = _prepare_output(
        source_root, output_root, preview, selected_cells
    )
    cells = report["cells"]
    detail_candidates, detail_cells, detail_ids, detail_paths, social_detail_paths = (
        _render_detail_sets(
            cells, selected_cells, source_root, output_root, dpi, preview
        )
    )
    comparison_rows, comparison_paths, comparison_ledger = _render_comparisons(
        cells, detail_candidates, source_root, output_root, dpi, preview
    )
    groups, matrix_paths = _render_matrices(
        cells, detail_candidates, source_root, output_root, dpi, preview
    )
    availability, social_density, social_overhead, social_pattern_density = (
        _write_ledgers(
            cells,
            output_root,
            detail_ids,
            comparison_rows,
        )
    )
    social_density_count, social_overhead_count, social_pattern_density_count = (
        _social_export_counts(
            output_root, comparison_rows, social_overhead, social_pattern_density
        )
    )

    source_files = _plot_source_hashes()
    output_files = [
        *detail_paths,
        *social_detail_paths,
        *comparison_paths,
        *matrix_paths,
        availability,
        social_density,
        social_overhead,
        social_pattern_density,
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
        "comparison_availability_ledger": comparison_ledger.relative_to(
            output_root
        ).as_posix(),
        "facebook_ready_index": social_density.relative_to(output_root).as_posix(),
        "facebook_ready_image_count": social_density_count,
        "facebook_overhead_index": social_overhead.relative_to(output_root).as_posix(),
        "facebook_overhead_image_count": social_overhead_count,
        "facebook_pattern_density_index": social_pattern_density.relative_to(
            output_root
        ).as_posix(),
        "facebook_pattern_density_image_count": social_pattern_density_count,
        "outputs": {
            path.relative_to(output_root).as_posix(): sha256_file(path)
            for path in output_files
        },
        "interpretation": (
            "Figures use saved supported simulation rows; unsupported/infeasible cells remain in availability.csv. "
            "This postprocessor performs no simulation or score calculation."
        ),
    }
    _assert_inputs_unchanged(source_root, report, preview=preview)
    manifest_path = output_root / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_root", type=Path, help="integrity-checked matched-study report root"
    )
    parser.add_argument("output_root", type=Path, help="new figure output root")
    parser.add_argument(
        "--preview", action="store_true", help="explicitly label a pilot preview"
    )
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        help="pilot preview cell; repeat to select multiple",
    )
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
