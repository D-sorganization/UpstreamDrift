"""Deterministic availability and social-export ledgers for final figures."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _club_config(cell: dict[str, Any]) -> dict[str, Any]:
    config = cell.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("club"), dict):
        raise ValueError(f"cell {cell.get('cell')} has no club configuration")
    return config["club"]


def _write_availability_ledger(cells: list[dict[str, Any]], output_root: Path) -> Path:
    """Write the planned cell population in input order."""
    availability = output_root / "availability.csv"
    with availability.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "cell",
                "status",
                "reason",
                "club",
                "lean_deg",
                "curve_scale",
                "shaft_rotation_sd_deg",
            ),
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
    return availability


def _write_facebook_comparison_ledger(
    comparison_rows: list[dict[str, Any]], output_root: Path
) -> Path:
    """Write social comparison availability in planned pair order."""
    social_density = output_root / "facebook_ready.csv"
    with social_density.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "comparison",
                "pattern",
                "club",
                "profile",
                "first_cell",
                "second_cell",
                "first_condition",
                "second_condition",
                "status",
                "reason",
                "target_x_m",
                "nominal_lean_deg",
                "png",
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
                        "comparison",
                        "pattern",
                        "club",
                        "profile",
                        "first_cell",
                        "second_cell",
                        "first_condition",
                        "second_condition",
                        "status",
                        "reason",
                        "target_x_m",
                        "nominal_lean_deg",
                        "png",
                    )
                }
            )
    return social_density


def _write_facebook_pattern_density_ledger(
    selected_social: list[dict[str, Any]], output_root: Path, rendered_cells: set[str]
) -> Path:
    """Record selected per-pattern density exports and infeasible fallbacks."""
    social_pattern_density = output_root / "facebook_pattern_density.csv"
    with social_pattern_density.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "cell",
                "status",
                "reason",
                "club",
                "lean_deg",
                "curve_scale",
                "shaft_rotation_sd_deg",
                "selection_note",
                "png_density",
            ),
        )
        writer.writeheader()
        for cell in selected_social:
            config = cell["config"]
            club = _club_config(cell)
            noise = config.get("noise", {}).get("standard_deviations", [])
            density_source = (
                output_root / "social" / "details" / cell["cell"] / "density.png"
            )
            completed = (
                cell.get("status") == "completed"
                and cell["cell"] in rendered_cells
                and density_source.is_file()
            )
            writer.writerow(
                {
                    "cell": cell["cell"],
                    "status": cell["status"],
                    "reason": cell.get("reason", ""),
                    "club": club.get("club_id"),
                    "lean_deg": club.get("shaft_lean_deg"),
                    "curve_scale": config.get("curve_scale"),
                    "shaft_rotation_sd_deg": noise[0] if noise else "",
                    "selection_note": cell["selection_note"],
                    "png_density": density_source.relative_to(output_root).as_posix()
                    if completed
                    else "",
                }
            )
    return social_pattern_density


def _write_facebook_overhead_ledger(
    selected_social: list[dict[str, Any]], output_root: Path, rendered_cells: set[str]
) -> Path:
    """Record selected raw-overhead exports and infeasible fallbacks."""
    social_overhead = output_root / "facebook_overhead.csv"
    with social_overhead.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "cell",
                "status",
                "reason",
                "club",
                "lean_deg",
                "curve_scale",
                "shaft_rotation_sd_deg",
                "selection_note",
                "png_overhead",
            ),
        )
        writer.writeheader()
        for cell in selected_social:
            config = cell["config"]
            noise = config.get("noise", {}).get("standard_deviations", [])
            completed = (
                cell.get("status") == "completed" and cell["cell"] in rendered_cells
            )
            overhead_source = (
                output_root / "social" / "details" / cell["cell"] / "overhead.png"
            )
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
                    "png_overhead": overhead_source.relative_to(output_root).as_posix()
                    if completed
                    else "",
                }
            )
    return social_overhead


def _write_comparison_ledger(rows: list[dict[str, Any]], output_root: Path) -> Path:
    path = output_root / "comparison_availability.csv"
    fields = (
        "comparison",
        "pattern",
        "club",
        "profile",
        "first_cell",
        "second_cell",
        "first_condition",
        "second_condition",
        "status",
        "reason",
        "target_x_m",
        "png",
        "pdf",
        "svg",
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
