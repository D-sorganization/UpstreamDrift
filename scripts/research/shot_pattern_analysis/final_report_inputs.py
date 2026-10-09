"""Source-bound figure and interpretation inputs for the final report."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def profile_from_cell(name: str) -> str:
    # The validated plan is the authority for the cell name; parse its suffix
    # without importing the simulation runner into the pure report formatter.
    match = re.search(r"_rotation\d+_(.+)$", name)
    if match is None:
        raise ValueError(f"cell name lacks a noise profile: {name}")
    return match.group(1)


def verified_original_figure_links(
    figure_root: Path, report: dict, *, report_root: Path
) -> list[dict]:
    """Select real academic/social detail assets bound to the final matrix."""
    figure_root = Path(figure_root).resolve()
    report_root = Path(report_root).resolve()
    manifest_path = figure_root / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "matched-study-figure-production/1"
        or manifest.get("kind") != "final"
        or manifest.get("source_stage") != report["stage"]
        or manifest.get("plan_sha256") != report["plan_sha256"]
        or manifest.get("cell_manifest_sha256") != report["cell_manifest_sha256"]
    ):
        raise ValueError("figure manifest differs from final matrix")
    outputs = manifest["outputs"]
    links: list[dict] = []
    for club in sorted({cell["config"]["club"]["club_id"] for cell in report["cells"]}):
        candidates = sorted(
            (
                cell
                for cell in report["cells"]
                if cell["status"] == "completed"
                and cell["config"]["club"]["club_id"] == club
                and cell["config"]["curve_scale"] == 1
                and cell["config"]["noise"]["standard_deviations"][0] == 2
                and profile_from_cell(cell["cell"]) == "combined2"
            ),
            key=lambda cell: cell["config"]["club"]["shaft_lean_deg"],
        )
        selected = next(
            (
                cell
                for cell in candidates
                if all(
                    f"{prefix}details/{cell['cell']}/{kind}.png" in outputs
                    for prefix in ("", "social/")
                    for kind in ("density", "overhead")
                )
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"no academic and social figure pair for {club}")
        for presentation, prefix in (("academic", ""), ("social", "social/")):
            for kind in ("density", "overhead"):
                relative = f"{prefix}details/{selected['cell']}/{kind}.png"
                path = (figure_root / relative).resolve()
                if not path.is_relative_to(figure_root) or not path.is_file():
                    raise ValueError(f"figure artifact missing: {relative}")
                if hashlib.sha256(path.read_bytes()).hexdigest() != outputs[relative]:
                    raise ValueError(f"figure artifact changed: {relative}")
                links.append(
                    {
                        "club": club,
                        "kind": kind,
                        "presentation": presentation,
                        "source_protocol": "original_primary"
                        if report["stage"] == "final"
                        else "original_control",
                        "source_kind": "individual_supported_endpoints"
                        if kind == "density"
                        else "saved_illustrative_trajectory_samples",
                        "relative_path": path.relative_to(report_root).as_posix(),
                    }
                )
    return links


def verified_original_figure_inventory(
    figure_root: Path, report: dict, *, report_root: Path
) -> str:
    """Return the hash-bound full availability index for one original matrix."""
    figure_root = Path(figure_root).resolve()
    report_root = Path(report_root).resolve()
    # The same strict manifest and selected-asset validation guards this index.
    verified_original_figure_links(figure_root, report, report_root=report_root)
    manifest = json.loads((figure_root / "figure_manifest.json").read_text())
    relative = manifest.get("availability_ledger")
    if relative != "availability.csv":
        raise ValueError("original figure availability inventory is missing")
    path = figure_root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest[
        "outputs"
    ].get(relative):
        raise ValueError("original figure availability inventory changed")
    return path.relative_to(report_root).as_posix()


def assemble_qualified_interpretation(
    primary_root: Path,
    primary: dict,
    control_root: Path,
    control: dict,
    supplement: dict,
    *,
    primary_figures: Path,
    control_figures: Path,
    supplement_dir: Path,
    report_root: Path,
) -> dict:
    """Compose selected interpretation from independently qualified inputs."""
    from src.tools.shot_pattern_analysis.matched_interpretation import (
        describe_effect_modification,
        describe_saved_cell,
        describe_scoring,
    )

    report_root = Path(report_root).resolve()
    selected = [
        cell
        for cell in primary["cells"]
        if cell["config"]["curve_scale"] == 1
        and cell["config"]["noise"]["standard_deviations"][0] == 2
        and profile_from_cell(cell["cell"]) == "combined2"
    ]
    if len(selected) != 9 or {
        (cell["config"]["club"]["club_id"], cell["config"]["club"]["shaft_lean_deg"])
        for cell in selected
    } != {
        (club, lean)
        for club in ("driver", "seven_iron", "pitching_wedge")
        for lean in (0, 5, 10)
    }:
        raise ValueError("selected interpretation requires nine club/lean conditions")
    selected.sort(
        key=lambda cell: (
            cell["config"]["club"]["club_id"],
            cell["config"]["club"]["shaft_lean_deg"],
        )
    )
    explained = [
        describe_saved_cell(Path(primary_root) / cell["cell"]) for cell in selected
    ]
    scored = [(cell["cell"], describe_scoring(cell)) for cell in selected]
    comparisons = supplement["comparisons"]
    if comparisons.get("schema") != "matched-supplement-comparisons/1":
        raise ValueError("qualified supplement comparison schema differs")
    # The separately qualified inventory can contain many comparisons; show
    # source-named direct contrasts here and link its complete CSV below.
    pairs = [
        describe_effect_modification(pair)
        for pair in comparisons["comparisons"]
        if pair.get("comparison_kind") in ("curve_strength", "shaft_rotation_sd")
        and pair.get("status") in ("available", "conditional_only")
    ]
    representative_effects: list[dict] = []
    for club in ("driver", "seven_iron", "pitching_wedge"):
        for kind in ("curve_strength", "shaft_rotation_sd"):
            chosen = next(
                (
                    pair
                    for pair in pairs
                    if pair["comparison_kind"] == kind
                    and all(
                        name.startswith(club + "_") for name in pair["source_cells"]
                    )
                    and "combined2" in pair["source_cells"][0]
                    and "lean0" in pair["source_cells"][0]
                ),
                None,
            )
            if chosen is not None:
                representative_effects.append(chosen)
    links = [
        *verified_original_figure_links(
            primary_figures, primary, report_root=report_root
        ),
        *verified_original_figure_links(
            control_figures, control, report_root=report_root
        ),
    ]
    figure_inventories = [
        verified_original_figure_inventory(
            primary_figures, primary, report_root=report_root
        ),
        verified_original_figure_inventory(
            control_figures, control, report_root=report_root
        ),
    ]
    ledger = supplement["figure_ledger"]
    if ledger.get("schema") != "matched-supplement-figure-ledger/1":
        raise ValueError("qualified supplement figure inventory differs")
    supplemental_assets: set[tuple[str, str, str]] = set()
    for item in ledger["figures"]:
        if item.get("figure_kind") not in (
            "density_social_png",
            "overhead_social_png",
            "density_png",
            "overhead_png",
            "paired_density_social_png",
            "paired_overhead_social_png",
        ):
            continue
        source_protocol = item.get("source_protocol") or "paired supplement"
        presentation = "social" if "social" in item["figure_kind"] else "academic"
        kind = "overhead" if "overhead" in item["figure_kind"] else "density"
        asset_key = (source_protocol, presentation, kind)
        if asset_key in supplemental_assets:
            continue
        supplemental_assets.add(asset_key)
        relative = Path(item["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("supplement figure path must be relative")
        path = Path(supplement_dir).resolve() / "figures" / relative
        links.append(
            {
                "club": item.get("club_id", "supplement"),
                "kind": kind,
                "presentation": presentation,
                "source_protocol": source_protocol,
                "source_kind": item["source_kind"],
                "relative_path": path.relative_to(report_root).as_posix(),
            }
        )
    report = supplement["report"]
    if (
        report.get("schema") != "matched-supplement-report/1"
        or report.get("unique_executed_cells") != 26
    ):
        raise ValueError("additive supplement inventory is incomplete")
    receipt_path = Path(supplement_dir) / "REPORT_SUPPLEMENT_PIPELINE.json"
    return {
        "selected_cells": explained,
        "scores": scored,
        "effects": representative_effects,
        "figure_links": links,
        "figure_inventories": figure_inventories,
        "supplement": {
            "status": "qualified",
            "unique_executed_cells": report["unique_executed_cells"],
            "verified_control_aliases": sum(
                row["proof_status"] == "verified" for row in report["aliases"]
            ),
            "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "comparison_inventory": (
                Path(supplement_dir).resolve() / "REPORT_SUPPLEMENT_COMPARISONS.csv"
            )
            .relative_to(report_root)
            .as_posix(),
            "figure_inventory": (
                Path(supplement_dir).resolve() / "REPORT_SUPPLEMENT_FIGURES.json"
            )
            .relative_to(report_root)
            .as_posix(),
        },
    }
