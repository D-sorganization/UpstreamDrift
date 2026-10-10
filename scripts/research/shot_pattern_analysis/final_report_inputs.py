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
    figure_root: Path, report: dict, *, report_root: Path, protocol: str
) -> list[dict]:
    """Select real academic/social detail assets bound to the final matrix."""
    if protocol not in ("original_primary", "original_control"):
        raise ValueError("unknown original figure protocol")
    expected_cells = 324 if protocol == "original_primary" else 72
    if (
        report["planned_cells"] in (324, 72)
        and report["planned_cells"] != expected_cells
    ):
        raise ValueError("figure protocol disagrees with matrix plan")
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
                        "source_protocol": protocol,
                        "source_kind": "individual_supported_endpoints"
                        if kind == "density"
                        else "saved_illustrative_trajectory_samples",
                        "relative_path": path.relative_to(report_root).as_posix(),
                    }
                )
    return links


def verified_original_figure_inventory(
    figure_root: Path, report: dict, *, report_root: Path, protocol: str
) -> str:
    """Return the hash-bound full availability index for one original matrix."""
    figure_root = Path(figure_root).resolve()
    report_root = Path(report_root).resolve()
    # The same strict manifest and selected-asset validation guards this index.
    verified_original_figure_links(
        figure_root, report, report_root=report_root, protocol=protocol
    )
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


def supplement_figure_links(
    entries: list[dict],
    selected_effects: list[dict],
    *,
    supplement_dir: Path,
    report_root: Path,
) -> list[dict]:
    """Select PNG links from an already hash-verified supplemental ledger.

    Paired assets must describe the exact ordered conditions used in the
    selected effect tables. Keep each physical pattern; do not deduplicate it
    against another pattern. The caller verifies asset bytes at publication.
    """
    wanted = {
        (
            effect["comparison_kind"],
            tuple(effect["source_cells"]),
            tuple(effect["source_protocols"]),
        )
        for effect in selected_effects
    }
    links: list[dict] = []
    seen: set[tuple] = set()
    root = Path(supplement_dir).resolve() / "figures"
    report_root = Path(report_root).resolve()
    for entry in entries:
        kind_name = entry.get("figure_kind", "")
        match = re.fullmatch(
            r"(?:paired_)?(density|overhead)(?:_(straight|draw|fade))?(_social)?_png",
            kind_name,
        )
        if match is None or entry.get("status") == "unavailable":
            continue
        kind, filename_pattern, social = match.groups()
        presentation = "social" if social else "academic"
        if entry.get("presentation", presentation) != presentation:
            raise ValueError("supplement figure presentation disagrees with asset kind")
        pattern = entry.get("physical_pattern")
        if filename_pattern and pattern != filename_pattern.title():
            raise ValueError("supplement physical pattern disagrees with asset kind")
        cells: tuple[str, ...]
        protocols: tuple[str, ...]
        identity: tuple
        paired = "first_cell" in entry or "second_cell" in entry
        if paired:
            cells = (entry["first_cell"], entry["second_cell"])
            protocols = (entry["first_protocol"], entry["second_protocol"])
            identity = (entry.get("comparison_kind"), cells, protocols)
            if identity not in wanted:
                continue
            if (
                kind == "density"
                and presentation == "social"
                and pattern not in ("Straight", "Draw", "Fade")
            ):
                raise ValueError(
                    "paired social density must identify a physical pattern"
                )
            club = str(cells[0]).split("_lean", 1)[0]
            protocol = " → ".join(dict.fromkeys(protocols))
        else:
            cells = (entry["cell"],)
            protocol = entry.get("source_protocol", "supplement")
            protocols = (protocol,)
            club = str(entry.get("club_id") or cells[0]).split("_lean", 1)[0]
            identity = (protocol, club)
        asset_key = (identity, presentation, kind, pattern)
        if asset_key in seen:
            continue
        seen.add(asset_key)
        relative = Path(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("supplement figure path must remain relative")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("supplement figure path escapes its artifact root")
        links.append(
            {
                "club": club,
                "kind": kind,
                "presentation": presentation,
                "source_protocol": protocol,
                "source_protocols": protocols,
                "source_cells": cells,
                "physical_pattern": pattern,
                "comparison_kind": entry.get("comparison_kind"),
                "source_kind": entry["source_kind"],
                "relative_path": path.relative_to(report_root).as_posix(),
            }
        )
    return links


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
            primary_figures,
            primary,
            report_root=report_root,
            protocol="original_primary",
        ),
        *verified_original_figure_links(
            control_figures,
            control,
            report_root=report_root,
            protocol="original_control",
        ),
    ]
    figure_inventories = [
        verified_original_figure_inventory(
            primary_figures,
            primary,
            report_root=report_root,
            protocol="original_primary",
        ),
        verified_original_figure_inventory(
            control_figures,
            control,
            report_root=report_root,
            protocol="original_control",
        ),
    ]
    ledger = supplement["figure_ledger"]
    if ledger.get("schema") != "matched-supplement-figure-ledger/1":
        raise ValueError("qualified supplement figure inventory differs")
    links.extend(
        supplement_figure_links(
            ledger["figures"],
            representative_effects,
            supplement_dir=supplement_dir,
            report_root=report_root,
        )
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
