"""Source-bound figure and interpretation inputs for the final report."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


def profile_from_cell(name: str) -> str:
    # The validated plan is the authority for the cell name; parse its suffix
    # without importing the simulation runner into the pure report formatter.
    match = re.search(r"_rotation\d+_(.+)$", name)
    if match is None:
        raise ValueError(f"cell name lacks a noise profile: {name}")
    return match.group(1)


def select_bounds_control_pairs(
    comparisons: list[dict],
) -> tuple[list[dict], list[str]]:
    """Select exact reviewed curve-2, SD2, combined2 fixed-target 0→5 controls."""
    from src.tools.shot_pattern_analysis.matched_interpretation import (
        describe_effect_modification,
    )

    targets = (
        ("driver", "Driver", "original_control/1", "driver-control-speed-bounds/1"),
        (
            "pitching_wedge",
            "Pitching Wedge",
            "pw-primary-face-bounds/1",
            "pw-control-face-bounds/1",
        ),
    )
    selected: list[dict] = []
    unavailable: list[str] = []
    for club, label, protocol_a, protocol_b in targets:
        expected = (
            (protocol_a, f"{club}_lean0_curve2_rotation2_combined2"),
            (protocol_b, f"{club}_lean5_curve2_rotation2_combined2"),
        )
        matches = []
        for pair in comparisons:
            sources = pair.get("sources")
            if (
                pair.get("comparison_kind") != "across_lean"
                or not isinstance(sources, list)
                or len(sources) != 2
            ):
                continue
            if not all(isinstance(source, dict) for source in sources):
                continue
            identity = tuple(
                (source.get("protocol_id"), source.get("cell")) for source in sources
            )
            if identity == expected:
                matches.append(pair)
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous duplicate bounds-control comparison for {club}"
            )
        description = f"{label} 0°→5° (Curve 2, Shaft SD 2°, Combined-2; {protocol_a} → {protocol_b})"
        if not matches:
            unavailable.append(
                f"{description}: Unavailable (absent from verified supplement comparisons)."
            )
            continue
        effect = describe_effect_modification(matches[0])
        if effect["status"] == "unavailable":
            reason = matches[0].get("reason") or "qualified comparison unavailable"
            unavailable.append(f"{description}: Unavailable ({reason}).")
        elif any(
            effect[key] is None
            for key in ("draw_minus_straight_change", "fade_minus_straight_change")
        ):
            scoring = matches[0].get("course_scoring_effect_modification", {})
            reason = (
                scoring.get("reason") or "relative scoring effects were not emitted"
            )
            unavailable.append(f"{description}: Unavailable ({reason}).")
        else:
            selected.append(effect)
    return selected, unavailable


def select_representative_interaction_pairs(
    comparisons: list[dict],
) -> list[dict]:
    """Select the exact 12 draft representative interaction contrasts.

    The qualified draft specifies 12 contrasts across all three clubs, both
    shaft_only and combined2 profiles, SD 1→2° curve ×1, and curve ×1→×2 at SD 2°.
    Pitching wedge curve ×2 uses the verified supplemental pw-primary-face-bounds/1
    protocol. Selection is invariant to input order and fails closed on missing
    or duplicate identities.
    """
    from src.tools.shot_pattern_analysis.matched_interpretation import (
        describe_effect_modification,
    )

    targets: list[tuple[str, str, str, str, str, str, str]] = []
    for club in ("driver", "seven_iron", "pitching_wedge"):
        for profile in ("shaft_only", "combined2"):
            # Contrast 1: SD 1 -> 2, Curve 1
            targets.append(
                (
                    club,
                    profile,
                    "shaft_rotation_sd",
                    f"{club}_lean0_curve1_rotation1_{profile}",
                    f"{club}_lean0_curve1_rotation2_{profile}",
                    "original_control/1",
                    "original_control/1",
                )
            )
            # Contrast 2: Curve 1 -> 2, SD 2
            proto_b = (
                "pw-primary-face-bounds/1"
                if club == "pitching_wedge"
                else "original_control/1"
            )
            targets.append(
                (
                    club,
                    profile,
                    "curve_strength",
                    f"{club}_lean0_curve1_rotation2_{profile}",
                    f"{club}_lean0_curve2_rotation2_{profile}",
                    "original_control/1",
                    proto_b,
                )
            )

    selected: list[dict] = []
    for club, profile, kind, cell_a, cell_b, proto_a, proto_b in targets:
        expected = ((proto_a, cell_a), (proto_b, cell_b))
        matches = []
        for pair in comparisons:
            sources = pair.get("sources")
            if (
                pair.get("comparison_kind") != kind
                or not isinstance(sources, list)
                or len(sources) != 2
            ):
                continue
            if not all(isinstance(s, dict) for s in sources):
                continue
            identity = tuple((s.get("protocol_id"), s.get("cell")) for s in sources)
            if identity == expected:
                matches.append(pair)

        if len(matches) > 1:
            raise ValueError(
                f"ambiguous duplicate comparison for {club} {profile} {kind}: {cell_a} -> {cell_b}"
            )
        if not matches:
            raise ValueError(
                f"missing required representative comparison for {club} {profile} {kind}: {cell_a} -> {cell_b}"
            )

        match = matches[0]
        effect = describe_effect_modification(match)
        if effect.get("status") not in ("available", "conditional_only"):
            raise ValueError(
                f"representative comparison is unavailable: {club} {profile} {kind}"
            )
        if any(
            effect[key] is None
            for key in ("draw_minus_straight_change", "fade_minus_straight_change")
        ):
            raise ValueError(
                f"required relative scoring interaction unavailable: {cell_a}"
            )
        selected.append(effect)

    return selected


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


def _index_source_cells(cells: list[dict], source: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for cell in cells:
        name = cell["cell"]
        if name in indexed:
            raise ValueError(f"duplicate {source} cell: {name}")
        indexed[name] = cell
    return indexed


def _require_population_counts(record: dict, cell_name: str, prefix: str) -> None:
    summary = record.get("summary", {})
    for pattern in ("Straight", "Draw", "Fade"):
        if pattern not in summary or "supported_n" not in summary[pattern]:
            raise ValueError(
                f"{prefix}cell lacks individual support count: {cell_name}"
            )
    paired = record.get("paired_comparisons", {})
    if "paired_n" not in paired:
        raise ValueError(f"{prefix}cell lacks common paired count: {cell_name}")


def _resolve_selected_cell(
    cell_name: str, primary_cell: dict, supplement_cells: dict
) -> dict:
    status = primary_cell.get("status")
    if status == "completed":
        _require_population_counts(primary_cell, cell_name, "")
        return {**primary_cell, "source_protocol": "original_primary/1"}
    if status == "unavailable":
        if cell_name not in supplement_cells:
            raise ValueError(
                f"missing resolved supplement cell for unavailable primary: {cell_name}"
            )
        entry = supplement_cells[cell_name]
        if entry.get("status") != "completed":
            raise ValueError(f"supplement cell is not completed: {cell_name}")
        summary = entry.get("summary", {})
        _require_population_counts(summary, cell_name, "supplement ")
        return {
            **summary,
            "cell": cell_name,
            "source_protocol": entry.get("source_protocol"),
            "original_status": "unavailable",
        }
    raise ValueError(f"cell has invalid status {status}: {cell_name}")


def assemble_resolved_selected_cells(primary: dict, supplement: dict) -> list[dict]:
    """Assemble the resolved 24-cell zero-lean selection with explicit protocols.

    Preserves original primary accounting for the 20 completed primary cells
    (protocol original_primary/1) and integrates the verified supplemental
    cell records for the four resolved PW curve-2 cells (protocol
    pw-primary-face-bounds/1). Guards against missing or duplicate identities
    and asserts separate individual endpoint and common-three scoring counts.
    """
    clubs = ("driver", "seven_iron", "pitching_wedge")
    curves = (1.0, 2.0)
    rotations = (1.0, 2.0)
    profiles = ("shaft_only", "combined2")

    primary_cells = _index_source_cells(primary.get("cells", []), "primary")
    supp_report = supplement.get("report", {})
    supp_unique = _index_source_cells(supp_report.get("unique_cells", []), "supplement")

    resolved: list[dict] = []
    seen: set[str] = set()

    for club in clubs:
        for curve in curves:
            for rot in rotations:
                for prof in profiles:
                    cell_name = (
                        f"{club}_lean0_curve{int(curve)}_rotation{int(rot)}_{prof}"
                    )
                    if cell_name in seen:
                        raise ValueError(f"duplicate requested cell: {cell_name}")
                    seen.add(cell_name)

                    if cell_name not in primary_cells:
                        raise ValueError(f"missing requested primary cell: {cell_name}")

                    resolved.append(
                        _resolve_selected_cell(
                            cell_name, primary_cells[cell_name], supp_unique
                        )
                    )

    if len(resolved) != 24:
        raise ValueError(f"expected exactly 24 resolved cells, got {len(resolved)}")

    from .report_synthesis import validate_resolved_selection

    validate_resolved_selection(resolved)
    return resolved


@dataclass(frozen=True)
class InterpretationAssets:
    """Original/supplemental artifact roots and their publication root."""

    primary_figures: Path
    control_figures: Path
    supplement_dir: Path
    report_root: Path


def _selected_interpretation_cells(primary: dict) -> list[dict]:
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
    return selected


def _original_figure_assets(
    primary: dict, control: dict, assets: InterpretationAssets, report_root: Path
) -> tuple[list[dict], list[str]]:
    links = [
        *verified_original_figure_links(
            assets.primary_figures,
            primary,
            report_root=report_root,
            protocol="original_primary",
        ),
        *verified_original_figure_links(
            assets.control_figures,
            control,
            report_root=report_root,
            protocol="original_control",
        ),
    ]
    figure_inventories = [
        verified_original_figure_inventory(
            assets.primary_figures,
            primary,
            report_root=report_root,
            protocol="original_primary",
        ),
        verified_original_figure_inventory(
            assets.control_figures,
            control,
            report_root=report_root,
            protocol="original_control",
        ),
    ]
    return links, figure_inventories


def _qualified_supplement_identity(
    supplement: dict, supplement_dir: Path
) -> tuple[dict, dict]:
    report = supplement["report"]
    if (
        report.get("schema") != "matched-supplement-report/1"
        or report.get("unique_executed_cells") != 26
    ):
        raise ValueError("additive supplement inventory is incomplete")
    publication = supplement.get("publication_identity")
    if publication is None:
        # Ordinary in-memory callers retain the original receipt contract.
        receipt_path = Path(supplement_dir) / "REPORT_SUPPLEMENT_PIPELINE.json"
        digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        publication = {
            "kind": "pipeline",
            "receipt_filename": receipt_path.name,
            "receipt_sha256": digest,
            "parent_receipt_sha256": digest,
        }
    receipt_path = Path(supplement_dir) / publication["receipt_filename"]
    if (
        hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        != publication["receipt_sha256"]
    ):
        raise ValueError("supplement publication identity changed")
    return report, publication


def assemble_qualified_interpretation(
    primary_root: Path,
    primary: dict,
    control_root: Path,
    control: dict,
    supplement: dict,
    *,
    assets: InterpretationAssets,
) -> dict:
    """Compose selected interpretation from independently qualified inputs."""
    from src.tools.shot_pattern_analysis.matched_interpretation import (
        describe_saved_cell,
        describe_scoring,
    )

    report_root = Path(assets.report_root).resolve()
    selected = _selected_interpretation_cells(primary)
    explained = [
        describe_saved_cell(Path(primary_root) / cell["cell"]) for cell in selected
    ]
    scored = [(cell["cell"], describe_scoring(cell)) for cell in selected]
    comparisons = supplement["comparisons"]
    if comparisons.get("schema") != "matched-supplement-comparisons/1":
        raise ValueError("qualified supplement comparison schema differs")
    resolved_selected_cells = assemble_resolved_selected_cells(primary, supplement)
    representative_effects = select_representative_interaction_pairs(
        comparisons["comparisons"]
    )
    bounds_effects, bounds_unavailable = select_bounds_control_pairs(
        comparisons["comparisons"]
    )
    links, figure_inventories = _original_figure_assets(
        primary, control, assets, report_root
    )
    ledger = supplement["figure_ledger"]
    if ledger.get("schema") != "matched-supplement-figure-ledger/1":
        raise ValueError("qualified supplement figure inventory differs")
    links.extend(
        supplement_figure_links(
            ledger["figures"],
            representative_effects + bounds_effects,
            supplement_dir=assets.supplement_dir,
            report_root=report_root,
        )
    )
    report, publication = _qualified_supplement_identity(
        supplement, assets.supplement_dir
    )
    return {
        "selected_cells": explained,
        "resolved_selected_cells": resolved_selected_cells,
        "scores": scored,
        "effects": representative_effects,
        "bounds_control": {
            "effects": bounds_effects,
            "unavailable": bounds_unavailable,
        },
        "figure_links": links,
        "figure_inventories": figure_inventories,
        "supplement": {
            "status": "qualified",
            "unique_executed_cells": report["unique_executed_cells"],
            "verified_control_aliases": sum(
                row["proof_status"] == "verified" for row in report["aliases"]
            ),
            "receipt_sha256": publication["receipt_sha256"],
            "publication_identity": publication,
            "comparison_inventory": (
                Path(assets.supplement_dir).resolve()
                / "REPORT_SUPPLEMENT_COMPARISONS.csv"
            )
            .relative_to(report_root)
            .as_posix(),
            "figure_inventory": (
                Path(assets.supplement_dir).resolve() / "REPORT_SUPPLEMENT_FIGURES.json"
            )
            .relative_to(report_root)
            .as_posix(),
        },
    }
