"""Build a source-bound narrative from immutable matched-shot matrix results.

Usage: python -m scripts.research.shot_pattern_analysis.build_final_report \
    PRIMARY_ROOT CONTROL_ROOT REPORT.md
The input cells are validated by the public matched-report and scoring APIs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


from .final_report_inputs import (
    InterpretationAssets,
    assemble_qualified_interpretation,
    profile_from_cell as _profile,
    verified_original_figure_inventory,
    verified_original_figure_links,
)
from .report_renderer import (
    PATTERNS,
    _count_matrix,
    _interval,
    render_cell_inventory,
    _direction_counts,
    _effect_summary,
    _profile_control_lines,
    _experimental_control_lines,
    _anchor_lines,
    _selected_comparison_lines,
    _resolved_selected_comparison_lines,
    render_club_appendices,
    _assert_markdown_output_sizes,
    _sensitivity_lines,
    _lean_lines,
    render_lean_inventory,
    _bounds_control_lines,
    _conclusions_and_synthesis_lines,
    _qualified_interpretation_lines,
    render_final_report,
)


def assert_matrix_inputs_unchanged(root: Path, report: dict) -> None:
    """Reject a report if source-bound input files changed during aggregation."""
    root = Path(root)
    paths = {"matrix_plan.json": report["plan_sha256"]}
    paths.update(
        {
            f"{name}/manifest.json": digest
            for name, digest in report["cell_manifest_sha256"].items()
        }
    )
    for name, digest in paths.items():
        path = root / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise RuntimeError(f"matrix input changed during report assembly: {name}")


def validate_final_plans(
    primary: dict, control: dict, *, common_targets: dict[str, float]
) -> None:
    """Require the exact accepted primary/control factorial protocols."""
    if (
        primary.get("stage") != "final"
        or primary.get("planned_cells") != 324
        or len(primary.get("cells", [])) != 324
    ):
        raise ValueError("primary final protocol requires 324 cells")
    if (
        control.get("stage") != "final"
        or control.get("planned_cells") != 72
        or len(control.get("cells", [])) != 72
    ):
        raise ValueError("control final protocol requires 72 cells")
    from src.tools.shot_pattern_analysis.matched_report import (
        validate_original_matrix_protocol,
    )

    validate_original_matrix_protocol(primary, protocol="original_primary")
    validate_original_matrix_protocol(
        control,
        protocol="original_control",
        common_targets=common_targets,
    )


def _write_atomic(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        staged = Path(handle.name)
        handle.write(content)
    try:
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def _report_source_hashes() -> dict[str, str]:
    """Bind the public interpretation, qualification, and builder implementations."""
    repository = Path(__file__).resolve().parents[3]
    sources = (
        Path(__file__).resolve(),
        repository / "scripts/research/shot_pattern_analysis/final_report_inputs.py",
        repository / "scripts/research/shot_pattern_analysis/report_synthesis.py",
        repository / "scripts/research/shot_pattern_analysis/report_renderer.py",
        repository / "scripts/research/shot_pattern_analysis/report_tables.py",
        repository
        / "scripts/research/shot_pattern_analysis/report_publication_layout.py",
        repository / "scripts/research/shot_pattern_analysis/supplement_publication.py",
        repository
        / "scripts/research/shot_pattern_analysis/supplement_parent_verifier.py",
        repository / "src/tools/shot_pattern_analysis/matched_interpretation.py",
        repository / "src/tools/shot_pattern_analysis/matched_formatting.py",
        repository / "src/tools/shot_pattern_analysis/matched_report.py",
        repository / "src/tools/shot_pattern_analysis/matched_lean_control.py",
        repository / "src/tools/shot_pattern_analysis/matched_sensitivity.py",
        repository / "src/tools/shot_pattern_analysis/matched_supplement_pipeline.py",
        repository / "src/tools/shot_pattern_analysis/matched_supplement_report.py",
        repository / "src/tools/shot_pattern_analysis/matched_supplement_figures.py",
    )
    return {
        path.relative_to(repository).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sources
    }


def final_publication_receipt(
    *,
    output_paths: tuple[Path, ...],
    source_sha256: dict[str, str],
    primary_plan_sha256: str,
    control_plan_sha256: str,
    figure_manifest_sha256: dict[str, str],
    supplement_receipt_sha256: str,
    supplement_publication_identity: dict | None = None,
    archive_provenance: Mapping[str, str | None] | None = None,
) -> dict:
    """Bind every editable report output to its validated input identities."""
    if len({path.parent.resolve() for path in output_paths}) != 1:
        raise ValueError("publication outputs must share one directory")
    for path in output_paths:
        if not path.is_file():
            raise ValueError(f"publication output is missing: {path.name}")
    return {
        "schema": "matched-final-publication-receipt/1",
        "source_sha256": dict(source_sha256),
        "primary_plan_sha256": primary_plan_sha256,
        "control_plan_sha256": control_plan_sha256,
        "figure_manifest_sha256": dict(figure_manifest_sha256),
        "supplement_receipt_sha256": supplement_receipt_sha256,
        "supplement_publication_identity": supplement_publication_identity,
        "supersedes_publication_receipt_sha256": (archive_provenance or {}).get(
            "supersedes_publication_receipt_sha256"
        ),
        "source_archive_manifest_sha256": (archive_provenance or {}).get(
            "source_archive_manifest_sha256"
        ),
        "output_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output_paths
        },
    }


def _split_archive_provenance(report_root: Path) -> tuple[str | None, str | None]:
    """Compatibility wrapper for archived report-split provenance validation."""
    from .report_publication_layout import split_archive_provenance

    return split_archive_provenance(report_root)


@dataclass(frozen=True)
class PublicationInputContext:
    """Typed source identities checked immediately before report publication."""

    primary_root: Path
    primary: dict
    control_root: Path
    control: dict
    primary_figures: Path
    control_figures: Path
    figure_manifest_sha256: dict[Path, str]
    supplement_dir: Path
    supplement_receipt_sha256: str
    source_sha256: dict[str, str]
    report_root: Path


def assert_publication_inputs_unchanged(
    context: PublicationInputContext | None = None, **legacy_arguments: Any
) -> None:
    """Revalidate inputs, accepting legacy keyword arguments for compatibility."""
    if context is None:
        context = PublicationInputContext(**legacy_arguments)
    elif legacy_arguments:
        raise TypeError("pass a publication context or legacy fields, not both")
    _assert_publication_inputs_unchanged(context)


def _assert_publication_inputs_unchanged(context: PublicationInputContext) -> None:
    """Revalidate every source-bound input before writing any publication output."""
    from .supplement_publication import load_verified_supplement_publication

    for root, report in (
        (context.primary_root, context.primary),
        (context.control_root, context.control),
    ):
        assert_matrix_inputs_unchanged(root, report)
    if _report_source_hashes() != context.source_sha256:
        raise RuntimeError("report interpretation source changed during assembly")
    if any(
        hashlib.sha256((root / "figure_manifest.json").read_bytes()).hexdigest()
        != digest
        for root, digest in context.figure_manifest_sha256.items()
    ):
        raise RuntimeError("figure manifest changed during report assembly")
    for root, report, protocol in (
        (context.primary_figures, context.primary, "original_primary"),
        (context.control_figures, context.control, "original_control"),
    ):
        verified_original_figure_links(
            root, report, report_root=context.report_root, protocol=protocol
        )
        verified_original_figure_inventory(
            root, report, report_root=context.report_root, protocol=protocol
        )
    verified = load_verified_supplement_publication(context.supplement_dir)
    if (
        verified["publication_identity"]["receipt_sha256"]
        != context.supplement_receipt_sha256
    ):
        raise RuntimeError("supplement receipt changed during report assembly")
    if _report_source_hashes() != context.source_sha256:
        raise RuntimeError(
            "report interpretation source changed during final validation"
        )


def load_supplement_at_stable_receipt(directory: Path) -> tuple[dict, str]:
    """Bind validated values to the same receipt observed before their load."""
    from .supplement_publication import load_verified_supplement_publication

    supplement = load_verified_supplement_publication(directory)
    return supplement, supplement["publication_identity"]["receipt_sha256"]


@dataclass(frozen=True)
class ReportBuildPaths:
    primary_root: Path
    control_root: Path
    destination: Path
    report_root: Path
    supplement_dir: Path
    primary_figures: Path
    control_figures: Path


@dataclass(frozen=True)
class LoadedReportInputs:
    sources_before: dict[str, str]
    figure_manifests_before: dict[Path, str]
    primary: dict
    control: dict
    supplement: dict
    supplement_receipt_before: str
    common_targets: dict[str, float]


def _resolve_report_paths(
    primary_root: Path,
    control_root: Path,
    destination: Path,
    supplement_dir: Path | None,
    primary_figures: Path | None,
    control_figures: Path | None,
) -> ReportBuildPaths:
    primary_root, control_root, destination = map(
        Path, (primary_root, control_root, destination)
    )
    report_root = destination.parent.resolve()
    return ReportBuildPaths(
        primary_root=primary_root,
        control_root=control_root,
        destination=destination,
        report_root=report_root,
        supplement_dir=Path(supplement_dir or report_root / "supplement_report_v1"),
        primary_figures=Path(primary_figures or report_root / "final_figures_v1"),
        control_figures=Path(
            control_figures or report_root / "control_final_figures_v1"
        ),
    )


def _load_report_inputs(paths: ReportBuildPaths) -> LoadedReportInputs:
    from src.tools.shot_pattern_analysis.matched_report import (
        load_matrix_report,
        load_reviewed_common_targets,
    )

    sources_before = _report_source_hashes()
    figure_roots = (paths.primary_figures, paths.control_figures)
    figure_manifests_before = {
        root: hashlib.sha256((root / "figure_manifest.json").read_bytes()).hexdigest()
        for root in figure_roots
    }
    primary = load_matrix_report(paths.primary_root)
    control = load_matrix_report(paths.control_root)
    supplement, supplement_receipt_before = load_supplement_at_stable_receipt(
        paths.supplement_dir
    )
    common_targets, _ = load_reviewed_common_targets(paths.primary_root.parent)
    validate_final_plans(primary, control, common_targets=common_targets)
    return LoadedReportInputs(
        sources_before=sources_before,
        figure_manifests_before=figure_manifests_before,
        primary=primary,
        control=control,
        supplement=supplement,
        supplement_receipt_before=supplement_receipt_before,
        common_targets=common_targets,
    )


def _analyze_report_inputs(
    paths: ReportBuildPaths, loaded: LoadedReportInputs
) -> tuple[dict, dict[str, dict]]:
    from src.tools.shot_pattern_analysis.matched_lean_control import analyze_across_lean
    from src.tools.shot_pattern_analysis.matched_sensitivity import (
        analyze_saved_sensitivity,
    )

    lean = analyze_across_lean(paths.control_root)
    sensitivity: dict[str, dict] = {}
    for club in ("driver", "seven_iron", "pitching_wedge"):
        candidates = [
            cell
            for cell in loaded.primary["cells"]
            if cell["status"] == "completed"
            and cell["config"]["club"]["club_id"] == club
            and cell["config"]["club"]["shaft_lean_deg"] == 0
            and cell["config"]["curve_scale"] == 1
            and cell["config"]["noise"]["standard_deviations"][0] == 1
            and _profile(cell["cell"]) == "combined2"
        ]
        if len(candidates) == 1:
            name = candidates[0]["cell"]
            sensitivity[name] = analyze_saved_sensitivity(
                paths.primary_root / name, paths.primary_root / "matrix_plan.json"
            )
    return lean, sensitivity


def _assemble_interpretation(
    paths: ReportBuildPaths, loaded: LoadedReportInputs
) -> dict:
    assets = InterpretationAssets(
        primary_figures=paths.primary_figures,
        control_figures=paths.control_figures,
        supplement_dir=paths.supplement_dir,
        report_root=paths.report_root,
    )
    return assemble_qualified_interpretation(
        paths.primary_root,
        loaded.primary,
        paths.control_root,
        loaded.control,
        loaded.supplement,
        assets=assets,
    )


def _publish_report_outputs(
    paths: ReportBuildPaths,
    loaded: LoadedReportInputs,
    lean: dict,
    sensitivity: dict[str, dict],
    interpretation: dict,
) -> Path:
    destination = paths.destination
    rendered = render_final_report(
        loaded.primary,
        loaded.control,
        lean,
        sensitivity,
        interpretation=interpretation,
        report_filename=destination.name,
    )
    appendix_paths = {
        club: destination.with_name(f"{destination.stem}_{club.upper()}.md")
        for club in ("driver", "seven_iron", "pitching_wedge")
    }
    appendix_contents = render_club_appendices(
        interpretation, report_filename=destination.name
    )
    _assert_markdown_output_sizes(
        {
            destination.name: rendered,
            **{
                appendix_paths[club].name: appendix_contents[club]
                for club in appendix_paths
            },
        }
    )
    assert_publication_inputs_unchanged(
        PublicationInputContext(
            primary_root=paths.primary_root,
            primary=loaded.primary,
            control_root=paths.control_root,
            control=loaded.control,
            primary_figures=paths.primary_figures,
            control_figures=paths.control_figures,
            figure_manifest_sha256=loaded.figure_manifests_before,
            supplement_dir=paths.supplement_dir,
            supplement_receipt_sha256=loaded.supplement_receipt_before,
            source_sha256=loaded.sources_before,
            report_root=paths.report_root,
        )
    )
    cell_inventory_path = destination.with_name("REPORT_CELLS.csv")
    lean_inventory_path = destination.with_name("REPORT_LEAN.csv")
    receipt_path = destination.with_name("REPORT_PUBLICATION_RECEIPT.json")
    previous_receipt_sha256 = (
        hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        if receipt_path.is_file()
        else None
    )
    archived_receipt_sha256, archive_manifest_sha256 = _split_archive_provenance(
        paths.report_root
    )
    supersedes_receipt_sha256 = archived_receipt_sha256 or previous_receipt_sha256
    _write_atomic(
        cell_inventory_path, render_cell_inventory(loaded.primary, loaded.control)
    )
    _write_atomic(lean_inventory_path, render_lean_inventory(lean))
    _write_atomic(destination, rendered)
    for club, path in appendix_paths.items():
        _write_atomic(path, appendix_contents[club])
    publication_receipt = final_publication_receipt(
        output_paths=(
            destination,
            cell_inventory_path,
            lean_inventory_path,
            *appendix_paths.values(),
        ),
        source_sha256=loaded.sources_before,
        primary_plan_sha256=loaded.primary["plan_sha256"],
        control_plan_sha256=loaded.control["plan_sha256"],
        figure_manifest_sha256={
            root.name: digest for root, digest in loaded.figure_manifests_before.items()
        },
        supplement_receipt_sha256=interpretation["supplement"]["receipt_sha256"],
        supplement_publication_identity=interpretation["supplement"].get(
            "publication_identity"
        ),
        archive_provenance={
            "supersedes_publication_receipt_sha256": supersedes_receipt_sha256,
            "source_archive_manifest_sha256": archive_manifest_sha256,
        },
    )
    _write_atomic(
        receipt_path,
        json.dumps(publication_receipt, indent=2, sort_keys=True) + "\n",
    )
    return destination


def build_final_report(
    primary_root: Path,
    control_root: Path,
    destination: Path,
    *,
    supplement_dir: Path | None = None,
    primary_figures: Path | None = None,
    control_figures: Path | None = None,
) -> Path:
    """Validate saved matrices, then atomically publish editable Markdown."""
    paths = _resolve_report_paths(
        primary_root,
        control_root,
        destination,
        supplement_dir,
        primary_figures,
        control_figures,
    )
    loaded = _load_report_inputs(paths)
    lean, sensitivity = _analyze_report_inputs(paths, loaded)
    interpretation = _assemble_interpretation(paths, loaded)
    return _publish_report_outputs(paths, loaded, lean, sensitivity, interpretation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_root", type=Path)
    parser.add_argument("control_root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--supplement-dir", type=Path)
    parser.add_argument("--primary-figures", type=Path)
    parser.add_argument("--control-figures", type=Path)
    arguments = parser.parse_args()
    print(
        build_final_report(
            arguments.primary_root,
            arguments.control_root,
            arguments.destination,
            supplement_dir=arguments.supplement_dir,
            primary_figures=arguments.primary_figures,
            control_figures=arguments.control_figures,
        )
    )


if __name__ == "__main__":
    main()
