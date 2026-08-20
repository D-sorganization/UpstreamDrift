"""Publish validated structural figure data and its reviewer-facing figure."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from scripts.research.proximal_distal_energy.articulated_structural_cell_evidence import (
    Array,
    load_structural_cell_evidence,
)
from scripts.research.proximal_distal_energy.articulated_structural_figure import (
    render_structural_figure,
)
from scripts.research.proximal_distal_energy.articulated_structural_figure_data import (
    PackKey,
    build_structural_figure_data,
    validate_structural_figure_data,
    write_structural_figure_data,
)
from scripts.research.proximal_distal_energy.articulated_structural_result import (
    validate_structural_propagation_bundle_against_plan,
)


def _load_bound_packs(
    result_path: Path, result: dict[str, Any]
) -> dict[PackKey, dict[str, Array]]:
    packs = {}
    for corner in result["corners"]:
        artifact = PurePosixPath(corner["cell_evidence_artifact"])
        key = (corner["corner_id"], corner["pathway"])
        packs[key] = load_structural_cell_evidence(
            result_path.parent.joinpath(*artifact.parts)
        )
    return packs


def publish_structural_figure_bundle(
    *,
    result_path: Path,
    plan_path: Path,
    figure_data_output: Path,
    figure_output: Path,
) -> dict[str, Any]:
    """Validate every authority and pack before emitting both publication assets."""

    if figure_data_output.suffix.lower() != ".json":
        raise ValueError("structural figure data output must be JSON")
    if figure_output.suffix.lower() not in (".svg", ".pdf"):
        raise ValueError("structural figure output must be SVG or PDF")
    if figure_data_output.resolve() == figure_output.resolve():
        raise ValueError("structural publication outputs must be distinct")
    result = validate_structural_propagation_bundle_against_plan(result_path, plan_path)
    packs = _load_bound_packs(result_path, result)
    record = build_structural_figure_data(result, packs)
    write_structural_figure_data(record, figure_data_output)
    validate_structural_figure_data(figure_data_output, result["result_sha256"])
    render_structural_figure(record, figure_output)
    return record


__all__ = ["publish_structural_figure_bundle"]
