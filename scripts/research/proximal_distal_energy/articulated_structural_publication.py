"""Publish validated structural figure data and its reviewer-facing figure."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from collections.abc import Sequence
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

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
FIGURES = ROOT / "docs/research/proximal_distal_energy_transfer/figures"
DEFAULT_RESULT = DATA / "articulated_structural_propagation_result.json"
DEFAULT_PLAN = DATA / "articulated_structural_propagation_plan.json"
DEFAULT_FIGURE_DATA = DATA / "articulated_structural_figure_data.json"
DEFAULT_FIGURE = FIGURES / "articulated_structural_sensitivity.svg"


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


def main(argv: Sequence[str] | None = None) -> None:
    """Publish the canonical bundle or explicitly supplied review artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--figure-data", type=Path, default=DEFAULT_FIGURE_DATA)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args(argv)
    publish_structural_figure_bundle(
        result_path=args.result,
        plan_path=args.plan,
        figure_data_output=args.figure_data,
        figure_output=args.figure,
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "publish_structural_figure_bundle"]
