"""Render saved three-club social pairs and measure decorated-text clearance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from matplotlib.figure import Figure

from src.tools.shot_pattern_analysis.matched_export import matched_source_snapshot
from docs.research.shot_pattern_analysis.phase2.review_followups import (
    footer_review_helpers,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(main: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("review output must be new")
    output.mkdir(parents=True)
    phase = main / "docs/research/shot_pattern_analysis/phase2"
    data = phase / "final_v1"
    plan = json.loads((data / "matrix_plan.json").read_text())
    execution = matched_source_snapshot()
    accepted = json.loads((phase / "source_transition_supplement.json").read_text())[
        "new_execution"
    ]
    key = "src/tools/shot_pattern_analysis/academic_plots.py"
    assert {k: v for k, v in execution.items() if k != "source_sha256"} == {
        k: v for k, v in accepted.items() if k != "source_sha256"
    }
    assert set(execution["source_sha256"]) == set(accepted["source_sha256"])
    assert all(
        v == execution["source_sha256"][k]
        for k, v in accepted["source_sha256"].items()
        if k != key
    )
    source_paths = sorted(Path("src/tools/shot_pattern_analysis").glob("*.py")) + [
        Path(__file__).resolve(),
        Path(footer_review_helpers.__file__).resolve(),
    ]
    sources = {str(p): sha(p) for p in source_paths}
    inputs: dict[str, dict[str, str]] = {}
    rendered: list[footer_review_helpers.FigureGeometry] = []
    artifacts: dict[str, str] = {}
    previews: dict[str, str] = {}
    original_save = Figure.savefig

    def save(figure, path, *args, **kwargs):
        if str(path).endswith(".png"):
            rendered.append(footer_review_helpers.inspect_figure(figure, path))
        return original_save(figure, path, *args, **kwargs)

    Figure.savefig = save  # type: ignore[method-assign,assignment]
    try:
        for club in ("driver", "seven_iron", "pitching_wedge"):
            footer_review_helpers.render_club(
                club, data, output, plan, inputs, artifacts, previews, sha
            )
    finally:
        Figure.savefig = original_save  # type: ignore[method-assign]
    assert all(sha(Path(p)) == digest for p, digest in sources.items())
    assert matched_source_snapshot() == execution
    assert all(
        sha(data / name / p) == digest
        for name, files in inputs.items()
        for p, digest in files.items()
    )
    assert all(
        sha(output / p) == digest for p, digest in {**artifacts, **previews}.items()
    )
    result = {
        "scope": "Saved-Data Social-Pair Footer Repair; No Simulation or Final Study Acceptance",
        "status": "visual_review_pending",
        "sources": sources,
        "inputs": inputs,
        "execution": execution,
        "only_accepted_execution_delta": key,
        "artifacts": artifacts,
        "previews": previews,
        "geometry": rendered,
    }
    (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.main, args.output)
