"""Render one saved final-size pair for component QA, never full-study acceptance."""

import hashlib
import json
import sys
from pathlib import Path
from src.tools.shot_pattern_analysis.matched_report import load_validated_cell
from src.tools.shot_pattern_analysis.matched_supplement_figures import (
    _render_pair_figures,
)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


root = Path("docs/research/shot_pattern_analysis/phase2/final_v1").resolve()
out = Path(sys.argv[1])
if out.exists():
    raise RuntimeError("Use a new output directory for each source-bound review")
plan = json.loads((root / "matrix_plan.json").read_text())
names = [f"driver_lean5_curve1_rotation{sd}_combined2" for sd in (1, 2)]
source_paths = [
    Path("src/tools/shot_pattern_analysis") / name
    for name in (
        "matched_supplement_figures.py",
        "matched_report.py",
        "density.py",
        "academic_plots.py",
    )
]
sources = {str(p): sha(p) for p in source_paths}
refs = []
for name in names:
    cell = load_validated_cell(root / name, plan["cells"][name], plan["execution"])
    refs.append(
        {
            "cell": name,
            "status": cell["status"],
            "config": cell["config"],
            "summary": cell,
            "source_root": str(root),
            "source_protocol": "original_primary/1",
            "source_cell_manifest_sha256": sha(root / name / "manifest.json"),
        }
    )
pair = {
    "first": refs[0],
    "second": refs[1],
    "comparison_kind": "shaft_rotation_sd",
    "target_x_m": refs[0]["summary"]["target_x_m"],
}
ledger = _render_pair_figures(pair, out)
assert len(ledger) == 8, [row.get("figure_kind") for row in ledger]
assert len({row["relative_path"] for row in ledger}) == 8
for row in ledger:
    p = out / row["relative_path"]
    assert sha(p) == row["sha256"]
    if row["figure_kind"].startswith("density"):
        assert row["source_kind"] == "six_population_common_index_comparison"
        assert row["common_supported_n"] == 9939
    else:
        assert row["source_kind"] == "saved_illustrative_trajectory_samples"
        assert row.get("common_supported_n") is None
for ref in refs:
    load_validated_cell(root / ref["cell"], ref["config"], plan["execution"])
    assert (
        sha(root / ref["cell"] / "manifest.json") == ref["source_cell_manifest_sha256"]
    )
assert sources == {str(p): sha(p) for p in source_paths}, (
    "Figure source changed during review"
)
(out / "component_receipt.json").write_text(
    json.dumps(
        {
            "scope": "One saved final-size pair; not full-study acceptance or social visual qualification",
            "sources": sources,
            "source_refs": [
                {k: v for k, v in ref.items() if k not in ("summary", "config")}
                for ref in refs
            ],
            "ledger": ledger,
        },
        indent=2,
    )
    + "\n"
)
sys.stdout.write(
    json.dumps({"output": str(out), "assets": len(ledger), "ledger": ledger}, indent=2)
    + "\n"
)
