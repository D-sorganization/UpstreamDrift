"""Reproduce one saved driver pair; this never qualifies the full study."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from src.tools.shot_pattern_analysis.matched_export import matched_source_snapshot
from src.tools.shot_pattern_analysis.matched_report import load_validated_cell
from src.tools.shot_pattern_analysis.matched_sensitivity import _supported_endpoints
from src.tools.shot_pattern_analysis.matched_pair_social import (
    write_pair_social_density,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("main_worktree", type=Path)
parser.add_argument("output_dir", type=Path)
args = parser.parse_args()
main = args.main_worktree.resolve()
root = main / "docs/research/shot_pattern_analysis/phase2/final_v1"
transition_path = (
    main
    / "docs/research/shot_pattern_analysis/phase2/source_transition_supplement.json"
)
transition = json.loads(transition_path.read_text())
execution = matched_source_snapshot()
assert execution == transition["new_execution"], (
    "Accepted presentation execution changed"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


source = Path("src/tools/shot_pattern_analysis/matched_pair_social.py")
source_hash = sha(source)
report_sources = {
    str(path): sha(path)
    for path in (
        source,
        Path(__file__),
        Path("src/tools/shot_pattern_analysis/matched_report.py"),
        Path("src/tools/shot_pattern_analysis/matched_sensitivity.py"),
    )
}
plan_path = root / "matrix_plan.json"
plan_hash = sha(plan_path)
plan = json.loads(plan_path.read_text())
names = [f"driver_lean5_curve1_rotation{sd}_combined2" for sd in (1, 2)]
refs: list[dict[str, Any]] = []
loaded = []
for name in names:
    manifest = root / name / "manifest.json"
    before = sha(manifest)
    summary = load_validated_cell(root / name, plan["cells"][name], plan["execution"])
    assert summary["status"] == "completed"
    assert summary["config"]["n_shots"] == 10000
    assert max(v["residual_m"] for v in summary["nominals"].values()) <= 0.05
    endpoints, invalid = _supported_endpoints(root / name / "shots.csv.gz", 10000)
    loaded.append(endpoints)
    refs.append({"cell": name, "manifest_sha256": before, "invalid_counts": invalid})
    assert sha(manifest) == before
common = sorted(
    set.intersection(
        *(
            {i for i, _, _ in endpoint_map[p]}
            for endpoint_map in loaded
            for p in ("Straight", "Draw", "Fade")
        )
    )
)
assert len(common) == 9939
point_sets = {}
for label, endpoint_map in zip(
    ("Shaft Twist SD 1°", "Shaft Twist SD 2°"), loaded, strict=True
):
    for p in ("Straight", "Draw", "Fade"):
        values = {i: (x, y) for i, x, y in endpoint_map[p]}
        point_sets[f"{p} · {label}"] = [values[i] for i in common]
out = args.output_dir.resolve()
assert not out.exists(), "Use a fresh destination"
outputs = write_pair_social_density(
    point_sets,
    out / "driver_sd_comparison",
    condition_labels=("Shaft Twist SD 1°", "Shaft Twist SD 2°"),
    scenario="Lean 5° · Curve ×1 · Path SD 0.5° · Speed SD 2 mph · Strike SD 0.25 in/Axis",
    social_club="Driver",
    target_point_m=(summary["target_x_m"], 0.0),
    total_n=10000,
)
assert len(outputs) == 9
for ref in refs:
    name = ref["cell"]
    load_validated_cell(root / name, plan["cells"][name], plan["execution"])
    assert sha(root / name / "manifest.json") == ref["manifest_sha256"]
assert sha(plan_path) == plan_hash
assert sha(source) == source_hash
assert all(sha(Path(path)) == digest for path, digest in report_sources.items())
assert matched_source_snapshot() == execution
receipt = {
    "scope": "Actual Saved 10,000-Shot Pair Component; Visual Review Pending, Not Full-Study Acceptance",
    "source_transition_sha256": sha(transition_path),
    "execution": execution,
    "wrapper_sha256": source_hash,
    "report_sources": report_sources,
    "plan_sha256": plan_hash,
    "inputs": refs,
    "common_six_n": len(common),
    "outputs": {k: {"path": str(p), "sha256": sha(p)} for k, p in outputs.items()},
}
(out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
sys.stdout.write(
    json.dumps(
        {"output": str(out), "common_six_n": len(common), "assets": len(outputs)}
    )
)
