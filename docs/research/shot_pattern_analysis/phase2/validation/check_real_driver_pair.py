"""Check one real final-size SD pair against independently assembled contrasts.

This component check does not accept an incomplete final study.
"""

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from src.tools.shot_pattern_analysis.matched_lean_control import (
    analyze_explicit_source_pair,
)
from src.tools.shot_pattern_analysis.course_score_vectors import (
    score_course_endpoint_vectors,
)

root = Path("docs/research/shot_pattern_analysis/phase2/final_v1").resolve()
lean = int(sys.argv[1]) if len(sys.argv) > 1 else 0
assert lean in (0, 5), "component check covers only declared lean 0 and 5 cases"
names = [f"driver_lean{lean}_curve1_rotation{sd}_combined2" for sd in (1, 2)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


refs = [
    {
        "source_root": str(root),
        "cell": name,
        "source_protocol": "original_primary/1",
        "execution_kind": "unique",
        "source_plan_sha256": sha(root / "matrix_plan.json"),
        "source_cell_manifest_sha256": sha(root / name / "manifest.json"),
        "source_csv_sha256": sha(root / name / "shots.csv.gz"),
    }
    for name in names
]
target = json.loads((root / names[0] / "summary.json").read_text())["target_x_m"]
actual = analyze_explicit_source_pair(
    refs[0], refs[1], comparison_kind="shaft_rotation_sd", target_x_m=target
)
patterns = ("Straight", "Draw", "Fade")
groups = []
for name in names:
    group: dict[str, dict[int, tuple[float, float]]] = {
        pattern: {} for pattern in patterns
    }
    with gzip.open(root / name / "shots.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            if row["status"] == "supported":
                group[row["pattern"]][int(row["shot_index"])] = (
                    float(row["carry_x_m"]),
                    float(row["carry_y_m"]),
                )
    groups.append(group)
indices = sorted(
    set.intersection(*(set(group[pattern]) for group in groups for pattern in patterns))
)
assert len(indices) == actual["common_supported_n"]
assert len(indices) == 10000 if lean == 0 else 2 <= len(indices) < 10000
vectors = [
    score_course_endpoint_vectors(
        {
            pattern: [(index, *group[pattern][index]) for index in indices]
            for pattern in patterns
        },
        club_id="driver",
        target_x_m=target,
        green_radius_m=15.0,
    ).scores
    for group in groups
]
independent = {
    pattern: float(
        np.mean(
            (vectors[1][pattern] - vectors[1]["Straight"])
            - (vectors[0][pattern] - vectors[0]["Straight"])
        )
    )
    for pattern in ("Draw", "Fade")
}
errors = {
    pattern: abs(
        value - actual["course_scoring_effect_modification"][pattern]["estimate"]
    )
    for pattern, value in independent.items()
}
assert max(errors.values()) <= 1e-12
for ref in refs:
    assert (
        sha(root / ref["cell"] / "manifest.json") == ref["source_cell_manifest_sha256"]
    )
    assert sha(root / ref["cell"] / "shots.csv.gz") == ref["source_csv_sha256"]
result = {
    "schema": "real-saved-pair-qualification/1",
    "status": "passed_component_check",
    "scope": "One completed 10,000-shot driver SD1-to-SD2 pair; not full-matrix or final-study acceptance",
    "shaft_lean_deg": lean,
    "pair_result": actual,
    "independently_assembled_common_n": len(indices),
    "independent_mean_interaction_sg": independent,
    "max_abs_mean_difference_by_pattern": errors,
    "limitation": "Per-shot values use the already parity-qualified public scorer; this check independently assembles sorted common indices and the difference-of-differences, not a separate physical or scoring model.",
    "script_sha256": sha(Path(__file__)),
}
Path(f"/tmp/p2-real-driver-lean{lean}-pair-review.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
sys.stdout.write(
    json.dumps(
        {
            "status": result["status"],
            "common_n": len(indices),
            "means": independent,
            "errors": errors,
        },
        indent=2,
    )
    + "\n"
)
