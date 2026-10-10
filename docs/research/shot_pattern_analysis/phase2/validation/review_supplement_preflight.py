"""Read-only independent replay of saved supplemental preflight contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from docs.research.shot_pattern_analysis.phase2.validation.supplement_audit_protocol import (
    validate_control_aliases,
    validate_supplement_plan,
)
from docs.research.shot_pattern_analysis.phase2.validation.supplemental_refinement import (
    COMPARISONS,
    REFINEMENT_SETTINGS,
    load_qualified_nominals,
    tail_cases,
)
from src.tools.shot_pattern_analysis.matched_export import matched_source_snapshot
from src.tools.shot_pattern_analysis.matched_report import load_matrix_report


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    phase = Path("docs/research/shot_pattern_analysis/phase2")
    validation = phase / "validation"
    output = validation / "supplement_independent_preflight_review.json"
    assert not output.exists()
    inputs_path = validation / "supplement_refinement_inputs_v1.json"
    refinement_path = validation / "supplement_refinement_v1.json"
    execution = matched_source_snapshot()
    files = [
        inputs_path,
        refinement_path,
        phase / "source_transition_supplement.json",
        validation / "supplement_pilot_aliases.json",
        validation / "supplement_composition.json",
        validation / "SUPPLEMENT_COMPOSITION_TESTS.txt",
        Path(__file__),
    ]
    inputs, nominals = load_qualified_nominals(inputs_path)
    assert inputs["execution"] == execution and len(nominals) == 9
    audited_rows = 0
    protocols = {}
    for protocol, entry in inputs["pilots"].items():
        root = (validation / entry["root"]).resolve()
        plan_path = root / "matrix_plan.json"
        plan = json.loads(plan_path.read_text())
        review = validate_supplement_plan(plan, "pilot", phase)
        audit_path = (validation / entry["audit"]["path"]).resolve()
        audit = json.loads(audit_path.read_text())
        assert sha(audit_path) == entry["audit"]["sha256"]
        assert audit["pass"] and audit["complete"] and audit["failures"] == []
        assert audit["execution"] == execution and audit["plan_sha256"] == sha(
            plan_path
        )
        assert audit["script_sha256"] == sha(validation / "independent_final_audit.py")
        assert audit["protocol_auditor_sha256"] == sha(
            validation / "supplement_audit_protocol.py"
        )
        for path, digest in audit["audit_dependency_sha256"].items():
            assert sha(Path(path)) == digest
        report = load_matrix_report(root)
        assert set(audit["cells"]) == set(plan["cells"])
        assert audit["audited_cells"] == audit["planned_cells"] == len(plan["cells"])
        for cell in report["cells"]:
            record = audit["cells"][cell["cell"]]
            assert cell["status"] == record["status"] == "completed"
            assert record["rows"] == 96
            assert (
                report["cell_manifest_sha256"][cell["cell"]]
                == record["manifest_sha256"]
            )
        audited_rows += sum(c["rows"] for c in audit["cells"].values())
        files.extend([plan_path, audit_path])
        protocols[protocol] = review
    assert audited_rows == 2496
    alias_review = validate_control_aliases(
        validation / "supplement_pilot_aliases.json", "pilot", phase
    )
    assert alias_review["verified_aliases"] == 4 and alias_review["new_executions"] == 0
    refinement = json.loads(refinement_path.read_text())
    assert refinement["execution"] == execution and refinement["inputs_sha256"] == sha(
        inputs_path
    )
    assert refinement["script_sha256"] == sha(validation / "supplemental_refinement.py")
    assert refinement["comparison_helper_sha256"] == sha(
        validation / "independent_preflight.py"
    )
    assert refinement["settings_steps_dt"] == [list(x) for x in REFINEMENT_SETTINGS]
    assert (
        refinement["gate_m"] == 0.05
        and refinement["pass"]
        and refinement["failures"] == []
    )
    expected = [(nominal, case) for nominal in nominals for case in tail_cases()]
    assert refinement["cases"] == len(refinement["rows"]) == len(expected) == 333
    maxima = dict.fromkeys(COMPARISONS, 0.0)
    fine_residual = 0.0
    unsupported = 0
    for row, (nominal, case) in zip(refinement["rows"], expected, strict=True):
        assert all(row[key] == value for key, value in nominal.items())
        assert row["case"] == case and len(row["runs"]) == 6
        assert set(row["comparisons"]) == set(COMPARISONS)
        for name, (a, b) in COMPARISONS.items():
            left, right = row["runs"][a], row["runs"][b]
            observed = row["comparisons"][name]
            if left["status"] == right["status"] == "valid":
                delta = math.dist(left["endpoint_m"], right["endpoint_m"])
                assert math.isfinite(delta) and delta <= 0.05
                assert observed == {
                    "status": "evaluated",
                    "change_m": delta,
                    "pass": True,
                }
                maxima[name] = max(maxima[name], delta)
            else:
                assert left["status"] == right["status"] == "unsupported"
                assert left["reason"] == right["reason"]
                assert observed == {"status": "same_unsupported_domain", "pass": None}
                unsupported += 1
        if case["name"] == "nominal":
            delta = math.dist(row["runs"][4]["endpoint_m"], [row["target_x_m"], 0])
            assert delta == row["nominal_fine_residual_m"] and delta <= 0.05
            fine_residual = max(fine_residual, delta)
    composition = json.loads((validation / "supplement_composition.json").read_text())
    assert composition["tests"]["passed"] == 97
    assert (
        sha(validation / composition["tests"]["log"]) == composition["tests"]["sha256"]
    )
    files.extend(
        [
            validation / "supplemental_refinement.py",
            validation / "independent_preflight.py",
            validation / "independent_final_audit.py",
            validation / "supplement_audit_protocol.py",
        ]
    )
    assert matched_source_snapshot() == execution
    result = {
        "status": "passed_saved_preflight_review",
        "execution": execution,
        "protocol_reviews": protocols,
        "audited_rows": audited_rows,
        "alias_review": alias_review,
        "refinement_cases": 333,
        "nominals": 9,
        "gate_m": 0.05,
        "comparison_maxima_m": maxima,
        "fine_nominal_residual_max_m": fine_residual,
        "same_unsupported_comparisons": unsupported,
        "evidence_sha256": {str(path): sha(path) for path in files},
        "limitations": [
            "Saved numerical receipts independently checked; no new flights executed.",
            "Finite selected tails do not certify unbounded Gaussian tails or empirical accuracy.",
            "Visual pilot gate and final execution acceptance remain separate.",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
