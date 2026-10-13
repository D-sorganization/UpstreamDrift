"""Predeclared finite-grid supplemental refinement; execution awaits qualified pilots."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from docs.research.shot_pattern_analysis.phase2.validation.independent_preflight import (
    SETTINGS,
    comparison,
)
from docs.research.shot_pattern_analysis.phase2.validation.supplement_audit_protocol import (
    PROTOCOLS,
    load_json_strict,
    validate_supplement_plan,
)
from src.tools.shot_pattern_analysis.matched_config import MatchedConfig
from src.tools.shot_pattern_analysis.matched_export import matched_source_snapshot
from src.tools.shot_pattern_analysis.matched_physics import (
    MatchedShotPhysics,
    UnsupportedShot,
)
from src.tools.shot_pattern_analysis.matched_report import load_matrix_report

GATE_M = 0.05
REFINEMENT_SETTINGS = (*SETTINGS, (128, 0.02))
SD = (2.0, 0.5, 0.89408, 0.00635, 0.00635)
COMPARISONS = {
    "contact_coarse": (0, 1),
    "contact_fine": (1, 2),
    "contact_primary_to_fine": (0, 2),
    "flight_coarse": (3, 2),
    "flight_fine": (2, 4),
    "flight_primary_to_fine": (3, 4),
    "primary_to_fine": (5, 4),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tail_cases() -> list[dict[str, Any]]:
    """37 selected primitive-error cases, not Gaussian joint quantile regions."""
    cases = [{"name": "nominal", "offsets": [0.0] * 5}]
    for axis in range(5):
        for level in (-3, -2, 2, 3):
            offsets = [0.0] * 5
            offsets[axis] = level * SD[axis]
            cases.append({"name": f"axis{axis}_{level:+d}sd", "offsets": offsets})
    signs = ((1, 1, 1, 1, 1), (1, -1, 1, -1, 1), (1, 1, -1, -1, -1), (1, -1, -1, 1, -1))
    for index, sign in enumerate(signs):
        for level in (-3, -2, 2, 3):
            cases.append(
                {
                    "name": f"joint{index}_{level:+d}sd",
                    "offsets": [level * a * b for a, b in zip(sign, SD, strict=True)],
                }
            )
    return cases


def evaluate_case(
    physics: Any, config: MatchedConfig, nominal: dict, offsets: list[float]
) -> dict:
    twist, path, speed, horizontal, vertical = offsets
    runs = []
    for steps, dt in REFINEMENT_SETTINGS:
        actual = replace(
            config,
            club_speed_mps=nominal["speed_mps"] + speed,
            steps_per_compression=steps,
            dt_s=dt,
        )
        try:
            outcome = physics.simulate_delivery(
                nominal["face_deg"],
                nominal["path_deg"] + path,
                actual,
                shaft_rotation_deg=twist,
                contact_horizontal_m=horizontal,
                contact_vertical_m=vertical,
            ).outcome
            endpoint = [outcome.carry_x_m, outcome.carry_y_m]
            if not all(math.isfinite(v) for v in endpoint):
                raise ValueError("nonfinite endpoint")
            runs.append({"status": "valid", "endpoint_m": endpoint})
        except UnsupportedShot as exc:
            runs.append({"status": "unsupported", "reason": exc.reason})
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            runs.append(
                {"status": "numerical_error", "error": f"{type(exc).__name__}: {exc}"}
            )
    return {
        "runs": runs,
        "comparisons": {
            name: comparison(runs[a], runs[b]) for name, (a, b) in COMPARISONS.items()
        },
    }


def _bound_file(base: Path, reference: dict) -> Path:
    path = (base / reference["path"]).resolve()
    if sha(path) != reference["sha256"]:
        raise ValueError("refinement input evidence hash mismatch")
    return path


def load_qualified_nominals(manifest_path: Path) -> tuple[dict, list[dict]]:
    """Reject absent/unqualified pilots before constructing a flight engine."""
    manifest = load_json_strict(manifest_path)
    if manifest.get("schema") != "supplement-refinement-inputs/1":
        raise ValueError("invalid refinement inputs")
    review = load_json_strict(
        _bound_file(manifest_path.parent, manifest["source_review"])
    )
    execution = manifest["execution"]
    if (
        review.get("status") != "independently_accepted"
        or review.get("new_execution") != execution
    ):
        raise ValueError("source-transition review does not qualify this execution")
    if set(manifest["pilots"]) != set(PROTOCOLS):
        raise ValueError("three qualified pilots required")
    evidence = Path(__file__).resolve().parents[1]
    nominals = []
    for protocol, item in manifest["pilots"].items():
        root = (manifest_path.parent / item["root"]).resolve()
        plan_path = root / "matrix_plan.json"
        plan = load_json_strict(plan_path)
        validate_supplement_plan(plan, "pilot", evidence)
        if plan["supplement_protocol"] != protocol or plan["execution"] != execution:
            raise ValueError("pilot execution/protocol mismatch")
        audit = load_json_strict(_bound_file(manifest_path.parent, item["audit"]))
        if not (
            audit.get("pass") is True
            and audit.get("complete") is True
            and audit.get("data_stage") == "pilot"
            and audit.get("execution") == execution
            and audit.get("plan_sha256") == sha(plan_path)
        ):
            raise ValueError("pilot lacks a matching complete independent audit")
        report = load_matrix_report(root)
        if any(
            audit["cells"].get(name, {}).get("manifest_sha256") != digest
            for name, digest in report["cell_manifest_sha256"].items()
        ):
            raise ValueError("pilot manifests differ from independently audited bytes")
        selected = [
            cell
            for cell in report["cells"]
            if cell["cell"].endswith("_rotation2_combined2")
        ]
        if len(selected) != 1 or any(
            cell["status"] != "completed" for cell in report["cells"]
        ):
            raise ValueError("supplemental nominal coverage incomplete")
        chosen = selected[0]
        if any(cell["nominals"] != chosen["nominals"] for cell in report["cells"]):
            raise ValueError("nominals differ across noise profiles")
        if (
            chosen["config"]["club"]["steps_per_compression"] != 128
            or chosen["config"]["club"]["dt_s"] != 0.02
        ):
            raise ValueError("unexpected primary numerical settings")
        for pattern in ("Straight", "Draw", "Fade"):
            nominals.append(
                {
                    "protocol": protocol,
                    "cell": chosen["cell"],
                    "pattern": pattern,
                    "club": chosen["config"]["club"],
                    "nominal": chosen["nominals"][pattern],
                    "target_x_m": chosen["target_x_m"],
                    "plan_sha256": sha(plan_path),
                    "manifest_sha256": sha(root / chosen["cell"] / "manifest.json"),
                }
            )
    return manifest, nominals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refinement receipts are immutable; choose a new output")
    inputs_hash = sha(args.inputs)
    manifest, nominals = load_qualified_nominals(args.inputs)
    before = matched_source_snapshot()
    if before != manifest["execution"]:
        raise ValueError("installed execution differs from qualified pilot")
    physics = MatchedShotPhysics()
    rows = []
    for record in nominals:
        config = MatchedConfig(**record["club"])
        for case in tail_cases():
            result = evaluate_case(physics, config, record["nominal"], case["offsets"])
            if case["name"] == "nominal":
                fine = result["runs"][4]
                result["nominal_fine_residual_m"] = (
                    math.dist(fine["endpoint_m"], [record["target_x_m"], 0])
                    if fine["status"] == "valid"
                    else None
                )
            rows.append({**record, "case": case, **result})
    if before != matched_source_snapshot() or sha(args.inputs) != inputs_hash:
        raise RuntimeError("refinement source/input drift")
    load_qualified_nominals(args.inputs)
    failures = [
        i
        for i, row in enumerate(rows)
        if any(value["pass"] is False for value in row["comparisons"].values())
        or (
            row["case"]["name"] == "nominal"
            and (
                row["nominal_fine_residual_m"] is None
                or row["nominal_fine_residual_m"] > GATE_M
            )
        )
    ]
    receipt = {
        "schema": "supplement-deterministic-refinement/1",
        "execution": before,
        "inputs_sha256": inputs_hash,
        "script_sha256": sha(Path(__file__)),
        "comparison_helper_sha256": sha(
            Path(__file__).with_name("independent_preflight.py")
        ),
        "gate_m": GATE_M,
        "settings_steps_dt": REFINEMENT_SETTINGS,
        "nominals": 9,
        "cases": len(rows),
        "failures": failures,
        "pass": not failures,
        "scope": "Finite selected deterministic sensitivities; does not certify unbounded Gaussian tails or empirical accuracy",
        "rows": rows,
    }
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
