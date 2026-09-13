"""Reclassify the archived audit77 arrays with an explicit floor safety factor.

The executed receipt used factor 1. This local pass re-applies the same tested
provider to the same archived analytic/central arrays and measured floors,
recording factor-1 and factor-2 verdicts side by side and the excess ratio of
every entry the factor-1 pass called failed. No arrays are recomputed and no
gate is changed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.shared.python.motion_matching.derivative_resolution import (
    classify_derivative_block,
    qualify_direction,
)

BLOCKS = (
    "markers",
    "endpoint_q",
    "endpoint_qd",
    "endpoint_scaled",
    "continuity_q",
    "continuity_qd",
    "continuity_scaled",
)
EXPECTED_ZERO = {"node_position": "continuity_qd", "node_velocity": "continuity_q"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit76", type=Path, required=True)
    parser.add_argument("--audit77", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=float, default=2.0)
    args = parser.parse_args()
    receipt = json.loads((args.audit77 / "receipt.json").read_text())
    result = {
        "receipt_sha256": hashlib.sha256(
            (args.audit77 / "receipt.json").read_bytes()
        ).hexdigest(),
        "factors": [1.0, args.factor],
        "rows": [],
        "excess_ratios": [],
    }
    qualification = {1.0: {}, args.factor: {}}
    for name, rows in receipt["verdicts"].items():
        series = {f: {k: [] for k in BLOCKS} for f in qualification}
        for row in rows:
            h, tol = row["step"], row["tolerance"]
            folder = args.audit76 if row["source"] == "audit76" else args.audit77
            suffix = f"{name}-{h:g}" + ("" if row["source"] == "audit76" else f"-{tol}")
            arrays = np.load(folder / f"comparison-{suffix}.npz")
            entry = {"direction": name, "step": h, "tolerance": tol, "blocks": {}}
            for key in BLOCKS:
                executed = row["blocks"][key]
                floor = executed["resolution_floor"]
                replay_error = floor * h
                verdicts = {}
                for factor in qualification:
                    verdict = classify_derivative_block(
                        arrays[f"analytic_{key}"],
                        arrays[f"central_{key}"],
                        step=h,
                        replay_error=replay_error,
                        expected_zero=EXPECTED_ZERO.get(name) == key,
                        floor_safety_factor=factor,
                    )
                    verdicts[factor] = verdict
                    series[factor][key].append(verdict)
                assert verdicts[1.0].verdict == executed["verdict"], (name, h, key)
                entry["blocks"][key] = {
                    str(f): v.verdict for f, v in verdicts.items()
                } | {
                    "absolute_l2_error": verdicts[1.0].absolute_l2_error,
                    "floor_factor_1": verdicts[1.0].resolution_floor,
                    "relative_l2_error": verdicts[1.0].relative_l2_error,
                }
                if verdicts[1.0].verdict == "failed":
                    result["excess_ratios"].append(
                        {
                            "direction": name,
                            "step": h,
                            "tolerance": tol,
                            "block": key,
                            "error_over_floor": verdicts[1.0].absolute_l2_error
                            / verdicts[1.0].resolution_floor,
                        }
                    )
            result["rows"].append(entry)
        for factor in qualification:
            qualification[factor][name] = {
                key: qualify_direction(series[factor][key]) for key in BLOCKS
            }
    result["qualification"] = {str(f): q for f, q in qualification.items()}
    result["max_error_over_floor"] = max(
        (r["error_over_floor"] for r in result["excess_ratios"]), default=0.0
    )
    result["status"] = (
        "qualified_with_factor"
        if all(all(v.values()) for v in qualification[args.factor].values())
        else "unqualified"
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["status"] == "qualified_with_factor" else 1)


if __name__ == "__main__":
    main()
