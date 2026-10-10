"""Reproduce source-qualified configuration checks for every selected pair."""

import logging
import sys
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.tools.shot_pattern_analysis import matched_lean_control as pairing
from src.tools.shot_pattern_analysis.matched_supplement_pipeline import (
    _load_original_sources,
    _pair_result,
    select_explicit_pairs,
)
from src.tools.shot_pattern_analysis.matched_supplement_report import (
    load_supplement_sources,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


source_files = [Path(pairing.__file__), Path(__file__)]
source_hashes = {str(p): sha(p) for p in source_files}
cells = []
for name, folder in [
    ("original_primary", "final_v1"),
    ("original_control", "control_final_v1"),
]:
    _, refs = _load_original_sources(ROOT / folder, name)
    cells.extend(refs)
protocols = {
    "pw-primary-face-bounds/1": ROOT / "supplement_final_v1/pw_primary_face",
    "pw-control-face-bounds/1": ROOT / "supplement_final_v1/pw_control_face",
    "driver-control-speed-bounds/1": ROOT / "supplement_final_v1/driver_control_speed",
}
s = load_supplement_sources(
    protocols,
    ROOT / "review_followups/SUPPLEMENTAL_EXECUTION_INVENTORY.json",
    stage="final",
)
for ref in s["unique_cells"]:
    ref["_execution"] = s["execution"]
    cells.append(ref)
for alias in s["aliases"]:
    if alias["proof_status"] == "verified":
        ref = next(
            r for r in s["unique_cells"] if r["cell"] == alias["alias_of_primary"]
        )
        cells.append(
            {
                **ref,
                **alias,
                "cell": alias["alias_cell"],
                "execution_kind": "provenance_alias",
                "summary": ref["summary"],
                "config": alias["requested_config"],
                "_execution": s["execution"],
            }
        )
pairs = select_explicit_pairs(cells)


class ConfigReached(Exception):
    pass


def stop_at_rows(*args, **kwargs):
    raise ConfigReached


original = pairing._primitive_errors_by_pattern
pairing._primitive_errors_by_pattern = stop_at_rows
transition = ROOT / "source_transition_supplement.json"
transition_sha = sha(transition)
results = []
for i, p in enumerate(pairs):
    row = {
        "kind": p["comparison_kind"],
        "first": p["first"]["cell"],
        "first_protocol": p["first"]["source_protocol"],
        "second": p["second"]["cell"],
        "second_protocol": p["second"]["source_protocol"],
    }
    try:
        _pair_result(p, transition, transition_sha)
    except ConfigReached:
        row["status"] = "configuration_and_nominal_checks_passed"
    except Exception as exc:
        logging.getLogger(__name__).exception("Validation record rejected")
        row.update(status="failed", error=repr(exc))
    else:
        raise AssertionError("preflight did not stop at row boundary")
    results.append(row)
    if i % 25 == 0:
        sys.stdout.write(str(f"{i + 1}/{len(pairs)}") + "\n")
        sys.stdout.flush()
pairing._primitive_errors_by_pattern = original
for path_str, h in source_hashes.items():
    assert sha(Path(path_str)) == h
receipt = {
    "scope": "All selected source/config/nominal checks, deliberately stops before primitive row read/statistical computations",
    "pair_count": len(pairs),
    "counts": dict(Counter(p["comparison_kind"] for p in pairs)),
    "source_hashes": source_hashes,
    "transition_sha256": transition_sha,
    "results": results,
    "failures": [r for r in results if r["status"] == "failed"],
}
(OUT / "preflight.json").write_text(json.dumps(receipt, indent=2) + "\n")
sys.stdout.write(
    str(
        json.dumps(
            {"pair_count": len(pairs), "failures": receipt["failures"]}, indent=2
        )
    )
    + "\n"
)
