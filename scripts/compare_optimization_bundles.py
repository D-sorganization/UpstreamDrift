"""Compare recorded Solution Bundle metrics; never imply forward acceptance.

Run from the repository root with ``python3 -m scripts.compare_optimization_bundles``.
This Slice 5 report consumes the existing strict JSON/NPZ codec. It does not
recompute flight, infer missing quantities, or independently replay dynamics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from src.shared.python.optimization.solution_bundle import (
    SolutionBundle,
    load_solution_bundle,
)

_FLIGHT_KEYS = {"carry_m", "ball_speed_ms", "clubhead_speed_ms", "lateral_dispersion_m"}
_ROLLOUT_KEYS = {"rollout_m", "total_distance_m"}


def _deltas(
    baseline: dict[str, float], candidate: dict[str, float], keys: set[str]
) -> dict[str, dict[str, float | None]]:
    rows = {}
    for key in sorted(keys):
        values = []
        for mapping in (baseline, candidate):
            value = mapping.get(key)
            if key in mapping and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{key} must be a finite numeric metric")
            values.append(value)
        a, b = values
        delta = None if a is None or b is None else b - a
        if delta is not None and not math.isfinite(delta):
            raise ValueError(f"{key} delta must be finite")
        rows[key] = {"baseline": a, "candidate": b, "delta": delta}
    return rows


def _provenance(bundle: SolutionBundle) -> dict[str, Any]:
    return {
        "solution_id": bundle.solution_id,
        "parent_id": bundle.parent_id,
        "git_sha": bundle.git_sha,
        "assumptions": list(bundle.assumptions),
        "control": bundle.control,
        "integrator": bundle.integrator,
        "optimizer_seed": bundle.optimizer_seed,
    }


def compare_bundles(baseline: str | Path, candidate: str | Path) -> dict[str, Any]:
    """Return deterministic recorded deltas (candidate minus baseline).

    Both directories must pass the codec's integrity/shape validation and share
    model, assets, coordinate conventions and environment. Missing metrics stay
    null. Negative feasibility margins are retained, without declaring an
    aggregate feasibility verdict. Replay drift remains explicitly unverified.
    Raises ValueError for incompatible inputs or invalid/nonfinite metrics.
    """
    a = load_solution_bundle(baseline)
    b = load_solution_bundle(candidate)
    compatibility = {}
    for field in ("model_hash", "asset_hashes", "frame", "joint_order", "environment"):
        value = getattr(a, field)
        if value != getattr(b, field):
            raise ValueError(f"incompatible {field}")
        compatibility[field] = list(value) if field == "joint_order" else value
    metric_keys = set(a.flight_metrics) | set(b.flight_metrics) | _FLIGHT_KEYS
    return {
        "schema_version": "optimization-bundle-comparison/1",
        "evidence": "recorded metrics only; no physical acceptance verdict",
        "baseline": _provenance(a),
        "candidate": _provenance(b),
        "compatibility": compatibility,
        "flight_metrics": _deltas(
            a.flight_metrics, b.flight_metrics, metric_keys - _ROLLOUT_KEYS
        ),
        "rollout_metrics": _deltas(
            a.flight_metrics, b.flight_metrics, metric_keys & _ROLLOUT_KEYS
        ),
        "feasibility_margins": _deltas(
            a.feasibility_margins,
            b.feasibility_margins,
            set(a.feasibility_margins) | set(b.feasibility_margins),
        ),
        "forward_replay": {"status": "unverified", "drift": None},
        "limitations": [
            "Metric names and units are producer supplied; no alias conversion.",
            "Rollout optimization is deferred and rollout metrics are unvalidated.",
            "Recorded metrics and margins are not independently recomputed.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Print a strict JSON comparison of two immutable saved bundles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_bundles(args.baseline, args.candidate)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
