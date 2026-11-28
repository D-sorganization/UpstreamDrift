"""Report sampled rank/tolerance sensitivity using the existing NSX accounting.

Run ``python3 -m scripts.research.nullspace_rank_audit`` for a synthetic
near-rank-loss control. This research diagnostic does not admit a trajectory,
solve an allocation, or compare raw null-space basis coordinates (Refs #15494).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import json
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.contracts import ClosedLoopSnapshot
from src.shared.python.loop_nullspace.decomposition import (
    CharacteristicScales,
    dimension_accounting,
)
from src.shared.python.loop_nullspace.fixtures import (
    generic_planar_config,
    planar_golfer_snapshot,
)


def audit_rank_history(
    snapshots: Sequence[ClosedLoopSnapshot],
    times: ArrayLike,
    rtols: ArrayLike,
    *,
    scales: CharacteristicScales | None = None,
) -> dict[str, Any]:
    """Return JSON-ready counts indexed by tolerance, then sampled frame.

    Preconditions: nonempty snapshots with identical topology and ground-load
    dimension; finite strictly increasing times in seconds; at least two unique
    finite relative tolerances in (0, 1). Raises ValueError on invalid data and
    TypeError on non-snapshot inputs or invalid scales.

    Postconditions: every count is supplied by ``dimension_accounting``.
    ``changed_frames`` contains the later index of each adjacent count change;
    ``tolerance_sensitive_frames`` lists frames whose counts differ across the
    supplied tolerances. Neither result establishes feasibility or invariance
    between samples. No candidate is admitted, clipped, or silently repaired.
    """
    frames = tuple(snapshots)
    if not frames:
        raise ValueError("snapshots must be nonempty")
    if any(not isinstance(frame, ClosedLoopSnapshot) for frame in frames):
        raise TypeError("snapshots must contain only ClosedLoopSnapshot values")
    time = np.asarray(times, dtype=float)
    if (
        time.shape != (len(frames),)
        or not np.all(np.isfinite(time))
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError("times must be finite, strictly increasing, one per snapshot")
    tolerances = np.asarray(rtols, dtype=float)
    if (
        tolerances.ndim != 1
        or tolerances.size < 2
        or not np.all(np.isfinite(tolerances))
        or np.any((tolerances <= 0) | (tolerances >= 1))
        or np.unique(tolerances).size != tolerances.size
    ):
        raise ValueError(
            "rtols must contain at least two unique finite values in (0, 1)"
        )
    sc = CharacteristicScales() if scales is None else scales
    if not isinstance(sc, CharacteristicScales):
        raise TypeError("scales must be CharacteristicScales")
    ground_shapes = {None if f.J_g is None else f.J_g.shape for f in frames}
    if any(f.topology != frames[0].topology for f in frames) or len(ground_shapes) != 1:
        raise ValueError("snapshots must share topology and ground-load dimension")
    sweeps = []
    for rtol in tolerances:
        counts = [
            dimension_accounting(f, scales=sc, rank_rtol=float(rtol)).to_dict()
            for f in frames
        ]
        sweeps.append(
            {
                "rtol": float(rtol),
                "counts": counts,
                "changed_frames": [
                    i for i in range(1, len(frames)) if counts[i] != counts[i - 1]
                ],
            }
        )
    return {
        "schema_version": 1,
        "classification": "diagnostic_only",
        "times": time.tolist(),
        "time_units": "s",
        "count_units": "1",
        "scales": sc.to_dict(),
        "rank_rule": "sigma > rtol * max(sigma_max, 1) on NSX scaled load matrices",
        "sweeps": sweeps,
        "tolerance_sensitive_frames": [
            i
            for i in range(len(frames))
            if any(s["counts"][i] != sweeps[0]["counts"][i] for s in sweeps[1:])
        ],
        "non_claims": [
            "not a replay, allocation, admission, or feasibility certificate",
            "unchanged sampled rank does not prove a physical invariant",
            "no conclusion between samples or outside the supplied tolerance grid",
            "caller must declare model, contact regime, load policy and provenance",
        ],
    }


def main() -> None:
    """Write a strict JSON synthetic control to stdout; never claim a golfer run."""
    baseline = planar_golfer_snapshot(generic_planar_config()).snapshot
    frames = []
    for epsilon in (1.0, 1e-7, 0.0):
        jacobian = np.zeros_like(baseline.J_c)
        jacobian[0, 0], jacobian[1, 1] = 1.0, epsilon
        frames.append(replace(baseline, J_c=jacobian))
    report = audit_rank_history(frames, [0.0, 0.1, 0.2], [1e-9, 1e-5])
    report["fixture"] = "synthetic_diagonal_closure"
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
