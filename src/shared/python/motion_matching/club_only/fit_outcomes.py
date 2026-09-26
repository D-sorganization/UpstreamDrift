"""Recorded club-only fit outcomes read from committed evidence receipts (#10602).

CO-04 (pendulum match) and CO-05 (body candidates) write ``outcomes`` lists into
their evidence receipts. The CO-08 matrix scores a cell only from such a
recorded outcome, and only when every metric the gates judge was actually
recorded — a missing metric is never filled in with a default.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "REQUIRED_FIT_METRICS",
    "load_fit_outcomes",
    "missing_fit_metrics",
]

# Metrics the frozen CO-02 gates judge; each must be recorded, finite and >= 0.
REQUIRED_FIT_METRICS: tuple[str, ...] = (
    "original_3d_rmse_m",
    "in_plane_rmse_m",
    "grip_position_rmse_m",
    "face_position_rmse_m",
    "coverage_fraction",
    "closure_residual_m",
)


def load_fit_outcomes(path: Path | str) -> dict[tuple[str, str], dict[str, Any]]:
    """Return recorded outcomes keyed by ``(model_id, trial_id)``.

    A missing receipt yields ``{}`` (nothing recorded). Entries without both
    identifiers are ignored because they cannot be attributed to a cell.
    """
    evidence_path = Path(path)
    if not evidence_path.is_file():
        return {}
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    outcomes: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in payload.get("outcomes", []):
        model_id = entry.get("model_id")
        trial_id = entry.get("trial_id")
        if model_id and trial_id:
            outcomes[(str(model_id), str(trial_id))] = dict(entry)
    return outcomes


def _is_recorded_metric(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0.0


def missing_fit_metrics(outcome: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the required metrics ``outcome`` lacks (empty when complete).

    ``contact_feasible`` must be an explicit bool and ``coverage_fraction`` must
    lie in [0, 1]; neither is ever assumed.
    """
    missing = [
        name
        for name in REQUIRED_FIT_METRICS
        if not _is_recorded_metric(outcome.get(name))
    ]
    coverage = outcome.get("coverage_fraction")
    if isinstance(coverage, (int, float)) and coverage > 1.0:
        missing.append("coverage_fraction")
    if not isinstance(outcome.get("contact_feasible"), bool):
        missing.append("contact_feasible")
    return tuple(missing)
