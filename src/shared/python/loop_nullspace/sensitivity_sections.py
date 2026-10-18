"""JSON forms of the sensitivity stage for report schema v2 (NSX-99, #12728).

Pure serialisation of values the NSX-90/91/95 modules already computed; there is
no mathematics here beyond picking the primary outcome's row and ``sqrt`` of a
variance.  Every number a UI may show is a
:func:`~.report_serialization.quantity_dict` with its six evidence axes (and,
when present, the orthogonal ``conditioning`` axis) so that
``plot_model_decisions`` can ask :func:`~.output_policy_integration.
display_decision` about it unchanged.

Only the primary outcome's quantities are written.  Rows of the kernel, variance
and budget for the other outcomes are not serialised: nothing downstream draws
them, and an undrawn number would be an ungated one.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.report_serialization import (
    json_float,
    quantity_dict,
)
from src.shared.python.loop_nullspace.strategy_arms import ArmResult
from src.shared.python.loop_nullspace.strategy_campaign import (
    ComparisonReport,
    RankedArm,
)

__all__ = [
    "arm_dict",
    "arm_sd_dict",
    "budget_records",
    "comparison_dict",
    "primary_row",
    "sd_record",
    "unavailable_record",
]

_SQUARED = re.compile(r"^\((.+)\)\^2$")


def unavailable_record(rec: QuantityRecord, reason: str) -> QuantityRecord:
    """``rec`` with its numbers replaced by NaN and an explicit reason."""
    return dataclasses.replace(
        rec,
        values=np.full(np.shape(rec.values), np.nan),
        allocation_dependence=AllocationDependence.UNAVAILABLE,
        reason=reason,
    )


def primary_row(rec: QuantityRecord, axis: int, index: int) -> QuantityRecord:
    """``rec`` restricted to outcome ``index`` along ``axis`` (axis is dropped)."""
    values = np.take(np.asarray(rec.values), index, axis=axis)
    return dataclasses.replace(rec, values=np.atleast_1d(values))


def sd_record(variance: QuantityRecord, units: str) -> QuantityRecord:
    """Standard deviation of a variance record (same evidence, ``sqrt`` values)."""
    values = np.asarray(variance.values, dtype=float)
    if variance.allocation_dependence is AllocationDependence.UNAVAILABLE:
        return dataclasses.replace(variance, units=units)
    return dataclasses.replace(variance, values=np.sqrt(values), units=units)


def budget_records(
    variance: QuantityRecord,
    classes: Sequence[str],
    parts: np.ndarray,
    by_time: np.ndarray,
    total: float,
) -> dict[str, QuantityRecord]:
    """Parts, shares and per-step parts of the primary outcome's variance.

    All three inherit the evidence axes of ``variance`` (they are exactly its
    decomposition).  A non-positive total has no shares: those are
    ``UNAVAILABLE`` with the reason, never ``NaN`` or zero-filled.
    """
    if len(classes) != parts.shape[0]:
        raise ValueError("one part is needed per input class")
    base = dataclasses.replace(variance, values=parts)
    out = {
        "parts": base,
        "by_time": dataclasses.replace(variance, values=by_time),
    }
    if total > 0.0 and math.isfinite(total):
        shares = dataclasses.replace(variance, values=parts / total, units="1")
    else:
        shares = unavailable_record(
            dataclasses.replace(variance, values=parts, units="1"),
            "the total variance is not positive; shares are undefined",
        )
    out["shares"] = shares
    return out


def _sd_units(variance_units: str) -> str | None:
    match = _SQUARED.match(variance_units)
    return None if match is None else match.group(1)


def arm_sd_dict(arm: ArmResult) -> dict[str, Any]:
    """The arm's face-angle SD quantity, or the reason there is none."""
    if arm.record is None or arm.face_variance is None:
        return {
            "label": arm.label,
            "sd": None,
            "reason": "the arm produced no variance",
        }
    units = _sd_units(arm.record.units)
    if units is None:
        return {
            "label": arm.label,
            "sd": None,
            "reason": f"variance units {arm.record.units!r} are not a square",
        }
    rec = dataclasses.replace(
        arm.record, values=np.array([math.sqrt(arm.face_variance)]), units=units
    )
    return {"label": arm.label, "sd": quantity_dict(rec), "reason": None}


def _num(value: float | None) -> float | None:
    return None if value is None else json_float(value)


def arm_dict(row: RankedArm) -> dict[str, Any]:
    """One arm of the comparison with its status, place and ratio quantity."""
    arm = row.arm
    ratio = None if row.ratio_record is None else quantity_dict(row.ratio_record)
    return {
        "label": arm.label,
        "strategy": arm.strategy.value,
        "status": arm.status.value,
        "reason": arm.reason,
        "admitted": bool(row.admitted),
        "rank": row.rank,
        "exclusion": None if row.exclusion is None else row.exclusion.value,
        "beats_baseline": bool(row.beats_baseline),
        "chs_mps": _num(arm.chs_mps),
        "effort": _num(arm.effort),
        "effort_basis": arm.effort_basis,
        "certificate": arm.certificate,
        "mechanisms": list(arm.mechanisms),
        "hypotheses": list(arm.hypotheses),
        "unverified_hypothesis_ids": list(arm.unverified_hypothesis_ids),
        "ratio": ratio,
    }


def comparison_dict(report: ComparisonReport) -> dict[str, Any]:
    """The ranked comparison as the ``strategies`` section body."""
    settings: Mapping[str, Any] = {
        "reference_chs_mps": json_float(report.settings.reference_chs_mps),
        "chs_rel_tol": json_float(report.settings.chs_rel_tol),
        "effort_cap": _num(report.settings.effort_cap),
        "effort_basis": report.settings.effort_basis,
    }
    return {
        "mode": report.mode.value,
        "settings": dict(settings),
        "primary_outcome": report.primary_outcome,
        "ranking_basis": report.ranking_basis,
        "assumptions": list(report.assumptions),
        "arms": [arm_dict(r) for r in report.rows],
        "fronts": {k: list(v) for k, v in report.fronts.items()},
        "failure_rate": json_float(report.failure_rate),
        "not_attempted": list(report.not_attempted),
        "winner": report.winner,
        "non_claim": report.non_claim,
    }
