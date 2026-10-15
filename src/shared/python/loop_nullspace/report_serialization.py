"""JSON forms of NSX results for the report contract (NSX-40, #12572).

Pure serialisation, no mathematics: each function turns an existing NSX value
object into the plain ``dict`` that ``schemas/loop-nullspace-report-v1.json``
describes.  Every quantity is written by :func:`quantity_dict` with all six
evidence axes of :class:`~src.shared.python.loop_nullspace.evidence.QuantityRecord`
(allocation dependence, model provenance, input provenance, validation status,
conventions, hypotheses) plus its units and, when unavailable, its reason.

Non-finite numbers never reach the JSON: an unavailable quantity has
``values: null`` and a bound that is not ``BOUNDED`` is ``null``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.admission import (
    AdmissionReport,
    VerdictKind,
)
from src.shared.python.loop_nullspace.allocation_spec import AllocationResult
from src.shared.python.loop_nullspace.envelopes import BoundStatus, EnvelopeSeries
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.invariants import InvariantReport
from src.shared.python.loop_nullspace.registry import HypothesisSet, Registry

__all__ = [
    "admission_dict",
    "allocation_dict",
    "envelope_dict",
    "hypotheses_dict",
    "invariants_dict",
    "json_float",
    "quantity_dict",
]

_LEVELS = ("phi", "phi_dot", "phi_ddot")


def json_float(value: float) -> float | None:
    """``value`` as a JSON number, or ``None`` when it is NaN or infinite."""
    x = float(value)
    return x if math.isfinite(x) else None


def _array(values: Any) -> Any:
    """Nested lists of an array, with non-finite entries as ``None``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return json_float(float(arr))
    return (
        [_array(row) for row in arr]
        if arr.ndim > 1
        else [json_float(float(x)) for x in arr]
    )


def quantity_dict(rec: QuantityRecord) -> dict[str, Any]:
    """A quantity with its units and all six evidence axes."""
    unavailable = rec.allocation_dependence is AllocationDependence.UNAVAILABLE
    return {
        "values": None if unavailable else _array(rec.values),
        "units": rec.units,
        "allocation_dependence": rec.allocation_dependence.value,
        "validation_status": rec.validation_status.value,
        "model_provenance": {
            "name": rec.model_provenance.name,
            "version": rec.model_provenance.version,
            "content_hash": rec.model_provenance.content_hash,
        },
        "input_provenance": {
            "kind": rec.input_provenance.kind,
            "source": rec.input_provenance.source,
            "note": rec.input_provenance.note,
        },
        "conventions": list(rec.conventions),
        "hypotheses": list(rec.hypotheses),
        "reason": rec.reason,
    }


def invariants_dict(
    report: InvariantReport, frame_indices: Sequence[int]
) -> dict[str, Any]:
    """Per-frame invariant quantities, keyed by the original frame index."""
    gates = report.gates
    return {
        "metric_label": report.metric_label,
        "gates": {
            "phi_dot_tol": gates.phi_dot_tol,
            "phi_ddot_tol": gates.phi_ddot_tol,
            "length_scale_m": gates.length_scale_m,
            "time_scale_s": gates.time_scale_s,
        },
        "frames": [
            {
                "index": int(frame_indices[f.index]),
                "quantities": {k: quantity_dict(v) for k, v in f.records.items()},
            }
            for f in report.frames
        ],
    }


def admission_dict(
    report: AdmissionReport, frame_indices: Sequence[int]
) -> dict[str, Any]:
    """The admission verdict, its reasons and the three scaled residual levels."""
    verdict = report.verdict
    gates = report.gates
    return {
        "verdict": verdict.kind.value,
        "reasons": list(verdict.reasons),
        "failing_frames": [int(frame_indices[k]) for k in verdict.failing_frames],
        "repaired": verdict.kind is VerdictKind.REPAIRED,
        "derivative_method": report.derivative_method,
        "noise_floor": json_float(report.noise_floor),
        "position_evaluated": report.position_evaluated,
        "gates": {
            "phi_tol": gates.phi_tol,
            "phi_dot_tol": gates.phi_dot_tol,
            "phi_ddot_tol": gates.phi_ddot_tol,
            "require_position": gates.require_position,
            "length_scale_m": gates.length_scale_m,
            "time_scale_s": gates.time_scale_s,
        },
        "residuals": {
            level: quantity_dict(report.residual_record(level)) for level in _LEVELS
        },
    }


def _status_names(rows: Sequence[Sequence[BoundStatus]], j: int) -> list[str]:
    return [row[j].name for row in rows]


def envelope_dict(
    series: EnvelopeSeries,
    records: dict[str, QuantityRecord],
    frame_indices: Sequence[int],
    conflicts: dict[int, tuple[str, ...]],
) -> dict[str, Any]:
    """Static envelopes: quantity records, raw bounds and the per-frame status.

    ``conflicts`` maps a position to the minimal conflicting hypotheses of an
    infeasible frame.  The finite bounds stay available per quantity even when
    the record is ``UNAVAILABLE``; infeasible frames are listed, never dropped.
    """
    bounds = {
        name: {
            "units": series.units[j],
            "lower": _array(series.lower[:, j]),
            "upper": _array(series.upper[:, j]),
            "status_lower": _status_names(series.status_lower, j),
            "status_upper": _status_names(series.status_upper, j),
        }
        for j, name in enumerate(series.names)
    }
    frames = [
        {
            "index": int(frame_indices[k]),
            "chebyshev_status": series.chebyshev_status[k].name,
            "chebyshev_margin": json_float(series.chebyshev_margin[k]),
            "feasible": series.chebyshev_status[k] is not BoundStatus.INFEASIBLE,
            "conflict_hypotheses": list(conflicts.get(k, ())),
            "hypothesis_ids": list(series.frame_hypothesis_ids[k]),
            "rests_on_unverified": bool(series.frame_rests_on_unverified[k]),
        }
        for k in range(series.n_frames)
    ]
    return {
        "label": series.label,
        "chebyshev_metric": series.chebyshev_metric,
        "quantities": {k: quantity_dict(v) for k, v in records.items()},
        "bounds": bounds,
        "frames": frames,
        "simplex_iterations": int(series.simplex_iterations),
        "warm_started": bool(series.warm_started),
        "non_claims": list(series.non_claims),
    }


def allocation_dict(
    result: AllocationResult,
    frame_indices: Sequence[int],
    excluded: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """The hypothesis-conditioned allocation, or its honest failure status."""
    out: dict[str, Any] = {
        "status": result.status.value,
        "reason": result.reason or None,
        "frame_indices": [int(i) for i in frame_indices],
        "excluded_frames": [dict(e) for e in excluded],
        "iterations": int(result.iterations),
        "solve_time_s": json_float(result.solve_time_s),
        "non_claim": result.non_claim,
        "torque": None if result.record is None else quantity_dict(result.record),
        "objective_terms": {
            k: json_float(v) for k, v in result.objective_terms.items()
        },
        "objective_total": None
        if result.objective_total is None
        else json_float(result.objective_total),
        "kkt_residuals": None
        if result.kkt_residuals is None
        else {
            k: json_float(v)
            for k, v in dataclasses.asdict(result.kkt_residuals).items()
        },
        "unverified_hypothesis_ids": list(result.unverified_hypothesis_ids),
        "power_reduction": None
        if result.power_reduction is None
        else dataclasses.asdict(result.power_reduction),
        "convexity": None
        if result.convexity is None
        else {
            **dataclasses.asdict(result.convexity),
            "min_curvature_ratio": json_float(result.convexity.min_curvature_ratio),
            "enabled_terms": list(result.convexity.enabled_terms),
        },
    }
    return out


def hypotheses_dict(registry: Registry, preset: HypothesisSet) -> dict[str, Any]:
    """The preset's hypotheses and conventions with the registry version."""
    return {
        "registry_version": registry.registry_version,
        "preset": preset.id,
        "preset_label": preset.label,
        "hypotheses": [
            {
                "id": h.id,
                "statement": h.statement,
                "verification_status": h.verification_status,
                "citation_id": h.source.citation_id,
                "source_note": h.source.note,
            }
            for h in preset.hypotheses
        ],
        "conventions": [
            {"id": c.id, "statement": c.statement} for c in preset.conventions
        ],
    }
