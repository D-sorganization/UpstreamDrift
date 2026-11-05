"""Observability of the full-body-v2 trunk DOFs from the marker set (SHOW-A5, #14554).

The fit residual is marker position minus model marker position. Its Jacobian
with respect to the six v2 trunk coordinates tells whether lumbar and thoracic
angles can be told apart: near-parallel lumbar/thoracic columns (per axis) or a
large condition number mean the data cannot separate them, and the panel
resolution (epic #14549, Argon) is then to couple them with the cited
regional split instead of fitting free DOFs.

:func:`marker_jacobian` is a central finite difference of the spec forward
kinematics, :func:`observability_report` evaluates it at named key frames and
:func:`decide_trunk_coupling` turns the report into the free-vs-coupled choice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.model_appearance.skinned_body import spec_body_poses

Array = NDArray[np.float64]

__all__ = ["decide_trunk_coupling", "marker_jacobian", "observability_report"]

STEP_RAD = 1e-6
RANK_RTOL = 1e-6
#: Decision thresholds (recorded in the ADR): separable only if both hold.
MAX_CONDITION = 1.0e3
MAX_AXIS_COSINE = 0.95
_AXES = ("flexion", "lateral_bending", "axial_rotation")


def _marker_points(spec: Mapping[str, Any]) -> dict[str, tuple[str, Array]]:
    """Attached marker -> (body, 4-vector point in that body), from frame + offset."""
    frames = {f["name"]: f for f in spec["frames"]}
    bodies = {b["name"].rsplit("/", 1)[-1]: b["name"] for b in spec["bodies"]}
    out: dict[str, tuple[str, Array]] = {}
    for name, att in spec["marker_attachments"].items():
        if att["offset_m"] is None:  # unattached (no body in this model): no residual
            continue
        offset = np.append(np.asarray(att["offset_m"], dtype=float), 1.0)
        if att["body"] in frames:
            frame = frames[att["body"]]
            out[name] = (frame["body"], np.asarray(frame["placement"]) @ offset)
        elif att["body"] in bodies:
            out[name] = (bodies[att["body"]], offset)
        else:
            raise ValueError(f"marker {name!r} attaches to unknown {att['body']!r}")
    return out


def _positions(spec: Mapping[str, Any], q: Mapping[str, float]) -> Array:
    poses = spec_body_poses(spec, q)
    pts = _marker_points(spec)
    return np.concatenate([(poses[b] @ p)[:3] for b, p in pts.values()])


def marker_jacobian(
    spec: Mapping[str, Any],
    q: Sequence[float],
    coordinates: Sequence[str],
    *,
    step: float = STEP_RAD,
) -> Array:
    """Jacobian (3 * markers, n) of marker world positions wrt ``coordinates``.

    ``q`` is a full coordinate vector in ``spec["coordinate_order"]``. Central
    differences about ``q``; postcondition: finite.
    """
    order = list(spec["coordinate_order"])
    require(len(q) == len(order), "q must follow the spec coordinate_order")
    require(set(coordinates) <= set(order), "unknown coordinate requested")
    require(step > 0.0, "step must be positive")
    base = dict(zip(order, (float(v) for v in q), strict=True))
    cols = []
    for name in coordinates:
        hi, lo = dict(base), dict(base)
        hi[name] += step
        lo[name] -= step
        cols.append((_positions(spec, hi) - _positions(spec, lo)) / (2.0 * step))
    jac = np.stack(cols, axis=1)
    ensure(bool(np.isfinite(jac).all()), "Jacobian must be finite")
    return jac


def _axis_cosines(jac: Array, coordinates: Sequence[str]) -> dict[str, float]:
    out = {}
    for axis in _AXES:
        a = jac[:, list(coordinates).index(f"lumbar_{axis}")]
        b = jac[:, list(coordinates).index(f"thoracic_{axis}")]
        out[axis] = float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return out


def observability_report(
    spec: Mapping[str, Any],
    q_events: Mapping[str, Sequence[float]],
    coordinates: Sequence[str],
) -> dict[str, Any]:
    """SVD, condition number, rank and per-axis lumbar/thoracic cosines per event.

    ``coordinates`` must hold the lumbar and thoracic coordinate of each axis.
    Postcondition: singular values are sorted descending; ``rank`` counts those
    above ``RANK_RTOL`` times the largest.
    """
    events: dict[str, Any] = {}
    for event, q in q_events.items():
        jac = marker_jacobian(spec, q, coordinates)
        sv = np.linalg.svd(jac, compute_uv=False)
        events[event] = {
            "singular_values": [float(s) for s in sv],
            "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
            "rank": int(np.count_nonzero(sv > RANK_RTOL * sv[0])),
            "axis_cosines": _axis_cosines(jac, coordinates),
        }
    return {"coordinates": list(coordinates), "events": events}


def decide_trunk_coupling(report: Mapping[str, Any]) -> dict[str, Any]:
    """Free vs coupled decision from an :func:`observability_report`.

    Free only when, at every event, the condition number is below
    :data:`MAX_CONDITION` and every axis cosine below :data:`MAX_AXIS_COSINE`;
    otherwise coupled. ``evidence`` lists each violated criterion.
    """
    require(bool(report.get("events")), "report has no events")
    evidence = []
    for event, ev in report["events"].items():
        if ev["condition_number"] > MAX_CONDITION:
            evidence.append(
                f"{event}: condition number {ev['condition_number']:.3g} > "
                f"{MAX_CONDITION:g}"
            )
        for axis, cos in ev["axis_cosines"].items():
            if cos > MAX_AXIS_COSINE:
                evidence.append(
                    f"{event}: {axis} lumbar/thoracic columns cosine {cos:.3f} > "
                    f"{MAX_AXIS_COSINE}"
                )
    return {"coupled": bool(evidence), "evidence": evidence}
