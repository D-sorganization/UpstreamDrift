"""Solver-independent probe certificates and their numpy verifier (NSX-25, #12568).

A *leaf* certifies one affine functional ``f(z) = const + c . z`` (the
predicate normalised to ``f >= theta``) over a :class:`~.trajectory_system.StackedSystem`
``{G z <= h, b_j - A_j z in SOC}``.  Stacking ``A = [G; A_1; ...]`` and
``b = [h; b_1; ...]``, a leaf is one of three kinds:

``LOWER_BOUND``
    a primal *witness* ``w`` (feasible, ``f(w)`` its value) and a dual vector
    ``d`` with ``d >= 0`` on the polyhedral rows, ``d_j`` in the second-order
    cone on each cone block (self-dual) and stationarity ``c + A^T d = 0``.
    Weak duality then gives ``f(z) >= const - b . d`` for *every* feasible
    ``z``: the certified lower bound.  The witness is an upper bound on
    ``min f``, so the pair brackets the optimum and ``bound <= f(w)``.
``FARKAS``
    ``d`` in the dual cone with ``A^T d = 0`` and ``b . d < 0``: the set is
    empty, so nothing can be claimed about the predicate.
``WITNESS_ONLY``
    ``f`` is unbounded below on the set; a feasible witness with ``f`` below the
    threshold proves the predicate false.

Verdict rule (explicit tolerance ``atol = SIGN_ATOL_RTOL * scale``, never a
bare zero, mirroring the envelope sign classes): ``HOLDS`` iff the certified
bound exceeds ``theta + atol``; ``FAILS`` iff a feasible witness has
``f < theta - atol``; otherwise ``UNDECIDED("BOUNDARY")`` (the optimum is
within ``atol`` of the threshold).

The verifier :func:`verify_leaf` uses only numpy and the system data.  The
stationarity and cone checks hold up to ``DUAL_TOL`` (scaled); a witness may
violate a row by the HiGHS feasibility default ``PRIMAL_TOL`` (scaled).  The
certificate is therefore exact up to the reported residuals, not a
rational-arithmetic proof.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
    ValidationStatus,
)
from src.shared.python.loop_nullspace.trajectory_system import StackedSystem

__all__ = [
    "DUAL_TOL",
    "KIND_FARKAS",
    "KIND_LOWER",
    "KIND_WITNESS",
    "PRIMAL_TOL",
    "LeafCertificate",
    "Verdict",
    "Witness",
    "arr_or_none",
    "leaf_verdict",
    "record_from_dict",
    "record_to_dict",
    "verify_leaf",
]

PRIMAL_TOL = 1e-7
DUAL_TOL = 1e-8
KIND_LOWER = "LOWER_BOUND"
KIND_FARKAS = "FARKAS"
KIND_WITNESS = "WITNESS_ONLY"
_KINDS = (KIND_LOWER, KIND_FARKAS, KIND_WITNESS)


class Verdict(Enum):
    """Outcome of a probe."""

    HOLDS = "HOLDS"
    FAILS = "FAILS"
    UNDECIDED = "UNDECIDED"


def arr_or_none(value: Any) -> Array | None:
    """Read-only float copy, or ``None``."""
    if value is None:
        return None
    arr = np.array(value, dtype=float, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False)
class LeafCertificate:
    """Certificate for one normalised functional ``f(z) = const + c . z >= theta``.

    ``frame`` is the probed frame (``None`` for a window sum).  ``c`` is over
    the stacked variable.  See the module docstring for the three ``kind``s.
    """

    kind: str
    frame: int | None
    c: Array
    const: float
    threshold: float
    atol: float
    verdict: Verdict
    reason: str = ""
    witness: Array | None = None
    witness_value: float | None = None
    dual: Array | None = None
    lower_bound: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"leaf kind {self.kind!r} not in {_KINDS}")
        if not (self.atol >= 0.0):
            raise ValueError("atol must be non-negative")
        object.__setattr__(self, "c", arr_or_none(self.c))
        object.__setattr__(self, "witness", arr_or_none(self.witness))
        object.__setattr__(self, "dual", arr_or_none(self.dual))

    def to_dict(self) -> dict[str, Any]:
        """JSON-able dict (arrays as float lists; ``None`` stays ``None``)."""

        def lst(a: Array | None) -> list[float] | None:
            return None if a is None else [float(v) for v in a]

        return {
            "kind": self.kind,
            "frame": self.frame,
            "c": lst(self.c),
            "const": self.const,
            "threshold": self.threshold,
            "atol": self.atol,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "witness": lst(self.witness),
            "witness_value": self.witness_value,
            "dual": lst(self.dual),
            "lower_bound": self.lower_bound,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LeafCertificate:
        """Inverse of :meth:`to_dict`; raises ``ValueError`` on malformed input."""
        try:
            return cls(
                kind=d["kind"],
                frame=d["frame"],
                c=d["c"],
                const=float(d["const"]),
                threshold=float(d["threshold"]),
                atol=float(d["atol"]),
                verdict=Verdict(d["verdict"]),
                reason=d.get("reason", ""),
                witness=d.get("witness"),
                witness_value=d.get("witness_value"),
                dual=d.get("dual"),
                lower_bound=d.get("lower_bound"),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed leaf certificate: {exc!r}") from exc


@dataclass(frozen=True, eq=False)
class Witness:
    """A feasible point violating the predicate (``FAILS``).

    ``z`` holds one coordinate vector per frame; ``x`` the full
    ``(tau, lambda, f_g)`` per frame when the set carries a decomposition.
    ``value`` is the predicate quantity at the witness (original sign).
    """

    frame: int
    z: tuple[Array, ...]
    value: float
    x: tuple[Array, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-able dict."""
        return {
            "frame": self.frame,
            "z": [[float(v) for v in a] for a in self.z],
            "value": self.value,
            "x": None if self.x is None else [[float(v) for v in a] for a in self.x],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Witness:
        """Inverse of :meth:`to_dict`."""
        try:
            x = d.get("x")
            return cls(
                int(d["frame"]),
                tuple(arr_or_none(a) for a in d["z"]),  # type: ignore[misc]
                float(d["value"]),
                None if x is None else tuple(arr_or_none(a) for a in x),  # type: ignore[misc]
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed witness: {exc!r}") from exc


# ----------------------------------------------------------------- verdict
def leaf_verdict(
    kind: str,
    lower_bound: float | None,
    witness_value: float | None,
    threshold: float,
    atol: float,
) -> tuple[Verdict, str]:
    """Verdict and reason implied by the numbers of a leaf (see module docs)."""
    if kind == KIND_FARKAS:
        return Verdict.UNDECIDED, "EMPTY_FEASIBLE_SET"
    if witness_value is not None and witness_value < threshold - atol:
        return Verdict.FAILS, ""
    if kind == KIND_WITNESS:
        return Verdict.UNDECIDED, "WITNESS_NOT_VIOLATING"
    if lower_bound is None:
        return Verdict.UNDECIDED, "NO_DUAL_CERTIFICATE"
    if lower_bound > threshold + atol:
        return Verdict.HOLDS, ""
    return Verdict.UNDECIDED, "BOUNDARY"


# ---------------------------------------------------------------- verifier
def _in_cone(v: Array, scale: float) -> bool:
    return bool(v[0] >= float(np.linalg.norm(v[1:])) - DUAL_TOL * (1.0 + scale))


def _check_witness(leaf: LeafCertificate, s: StackedSystem) -> list[str]:
    msgs: list[str] = []
    w = np.asarray(leaf.witness)
    if w.shape != (s.n_var,):
        return [f"witness has shape {w.shape}, expected ({s.n_var},)"]
    if s.G.shape[0]:
        slack = s.G @ w - s.h
        bad = np.flatnonzero(slack > PRIMAL_TOL * (1.0 + np.abs(s.h)))
        msgs += [f"witness violates row {s.row_labels[i]!r}" for i in bad[:3]]
    for cone in s.cones:
        v = cone.b - cone.A @ w
        if not _in_cone(v, float(np.max(np.abs(v)))):
            msgs.append(f"witness leaves cone {cone.label!r}")
    value = leaf.const + float(np.asarray(leaf.c) @ w)
    if leaf.witness_value is None or abs(value - leaf.witness_value) > DUAL_TOL * (
        1.0 + abs(value)
    ):
        msgs.append(f"stored witness value {leaf.witness_value} != recomputed {value}")
    return msgs


def _dual_cone_ok(d: Array, s: StackedSystem) -> list[str]:
    msgs: list[str] = []
    m = s.G.shape[0]
    top = float(np.max(np.abs(d))) if d.size else 0.0
    if m and float(np.min(d[:m])) < -DUAL_TOL * (1.0 + top):
        msgs.append("dual has a negative multiplier on a polyhedral row")
    pos = m
    for cone in s.cones:
        blk = d[pos : pos + cone.b.size]
        pos += cone.b.size
        if not _in_cone(blk, top):
            msgs.append(f"dual block of cone {cone.label!r} is outside the cone")
    return msgs


def _check_dual(leaf: LeafCertificate, s: StackedSystem) -> list[str]:
    d = np.asarray(leaf.dual)
    if d.shape != (s.n_dual,):
        return [f"dual has shape {d.shape}, expected ({s.n_dual},)"]
    msgs = _dual_cone_ok(d, s)
    a_all, b_all = s.stacked_matrix()
    c = np.asarray(leaf.c)
    if leaf.kind == KIND_FARKAS:
        res = float(np.max(np.abs(a_all.T @ d))) if s.n_var else 0.0
        gap = float(b_all @ d)
        if res > DUAL_TOL * (1.0 + float(np.sum(np.abs(d)))):
            msgs.append(f"Farkas stationarity residual {res:.3e} too large")
        if not gap < -DUAL_TOL * float(np.sum(np.abs(d))):
            msgs.append(f"Farkas gap b.d = {gap:.3e} is not negative enough")
        return msgs
    res = float(np.max(np.abs(c + a_all.T @ d))) if s.n_var else 0.0
    if res > DUAL_TOL * (1.0 + float(np.max(np.abs(c), initial=0.0))):
        msgs.append(f"stationarity residual {res:.3e} too large")
    bound = leaf.const - float(b_all @ d)
    if leaf.lower_bound is None or abs(bound - leaf.lower_bound) > DUAL_TOL * (
        1.0 + abs(bound)
    ):
        msgs.append(f"stored bound {leaf.lower_bound} != recomputed {bound}")
    wv = leaf.witness_value
    if wv is not None and bound > wv + DUAL_TOL * (1.0 + abs(wv)):
        msgs.append("weak duality violated: certified bound exceeds witness value")
    return msgs


def verify_leaf(leaf: LeafCertificate, system: StackedSystem) -> list[str]:
    """Re-check a leaf against ``system`` with plain numpy; empty list means valid.

    Checks witness feasibility (rows and cones), the stored value, dual
    signs/cone membership, stationarity or the Farkas conditions, the stored
    bound, weak duality, and that the stored verdict is the one the numbers
    imply.  Never calls a solver.
    """
    if not isinstance(leaf, LeafCertificate):
        raise TypeError(f"leaf must be LeafCertificate, got {type(leaf)!r}")
    if np.asarray(leaf.c).shape != (system.n_var,):
        return [f"c has shape {np.asarray(leaf.c).shape}, expected ({system.n_var},)"]
    msgs: list[str] = []
    if leaf.kind in (KIND_LOWER, KIND_WITNESS):
        if leaf.witness is None:
            msgs.append("leaf has no witness")
        else:
            msgs += _check_witness(leaf, system)
    if leaf.kind in (KIND_LOWER, KIND_FARKAS):
        if leaf.dual is None:
            msgs.append("leaf has no dual certificate")
        else:
            msgs += _check_dual(leaf, system)
    if not msgs:
        want, _ = leaf_verdict(
            leaf.kind,
            leaf.lower_bound,
            leaf.witness_value,
            leaf.threshold,
            leaf.atol,
        )
        if want is not leaf.verdict:
            msgs.append(
                f"stored verdict {leaf.verdict.value} but numbers imply {want.value}"
            )
    return msgs


# ------------------------------------------------------- record <-> dict
def record_to_dict(rec: QuantityRecord | None) -> dict[str, Any] | None:
    """JSON-able form of a :class:`QuantityRecord` (``None`` stays ``None``)."""
    if rec is None:
        return None
    vals = np.asarray(rec.values, dtype=float)
    return {
        "values": [None if np.isnan(v) else float(v) for v in vals.ravel()],
        "shape": list(vals.shape),
        "units": rec.units,
        "allocation_dependence": rec.allocation_dependence.value,
        "model": [
            rec.model_provenance.name,
            rec.model_provenance.version,
            rec.model_provenance.content_hash,
        ],
        "input": [
            rec.input_provenance.kind,
            rec.input_provenance.source,
            rec.input_provenance.note,
        ],
        "validation_status": rec.validation_status.value,
        "conventions": list(rec.conventions),
        "hypotheses": list(rec.hypotheses),
        "reason": rec.reason,
    }


def record_from_dict(d: dict[str, Any] | None) -> QuantityRecord | None:
    """Inverse of :func:`record_to_dict`."""
    if d is None:
        return None
    try:
        vals = np.array(
            [np.nan if v is None else v for v in d["values"]], dtype=float
        ).reshape(d["shape"])
        return QuantityRecord(
            values=vals,
            units=d["units"],
            allocation_dependence=AllocationDependence(d["allocation_dependence"]),
            model_provenance=ModelProvenance(*d["model"]),
            input_provenance=InputProvenance(*d["input"]),
            validation_status=ValidationStatus(d["validation_status"]),
            conventions=tuple(d["conventions"]),
            hypotheses=tuple(d["hypotheses"]),
            reason=d["reason"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed quantity record: {exc!r}") from exc


def dumps(obj: dict[str, Any]) -> str:
    """Deterministic JSON text (sorted keys; floats use the shortest round-trip repr)."""
    return json.dumps(obj, sort_keys=True, allow_nan=False)
