"""Certified affine probes and trajectory-coupled envelopes (NSX-25, #12568).

A *probe* answers a yes/no question about an affine functional of the
trajectory feasible set, for example "is the lead-hand axial load positive in
every frame of window ``W`` under hypotheses ``H``?".  It is one LP (or SOCP
with exact cones) over the stacked, rate-coupled set of
:mod:`.trajectory_system`, and the answer carries a certificate that
:func:`verify_certificate` re-checks with plain numpy
(:mod:`.probe_certificates`):

``HOLDS``
    the certified lower bound (weak duality) clears the threshold by ``atol``.
``FAILS``
    a feasible *witness* violates the predicate by more than ``atol``.
``UNDECIDED(reason)``
    ``BOUNDARY`` (optimum within ``atol`` of the threshold),
    ``EMPTY_FEASIBLE_SET`` (Farkas certificate: nothing can be claimed) or a
    certificate the verifier rejected.

Only affine predicates exist: :class:`AffinePredicate` is built from an affine
quantity (:class:`~.envelopes.EnvelopeQuantity` or a quantity factory) or, via
:meth:`AffinePredicate.from_function`, from a callable that is tested for
affinity and rejected with :class:`NonAffinePredicateError` otherwise.
Non-convex predicates are out of scope.

Non-claims: a certificate holds only under the stated hypotheses (the rows of
the feasible sets and the rate bounds); it is not a statement about the true
loads of any athlete.  Cones enter exactly (never through the polygon), so
neither direction depends on the declared linearisation.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.envelopes import (
    SIGN_ATOL_RTOL,
    BoundStatus,
    EnvelopeQuantity,
    SignClass,
    classify_sign,
)
from src.shared.python.loop_nullspace.evidence import (
    Array,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.pareto import (
    ParetoFront,
    ParetoPoint,
    pareto_front,
)
from src.shared.python.loop_nullspace.probe_certificates import (
    KIND_FARKAS,
    KIND_LOWER,
    KIND_WITNESS,
    LeafCertificate,
    Verdict,
    Witness,
    dumps,
    leaf_verdict,
    record_from_dict,
    record_to_dict,
    verify_leaf,
)
from src.shared.python.loop_nullspace.solvers import SolveStatus
from src.shared.python.loop_nullspace.trajectory_envelope import (
    TrajectoryEnvelope,
    trajectory_envelope,
)
from src.shared.python.loop_nullspace.trajectory_system import (
    QuantitySpec,
    RateBound,
    StackedSystem,
    SystemSolver,
    family_record,
    select_quantity,
    stack_system,
    unverified_ids,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AffinePredicate",
    "Certificate",
    "NonAffinePredicateError",
    "ParetoFront",
    "ParetoPoint",
    "RateBound",
    "TrajectoryEnvelope",
    "VerificationReport",
    "Verdict",
    "Witness",
    "certify",
    "pareto_front",
    "trajectory_envelope",
    "verify_certificate",
]

SCHEMA = "loop-nullspace-probe/1"
AGGREGATIONS = ("ALL", "SUM")
SENSES = {"ge": 1.0, "le": -1.0}
AFFINITY_RTOL = 1e-9
NON_CLAIM = (
    "A certificate holds only under the stated hypotheses (the declared "
    "feasible-set rows and rate bounds); it is not a statement about the true "
    "loads or torques of any athlete."
)


class NonAffinePredicateError(ValueError):
    """A predicate that is not an affine functional of the null coordinates."""


# ------------------------------------------------------------- predicate
@dataclass(frozen=True, eq=False)
class AffinePredicate:
    """``q(z) >= threshold`` (``sense="ge"``) or ``q(z) <= threshold`` (``"le"``).

    ``quantity`` is an :class:`EnvelopeQuantity`, a sequence of them, or a
    factory ``FeasibleSet -> sequence``; ``quantity_name`` selects one when
    several are returned.  The tolerance ``atol = SIGN_ATOL_RTOL * q.scale``
    (the NSX-22 sign tolerance) separates HOLDS/FAILS from BOUNDARY.

    Raises ``TypeError`` for a non-affine object, ``ValueError`` for an unknown
    ``sense`` or a non-finite ``threshold``.
    """

    quantity: QuantitySpec
    sense: str
    threshold: float
    quantity_name: str | None = None
    label: str = ""

    def __post_init__(self) -> None:
        q = self.quantity
        if not (
            isinstance(q, EnvelopeQuantity)
            or callable(q)
            or isinstance(q, (list, tuple))
        ):
            raise TypeError(
                "an AffinePredicate needs an EnvelopeQuantity, a sequence of them or a "
                f"quantity factory (affine functionals only), got {type(q)!r}"
            )
        if self.sense not in SENSES:
            raise ValueError(f"sense must be 'ge' or 'le', got {self.sense!r}")
        if not math.isfinite(self.threshold):
            raise ValueError(f"threshold must be finite, got {self.threshold!r}")
        object.__setattr__(self, "threshold", float(self.threshold))

    @classmethod
    def from_function(
        cls,
        fn: Callable[[Array], float],
        n_z: int,
        sense: str,
        threshold: float,
        *,
        units: str,
        scale: float,
        label: str = "",
    ) -> AffinePredicate:
        """Build from a callable on ``z`` after testing it for affinity.

        The callable is evaluated at the origin and the unit vectors to get the
        coefficients, then at random points: ``|fn(z) - (const + coeff . z)|``
        must stay below ``AFFINITY_RTOL`` relative.  Raises
        :class:`NonAffinePredicateError` otherwise.
        """
        if isinstance(n_z, bool) or not isinstance(n_z, int) or n_z < 1:
            raise ValueError(f"n_z must be a positive integer, got {n_z!r}")
        const = float(fn(np.zeros(n_z)))
        coeff = np.array([float(fn(e)) - const for e in np.eye(n_z)])
        rng = np.random.default_rng(12568)
        for _ in range(8):
            z = rng.normal(size=n_z) * 3.0
            want = const + float(coeff @ z)
            got = float(fn(z))
            if abs(got - want) > AFFINITY_RTOL * (1.0 + abs(want) + abs(got)):
                raise NonAffinePredicateError(
                    "predicate is not affine in z: probes accept affine "
                    f"predicates only (|fn(z) - affine fit| = {abs(got - want):.3e})"
                )
        q = EnvelopeQuantity(label or "predicate", coeff, const, units, scale)
        return cls(q, sense, threshold, label=label or "predicate")

    def quantity_for(self, fs: FeasibleSet) -> EnvelopeQuantity:
        """The affine quantity on ``fs`` (coefficient length checked)."""
        return select_quantity(self.quantity, self.quantity_name, fs)

    @property
    def sign(self) -> float:
        """``+1`` for ``ge``, ``-1`` for ``le`` (normalises to ``f >= theta``)."""
        return SENSES[self.sense]

    def describe(self, qty: EnvelopeQuantity) -> dict[str, Any]:
        """JSON-able description (the coefficients live in the leaves)."""
        return {
            "name": self.label or qty.name,
            "quantity": qty.name,
            "sense": self.sense,
            "threshold": self.threshold,
            "units": qty.units,
            "scale": qty.scale,
        }


# ------------------------------------------------------------ certificate
@dataclass(frozen=True, eq=False)
class Certificate:
    """Verdict of a probe with everything needed to re-check it.

    ``witness`` is set exactly when ``verdict is FAILS``.  ``leaves`` are the
    per-frame (or window-sum) certificates; ``system_digest`` binds them to the
    exact stacked data.  Serialise with :meth:`to_json`; the round trip is
    exact for every array and number.
    """

    verdict: Verdict
    reason: str
    window: tuple[int, ...]
    aggregation: str
    predicate: dict[str, Any]
    leaves: tuple[LeafCertificate, ...]
    witness: Witness | None
    hypotheses: tuple[str, ...]
    unverified_hypothesis_ids: tuple[str, ...]
    system_digest: str
    cone_mode: str
    coupling: tuple[dict[str, Any], ...]
    solver: str
    record: QuantityRecord | None = field(default=None, repr=False)
    non_claim: str = NON_CLAIM
    schema: str = SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """JSON-able dict."""
        return {
            "schema": self.schema,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "window": list(self.window),
            "aggregation": self.aggregation,
            "predicate": self.predicate,
            "leaves": [leaf.to_dict() for leaf in self.leaves],
            "witness": None if self.witness is None else self.witness.to_dict(),
            "hypotheses": list(self.hypotheses),
            "unverified_hypothesis_ids": list(self.unverified_hypothesis_ids),
            "system_digest": self.system_digest,
            "cone_mode": self.cone_mode,
            "coupling": list(self.coupling),
            "solver": self.solver,
            "record": record_to_dict(self.record),
            "non_claim": self.non_claim,
        }

    def to_json(self) -> str:
        """Deterministic JSON text."""
        return dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> Certificate:
        """Parse :meth:`to_json` output; raises ``ValueError`` if malformed."""
        import json

        try:
            d = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON: {exc}") from exc
        if not isinstance(d, dict) or d.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} certificate")
        try:
            wit = d["witness"]
            return cls(
                verdict=Verdict(d["verdict"]),
                reason=d["reason"],
                window=tuple(int(k) for k in d["window"]),
                aggregation=d["aggregation"],
                predicate=d["predicate"],
                leaves=tuple(LeafCertificate.from_dict(x) for x in d["leaves"]),
                witness=None if wit is None else Witness.from_dict(wit),
                hypotheses=tuple(d["hypotheses"]),
                unverified_hypothesis_ids=tuple(d["unverified_hypothesis_ids"]),
                system_digest=d["system_digest"],
                cone_mode=d["cone_mode"],
                coupling=tuple(d["coupling"]),
                solver=d["solver"],
                record=record_from_dict(d["record"]),
                non_claim=d["non_claim"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed certificate: {exc!r}") from exc


@dataclass(frozen=True)
class VerificationReport:
    """Outcome of :func:`verify_certificate`; ``ok`` iff ``messages`` is empty."""

    ok: bool
    messages: tuple[str, ...]


# --------------------------------------------------------------- helpers
def _validated_window(window: Any, n_frames: int) -> tuple[int, ...]:
    if window is None:
        return tuple(range(n_frames))
    items = list(window)
    if not items:
        raise ValueError("window must contain at least one frame")
    out = []
    for k in items:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise TypeError(f"window entries must be integers, got {k!r}")
        if not 0 <= int(k) < n_frames:
            raise ValueError(f"window frame {k} outside [0, {n_frames})")
        out.append(int(k))
    if len(set(out)) != len(out):
        raise ValueError(f"window has duplicate frames: {items}")
    return tuple(sorted(out))


def _functional_specs(
    system: StackedSystem,
    qty: dict[int, EnvelopeQuantity],
    pred: AffinePredicate,
    aggregation: str,
) -> list[tuple[int | None, Array, float, float]]:
    """``(frame, c, const, atol)`` of each normalised functional ``f >= theta``."""
    s = pred.sign
    if aggregation == "SUM":
        c = sum(s * system.embed(k, q.coeff) for k, q in qty.items())
        const = sum(s * q.const for q in qty.values())
        atol = SIGN_ATOL_RTOL * sum(q.scale for q in qty.values())
        return [(None, np.asarray(c), float(const), atol)]
    return [
        (k, s * system.embed(k, q.coeff), s * q.const, SIGN_ATOL_RTOL * q.scale)
        for k, q in qty.items()
    ]


def _witness_leaf(
    system: StackedSystem,
    c: Array,
    const: float,
    thr: float,
    margin: float,
) -> tuple[Array, float]:
    """A feasible point with ``f <= thr - margin`` for a functional unbounded below."""
    shifted = system.with_row(c, thr - const - margin, "probe:violation")
    with SystemSolver(shifted) as solver:
        out = solver.feasibility()
    if out.status is not SolveStatus.OPTIMAL or out.x is None:
        raise RuntimeError(
            f"unbounded functional but no violating point found ({out.status.name})"
        )
    return out.x, const + float(c @ out.x)


def _solve_leaf(
    solver: SystemSolver,
    spec: tuple[int | None, Array, float, float],
    thr: float,
    b_all: Array,
) -> LeafCertificate:
    frame, c, const, atol = spec
    system = solver.system
    out = solver.minimize(c)
    common = {"frame": frame, "c": c, "const": const, "threshold": thr, "atol": atol}
    if out.status is SolveStatus.INFEASIBLE:
        verdict, reason = leaf_verdict(KIND_FARKAS, None, None, thr, atol)
        return LeafCertificate(
            KIND_FARKAS, verdict=verdict, reason=reason, dual=out.dual, **common
        )
    if out.status is SolveStatus.UNBOUNDED:
        w, value = _witness_leaf(system, c, const, thr, max(1.0, 10.0 * atol))
        verdict, reason = leaf_verdict(KIND_WITNESS, None, value, thr, atol)
        return LeafCertificate(
            KIND_WITNESS,
            verdict=verdict,
            reason=reason,
            witness=w,
            witness_value=value,
            **common,
        )
    assert out.x is not None
    value = const + float(c @ out.x)
    bound = None if out.dual is None else const - float(b_all @ out.dual)
    verdict, reason = leaf_verdict(KIND_LOWER, bound, value, thr, atol)
    return LeafCertificate(
        KIND_LOWER,
        verdict=verdict,
        reason=reason,
        witness=out.x,
        witness_value=value,
        dual=out.dual,
        lower_bound=bound,
        **common,
    )


def _combine(leaves: Sequence[LeafCertificate]) -> tuple[Verdict, str, int]:
    """Certificate verdict, reason and index of the leaf supplying the witness."""
    for i, leaf in enumerate(leaves):
        if leaf.verdict is Verdict.FAILS:
            return Verdict.FAILS, "", i
    if all(leaf.verdict is Verdict.HOLDS for leaf in leaves):
        return Verdict.HOLDS, "", -1
    for leaf in leaves:
        if leaf.reason == "EMPTY_FEASIBLE_SET":
            return Verdict.UNDECIDED, "EMPTY_FEASIBLE_SET", -1
    first = next(leaf for leaf in leaves if leaf.verdict is not Verdict.HOLDS)
    return Verdict.UNDECIDED, first.reason, -1


def _make_witness(
    system: StackedSystem,
    frames: Sequence[FeasibleSet],
    leaf: LeafCertificate,
    sign: float,
) -> Witness:
    zs = system.split(leaf.witness)
    xs = []
    for fs, z in zip(frames, zs, strict=True):
        dec = fs.decomposition
        xs.append(None if dec is None or z.size != dec.nullity else dec.x(z))
    x = None if any(v is None for v in xs) else tuple(xs)
    frame = -1 if leaf.frame is None else leaf.frame
    return Witness(frame, zs, sign * float(leaf.witness_value), x)  # type: ignore[arg-type]


def _record_for(
    frames: Sequence[FeasibleSet],
    leaves: Sequence[LeafCertificate],
    pred: AffinePredicate,
    qty: EnvelopeQuantity,
    hyps: Sequence[str],
) -> QuantityRecord:
    s = pred.sign
    bounds = [
        np.nan if leaf.lower_bound is None else s * leaf.lower_bound for leaf in leaves
    ]
    bad = any(np.isnan(b) for b in bounds)
    side = "lower" if s > 0 else "upper"
    return family_record(
        frames,
        np.array(bounds),
        qty.units,
        hyps,
        source=f"certified {side} bound of {qty.name}",
        note="dual-certified bound of the affine predicate quantity",
        unavailable="no certified bound (empty set or unbounded direction)"
        if bad
        else None,
    )


def certify(
    feasible_sets: Sequence[FeasibleSet],
    predicate: AffinePredicate,
    window: Sequence[int] | None = None,
    *,
    coupling: Sequence[RateBound] = (),
    aggregation: str = "ALL",
) -> Certificate:
    """Certify an affine predicate over the trajectory-coupled feasible set.

    ``aggregation="ALL"`` (default) asks the predicate of *every* frame in
    ``window`` (one LP per frame on one persistent model); ``"SUM"`` asks it of
    the window sum ``sum_k q_k`` (one LP).  ``coupling`` adds hard rate bounds
    between frames; without it the frames are independent.

    Preconditions: non-empty ``feasible_sets``; ``predicate`` an
    :class:`AffinePredicate` whose quantity has the right length on every frame
    of the window (``ValueError``/``TypeError``); ``window`` distinct valid frame
    indices (default all).  Postconditions: the returned leaves pass
    :func:`verify_leaf` (a leaf the verifier rejects is downgraded to
    ``UNDECIDED`` with reason ``CERTIFICATE_REJECTED``, never reported as
    HOLDS/FAILS); ``witness`` is set iff ``FAILS``; a statement is always
    conditional on ``hypotheses``.  Raises ``RuntimeError`` only when the
    solver is numerically inconclusive.
    """
    frames = tuple(feasible_sets)
    if not isinstance(predicate, AffinePredicate):
        raise TypeError(f"predicate must be AffinePredicate, got {type(predicate)!r}")
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}"
        )
    system = stack_system(frames, coupling, cones="exact")
    win = _validated_window(window, len(frames))
    qty = {k: predicate.quantity_for(frames[k]) for k in win}
    thr = predicate.sign * predicate.threshold
    specs = _functional_specs(system, qty, predicate, aggregation)
    _, b_all = system.stacked_matrix()
    with SystemSolver(system) as solver:
        leaves = [_solve_leaf(solver, sp, thr, b_all) for sp in specs]
    leaves = [_checked(leaf, system) for leaf in leaves]
    verdict, reason, idx = _combine(leaves)
    witness = (
        _make_witness(system, frames, leaves[idx], predicate.sign)
        if verdict is Verdict.FAILS
        else None
    )
    hyps = tuple(dict.fromkeys(system.hypothesis_ids))
    first_q = qty[win[0]]
    logger.info("NSX-25 certify %s over %s -> %s", predicate.label, win, verdict.value)
    return Certificate(
        verdict=verdict,
        reason=reason,
        window=win,
        aggregation=aggregation,
        predicate=predicate.describe(first_q),
        leaves=tuple(leaves),
        witness=witness,
        hypotheses=hyps,
        unverified_hypothesis_ids=unverified_ids(hyps),
        system_digest=system.digest,
        cone_mode=system.cone_mode,
        coupling=tuple(rb.describe() for rb in coupling),
        solver="clarabel" if system.cones else "highspy",
        record=_record_for(frames, leaves, predicate, first_q, hyps),
    )


def _checked(leaf: LeafCertificate, system: StackedSystem) -> LeafCertificate:
    msgs = verify_leaf(leaf, system)
    if not msgs:
        return leaf
    logger.warning("probe certificate rejected by verifier: %s", msgs)
    return LeafCertificate(
        leaf.kind,
        leaf.frame,
        leaf.c,
        leaf.const,
        leaf.threshold,
        leaf.atol,
        Verdict.UNDECIDED,
        f"CERTIFICATE_REJECTED: {msgs[0]}",
        leaf.witness,
        leaf.witness_value,
        leaf.dual,
        leaf.lower_bound,
    )


def verify_certificate(
    cert: Certificate,
    feasible_sets: Sequence[FeasibleSet],
    *,
    coupling: Sequence[RateBound] = (),
    predicate: AffinePredicate | None = None,
) -> VerificationReport:
    """Re-check ``cert`` against the data with plain numpy (no solver).

    The system is rebuilt from ``feasible_sets`` and ``coupling`` and must have
    the certificate's digest; every leaf is re-checked by :func:`verify_leaf`;
    the combined verdict and the witness must match; when ``predicate`` is
    given the leaf functionals must equal the predicate's.  Returns a report;
    never raises for an invalid certificate.
    """
    if not isinstance(cert, Certificate):
        raise TypeError(f"cert must be Certificate, got {type(cert)!r}")
    system = stack_system(feasible_sets, coupling, cones=cert.cone_mode)
    msgs: list[str] = []
    if system.digest != cert.system_digest:
        msgs.append("system digest differs: certificate belongs to different data")
        return VerificationReport(False, tuple(msgs))
    for leaf in cert.leaves:
        msgs += [f"leaf frame {leaf.frame}: {m}" for m in verify_leaf(leaf, system)]
    verdict, _, idx = _combine(cert.leaves)
    if verdict is not cert.verdict:
        msgs.append(
            f"stored verdict {cert.verdict.value} but leaves imply {verdict.value}"
        )
    if cert.verdict is Verdict.FAILS:
        msgs += _check_witness_consistency(cert, system, idx)
    elif cert.witness is not None:
        msgs.append("a non-FAILS certificate must not carry a witness")
    if predicate is not None:
        msgs += _check_functionals(cert, system, feasible_sets, predicate)
    return VerificationReport(not msgs, tuple(msgs))


def _check_witness_consistency(
    cert: Certificate, system: StackedSystem, idx: int
) -> list[str]:
    if cert.witness is None or idx < 0:
        return ["FAILS certificate has no witness"]
    stacked = np.concatenate(cert.witness.z) if cert.witness.z else np.zeros(0)
    leaf_w = cert.leaves[idx].witness
    if (
        leaf_w is None
        or stacked.shape != leaf_w.shape
        or not np.array_equal(stacked, leaf_w)
    ):
        return ["witness differs from the failing leaf's witness"]
    return []


def _check_functionals(
    cert: Certificate,
    system: StackedSystem,
    frames: Sequence[FeasibleSet],
    pred: AffinePredicate,
) -> list[str]:
    qty = {k: pred.quantity_for(frames[k]) for k in cert.window}
    want = _functional_specs(system, qty, pred, cert.aggregation)
    msgs = []
    if len(want) != len(cert.leaves):
        return ["leaf count differs from the predicate's functionals"]
    thr = pred.sign * pred.threshold
    for (frame, c, const, _), leaf in zip(want, cert.leaves, strict=True):
        same = (
            frame == leaf.frame
            and np.allclose(c, leaf.c, rtol=0, atol=1e-12)
            and abs(const - leaf.const) <= 1e-12
            and abs(thr - leaf.threshold) <= 1e-12
        )
        if not same:
            msgs.append(
                f"leaf frame {leaf.frame}: functional differs from the predicate"
            )
    return msgs
