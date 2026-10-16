"""Inertial parameter ensemble over unidentified directions (NSX-62, #12587).

Rigid-body inverse dynamics is linear in the ten inertial parameters of each body,
``Y(q, v, a) pi = tau``.  With the ground reaction and the (known) club wrench
observed, the unactuated root rows ``Y_u pi = w`` contain no unknown input, so the
row space of the regressor stacked over a window is the part of ``pi`` that the
window identifies, and its null space is what the data cannot see
(``structural_observability`` in ``estimation.mosaic.identifiability_subspace``
uses the same rows).  Joint-torque envelopes depend on ``pi`` through the
actuated rows, which the data do not constrain, so the envelope is only as certain
as the inertia is identified.

:func:`identifiability_basis` splits the *sampled* parameters (bodies with
positive mass that are not declared known) into the identified row space and the
unidentified null space of the column-scaled stacked regressor.  The rank is the
smaller of the SVD and the column-pivoted QR rank, so a combination is only called
identified when both agree.  :func:`sample_parameters` draws

    ``pi = pi0 + D N N^T g``,   ``g ~ N(0, sigma_rel^2 I)``

(``N`` the null basis in the scaled coordinates, ``D`` the scale), with the draw
``g`` shaped by the cone geometry: each body perturbs as
``J' = J^(1/2) (I + Delta) J^(1/2)`` with ``Delta`` Gaussian, so a thin or
nearly degenerate body is perturbed less in its flat directions.  A sample is
kept only when every sampled body's pseudo-inertia is positive semidefinite (the
Wensing-Kim-Slotine LMI); the others are *counted* by reason.  A violating sample
is never projected back onto the cone: it is rejected, because a projection would
move the identified combinations.  :func:`evaluate_parameter_ensemble` rebuilds
the envelopes per sample and reports the sensitivity of every quantity, reusing
:func:`envelope_union_intersection` of the NSX-61 ensemble.

Declared hypothesis ``H-INERTIAL-PARAMETER-ENSEMBLE-PRIOR``: a Gaussian in the
scaled coordinates, restricted to the null space.  It is a modelling choice, not
a population or measurement distribution.  Non-claims: no subject-specific
inertia estimate; bi-level inertia learning is out of scope; identification is
conditional on the observed ground reaction and club wrench and holds only for
combinations in the window's row space.  No engine is imported.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy.linalg import block_diag, qr

from src.shared.python.estimation.mosaic.inertial import (
    is_spatial_consistent,
    pseudo_inertia,
)
from src.shared.python.loop_nullspace import registry
from src.shared.python.loop_nullspace.decomposition import BalanceInconsistentError
from src.shared.python.loop_nullspace.ensemble import (
    EnsembleEnvelope,
    envelope_union_intersection,
)
from src.shared.python.loop_nullspace.envelopes import BoundStatus, EnvelopeSeries
from src.shared.python.loop_nullspace.evidence import Array

logger = logging.getLogger(__name__)

__all__ = [
    "PARAMETER_ENSEMBLE_HYPOTHESIS_ID",
    "InertialBasis",
    "InertialModel",
    "ParameterSamples",
    "ParameterSamplingHypothesis",
    "ParameterSensitivityReport",
    "QuantitySensitivity",
    "evaluate_parameter_ensemble",
    "identifiability_basis",
    "sample_parameters",
    "stacked_unactuated_regressor",
    "swap_inertia_layout",
]

PARAMETER_ENSEMBLE_HYPOTHESIS_ID = "H-INERTIAL-PARAMETER-ENSEMBLE-PRIOR"
PARAMS_PER_BODY = 10
_LMI_TOLERANCE = 1e-12
_EIGENVALUE_FLOOR = 1e-12
_MIN_GYRATION_M = 1e-3
_WIDTH_FLOOR = 1e-9
_NON_CLAIMS = (
    "no subject-specific inertia claim: samples are draws from a declared prior",
    "identification holds only for the window's regressor row space, given the "
    "observed ground reaction and club wrench",
    "bi-level inertia learning is out of scope",
    "interval hull of member envelopes: not a probability or confidence set",
)


class InertialModel(Protocol):
    """A model whose body inertial parameters can be read and replaced.

    Parameters are flat, body-major, ten per body in the Pinocchio dynamic
    parameter layout ``[m, h(3), Ixx, Ixy, Iyy, Ixz, Iyz, Izz]`` (inertia about
    the body-frame origin).
    """

    def inertial_parameters(self) -> Array: ...

    def set_inertial_parameters(self, pi: Array) -> None: ...


def swap_inertia_layout(parameters: Array) -> Array:
    """Convert ``(bodies, 10)`` parameters between the Pinocchio and LMI layouts.

    Pinocchio orders the inertia ``Ixx, Ixy, Iyy, Ixz, Iyz, Izz``;
    ``estimation.mosaic.inertial`` orders it ``Ixx, Ixy, Ixz, Iyy, Iyz, Izz``.
    The conversion swaps entries 6 and 7 and is its own inverse.
    """
    arr = np.asarray(parameters, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != PARAMS_PER_BODY:
        raise ValueError(f"parameters must be (bodies, 10), got {arr.shape}")
    out = arr.copy()
    out[:, [6, 7]] = arr[:, [7, 6]]
    return out


@dataclass(frozen=True)
class ParameterSamplingHypothesis:
    """The declared sampling law (``H-INERTIAL-PARAMETER-ENSEMBLE-PRIOR``).

    ``sigma_rel`` is the standard deviation of each entry of ``Delta`` in
    ``J' = J^(1/2) (I + Delta) J^(1/2)`` (``J`` the nominal 4x4 pseudo-inertia
    of a body) before the draw is restricted to the unidentified directions.
    It is dimensionless and a spectral norm of ``Delta`` below one keeps the body
    inside the cone.
    """

    sigma_rel: float
    hypothesis_id: str = PARAMETER_ENSEMBLE_HYPOTHESIS_ID

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma_rel) or self.sigma_rel <= 0.0:
            raise ValueError(
                f"sigma_rel must be positive and finite, got {self.sigma_rel}"
            )
        registry.hypothesis(self.hypothesis_id)  # must be declared (ValueError)


@dataclass(frozen=True, eq=False)
class InertialBasis:
    """Identified and unidentified directions of the sampled parameters.

    Coordinates are the *scaled* free parameters ``eta = (pi - pi0) / D`` over
    the sampled bodies.  ``identified_basis`` (``n_free x rank``) spans the row
    space of the scaled stacked regressor and ``null_basis`` its orthogonal
    complement.  ``base_columns`` are the QR pivot columns (indices into the
    full parameter vector) that survive the rank decision.
    """

    pi0: Array
    scale: Array
    free: np.ndarray
    sampled_bodies: tuple[int, ...]
    regressor: Array
    identified_basis: Array
    null_basis: Array
    base_columns: tuple[int, ...]
    rank: int
    qr_rank: int
    singular_values: Array
    rtol: float

    @property
    def n_bodies(self) -> int:
        """Number of bodies (ten parameters each)."""
        return int(self.pi0.size // PARAMS_PER_BODY)

    @property
    def n_free(self) -> int:
        """Number of sampled parameters."""
        return int(self.free.sum())

    @property
    def null_dimension(self) -> int:
        """Dimension of the unidentified subspace of the sampled parameters."""
        return int(self.null_basis.shape[1])

    def identified_drift(self, pi: Array) -> Array:
        """Identified combinations ``V_r^T (pi - pi0) / D`` of ``pi`` (zero if fixed)."""
        pi = np.asarray(pi, dtype=np.float64)
        if pi.shape != self.pi0.shape:
            raise ValueError(f"pi must have shape {self.pi0.shape}, got {pi.shape}")
        eta = ((pi - self.pi0) / self.scale)[self.free]
        return np.asarray(self.identified_basis.T @ eta)


@dataclass(frozen=True, eq=False)
class ParameterSamples:
    """Accepted samples plus the full rejection accounting.

    ``rejected_by_reason`` counts each rejected draw once; ``violating_bodies``
    counts, per body index, how often that body's pseudo-inertia was not PSD
    (a draw can violate on several bodies).  ``max_identified_drift`` is the
    largest scaled change of any identified combination over the accepted draws.
    """

    basis: InertialBasis
    accepted: tuple[Array, ...]
    sample_indices: tuple[int, ...]
    n_requested: int
    seed: int
    sigma_rel: float
    hypothesis_id: str
    rejected_by_reason: dict[str, int] = field(default_factory=dict)
    violating_bodies: dict[int, int] = field(default_factory=dict)
    max_identified_drift: float = 0.0

    @property
    def n_accepted(self) -> int:
        """Number of samples inside the LMI."""
        return len(self.accepted)

    @property
    def n_rejected(self) -> int:
        """Number of draws that violated the LMI."""
        return sum(self.rejected_by_reason.values())

    @property
    def sampled_bodies(self) -> tuple[int, ...]:
        """Indices of the bodies that were sampled."""
        return self.basis.sampled_bodies


# ------------------------------------------------------------------- basis --
def _body_scale(bodies: Array) -> Array:
    """Per-parameter scale ``(m, m r x3, m r^2 x6)``; ones for massless bodies."""
    mass = bodies[:, 0]
    scale = np.ones_like(bodies)
    live = mass > 0.0
    trace = bodies[:, 4] + bodies[:, 6] + bodies[:, 9]
    gyration = np.full_like(mass, _MIN_GYRATION_M)
    gyration[live] = np.maximum(np.sqrt(trace[live] / mass[live]), _MIN_GYRATION_M)
    powers = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=np.float64)
    scale[live] = mass[live, None] * gyration[live, None] ** powers
    return scale


def _validate_basis_inputs(
    regressor: Array, pi0: Array, fixed_bodies: Sequence[int], rtol: float
) -> tuple[Array, Array, tuple[int, ...]]:
    y = np.asarray(regressor, dtype=np.float64)
    p = np.asarray(pi0, dtype=np.float64)
    if y.ndim != 2 or p.ndim != 1:
        raise ValueError("regressor must be 2-D and pi0 1-D")
    if y.shape[1] != p.size or p.size % PARAMS_PER_BODY:
        raise ValueError(
            f"regressor has {y.shape[1]} columns but pi0 has {p.size} parameters "
            f"(need 10 per body)"
        )
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("regressor and pi0 must be finite")
    if not math.isfinite(rtol) or not 0.0 < rtol < 1.0:
        raise ValueError(f"rtol must lie in (0, 1), got {rtol}")
    n_bodies = p.size // PARAMS_PER_BODY
    fixed = tuple(int(b) for b in fixed_bodies)
    if any(not 0 <= b < n_bodies for b in fixed):
        raise ValueError(f"fixed body index out of range [0, {n_bodies}): {fixed}")
    return y, p, fixed


def identifiability_basis(
    regressor: Array,
    pi0: Array,
    *,
    fixed_bodies: Sequence[int] = (),
    rtol: float = 1e-8,
) -> InertialBasis:
    """Identified row space and unidentified null space of ``regressor``.

    ``regressor`` is the unactuated-row regressor stacked over the window (see
    :func:`stacked_unactuated_regressor`), ``pi0`` the nominal parameters.
    Bodies with non-positive mass are structurally absent and ``fixed_bodies``
    are declared known (for example the club carrier); neither is sampled, but
    both stay in the regressor product ``Y pi``.

    Preconditions (``ValueError``): finite inputs, ten parameters per body,
    ``rtol`` in (0, 1), every nominal body physically consistent (PSD
    pseudo-inertia), at least one sampled body.  Postconditions: the columns of
    ``identified_basis`` and ``null_basis`` are orthonormal and mutually
    orthogonal, ``rank`` is ``min(SVD rank, column-pivoted QR rank)`` of
    ``regressor[:, free] * scale`` at relative tolerance ``rtol``.
    """
    y, p, fixed = _validate_basis_inputs(regressor, pi0, fixed_bodies, rtol)
    bodies = p.reshape(-1, PARAMS_PER_BODY)
    bad = np.flatnonzero(~is_spatial_consistent(swap_inertia_layout(bodies)))
    if bad.size:
        raise ValueError(
            f"nominal bodies {bad.tolist()} are not physically consistent "
            "(pseudo-inertia not positive semidefinite)"
        )
    sampled = tuple(
        b for b in range(bodies.shape[0]) if bodies[b, 0] > 0.0 and b not in fixed
    )
    if not sampled:
        raise ValueError("no sampled body: every body is massless or fixed")
    scale = _body_scale(bodies).ravel()
    free = np.zeros(p.size, dtype=bool)
    for b in sampled:
        free[b * PARAMS_PER_BODY : (b + 1) * PARAMS_PER_BODY] = True
    scaled = y[:, free] * scale[free]
    _, singular, vt = np.linalg.svd(scaled, full_matrices=True)
    top = float(singular[0]) if singular.size else 0.0
    svd_rank = int(np.count_nonzero(singular > rtol * top)) if top > 0.0 else 0
    diag, pivots = _qr_diagonal(scaled)
    qr_rank = int(np.count_nonzero(diag > rtol * diag[0])) if diag[0] > 0.0 else 0
    rank = min(svd_rank, qr_rank)
    if svd_rank != qr_rank:
        logger.warning("SVD rank %d != QR rank %d; using %d", svd_rank, qr_rank, rank)
    free_index = np.flatnonzero(free)
    return InertialBasis(
        pi0=p.copy(),
        scale=scale,
        free=free,
        sampled_bodies=sampled,
        regressor=y.copy(),
        identified_basis=vt[:rank].T.copy(),
        null_basis=vt[rank:].T.copy(),
        base_columns=tuple(int(free_index[c]) for c in pivots[:rank]),
        rank=rank,
        qr_rank=qr_rank,
        singular_values=singular,
        rtol=float(rtol),
    )


def _qr_diagonal(scaled: Array) -> tuple[Array, np.ndarray[Any, np.dtype[np.intp]]]:
    """``|diag R|`` and the column pivots of a column-pivoted QR."""
    if scaled.shape[0] == 0:
        return np.zeros(1), np.arange(scaled.shape[1], dtype=np.intp)
    r, pivots = qr(scaled, mode="r", pivoting=True)
    return np.abs(np.diag(r)), np.asarray(pivots, dtype=np.intp)


def stacked_unactuated_regressor(
    model: Any, snapshots: Sequence[Any], *, rtol: float = 1e-6
) -> Array:
    """Unactuated rows of the torque regressor, stacked over ``snapshots``.

    ``model`` provides ``torque_regressor(q, v, a)`` of shape ``(nv, n_pi)`` and
    ``inertial_parameters()``.  Each snapshot's unactuated rows must reproduce
    ``M a + h`` at the nominal parameters (relative ``rtol`` of the row norm),
    which ties the regressor to the same model the snapshots were built from.

    Preconditions: at least one snapshot (``ValueError``) and agreement of the
    regressor product with the snapshot dynamics (``ValueError``).  Postcondition:
    shape ``(sum of unactuated rows, n_pi)``.
    """
    if len(snapshots) < 1:
        raise ValueError("at least one snapshot is required")
    pi0 = np.asarray(model.inertial_parameters(), dtype=np.float64)
    blocks = []
    for i, snap in enumerate(snapshots):
        rows = list(snap.topology.unactuated_indices)
        y = np.asarray(model.torque_regressor(snap.q, snap.v, snap.a))[rows]
        expected = (snap.M @ snap.a + snap.h)[rows]
        mismatch = float(np.linalg.norm(y @ pi0 - expected))
        if mismatch > rtol * max(float(np.linalg.norm(expected)), 1.0):
            raise ValueError(
                f"regressor does not reproduce M a + h on snapshot {i} "
                f"(residual {mismatch:.3g})"
            )
        blocks.append(y)
    return np.vstack(blocks)


# ---------------------------------------------------------------- sampling --
def _check_draw_arguments(basis: InertialBasis, n: int, seed: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int")
    if basis.null_dimension < 1:
        raise ValueError(
            "no unidentified direction to sample: the regressor identifies every "
            "sampled parameter"
        )


def _body_whitening(body: Array) -> Array:
    """Map ``delta pi`` of one body (Pinocchio layout) to ``vec(Delta)``.

    ``Delta = J^(-1/2) dJ J^(-1/2)`` with ``dJ`` the pseudo-inertia change; the
    10-vector holds the upper triangle of ``Delta`` with off-diagonal entries
    times ``sqrt(2)``, so its Euclidean norm is ``|Delta|_F``.  Eigenvalues of
    ``J`` are floored at ``1e-12`` of the largest, which only relaxes the
    penalty on a nearly flat direction and so can add rejections, never hide one.
    """
    pseudo = pseudo_inertia(swap_inertia_layout(body[None, :]))[0]
    values, vectors = np.linalg.eigh(pseudo)
    values = np.maximum(values, _EIGENVALUE_FLOOR * max(float(values[-1]), 1e-300))
    inv_root = (vectors / np.sqrt(values)) @ vectors.T
    units = pseudo_inertia(swap_inertia_layout(np.eye(PARAMS_PER_BODY)))
    delta = inv_root @ units @ inv_root
    rows, cols = np.triu_indices(4)
    weights = np.where(rows == cols, 1.0, np.sqrt(2.0))
    return np.asarray(delta[:, rows, cols] * weights).T


def _whitening(basis: InertialBasis) -> Array:
    """Block-diagonal whitening of the sampled parameters (``n_free`` square)."""
    bodies = basis.pi0.reshape(-1, PARAMS_PER_BODY)
    return block_diag(*(_body_whitening(bodies[b]) for b in basis.sampled_bodies))


def _violations(basis: InertialBasis, pi: Array) -> list[int]:
    """Sampled bodies whose pseudo-inertia is not PSD at ``pi``."""
    sampled = list(basis.sampled_bodies)
    bodies = swap_inertia_layout(pi.reshape(-1, PARAMS_PER_BODY)[sampled])
    eigenvalues = np.linalg.eigvalsh(pseudo_inertia(bodies))
    failing = np.flatnonzero(eigenvalues.min(axis=1) < -_LMI_TOLERANCE)
    return [sampled[int(i)] for i in failing]


def sample_parameters(
    basis: InertialBasis, hypothesis: ParameterSamplingHypothesis, n: int, seed: int
) -> ParameterSamples:
    """Draw ``n`` parameter vectors along the unidentified directions.

    Preconditions (``ValueError``/``TypeError``): ``n >= 1``, integer ``seed``, a
    non-empty null space.  Postconditions: draw ``k`` uses row ``k`` of one
    seeded normal matrix, so rejections never shift later draws;
    ``n_accepted + n_rejected == n``; every accepted sample keeps all sampled
    bodies' pseudo-inertias positive semidefinite and the identified
    combinations fixed to rounding; massless and fixed bodies are untouched.  A
    draw outside the LMI is counted under ``pseudo_inertia_not_psd`` and never
    projected.
    """
    _check_draw_arguments(basis, n, seed)
    noise = hypothesis.sigma_rel * np.random.default_rng(seed).normal(
        size=(n, basis.n_free)
    )
    # Orthogonal projection, in the whitened (Delta) metric, onto the achievable
    # parameter changes D N z: z = argmin |B z - g| with B = W D N.
    whitening = _whitening(basis)
    embedded = basis.scale[basis.free, None] * basis.null_basis
    coefficients = np.linalg.lstsq(whitening @ embedded, noise.T, rcond=None)[0].T
    eta = coefficients @ basis.null_basis.T
    accepted, indices = [], []
    rejected: Counter[str] = Counter()
    bodies: Counter[int] = Counter()
    worst = 0.0
    for k in range(n):
        pi = basis.pi0.copy()
        pi[basis.free] += eta[k] * basis.scale[basis.free]
        failing = _violations(basis, pi)
        if failing:
            rejected["pseudo_inertia_not_psd"] += 1
            bodies.update(failing)
            continue
        drift = basis.identified_drift(pi)
        worst = max(worst, float(np.max(np.abs(drift))) if drift.size else 0.0)
        accepted.append(pi)
        indices.append(k)
    logger.info("parameter ensemble: %d/%d inside the LMI", len(accepted), n)
    return ParameterSamples(
        basis=basis,
        accepted=tuple(accepted),
        sample_indices=tuple(indices),
        n_requested=n,
        seed=seed,
        sigma_rel=hypothesis.sigma_rel,
        hypothesis_id=hypothesis.hypothesis_id,
        rejected_by_reason=dict(rejected),
        violating_bodies=dict(bodies),
        max_identified_drift=worst,
    )


# -------------------------------------------------------------- sensitivity --
@dataclass(frozen=True)
class QuantitySensitivity:
    """Sensitivity of one quantity to the unidentified inertial parameters.

    ``union_width_over_nominal_*`` are over the frames bounded in both the union
    and the nominal envelope; ``max_bound_shift`` is the largest absolute move of
    a bound of any member from the nominal bound (the quantity's units);
    ``max_relative_bound_shift`` divides each by the nominal width.
    """

    name: str
    units: str
    frames_compared: int
    union_width_over_nominal_median: float | None
    union_width_over_nominal_min: float | None
    union_width_over_nominal_max: float | None
    max_bound_shift: float
    max_relative_bound_shift: float | None
    frames_with_sign_class_disagreement: int
    frames_losing_decided_sign: int


@dataclass(frozen=True, eq=False)
class ParameterSensitivityReport:
    """Per-quantity sensitivity and the full sample accounting."""

    envelope: EnsembleEnvelope
    quantities: tuple[QuantitySensitivity, ...]
    n_requested: int
    n_lmi_rejected: int
    n_refused_unbalanced: int
    n_enveloped: int
    member_frames_infeasible: int
    hypothesis_ids: tuple[str, ...]
    identified_rank: int
    null_dimension: int
    max_identified_drift: float
    non_claims: tuple[str, ...] = _NON_CLAIMS

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary (the envelope arrays are not included)."""
        return {
            "n_requested": self.n_requested,
            "n_lmi_rejected": self.n_lmi_rejected,
            "n_refused_unbalanced": self.n_refused_unbalanced,
            "n_enveloped": self.n_enveloped,
            "member_frames_infeasible": self.member_frames_infeasible,
            "identified_rank": self.identified_rank,
            "null_dimension": self.null_dimension,
            "max_identified_drift": self.max_identified_drift,
            "hypothesis_ids": list(self.hypothesis_ids),
            "quantities": {
                q.name: {k: v for k, v in vars(q).items() if k != "name"}
                for q in self.quantities
            },
            "non_claims": list(self.non_claims),
        }


def _class_name(series: EnvelopeSeries, t: int, j: int) -> str:
    cls = series.sign_class[t]
    if not cls:
        return "NONE"
    entry = cls[j]
    return "NONE" if entry is None else entry.name


def _stats(values: Array) -> tuple[float | None, float | None, float | None]:
    if values.size == 0:
        return None, None, None
    return float(np.median(values)), float(values.min()), float(values.max())


def _bound_shift(nominal: EnvelopeSeries, members: Sequence[EnvelopeSeries], j: int):
    """Largest absolute and relative move of any bound of quantity ``j``."""
    width = nominal.upper[:, j] - nominal.lower[:, j]
    absolute, relative = 0.0, None
    for m in members:
        for ref, got in ((nominal.lower, m.lower), (nominal.upper, m.upper)):
            ok = np.isfinite(ref[:, j]) & np.isfinite(got[:, j])
            if not ok.any():
                continue
            shift = np.abs(got[ok, j] - ref[ok, j])
            absolute = max(absolute, float(shift.max()))
            wide = ok & (width > _WIDTH_FLOOR)
            if wide.any():
                rel = float((np.abs(got[wide, j] - ref[wide, j]) / width[wide]).max())
                relative = rel if relative is None else max(relative, rel)
    return absolute, relative


def _quantity_sensitivity(
    nominal: EnvelopeSeries,
    members: Sequence[EnvelopeSeries],
    union: EnsembleEnvelope,
    j: int,
) -> QuantitySensitivity:
    n_frames = nominal.n_frames
    nominal_width = nominal.upper[:, j] - nominal.lower[:, j]
    union_width = union.union_upper[:, j] - union.union_lower[:, j]
    ok = (
        np.isfinite(nominal_width)
        & np.isfinite(union_width)
        & (nominal_width > _WIDTH_FLOOR)
    )
    median, low, high = _stats(union_width[ok] / nominal_width[ok])
    shift, relative = _bound_shift(nominal, members, j)
    disagree = lost = 0
    for t in range(n_frames):
        classes = {_class_name(m, t, j) for m in (nominal, *members)} - {"NONE"}
        disagree += len(classes) > 1
        decided = _class_name(nominal, t, j) in ("POSITIVE", "NEGATIVE")
        lo, hi = union.union_lower[t, j], union.union_upper[t, j]
        lost += bool(decided and not (lo > 0.0 or hi < 0.0))
    return QuantitySensitivity(
        name=nominal.names[j],
        units=nominal.units[j],
        frames_compared=int(ok.sum()),
        union_width_over_nominal_median=median,
        union_width_over_nominal_min=low,
        union_width_over_nominal_max=high,
        max_bound_shift=shift,
        max_relative_bound_shift=relative,
        frames_with_sign_class_disagreement=int(disagree),
        frames_losing_decided_sign=int(lost),
    )


def evaluate_parameter_ensemble(
    samples: ParameterSamples,
    model: InertialModel,
    series_of: Callable[[], EnvelopeSeries],
    nominal: EnvelopeSeries,
) -> ParameterSensitivityReport:
    """Rebuild the envelopes per accepted sample and report the sensitivity.

    For each accepted sample the model's parameters are replaced and
    ``series_of()`` rebuilds the snapshots, feasible sets and the
    :class:`EnvelopeSeries` from the model's current state.  A sample whose root
    rows cannot be balanced (``BalanceInconsistentError``) is counted in
    ``n_refused_unbalanced`` and never repaired.  The model's nominal parameters
    are restored on every exit path.

    Preconditions (``ValueError``): the model currently holds ``samples.basis.pi0``.
    Postconditions: ``n_enveloped + n_refused_unbalanced == n_accepted``; the
    envelope's first member is ``nominal``, the others ``param-sample-<k>``; the
    union contains every member interval.
    """
    original = np.asarray(model.inertial_parameters(), dtype=np.float64)
    if not np.allclose(original, samples.basis.pi0, rtol=1e-9, atol=1e-12):
        raise ValueError("the model does not hold the nominal parameters of the basis")
    members, labels, refused = [], ["nominal"], 0
    try:
        for k, pi in zip(samples.sample_indices, samples.accepted, strict=True):
            model.set_inertial_parameters(pi)
            try:
                members.append(series_of())
            except BalanceInconsistentError:
                refused += 1
                continue
            labels.append(f"param-sample-{k}")
    finally:
        model.set_inertial_parameters(original)
    union = envelope_union_intersection(
        [nominal, *members], labels, hypothesis_ids=(samples.hypothesis_id,)
    )
    infeasible = sum(
        st is BoundStatus.INFEASIBLE for m in members for st in m.chebyshev_status
    )
    return ParameterSensitivityReport(
        envelope=union,
        quantities=tuple(
            _quantity_sensitivity(nominal, members, union, j)
            for j in range(len(nominal.names))
        ),
        n_requested=samples.n_requested,
        n_lmi_rejected=samples.n_rejected,
        n_refused_unbalanced=refused,
        n_enveloped=len(members),
        member_frames_infeasible=int(infeasible),
        hypothesis_ids=(samples.hypothesis_id,),
        identified_rank=samples.basis.rank,
        null_dimension=samples.basis.null_dimension,
        max_identified_drift=samples.max_identified_drift,
    )
