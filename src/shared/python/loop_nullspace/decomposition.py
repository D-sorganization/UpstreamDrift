"""Closed-loop decomposition of ``A x = b`` on ``ForceNullSpace`` (NSX-15).

One instant of the closed-loop dynamics is the linear system::

    A x = b,   A = [S^T  J_c^T  (J_g^T)],   x = (tau, lambda, (f_g)),
    b = M a + h,

for the actuated torques ``tau``, the closure multipliers ``lambda`` and the
optional ground-contact loads ``f_g``.  ``decompose`` reuses
``ForceNullSpace.from_balance`` (scaled SVD, characteristic scales
``F_c = 100 N`` and ``M_c = 5 N m``, convention ``C-SCALE-FORCE-MOMENT``) for
the null basis, the rank and the scaled spectrum, and it adds the dimension
accounting of the specification:

* ``rank A = n_a + r_u`` with ``r_u = rank [J_cu^T J_gu^T]``;
* multiplier gauge ``m - rank J_c`` (directions of ``lambda`` that change no
  load);
* attainable torque variation ``rank J_c^T - rank J_cu^T`` (``f_g`` fixed), and
  the ``f_g``-free versions on the stacked ``[J_c; J_g]``.

``zero_side_solve`` classifies the question "what multiplier would a side that
supplies no torque need?" into the three cases of the specification and never
least-squares an inconsistent system silently.  ``min_metric_torque`` returns
the minimum-``W``-norm torque for a *declared* metric ``W``; ``W = M^-1`` is
only the labelled metric ``C-METRIC-MINV`` and is never called "Gauss".

Non-claims: no allocation is preferred here, and null-space coordinates ``z``
are not physiological quantities.  Trajectory-level entry points accept only an
``AdmittedTrajectory`` (NSX-14).  Generalised-force rows use the moment scale
``M_c`` for every coordinate, because the snapshot does not encode joint types.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.admission import AdmittedTrajectory
from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    ClosureKind,
)
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    QuantityRecord,
    check_shape,
    check_symmetric,
    readonly_float,
)
from src.shared.python.motion_matching.force_nullspace.space import ForceNullSpace

__all__ = [
    "AllocationMetric",
    "BalanceInconsistentError",
    "CharacteristicScales",
    "Decomposition",
    "DimensionAccounting",
    "ZeroSideCase",
    "ZeroSideResult",
    "decompose",
    "decompose_trajectory",
    "dimension_accounting",
    "min_metric_torque",
    "min_metric_torque_trajectory",
    "zero_side_solve",
    "zero_side_trajectory",
]

SCALE_CONVENTION = "C-SCALE-FORCE-MOMENT: F_c = 100 N, M_c = 5 N m"
RANK_RTOL = 1e-10
BALANCE_TOL = 1e-9
_LSTSQ_RCOND = 1e-10
SIDES = ("lead", "trail", "club_joint")
_SIDE_PREFIXES = {
    "lead": ("lead", "club"),
    "trail": ("trail",),
    "club_joint": ("club",),
}
_ROW_KINDS = {
    ClosureKind.WELD6: "FFFMMM",
    ClosureKind.SHAFT_FREE5: "FFFMM",
    ClosureKind.BALL3: "FFF",
    ClosureKind.PLANAR_POINT2: "FF",
    ClosureKind.PLANAR_WELD3: "FFM",
}


class BalanceInconsistentError(ValueError):
    """The unactuated rows of ``A x = b`` cannot be balanced (never repaired)."""

    def __init__(self, message: str, residual_scaled: float) -> None:
        super().__init__(message)
        self.residual_scaled = float(residual_scaled)


@dataclass(frozen=True)
class CharacteristicScales:
    """Characteristic force (N) and moment (N m) for rank decisions."""

    force: float = 100.0
    moment: float = 5.0

    def __post_init__(self) -> None:
        for name in ("force", "moment"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"scale {name} must be positive, got {value}")

    def to_dict(self) -> dict[str, float]:
        """JSON-compatible form."""
        return {"force": float(self.force), "moment": float(self.moment)}


@dataclass(frozen=True)
class AllocationMetric:
    """A declared positive-definite metric ``W`` on the actuated torques.

    ``label`` names the convention; ``M^-1`` is available only through
    :meth:`inverse_mass`, which labels it ``C-METRIC-MINV``.
    """

    W: Array
    label: str

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("AllocationMetric requires a non-empty label")
        w = readonly_float(self.W, "AllocationMetric.W")
        if w.ndim != 2 or w.shape[0] != w.shape[1]:
            raise ValueError(f"AllocationMetric.W must be square, got shape {w.shape}")
        check_symmetric(w, "AllocationMetric.W")
        try:
            np.linalg.cholesky(0.5 * (w + w.T))
        except np.linalg.LinAlgError as exc:
            raise ValueError("AllocationMetric.W must be positive definite") from exc
        object.__setattr__(self, "W", w)

    @classmethod
    def identity(cls, n_actuated: int) -> AllocationMetric:
        """``W = I`` on ``n_actuated`` torques."""
        return cls(np.eye(int(n_actuated)), "C-METRIC-IDENTITY")

    @classmethod
    def inverse_mass(cls, snapshot: ClosedLoopSnapshot) -> AllocationMetric:
        """``W = (M^-1)[actuated, actuated]``, the labelled ``C-METRIC-MINV``."""
        idx = list(snapshot.topology.actuated_indices)
        inv = np.linalg.inv(snapshot.M)
        sub = inv[np.ix_(idx, idx)]
        return cls(0.5 * (sub + sub.T), "C-METRIC-MINV")


Weights = AllocationMetric | ArrayLike
WeightsSource = AllocationMetric | ArrayLike | Callable[[ClosedLoopSnapshot], Any]


def _coerce_metric(weights: Any, n_a: int) -> AllocationMetric:
    if isinstance(weights, AllocationMetric):
        metric = weights
    elif isinstance(weights, (np.ndarray, list, tuple)):
        metric = AllocationMetric(np.asarray(weights, dtype=float), "USER_DECLARED")
    else:
        raise TypeError(
            "weights must be an AllocationMetric or a matrix, got "
            f"{type(weights).__name__}"
        )
    if metric.W.shape != (n_a, n_a):
        raise ValueError(
            f"metric shape {metric.W.shape} does not match the {n_a} actuated "
            f"torques; expected {(n_a, n_a)}"
        )
    return metric


# ------------------------------------------------------------------ scales --


def _lambda_scales(kind: ClosureKind, s: CharacteristicScales) -> Array:
    return np.array([s.force if c == "F" else s.moment for c in _ROW_KINDS[kind]])


def _variable_scale(snap: ClosedLoopSnapshot, s: CharacteristicScales) -> Array:
    n_g = 0 if snap.J_g is None else snap.J_g.shape[0]
    return np.concatenate(
        [
            np.full(snap.topology.n_actuated, s.moment),
            _lambda_scales(snap.topology.closure_kind, s),
            np.full(n_g, s.force),
        ]
    )


def _row_scale(snap: ClosedLoopSnapshot, s: CharacteristicScales) -> Array:
    return np.full(snap.topology.n, s.moment)


def _rank(mat: Array, rtol: float) -> int:
    if mat.size == 0:
        return 0
    sv = np.linalg.svd(mat, compute_uv=False)
    return int(np.count_nonzero(sv > rtol * max(float(sv[0]), 1.0)))


def _check_options(rank_rtol: float, scales: CharacteristicScales | None) -> Any:
    if not 0.0 < rank_rtol < 1.0:
        raise ValueError(f"rank_rtol must lie in (0, 1), got {rank_rtol}")
    return scales if scales is not None else CharacteristicScales()


def _load_columns(snap: ClosedLoopSnapshot) -> Array:
    """``[J_c^T (J_g^T)]``: the non-torque columns of ``A`` (unscaled)."""
    cols = [snap.closure_rows().T]
    if snap.J_g is not None:
        cols.append(snap.J_g.T)
    return np.hstack(cols)


# -------------------------------------------------------- dimension counts --


@dataclass(frozen=True)
class DimensionAccounting:
    """Rank-revealing dimension counts of ``A x = b`` (specification Prop. 1)."""

    n: int
    n_a: int
    n_u: int
    m: int
    n_g: int
    dim_x: int
    rank_A: int
    nullity: int
    rank_J_c: int
    rank_J_cu_T: int
    multiplier_gauge: int
    attainable_torque_variation: int
    rank_stacked: int
    rank_stacked_u: int
    multiplier_gauge_fg_free: int
    attainable_torque_variation_fg_free: int

    def to_dict(self) -> dict[str, int]:
        """JSON-compatible form."""
        return {k: int(v) for k, v in self.__dict__.items()}


def dimension_accounting(
    snapshot: ClosedLoopSnapshot,
    *,
    scales: CharacteristicScales | None = None,
    rank_rtol: float = RANK_RTOL,
) -> DimensionAccounting:
    """Rank-revealing counts for ``snapshot``; needs no consistent balance.

    Ranks use the scaled matrices (relative tolerance ``rank_rtol`` of
    ``max(sigma_max, 1)``).  Postcondition: ``rank_A = n_a + r_u``,
    ``nullity = dim_x - rank_A`` and, when ``n_g = 0``,
    ``nullity = multiplier_gauge + attainable_torque_variation``.
    """
    sc = _check_options(rank_rtol, scales)
    topo = snapshot.topology
    n, n_a, m = topo.n, topo.n_actuated, topo.closure_kind.rows
    n_g = 0 if snapshot.J_g is None else snapshot.J_g.shape[0]
    unact = list(topo.unactuated_indices)
    scale_cols = _variable_scale(snapshot, sc)[n_a:]
    scaled = _load_columns(snapshot) * scale_cols[None, :] / sc.moment
    jc_t = scaled[:, :m]
    r_jc, r_jcu = _rank(jc_t, rank_rtol), _rank(jc_t[unact], rank_rtol)
    r_stack, r_stack_u = _rank(scaled, rank_rtol), _rank(scaled[unact], rank_rtol)
    dim_x = n_a + m + n_g
    rank_a = n_a + r_stack_u
    gauge, attain = m - r_jc, r_jc - r_jcu
    return DimensionAccounting(
        n=n,
        n_a=n_a,
        n_u=len(unact),
        m=m,
        n_g=n_g,
        dim_x=dim_x,
        rank_A=rank_a,
        nullity=dim_x - rank_a,
        rank_J_c=r_jc,
        rank_J_cu_T=r_jcu,
        multiplier_gauge=gauge,
        attainable_torque_variation=attain,
        rank_stacked=r_stack,
        rank_stacked_u=r_stack_u,
        multiplier_gauge_fg_free=m + n_g - r_stack,
        attainable_torque_variation_fg_free=r_stack - r_stack_u,
    )


# ------------------------------------------------------------ decomposition --


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


@dataclass(frozen=True)
class Decomposition:
    """``x = x_p + N z`` with rank-revealing diagnostics for one frame.

    ``x_p`` is the minimum-metric-torque particular solution (minimum scaled
    norm among those); ``N`` has orthonormal columns in scaled coordinates.
    Arrays are read-only.  ``records`` holds the user-visible quantities as
    :class:`QuantityRecord` objects.
    """

    x_p: Array
    N: Array
    counts: DimensionAccounting
    singular_values_scaled: Array
    condition_number: float
    equality_residual: float
    metric_label: str
    scales: CharacteristicScales
    records: dict[str, QuantityRecord] = field(repr=False, default_factory=dict)

    @property
    def n_actuated(self) -> int:
        """Number of actuated torques ``n_a``."""
        return self.counts.n_a

    @property
    def m(self) -> int:
        """Number of closure multipliers."""
        return self.counts.m

    @property
    def dim_x(self) -> int:
        """Length of ``x = (tau, lambda, f_g)``."""
        return self.counts.dim_x

    @property
    def rank_A(self) -> int:
        """Numerical rank of ``A`` (``n_a + r_u``)."""
        return self.counts.rank_A

    @property
    def nullity(self) -> int:
        """Dimension of the equality null space."""
        return self.counts.nullity

    @property
    def multiplier_gauge(self) -> int:
        """``m - rank J_c``."""
        return self.counts.multiplier_gauge

    @property
    def attainable_torque_variation(self) -> int:
        """``rank J_c^T - rank J_cu^T`` (``f_g`` fixed)."""
        return self.counts.attainable_torque_variation

    @property
    def rank_J_c(self) -> int:
        """Numerical rank of the closure Jacobian."""
        return self.counts.rank_J_c

    @property
    def rank_J_cu_T(self) -> int:
        """Rank of the unactuated columns of ``J_c``."""
        return self.counts.rank_J_cu_T

    @property
    def torque_particular(self) -> Array:
        """Actuated torques of ``x_p``."""
        return self.x_p[: self.n_actuated]

    def x(self, z: ArrayLike) -> Array:
        """``x_p + N z`` for null coordinates ``z`` (length ``nullity``)."""
        zz = np.asarray(z, dtype=float)
        if zz.shape != (self.nullity,):
            raise ValueError(f"z must have shape ({self.nullity},), got {zz.shape}")
        return np.asarray(self.x_p + self.N @ zz)

    def torque(self, z: ArrayLike) -> Array:
        """Actuated torques of ``x_p + N z``."""
        return self.x(z)[: self.n_actuated]

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible per-frame diagnostics (no NaN/inf values)."""
        return {
            **self.counts.to_dict(),
            "x_p": self.x_p.tolist(),
            "N": self.N.tolist(),
            "singular_values_scaled": self.singular_values_scaled.tolist(),
            "condition_number": _finite_or_none(self.condition_number),
            "equality_residual": float(self.equality_residual),
            "metric_label": self.metric_label,
            "scales": self.scales.to_dict(),
            "conventions": [SCALE_CONVENTION],
        }


def _balance_check(snap: ClosedLoopSnapshot, sc: CharacteristicScales) -> None:
    """Raise unless the unactuated rows can be balanced by ``lambda, f_g``."""
    unact = list(snap.topology.unactuated_indices)
    if not unact:
        return
    cols = (
        _load_columns(snap)[unact]
        * _variable_scale(snap, sc)[snap.topology.n_actuated :][None, :]
    )
    b_u = snap.generalized_load()[unact]
    left, sv, _ = np.linalg.svd(cols / sc.moment, full_matrices=False)
    keep = left[:, : int(np.count_nonzero(sv > RANK_RTOL * max(sv[0], 1.0)))]
    rhs = b_u / sc.moment
    residual = float(np.linalg.norm(rhs - keep @ (keep.T @ rhs)))
    if residual > BALANCE_TOL * max(1.0, float(np.linalg.norm(rhs))):
        raise BalanceInconsistentError(
            "unactuated rows cannot be balanced by the closure and contact loads "
            f"(scaled residual {residual:.3e}); the state is not repaired here",
            residual,
        )


def _min_metric_particular(
    fns: ForceNullSpace, n_a: int, w: Array, torque_scale: float
) -> Array:
    """Minimum ``tau^T W tau`` member of ``x_p + N z``; min scaled ``|z|``.

    Directions of ``z`` that change no torque (the multiplier gauge) are cut by
    an absolute-scale threshold: ``N`` columns are unit in scaled coordinates,
    so a torque effect below ``1e-10 |W^1/2| torque_scale`` is numerical noise.
    """
    basis = fns.basis
    if basis.shape[1] == 0:
        return np.asarray(fns.particular)
    chol = np.linalg.cholesky(0.5 * (w + w.T))
    design = chol.T @ basis[:n_a]
    left, sv, right_t = np.linalg.svd(design, full_matrices=False)
    keep = sv > _LSTSQ_RCOND * float(np.linalg.norm(chol, 2)) * torque_scale
    target = -(chol.T @ fns.particular[:n_a])
    z = right_t[keep].T @ ((left[:, keep].T @ target) / sv[keep])
    return np.asarray(fns.particular + basis @ z)


def _scalar_record(
    snap: ClosedLoopSnapshot, value: float, units: str, conv: tuple[str, ...]
) -> QuantityRecord:
    common: dict[str, Any] = {
        "units": units,
        "model_provenance": snap.model_provenance,
        "input_provenance": snap.input_provenance,
        "conventions": conv,
    }
    if not np.isfinite(value):
        return QuantityRecord(
            values=np.array([np.nan]),
            allocation_dependence=AllocationDependence.UNAVAILABLE,
            reason="rank-deficient spectrum: condition number is infinite",
            **common,
        )
    return QuantityRecord(
        values=np.array([float(value)]),
        allocation_dependence=AllocationDependence.INVARIANT,
        **common,
    )


def _records(
    snap: ClosedLoopSnapshot,
    counts: DimensionAccounting,
    cond: float,
    tau: Array,
    label: str,
) -> dict[str, QuantityRecord]:
    conv = (SCALE_CONVENTION,)
    out = {
        name: _scalar_record(snap, float(getattr(counts, name)), "1", conv)
        for name in (
            "rank_A",
            "nullity",
            "multiplier_gauge",
            "attainable_torque_variation",
        )
    }
    out["condition_number"] = _scalar_record(snap, cond, "1", conv)
    out["torque_particular"] = QuantityRecord(
        values=tau,
        units="N m",
        allocation_dependence=AllocationDependence.FAMILY_DEPENDENT,
        model_provenance=snap.model_provenance,
        input_provenance=snap.input_provenance,
        conventions=(*conv, label),
        hypotheses=(f"minimum-norm torque for the declared metric {label}",),
    )
    return out


def decompose(
    snapshot: ClosedLoopSnapshot,
    *,
    weights: Weights,
    scales: CharacteristicScales | None = None,
    rank_rtol: float = RANK_RTOL,
) -> Decomposition:
    """Decompose ``A x = b`` into ``x_p + N z`` for ``snapshot``.

    ``weights`` is the declared metric on the actuated torques (an
    :class:`AllocationMetric`, or a bare matrix recorded as ``USER_DECLARED``);
    it selects the particular solution only, it does not prefer an allocation.

    Preconditions: ``snapshot`` is a :class:`ClosedLoopSnapshot` (``TypeError``),
    the metric matches ``n_a`` (``ValueError``), ``rank_rtol`` in ``(0, 1)``.
    Raises :class:`BalanceInconsistentError` if the unactuated rows cannot be
    balanced.  Postconditions: ``A x_p = b`` and ``A N = 0`` to the balance
    tolerance, ``N`` has ``dim_x - rank_A`` orthonormal (scaled) columns, and
    ``rank_A = n_a + r_u``.
    """
    if not isinstance(snapshot, ClosedLoopSnapshot):
        raise TypeError(
            f"decompose requires a ClosedLoopSnapshot, got {type(snapshot).__name__}"
        )
    sc = _check_options(rank_rtol, scales)
    n_a = snapshot.topology.n_actuated
    metric = _coerce_metric(weights, n_a)
    counts = dimension_accounting(snapshot, scales=sc, rank_rtol=rank_rtol)
    _balance_check(snapshot, sc)
    matrix = np.hstack([snapshot.actuated_rows().T, _load_columns(snapshot)])
    fns = ForceNullSpace.from_balance(
        matrix,
        snapshot.generalized_load(),
        variable_scale=_variable_scale(snapshot, sc),
        row_scale=_row_scale(snapshot, sc),
        rank_rtol=rank_rtol,
        balance_tolerance=BALANCE_TOL,
    )
    if fns.rank != counts.rank_A:
        raise ValueError(
            f"rank accounting mismatch: SVD rank {fns.rank} != n_a + r_u = "
            f"{counts.rank_A}"
        )
    x_p = _min_metric_particular(fns, n_a, metric.W, sc.moment)
    info = fns.analyze()
    return Decomposition(
        x_p=readonly_float(x_p, "x_p"),
        N=readonly_float(fns.basis, "N"),
        counts=counts,
        singular_values_scaled=readonly_float(info.singular_values, "sv"),
        condition_number=info.condition_number,
        equality_residual=info.equality_residual,
        metric_label=metric.label,
        scales=sc,
        records=_records(
            snapshot, counts, info.condition_number, x_p[:n_a], metric.label
        ),
    )


def min_metric_torque(
    snapshot: ClosedLoopSnapshot,
    W: Weights,
    *,
    scales: CharacteristicScales | None = None,
) -> Array:
    """Minimum ``tau^T W tau`` actuated torque that produces the measured motion.

    ``W`` must be declared; ``M^-1`` is allowed as the labelled metric
    ``AllocationMetric.inverse_mass`` (``C-METRIC-MINV``), never as "Gauss".
    Postcondition: shape ``(n_a,)``; the torque, with its multipliers, satisfies
    the balance.  Free ground loads are included in the family when present.
    """
    return decompose(snapshot, weights=W, scales=scales).torque_particular


# --------------------------------------------------------------- zero side --


class ZeroSideCase(Enum):
    """The three outcomes of a zero-side solve."""

    CONSISTENT_UNIQUE = "CONSISTENT_UNIQUE"
    CONSISTENT_NONUNIQUE = "CONSISTENT_NONUNIQUE"
    INCONSISTENT = "INCONSISTENT"


@dataclass(frozen=True)
class ZeroSideResult:
    """Classification of ``J_{c,s}^T lambda = tau_ID,s`` for one side.

    ``multiplier`` is the minimum-residual, minimum-norm multiplier (damped when
    ``damping > 0``); for ``INCONSISTENT`` it is labelled
    ``"minimum-residual side effort"`` and the side is never called zero.
    ``residual`` is in physical units, ``residual_scaled`` in characteristic
    units; ``nullity = m - rank`` is the affine solution-set dimension when
    consistent.
    """

    side: str
    indices: tuple[int, ...]
    case: ZeroSideCase
    n_equations: int
    n_unknowns: int
    rank: int
    nullity: int
    residual: float
    residual_scaled: float
    multiplier: Array
    singular_values: Array
    condition_number: float
    damping: float
    rank_rtol: float
    label: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form."""
        return {
            "side": self.side,
            "indices": list(self.indices),
            "case": self.case.value,
            "n_equations": self.n_equations,
            "n_unknowns": self.n_unknowns,
            "rank": self.rank,
            "nullity": self.nullity,
            "residual": float(self.residual),
            "residual_scaled": float(self.residual_scaled),
            "multiplier": self.multiplier.tolist(),
            "singular_values": self.singular_values.tolist(),
            "condition_number": _finite_or_none(self.condition_number),
            "damping": float(self.damping),
            "rank_rtol": float(self.rank_rtol),
            "label": self.label,
        }


def _side_indices(
    snap: ClosedLoopSnapshot, side: str, indices: Sequence[int] | None
) -> tuple[int, ...]:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    n = snap.topology.n
    if indices is not None:
        idx = tuple(int(i) for i in indices)
        if not idx or len(set(idx)) != len(idx) or not all(0 <= i < n for i in idx):
            raise ValueError(f"indices must be unique coordinates in 0..{n - 1}")
        return idx
    prefixes = _SIDE_PREFIXES[side]
    found = tuple(
        i
        for i, name in enumerate(snap.topology.coordinate_order)
        if name.startswith(prefixes)
    )
    if not found:
        raise ValueError(
            f"no coordinate of {snap.topology.coordinate_order} belongs to side "
            f"{side!r}; pass indices explicitly"
        )
    return found


def zero_side_solve(
    snapshot: ClosedLoopSnapshot,
    side: str,
    *,
    indices: Sequence[int] | None = None,
    damping: float = 0.0,
    rank_rtol: float = RANK_RTOL,
    consistency_tol: float = BALANCE_TOL,
    scales: CharacteristicScales | None = None,
) -> ZeroSideResult:
    """Classify the zero-``side`` question into its three cases.

    Solves ``J_{c,s}^T lambda = b_s`` for the coordinates ``I_s`` of ``side``
    (``"lead"``, ``"trail"`` or ``"club_joint"``, resolved from the coordinate
    names unless ``indices`` is given).  ``CONSISTENT_*`` requires the scaled
    residual ``|| (I - J^T (J^T)^+) b_s ||`` to be within ``consistency_tol``;
    otherwise the case is ``INCONSISTENT`` and the minimum-residual multiplier
    is returned with its label.  ``damping`` (Tikhonov, ``>= 0``) only shapes
    the returned multiplier, is recorded, and never changes the classification.

    Raises ``ValueError`` for an unknown side, bad indices or negative damping.
    """
    sc = _check_options(rank_rtol, scales)
    if not np.isfinite(damping) or damping < 0.0:
        raise ValueError(f"damping must be finite and non-negative, got {damping}")
    idx = _side_indices(snapshot, side, indices)
    kind = snapshot.topology.closure_kind
    lam_scale = _lambda_scales(kind, sc)
    jt = snapshot.closure_rows()[:, list(idx)].T
    rows = _row_scale(snapshot, sc)[list(idx)]
    scaled = jt * lam_scale[None, :] / rows[:, None]
    rhs = snapshot.generalized_load()[list(idx)] / rows
    left, sv, right_t = np.linalg.svd(scaled, full_matrices=False)
    rank = int(np.count_nonzero(sv > rank_rtol * max(float(sv[0]), 1.0)))
    resid_vec = rhs - left[:, :rank] @ (left[:, :rank].T @ rhs)
    res_scaled = float(np.linalg.norm(resid_vec))
    gain = sv[:rank] / (sv[:rank] ** 2 + damping**2)
    y = right_t[:rank].T @ (gain * (left[:, :rank].T @ rhs))
    m = scaled.shape[1]
    consistent = res_scaled <= consistency_tol * max(1.0, float(np.linalg.norm(rhs)))
    if not consistent:
        case = ZeroSideCase.INCONSISTENT
    elif rank == m:
        case = ZeroSideCase.CONSISTENT_UNIQUE
    else:
        case = ZeroSideCase.CONSISTENT_NONUNIQUE
    cond = float(sv[0] / sv[rank - 1]) if rank > 0 else float("inf")
    return ZeroSideResult(
        side=side,
        indices=idx,
        case=case,
        n_equations=len(idx),
        n_unknowns=m,
        rank=rank,
        nullity=m - rank,
        residual=float(np.linalg.norm(rows * resid_vec)),
        residual_scaled=res_scaled,
        multiplier=readonly_float(lam_scale * y, "multiplier"),
        singular_values=readonly_float(sv, "singular_values"),
        condition_number=cond,
        damping=float(damping),
        rank_rtol=float(rank_rtol),
        label="minimum-residual side effort"
        if case is ZeroSideCase.INCONSISTENT
        else "",
    )


# -------------------------------------------------------------- trajectories --


def _frames(trajectory: Any, who: str) -> tuple[ClosedLoopSnapshot, ...]:
    if not isinstance(trajectory, AdmittedTrajectory):
        raise TypeError(
            f"{who} requires an AdmittedTrajectory (build one with "
            "AdmittedTrajectory.from_report(admit(...))), got "
            f"{type(trajectory).__name__}"
        )
    return trajectory.snapshots


def _frame_weights(weights: WeightsSource, snap: ClosedLoopSnapshot) -> Any:
    if callable(weights) and not isinstance(weights, (AllocationMetric, np.ndarray)):
        return weights(snap)
    return weights


def decompose_trajectory(
    trajectory: AdmittedTrajectory,
    *,
    weights: WeightsSource,
    scales: CharacteristicScales | None = None,
) -> tuple[Decomposition, ...]:
    """:func:`decompose` on every frame of an admitted trajectory.

    ``weights`` is a metric, or a callable ``snapshot -> metric`` (for example
    ``AllocationMetric.inverse_mass``).  Raises ``TypeError`` for anything but
    an :class:`AdmittedTrajectory`.
    """
    return tuple(
        decompose(s, weights=_frame_weights(weights, s), scales=scales)
        for s in _frames(trajectory, "decompose_trajectory")
    )


def zero_side_trajectory(
    trajectory: AdmittedTrajectory, side: str, **options: Any
) -> tuple[ZeroSideResult, ...]:
    """:func:`zero_side_solve` on every frame of an admitted trajectory."""
    return tuple(
        zero_side_solve(s, side, **options)
        for s in _frames(trajectory, "zero_side_trajectory")
    )


def min_metric_torque_trajectory(
    trajectory: AdmittedTrajectory,
    W: WeightsSource,
    *,
    scales: CharacteristicScales | None = None,
) -> tuple[Array, ...]:
    """:func:`min_metric_torque` on every frame of an admitted trajectory."""
    return tuple(
        min_metric_torque(s, _frame_weights(W, s), scales=scales)
        for s in _frames(trajectory, "min_metric_torque_trajectory")
    )
