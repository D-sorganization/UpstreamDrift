"""Static torque and hand-split envelopes with Chebyshev margin and sign classification (NSX-22, #12565).

Computes per-frame lower and upper bounds of linear quantities over the
equality-reduced :class:`FeasibleSet`, classifies sign determinacy
(``POSITIVE``, ``NEGATIVE``, ``INDETERMINATE``, ``ZERO``), and computes the
Chebyshev radius and center in the declared metric.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

from src.shared.python.loop_nullspace.feasible_set import (
    FeasibleSet,
    FloatArray,
    _readonly_1d,
    _readonly_2d,
)

BoundStatus = Literal["BOUNDED", "UNBOUNDED", "INFEASIBLE"]
SignClass = Literal["POSITIVE", "NEGATIVE", "INDETERMINATE", "ZERO"]


@dataclass(frozen=True)
class LinearQuantitySpec:
    """Linear scalar quantity ``q(z) = offset + c_z^T z`` or ``offset + c_x^T x(z)``."""

    name: str
    c_z: FloatArray | None = None
    offset: float = 0.0
    c_x: FloatArray | None = None
    index: int | None = None
    units: str = "N*m"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("LinearQuantitySpec.name must be a non-empty string")
        object.__setattr__(self, "name", self.name.strip())
        if self.c_z is not None:
            object.__setattr__(self, "c_z", _readonly_1d(self.c_z, "c_z"))
        if self.c_x is not None:
            object.__setattr__(self, "c_x", _readonly_1d(self.c_x, "c_x"))
        off = float(self.offset)
        if not math.isfinite(off):
            raise ValueError("offset must be finite")
        object.__setattr__(self, "offset", off)

    def resolve(self, fs: FeasibleSet) -> tuple[FloatArray, float]:
        """Resolve ``(c_z, constant_offset)`` in the ``z``-coordinates of ``fs``."""
        if self.c_z is not None:
            if self.c_z.shape != (fs.dim_z,):
                raise ValueError(
                    f"c_z length ({self.c_z.shape[0]}) must match fs.dim_z ({fs.dim_z})"
                )
            return self.c_z, self.offset
        if self.c_x is not None:
            if self.c_x.shape != (fs.dim_x,):
                raise ValueError(
                    f"c_x length ({self.c_x.shape[0]}) must match fs.dim_x ({fs.dim_x})"
                )
            c_z = _readonly_1d(self.c_x @ fs.null_basis, "c_z")
            const = float(self.offset + self.c_x @ fs.x_particular)
            return c_z, const
        if self.index is not None:
            idx = int(self.index)
            if idx < 0 or idx >= fs.dim_x:
                raise ValueError(f"index {idx} out of bounds for dim_x={fs.dim_x}")
            c_z = _readonly_1d(fs.null_basis[idx, :], "c_z")
            const = float(self.offset + fs.x_particular[idx])
            return c_z, const
        raise ValueError(
            f"LinearQuantitySpec {self.name!r} must specify c_z, c_x, or index"
        )


@dataclass(frozen=True)
class QuantityEnvelope:
    """Per-frame lower/upper static bounds and sign classification for one quantity."""

    name: str
    lower: FloatArray
    upper: FloatArray
    lower_status: tuple[BoundStatus, ...]
    upper_status: tuple[BoundStatus, ...]
    sign_class: tuple[SignClass, ...]
    units: str = "N*m"
    hypotheses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        lo = np.array(self.lower, dtype=np.float64, copy=True)
        hi = np.array(self.upper, dtype=np.float64, copy=True)
        if lo.ndim != 1 or hi.ndim != 1 or lo.shape != hi.shape:
            raise ValueError("lower and upper must be 1-D arrays of equal length")
        lo.setflags(write=False)
        hi.setflags(write=False)
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)
        object.__setattr__(self, "lower_status", tuple(self.lower_status))
        object.__setattr__(self, "upper_status", tuple(self.upper_status))
        object.__setattr__(self, "sign_class", tuple(self.sign_class))
        object.__setattr__(self, "hypotheses", tuple(self.hypotheses))

    @property
    def is_bounded(self) -> bool:
        """True if every frame has finite lower and upper bounds."""
        return all(s == "BOUNDED" for s in self.lower_status) and all(
            s == "BOUNDED" for s in self.upper_status
        )

    @property
    def width(self) -> FloatArray:
        """Per-frame envelope width ``upper - lower`` (read-only)."""
        diff = np.asarray(self.upper - self.lower, dtype=np.float64)
        diff.setflags(write=False)
        return diff


@dataclass(frozen=True)
class EnvelopeSeries:
    """Static per-frame envelope collection with Chebyshev margins (NSX-22)."""

    envelopes: tuple[QuantityEnvelope, ...]
    chebyshev_margins: FloatArray
    chebyshev_centers: tuple[FloatArray, ...] = ()
    series_label: str = "STATIC"
    evidence_class: str = "ENVELOPE"
    warm_start_used: bool = True
    total_lp_iterations: int = 0

    def __post_init__(self) -> None:
        margins = np.array(self.chebyshev_margins, dtype=np.float64, copy=True)
        margins.setflags(write=False)
        object.__setattr__(self, "envelopes", tuple(self.envelopes))
        object.__setattr__(self, "chebyshev_margins", margins)
        object.__setattr__(self, "chebyshev_centers", tuple(self.chebyshev_centers))

    def __getitem__(self, key: int | str) -> QuantityEnvelope:
        """Look up a :class:`QuantityEnvelope` by integer index or quantity name."""
        if isinstance(key, int):
            return self.envelopes[key]
        return self.by_name(key)

    def by_name(self, name: str) -> QuantityEnvelope:
        """Return the :class:`QuantityEnvelope` matching ``name``."""
        for env in self.envelopes:
            if env.name == name:
                return env
        raise KeyError(f"Unknown quantity envelope: {name!r}")

    @property
    def lower_matrix(self) -> FloatArray:
        """2-D array of lower bounds with shape ``(n_frames, n_quantities)``."""
        if not self.envelopes:
            empty = np.zeros((self.chebyshev_margins.size, 0), dtype=np.float64)
            empty.setflags(write=False)
            return empty
        mat = np.column_stack([e.lower for e in self.envelopes])
        mat.setflags(write=False)
        return mat

    @property
    def upper_matrix(self) -> FloatArray:
        """2-D array of upper bounds with shape ``(n_frames, n_quantities)``."""
        if not self.envelopes:
            empty = np.zeros((self.chebyshev_margins.size, 0), dtype=np.float64)
            empty.setflags(write=False)
            return empty
        mat = np.column_stack([e.upper for e in self.envelopes])
        mat.setflags(write=False)
        return mat


def classify_sign(
    lower: float,
    upper: float,
    *,
    zero_tol: float = 1e-9,
    lower_status: BoundStatus = "BOUNDED",
    upper_status: BoundStatus = "BOUNDED",
) -> SignClass:
    """Classify the sign determinacy of an interval ``[lower, upper]`` within ``zero_tol``."""
    if not math.isfinite(zero_tol) or zero_tol < 0.0:
        raise ValueError(f"zero_tol must be finite and >= 0, got {zero_tol}")
    if lower_status == "INFEASIBLE" or upper_status == "INFEASIBLE":
        return "INDETERMINATE"

    lo_finite = not math.isnan(lower) and math.isfinite(lower)
    hi_finite = not math.isnan(upper) and math.isfinite(upper)

    if lo_finite and lower > zero_tol:
        return "POSITIVE"
    if hi_finite and upper < -zero_tol:
        return "NEGATIVE"
    if lo_finite and hi_finite and abs(lower) <= zero_tol and abs(upper) <= zero_tol:
        return "ZERO"
    return "INDETERMINATE"


def _solve_directional_lp(
    c_z: FloatArray,
    offset: float,
    a_ub: FloatArray,
    b_ub: FloatArray,
    *,
    warm_basis: FloatArray | None = None,
) -> tuple[float, BoundStatus, int, FloatArray | None]:
    """Minimize ``offset + c_z^T z`` s.t. ``a_ub z <= b_ub``; never clamp unbounded rays."""
    m_rows, dim_z = a_ub.shape
    if dim_z == 0:
        if m_rows > 0 and np.any(b_ub < -1e-9):
            return float("nan"), "INFEASIBLE", 0, None
        return float(offset), "BOUNDED", 0, np.zeros(0, dtype=np.float64)

    if np.all(np.abs(c_z) <= 1e-15):
        if m_rows > 0:
            res_feas = linprog(
                np.zeros(dim_z),
                A_ub=a_ub,
                b_ub=b_ub,
                bounds=[(None, None)] * dim_z,
                method="highs",
            )
            if res_feas.status == 2:
                return (
                    float("nan"),
                    "INFEASIBLE",
                    int(getattr(res_feas, "nit", 0)),
                    None,
                )
        return float(offset), "BOUNDED", 0, warm_basis

    if m_rows == 0:
        return float("nan"), "UNBOUNDED", 0, None

    res = linprog(
        c_z,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * dim_z,
        method="highs",
    )
    nit = int(getattr(res, "nit", 0))
    if res.status == 0 and res.x is not None:
        sol_z = np.asarray(res.x, dtype=np.float64)
        return float(offset + c_z @ sol_z), "BOUNDED", nit, sol_z
    if res.status == 3:
        return float("nan"), "UNBOUNDED", nit, None
    return float("nan"), "INFEASIBLE", nit, None


def _resolve_quantity_specs(
    first_fs: FeasibleSet,
    quantities: Any,
) -> tuple[LinearQuantitySpec, ...]:
    """Normalize ``quantities`` argument into a tuple of :class:`LinearQuantitySpec`."""
    if quantities is None or quantities == "tau":
        return tuple(
            LinearQuantitySpec(name=f"tau_{i}", index=i, units="N*m")
            for i in range(first_fs.actuated_count)
        )
    if quantities == "x":
        return tuple(
            LinearQuantitySpec(name=f"x_{i}", index=i, units="mixed_SI")
            for i in range(first_fs.dim_x)
        )
    if quantities == "z":
        eye = np.eye(first_fs.dim_z, dtype=np.float64)
        return tuple(
            LinearQuantitySpec(name=f"z_{j}", c_z=eye[j], offset=0.0, units="1")
            for j in range(first_fs.dim_z)
        )
    if isinstance(quantities, Mapping):
        specs: list[LinearQuantitySpec] = []
        for k, v in quantities.items():
            vec = _readonly_1d(v, str(k))
            if vec.shape == (first_fs.dim_z,):
                specs.append(LinearQuantitySpec(name=str(k), c_z=vec))
            else:
                specs.append(LinearQuantitySpec(name=str(k), c_x=vec))
        return tuple(specs)

    out: list[LinearQuantitySpec] = []
    for idx, item in enumerate(quantities):
        if isinstance(item, LinearQuantitySpec):
            out.append(item)
        elif isinstance(item, (int, np.integer)):
            out.append(LinearQuantitySpec(name=f"q_{int(item)}", index=int(item)))
        elif isinstance(item, np.ndarray):
            vec = _readonly_1d(item, f"quantity_{idx}")
            if vec.shape == (first_fs.dim_z,):
                out.append(LinearQuantitySpec(name=f"q_{idx}", c_z=vec))
            else:
                out.append(LinearQuantitySpec(name=f"q_{idx}", c_x=vec))
        else:
            raise TypeError(f"Unsupported quantity specification: {item!r}")
    return tuple(out)


def _dual_row_norms(a_ub: FloatArray, metric_weights: ArrayLike | None) -> FloatArray:
    """Compute dual norm ``||a_i||_{W^{-1}}`` for each row of ``a_ub``."""
    m_rows, dim_z = a_ub.shape
    if m_rows == 0:
        return np.zeros(0, dtype=np.float64)
    if metric_weights is None:
        return np.linalg.norm(a_ub, axis=1)

    w_arr = np.asarray(metric_weights, dtype=np.float64)
    if not np.isfinite(w_arr).all():
        raise ValueError("metric_weights must be finite")
    if w_arr.ndim == 1:
        if w_arr.shape != (dim_z,) or np.any(w_arr <= 0.0):
            raise ValueError(f"1-D metric_weights must be positive of length {dim_z}")
        inv_w = 1.0 / w_arr
        return np.sqrt(np.sum((a_ub**2) * inv_w[None, :], axis=1))
    if w_arr.ndim == 2:
        w_mat = _readonly_2d(w_arr, "metric_weights")
        if w_mat.shape != (dim_z, dim_z):
            raise ValueError(f"2-D metric_weights must have shape ({dim_z}, {dim_z})")
        w_inv_a = np.linalg.solve(w_mat, a_ub.T).T
        return np.sqrt(np.maximum(0.0, np.sum(a_ub * w_inv_a, axis=1)))
    raise ValueError("metric_weights must be 1-D or 2-D")


def _centered_point_in_eroded_polytope(
    a_ub: FloatArray,
    b_eroded: FloatArray,
    lp_center: FloatArray,
) -> FloatArray:
    """Find the coordinate-centered Chebyshev center within ``A_ub z <= b_eroded``."""
    dim_z = int(a_ub.shape[1])
    z_lo = np.empty(dim_z, dtype=np.float64)
    z_hi = np.empty(dim_z, dtype=np.float64)
    bounds = [(None, None)] * dim_z
    for j in range(dim_z):
        e_j = np.zeros(dim_z, dtype=np.float64)
        e_j[j] = 1.0
        res_min = linprog(e_j, A_ub=a_ub, b_ub=b_eroded, bounds=bounds, method="highs")
        res_max = linprog(-e_j, A_ub=a_ub, b_ub=b_eroded, bounds=bounds, method="highs")
        if (
            res_min.status == 0
            and res_max.status == 0
            and res_min.x is not None
            and res_max.x is not None
        ):
            z_lo[j] = float(res_min.x[j])
            z_hi[j] = float(res_max.x[j])
        else:
            return _readonly_1d(lp_center, "chebyshev_center")

    midpoint = 0.5 * (z_lo + z_hi)
    if np.all(a_ub @ midpoint <= b_eroded + 1e-9):
        return _readonly_1d(midpoint, "chebyshev_center")
    return _readonly_1d(lp_center, "chebyshev_center")


def chebyshev_ball(
    feasible_set: FeasibleSet,
    *,
    metric_weights: ArrayLike | None = None,
    tol: float = 1e-9,
) -> tuple[float, FloatArray]:
    """Compute Chebyshev radius ``r >= 0`` and center ``z_cheb`` per Eq. (17) of NSX-00."""
    if not isinstance(feasible_set, FeasibleSet):
        raise TypeError("feasible_set must be a FeasibleSet instance")
    dim_z = feasible_set.dim_z
    nan_center = np.full(dim_z, np.nan, dtype=np.float64)
    nan_center.setflags(write=False)

    if feasible_set.is_empty(tol=tol):
        return 0.0, nan_center
    if dim_z == 0:
        zero_center = np.zeros(0, dtype=np.float64)
        zero_center.setflags(write=False)
        return 0.0, zero_center

    a_ub = feasible_set.A_ub
    b_ub = feasible_set.b_ub
    if a_ub.shape[0] == 0:
        return float("nan"), nan_center

    row_norms = _dual_row_norms(a_ub, metric_weights)
    # Maximize r subject to a_i^T z + ||a_i||_* r <= b_i, r >= 0
    c_lp = np.zeros(dim_z + 1, dtype=np.float64)
    c_lp[-1] = -1.0
    a_cheb = np.column_stack([a_ub, row_norms])
    bounds = [(None, None)] * dim_z + [(0.0, None)]
    res = linprog(c_lp, A_ub=a_cheb, b_ub=b_ub, bounds=bounds, method="highs")

    if res.status == 3:
        return float("nan"), nan_center
    if res.status != 0 or res.x is None:
        return 0.0, nan_center

    r_star = max(0.0, float(res.x[-1]))
    lp_z = np.asarray(res.x[:dim_z], dtype=np.float64)
    b_eroded = b_ub - r_star * row_norms
    center = _centered_point_in_eroded_polytope(a_ub, b_eroded, lp_z)
    return r_star, center


def chebyshev_margin(
    feasible_set: FeasibleSet,
    *,
    metric_weights: ArrayLike | None = None,
    tol: float = 1e-9,
) -> float:
    """Return the Chebyshev inscribed ball radius ``r >= 0`` (``0.0`` when infeasible)."""
    radius, _ = chebyshev_ball(feasible_set, metric_weights=metric_weights, tol=tol)
    return radius


def _solve_quantity_bounds(
    fs: FeasibleSet,
    spec: LinearQuantitySpec,
    w_min: FloatArray | None,
    w_max: FloatArray | None,
    zero_tol: float,
) -> tuple[
    float,
    float,
    BoundStatus,
    BoundStatus,
    SignClass,
    int,
    FloatArray | None,
    FloatArray | None,
]:
    """Solve directional min/max LPs and sign classification for one quantity at one frame."""
    c_z, offset = spec.resolve(fs)
    a_ub, b_ub = fs.A_ub, fs.b_ub
    lo_val, lo_st, it_lo, z_lo = _solve_directional_lp(
        c_z, offset, a_ub, b_ub, warm_basis=w_min
    )
    neg_hi_val, hi_st, it_hi, z_hi = _solve_directional_lp(
        -c_z, -offset, a_ub, b_ub, warm_basis=w_max
    )
    hi_val = -neg_hi_val if hi_st == "BOUNDED" else float("nan")
    sign_cl = classify_sign(
        lo_val, hi_val, zero_tol=zero_tol, lower_status=lo_st, upper_status=hi_st
    )
    return lo_val, hi_val, lo_st, hi_st, sign_cl, it_lo + it_hi, z_lo, z_hi


def static_envelopes(
    feasible_sets: Sequence[FeasibleSet] | FeasibleSet,
    quantities: Any = None,
    *,
    zero_tol: float = 1e-9,
    metric_weights: ArrayLike | None = None,
    warm_start: bool = True,
) -> EnvelopeSeries:
    """Compute per-frame static min/max envelopes, sign classes, and Chebyshev margins."""
    fs_Sequence: tuple[FeasibleSet, ...] = (
        (feasible_sets,)
        if isinstance(feasible_sets, FeasibleSet)
        else tuple(feasible_sets)
    )
    if not fs_Sequence:
        raise ValueError("feasible_sets must be non-empty")
    for fs in fs_Sequence:
        if not isinstance(fs, FeasibleSet):
            raise TypeError("Every entry in feasible_sets must be a FeasibleSet")

    specs = _resolve_quantity_specs(fs_Sequence[0], quantities)
    n_frames = len(fs_Sequence)
    n_q = len(specs)

    lowers = np.full((n_frames, n_q), np.nan, dtype=np.float64)
    uppers = np.full((n_frames, n_q), np.nan, dtype=np.float64)
    lo_statuses: list[list[BoundStatus]] = [[] for _ in range(n_q)]
    hi_statuses: list[list[BoundStatus]] = [[] for _ in range(n_q)]
    signs: list[list[SignClass]] = [[] for _ in range(n_q)]

    margins = np.zeros(n_frames, dtype=np.float64)
    centers: list[FloatArray] = []
    total_iters = 0
    prev_min_z: list[FloatArray | None] = [None] * n_q
    prev_max_z: list[FloatArray | None] = [None] * n_q

    for t_idx, fs in enumerate(fs_Sequence):
        if fs.is_empty(tol=zero_tol):
            nan_c = np.full(fs.dim_z, np.nan, dtype=np.float64)
            nan_c.setflags(write=False)
            margins[t_idx] = 0.0
            centers.append(nan_c)
            for q_idx in range(n_q):
                lo_statuses[q_idx].append("INFEASIBLE")
                hi_statuses[q_idx].append("INFEASIBLE")
                signs[q_idx].append("INDETERMINATE")
            continue

        r_t, c_t = chebyshev_ball(fs, metric_weights=metric_weights, tol=zero_tol)
        margins[t_idx] = r_t
        centers.append(c_t)

        for q_idx, spec in enumerate(specs):
            w_min = prev_min_z[q_idx] if warm_start else None
            w_max = prev_max_z[q_idx] if warm_start else None
            lo_val, hi_val, lo_st, hi_st, sign_cl, iters, z_lo, z_hi = (
                _solve_quantity_bounds(fs, spec, w_min, w_max, zero_tol)
            )
            total_iters += iters
            prev_min_z[q_idx] = z_lo
            prev_max_z[q_idx] = z_hi
            lowers[t_idx, q_idx] = lo_val
            uppers[t_idx, q_idx] = hi_val
            lo_statuses[q_idx].append(lo_st)
            hi_statuses[q_idx].append(hi_st)
            signs[q_idx].append(sign_cl)

    union_hyps: list[str] = []
    for fs in fs_Sequence:
        union_hyps.extend(fs.hypothesis_ids)
    dedup_hyps = tuple(dict.fromkeys(union_hyps))

    q_envelopes = tuple(
        QuantityEnvelope(
            name=specs[q_idx].name,
            lower=lowers[:, q_idx],
            upper=uppers[:, q_idx],
            lower_status=tuple(lo_statuses[q_idx]),
            upper_status=tuple(hi_statuses[q_idx]),
            sign_class=tuple(signs[q_idx]),
            units=specs[q_idx].units,
            hypotheses=dedup_hyps,
        )
        for q_idx in range(n_q)
    )
    return EnvelopeSeries(
        envelopes=q_envelopes,
        chebyshev_margins=margins,
        chebyshev_centers=tuple(centers),
        series_label="STATIC",
        evidence_class="ENVELOPE",
        warm_start_used=bool(warm_start),
        total_lp_iterations=total_iters,
    )


__all__ = [
    "BoundStatus",
    "EnvelopeSeries",
    "LinearQuantitySpec",
    "QuantityEnvelope",
    "SignClass",
    "chebyshev_ball",
    "chebyshev_margin",
    "classify_sign",
    "static_envelopes",
]
