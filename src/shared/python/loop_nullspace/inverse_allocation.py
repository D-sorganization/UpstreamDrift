"""Inverse optimal control of the NSX-23 allocation weights (NSX-71, #12589).

Given observed torques, find the objective weights ``a_j`` under which the
observation is the optimum of :func:`~.allocation.allocate_trajectory`, say
when they are not identifiable, and replay the allocator (never a second
solver) to score weights on other data.

Method: KKT-residual (stationarity) estimator, chosen over a bilevel fit.  The
objective is linear in its weights, ``J = sum_j a_j J_j(tau)``, and the inner
problem is convex with weight-independent constraints.  With the observed
torque ``tau*`` in the interior of the feasible torque set, optimality means
the reduced gradient vanishes, ``sum_j a_j N_k^T grad_tau J_j(tau*) = 0`` in
every frame ``k``: a *linear* system ``G a = 0``.  Minimising ``|G a|`` over a
normalisation of ``a`` is a convex QP with a global optimum and exact
recovery in the noise-free case, where a bilevel fit is non-convex and needs
an allocator solve per evaluation.  Its cost is that it is an equation-error
estimator (biased under noise, blind to constraint multipliers), so frames
where a constraint is binding are dropped and counted, and the fitted weights
are always scored by replaying the allocator.

Identifiability.  The objective is invariant under ``a -> c a`` (``c > 0``);
the fit fixes the scale by ``sum u = 1`` after scaling each column of ``G`` to
unit norm.  Anything beyond that gauge is reported: a gradient column that is
numerically zero (``unidentified_features``, weight ``NaN``), and directions
along which the fit criterion is flat on the normalised simplex
(``null_directions``, in weight coordinates), e.g. a zero reference makes the
effort and reference gradients identical so only their sum is identified.

Non-claims: weights explain the supplied torques; fitted to inverse-dynamics
or simulation torques they say nothing about physiological effort or hand
loads.  The observation is projected onto each frame's torque family; the
projection residual is reported, not hidden.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg as sla

from src.shared.python.loop_nullspace.allocation import (
    AllocationResult,
    GrossPowerSpec,
    ObjectiveSpec,
    allocate_trajectory,
)
from src.shared.python.loop_nullspace.evidence import Array, readonly_float
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.solvers import (
    ConicSettings,
    QuadraticObjective,
    SolveStatus,
    SOCBlock,
    solve_socp,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InverseFit",
    "InverseProblem",
    "InverseSettings",
    "Stationarity",
    "block_bootstrap_log_ratio",
    "fit_allocation_weights",
    "fit_from_blocks",
    "relative_torque_error",
    "replay_allocation",
    "stationarity",
    "term_gradients",
]

FEATURES = ("effort", "rate", "reference", "gross_power")
_QP_TOL = 1e-12
_RANK_RTOL = 1e-10


@dataclass(frozen=True)
class InverseSettings:
    """Numerical controls of the fit (defaults are the NSX-71 pre-registration).

    ``interior_margin``: a frame is *interior* when some torque-preserving
    choice of the gauge coordinates leaves every constraint slack above this
    (normalised rows, cone slack ``t - |u|``).  ``sv_rtol``: a direction is
    unidentified when its curvature on the normalised simplex is below
    ``sv_rtol**2`` of the largest.  ``column_rtol``: a gradient column with
    norm below ``column_rtol`` of the largest is treated as zero.
    """

    interior_margin: float = 1e-6
    sv_rtol: float = 1e-2
    column_rtol: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("interior_margin", "sv_rtol", "column_rtol"):
            val = getattr(self, name)
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {val!r}")


@dataclass(frozen=True, eq=False)
class InverseProblem:
    """Observed torques on a window of feasible sets.

    ``observed_torque`` is ``(frames, n_a)``; ``reference`` (same shape) enables
    the reference feature; ``gross_power`` enables the gross-power feature
    (its ``weight`` is ignored, the fit supplies the coefficient).
    """

    feasible_sets: tuple[FeasibleSet, ...]
    times_s: Array
    torque_scale: Array
    observed_torque: Array
    reference: Array | None = None
    gross_power: GrossPowerSpec | None = None

    def __post_init__(self) -> None:
        frames = tuple(self.feasible_sets)
        if not frames:
            raise ValueError("feasible_sets must contain at least one frame")
        for k, fs in enumerate(frames):
            if not isinstance(fs, FeasibleSet) or fs.decomposition is None:
                raise ValueError(
                    f"frame {k} must be a FeasibleSet with a decomposition"
                )
        times = readonly_float(self.times_s, "times_s")
        scale = readonly_float(self.torque_scale, "torque_scale")
        obs = readonly_float(self.observed_torque, "observed_torque")
        if times.shape != (len(frames),):
            raise ValueError(f"times_s needs {len(frames)} entries, got {times.shape}")
        if len(frames) > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be strictly increasing")
        if scale.ndim != 1 or scale.size == 0 or np.any(scale <= 0.0):
            raise ValueError("torque_scale must be 1-D with positive entries")
        if obs.shape != (len(frames), scale.size):
            raise ValueError(
                f"observed_torque must have shape {(len(frames), scale.size)}, "
                f"got {obs.shape}"
            )
        ref = None
        if self.reference is not None:
            ref = readonly_float(self.reference, "reference")
            if ref.shape != obs.shape:
                raise ValueError(
                    f"reference must have shape {obs.shape}, got {ref.shape}"
                )
        gp = self.gross_power
        if gp is not None and (
            not isinstance(gp, GrossPowerSpec) or gp.joint_velocity.shape != obs.shape
        ):
            raise ValueError(
                f"gross_power must be a GrossPowerSpec with velocity {obs.shape}"
            )
        for fs in frames:
            if fs.decomposition.n_actuated != scale.size:
                raise ValueError(
                    "every frame must have len(torque_scale) actuated torques"
                )
        for name, val in (
            ("feasible_sets", frames),
            ("times_s", times),
            ("torque_scale", scale),
            ("observed_torque", obs),
            ("reference", ref),
        ):
            object.__setattr__(self, name, val)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Enabled features, in the fixed order of :data:`FEATURES`."""
        on = {
            "effort": True,
            "rate": len(self.feasible_sets) > 1,
            "reference": self.reference is not None,
            "gross_power": self.gross_power is not None,
        }
        return tuple(n for n in FEATURES if on[n])


@dataclass(frozen=True, eq=False)
class InverseFit:
    """Result of the stationarity fit.

    ``weights`` are in objective units (arbitrary positive common scale;
    ``NaN`` for unidentified gradient columns); ``simplex_weights`` are the
    normalised ``u`` (sum 1 over the informative columns).  ``kkt_residual`` is
    ``|G~ u|`` with unit-norm columns, in ``[0, 1]``.  ``null_directions``
    (``features x m``, unit columns, weight coordinates) are the directions the
    data cannot fix.  ``identifiable`` is False if any is present or any
    feature is unidentified.
    """

    feature_names: tuple[str, ...]
    weights: Array
    simplex_weights: Array
    kkt_residual: float
    singular_values: Array
    second_smallest_sv_ratio: float
    tangent_curvature_ratio: float
    identifiable: bool
    unidentified_features: tuple[str, ...]
    null_directions: Array
    n_rows: int
    used_frames: tuple[int, ...] = ()
    dropped_binding: tuple[int, ...] = ()
    dropped_infeasible: tuple[int, ...] = ()
    dropped_nonsmooth: tuple[int, ...] = ()
    family_residual: Array = field(default_factory=lambda: np.zeros(0))

    def weight_dict(self) -> dict[str, float]:
        """Feature name to fitted weight (``NaN`` where unidentified)."""
        return {
            n: float(w) for n, w in zip(self.feature_names, self.weights, strict=True)
        }

    def ratio(self, numerator: str, denominator: str) -> float:
        """Weight ratio ``a_numerator / a_denominator`` (scale-free)."""
        w = self.weight_dict()
        for name in (numerator, denominator):
            if name not in w:
                raise ValueError(f"unknown feature {name!r}; have {self.feature_names}")
        return w[numerator] / w[denominator]


@dataclass(frozen=True, eq=False)
class Stationarity:
    """Per-frame stationarity rows ``blocks[k] = N_k^T [grad_j]`` and diagnostics."""

    feature_names: tuple[str, ...]
    blocks: tuple[Array, ...]
    used_frames: tuple[int, ...]
    dropped_binding: tuple[int, ...]
    dropped_infeasible: tuple[int, ...]
    dropped_nonsmooth: tuple[int, ...]
    family_residual: Array
    projected_torque: Array
    scales: Array


# --------------------------------------------------------------- gradients
def _trapezoid(times: Array) -> Array:
    if times.size == 1:
        return np.ones(1)
    half = np.diff(times) / 2.0
    out = np.zeros(times.size)
    out[:-1] += half
    out[1:] += half
    return out


def _gross_gradient(problem: InverseProblem, tau: Array, omega: Array) -> Array:
    gp = problem.gross_power
    p = tau * gp.joint_velocity
    slope = np.where(
        p > 0.0,
        gp.c_plus,
        np.where(p < 0.0, -gp.c_minus, 0.5 * (gp.c_plus - gp.c_minus)),
    )
    return omega[:, None] * slope * gp.joint_velocity


def term_gradients(problem: InverseProblem, tau: Any) -> dict[str, Array]:
    """Gradient of each unit-weight objective term w.r.t. ``tau`` (frames x n_a).

    Terms are those of :class:`~.allocation_spec.ObjectiveSpec`.  The
    gross-power gradient at ``p = 0`` is the sub-gradient midpoint.
    """
    tau = readonly_float(tau, "tau")
    if tau.shape != problem.observed_torque.shape:
        raise ValueError(f"tau must have shape {problem.observed_torque.shape}")
    s2 = problem.torque_scale**2
    omega = _trapezoid(problem.times_s)
    grads = {"effort": 2.0 * omega[:, None] * tau / s2}
    if "rate" in problem.feature_names:
        slope = np.diff(tau, axis=0) / np.diff(problem.times_s)[:, None]
        rate = np.zeros_like(tau)
        rate[1:] += slope
        rate[:-1] -= slope
        grads["rate"] = 2.0 * rate / s2
    if problem.reference is not None:
        grads["reference"] = 2.0 * omega[:, None] * (tau - problem.reference) / s2
    if problem.gross_power is not None:
        grads["gross_power"] = _gross_gradient(problem, tau, omega)
    return grads


# ------------------------------------------------------ frame classification
def _gauge_basis(n_mat: Array) -> Array:
    _u, sv, vt = np.linalg.svd(n_mat, full_matrices=True)
    rank = int(np.sum(sv > _RANK_RTOL * (sv[0] if sv.size else 1.0)))
    return vt[rank:].T


def _row_system(fs: FeasibleSet, z0: Array, gauge: Array):
    rows = [r for r in fs.rows if r.kind != "cone_linear" and np.linalg.norm(r.g) > 0.0]
    if not rows:
        return None, None
    g = np.array([r.g for r in rows])
    norm = np.linalg.norm(g, axis=1)
    a = np.hstack([(g @ gauge) / norm[:, None], np.ones((len(rows), 1))])
    return a, (np.array([r.h for r in rows]) - g @ z0) / norm


def _max_margin(fs: FeasibleSet, z0: Array, gauge: Array) -> float | None:
    """Largest uniform slack over torque-preserving ``z = z0 + gauge y``.

    ``None`` when the solve is inconclusive.  ``+inf`` when nothing constrains.
    """
    a_ub, b_ub = _row_system(fs, z0, gauge)
    n_y = gauge.shape[1]
    cones = []
    for c in fs.cones:
        a = np.zeros((c.A.shape[0], n_y + 1))
        a[:, :n_y] = c.A @ gauge
        a[0, n_y] = 1.0
        cones.append(SOCBlock(a, c.b - c.A @ z0))
    if a_ub is None and not cones:
        return float("inf")
    cap = np.zeros((1, n_y + 1))
    cap[0, n_y] = 1.0
    a_all = cap if a_ub is None else np.vstack([a_ub, cap])
    b_all = np.array([1.0]) if b_ub is None else np.append(b_ub, 1.0)
    cost = np.zeros(n_y + 1)
    cost[n_y] = -1.0
    res = solve_socp(
        QuadraticObjective(cost, np.zeros((n_y + 1, n_y + 1))),
        None,
        None,
        a_all,
        b_all,
        cones,
        settings=ConicSettings(tol=_QP_TOL),
    )
    if res.status is not SolveStatus.OPTIMAL or res.x is None:
        return None
    return float(res.x[n_y])


def _classify(fs: FeasibleSet, z0: Array, gauge: Array, tol: float) -> str:
    margin = _max_margin(fs, z0, gauge)
    if margin is None or abs(margin) <= tol:
        return "binding"
    return "interior" if margin > 0.0 else "infeasible"


# -------------------------------------------------------------- stationarity
def _project(problem: InverseProblem) -> tuple[Array, Array, list[tuple[Array, Array]]]:
    """Observation projected onto each frame's torque family and its residual."""
    s = problem.torque_scale
    proj = np.zeros_like(problem.observed_torque)
    resid = np.zeros(len(problem.feasible_sets))
    geom = []
    for k, fs in enumerate(problem.feasible_sets):
        dec = fs.decomposition
        n_a, t0 = s.size, dec.x_p[: s.size]
        n_mat = dec.N[:n_a]
        z0 = np.linalg.lstsq(n_mat, problem.observed_torque[k] - t0, rcond=None)[0]
        proj[k] = t0 + n_mat @ z0
        den = np.linalg.norm(problem.observed_torque[k] / s)
        resid[k] = np.linalg.norm((problem.observed_torque[k] - proj[k]) / s) / max(
            den, 1e-300
        )
        geom.append((z0, _gauge_basis(n_mat)))
    return proj, resid, geom


def stationarity(
    problem: InverseProblem, settings: InverseSettings | None = None
) -> Stationarity:
    """Stationarity rows of the observation on the interior, smooth frames.

    Postconditions: ``used_frames``, ``dropped_binding``, ``dropped_infeasible``
    and ``dropped_nonsmooth`` partition the frame indices; ``blocks`` has one
    ``(n_z_k, features)`` array per used frame.
    """
    if not isinstance(problem, InverseProblem):
        raise TypeError(f"problem must be InverseProblem, got {type(problem)!r}")
    cfg = settings or InverseSettings()
    proj, resid, geom = _project(problem)
    names = problem.feature_names
    grads = term_gradients(problem, proj)
    nonsmooth = _nonsmooth_frames(problem, proj)
    used, binding, infeasible, smooth = [], [], [], []
    blocks = []
    scales = np.zeros(len(names))
    for k, fs in enumerate(problem.feasible_sets):
        kind = _classify(fs, *geom[k], cfg.interior_margin)
        if kind == "infeasible":
            infeasible.append(k)
        elif kind == "binding":
            binding.append(k)
        elif k in nonsmooth:
            smooth.append(k)
        else:
            used.append(k)
            n_mat = fs.decomposition.N[: problem.torque_scale.size]
            blocks.append(np.column_stack([n_mat.T @ grads[n][k] for n in names]))
            full = np.array([np.linalg.norm(grads[n][k]) for n in names])
            scales += np.linalg.norm(n_mat, 2) * full
    return Stationarity(
        names,
        tuple(blocks),
        tuple(used),
        tuple(binding),
        tuple(infeasible),
        tuple(smooth),
        resid,
        proj,
        scales,
    )


def _nonsmooth_frames(problem: InverseProblem, tau: Array) -> set[int]:
    gp = problem.gross_power
    if gp is None:
        return set()
    p = np.abs(tau * gp.joint_velocity)
    kink = p <= 1e-9 * max(float(np.max(p)), 1e-300)
    return {int(k) for k in np.flatnonzero(np.any(kink, axis=1))}


# ----------------------------------------------------------------------- fit
def _simplex_qp(gram: Array) -> Array:
    n = gram.shape[0]
    res = solve_socp(
        QuadraticObjective(np.zeros(n), 2.0 * gram),
        np.ones((1, n)),
        np.ones(1),
        -np.eye(n),
        np.zeros(n),
        (),
        settings=ConicSettings(tol=_QP_TOL),
    )
    if res.status is not SolveStatus.OPTIMAL or res.x is None:
        raise RuntimeError(f"simplex QP failed: {res.status.name} ({res.reason})")
    u = np.maximum(np.asarray(res.x), 0.0)
    return u / u.sum()


def _flat_directions(gram: Array, cfg: InverseSettings) -> tuple[Array, float]:
    """Unit directions with ``sum d = 0`` along which the criterion is flat."""
    n = gram.shape[0]
    if n < 2:
        return np.zeros((n, 0)), 1.0
    tangent = sla.null_space(np.ones((1, n)))
    lam, vec = np.linalg.eigh(tangent.T @ gram @ tangent)
    top = max(float(np.max(np.linalg.eigvalsh(gram))), 1e-300)
    ratio = float(max(lam[0], 0.0) / top)
    flat = lam <= cfg.sv_rtol**2 * top
    return tangent @ vec[:, flat], ratio


def fit_from_blocks(
    blocks: Sequence[Any],
    names: Sequence[str],
    settings: InverseSettings | None = None,
    *,
    scales: Any = None,
) -> InverseFit:
    """Fit weights from stacked stationarity blocks ``(rows_k, features)``.

    Preconditions: at least one block, all with ``len(names)`` columns
    (``ValueError``); ``scales`` (one per column) is the unprojected gradient
    scale used to detect zero columns.  Postconditions: see :class:`InverseFit`; the scale
    gauge is fixed by ``sum u = 1`` and is the only silent normalisation.
    """
    cfg = settings or InverseSettings()
    names = tuple(names)
    mats = [np.atleast_2d(np.asarray(b, dtype=float)) for b in blocks]
    if not mats or any(m.shape[1] != len(names) for m in mats):
        raise ValueError(f"need >= 1 block with {len(names)} columns each")
    g = np.vstack(mats)
    if not np.all(np.isfinite(g)):
        raise ValueError("stationarity rows must be finite")
    norms = np.linalg.norm(g, axis=0)
    ref = norms if scales is None else np.asarray(scales, dtype=float)
    if ref.shape != norms.shape:
        raise ValueError(f"scales must have {len(names)} entries")
    live = norms > cfg.column_rtol * np.maximum(
        ref if scales is not None else float(norms.max()), 1e-300
    )
    dead = tuple(n for n, ok in zip(names, live, strict=True) if not ok)
    idx = np.flatnonzero(live)
    full = len(names)
    weights = np.full(full, np.nan)
    simplex = np.zeros(full)
    if idx.size == 0:
        return _empty_fit(names, dead, g.shape[0], full)
    scaled = g[:, idx] / norms[idx]
    gram = scaled.T @ scaled
    u = _simplex_qp(gram) if idx.size > 1 else np.ones(1)
    simplex[idx] = u
    weights[idx] = u / norms[idx]
    flat, tcr = _flat_directions(gram, cfg)
    sv = np.sqrt(np.maximum(np.linalg.eigvalsh(gram)[::-1], 0.0))
    second = float(sv[-2] / sv[0]) if sv.size > 1 and sv[0] > 0 else 1.0
    directions = _to_weight_space(flat, norms[idx], idx, full, dead, names)
    return InverseFit(
        names,
        weights,
        simplex,
        float(np.linalg.norm(scaled @ u)),
        sv,
        second,
        tcr,
        directions.shape[1] == 0 and not dead,
        dead,
        directions,
        g.shape[0],
    )


def _to_weight_space(flat, norms, idx, full, dead, names) -> Array:
    cols = []
    for d in flat.T:
        vec = np.zeros(full)
        vec[idx] = d / norms
        cols.append(vec / np.linalg.norm(vec))
    for name in dead:
        unit = np.zeros(full)
        unit[names.index(name)] = 1.0
        cols.append(unit)
    return np.column_stack(cols) if cols else np.zeros((full, 0))


def _empty_fit(names, dead, n_rows, full) -> InverseFit:
    return InverseFit(
        names,
        np.full(full, np.nan),
        np.zeros(full),
        1.0,
        np.zeros(0),
        0.0,
        0.0,
        False,
        dead,
        np.eye(full),
        n_rows,
    )


def fit_allocation_weights(
    problem: InverseProblem,
    settings: InverseSettings | None = None,
    *,
    precomputed: Stationarity | None = None,
) -> InverseFit:
    """Fit the allocation weights to the observed torques of ``problem``.

    Raises ``TypeError`` for a non-problem and ``ValueError`` when no frame is
    usable (all binding, infeasible or non-smooth).
    """
    if not isinstance(problem, InverseProblem):
        raise TypeError(f"problem must be InverseProblem, got {type(problem)!r}")
    stat = precomputed or stationarity(problem, settings)
    if not stat.blocks:
        raise ValueError(
            "no usable frame: binding "
            f"{stat.dropped_binding}, infeasible {stat.dropped_infeasible}, "
            f"non-smooth {stat.dropped_nonsmooth}"
        )
    fit = fit_from_blocks(stat.blocks, stat.feature_names, settings, scales=stat.scales)
    logger.info(
        "NSX-71 fit: %d/%d frames used, rho_KKT=%.3g, identifiable=%s",
        len(stat.used_frames),
        len(stat.family_residual),
        fit.kkt_residual,
        fit.identifiable,
    )
    return dataclasses.replace(
        fit,
        used_frames=stat.used_frames,
        dropped_binding=stat.dropped_binding,
        dropped_infeasible=stat.dropped_infeasible,
        dropped_nonsmooth=stat.dropped_nonsmooth,
        family_residual=stat.family_residual,
    )


# ----------------------------------------------------------------- bootstrap
def block_bootstrap_log_ratio(
    source: InverseProblem | Stationarity,
    numerator: str,
    denominator: str,
    *,
    resamples: int = 200,
    block: int = 8,
    seed: int = 12589,
    settings: InverseSettings | None = None,
) -> tuple[float, float]:
    """5th and 95th percentile of ``log10(a_num / a_den)`` over block resamples.

    Moving blocks of ``block`` consecutive used frames are drawn with
    replacement (seeded ``numpy.random.default_rng``) until the original number
    of frames is reached; resamples with an unidentified numerator or
    denominator are skipped.  ``ValueError`` if fewer than half succeed.
    """
    stat = (
        source if isinstance(source, Stationarity) else stationarity(source, settings)
    )
    names = stat.feature_names
    if resamples < 2 or block < 1:
        raise ValueError("resamples must be >= 2 and block >= 1")
    if numerator not in names or denominator not in names:
        raise ValueError(f"features must be in {names}")
    n = len(stat.blocks)
    if n < 1:
        raise ValueError("no usable frames to resample")
    rng = np.random.default_rng(seed)
    starts = max(n - block + 1, 1)
    values = []
    for _ in range(resamples):
        picks: list[int] = []
        while len(picks) < n:
            first = int(rng.integers(0, starts))
            picks.extend(range(first, min(first + block, n)))
        fit = fit_from_blocks(
            [stat.blocks[i] for i in picks[:n]], names, settings, scales=stat.scales
        )
        w = fit.weight_dict()
        if (
            np.isfinite(w[numerator])
            and np.isfinite(w[denominator])
            and w[denominator] > 0
        ):
            if w[numerator] > 0:
                values.append(np.log10(w[numerator] / w[denominator]))
    if len(values) < resamples // 2:
        raise ValueError(f"only {len(values)} of {resamples} resamples were usable")
    lo, hi = np.percentile(values, [5.0, 95.0])
    return float(lo), float(hi)


# -------------------------------------------------------------------- replay
def replay_allocation(
    problem: InverseProblem, weights: Mapping[str, float]
) -> AllocationResult:
    """Run :func:`allocate_trajectory` on the problem's sets with ``weights``.

    ``weights`` maps feature names to non-negative values (missing means 0).
    A reference weight above 0 needs ``problem.reference`` and a gross-power
    weight above 0 needs ``problem.gross_power`` (``ValueError`` otherwise).
    Postcondition: the allocator's result, unchanged.
    """
    unknown = set(weights) - set(FEATURES)
    if unknown:
        raise ValueError(f"unknown weights {sorted(unknown)}; features are {FEATURES}")
    w = {n: float(weights.get(n, 0.0)) for n in FEATURES}
    if w["reference"] > 0.0 and problem.reference is None:
        raise ValueError("reference weight > 0 but the problem has no reference")
    if w["gross_power"] > 0.0 and problem.gross_power is None:
        raise ValueError("gross_power weight > 0 but the problem has no gross_power")
    gross = (
        dataclasses.replace(problem.gross_power, weight=w["gross_power"])
        if w["gross_power"] > 0.0
        else None
    )
    spec = ObjectiveSpec(
        times_s=problem.times_s,
        torque_scale=problem.torque_scale,
        effort_weight=w["effort"],
        rate_weight=w["rate"],
        reference_weight=w["reference"],
        gross_power=gross,
    )
    ref = problem.reference if w["reference"] > 0.0 else None
    return allocate_trajectory(problem.feasible_sets, spec, reference=ref)


def relative_torque_error(
    problem: InverseProblem, tau_alloc: Any, tau_obs: Any
) -> float:
    """``|S^-1 (tau_alloc - tau_obs)|_F / |S^-1 tau_obs|_F`` (capacity-scaled)."""
    a = readonly_float(tau_alloc, "tau_alloc")
    b = readonly_float(tau_obs, "tau_obs")
    if a.shape != b.shape or a.shape[-1] != problem.torque_scale.size:
        raise ValueError(
            f"torque arrays must share shape (frames, {problem.torque_scale.size})"
        )
    den = np.linalg.norm(b / problem.torque_scale)
    if den == 0.0:
        raise ValueError("observed torque is identically zero")
    return float(np.linalg.norm((a - b) / problem.torque_scale) / den)
