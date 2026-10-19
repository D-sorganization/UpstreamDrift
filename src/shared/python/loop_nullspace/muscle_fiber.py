"""The activation fibre at fixed net torque (NSX-94, #12723).

The set of activations that produce one muscle torque,

    F(tau) = { a in [0, 1]^m : G a = tau - R^T f_p },       G = R^T diag(F_a),

is the *activation fibre*.  The equality ``G a = b`` alone is an affine space
parallel to ``ker G`` (dimension ``m - rank G``, found with the repository's
:class:`ForceNullSpace`, never re-implemented here); the activation box turns it
into a bounded polytope.  The dimension of the polytope itself is the dimension
of its *affine hull*, and that equals ``m - rank G`` only when a strictly
box-feasible point exists (a sufficient condition, not a necessary one).  When
the box forces a coordinate to a constant (an *implicit equality*, e.g.
``G = [1 1 0]``, ``b = 2``, ``a in [0, 1]^3`` forces ``a1 = a2 = 1``) the hull
is smaller; the hull is therefore *computed* (:attr:`ActivationFiber.hull`) from
implicit-equality detection, never assumed from the rank.  Moving along the
fibre changes no joint torque at all and changes only which muscles pull against
which -- co-contraction -- so a torque envelope, however tight, quotients it away
(review finding A4 of ``muscle_level_view.md``).

Co-contraction is never inferred.  The experiment *declares* a
:class:`CocontractionIndex` ``c(a) = w^T a`` (default: total activation) and
sweeps its level; for a level inside ``[c_min, c_max]`` the representative is the
unique minimum-norm point

    a*(c) = argmin 1/2 ||a||^2   s.t.  G a = b,  w^T a = c,  0 <= a <= 1,

the per-level counterpart of the static-optimisation criterion
``min sum a^2`` used by ``myofullbody.redundancy`` and
``musculoskeletal_static_opt``.  It is a declared selection rule, not physiology:
inside one level the remaining sub-fibre is unexplored, and
:meth:`ActivationFiber.sample` shows it.  Its dimension is
``m - rank G - 1`` only if, additionally, ``w`` is not in ``range(G^T)`` and the
level lies strictly inside ``(c_min, c_max)`` and the fibre is full-dimensional;
:meth:`ActivationFiber.level_subfibre_dimension` computes it from the hull of the
sub-fibre instead of assuming it (``w = e3`` on the segment above gives a point).

Numerics.

* ``a*(c)`` is the projection of the origin onto a polytope.  The dual is
  ``a(mu) = clip(-E^T mu, 0, 1)`` with ``E = [G; w^T]`` and the piecewise
  linear monotone equation ``E a(mu) = e`` is solved by a semismooth Newton
  iteration with backtracking; once the active set is identified the step is
  exact, so the equality residual is at machine precision (``~1e-15`` scaled),
  far inside the ``1e-12`` the issue demands.
* The level range ``[c_min, c_max]`` is a pair of linear programs solved with
  ``scipy.optimize.linprog`` (HiGHS) at tightened tolerances.  Each fibre is a
  one-shot problem, so the persistent ``highspy`` sessions of
  :mod:`solvers` (epic rule R11) would add nothing; the backend is recorded in
  :data:`LP_BACKEND`.  ``LEVEL_INSET`` keeps swept levels off the LP endpoints.
* The affine hull is found by implicit-equality detection: a coordinate whose
  LP range over the fibre is at most :data:`HULL_TOL` is fixed, and the free
  coordinates get a :class:`ForceNullSpace` basis of the reduced system.  In
  those hull coordinates the fibre has a non-empty interior, so
  :meth:`ActivationFiber.sample` runs the package's :func:`hit_and_run` there
  from the Chebyshev centre; only a hull of dimension zero is a singleton, and
  the result says so (:class:`FibreSample`).  Samples are *coverage of the fibre*, not a probability
  statement about a golfer (epic rule R14: sampled coverage, not certified
  bounds); mixing is not certified.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog

from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.feasible_rows import FeasibleRow
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.hit_and_run import TrajectoryDomain, hit_and_run
from src.shared.python.loop_nullspace.muscle_frame import (
    RESERVE_FRACTION_LIMIT,
    FrameQualification,
    MuscleFrame,
    qualify_frame,
    require_qualified,
)
from src.shared.python.motion_matching.force_nullspace.space import ForceNullSpace

__all__ = [
    "HULL_TOL",
    "LEVEL_INSET",
    "LP_BACKEND",
    "ActivationFiber",
    "CocontractionIndex",
    "FiberInfeasibleError",
    "FibreHull",
    "FibreSample",
    "activation_fiber",
    "muscle_subset_index",
    "total_activation",
]

LP_BACKEND = "scipy.optimize.linprog[highs]"
"""Backend of the level-range and Chebyshev LPs, recorded in result provenance."""

_LP_TOL = 1e-10
"""HiGHS primal/dual feasibility tolerance (default ``1e-7``) tightened by three
orders of magnitude so a swept level sits inside the true range."""

LEVEL_INSET = 1e-6
"""Fraction of the level range kept clear of each LP endpoint.  HiGHS reports the
endpoints to ``~10 * _LP_TOL``; ``1e-6`` of the range is four orders larger."""

_PROJ_TOL = 1e-13
"""Scaled equality residual accepted by the dual projection.  The iteration is
finite and exact, so ``1e-13`` is ~450 ``eps`` of headroom, not a tuned slack."""

HULL_TOL = 1e-8
"""Largest LP range (activation units) of a coordinate still treated as *fixed*,
i.e. an implicit equality.  HiGHS, tightened to ``_LP_TOL = 1e-10``, reports an
optimum to ``~10 * _LP_TOL = 1e-9``; ``HULL_TOL`` is a further factor ten, so a
range at the solver's own resolution is never mistaken for a free direction and a
genuinely free direction (any fibre wider than ``1e-8`` of the full activation
range) is never discarded.  A fibre thinner than this is reported at the lower
dimension; that is the declared resolution, not a tuned slack."""


class FiberInfeasibleError(ValueError):
    """The requested fibre point or level does not exist (fail closed)."""


_BOX_HYPOTHESIS = "H-ACTIVATION-BOX"
"""Hypothesis id of the ``0 <= a <= 1`` rows handed to :func:`hit_and_run`."""


def _solve_lp(cost: Array, what: str, **kwargs: Any) -> tuple[Array, float]:
    """HiGHS at the module's tightened tolerances; fail closed on non-optimum."""
    res = linprog(
        cost,
        method="highs",
        options={
            "primal_feasibility_tolerance": _LP_TOL,
            "dual_feasibility_tolerance": _LP_TOL,
        },
        **kwargs,
    )
    if res.status != 0 or res.x is None or res.fun is None:
        raise FiberInfeasibleError(f"{what} failed: {res.message}")
    return np.asarray(res.x), float(res.fun)


@dataclass(frozen=True, eq=False)
class FibreHull:
    """The affine hull of a fibre: ``{a : a_fixed = values, a = p + B z}``.

    Attributes:
        free: Indices of the coordinates that vary over the fibre.
        particular: ``p``, ``(m,)``; equals the constant on every fixed index.
        basis: ``B``, ``(m, d)``; its rows on fixed indices are zero.
        dimension: ``d``, the dimension of the fibre (of its affine hull).

    The fibre has a non-empty interior in the coordinates ``z`` (relative
    interior of a polytope), which is what makes the Chebyshev centre and
    hit-and-run well posed there.
    """

    free: NDArray[np.intp]
    particular: Array
    basis: Array

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[1])


def _affine_hull(a_eq: Array, b_eq: Array, rank_rtol: float) -> FibreHull:
    """True affine hull of ``{a in [0,1]^m : a_eq a = b_eq}``.

    A coordinate is an implicit equality iff its LP range over the set is at most
    :data:`HULL_TOL` (two LPs each); the hull is then ``{a_eq a = b_eq, a_fixed =
    const}`` and its direction space is the :class:`ForceNullSpace` basis of the
    free columns.  Preconditions: the set is non-empty (else
    :class:`FiberInfeasibleError`).  Postconditions: ``dimension <= m - rank``,
    with equality when a strictly box-feasible point exists.
    """
    m = a_eq.shape[1]
    lo, hi = np.empty(m), np.empty(m)
    for j in range(m):
        unit = np.zeros(m)
        unit[j] = 1.0
        kw = {"A_eq": a_eq, "b_eq": b_eq, "bounds": (0.0, 1.0)}
        lo[j] = _solve_lp(unit, "hull LP", **kw)[1]
        hi[j] = -_solve_lp(-unit, "hull LP", **kw)[1]
    fixed = (hi - lo) <= HULL_TOL
    mid = 0.5 * (lo + hi)
    for bound in (0.0, 1.0):
        mid[np.abs(mid - bound) <= HULL_TOL] = bound
    particular = np.where(fixed, mid, 0.0)
    free = np.flatnonzero(~fixed)
    basis = np.zeros((m, 0))
    if free.size:
        rhs = b_eq - a_eq[:, fixed] @ mid[fixed]
        fns = ForceNullSpace.from_balance(
            a_eq[:, free],
            rhs,
            rank_rtol=rank_rtol,
            balance_tolerance=max(1, m) * HULL_TOL,
        )
        particular[free] = np.asarray(fns.particular)
        null = np.asarray(fns.basis)
        basis = np.zeros((m, null.shape[1]))
        basis[free, :] = null
    for arr in (free, particular, basis):
        arr.flags.writeable = False
    return FibreHull(free=free, particular=particular, basis=basis)


@dataclass(frozen=True, eq=False)
class FibreSample:
    """Fibre samples with the dimension they were drawn in.

    ``samples`` is ``(n, m)``.  ``is_singleton`` is ``True`` exactly when
    ``hull_dimension == 0``: then every row is the fibre's single point, which is
    the honest answer, not a sampler failure.
    """

    samples: Array
    hull_dimension: int

    @property
    def is_singleton(self) -> bool:
        return self.hull_dimension == 0


@dataclass(frozen=True)
class CocontractionIndex:
    """A declared linear co-contraction functional ``c(a) = w^T a``.

    Preconditions (``ValueError``): finite non-zero 1-D ``weights``.
    """

    name: str
    weights: Array

    def __post_init__(self) -> None:
        w = np.array(self.weights, dtype=np.float64, copy=True)
        if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)) or not np.any(w):
            raise ValueError("weights must be a finite, non-zero 1-D vector")
        if not self.name:
            raise ValueError("name must be non-empty")
        w.flags.writeable = False
        object.__setattr__(self, "weights", w)

    def __call__(self, activation: ArrayLike) -> float:
        a = np.asarray(activation, dtype=np.float64)
        if a.shape != self.weights.shape:
            raise ValueError(
                f"activation must have shape {self.weights.shape} to match weights"
            )
        return float(self.weights @ a)


def total_activation(n_muscles: int) -> CocontractionIndex:
    """``c(a) = sum_k a_k``: the default declared co-contraction level."""
    return CocontractionIndex("total_activation", np.ones(n_muscles))


def muscle_subset_index(
    frame: MuscleFrame, names: Sequence[str], *, name: str | None = None
) -> CocontractionIndex:
    """Total activation of the named muscles, e.g. the antagonists of a movement.

    Preconditions (``ValueError``): ``names`` non-empty and all in ``frame``.
    """
    missing = [n for n in names if n not in frame.names]
    if not names or missing:
        raise ValueError(
            f"names must be a non-empty subset of the frame; bad: {missing}"
        )
    w = np.array([1.0 if n in names else 0.0 for n in frame.names])
    return CocontractionIndex(name or "activation:" + "+".join(names), w)


@dataclass(frozen=True)
class ActivationFiber:
    """The fibre ``{a in [0,1]^m : G a = b}`` with its null-space decomposition.

    ``nullity`` is ``dim ker G = m - rank G``, the dimension of the *equality*
    set before the box.  The dimension of the fibre itself is
    :attr:`hull_dimension`; it equals ``nullity`` when a strictly box-feasible
    point exists (sufficient, not necessary) and is smaller when the box forces
    some coordinates (implicit equalities), e.g. the segment
    ``G = [1 1 0]``, ``b = 2``, ``a in [0,1]^3`` has ``nullity`` 2 and hull
    dimension 1.

    Attributes:
        frame: The :class:`MuscleFrame` the fibre belongs to.
        requested_torque: The torque asked for, ``(n,)``.
        target_torque: The muscle-reachable torque used (equals the request
            when the reserve is zero), ``(n,)``.
        qualification: The passed :class:`FrameQualification`.
        particular, basis: ``a = particular + basis @ z`` solves ``G a = b``.
        rank, nullity: Of ``G`` (``ForceNullSpace``), before the box.
        rank_rtol: The SVD rank gap used for ``G`` and for the hull's reduced
            system.
    """

    frame: MuscleFrame
    requested_torque: Array
    target_torque: Array
    qualification: FrameQualification
    gain: Array
    offset: Array
    particular: Array
    basis: Array
    rank: int
    nullity: int
    row_scale: Array
    rank_rtol: float = 1e-12

    @property
    def reserve_fraction(self) -> float:
        """``||tau_requested - tau_target|| / ||tau_requested||``."""
        return self.qualification.reserve_fraction

    # ------------------------------------------------------------- queries
    def residual(self, activation: ArrayLike) -> float:
        """``max |G a - b|`` in N m."""
        a = np.asarray(activation, dtype=np.float64)
        return float(np.max(np.abs(self.gain @ a - self.offset)))

    def contains(self, activation: ArrayLike, tol: float = 1e-9) -> bool:
        """Whether ``a`` lies on the fibre to ``tol`` (N m and activation units)."""
        a = np.asarray(activation, dtype=np.float64)
        in_box = bool(np.all(a >= -tol) and np.all(a <= 1.0 + tol))
        return in_box and self.residual(a) <= tol * float(np.max(self.row_scale))

    def _check_index(self, index: CocontractionIndex) -> Array:
        if index.weights.shape != (self.frame.n_muscles,):
            raise ValueError(
                f"index weights must have shape ({self.frame.n_muscles},), "
                f"got {index.weights.shape}"
            )
        return index.weights

    def _equality(self) -> tuple[Array, Array]:
        return self.gain / self.row_scale[:, None], self.offset / self.row_scale

    @cached_property
    def hull(self) -> FibreHull:
        """The fibre's true affine hull (implicit equalities detected by LP)."""
        a_eq, b_eq = self._equality()
        return _affine_hull(a_eq, b_eq, self.rank_rtol)

    @property
    def hull_dimension(self) -> int:
        """Dimension of the fibre (of its affine hull); ``<= nullity``."""
        return self.hull.dimension

    def index_range(self, index: CocontractionIndex) -> tuple[float, float]:
        """Exact ``(min, max)`` of ``w^T a`` over the fibre (two LPs).

        Raises :class:`FiberInfeasibleError` if HiGHS does not report an optimum.
        """
        w = self._check_index(index)
        a_eq, b_eq = self._equality()
        out = []
        for sign in (1.0, -1.0):
            _, fun = _solve_lp(
                sign * w, "fibre LP", A_eq=a_eq, b_eq=b_eq, bounds=(0.0, 1.0)
            )
            out.append(sign * fun)
        return out[0], out[1]

    def _rows(self) -> tuple[Array, Array]:
        """Box rows ``N z <= c`` in the hull coordinates (free coordinates only)."""
        n_mat = self.hull.basis[self.hull.free]
        p = self.hull.particular[self.hull.free]
        return np.vstack([n_mat, -n_mat]), np.concatenate([1.0 - p, p])

    def chebyshev(self) -> tuple[Array, float]:
        """Chebyshev centre (hull coordinates ``z``) and radius of the fibre.

        The radius is the largest ball, in activation units inside the hull,
        that fits in the box.  It is positive whenever :attr:`hull_dimension` is
        positive (the hull carries the interior) and ``0.0`` with an empty
        centre when the hull is a single point.
        """
        k = self.hull_dimension
        if k == 0:
            return np.zeros(0), 0.0
        rows, bounds = self._rows()
        norms = np.linalg.norm(rows, axis=1)
        cost = np.zeros(k + 1)
        cost[-1] = -1.0
        x, _ = _solve_lp(
            cost,
            "Chebyshev LP",
            A_ub=np.hstack([rows, norms[:, None]]),
            b_ub=bounds,
            bounds=[(None, None)] * k + [(0.0, None)],
        )
        return x[:k], float(x[k])

    @property
    def is_degenerate(self) -> bool:
        """``True`` when the fibre is a single point (hull dimension zero)."""
        return self.hull_dimension == 0

    # ------------------------------------------------------------- sampling
    def sample_result(
        self,
        n: int,
        *,
        rng: np.random.Generator,
        thin: int = 5,
        burn_in: int = 100,
    ) -> FibreSample:
        """``n`` hit-and-run samples of the fibre, with the dimension they span.

        Runs the package's :func:`hit_and_run` in the hull coordinates from the
        Chebyshev centre (seeded from ``rng``), so a lower-dimensional fibre is
        explored along its own directions.  Only when :attr:`hull_dimension` is
        zero is the single point returned ``n`` times, and the result is then
        flagged :attr:`FibreSample.is_singleton`.

        Preconditions: ``rng`` a ``numpy.random.Generator`` (``TypeError``);
        ``n >= 1``, ``thin >= 1``, ``burn_in >= 0`` (``ValueError``).
        Postconditions: every sample lies in the box exactly and satisfies
        ``G a = b`` to the hull-basis accuracy.  The samples are coverage, not a
        probability claim.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng must be a numpy Generator, got {type(rng)!r}")
        if n < 1 or thin < 1 or burn_in < 0:
            raise ValueError("n and thin must be >= 1 and burn_in >= 0")
        hull = self.hull
        d = hull.dimension
        if d == 0:
            point = np.clip(hull.particular, 0.0, 1.0)
            return FibreSample(np.tile(point, (n, 1)), 0)
        z0, radius = self.chebyshev()
        if not radius > 0.0:
            raise FiberInfeasibleError("hull has no interior (Chebyshev radius 0)")
        rows, bounds = self._rows()
        box = FeasibleSet.from_rows(
            d,
            [
                FeasibleRow(
                    rows[i], float(bounds[i]), f"box{i}", _BOX_HYPOTHESIS, "torque"
                )
                for i in range(rows.shape[0])
            ],
        )
        chain = hit_and_run(
            TrajectoryDomain([box], np.ones((1, 1))),
            z0,
            n_draws=n,
            seed=int(rng.integers(0, 2**63 - 1)),
            thin=thin,
            burn_in=burn_in,
        )
        acts = hull.particular[None, :] + chain.coefficients @ hull.basis.T
        return FibreSample(np.clip(acts, 0.0, 1.0), d)

    def sample(
        self,
        n: int,
        *,
        rng: np.random.Generator,
        thin: int = 5,
        burn_in: int = 100,
    ) -> Array:
        """``(n, m)`` samples of the fibre; see :meth:`sample_result`.

        Same preconditions and postconditions.  A fibre of hull dimension zero
        returns its single point ``n`` times; use :meth:`sample_result` to be told.
        """
        return self.sample_result(n, rng=rng, thin=thin, burn_in=burn_in).samples

    def level_subfibre_dimension(self, index: CocontractionIndex, level: float) -> int:
        """Dimension of the level sub-fibre ``{a in fibre : w^T a = level}``.

        Computed from the sub-fibre's own affine hull.  It equals
        ``m - rank G - 1`` only when ``w`` is not in ``range(G^T)``, ``level`` is
        strictly inside ``(c_min, c_max)`` and the fibre is full-dimensional; on
        the segment ``G = [1 1 0]`` with ``w = e3`` an interior level gives 0.

        Preconditions: finite ``level`` (``ValueError``); ``level`` within the
        fibre's index range to :data:`HULL_TOL` (:class:`FiberInfeasibleError`).
        """
        w = self._check_index(index)
        if not np.isfinite(level):
            raise ValueError("level must be finite")
        lo, hi = self.index_range(index)
        if not (lo - HULL_TOL <= level <= hi + HULL_TOL):
            raise FiberInfeasibleError(
                f"level {level:.6g} of {index.name!r} is outside the fibre's range "
                f"[{lo:.6g}, {hi:.6g}]"
            )
        a_eq, b_eq = self._equality()
        scale = float(np.linalg.norm(w, np.inf))
        a_lvl = np.vstack([a_eq, (w / scale)[None]])
        b_lvl = np.concatenate([b_eq, [level / scale]])
        return _affine_hull(a_lvl, b_lvl, self.rank_rtol).dimension

    # ------------------------------------------------------- level selection
    def point_at_level(
        self, index: CocontractionIndex, level: float, *, max_iter: int = 200
    ) -> Array:
        """Minimum-norm fibre point with ``w^T a = level`` (see module doc).

        Preconditions (``ValueError``): finite ``level``.  Raises
        :class:`FiberInfeasibleError` when ``level`` lies outside the fibre's
        range (the dual iteration cannot reduce the residual).  Postconditions:
        ``|G a - b|`` and ``|w^T a - level|`` are at machine precision and
        ``0 <= a <= 1`` exactly.
        """
        w = self._check_index(index)
        if not np.isfinite(level):
            raise ValueError("level must be finite")
        w_scale = max(float(np.linalg.norm(w, np.inf)), 1.0)
        level_scale = max(abs(float(level)), 1.0)
        e_mat = np.vstack(
            [self.gain / self.row_scale[:, None], (w / level_scale)[None]]
        )
        e_vec = np.concatenate([self.offset / self.row_scale, [level / level_scale]])
        del w_scale
        mu = np.zeros(e_mat.shape[0])

        def primal(m_vec: Array) -> Array:
            return np.clip(-e_mat.T @ m_vec, 0.0, 1.0)

        phi = e_mat @ primal(mu) - e_vec
        for _ in range(max_iter):
            if np.max(np.abs(phi)) <= _PROJ_TOL:
                return primal(mu)
            raw = -e_mat.T @ mu
            free = ((raw >= 0.0) & (raw <= 1.0)).astype(float)
            jac = (e_mat * free[None, :]) @ e_mat.T
            step = np.linalg.lstsq(jac, phi, rcond=None)[0]
            scale, base = 1.0, float(np.linalg.norm(phi))
            for _ in range(40):
                cand = mu + scale * step
                phi_c = e_mat @ primal(cand) - e_vec
                if float(np.linalg.norm(phi_c)) < base * (1.0 - 1e-4 * scale):
                    mu, phi = cand, phi_c
                    break
                scale *= 0.5
            else:
                break
        raise FiberInfeasibleError(
            f"level {level:.6g} of {index.name!r} is outside the fibre's range or "
            "the dual projection stalled"
        )


def activation_fiber(
    frame: MuscleFrame,
    torque: ArrayLike,
    *,
    reserve_limit: float = RESERVE_FRACTION_LIMIT,
    linearity_limit: float = RESERVE_FRACTION_LIMIT,
    rank_rtol: float = 1e-12,
) -> ActivationFiber:
    """Build the activation fibre of ``frame`` at the net muscle ``torque`` (N m).

    The fibre is the set of activations whose secant-law torque equals the
    muscle-reachable torque nearest the request; with zero reserve that is the
    request itself.  ``rank_rtol`` is :class:`ForceNullSpace`'s default
    (``1e-12``, the SVD gap relative to the largest singular value).

    Raises:
        NotQualifiedError: the reserve or linearity gate of
            :func:`qualify_frame` fails (fail closed).
        ValueError: ``torque`` not finite/non-zero/length ``n``.
    """
    qual = require_qualified(
        qualify_frame(
            frame,
            torque,
            reserve_limit=reserve_limit,
            linearity_limit=linearity_limit,
        )
    )
    target = np.asarray(qual.reachable_torque)
    gain = frame.gain_matrix
    offset = target - frame.passive_torque
    scale = max(float(np.max(np.abs(gain))), float(np.max(np.abs(target))), 1e-300)
    row_scale = np.full(frame.n_coordinates, scale)
    fns = ForceNullSpace.from_balance(
        gain, offset, row_scale=row_scale, rank_rtol=rank_rtol
    )
    particular, basis = np.array(fns.particular), np.array(fns.basis)
    for arr in (particular, basis, row_scale):
        arr.flags.writeable = False
    return ActivationFiber(
        frame=frame,
        requested_torque=np.asarray(torque, dtype=np.float64),
        target_torque=target,
        qualification=qual,
        gain=gain,
        offset=offset,
        particular=particular,
        basis=basis,
        rank=int(fns.rank),
        nullity=int(basis.shape[1]),
        row_scale=row_scale,
        rank_rtol=rank_rtol,
    )
