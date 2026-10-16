"""The activation fibre at fixed net torque (NSX-94, #12723).

The set of activations that produce one muscle torque,

    F(tau) = { a in [0, 1]^m : G a = tau - R^T f_p },       G = R^T diag(F_a),

is the *activation fibre*.  Its affine hull is parallel to ``ker G`` (dimension
``m - rank G``, found with the repository's :class:`ForceNullSpace`, never
re-implemented here); the activation box turns it into a bounded polytope.
Moving along it changes no joint torque at all and changes only which muscles
pull against which -- co-contraction -- so a torque envelope, however tight,
quotients it away (review finding A4 of ``muscle_level_view.md``).

Co-contraction is never inferred.  The experiment *declares* a
:class:`CocontractionIndex` ``c(a) = w^T a`` (default: total activation) and
sweeps its level; for a level inside ``[c_min, c_max]`` the representative is the
unique minimum-norm point

    a*(c) = argmin 1/2 ||a||^2   s.t.  G a = b,  w^T a = c,  0 <= a <= 1,

the per-level counterpart of the static-optimisation criterion
``min sum a^2`` used by ``myofullbody.redundancy`` and
``musculoskeletal_static_opt``.  It is a declared selection rule, not physiology:
inside one level the remaining sub-fibre (dimension ``m - rank G - 1``) is
unexplored, and :meth:`ActivationFiber.sample` shows it.

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
* :meth:`ActivationFiber.sample` is hit-and-run in the null coordinates from
  the Chebyshev centre.  Samples are *coverage of the fibre*, not a probability
  statement about a golfer (epic rule R14: sampled coverage, not certified
  bounds); mixing is not certified.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.muscle_frame import (
    RESERVE_FRACTION_LIMIT,
    FrameQualification,
    MuscleFrame,
    qualify_frame,
    require_qualified,
)
from src.shared.python.motion_matching.force_nullspace.space import ForceNullSpace

__all__ = [
    "LEVEL_INSET",
    "LP_BACKEND",
    "ActivationFiber",
    "CocontractionIndex",
    "FiberInfeasibleError",
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

_DEGENERATE_RADIUS = 1e-9
"""Chebyshev radius below which a fibre counts as collapsed (``balance_tolerance``
of :class:`ForceNullSpace` is the same ``1e-9``)."""


class FiberInfeasibleError(ValueError):
    """The requested fibre point or level does not exist (fail closed)."""


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

    Attributes:
        frame: The :class:`MuscleFrame` the fibre belongs to.
        requested_torque: The torque asked for, ``(n,)``.
        target_torque: The muscle-reachable torque used (equals the request
            when the reserve is zero), ``(n,)``.
        qualification: The passed :class:`FrameQualification`.
        particular, basis: ``a = particular + basis @ z`` solves ``G a = b``.
        rank, nullity: Of ``G`` (``ForceNullSpace``), before the box.
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

    def index_range(self, index: CocontractionIndex) -> tuple[float, float]:
        """Exact ``(min, max)`` of ``w^T a`` over the fibre (two LPs).

        Raises :class:`FiberInfeasibleError` if HiGHS does not report an optimum.
        """
        w = self._check_index(index)
        a_eq, b_eq = self._equality()
        out = []
        for sign in (1.0, -1.0):
            res = linprog(
                sign * w,
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=(0.0, 1.0),
                method="highs",
                options={
                    "primal_feasibility_tolerance": _LP_TOL,
                    "dual_feasibility_tolerance": _LP_TOL,
                },
            )
            if res.status != 0 or res.fun is None:
                raise FiberInfeasibleError(f"fibre LP failed: {res.message}")
            out.append(sign * float(res.fun))
        return out[0], out[1]

    def _rows(self) -> tuple[Array, Array]:
        n_mat = self.basis
        rows = np.vstack([n_mat, -n_mat])
        bounds = np.concatenate([1.0 - self.particular, self.particular])
        return rows, bounds

    def chebyshev(self) -> tuple[Array, float]:
        """Chebyshev centre (null coordinates ``z``) and radius of the fibre.

        The radius is the largest ball, in activation units inside the null
        space, that fits in the box; ``0`` marks a collapsed fibre.
        """
        k = self.nullity
        if k == 0:
            return np.zeros(0), 0.0
        rows, bounds = self._rows()
        norms = np.linalg.norm(rows, axis=1)
        a_ub = np.hstack([rows, norms[:, None]])
        cost = np.zeros(k + 1)
        cost[-1] = -1.0
        res = linprog(
            cost,
            A_ub=a_ub,
            b_ub=bounds,
            bounds=[(None, None)] * k + [(0.0, None)],
            method="highs",
            options={
                "primal_feasibility_tolerance": _LP_TOL,
                "dual_feasibility_tolerance": _LP_TOL,
            },
        )
        if res.status != 0 or res.x is None:
            raise FiberInfeasibleError(f"Chebyshev LP failed: {res.message}")
        return np.asarray(res.x[:k]), float(res.x[k])

    @property
    def is_degenerate(self) -> bool:
        """``True`` when the box collapses the fibre to (numerically) a point."""
        return self.chebyshev()[1] < _DEGENERATE_RADIUS

    # ------------------------------------------------------------- sampling
    def sample(
        self,
        n: int,
        *,
        rng: np.random.Generator,
        thin: int = 5,
        burn_in: int = 100,
    ) -> Array:
        """``n`` hit-and-run samples of the fibre, shape ``(n, m)``.

        Every sample satisfies ``G a = b`` to the null-basis accuracy and the
        box exactly (steps stop at the box).  A collapsed fibre returns its
        single point ``n`` times.  Preconditions: ``n >= 1``, ``thin >= 1``,
        ``burn_in >= 0``.  The samples are coverage, not a probability claim.
        """
        if n < 1 or thin < 1 or burn_in < 0:
            raise ValueError("n and thin must be >= 1 and burn_in >= 0")
        z, radius = self.chebyshev()
        if self.nullity == 0 or radius < _DEGENERATE_RADIUS:
            point = self.particular + self.basis @ z
            return np.tile(np.clip(point, 0.0, 1.0), (n, 1))
        rows, bounds = self._rows()
        out = np.empty((n, self.frame.n_muscles))
        for step in range(burn_in + n * thin):
            d = rng.normal(size=self.nullity)
            d /= np.linalg.norm(d)
            slope = rows @ d
            slack = np.maximum(bounds - rows @ z, 0.0)
            up, down = slope > 1e-14, slope < -1e-14
            t_hi = np.min(slack[up] / slope[up])
            t_lo = np.max(slack[down] / slope[down])
            z = z + rng.uniform(t_lo, t_hi) * d
            done = step - burn_in
            if done >= 0 and done % thin == thin - 1:
                out[done // thin] = np.clip(self.particular + self.basis @ z, 0.0, 1.0)
        return out

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
    )
