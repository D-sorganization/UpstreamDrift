"""Hit-and-run sampling of null-space trajectories (research spike, NSX-70, #12588).

Exploratory.  The null-space coordinate of every frame is restricted to a
low-rank temporal basis, ``z(t) = sum_k B[t, k] C[k, :]``, so the sampler works
on the coefficient matrix ``C`` (``D = K n_z`` unknowns).  The admissible set
in ``C`` is the intersection over frames of the per-frame feasible sets
(:class:`~.feasible_set.FeasibleSet`) pulled back through ``B``:

* polyhedral rows ``g . z(t) <= h`` (torque capacity, normal force, grip floor
  and the declared friction-cone linearisation) stay linear in ``C``;
* the exact second-order-cone blocks ``b - A z(t)`` in the cone are kept exact,
  and the chord through a cone is a quadratic root problem.

Hit-and-run (Smith 1984) picks a direction, intersects the line with the set
and moves to a uniform point of the chord.  With a direction distribution
symmetric under ``d -> -d`` the uniform distribution on the bounded convex set
is invariant (tested on a box, a simplex and a disk).  The target is *uniform
in the coefficients ``C``*, which is a declared sampling measure and not a
claim about which swings occur.  Non-claims: no physical prior, no
convergence guarantee beyond the measured diagnostics, no objective.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.shared.python.estimation.mosaic.kinematic_basis import BSplineBasis
from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.solvers import SolveStatus, SOCBlock, solve_socp

logger = logging.getLogger(__name__)

__all__ = [
    "HitAndRunResult",
    "TrajectoryDomain",
    "bspline_temporal_basis",
    "hit_and_run",
    "interior_point",
    "run_chains",
]

_PARALLEL_TOL = 1e-14
_MIN_INTERIOR_RADIUS = 1e-9
_START_FRACTION = 0.9


def bspline_temporal_basis(times: Array, n_coefficients: int, degree: int = 3) -> Array:
    """``(T, K)`` clamped B-spline position basis (MOSAIC's ``BSplineBasis``).

    Rows sum to one.  Preconditions are those of ``BSplineBasis.uniform``.
    """
    return np.array(BSplineBasis.uniform(times, n_coefficients, degree).position)


@dataclass(frozen=True, eq=False)
class _ConeFrame:
    frame: int
    A: Array
    b: Array


class TrajectoryDomain:
    """Admissible coefficient set ``{C : z(t) in feasible set of frame t}``.

    ``frames`` are per-frame feasible sets with equal ``n_z``; ``basis`` is the
    ``(T, K)`` temporal basis.  Coefficients are flat vectors of length
    ``K * n_z`` (row-major ``C``).  Immutable after construction.
    """

    def __init__(self, frames: Sequence[FeasibleSet], basis: Array) -> None:
        if len(frames) == 0:
            raise ValueError("need at least one frame")
        for fs in frames:
            if not isinstance(fs, FeasibleSet):
                raise TypeError(f"frames must be FeasibleSet, got {type(fs)!r}")
        b = np.array(basis, dtype=np.float64)
        if b.ndim != 2 or b.shape[0] != len(frames):
            raise ValueError(
                f"basis needs one row per frame ({len(frames)}), got {b.shape}"
            )
        if not np.all(np.isfinite(b)):
            raise ValueError("basis must be finite")
        n_z = frames[0].n_z
        if any(fs.n_z != n_z for fs in frames):
            raise ValueError("every frame must share the same n_z")
        self.basis = b
        self.n_frames = len(frames)
        self.n_z = n_z
        self.n_basis = b.shape[1]
        self.dimension = self.n_basis * n_z
        g, h, idx = [], [], []
        cones: list[_ConeFrame] = []
        for t, fs in enumerate(frames):
            g += [r.g for r in fs.rows]
            h += [r.h for r in fs.rows]
            idx += [t] * len(fs.rows)
            cones += [_ConeFrame(t, c.A, c.b) for c in fs.cones]
        self._g = np.array(g).reshape(-1, n_z)
        self._h = np.array(h)
        self._idx = np.array(idx, dtype=int)
        self._cones = tuple(cones)

    # -------------------------------------------------------------- geometry
    def trajectory(self, coefficients: Array) -> Array:
        """``(T, n_z)`` trajectory ``B C`` of one coefficient vector."""
        c = self._coeff(coefficients)
        return np.asarray(self.basis @ c, dtype=np.float64)

    def trajectories(self, coefficients: Array) -> Array:
        """``(N, T, n_z)`` trajectories of ``N`` coefficient vectors."""
        arr = np.asarray(coefficients, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.dimension:
            raise ValueError(f"coefficients must be (N, {self.dimension})")
        mats = arr.reshape(arr.shape[0], self.n_basis, self.n_z)
        return np.asarray(np.einsum("tk,nkj->ntj", self.basis, mats))

    def _coeff(self, coefficients: Array) -> Array:
        arr = np.asarray(coefficients, dtype=np.float64)
        if arr.shape != (self.dimension,):
            raise ValueError(
                f"start/coefficients must have shape ({self.dimension},), "
                f"got {arr.shape}"
            )
        return arr.reshape(self.n_basis, self.n_z)

    def min_slack(self, coefficients: Array) -> float:
        """Smallest row slack ``h - g . z`` and cone margin ``t - ||u||``.

        Positive iff the point is strictly inside every row and cone;
        ``+inf`` for a domain without constraints.
        """
        z = self.trajectory(coefficients)
        values = [math.inf]
        if self._h.size:
            values.append(float(np.min(self._h - self._row_dot(z))))
        for cone in self._cones:
            w = cone.b - cone.A @ z[cone.frame]
            values.append(float(w[0] - np.linalg.norm(w[1:])))
        return min(values)

    def contains(self, coefficients: Array, *, tol: float = 0.0) -> bool:
        """True iff the point violates no row or cone by more than ``tol``."""
        if tol < 0.0:
            raise ValueError("tol must be non-negative")
        return self.min_slack(coefficients) >= -tol

    def _row_dot(self, z: Array) -> Array:
        return np.asarray(np.einsum("mj,mj->m", self._g, z[self._idx]))

    def chord(self, coefficients: Array, direction: Array) -> tuple[float, float]:
        """Interval ``[lo, hi]`` of steps ``s`` keeping ``C + s d`` admissible.

        Raises ``ValueError`` if the line leaves the set on neither side
        (unbounded domain).  The point must be inside the set.
        """
        d = self._coeff(direction)
        z, dz = self.trajectory(coefficients), np.asarray(self.basis @ d)
        lo, hi = -math.inf, math.inf
        if self._h.size:
            slack = np.maximum(self._h - self._row_dot(z), 0.0)
            rate = self._row_dot(dz)
            pos, neg = rate > _PARALLEL_TOL, rate < -_PARALLEL_TOL
            if pos.any():
                hi = float(np.min(slack[pos] / rate[pos]))
            if neg.any():
                lo = float(np.max(slack[neg] / rate[neg]))
        for cone in self._cones:
            c_lo, c_hi = _cone_interval(cone, z[cone.frame], dz[cone.frame])
            lo, hi = max(lo, c_lo), min(hi, c_hi)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError("domain is unbounded along the sampled direction")
        return lo, hi


def _cone_interval(cone: _ConeFrame, z: Array, dz: Array) -> tuple[float, float]:
    """Steps ``s`` with ``b - A (z + s dz)`` in the second-order cone."""
    w, u = cone.b - cone.A @ z, cone.A @ dz
    lo, hi = -math.inf, math.inf
    if u[0] > _PARALLEL_TOL:  # t(s) = w0 - s u0 >= 0
        hi = w[0] / u[0]
    elif u[0] < -_PARALLEL_TOL:
        lo = w[0] / u[0]
    a = float(u[1:] @ u[1:] - u[0] ** 2)
    b = -2.0 * float(w[1:] @ u[1:] - w[0] * u[0])
    c = float(w[1:] @ w[1:] - w[0] ** 2)  # < 0 strictly inside
    scale = max(abs(a), abs(b), abs(c), 1.0)
    if abs(a) <= 1e-12 * scale:
        if b > 0.0:
            hi = min(hi, -c / b)
        elif b < 0.0:
            lo = max(lo, -c / b)
        return lo, hi
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return lo, hi  # a < 0 and no crossing: only the t >= 0 bound applies
    root = math.sqrt(disc)
    q = -0.5 * (b + math.copysign(root, b))
    r1, r2 = sorted((q / a, c / q if q != 0.0 else q / a))
    if a > 0.0:
        return max(lo, r1), min(hi, r2)
    if r1 > 0.0:  # a < 0, both roots positive: stay below the first root
        return lo, min(hi, r1)
    return max(lo, r2), hi


# ------------------------------------------------------------- start point
def interior_point(domain: TrajectoryDomain) -> Array:
    """Coefficient vector maximising the inscribed margin ``r`` (SOCP).

    Rows use the Euclidean margin in coefficient space; cones use the margin
    of the cone's first component.  Raises ``ValueError`` if the set is empty,
    has no interior (``r`` below ``1e-9``) or the margin is unbounded.
    """
    dim, g, idx = domain.dimension, domain._g, domain._idx
    rows = (domain.basis[idx][:, :, None] * g[:, None, :]).reshape(len(idx), dim)
    norms = np.linalg.norm(rows, axis=1)
    a_ub = np.hstack([rows, norms[:, None]]) if len(idx) else None
    blocks = []
    for cone in domain._cones:
        a_c = np.kron(domain.basis[cone.frame][None, :], cone.A)
        margin = np.zeros((a_c.shape[0], 1))
        margin[0, 0] = 1.0
        blocks.append(SOCBlock(np.hstack([a_c, margin]), cone.b))
    obj = np.zeros(dim + 1)
    obj[-1] = -1.0
    bounds = [(None, None)] * dim + [(0.0, None)]
    res = solve_socp(
        obj,
        None,
        None,
        a_ub,
        domain._h if domain._h.size else None,
        blocks,
        bounds=bounds,
    )
    if res.status is SolveStatus.UNBOUNDED:
        raise ValueError("domain is unbounded: the inscribed margin has no maximum")
    if res.status is not SolveStatus.OPTIMAL or res.x is None:
        raise ValueError(f"no interior point: {res.status.name} ({res.reason})")
    if res.x[-1] <= _MIN_INTERIOR_RADIUS:
        raise ValueError(f"no interior: inscribed margin {res.x[-1]:.3g}")
    return np.asarray(res.x[:-1], dtype=np.float64)


# ------------------------------------------------------------------ sampler
@dataclass(frozen=True, eq=False)
class HitAndRunResult:
    """One chain: ``(n_draws, D)`` coefficients plus how it was produced."""

    coefficients: Array
    seed: int
    burn_in: int
    thin: int
    start: Array

    @property
    def n_draws(self) -> int:
        return int(self.coefficients.shape[0])


def hit_and_run(
    domain: TrajectoryDomain,
    start: Array,
    *,
    n_draws: int,
    seed: int,
    thin: int = 1,
    burn_in: int = 0,
    direction_scale: Array | None = None,
) -> HitAndRunResult:
    """Run one hit-and-run chain and record every ``thin``-th state.

    Each step draws a Gaussian direction (optionally ``direction_scale @ xi``,
    symmetric so the uniform law stays invariant), normalises it, and moves
    to a uniform point of the chord.  Exactly two random variates are consumed
    per step, so ``thin`` only subsamples a fixed stream.

    Preconditions: ``start`` strictly inside the domain; ``n_draws >= 1``,
    ``thin >= 1``, ``burn_in >= 0``; ``direction_scale`` is ``(D, D)`` and
    invertible if given.  Postconditions: every stored state satisfies every
    constraint (to rounding); the chain is a pure function of
    ``(domain, start, seed, options)``.
    """
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}")
    if thin < 1 or burn_in < 0:
        raise ValueError(f"thin must be >= 1 and burn_in >= 0, got {thin}, {burn_in}")
    x: Array = np.array(domain._coeff(start), dtype=np.float64).ravel()
    if domain.min_slack(x) <= 0.0:
        raise ValueError("start must be strictly inside the domain")
    scale = _validated_scale(direction_scale, domain.dimension)
    rng = np.random.default_rng(seed)
    out = np.empty((n_draws, domain.dimension))
    for step in range(burn_in + n_draws * thin):
        xi = rng.standard_normal(domain.dimension)
        d = xi if scale is None else scale @ xi
        d = d / np.linalg.norm(d)
        lo, hi = domain.chord(x, d)
        x = x + (lo + (hi - lo) * rng.random()) * d
        if step >= burn_in and (step - burn_in + 1) % thin == 0:
            out[(step - burn_in) // thin] = x
    return HitAndRunResult(out, int(seed), burn_in, thin, np.array(start, float))


def _validated_scale(scale: Array | None, dim: int) -> Array | None:
    if scale is None:
        return None
    arr = np.asarray(scale, dtype=np.float64)
    if arr.shape != (dim, dim) or not np.all(np.isfinite(arr)):
        raise ValueError(f"direction_scale must be finite ({dim}, {dim})")
    if np.linalg.matrix_rank(arr) < dim:
        raise ValueError("direction_scale must be invertible")
    return arr


def run_chains(
    domain: TrajectoryDomain,
    *,
    n_chains: int,
    n_draws: int,
    seed: int,
    burn_in: int = 500,
    thin: int = 1,
    direction_scale: Array | None = None,
) -> list[HitAndRunResult]:
    """Independent chains from over-dispersed starts (reproducible from ``seed``).

    Each chain starts 90 % of the way from the inscribed centre to a random
    boundary point, which is deliberately far from the bulk so split R-hat can
    detect slow mixing.  Chain seeds are spawned from ``seed``.
    """
    if n_chains < 1:
        raise ValueError(f"n_chains must be >= 1, got {n_chains}")
    centre = interior_point(domain)
    children = np.random.SeedSequence(seed).spawn(n_chains)
    chains = []
    for child in children:
        chain_seed = int(child.generate_state(1)[0])
        rng = np.random.default_rng(chain_seed ^ 0x5EED)
        d = rng.standard_normal(domain.dimension)
        d /= np.linalg.norm(d)
        lo, hi = domain.chord(centre, d)
        begin = centre + _START_FRACTION * (hi if rng.random() < 0.5 else lo) * d
        chains.append(
            hit_and_run(
                domain,
                begin,
                n_draws=n_draws,
                seed=chain_seed,
                thin=thin,
                burn_in=burn_in,
                direction_scale=direction_scale,
            )
        )
    return chains
