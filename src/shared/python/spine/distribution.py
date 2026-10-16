"""Distribution of a pelvis-to-terminal rotation over spinal levels (SPN-2b).

Implements ADR-0055 item 7. Given a target rotation ``R_target`` and a
:class:`CouplingLaw` the independent coordinates solve::

    phi* = argmin || log( R_target^T R(phi) ) ||

where ``R(phi) = B @ R_chain(law.expand(phi)) @ E`` (``B``, ``E`` optional fixed
base and terminal rotations), ``R_chain`` is the ordered composition of
:meth:`SegmentalPose.compose` and ``log`` is the SO(3) rotation vector. The
coupling law is never altered: an unreachable target yields a non-zero
geodesic residual, reported, never repaired. Units: radians. Frames: ADR-0055
(X anterior, Y cranial, Z right).

Analytic Jacobian. With ``E = R_target^T R`` and ``r = log E``, a perturbation
``d phi`` gives ``E (I + [w]x)`` with ``w = sum_k (R_i B_i)^T a_ik dtheta_ik``
(``a_ik`` the ISB ZXY axis of angle ``k`` in the parent frame of level ``i``,
``B_i`` the product of the factors cranial to level ``i``), hence
``dr = Jr^-1(r) w`` with the right-Jacobian inverse of SO(3). Chaining
``dtheta/dphi`` from ``law.jacobian`` gives ``d r / d phi``. Finite differences
serve only as a test oracle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.shared.python.contracts import ensure
from src.shared.python.spine.coupling import CouplingLaw, level_rotations
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.segmental import (
    SegmentalKinematicsSeries,
    SegmentalPose,
)

__all__ = [
    "ChainOffsets",
    "DistributionResult",
    "DistributionSeriesResult",
    "ObservabilityReport",
    "distribute",
    "distribute_series",
    "distribute_series_detailed",
    "residual_and_jacobian",
]

_ROT_TOL = 1e-9
_RANK_RTOL = 1e-8
_NULL_FLAG = 1e-6
_SOLVER_TOL_FACTOR = 1e-5
_SOLVER_TOL_FLOOR = 1e-15


def _frozen(value: np.ndarray) -> np.ndarray:
    out = np.array(value, dtype=float, copy=True)
    out.setflags(write=False)
    return out


def _rotation(value: object, name: str) -> np.ndarray:
    try:
        rot = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array: {exc}") from exc
    if rot.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {rot.shape}")
    if not np.all(np.isfinite(rot)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(rot.T @ rot, np.eye(3), atol=_ROT_TOL, rtol=0) or (
        abs(float(np.linalg.det(rot)) - 1.0) > _ROT_TOL
    ):
        raise ValueError(f"{name} must be a proper rotation (orthonormal, det +1)")
    return rot


@dataclass(frozen=True)
class ChainOffsets:
    """Fixed rotations outside the law: ``R = base @ R_chain @ terminal``.

    Both must be proper (3, 3) rotations (``ValueError`` otherwise).
    """

    base_rotation: np.ndarray
    terminal_rotation: np.ndarray

    def __post_init__(self) -> None:
        base = _rotation(self.base_rotation, "base_rotation")
        term = _rotation(self.terminal_rotation, "terminal_rotation")
        object.__setattr__(self, "base_rotation", _frozen(base))
        object.__setattr__(self, "terminal_rotation", _frozen(term))


@dataclass(frozen=True)
class ObservabilityReport:
    """Rank analysis of the composed body-frame Jacobian ``d w / d phi`` at phi*.

    ``rank`` is the numerical rank of the (3, n_phi) Jacobian. ``null_space``
    (n_phi, n_phi - rank) spans coordinate directions that do not change the
    composed rotation to first order (not identified by the target);
    ``flagged_coordinates`` names coordinates with a non-negligible null-space
    component. ``unobservable_target_directions`` (3, 3 - rank) are body-frame
    rotation directions the law cannot produce at phi*.
    """

    rank: int
    singular_values: np.ndarray
    underdetermined: bool
    null_space: np.ndarray
    flagged_coordinates: tuple[str, ...]
    unobservable_target_directions: np.ndarray


@dataclass(frozen=True)
class DistributionResult:
    """Outcome of one distribution solve.

    ``residual_rad`` is the geodesic distance between the target and the
    returned pose (recomputed independently of the solver, agreement 1e-10).
    ``converged`` is the solver's convergence flag and does NOT imply a small
    residual. Arrays are read-only.
    """

    phi: np.ndarray
    pose: SegmentalPose
    level_angles: Mapping[SpinalLevel, np.ndarray]
    residual_rad: float
    converged: bool
    law_id: str
    law_version: str
    distribution_source: str
    observability: ObservabilityReport
    message: str

    def check_tolerance(self, tolerance_rad: float, *, raise_on_exceed: bool = False):
        """Return whether ``residual_rad <= tolerance_rad``; never alters the fit.

        Raises:
            ValueError: if ``tolerance_rad`` is not finite and positive, or if
                ``raise_on_exceed`` and the residual exceeds it.
        """
        tol = float(tolerance_rad)
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError(f"tolerance_rad must be finite and > 0, got {tol}")
        ok = self.residual_rad <= tol
        if not ok and raise_on_exceed:
            raise ValueError(
                f"residual {self.residual_rad:.3e} rad exceeds tolerance {tol:.3e} rad"
            )
        return ok


@dataclass(frozen=True)
class DistributionSeriesResult:
    """Per-frame results plus the packaged :class:`SegmentalKinematicsSeries`."""

    results: tuple[DistributionResult, ...]
    residual_rad: np.ndarray
    series: SegmentalKinematicsSeries


# ------------------------------------------------------------------- numerics


def _jr_inverse(rvec: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian of SO(3) at rotation vector ``rvec``."""
    theta = float(np.linalg.norm(rvec))
    skew = np.array(
        [[0.0, -rvec[2], rvec[1]], [rvec[2], 0.0, -rvec[0]], [-rvec[1], rvec[0], 0.0]]
    )
    if theta < 1e-4:
        coef = 1.0 / 12.0 + theta * theta / 720.0
    else:
        coef = 1.0 / theta**2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) + 0.5 * skew + coef * (skew @ skew)


def _isb_axes(angles: np.ndarray) -> np.ndarray:
    """Columns: axes of the ZXY angles (FE, LB, AR) in the level's parent frame."""
    sa, ca = np.sin(angles[0]), np.cos(angles[0])
    sb, cb = np.sin(angles[1]), np.cos(angles[1])
    return np.array([[0.0, ca, -sa * cb], [0.0, sa, ca * cb], [1.0, 0.0, sb]])


def _geodesic_vector(target: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(target.T @ rot).as_rotvec()


def _total_rotation(pose: SegmentalPose, fixed: ChainOffsets | None) -> np.ndarray:
    rot = pose.compose()[:3, :3]
    if fixed is None:
        return rot
    return fixed.base_rotation @ rot @ fixed.terminal_rotation


def _body_jacobian(
    pose: SegmentalPose,
    law: CouplingLaw,
    phi: np.ndarray,
    fixed: ChainOffsets | None,
) -> np.ndarray:
    """(3, n_phi) body-frame angular-velocity Jacobian ``d w / d phi``."""
    jac_theta = law.jacobian(phi)
    n_phi = phi.size
    items = pose.transforms
    factors = [it.neutral[:3, :3] @ it.rotation for it in items]
    suffix = np.eye(3) if fixed is None else fixed.terminal_rotation
    out = np.zeros((3, n_phi))
    for idx in range(len(items) - 1, -1, -1):
        angles = items[idx].isb_angles()
        into_frame = (items[idx].rotation @ suffix).T  # (R_i B_i)^T
        w_per_angle = into_frame @ _isb_axes(angles)  # (3, 3)
        out += w_per_angle @ jac_theta[items[idx].level]
        suffix = factors[idx] @ suffix
    return out


def residual_and_jacobian(
    phi: np.ndarray,
    law: CouplingLaw,
    target: np.ndarray,
    neutral: Mapping[SpinalLevel, np.ndarray] | None,
    fixed: ChainOffsets | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotation-vector residual ``log(R_target^T R(phi))`` and ``d r / d phi``.

    Returns ``((3,), (3, n_phi))``. ``target`` must already be a validated
    rotation. Does not modify the law.
    """
    pose = level_rotations(law, phi, neutral)
    resid = _geodesic_vector(target, _total_rotation(pose, fixed))
    jac = _jr_inverse(resid) @ _body_jacobian(pose, law, phi, fixed)
    return resid, jac


def _observability(jac_body: np.ndarray, names: tuple[str, ...]) -> ObservabilityReport:
    left, sing, vt = np.linalg.svd(jac_body, full_matrices=True)
    smax = float(sing[0]) if sing.size else 0.0
    rank = int(np.sum(sing > max(smax * _RANK_RTOL, 1e-12)))
    null = vt[rank:].T.copy()
    unobs = left[:, rank:].copy()
    flagged = tuple(
        name
        for name, row in zip(names, null, strict=True)
        if float(np.linalg.norm(row)) > _NULL_FLAG
    )
    for arr in (null, unobs):
        arr.setflags(write=False)
    return ObservabilityReport(
        rank=rank,
        singular_values=_frozen(sing),
        underdetermined=rank < jac_body.shape[1],
        null_space=null,
        flagged_coordinates=flagged,
        unobservable_target_directions=unobs,
    )


# ---------------------------------------------------------------- validation


def _check_bounds(
    bounds: Mapping[str, tuple[float, float]] | None, names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(len(names), -np.inf)
    upper = np.full(len(names), np.inf)
    if bounds is None:
        return lower, upper
    for key, pair in bounds.items():
        if key not in names:
            raise ValueError(f"bounds name unknown coordinate {key!r}; have {names}")
        lo, hi = (float(pair[0]), float(pair[1]))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"bounds for {key!r} must be finite, got {pair}")
        if lo >= hi:
            raise ValueError(f"bounds for {key!r} must satisfy lo < hi, got {pair}")
        lower[names.index(key)], upper[names.index(key)] = lo, hi
    return lower, upper


def _check_start(
    phi0: np.ndarray | None, n_phi: int, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    if phi0 is None:
        return np.clip(np.zeros(n_phi), lower, upper)
    start = np.asarray(phi0, dtype=float)
    if start.shape != (n_phi,) or not np.all(np.isfinite(start)):
        raise ValueError(
            f"phi0 must be finite with shape ({n_phi},), got {start.shape}"
        )
    if np.any(start < lower) or np.any(start > upper):
        raise ValueError("phi0 is outside bounds (infeasible start)")
    return start.copy()


def _check_inputs(target: object, law: object, tol_rad: float, max_iter: int):
    if not isinstance(law, CouplingLaw):
        raise TypeError(f"law must be a CouplingLaw, got {type(law).__name__}")
    rot = _rotation(target, "target_rotation")
    if not (np.isfinite(tol_rad) and tol_rad > 0.0):
        raise ValueError(f"tol_rad must be finite and > 0, got {tol_rad}")
    if int(max_iter) < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    return rot


# --------------------------------------------------------------------- solve


def distribute(
    target_rotation: np.ndarray,
    law: CouplingLaw,
    *,
    neutral: Mapping[SpinalLevel, np.ndarray] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    phi0: np.ndarray | None = None,
    tol_rad: float = 1e-8,
    max_iter: int = 200,
    fixed: ChainOffsets | None = None,
) -> DistributionResult:
    """Solve ``phi* = argmin ||log(R_target^T R(phi))||`` under ``law``.

    ``neutral`` maps levels to (4, 4) neutral offsets; ``bounds`` maps
    independent-coordinate names to finite ``(lo, hi)`` with ``lo < hi``;
    ``fixed`` adds base/terminal rotations outside the law. Uses
    :func:`scipy.optimize.least_squares` (trust-region reflective) with the
    analytic Jacobian of :func:`residual_and_jacobian`.

    Preconditions: ``law`` is a :class:`CouplingLaw` (``TypeError``);
    ``target_rotation`` is a finite proper (3, 3) rotation, bounds are finite
    and ordered, ``phi0`` is finite, correctly shaped and inside bounds,
    ``tol_rad > 0``, ``max_iter >= 1`` (``ValueError``).

    Postconditions: the law is untouched; ``residual_rad`` equals the
    geodesic distance of the returned pose to 1e-10; ``distribution_source``
    is ``coupling:<law_id>@<version>``; ``converged`` does not imply a small
    residual (an unreachable target converges to a positive one).
    """
    target = _check_inputs(target_rotation, law, tol_rad, max_iter)
    names = law.independent_names()
    lower, upper = _check_bounds(bounds, names)
    start = _check_start(phi0, len(names), lower, upper)

    # tol_rad is the residual scale the caller cares about; the trust-region
    # termination tolerances are set well below it so a reachable target
    # lands far inside tol_rad rather than just at its edge.
    solver_tol = max(tol_rad * _SOLVER_TOL_FACTOR, _SOLVER_TOL_FLOOR)

    def fun(x: np.ndarray) -> np.ndarray:
        return residual_and_jacobian(x, law, target, neutral, fixed)[0]

    def jac(x: np.ndarray) -> np.ndarray:
        return residual_and_jacobian(x, law, target, neutral, fixed)[1]

    sol = least_squares(
        fun,
        start,
        jac=jac,
        bounds=(lower, upper),
        method="trf",
        ftol=solver_tol,
        xtol=solver_tol,
        gtol=solver_tol,
        max_nfev=int(max_iter),
    )
    return _package(sol, law, target, neutral, fixed, names)


def _package(sol, law, target, neutral, fixed, names) -> DistributionResult:
    phi = np.asarray(sol.x, dtype=float)
    pose = level_rotations(law, phi, neutral)
    residual = float(
        np.linalg.norm(_geodesic_vector(target, _total_rotation(pose, fixed)))
    )
    ensure(
        abs(residual - float(np.linalg.norm(sol.fun))) < 1e-10,
        "residual must equal the recomputed geodesic distance",
        residual,
    )
    obs = _observability(_body_jacobian(pose, law, phi, fixed), names)
    angles = {lv: _frozen(a) for lv, a in law.expand(phi).items()}
    return DistributionResult(
        phi=_frozen(phi),
        pose=pose,
        level_angles=angles,
        residual_rad=residual,
        converged=bool(sol.status > 0),
        law_id=law.law_id,
        law_version=law.version,
        distribution_source=f"coupling:{law.law_id}@{law.version}",
        observability=obs,
        message=str(sol.message),
    )


# -------------------------------------------------------------------- series


def distribute_series_detailed(
    targets: np.ndarray,
    time_s: np.ndarray,
    law: CouplingLaw,
    *,
    tolerance_rad: float | None = None,
    **kwargs,
) -> DistributionSeriesResult:
    """Distribute every frame, warm-starting each solve from the previous phi.

    ``targets`` is (n, 3, 3), ``time_s`` (n,) strictly increasing. A frame is
    ``valid`` when its solve converged and (if ``tolerance_rad`` is given) its
    residual is within it; invalid frames are kept as solved, never repaired.
    ``kwargs`` are forwarded to :func:`distribute` (not ``phi0``, which seeds
    only the first frame).

    Raises:
        ValueError: on bad shapes, mismatched lengths or non-increasing time.
    """
    rots = np.asarray(targets, dtype=float)
    if rots.ndim != 3 or rots.shape[1:] != (3, 3):
        raise ValueError(f"targets must have shape (n, 3, 3), got {rots.shape}")
    times = np.asarray(time_s, dtype=float)
    if times.shape != (rots.shape[0],):
        raise ValueError(
            f"length mismatch: {rots.shape[0]} targets vs time_s {times.shape}"
        )
    if rots.shape[0] == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("time_s must be non-empty and strictly increasing")
    results: list[DistributionResult] = []
    for frame in rots:
        if results:
            kwargs["phi0"] = _clip_start(results[-1].phi, kwargs.get("bounds"), law)
        results.append(distribute(frame, law, **kwargs))
    resid = np.array([r.residual_rad for r in results])
    valid = np.array([r.converged for r in results], dtype=bool)
    if tolerance_rad is not None:
        valid &= resid <= float(tolerance_rad)
    series = SegmentalKinematicsSeries(
        times,
        tuple(r.pose for r in results),
        results[0].distribution_source,
        valid,
    )
    resid.setflags(write=False)
    return DistributionSeriesResult(tuple(results), resid, series)


def _clip_start(phi: np.ndarray, bounds, law: CouplingLaw) -> np.ndarray:
    lower, upper = _check_bounds(bounds, law.independent_names())
    return np.clip(phi, lower, upper)


def distribute_series(
    targets: np.ndarray, time_s: np.ndarray, law: CouplingLaw, **kwargs
) -> SegmentalKinematicsSeries:
    """Series variant of :func:`distribute` returning the kinematics series.

    See :func:`distribute_series_detailed` for the contract and per-frame
    residuals.
    """
    return distribute_series_detailed(targets, time_s, law, **kwargs).series
