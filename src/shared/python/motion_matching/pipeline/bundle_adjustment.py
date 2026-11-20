"""Multi-frame marker-offset bundle adjustment, engine-agnostic core (#15296).

One address frame cannot separate where a marker sits on a body from the pose
of that body. Over several frames it can: the offsets are shared, the poses are
not. ``solve_bundle`` solves every selected frame's pose and every marker's
body-frame offset together, with a Gaussian prior and hard bounds on each
offset, and says what the data did not decide: the prior's share of chi2, the
offsets resting on a bound and the offset directions the frames never excite.

Kinematics only, no engine: the caller supplies forward kinematics as a
callable. Nothing here is a physical acceptance of a fit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

Array = np.ndarray
#: pose ``(n_q,)`` -> world frames of every body ``(n_bodies, 4, 4)``.
ForwardKinematics = Callable[[Array], Array]


@dataclass(frozen=True)
class BundleSettings:
    """Every number the adjustment uses. Defaults marked *experiment* are the
    values of the out-of-repository fit this module ports (see ``FIT_PATH.md``);
    they are choices of that fit, not measurements."""

    #: Marker residual scale, metres (*experiment*: skin-marker noise scale).
    marker_sigma_m: float = 0.010
    #: Hard offset bound, in prior sigmas either side of the seed (*experiment*).
    bound_sigmas: float = 3.0
    #: Fewest frames accepted; one frame cannot separate offset from pose.
    min_frames: int = 2
    #: A direction is weak when data carry under this multiple of the prior's
    #: information along it.
    weak_information_ratio: float = 1.0
    #: A marker is named for a weak direction holding this share of it or more.
    weak_loading_share: float = 0.1
    #: An offset this close to its bound counts as on it, metres.
    bound_atol_m: float = 1e-5
    #: Forward-difference pose step of the identifiability Jacobian, radians.
    pose_fd_step: float = 1e-6
    #: Relative singular-value cut of a frame's pose Jacobian.
    rank_rcond: float = 1e-9
    #: Stop when an accepted step gains under this share of the cost, or after
    #: this many linearisations (the second is not convergence).
    tolerance: float = 1e-10
    max_iterations: int = 200
    #: Levenberg-Marquardt damping: start, growth on a refused step and decay
    #: on an accepted one (*experiment*: 5 and 3), range, and diagonal floor.
    damping_start: float = 1e-3
    damping_up: float = 5.0
    damping_down: float = 3.0
    damping_min: float = 1e-9
    damping_max: float = 1e10
    damping_floor: float = 1e-9

    def __post_init__(self) -> None:
        for item in fields(self):
            if not getattr(self, item.name) > 0:
                raise ValueError(f"BundleSettings.{item.name} must be positive")


@dataclass(frozen=True)
class BundleProblem:
    """K frames, M markers, n_q pose coordinates.

    ``observations`` ``(K, M, 3)`` are world positions, read only where
    ``valid`` ``(K, M)``. ``offset_seeds`` ``(M, 3)`` are both the start and the
    prior mean, ``offset_sigmas`` ``(M, 3)`` the prior sigmas. ``pose_prior`` is
    ``(mean, sigma)``, both ``(K, n_q)``; an infinite sigma is no prior.
    ``constraint`` maps a pose to already-scaled residuals of fixed length
    (loop closure, ground contact). ``body_jacobian`` maps a pose to
    ``(n_bodies, 6, n_q)``: world velocity of each body origin (rows 0-2)
    and world angular velocity (3-5) per unit pose rate; without it the pose
    derivative is differenced from ``fk``.
    """

    fk: ForwardKinematics
    marker_bodies: Array
    observations: Array
    valid: Array
    offset_seeds: Array
    offset_sigmas: Array
    pose_start: Array
    pose_bounds: tuple[Array, Array]
    pose_prior: tuple[Array, Array] | None = None
    constraint: Callable[[Array], Array] | None = None
    body_jacobian: Callable[[Array], Array] | None = None

    def check(self, settings: BundleSettings) -> None:
        """Raise ``TypeError`` / ``ValueError`` unless the problem is well formed."""
        if not callable(self.fk):
            raise TypeError("fk must be callable: pose -> (n_bodies, 4, 4)")
        for name in ("constraint", "body_jacobian"):
            if getattr(self, name) is not None and not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable or None")
        frames, markers = np.shape(self.observations)[:2]
        n_q = np.shape(self.pose_start)[-1]
        shapes = {
            "observations": (self.observations, (frames, markers, 3)),
            "valid": (self.valid, (frames, markers)),
            "marker_bodies": (self.marker_bodies, (markers,)),
            "offset_seeds": (self.offset_seeds, (markers, 3)),
            "offset_sigmas": (self.offset_sigmas, (markers, 3)),
            "pose_start": (self.pose_start, (frames, n_q)),
            "pose_bounds": (np.asarray(self.pose_bounds), (2, n_q)),
        }
        for name, (value, shape) in shapes.items():
            if np.shape(value) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if frames < settings.min_frames:
            raise ValueError(
                f"bundle adjustment needs at least {settings.min_frames} frames, "
                f"got {frames}: one frame cannot separate offset from pose"
            )
        seen = self.observations[np.asarray(self.valid, bool)]
        finite = {**shapes, "observations": (seen, None)}
        for name in ("offset_seeds", "offset_sigmas", "pose_start", "observations"):
            if not np.all(np.isfinite(finite[name][0])):
                raise ValueError(f"{name} must be finite (observations: where valid)")
        if not np.all(np.asarray(self.offset_sigmas) > 0):
            raise ValueError("offset_sigmas must be positive")
        if not np.all(self.pose_bounds[0] < self.pose_bounds[1]):
            raise ValueError("pose_bounds must have lower < upper")
        if self.pose_prior is not None:
            mean, sigma = self.pose_prior
            if np.shape(mean) != (frames, n_q) or np.shape(sigma) != (frames, n_q):
                raise ValueError(f"pose_prior arrays must have shape {(frames, n_q)}")
            if not np.all(np.asarray(sigma) > 0):
                raise ValueError("pose_prior sigmas must be positive (inf: none)")


@dataclass(frozen=True)
class BundleResult:
    """Postconditions: ``offsets`` lie within ``bound_sigmas`` prior sigmas of
    their seeds and ``poses`` within the pose bounds; the four chi2 terms sum
    to the minimised cost; ``on_bound`` lists ``(marker, axis)``;
    ``weakly_identified`` lists markers that load a data-weak offset direction
    whose information ratios are ``weak_information_ratios`` (ascending)."""

    offsets: Array
    poses: Array
    rms_m: float
    per_marker_rms_m: Array
    per_frame_rms_m: Array
    data_chi2: float
    constraint_chi2: float
    offset_prior_chi2: float
    pose_prior_chi2: float
    on_bound: tuple[tuple[int, int], ...]
    weakly_identified: tuple[int, ...]
    weak_information_ratios: tuple[float, ...]
    converged: bool

    @property
    def prior_share(self) -> float:
        """Share of the total chi2 carried by the offset and pose priors."""
        prior = self.offset_prior_chi2 + self.pose_prior_chi2
        total = prior + self.data_chi2 + self.constraint_chi2
        return prior / total if total > 0 else 0.0


def _marker_errors(prob: BundleProblem, k: int, pose: Array, offsets: Array) -> Array:
    """Model minus observed marker positions of frame ``k``; zero where invalid."""
    frames = np.asarray(prob.fk(pose))[prob.marker_bodies]
    model = np.einsum("mij,mj->mi", frames[:, :3, :3], offsets) + frames[:, :3, 3]
    return np.where(prob.valid[k][:, None], model - prob.observations[k], 0.0)


def _terms(prob: BundleProblem, sigma: float, poses: Array, offsets: Array) -> Any:
    """Residual blocks: data, constraint, offset prior, pose prior."""
    data = [_marker_errors(prob, k, pose, offsets) for k, pose in enumerate(poses)]
    closed = [] if prob.constraint is None else [prob.constraint(p) for p in poses]
    prior = (offsets - prob.offset_seeds) / prob.offset_sigmas
    out = [np.ravel(data) / sigma, np.ravel(closed), prior.ravel()]
    if prob.pose_prior is None:
        return [*out, np.zeros(0)]
    mean, pose_sigma = prob.pose_prior
    return [*out, ((poses - mean) / pose_sigma)[np.isfinite(pose_sigma)]]


def _cost(prob: BundleProblem, sigma: float, poses: Array, offsets: Array) -> float:
    return float(sum(term @ term for term in _terms(prob, sigma, poses, offsets)))


def _difference(fn: Callable[[Array], Any], pose: Array, step: float) -> Any:
    """``fn(pose)`` and its forward difference, the pose on the last axis."""
    base = np.atleast_1d(np.asarray(fn(pose), float))
    slope = np.empty((*base.shape, len(pose)))
    for j in range(len(pose)):
        moved = pose.copy()
        moved[j] += step
        slope[..., j] = (np.atleast_1d(np.asarray(fn(moved), float)) - base) / step
    return base, slope


def _linearise(
    prob: BundleProblem, settings: BundleSettings, k: int, state: tuple[Array, Array]
) -> tuple[Array, Array, Array]:
    """Frame ``k`` at ``state`` (pose, offsets): residual, d/d pose, d/d offsets.

    Rows are the scaled marker errors, then the constraint. The offset
    derivative is exact (the body rotation); the pose derivative comes from
    ``body_jacobian`` when the problem has one, else from differences of ``fk``.
    """
    pose, offsets = state
    sigma, step, bodies = (
        settings.marker_sigma_m,
        settings.pose_fd_step,
        prob.marker_bodies,
    )
    seen = np.asarray(prob.valid[k], bool)
    rotations = np.asarray(prob.fk(pose))[bodies][:, :3, :3]
    if prob.body_jacobian is None:
        slope = _difference(prob.fk, pose, step)[1][bodies]
        moved = np.einsum("mijq,mj->miq", slope[:, :3, :3], offsets) + slope[:, :3, 3]
    else:
        twist = np.asarray(prob.body_jacobian(pose), float)[bodies]
        arm = np.einsum("mij,mj->mi", rotations, offsets)[:, :, None]
        moved = twist[:, :3] + np.cross(twist[:, 3:], arm, axis=1)
    residual = [_marker_errors(prob, k, pose, offsets).ravel() / sigma]
    by_pose = [(moved * seen[:, None, None]).reshape(-1, len(pose)) / sigma]
    by_offset: Array = np.zeros((offsets.size, offsets.size))
    for m in np.flatnonzero(seen):
        by_offset[3 * m : 3 * m + 3, 3 * m : 3 * m + 3] = rotations[m] / sigma
    if prob.constraint is not None:
        closed, closed_slope = _difference(prob.constraint, pose, step)
        residual.append(closed)
        by_pose.append(closed_slope)
        by_offset = np.vstack([by_offset, np.zeros((len(closed), offsets.size))])
    return np.concatenate(residual), np.vstack(by_pose), by_offset


def _with_pose_prior(prob: BundleProblem, k: int, pose: Array, block: Any) -> Any:
    """``block`` of ``_linearise`` with frame ``k``'s pose-prior rows appended."""
    if prob.pose_prior is None:
        return block
    mean, sigmas = prob.pose_prior[0][k], prob.pose_prior[1][k]
    rows = np.flatnonzero(np.isfinite(sigmas))
    return (
        np.concatenate([block[0], (pose - mean)[rows] / sigmas[rows]]),
        np.vstack([block[1], np.eye(len(pose))[rows] / sigmas[rows, None]]),
        np.vstack([block[2], np.zeros((len(rows), block[2].shape[1]))]),
    )


def _free(value: Array, gradient: Array, lower: Array, upper: Array) -> Array:
    """False where ``value`` rests on a bound that descent would push through."""
    return ~(((value <= lower) & (gradient > 0)) | ((value >= upper) & (gradient < 0)))


def _schur_step(
    blocks: list[Any], prior: tuple[Array, Array], damping: float, floor: float
) -> tuple[Array, Array]:
    """Damped Gauss-Newton step (poses, offsets): each frame's pose block is
    eliminated, so the only coupled solve is the size of the offsets.
    ``prior`` is the offset prior's residual and weights (zero where held)."""
    information = np.diag(prior[1] ** 2)
    gradient = prior[1] * prior[0]
    for residual, _, by_offset in blocks:
        information += by_offset.T @ by_offset
        gradient += by_offset.T @ residual
    reduced = information + damping * np.diag(np.diag(information) + floor)
    solved = []
    for residual, by_pose, by_offset in blocks:
        hessian = by_pose.T @ by_pose
        hessian += damping * np.diag(np.diag(hessian) + floor)
        cross = by_pose.T @ by_offset
        solved.append(
            np.linalg.solve(hessian, np.column_stack([cross, by_pose.T @ residual]))
        )
        reduced -= cross.T @ solved[-1][:, :-1]
        gradient -= cross.T @ solved[-1][:, -1]
    step = -np.linalg.solve(reduced, gradient)
    return np.array([-(part[:, -1] + part[:, :-1] @ step) for part in solved]), step


def _minimise(
    prob: BundleProblem,
    settings: BundleSettings,
    box: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array, bool]:
    """Bounded Levenberg-Marquardt from the seeds: poses, offsets, converged.

    Converged means an accepted step gained less than ``tolerance`` of the
    cost, or no step lowers the cost even at ``damping_max``; running out of
    ``max_iterations`` is not convergence.
    """
    lower, upper, low, high = box
    sigma, weights = settings.marker_sigma_m, 1.0 / np.ravel(prob.offset_sigmas)
    poses = np.clip(np.asarray(prob.pose_start, float), lower, upper)
    offsets = np.array(prob.offset_seeds, float)
    cost, damping = _cost(prob, sigma, poses, offsets), settings.damping_start
    for _ in range(settings.max_iterations):
        blocks = []
        pull = (offsets - prob.offset_seeds).ravel() * weights
        gradient = weights * pull
        for k, pose in enumerate(poses):
            block = _linearise(prob, settings, k, (pose, offsets))
            residual, by_pose, by_offset = _with_pose_prior(prob, k, pose, block)
            gradient += by_offset.T @ residual
            by_pose = by_pose * _free(pose, by_pose.T @ residual, lower, upper)
            blocks.append((residual, by_pose, by_offset))
        free = _free(offsets.ravel(), gradient, low.ravel(), high.ravel())
        blocks = [(r, a, b * free) for r, a, b in blocks]
        while damping <= settings.damping_max:
            turn, shift = _schur_step(
                blocks, (pull, weights * free), damping, settings.damping_floor
            )
            trial = np.clip(poses + turn, lower, upper)
            moved = np.clip(offsets + shift.reshape(-1, 3), low, high)
            lowered = _cost(prob, sigma, trial, moved)
            if lowered < cost:
                break
            damping *= settings.damping_up
        else:
            return poses, offsets, True
        gain, cost = (cost - lowered) / cost, lowered
        poses, offsets = trial, moved
        damping = max(damping / settings.damping_down, settings.damping_min)
        if gain < settings.tolerance:
            return poses, offsets, True
    return poses, offsets, False


def _offset_information(
    prob: BundleProblem, settings: BundleSettings, poses: Array, offsets: Array
) -> Array:
    """Data information on the offsets with every pose projected out, in units
    of the offset prior's information (``(3M, 3M)``, symmetric).

    Pose priors are left out on purpose: they are what this must expose.
    Constraints stay in: they restrict the poses.
    """
    lower, upper = prob.pose_bounds
    info = np.zeros((offsets.size, offsets.size))
    for k, pose in enumerate(poses):
        _, by_pose, by_offset = _linearise(prob, settings, k, (pose, offsets))
        inside = (pose > lower) & (pose < upper)
        basis, singular, _ = np.linalg.svd(by_pose[:, inside], full_matrices=False)
        basis = basis[:, singular > settings.rank_rcond * max(singular.max(), 1.0)]
        reduced = by_offset - basis @ (basis.T @ by_offset)
        info += reduced.T @ reduced
    scale = np.asarray(prob.offset_sigmas, float).ravel()
    return info * np.outer(scale, scale)


def _weak_directions(
    information: Array, settings: BundleSettings
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    values, vectors = np.linalg.eigh(information)
    weak = values < settings.weak_information_ratio
    shape = (len(values) // 3, 3, int(weak.sum()))
    loading = (vectors[:, weak] ** 2).reshape(shape).sum(axis=1)
    named = np.flatnonzero((loading >= settings.weak_loading_share).any(axis=1))
    return tuple(int(m) for m in named), tuple(float(v) for v in values[weak])


def _rms(errors: Array, valid: Array, axis: int | None) -> Any:
    total = (np.sum(errors**2, axis=-1) * valid).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(total / valid.sum(axis=axis))


def solve_bundle(prob: BundleProblem, settings: BundleSettings) -> BundleResult:
    """Jointly solve the K poses and M offsets of ``prob``. Preconditions:
    ``BundleProblem.check``; postconditions: ``BundleResult``."""
    if not isinstance(settings, BundleSettings):
        raise TypeError("settings must be a BundleSettings")
    prob.check(settings)
    seeds = np.asarray(prob.offset_seeds, float)
    reach = settings.bound_sigmas * np.asarray(prob.offset_sigmas, float)
    lower, upper = (np.asarray(b, float) for b in prob.pose_bounds)
    poses, offsets, converged = _minimise(
        prob, settings, (lower, upper, seeds - reach, seeds + reach)
    )
    chi2 = [float(t @ t) for t in _terms(prob, settings.marker_sigma_m, poses, offsets)]
    errors = np.array(
        [_marker_errors(prob, k, q, offsets) for k, q in enumerate(poses)]
    )
    valid = np.asarray(prob.valid, bool)
    pinned = np.abs(np.abs(offsets - seeds) - reach) <= settings.bound_atol_m
    weak, ratios = _weak_directions(
        _offset_information(prob, settings, poses, offsets), settings
    )
    return BundleResult(
        offsets=offsets,
        poses=poses,
        rms_m=float(_rms(errors, valid, None)),
        per_marker_rms_m=_rms(errors, valid, 0),
        per_frame_rms_m=_rms(errors, valid, 1),
        data_chi2=chi2[0],
        constraint_chi2=chi2[1],
        offset_prior_chi2=chi2[2],
        pose_prior_chi2=chi2[3],
        on_bound=tuple((int(m), int(a)) for m, a in np.argwhere(pinned)),
        weakly_identified=weak,
        weak_information_ratios=ratios,
        converged=converged,
    )
