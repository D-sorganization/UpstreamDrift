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
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

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
    #: ``least_squares`` stopping tolerance (ftol, xtol and gtol) and budget.
    tolerance: float = 1e-10
    max_evaluations: int = 200

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

    def check(self, settings: BundleSettings) -> None:
        """Raise ``TypeError`` / ``ValueError`` unless the problem is well formed."""
        if not callable(self.fk):
            raise TypeError("fk must be callable: pose -> (n_bodies, 4, 4)")
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
    their seeds and ``poses`` within the pose bounds; the three chi2 terms sum
    to the minimised cost; ``on_bound`` lists ``(marker, axis)``;
    ``weakly_identified`` lists markers that load a data-weak offset direction
    whose information ratios are ``weak_information_ratios`` (ascending)."""

    offsets: Array
    poses: Array
    rms_m: float
    per_marker_rms_m: Array
    per_frame_rms_m: Array
    data_chi2: float
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
        total = prior + self.data_chi2
        return prior / total if total > 0 else 0.0


def _marker_errors(prob: BundleProblem, k: int, pose: Array, offsets: Array) -> Array:
    """Model minus observed marker positions of frame ``k``; zero where invalid."""
    frames = np.asarray(prob.fk(pose))[prob.marker_bodies]
    model = np.einsum("mij,mj->mi", frames[:, :3, :3], offsets) + frames[:, :3, 3]
    return np.where(prob.valid[k][:, None], model - prob.observations[k], 0.0)


def _terms(prob: BundleProblem, sigma: float, x: Array) -> list[Array]:
    """Residual blocks: data, offset prior, pose prior."""
    frames, n_q = prob.pose_start.shape
    poses = x[: frames * n_q].reshape(frames, n_q)
    offsets = x[frames * n_q :].reshape(-1, 3)
    data = [_marker_errors(prob, k, poses[k], offsets) for k in range(frames)]
    prior = (offsets - prob.offset_seeds) / prob.offset_sigmas
    out = [np.ravel(data) / sigma, prior.ravel()]
    if prob.pose_prior is None:
        return [*out, np.zeros(0)]
    mean, pose_sigma = prob.pose_prior
    return [*out, ((poses - mean) / pose_sigma)[np.isfinite(pose_sigma)]]


def _sparsity(prob: BundleProblem) -> lil_matrix:
    """Which residual depends on which unknown (poses first, then offsets)."""
    frames, n_q = prob.pose_start.shape
    size = 3 * len(prob.marker_bodies)
    n_pose, n_data = frames * n_q, frames * size
    sigmas = np.zeros(0) if prob.pose_prior is None else prob.pose_prior[1]
    priors = np.flatnonzero(np.isfinite(sigmas))
    pattern = lil_matrix((n_data + size + len(priors), n_pose + size), dtype=int)
    for k in range(frames):
        pattern[k * size : (k + 1) * size, k * n_q : (k + 1) * n_q] = 1
        for row in range(size):
            first = n_pose + row - row % 3
            pattern[k * size + row, first : first + 3] = 1
    for row, col in enumerate([*range(n_pose, n_pose + size), *priors]):
        pattern[n_data + row, col] = 1
    return pattern


def _offset_information(
    prob: BundleProblem, settings: BundleSettings, poses: Array, offsets: Array
) -> Array:
    """Data information on the offsets with every pose projected out, in units
    of the offset prior's information (``(3M, 3M)``, symmetric).

    Pose priors are left out on purpose: they are what this must expose.
    """
    sigma, step = settings.marker_sigma_m, settings.pose_fd_step
    lower, upper = prob.pose_bounds
    info = np.zeros((offsets.size, offsets.size))
    for k, pose in enumerate(poses):
        base = _marker_errors(prob, k, pose, offsets).ravel()
        free = np.flatnonzero((pose - lower > step) & (upper - pose > step))
        jac_pose = np.zeros((len(base), len(free)))
        for col, j in enumerate(free):
            moved = pose.copy()
            moved[j] += step
            shifted = _marker_errors(prob, k, moved, offsets).ravel()
            jac_pose[:, col] = (shifted - base) / (step * sigma)
        jac_off = np.zeros((len(base), offsets.size))
        rotations = np.asarray(prob.fk(pose))[prob.marker_bodies][:, :3, :3]
        for m in np.flatnonzero(prob.valid[k]):
            jac_off[3 * m : 3 * m + 3, 3 * m : 3 * m + 3] = rotations[m] / sigma
        basis, singular, _ = np.linalg.svd(jac_pose, full_matrices=False)
        basis = basis[:, singular > settings.rank_rcond * max(singular.max(), 1.0)]
        reduced = jac_off - basis @ (basis.T @ jac_off)
        info += reduced.T @ reduced
    scale = np.asarray(prob.offset_sigmas, float).ravel()
    return info * np.outer(scale, scale)


def _weak_directions(
    information: Array, settings: BundleSettings
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    values, vectors = np.linalg.eigh(information)
    weak = values < settings.weak_information_ratio
    loading = (vectors[:, weak] ** 2).reshape(-1, 3, int(weak.sum())).sum(axis=1)
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
    frames, n_q = prob.pose_start.shape
    sigma, tolerance = settings.marker_sigma_m, settings.tolerance

    def residual(x: Array) -> Array:
        return np.concatenate(_terms(prob, sigma, x))

    solution = least_squares(
        residual,
        np.concatenate([np.clip(prob.pose_start, lower, upper).ravel(), seeds.ravel()]),
        jac_sparsity=_sparsity(prob),
        bounds=(
            np.concatenate([np.tile(lower, frames), (seeds - reach).ravel()]),
            np.concatenate([np.tile(upper, frames), (seeds + reach).ravel()]),
        ),
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
        max_nfev=settings.max_evaluations,
    )
    poses = solution.x[: frames * n_q].reshape(frames, n_q)
    offsets = solution.x[frames * n_q :].reshape(-1, 3)
    chi2 = [float(term @ term) for term in _terms(prob, sigma, solution.x)]
    errors = np.array(
        [_marker_errors(prob, k, poses[k], offsets) for k in range(frames)]
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
        offset_prior_chi2=chi2[1],
        pose_prior_chi2=chi2[2],
        on_bound=tuple((int(m), int(a)) for m, a in np.argwhere(pinned)),
        weakly_identified=weak,
        weak_information_ratios=ratios,
        converged=bool(solution.status > 0),
    )
