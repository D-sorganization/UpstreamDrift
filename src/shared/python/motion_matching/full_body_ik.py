"""Engine-agnostic inverse kinematics and marker trajectory tracking (FB-4).

Provides least-squares inverse kinematics solving across motion capture trajectories
given an engine's forward kinematics callable (``pose_fn``) and optional loop-closure
constraint callable (``closure_fn``). Warm-starts sequential frames to guarantee fast
convergence across dense kinematic captures. Consolidates ground-support IK (MS-11 #10330).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.constraint_kinematics import (
    ConstraintLinearization,
    ConstraintOptions,
)
from src.shared.python.motion_matching.marker_calibration import Offsets, Pose
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

Array: TypeAlias = NDArray[np.float64]
PoseFn: TypeAlias = Callable[[Array], Mapping[str, Pose]]
ClosureFn: TypeAlias = Callable[[Array], Array]


@dataclass(frozen=True)
class PoseFit:
    """Result of one pose solve.

    ``iterations`` counts LM loop iterations for the default solver and
    residual evaluations (SciPy ``nfev``) for opt-in TRF.
    """

    q: Array
    marker_rms_m: float
    per_marker_m: dict[str, float]
    closure_error_m: float
    closure_error_rad: float
    lowest_sphere_height_m: float
    iterations: int


@dataclass(frozen=True)
class SolvePoseOptions:
    """Options for single-pose least-squares IK solve."""

    iterations: int = 60
    prior_weight: float = 1e-2
    closure_weight: float = 1e2
    closure_rotation_weight: float | None = None
    ground_weight: float = 1e2
    flat_feet: bool | Sequence[str] = False
    balance_weight: float = 0.0
    bounds: Mapping[str, tuple[float, float]] | None = None
    anchors: Mapping[str, Sequence[float] | Array] | None = None
    damping: float = 1e-4
    locked: Mapping[str, float] | None = None
    tolerance_m: float = 1e-7
    marker_weights: Mapping[str, float] | None = None
    prior_weights: Mapping[str, float] | None = None
    axis_targets: (
        Mapping[str, tuple[Sequence[float], Sequence[float], float]] | None
    ) = None
    com_target: tuple[Sequence[float], float] | None = None
    solver: Literal["lm", "trf"] = "lm"
    #: Pose the joint prior pulls toward; ``None`` uses the seed ``q_init``.
    #: A restart passes its unjittered start here so the jitter only moves
    #: the seed, never the prior (#12042).
    prior_anchor: Sequence[float] | Array | None = None
    #: Weak posture prior (#12042): ``{coordinate: (target_rad, weight)}``
    #: adds ``sqrt(weight) * (q - target)`` rows. A weak weight only decides
    #: coordinates the markers leave under-determined.
    posture_prior: Mapping[str, tuple[float, float]] | None = None


@dataclass(frozen=True)
class SolveTrajectoryOptions:
    """Options for multi-frame trajectory least-squares IK solve."""

    frames: Sequence[int] | None = None
    flat_feet_per_frame: Sequence[Sequence[str]] | None = None
    plant_stance: bool = False
    prior_trajectory: Array | None = None
    restarts: int = 0
    restart_threshold_m: float = 0.0
    restart_spread_rad: float = 0.5
    restart_margin_m: float = 0.0
    axis_targets_per_frame: Sequence[Mapping[str, Any] | None] | None = None
    com_targets_per_frame: Sequence[tuple[Sequence[float], float] | None] | None = None
    locked_per_frame: Sequence[Mapping[str, float] | None] | None = None
    #: Continuity-preserving restarts (#12042). ``None`` keeps the legacy free
    #: restarts. A value keeps each retry's joint prior anchored to the frame's
    #: start pose and accepts a retry only when no joint coordinate (index 6
    #: on) moves further than this many radians from that start pose, so a
    #: restart cannot hop to another branch between two frames.
    restart_max_step_rad: float | None = None
    #: Anchor each retry's joint prior to the frame's start pose instead of
    #: the jittered seed (implied by ``restart_max_step_rad``).
    restart_anchor_prior: bool = False
    #: Draw each frame's restart jitter from a generator seeded by the frame
    #: index, so a perturbation that changes which earlier frames restarted
    #: cannot shift the jitter of every later frame.
    restart_seed_per_frame: bool = False


def _rotation_error(r_a: Array, r_b: Array) -> Array:
    """Small-angle rotation vector taking frame b onto frame a (world axes)."""
    r = r_a @ r_b.T
    return 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])


def _skew3(v: Array) -> Array:
    """Skew-symmetric matrix [v]_x such that [v]_x @ w = v x w."""
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )


def _validate_axis_spec(
    body_axis: Sequence[float], world_dir: Sequence[float], weight: float
) -> tuple[Array, Array]:
    """Validate and normalize axis target vectors."""
    if weight < 0:
        raise ValueError("Axis weights must be nonnegative")
    a = np.asarray(body_axis, dtype=float)
    d = np.asarray(world_dir, dtype=float)
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(d) < 1e-12:
        raise ValueError("Axis targets need nonzero vectors")
    return a / np.linalg.norm(a), d / np.linalg.norm(d)


def _resolve_closure_weights(
    pos_weight: float, rot_weight: float | None
) -> tuple[float, float, bool]:
    """Validate closure weights and return (pos_w, rot_w, should_skip)."""
    resolved_rot = pos_weight if rot_weight is None else rot_weight
    if resolved_rot < 0:
        raise ValueError("Closure rotation weight must be nonnegative")
    skip = pos_weight <= 0 and resolved_rot <= 0
    return pos_weight, resolved_rot, skip


def continuous_branches(
    q: Array,
    coordinate_order: Sequence[str],
    gimbals: Sequence[tuple[str, str, str]] = (),
) -> Array:
    """Return ``q`` with every non-root coordinate unwrapped by 2 pi and each
    XYZ gimbal (``Rx Ry Rz`` triple) on the Euler branch nearest previous frame.
    """
    out = np.asarray(q, dtype=float).copy()
    if out.ndim != 2 or out.shape[1] != len(coordinate_order):
        raise ValueError("q must be (frames, coordinates) matching the order")
    for triple in gimbals:
        if any(name not in coordinate_order for name in triple):
            raise ValueError(f"Unknown gimbal coordinates {triple}")
    out[:, 6:] = np.unwrap(out[:, 6:], axis=0)
    for triple in gimbals:
        ix, iy, iz = (coordinate_order.index(name) for name in triple)
        for k in range(1, out.shape[0]):
            prev = out[k - 1, [ix, iy, iz]]
            here = out[k, [ix, iy, iz]]
            alt = np.array([here[0] + np.pi, np.pi - here[1], here[2] + np.pi])
            alt = prev + (alt - prev + np.pi) % (2 * np.pi) - np.pi
            here = prev + (here - prev + np.pi) % (2 * np.pi) - np.pi
            best = alt if np.abs(alt - prev).sum() < np.abs(here - prev).sum() else here
            out[k, [ix, iy, iz]] = best
    return out


def _run_lm_loop(
    residual_fn: Callable[[Array], tuple[Array, Array]],
    q_init: Array,
    free: NDArray[np.bool_],
    low: Array,
    high: Array,
    *,
    iterations: int = 60,
    damping: float = 1e-4,
    tolerance_m: float = 1e-7,
) -> tuple[Array, int]:
    """Levenberg-Marquardt optimizer loop for pose fitting."""
    q = q_init.copy()
    residual, jacobian = residual_fn(q)
    cost = float(residual @ residual)
    lam = max(damping, 1e-12)
    done = 0
    while done < iterations:
        done += 1
        gram = jacobian.T @ jacobian
        diag = np.diag(np.diag(gram) + 1e-9)
        step = np.linalg.solve(gram + lam * diag, -jacobian.T @ residual)
        trial = q.copy()
        trial[free] += step
        trial = np.clip(trial, low, high)
        trial_residual, trial_jacobian = residual_fn(trial)
        trial_cost = float(trial_residual @ trial_residual)
        if trial_cost < cost:
            improvement = cost - trial_cost
            q, residual, jacobian, cost = (
                trial,
                trial_residual,
                trial_jacobian,
                trial_cost,
            )
            lam = max(lam / 3.0, 1e-12)
            if improvement < tolerance_m**2 and np.linalg.norm(step) < 1e-9:
                break
        else:
            lam *= 10.0
            if lam > 1e12:
                break
    return q, done


def _run_trf_loop(
    residual_fn: Callable[[Array], tuple[Array, Array]],
    q_init: Array,
    free: NDArray[np.bool_],
    low: Array,
    high: Array,
    *,
    iterations: int,
) -> tuple[Array, int]:
    """Bounded trust-region solve on free DOFs, preserving every locked value."""
    if not np.any(free):
        residual_fn(q_init)
        return q_init.copy(), 0
    cached_x: Array | None = None
    cached_pair: tuple[Array, Array] | None = None

    def evaluate(x: Array) -> tuple[Array, Array]:
        nonlocal cached_x, cached_pair
        if cached_x is None or not np.array_equal(x, cached_x):
            pose = q_init.copy()
            pose[free] = x
            cached_pair = residual_fn(pose)
            cached_x = x.copy()
        if cached_pair is None:
            raise RuntimeError("TRF residual evaluation produced no result")
        return cached_pair

    def _residual_fn(x: np.ndarray) -> np.ndarray:
        return evaluate(x)[0]

    def _jacobian_fn(x: np.ndarray) -> np.ndarray:
        return evaluate(x)[1]

    result = least_squares(
        _residual_fn,
        q_init[free],
        jac=_jacobian_fn,
        bounds=(low[free], high[free]),
        method="trf",
        x_scale="jac",
        # A seed on an active bound starts with a tiny trust region. Cost-only
        # stopping would mistake its first tiny improving step for convergence.
        ftol=None,
        max_nfev=iterations,
    )
    pose = q_init.copy()
    pose[free] = result.x
    return pose, int(result.nfev)


def solve_full_body_ik_trajectory(
    pose_fn: PoseFn,
    offsets: Offsets,
    capture: TourCapture,
    initial_q: Array,
    *,
    closure_fn: ClosureFn | None = None,
    closure_weight: float = 10.0,
    reg_weight: float = 1e-3,
    max_nfev: int = 50,
    coordinate_bounds: tuple[Array, Array] | None = None,
) -> Array:
    """Solve trajectory IK; explicit finite source bounds opt into bounded TRF.

    Omission preserves the original LM solver. Bounds never clip the seed or
    widen source anatomy. Finite output is not an optimization acceptance test.
    """
    q0 = np.asarray(initial_q, dtype=float)
    if q0.ndim != 1 or not np.isfinite(q0).all():
        raise ValueError("initial_q must be a finite 1D array")
    if capture.frames < 1:
        raise ValueError("capture must contain at least 1 frame")
    missing = [label for label in capture.labels if label not in offsets]
    if missing:
        raise ValueError(f"Capture labels missing from offsets: {missing}")
    if closure_weight < 0:
        raise ValueError("closure_weight must be non-negative")
    if reg_weight < 0:
        raise ValueError("reg_weight must be non-negative")
    low, high = _trajectory_bounds(q0, coordinate_bounds)

    n_coords = q0.size
    q_out = np.zeros((capture.frames, n_coords), dtype=float)
    q_curr = q0.copy()

    marker_indices = list(range(len(capture.labels)))
    marker_bodies = [offsets[label][0] for label in capture.labels]
    marker_offsets = [
        np.asarray(offsets[label][1], dtype=float) for label in capture.labels
    ]

    for f in range(capture.frames):
        valid_indices = [i for i in marker_indices if capture.valid[f, i]]
        if not valid_indices:
            q_out[f] = q_curr
            continue

        target_points = capture.points_m[f]
        q_ref = q_curr.copy()

        def residual(
            q_eval: Array,
            v_idx: list[int] = valid_indices,
            targets: Array = target_points,
            ref_q: Array = q_ref,
        ) -> Array:
            poses = pose_fn(q_eval)
            diffs = []
            for i in v_idx:
                body = marker_bodies[i]
                offset = marker_offsets[i]
                r, t = poses[body]
                p_pred = r @ offset + t
                diffs.append(p_pred - targets[i])

            res = np.concatenate(diffs)
            if closure_fn is not None and closure_weight > 0.0:
                closure_err = closure_fn(q_eval)
                res = np.concatenate([res, closure_weight * closure_err])
            if reg_weight > 0.0:
                res = np.concatenate([res, reg_weight * (q_eval - ref_q)])
            return res

        sol = least_squares(
            residual,
            q_curr,
            method="lm" if coordinate_bounds is None else "trf",
            bounds=(low, high),
            max_nfev=max_nfev,
            # As in _run_trf_loop, a tiny first step from a bound is not convergence.
            ftol=1e-8 if coordinate_bounds is None else None,
        )
        if np.isfinite(sol.x).all():
            q_curr = sol.x.copy()
        q_out[f] = q_curr

    return q_out


def _trajectory_bounds(
    initial_q: Array,
    bounds: tuple[Array, Array] | None,
) -> tuple[Array, Array]:
    """Validate explicit finite bounds before any engine geometry evaluation."""
    if bounds is None:
        return np.full_like(initial_q, -np.inf), np.full_like(initial_q, np.inf)
    low, high = (np.asarray(values, dtype=float) for values in bounds)
    if (
        low.shape != initial_q.shape
        or high.shape != initial_q.shape
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or np.any(low >= high)
        or np.any(initial_q < low)
        or np.any(initial_q > high)
    ):
        raise ValueError(
            "Coordinate bounds must be finite, ordered and contain initial_q"
        )
    return low.copy(), high.copy()


def compute_marker_rms_trajectory(
    pose_fn: PoseFn,
    offsets: Offsets,
    capture: TourCapture,
    q_trajectory: Array,
) -> tuple[Array, dict[str, float], float]:
    """Compute per-frame, per-marker, and overall RMS errors for a solved trajectory."""
    q_traj = np.asarray(q_trajectory, dtype=float)
    if q_traj.shape[0] != capture.frames or q_traj.ndim != 2:
        raise ValueError("q_trajectory rows must match capture frame count")

    per_frame_errors: list[float] = []
    all_errors: list[float] = []
    per_marker_errors: dict[str, list[float]] = {label: [] for label in capture.labels}

    for f in range(capture.frames):
        poses = pose_fn(q_traj[f])
        frame_sq_errors: list[float] = []
        for i, label in enumerate(capture.labels):
            if capture.valid[f, i]:
                body, offset = offsets[label]
                r, t = poses[body]
                p_pred = r @ np.asarray(offset, dtype=float) + t
                err = float(np.linalg.norm(p_pred - capture.points_m[f, i]))
                all_errors.append(err)
                frame_sq_errors.append(err**2)
                per_marker_errors[label].append(err)

        if frame_sq_errors:
            per_frame_errors.append(float(np.sqrt(np.mean(frame_sq_errors))))
        else:
            per_frame_errors.append(0.0)

    per_marker_rms = {
        label: float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0
        for label, errs in per_marker_errors.items()
    }
    total_rms = float(np.sqrt(np.mean(np.square(all_errors)))) if all_errors else 0.0

    return np.asarray(per_frame_errors, dtype=float), per_marker_rms, total_rms


def parse_specification(
    specification: Mapping[str, Any] | bytes | str,
) -> dict[str, Any]:
    """Normalize specification input to a dictionary."""
    if isinstance(specification, bytes):
        return json.loads(specification.decode("utf-8"))
    if isinstance(specification, str):
        return json.loads(specification)
    return dict(specification)


def get_marker_bodies(specification: Mapping[str, Any]) -> list[str]:
    """Extract sorted list of unique body names referenced by marker attachments."""
    marker_attachments = specification.get("marker_attachments", {})
    return sorted({v["body"] for v in marker_attachments.values() if "body" in v})


@dataclass(frozen=True)
class _PosePrep:
    mask: NDArray[np.bool_]
    pinned: frozenset[str]
    planted: dict[str, Array]
    com_goal: tuple[Array, float] | None
    row_scale: Array
    q: Array
    low: Array
    high: Array
    free: NDArray[np.bool_]
    nv: int
    sqrt_prior: Array
    axes: list[Any]
    anchor: Array
    posture: tuple[NDArray[np.intp], Array, Array]


def _prior_anchor(q_init: Array, anchor: Sequence[float] | Array | None) -> Array:
    """Pose the joint prior pulls toward: ``anchor`` if given, else ``q_init``.

    Raises ``ValueError`` when ``anchor`` is not a finite vector shaped like
    ``q_init``.
    """
    seed = np.asarray(q_init, dtype=float)
    if anchor is None:
        return seed
    target = np.asarray(anchor, dtype=float)
    if target.shape != seed.shape or not np.isfinite(target).all():
        raise ValueError(
            f"prior_anchor must be a finite vector of shape {seed.shape}, "
            f"got {target.shape}"
        )
    return target


def _restart_admissible(
    retry_q: Array, start: Array, max_step_rad: float | None
) -> bool:
    """True when a restart may replace the frame's fit (#12042).

    With ``max_step_rad`` set, no joint coordinate (index 6 on; 0-5 are the
    floating root) may move further than ``max_step_rad`` from ``start``.
    """
    if max_step_rad is None:
        return True
    step = np.abs(np.asarray(retry_q, dtype=float)[6:] - np.asarray(start)[6:])
    return bool(step.size == 0 or float(step.max()) <= max_step_rad)


class BaseFullBodyIK:
    """Base class for engine-specific full-body IK adapters."""

    supports_trf: bool = False

    def __init__(
        self,
        specification: Mapping[str, Any] | bytes | str | None = None,
        *,
        coordinate_order: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
    ) -> None:
        if specification is not None:
            self.specification: dict[str, Any] = parse_specification(specification)
            self.coordinate_order: tuple[str, ...] = tuple(
                self.specification.get("coordinate_order", [])
            )
            self.marker_bodies: list[str] = get_marker_bodies(self.specification)
        else:
            self.specification = {}
            self.coordinate_order = tuple(coordinate_order or ())
            self.marker_bodies = []
        self.labels: tuple[str, ...] = tuple(labels or ())

    def pose_fn(self, q: Array) -> dict[str, Pose]:
        """Compute world poses (R, t) for all bodies referenced by markers."""
        raise NotImplementedError  # tracked: #10330

    def closure_residuals(self, q: Array) -> Array:
        """Evaluate position residual between dual-grip weld frames/sites in world."""
        raise NotImplementedError  # tracked: #10330

    def constraint_residual_jacobian(
        self, q: Array, options: ConstraintOptions
    ) -> ConstraintLinearization:
        """Linearize declared geometry, or explicitly reject unsupported engines."""
        raise NotImplementedError(  # tracked: #10330
            "Native constraint linearization is unsupported"
        )

    def _set(self, q: Array) -> None:
        """Set generalized coordinates on underlying physics model."""
        raise NotImplementedError  # tracked: #10330

    def _positions(self) -> Array:
        """Forward kinematics returning marker positions as (M, 3) array."""
        raise NotImplementedError  # tracked: #10330

    def _marker_jacobian(self, positions: Array) -> Array:
        """Jacobian of marker positions with respect to generalized coordinates."""
        raise NotImplementedError  # tracked: #10330

    def _axis_rows(self, targets: Any) -> list[Any]:
        """Convert axis targets into residual rows and Jacobians."""
        return []

    def _append_closure(
        self,
        rows: list[Array],
        jacs: list[Array],
        pos_weight: float,
        rot_weight: float,
    ) -> None:
        """Append dual-grip weld loop closure residual and Jacobian rows."""

    def _append_ground(
        self,
        rows: list[Array],
        jacs: list[Array],
        ground: Any,
        weight: float,
        pinned: Any,
    ) -> None:
        """Append ground penetration avoidance penalty rows."""

    def _append_anchors(
        self,
        rows: list[Array],
        jacs: list[Array],
        planted: Any,
        weight: float,
    ) -> None:
        """Append planted stance foot anchor rows."""

    def _append_balance(
        self,
        rows: list[Array],
        jacs: list[Array],
        ground: Any,
        weight: float,
    ) -> None:
        """Append center of mass balance projection rows."""

    def _append_com_target(
        self,
        rows: list[Array],
        jacs: list[Array],
        ground: Any,
        com_goal: Any,
    ) -> None:
        """Append center of mass target tracking rows."""

    def _append_axes(
        self,
        rows: list[Array],
        jacs: list[Array],
        axes: Any,
    ) -> None:
        """Append directional axis alignment rows."""

    def _sphere_heights(self, ground: Any) -> dict[str, float]:
        """Return heights of foot contact spheres above ground."""
        return {}

    def closure_error(self, q: Array) -> tuple[float, float]:
        """Return (position_error_m, rotation_error_rad) for dual-grip closure."""
        return 0.0, 0.0

    def sphere_ground_points(self, q: Array, ground: Any) -> dict[str, Array]:
        """Return ground contact sphere points at configuration q."""
        return {}

    def ik_fn(
        self,
        offsets: Offsets,
        capture: TourCapture,
        initial_q: Array | None = None,
        *,
        closure_weight: float = 10.0,
        max_nfev: int = 50,
    ) -> Array:
        """Solve least-squares IK across all frames of the capture."""
        initial: Array
        if initial_q is None:
            initial = np.zeros(len(self.coordinate_order), dtype=float)
            waist_labels = ("WaistLeft", "WaistRight", "WaistLBack", "WaistRBack")
            if all(lbl in capture.labels for lbl in waist_labels):
                indices = [capture.index(lbl) for lbl in waist_labels]
                centroid = np.nanmean(capture.points_m[0, indices], axis=0)
                initial[0] = centroid[0]
                initial[1] = centroid[1]
                initial[2] = centroid[2]
        else:
            initial = np.asarray(initial_q, dtype=float)

        return solve_full_body_ik_trajectory(
            self.pose_fn,
            offsets,
            capture,
            initial,
            closure_fn=self.closure_residuals,
            closure_weight=closure_weight,
            max_nfev=max_nfev,
        )

    def evaluate_trajectory_rms(
        self,
        offsets: Offsets,
        capture: TourCapture,
        q_trajectory: Array,
    ) -> tuple[Array, dict[str, float], float]:
        """Compute per-frame, per-marker, and overall RMS errors."""
        return compute_marker_rms_trajectory(
            self.pose_fn, offsets, capture, q_trajectory
        )

    # -- Ground-support IK solver extensions (MS-11 #10330) ------------------

    @staticmethod
    def _plane_basis(ground: GroundPlane) -> Array:
        """Two orthonormal in-plane axes (rows) of the ground plane."""
        n = np.asarray(ground.normal, dtype=float)
        n = n / np.linalg.norm(n)
        return np.asarray(np.linalg.svd(np.eye(3) - np.outer(n, n))[0][:, :2].T)

    def _pinned_spheres(self, flat_feet: bool | Sequence[str]) -> frozenset[str]:
        spheres = getattr(self, "_spheres", {})
        if isinstance(flat_feet, bool):
            return frozenset(spheres) if flat_feet else frozenset()
        names = frozenset(flat_feet)
        unknown = names - set(spheres)
        if unknown:
            raise ValueError(f"Unknown contact spheres: {sorted(unknown)}")
        return names

    def _bounds(
        self, bounds: Mapping[str, tuple[float, float]] | None
    ) -> tuple[Array, Array]:
        low = np.full(len(self.coordinate_order), -np.inf)
        high = np.full(len(self.coordinate_order), np.inf)
        for name, (lo, hi) in (bounds or {}).items():
            if name not in self.coordinate_order:
                raise ValueError(f"Unknown bounded coordinate {name}")
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
                raise ValueError(f"Bounds for {name} must be finite with low < high")
            index = self.coordinate_order.index(name)
            low[index], high[index] = float(lo), float(hi)
        return low, high

    def _anchor_targets(
        self, anchors: Mapping[str, Sequence[float] | Array] | None, ground: GroundPlane
    ) -> dict[str, Array]:
        if not anchors:
            return {}
        spheres = getattr(self, "_spheres", {})
        n = np.asarray(ground.normal, dtype=float)
        n = n / np.linalg.norm(n)
        targets: dict[str, Array] = {}
        for name, point in anchors.items():
            if name not in spheres:
                raise ValueError(f"Unknown contact sphere to anchor: {name}")
            p = np.asarray(point, dtype=float)
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError(f"Anchor for {name} must be a finite 3-vector")
            radius = spheres[name][1]
            targets[name] = p - (p @ n - ground.height_m) * n + radius * n
        return targets

    def _prepare_pose_fit(
        self,
        targets: Array,
        valid: NDArray[Any],
        q_init: Array,
        ground: GroundPlane,
        opts: SolvePoseOptions,
    ) -> _PosePrep:
        targets_arr = np.asarray(targets, dtype=float)
        mask = np.asarray(valid, dtype=bool)
        if targets_arr.shape != (len(self.labels), 3) or mask.shape != (
            len(self.labels),
        ):
            raise ValueError("Targets must be (markers, 3) with a validity vector")
        mask = mask & np.isfinite(targets_arr).all(axis=1)
        pinned = self._pinned_spheres(opts.flat_feet)
        planted = self._anchor_targets(opts.anchors, ground)
        pinned = pinned - frozenset(planted)
        if mask.sum() < 3 and not pinned:
            raise ValueError("At least three valid markers are required")
        if (
            opts.iterations < 1
            or min(opts.prior_weight, opts.closure_weight, opts.ground_weight) < 0
        ):
            raise ValueError("Iterations must be positive and weights nonnegative")
        if opts.balance_weight < 0:
            raise ValueError("Iterations must be positive and weights nonnegative")
        com_goal: tuple[Array, float] | None = None
        if opts.com_target is not None:
            goal = np.asarray(opts.com_target[0], dtype=float)
            if (
                goal.shape != (2,)
                or not np.isfinite(goal).all()
                or opts.com_target[1] < 0
            ):
                raise ValueError(
                    "com_target needs a finite plane point and weight >= 0"
                )
            com_goal = (goal, float(opts.com_target[1]))
        sqrt_marker = np.ones(len(self.labels))
        for label, weight in (opts.marker_weights or {}).items():
            if label not in self.labels:
                raise ValueError(f"Unknown marker {label}")
            if weight < 0:
                raise ValueError("Marker weights must be nonnegative")
            sqrt_marker[self.labels.index(label)] = np.sqrt(weight)
        row_scale = np.repeat(sqrt_marker[mask], 3)
        q = np.asarray(q_init, dtype=float).copy()
        low, high = self._bounds(opts.bounds)
        q = np.clip(q, low, high)
        free = np.ones(len(q), dtype=bool)
        for name, value in (opts.locked or {}).items():
            if name not in self.coordinate_order:
                raise ValueError(f"Unknown locked coordinate {name}")
            index = self.coordinate_order.index(name)
            q[index] = float(value)
            free[index] = False
        model_obj: Any = getattr(self, "model", None) or getattr(self, "adapter", None)
        nv: int = (
            getattr(model_obj, "nv", len(self.coordinate_order))
            if model_obj is not None
            else len(self.coordinate_order)
        )
        prior_diag = np.full(nv, float(opts.prior_weight))
        for name, weight in (opts.prior_weights or {}).items():
            if name not in self.coordinate_order:
                raise ValueError(f"Unknown prior coordinate {name}")
            if weight < 0:
                raise ValueError("Prior weights must be nonnegative")
            prior_diag[self.coordinate_order.index(name)] = float(weight)
        return _PosePrep(
            mask=mask,
            pinned=pinned,
            planted=planted,
            com_goal=com_goal,
            row_scale=row_scale,
            q=q,
            low=low,
            high=high,
            free=free,
            nv=nv,
            sqrt_prior=np.sqrt(prior_diag),
            axes=self._axis_rows(opts.axis_targets),
            anchor=_prior_anchor(q_init, opts.prior_anchor),
            posture=self._posture_rows(opts.posture_prior),
        )

    def _posture_rows(
        self, prior: Mapping[str, tuple[float, float]] | None
    ) -> tuple[NDArray[np.intp], Array, Array]:
        """Indices, targets and square-root weights of the posture prior."""
        entries: Mapping[str, tuple[float, float]] = prior or {}
        names = list(entries)
        for name in names:
            if name not in self.coordinate_order:
                raise ValueError(f"Unknown posture prior coordinate {name}")
        values = np.asarray([entries[n] for n in names], dtype=np.float64).reshape(
            -1, 2
        )
        if not np.isfinite(values).all() or (values[:, 1] < 0).any():
            raise ValueError("Posture prior targets must be finite, weights >= 0")
        index = np.array([self.coordinate_order.index(n) for n in names], dtype=int)
        return index, values[:, 0], np.sqrt(values[:, 1])

    def _pose_residual_stack(
        self,
        q_k: Array,
        prep: _PosePrep,
        targets_arr: Array,
        q_init: Array,
        ground: GroundPlane,
        opts: SolvePoseOptions,
    ) -> tuple[Array, Array]:
        """Marker, prior, closure, ground, and balance residuals for one pose."""
        self._set(q_k)
        positions = self._positions()
        jac = self._marker_jacobian(positions)
        rows = [
            prep.row_scale * (positions[prep.mask] - targets_arr[prep.mask]).reshape(-1)
        ]
        jacs = [prep.row_scale[:, None] * jac[prep.mask].reshape(-1, prep.nv)]
        rows.append(prep.sqrt_prior * (q_k - q_init))
        jacs.append(prep.sqrt_prior * np.eye(prep.nv))
        index, target, sqrt_w = prep.posture
        if index.size:
            rows.append(sqrt_w * (q_k[index] - target))
            jacs.append(sqrt_w[:, None] * np.eye(prep.nv)[index])
        rot_w = (
            opts.closure_weight
            if opts.closure_rotation_weight is None
            else opts.closure_rotation_weight
        )
        self._append_closure(rows, jacs, opts.closure_weight, rot_w)
        self._append_ground(rows, jacs, ground, opts.ground_weight, prep.pinned)
        self._append_anchors(rows, jacs, prep.planted, opts.ground_weight)
        self._append_balance(rows, jacs, ground, opts.balance_weight)
        self._append_com_target(rows, jacs, ground, prep.com_goal)
        self._append_axes(rows, jacs, prep.axes)
        return np.concatenate(rows), np.concatenate(jacs)[:, prep.free]

    def _finalize_pose_fit(
        self,
        q: Array,
        targets_arr: Array,
        prep: _PosePrep,
        *,
        ground: GroundPlane,
        iterations: int,
    ) -> PoseFit:
        """Build a ``PoseFit`` from a solved configuration."""
        self._set(q)
        diff = self._positions() - targets_arr
        errors = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        rms = (
            float(np.sqrt(np.mean(errors[prep.mask] ** 2))) if prep.mask.any() else 0.0
        )
        per_marker = {
            label: float(errors[k])
            for k, label in enumerate(self.labels)
            if prep.mask[k]
        }
        heights = self._sphere_heights(ground)
        pos_err, rot_err = self.closure_error(q)
        return PoseFit(
            q=q,
            marker_rms_m=rms,
            per_marker_m=per_marker,
            closure_error_m=pos_err,
            closure_error_rad=rot_err,
            lowest_sphere_height_m=min(heights.values()),
            iterations=iterations,
        )

    def solve_pose(
        self,
        targets: Array,
        valid: NDArray[Any],
        q_init: Array,
        *,
        ground: GroundPlane,
        options: SolvePoseOptions | None = None,
        **kwargs: Any,
    ) -> PoseFit:
        """Least-squares pose for one frame of marker targets."""
        opts = SolvePoseOptions(**kwargs) if options is None else options
        if opts.solver not in ("lm", "trf"):
            raise ValueError("Pose solver must be 'lm' or 'trf'")
        if opts.solver == "trf" and not self.supports_trf:
            raise ValueError("Provider does not support fixed-row TRF pose solving")
        targets = np.asarray(targets, dtype=float)
        prep = self._prepare_pose_fit(targets, valid, q_init, ground, opts)
        if opts.solver == "trf" and (
            not np.isfinite(prep.q).all()
            or np.any(prep.q < prep.low)
            or np.any(prep.q > prep.high)
        ):
            raise ValueError(
                "Locked coordinates must be finite and inside authored bounds"
            )

        def residuals(q_k: Array) -> tuple[Array, Array]:
            return self._pose_residual_stack(
                q_k, prep, targets, prep.anchor, ground, opts
            )

        if opts.solver == "trf":
            q, done = _run_trf_loop(
                residuals,
                prep.q,
                prep.free,
                prep.low,
                prep.high,
                iterations=opts.iterations,
            )
            return self._finalize_pose_fit(
                q, targets, prep, ground=ground, iterations=done
            )
        q, done = _run_lm_loop(
            residuals,
            prep.q,
            prep.free,
            prep.low,
            prep.high,
            iterations=opts.iterations,
            damping=opts.damping,
            tolerance_m=opts.tolerance_m,
        )
        return self._finalize_pose_fit(q, targets, prep, ground=ground, iterations=done)

    def _validate_trajectory_options(
        self,
        targets: Array,
        valid: NDArray[Any],
        opts: SolveTrajectoryOptions,
    ) -> tuple[Array, NDArray[np.bool_], list[int], Array | None]:
        targets_arr = np.asarray(targets, dtype=float)
        mask = np.asarray(valid, dtype=bool)
        if targets_arr.ndim != 3 or mask.shape != targets_arr.shape[:2]:
            raise ValueError("Trajectory targets must be (frames, markers, 3)")
        indices = (
            list(range(targets_arr.shape[0]))
            if opts.frames is None
            else list(opts.frames)
        )
        if (
            opts.flat_feet_per_frame is not None
            and len(opts.flat_feet_per_frame) != targets_arr.shape[0]
        ):
            raise ValueError("flat_feet_per_frame needs one entry per capture frame")
        if opts.plant_stance and opts.flat_feet_per_frame is None:
            raise ValueError("plant_stance needs flat_feet_per_frame")
        prior: Array | None = None
        if opts.prior_trajectory is not None:
            prior = np.asarray(opts.prior_trajectory, dtype=float)
            if prior.shape != (targets_arr.shape[0], len(self.coordinate_order)):
                raise ValueError("prior_trajectory must be (frames, coordinates)")
        if (
            min(
                opts.restarts,
                opts.restart_threshold_m,
                opts.restart_spread_rad,
                opts.restart_margin_m,
            )
            < 0
        ):
            raise ValueError("Restart settings must be nonnegative")
        if opts.restart_max_step_rad is not None and not (
            np.isfinite(opts.restart_max_step_rad) and opts.restart_max_step_rad > 0
        ):
            raise ValueError("restart_max_step_rad must be finite and positive")
        if (
            opts.axis_targets_per_frame is not None
            and len(opts.axis_targets_per_frame) != targets_arr.shape[0]
        ):
            raise ValueError("axis_targets_per_frame needs one entry per capture frame")
        if (
            opts.com_targets_per_frame is not None
            and len(opts.com_targets_per_frame) != targets_arr.shape[0]
        ):
            raise ValueError("com_targets_per_frame needs one entry per capture frame")
        if (
            opts.locked_per_frame is not None
            and len(opts.locked_per_frame) != targets_arr.shape[0]
        ):
            raise ValueError("locked_per_frame needs one entry per capture frame")
        return targets_arr, mask, indices, prior

    def solve_trajectory(
        self,
        targets: Array,
        valid: NDArray[Any],
        q_init: Array,
        *,
        ground: GroundPlane,
        options: SolveTrajectoryOptions | None = None,
        **kwargs: Any,
    ) -> tuple[Array, list[PoseFit]]:
        """Solve consecutive frames, each warm-started from the previous one."""
        traj_field_names = {f.name for f in fields(SolveTrajectoryOptions)}
        traj_kwargs = {k: v for k, v in kwargs.items() if k in traj_field_names}
        frame_kwargs = {k: v for k, v in kwargs.items() if k not in traj_field_names}
        opts = SolveTrajectoryOptions(**traj_kwargs) if options is None else options
        targets, mask, indices, prior = self._validate_trajectory_options(
            targets, valid, opts
        )

        rng = np.random.default_rng(0)
        q = np.asarray(q_init, dtype=float)
        fits: list[PoseFit] = []
        anchors: dict[str, Array] = {}

        for k in indices:
            if opts.restart_seed_per_frame:
                rng = np.random.default_rng((0, k))
            stance = (
                ()
                if opts.flat_feet_per_frame is None
                else tuple(opts.flat_feet_per_frame[k])
            )
            if opts.plant_stance:
                anchors = {name: pt for name, pt in anchors.items() if name in stance}
            start = q if prior is None else prior[k]
            f_opts = dict(frame_kwargs)
            if opts.axis_targets_per_frame is not None:
                f_opts["axis_targets"] = opts.axis_targets_per_frame[k]
            if opts.com_targets_per_frame is not None:
                f_opts["com_target"] = opts.com_targets_per_frame[k]
            if opts.locked_per_frame is not None:
                f_opts["locked"] = opts.locked_per_frame[k]

            fit = self.solve_pose(
                targets[k],
                mask[k],
                start,
                ground=ground,
                flat_feet=stance if opts.flat_feet_per_frame is not None else False,
                anchors=anchors if opts.plant_stance else None,
                **f_opts,
            )
            anchored = (
                opts.restart_anchor_prior or opts.restart_max_step_rad is not None
            )
            retry_opts = (
                {**f_opts, "prior_anchor": np.asarray(start, dtype=float)}
                if anchored
                else f_opts
            )
            for _ in range(
                opts.restarts if fit.marker_rms_m > opts.restart_threshold_m else 0
            ):
                jittered = np.asarray(start, dtype=float).copy()
                jittered[6:] += rng.uniform(
                    -opts.restart_spread_rad, opts.restart_spread_rad, len(jittered) - 6
                )
                retry = self.solve_pose(
                    targets[k],
                    mask[k],
                    jittered,
                    ground=ground,
                    flat_feet=stance if opts.flat_feet_per_frame is not None else False,
                    anchors=anchors if opts.plant_stance else None,
                    **retry_opts,
                )
                if retry.marker_rms_m < fit.marker_rms_m - opts.restart_margin_m and (
                    _restart_admissible(retry.q, start, opts.restart_max_step_rad)
                ):
                    fit = retry
            if opts.plant_stance:
                points = self.sphere_ground_points(fit.q, ground)
                for name in stance:
                    anchors.setdefault(name, points[name])
            fits.append(fit)
            q = fit.q
        return np.array([fit.q for fit in fits]), fits
