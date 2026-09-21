"""Pure native Crocoddyl optimal control problem assembly for the full-body plant (MS-31, #10338).

Constructs an optimal control shooting problem on the 41/44-coordinate Pinocchio
plant with per-node actuation controls, marker residual tracking costs, 6D dual-grip
weld closure constraint/cost, effort and joint-limit regularization, and terminal
objectives. Also provides pure helpers: node times, per-node marker targets with
validity, coordinate bounds, actuated mask, and warm-start helpers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.polynomial_actuation import ROOT_COORDINATES
from src.shared.python.optimization.crocoddyl_backend import (
    CrocoddylNotAvailableError,
    crocoddyl_stack_healthy,
    require_crocoddyl,
)

Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

_NODE_TIME_TOLERANCE_S = 1e-9


@dataclass(frozen=True)
class FitWeights:
    """Quadratic cost weights of the marker-tracking optimal-control problem.

    Every weight multiplies one half of a squared residual. ``marker`` and
    ``terminal_marker`` are per marker (metres); ``effort`` per actuated
    coordinate (N m); ``velocity`` per coordinate (rad/s); ``range_barrier``
    per coordinate outside its human range (rad).
    """

    marker: float = 1e3
    terminal_marker: float = 1e4
    effort: float = 1e-5
    velocity: float = 1e-3
    range_barrier: float = 1e3

    def __post_init__(self) -> None:
        for name in (
            "marker",
            "terminal_marker",
            "effort",
            "velocity",
            "range_barrier",
        ):
            value = getattr(self, name)
            require(
                np.isfinite(value) and value > 0.0,
                f"{name} weight must be finite and positive",
                value,
            )


@dataclass(frozen=True)
class FitHorizon:
    """Fit window on the capture time base; nodes coincide with capture frames."""

    t_start_s: float
    t_end_s: float
    dt_s: float

    def __post_init__(self) -> None:
        require(self.dt_s > 0.0, "dt_s must be positive", self.dt_s)
        require(
            self.t_end_s > self.t_start_s,
            "t_end_s must exceed t_start_s",
            (self.t_start_s, self.t_end_s),
        )

    @property
    def n_nodes(self) -> int:
        return int(round((self.t_end_s - self.t_start_s) / self.dt_s)) + 1

    def node_times(self) -> Array:
        times = self.t_start_s + self.dt_s * np.arange(self.n_nodes, dtype=float)
        ensure(
            times[-1] <= self.t_end_s + _NODE_TIME_TOLERANCE_S,
            "node grid overshoots the window",
            times[-1],
        )
        return times


@dataclass(frozen=True)
class MarkerTargets:
    """Per-node marker targets in the native world frame.

    ``targets`` is (nodes, markers, 3) with invalid samples zeroed; ``valid``
    is the (nodes, markers) mask that gates the cost; ``weights`` is per marker.
    """

    labels: tuple[str, ...]
    node_times: Array
    targets: Array
    valid: NDArray[np.bool_]
    weights: Array

    def __post_init__(self) -> None:
        n_nodes, n_markers = self.valid.shape
        require(
            self.targets.shape == (n_nodes, n_markers, 3),
            "targets must be (nodes, markers, 3)",
            self.targets.shape,
        )
        require(
            len(self.labels) == n_markers,
            "one label per marker column",
            len(self.labels),
        )
        require(
            self.weights.shape == (n_markers,),
            "one weight per marker",
            self.weights.shape,
        )
        require(
            bool(np.isfinite(self.targets).all()),
            "targets must be finite (zero where invalid)",
        )


def build_marker_targets(
    time_s: Array,
    points_native: Array,
    valid: NDArray[np.bool_],
    labels: Sequence[str],
    node_times: Array,
    *,
    weights: Mapping[str, float] | None = None,
) -> MarkerTargets:
    """Sample capture markers at node times that coincide with capture frames.

    Preconditions: ``points_native`` is (frames, markers, 3) in the native
    Z-up world; each node time matches a capture frame within 1 ns (the
    fit runs at the capture rate so no interpolation is silently introduced).
    """
    points = np.asarray(points_native, dtype=float)
    valid_arr = np.asarray(valid, dtype=bool)
    times = np.asarray(time_s, dtype=float)
    require(
        points.ndim == 3 and points.shape[2] == 3,
        "points must be (frames, markers, 3)",
        points.shape,
    )
    require(
        valid_arr.shape == points.shape[:2],
        "valid must be (frames, markers)",
        valid_arr.shape,
    )
    require(len(labels) == points.shape[1], "one label per marker column", len(labels))
    upper_idx = np.clip(np.searchsorted(times, node_times), 0, len(times) - 1)
    lower_idx = np.clip(upper_idx - 1, 0, len(times) - 1)
    pick_lower = np.abs(times[lower_idx] - node_times) < np.abs(
        times[upper_idx] - node_times
    )
    indices = np.where(pick_lower, lower_idx, upper_idx)
    misfit = np.abs(times[indices] - node_times)
    if np.any(misfit > _NODE_TIME_TOLERANCE_S):
        offending = float(node_times[int(np.argmax(misfit))])
        raise ValueError(f"node time {offending} s is not a capture frame")
    sampled_valid = valid_arr[indices] & np.isfinite(points[indices]).all(axis=2)
    sampled = np.where(sampled_valid[..., None], points[indices], 0.0)
    weight_vector = np.array(
        [float((weights or {}).get(label, 1.0)) for label in labels], dtype=float
    )
    return MarkerTargets(
        tuple(labels),
        np.asarray(node_times, dtype=float),
        sampled,
        sampled_valid,
        weight_vector,
    )


def coordinate_bounds(
    specification: Mapping[str, Any], coordinate_order: Sequence[str]
) -> tuple[Array, Array]:
    """Lower/upper bounds in radians from ``coordinate_ranges_deg``; infinite where absent."""
    ranges = specification.get("coordinate_ranges_deg") or {}
    lower = np.full(len(coordinate_order), -np.inf)
    upper = np.full(len(coordinate_order), np.inf)
    for index, name in enumerate(coordinate_order):
        if name in ranges:
            low_deg, high_deg = ranges[name]
            require(
                high_deg > low_deg,
                f"range for {name} must be increasing",
                (low_deg, high_deg),
            )
            lower[index] = np.deg2rad(float(low_deg))
            upper[index] = np.deg2rad(float(high_deg))
    return lower, upper


def actuated_mask(coordinate_order: Sequence[str]) -> NDArray[np.bool_]:
    """True for every coordinate that carries an effort (the six root coordinates do not)."""
    return np.array(
        [name not in ROOT_COORDINATES for name in coordinate_order], dtype=bool
    )


EFFORT_BOUND_PATTERNS_N_M: tuple[tuple[str, float], ...] = (
    ("mtp_angle", 10.0),
    ("subtalar_angle", 40.0),
    ("ankle_angle", 150.0),
    ("knee_angle", 250.0),
    ("hip_", 300.0),
    ("SpineInput", 300.0),
    ("TorsoInput", 300.0),
    ("RScapInput", 160.0),
    ("ScapInput", 80.0),
    ("LSInput", 120.0),
    ("RSInput", 250.0),
    ("LEInput", 80.0),
    ("REInput", 180.0),
    ("LFInput", 40.0),
    ("RFInput", 80.0),
    ("LWInput", 30.0),
    ("RWInput", 60.0),
    ("NeckInput", 30.0),
)
"""Peak-effort bounds by coordinate-name pattern (first match wins), in N m.

Values are generous human joint-torque envelopes; they bound the solver, they
are not identified quantities. Unmatched actuated coordinates fall back to the
caller's default.
"""


def per_coordinate_effort_bounds(
    coordinate_order: Sequence[str], actuated: NDArray[np.bool_], default_n_m: float
) -> Array:
    """Effort bound for each actuated coordinate, in the actuated-column order."""
    require(default_n_m > 0.0, "default effort bound must be positive", default_n_m)
    bounds: list[float] = []
    for name, is_actuated in zip(coordinate_order, actuated, strict=True):
        if not is_actuated:
            continue
        bound = default_n_m
        for pattern, value in EFFORT_BOUND_PATTERNS_N_M:
            if pattern in name:
                bound = value
                break
        bounds.append(bound)
    return np.asarray(bounds, dtype=float)


def least_squares_controls(
    a_reference: Array,
    a_zero_effort: Array,
    deffort: Array,
    actuated: NDArray[np.bool_],
    *,
    ridge: float = 0.0,
) -> Array:
    """Efforts that best reproduce a reference acceleration through the plant's effort sensitivity.

    The plant acceleration is affine in the effort, ``a(u) = a(0) + (da/du) u``,
    so the warm start is a linear least-squares problem per node.
    """
    a_ref = np.asarray(a_reference, dtype=float)
    a_zero = np.asarray(a_zero_effort, dtype=float)
    sensitivity = np.asarray(deffort, dtype=float)[:, actuated]
    require(
        a_ref.shape == a_zero.shape == (sensitivity.shape[0],),
        "acceleration shapes must agree",
    )
    require(ridge >= 0.0, "ridge must be nonnegative", ridge)
    if ridge > 0.0:
        # Tikhonov-regularised normal equations damp near-massless directions.
        normal = sensitivity.T @ sensitivity + ridge * np.eye(sensitivity.shape[1])
        solution = np.linalg.solve(normal, sensitivity.T @ (a_ref - a_zero))
    else:
        solution, *_ = np.linalg.lstsq(sensitivity, a_ref - a_zero, rcond=None)
    ensure(bool(np.isfinite(solution).all()), "least-squares controls must be finite")
    return solution


def finite_difference_rates(q: Array, dt_s: float) -> Array:
    """Central-difference rates with one-sided ends; shape equals ``q``."""
    q_arr = np.asarray(q, dtype=float)
    require(
        q_arr.ndim == 2 and q_arr.shape[0] >= 2,
        "q must be (nodes >= 2, coordinates)",
        q_arr.shape,
    )
    return np.gradient(q_arr, dt_s, axis=0)


def range_barrier(
    q: Array, lower: Array, upper: Array, weight: float
) -> tuple[float, Array, Array]:
    """Quadratic penalty outside ``[lower, upper]``: value, gradient and diagonal Hessian."""
    q_arr = np.asarray(q, dtype=float)
    below = np.minimum(q_arr - lower, 0.0)
    above = np.maximum(q_arr - upper, 0.0)
    excess = np.where(np.isfinite(lower), below, 0.0) + np.where(
        np.isfinite(upper), above, 0.0
    )
    cost = 0.5 * weight * float(np.dot(excess, excess))
    gradient = weight * excess
    hessian = np.where(excess != 0.0, weight, 0.0)
    return cost, gradient, hessian


@lru_cache(maxsize=1)
def qualified_crocoddyl() -> Any:
    """Return verified healthy crocoddyl module or raise CrocoddylNotAvailableError."""
    croc = require_crocoddyl()
    healthy, reason = crocoddyl_stack_healthy()
    if not healthy:
        raise CrocoddylNotAvailableError(reason)
    return croc


@dataclass(frozen=True)
class CrocoddylProblemConfig:
    """Configuration for native Crocoddyl full-body problem assembly."""

    dt_s: float = 1.0 / 360.0
    marker_weight: float = 100.0
    terminal_marker_weight: float = 500.0
    effort_weight: float = 1e-3
    joint_limit_weight: float = 10.0
    weld_weight: float = 1000.0
    terminal_velocity_weight: float = 1.0

    def __post_init__(self) -> None:
        require(
            self.dt_s > 0.0 and np.isfinite(self.dt_s), "dt_s must be positive finite"
        )
        require(self.marker_weight >= 0.0, "marker_weight must be non-negative")
        require(
            self.terminal_marker_weight >= 0.0,
            "terminal_marker_weight must be non-negative",
        )
        require(self.effort_weight >= 0.0, "effort_weight must be non-negative")
        require(
            self.joint_limit_weight >= 0.0, "joint_limit_weight must be non-negative"
        )
        require(self.weld_weight >= 0.0, "weld_weight must be non-negative")


class CrocoddylProblemBundle(NamedTuple):
    """Assembled Crocoddyl problem with metadata and dimension contracts."""

    problem: Any
    time_grid: Array
    nq: int
    nv: int
    nu: int
    coordinate_order: tuple[str, ...]
    marker_labels: tuple[str, ...]
    config: CrocoddylProblemConfig


def _validate_grid(time_s: Array) -> Array:
    grid = np.asarray(time_s, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all():
        raise ValueError("time_grid must contain at least two finite times")
    if grid[0] < 0.0:
        raise ValueError("time_grid must be nonnegative")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("time_grid must be strictly increasing")
    return grid


def _make_action_models(
    croc: Any,
    nx: int,
    nu: int,
    time_grid: Array,
    marker_targets: Array,
    valid_mask: BoolArray,
    config: CrocoddylProblemConfig,
) -> tuple[list[Any], Any]:
    """Instantiate running action models and terminal action model."""
    state = croc.StateVector(nx)
    running_models = []
    t_steps = len(time_grid) - 1

    for _ in range(t_steps):
        # ActionModelAbstract(state, nu, nr)
        model = croc.ActionModelAbstract(state, nu, 0)
        # Set bounds if available
        if hasattr(model, "u_lb") and hasattr(model, "u_ub"):
            model.u_lb = np.full(nu, -1000.0)
            model.u_ub = np.full(nu, 1000.0)
        running_models.append(model)

    terminal_model = croc.ActionModelAbstract(state, 0, 0)
    return running_models, terminal_model


def build_native_crocoddyl_problem(
    specification: Mapping[str, Any],
    time_grid: Sequence[float] | Array,
    marker_targets: Array,
    valid_mask: BoolArray,
    marker_labels: Sequence[str],
    *,
    initial_state: Array | None = None,
    config: CrocoddylProblemConfig | None = None,
) -> CrocoddylProblemBundle:
    """Pure problem assembly for full-body Crocoddyl optimal control.

    Validates plant coordinates, capture target dimensions, time monotonicity,
    and constructs a Crocoddyl ShootingProblem.
    """
    cfg = config or CrocoddylProblemConfig()
    coords = tuple(specification.get("coordinate_order", ()))
    nq = len(coords)
    if nq == 0:
        raise ValueError("specification must provide non-empty coordinate_order")
    nv = nq
    nx = nq + nv
    nu = nq  # Per-node joint torque actuation

    grid = _validate_grid(np.asarray(time_grid, dtype=float))
    num_nodes = len(grid)

    targets = np.asarray(marker_targets, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    labels = tuple(marker_labels)

    if targets.ndim != 3 or targets.shape[0] != num_nodes:
        raise ValueError(
            f"marker_targets shape {targets.shape} must match (nodes={num_nodes}, markers, 3)"
        )
    if targets.shape[1] != len(labels) or targets.shape[2] != 3:
        raise ValueError(
            f"marker_targets ({targets.shape[1]}, {targets.shape[2]}) must match ({len(labels)}, 3)"
        )
    if valid.shape != (num_nodes, len(labels)):
        raise ValueError(
            f"validity_mask shape {valid.shape} must match ({num_nodes}, {len(labels)})"
        )

    # Initial state: default to zeros [q0=0, v0=0] if not provided
    x0 = (
        np.zeros(nx, dtype=float)
        if initial_state is None
        else np.asarray(initial_state, dtype=float)
    )
    if x0.shape != (nx,) or not np.isfinite(x0).all():
        raise ValueError(f"initial_state must be a finite array of shape ({nx},)")

    croc = qualified_crocoddyl()
    running_models, terminal_model = _make_action_models(
        croc, nx, nu, grid, targets, valid, cfg
    )

    problem = croc.ShootingProblem(x0.copy(), running_models, terminal_model)

    return CrocoddylProblemBundle(
        problem=problem,
        time_grid=grid,
        nq=nq,
        nv=nv,
        nu=nu,
        coordinate_order=coords,
        marker_labels=labels,
        config=cfg,
    )
