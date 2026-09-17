"""Pure assembly for the native Crocoddyl full-body fit (MS-31, #10338).

This module has no engine dependency. It turns a capture window and a
full-body document into the arrays the FDDP driver consumes: node times,
per-node marker targets with validity, coordinate bounds, the actuated
mask, and two warm-start helpers (finite-difference rates and least-squares
controls from the plant's effort sensitivity). Keeping it pure makes every
piece unit-testable without Pinocchio or Crocoddyl.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.polynomial_actuation import ROOT_COORDINATES

Array = NDArray[np.float64]

_NODE_TIME_TOLERANCE_S = 1e-9


@dataclass(frozen=True)
class FitWeights:
    """Quadratic cost weights of the marker-tracking optimal-control problem.

    Every weight multiplies one half of a squared residual. ``marker`` and
    ``terminal_marker`` are per marker (metres); ``effort`` per actuated
    coordinate (N m); ``velocity`` per coordinate (rad/s); ``range_barrier``
    per coordinate outside its human range (rad).
    """

    marker: float = 1.0
    terminal_marker: float = 10.0
    effort: float = 1e-5
    velocity: float = 1e-4
    range_barrier: float = 1e2

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


def least_squares_controls(
    a_reference: Array,
    a_zero_effort: Array,
    deffort: Array,
    actuated: NDArray[np.bool_],
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
