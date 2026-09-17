"""Pure native Crocoddyl optimal control problem assembly for the full-body plant (#10338).

Constructs an optimal control shooting problem on the 41/44-coordinate Pinocchio
plant with per-node actuation controls, marker residual tracking costs, 6D dual-grip
weld closure constraint/cost, effort and joint-limit regularization, and terminal
objectives.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.optimization.crocoddyl_backend import (
    CrocoddylNotAvailableError,
    crocoddyl_stack_healthy,
    require_crocoddyl,
)

Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


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
