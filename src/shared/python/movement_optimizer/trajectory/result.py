# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Data structures for optimisation results."""

from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import NDArray


@dataclass
class OptimizationResult:
    """Container for optimiser output.

    Attributes:
        t: Time grid, shape ``(N,)`` in seconds.
        q: Joint angles, shape ``(N, n_dof)`` in radians.
        qd: Joint velocities, shape ``(N, n_dof)`` in rad/s.
        qdd: Joint accelerations, shape ``(N, n_dof)`` in rad/s^2.
        torques: Joint torques, shape ``(N, n_dof)`` in N*m.
        power: Per-joint mechanical power, shape ``(N, n_dof)`` in W.
        com: Whole-body COM trajectory ``(x, y)``, shape ``(N, 2)`` in m.
        bar: Barbell trajectory ``(x, y)``, shape ``(N, 2)`` in m.
        success: True when the optimiser converged and all hard
            constraints (COM in inner BOS, joint limits) are satisfied.
        cost: Final scalar cost (lower is better).
        com_horizontal_range_cm: Peak-to-peak horizontal COM excursion.
        elapsed_s: Wall-clock seconds spent in :meth:`optimize`.
        n_evals: Total cost evaluations across all starts.
        n_joint_limit_violations: Number of trajectory samples whose
            joint angles fall outside ``q_bounds``.
    """

    t: NDArray
    q: NDArray
    qd: NDArray
    qdd: NDArray
    torques: NDArray
    power: NDArray
    com: NDArray
    bar: NDArray
    success: bool
    cost: float
    com_horizontal_range_cm: float
    elapsed_s: float = 0.0
    n_evals: int = 0
    n_joint_limit_violations: int = 0


@dataclass
class ProgressReport:
    """Snapshot of optimiser state emitted to a progress callback.

    Attributes:
        iteration: Cost evaluation counter.
        cost: Current cost value.
        best_cost: Best cost seen so far.
        improvement_pct: Percent improvement over the previous report.
        elapsed_s: Seconds since :meth:`optimize` started.
        cost_history: Running list of all observed cost values.
        is_stalled: True when the heuristic stall detector triggered.
        stall_reason: Human-readable reason when ``is_stalled``.
    """

    iteration: int
    cost: float
    best_cost: float
    improvement_pct: float
    elapsed_s: float
    cost_history: list[float] = field(default_factory=list)
    is_stalled: bool = False
    stall_reason: str = ""


class CancelledError(Exception):
    """Raised when the user cancels optimisation via ``cancel_event``."""
