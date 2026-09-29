"""Minimal records used by the visualiser and leaderboard writer.

These are intentionally light dataclasses — duck-typed against the future
canonical ``ClubTarget`` / ``FitResult`` defined in
``src/shared/python/motion_matching`` — so that this issue's deliverables
do not block on the simulate driver landing.

The fields here are exactly the union required by:

* the three canonical viz views (VISUALIZATION_SPEC.md), and
* the leaderboard JSON schema specified in issue #4133.

Anything richer (history, surrogate predictions, optimiser metadata) is
opaque to this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ClubTargetLike(Protocol):
    """Structural type for the measured target trajectory.

    Compatible with :class:`src.shared.python.motion_matching.club_target.ClubTarget`
    and any record exposing the same five attributes.
    """

    time: np.ndarray
    butt: np.ndarray
    clubhead: np.ndarray
    club_quat: np.ndarray
    impact_idx: int


@dataclass
class PinocchioVizFitResult:
    """Pinocchio fit result, viz/leaderboard-facing subset.

    Attributes:
        trial_id: Trial identifier — propagates to ``leaderboard.trial``.
        solver: Solver tag — defaults to the Pinocchio LM analytical-Jacobian
            stack (issue #4133 / PR #4168).
        butt_sim: Simulated butt position trajectory, shape ``(N, 3)``.
        clubhead_sim: Simulated clubhead position trajectory, ``(N, 3)``.
        club_quat_sim: Simulated club orientation as wxyz quaternions, ``(N, 4)``.
        time: Simulation time vector, shape ``(N,)``, seconds.
        joint_torques: Per-frame joint torques, ``(N, n_joints)`` (optional).
        clubhead_speed_mph: Clubhead linear speed, ``(N,)`` (optional).
        grip_rmse_mm: Final RMSE of butt position vs target, millimetres.
        clubhead_rmse_mm: Final RMSE of clubhead position vs target, millimetres.
        orientation_rmse_deg: Final RMSE of club orientation, degrees.
        total_work_J: Regularised total mechanical work, joules.
        wall_clock_s: Solver wall-clock time, seconds.
        n_iterations: Solver iteration count.
        commit: Git commit hash for traceability (best-effort, may be ``"unknown"``).
    """

    trial_id: str
    solver: str = "lm-analytical-jacobian"
    butt_sim: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    clubhead_sim: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    club_quat_sim: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    time: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    joint_torques: np.ndarray | None = None
    clubhead_speed_mph: np.ndarray | None = None
    grip_rmse_mm: float = 0.0
    clubhead_rmse_mm: float = 0.0
    orientation_rmse_deg: float = 0.0
    total_work_J: float = 0.0
    wall_clock_s: float = 0.0
    n_iterations: int = 0
    commit: str = "unknown"


FitResult = PinocchioVizFitResult
