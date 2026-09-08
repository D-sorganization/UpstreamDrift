"""Crocoddyl DDP/FDDP backend for swing optimization.

Part of epic #8390 (B4/#8399). Implements the running/terminal-model
design recorded in ``docs/technical/control-strategies-summary.md``: a
running model with control-effort and state regularization plus a
joint-limit barrier, and a terminal model driving the clubhead frame to a
target impact velocity — solved with Crocoddyl's FDDP over the shared B1
swing model (:mod:`.model_provider`), with analytic RNEA/ABA derivatives
from Pinocchio (no finite differences anywhere).

Availability: crocoddyl ships wheels for common Linux platforms but not
everywhere (conda-forge/WSL elsewhere). It is declared as the opt-in
``crocoddyl`` extra, excluded from ``all-engines``; absence degrades via
:class:`CrocoddylNotAvailableError` with an install hint, and importing
this module never fails.

Mixed-stack hazard: installing the PyPI ``crocoddyl`` wheel (cmeel-based,
bundling its own libpinocchio) alongside the PyPI ``pin`` wheel loads two
pinocchio binaries into one process. Observed symptoms range from silent
derivative corruption (FDDP "converges" in one iteration without moving)
to segmentation faults at larger horizons. :func:`crocoddyl_stack_healthy`
probes for this before any real solve, and :func:`solve_swing_ddp` returns
a diagnosed failure (recommending a consistent conda-forge install)
instead of risking a crash.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any

import numpy as np

from src.shared.python.motion_pipeline.model_bridge import rig_joint_link_name
from src.shared.python.optimization._swing_kinematics import JOINTS
from src.shared.python.optimization._swing_models import ClubModel, GolferModel
from src.shared.python.optimization.model_provider import (
    build_pinocchio_model,
    swing_joint_limits,
)

_INSTALL_HINT = (
    "crocoddyl is not installed. Install the crocoddyl extra where wheels "
    "exist (pip install 'upstream-drift[crocoddyl]'); on platforms without "
    "wheels use conda-forge (conda install -c conda-forge crocoddyl) or "
    "WSL. See docs/technical/control-strategies-summary.md."
)

# Terminal frame whose velocity represents the clubhead at impact.
_CLUBHEAD_JOINT = JOINTS[-1]


class CrocoddylNotAvailableError(RuntimeError):
    """Raised when crocoddyl is required but not importable."""


def crocoddyl_available() -> bool:
    """Whether ``crocoddyl`` (and pinocchio) are importable (mock-tolerant)."""
    for module in ("crocoddyl", "pinocchio"):
        try:
            if find_spec(module) is None:
                return False
        except (ValueError, ModuleNotFoundError):
            return False
    return True


def require_crocoddyl() -> Any:
    """Import and return ``crocoddyl``, raising with a hint if absent."""
    if not crocoddyl_available():
        raise CrocoddylNotAvailableError(_INSTALL_HINT)
    return import_module("crocoddyl")


_UNHEALTHY_STACK_MESSAGE = (
    "crocoddyl imported but its solver produced no descent on a probe "
    "problem with a strong terminal gradient — this is the known symptom "
    "of mixing the PyPI crocoddyl wheel (bundled libpinocchio) with the "
    "PyPI pin wheel in one process. Install both from conda-forge "
    "(conda install -c conda-forge crocoddyl pinocchio) for a consistent "
    "binary stack."
)


# Self-contained subprocess probe: build a sample manipulator, ask FDDP
# for a terminal end-effector velocity from rest, and require nonzero
# achieved speed. Runs out-of-process because broken mixed-wheel stacks
# segfault nondeterministically — a crash must be a *diagnosis*, never a
# host-process fatality. Exit codes: 0 healthy, 3 no-descent.
_PROBE_SCRIPT = r"""
import numpy as np
import pinocchio as pin
import crocoddyl

model = pin.buildSampleModelManipulator()
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFull(state)
nu = actuation.nu
run_costs = crocoddyl.CostModelSum(state, nu)
run_costs.addCost(
    "effort",
    crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state, nu)),
    1e-4,
)
running = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, run_costs),
    0.05,
)
frame = model.getFrameId("effector_body")
terminal_costs = crocoddyl.CostModelSum(state, nu)
terminal_costs.addCost(
    "impact",
    crocoddyl.CostModelResidual(
        state,
        crocoddyl.ResidualModelFrameVelocity(
            state,
            frame,
            pin.Motion(np.array([-2.0, 0.0, 0.0]), np.zeros(3)),
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            nu,
        ),
    ),
    10.0,
)
terminal = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminal_costs),
    0.0,
)
x0 = np.concatenate([pin.neutral(model), np.zeros(model.nv)])
problem = crocoddyl.ShootingProblem(x0, [running] * 3, terminal)
solver = crocoddyl.SolverFDDP(problem)
solver.solve([x0] * 4, problem.quasiStatic([x0] * 3), 25)
xs = np.asarray(solver.xs)
data = model.createData()
pin.forwardKinematics(model, data, xs[-1, : model.nq], xs[-1, model.nq :])
velocity = pin.getFrameVelocity(
    model, data, frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
)
speed = float(np.linalg.norm(np.asarray(velocity.linear)))
raise SystemExit(0 if speed > 0.1 else 3)
"""


def crocoddyl_stack_healthy(*, timeout_s: float = 90.0) -> tuple[bool, str]:
    """Probe whether the crocoddyl/pinocchio binary stack actually works.

    Runs a tiny FDDP problem in a subprocess and checks that the solver
    achieves descent (nonzero terminal end-effector speed toward a strong
    velocity target). A mixed PyPI-wheel stack fails this probe either by
    producing no descent or by crashing — both are reported as unhealthy
    rather than crashing the caller.

    Returns:
        ``(healthy, reason)`` — reason is empty when healthy.
    """
    if not crocoddyl_available():
        return False, _INSTALL_HINT
    import subprocess
    import sys

    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no user input
            [sys.executable, "-c", _PROBE_SCRIPT],
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, _UNHEALTHY_STACK_MESSAGE + " (probe timed out)"
    if completed.returncode == 0:
        return True, ""
    if completed.returncode < 0:
        return (
            False,
            _UNHEALTHY_STACK_MESSAGE
            + f" (probe crashed with signal {-completed.returncode})",
        )
    return False, _UNHEALTHY_STACK_MESSAGE


@dataclass(frozen=True)
class DdpSwingResult:
    """FDDP solve outcome.

    Attributes:
        success: Solver convergence flag.
        xs: State trajectory, shape ``(T+1, nq+nv)``.
        us: Control (torque) trajectory, shape ``(T, nu)``.
        terminal_speed: Clubhead frame linear speed at the final knot [m/s].
        cost: Final total cost.
        iterations: FDDP iterations executed.
        message: Human-readable status.
    """

    success: bool
    xs: np.ndarray
    us: np.ndarray
    terminal_speed: float
    cost: float
    iterations: int
    message: str


def solve_swing_ddp(
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
    *,
    horizon: int = 60,
    dt: float = 0.02,
    target_speed: float = 45.0,
    effort_weight: float = 1e-2,
    state_weight: float = 1e-1,
    limit_weight: float = 1e2,
    terminal_weight: float = 1e2,
    max_iterations: int = 200,
) -> DdpSwingResult:
    """Solve the swing as a DDP problem on the shared B1 model.

    Args:
        golfer: Anthropometrics (defaults to :class:`GolferModel`).
        club: Club parameters (defaults to :class:`ClubModel`).
        horizon: Number of running knots.
        dt: Integration step [s].
        target_speed: Desired clubhead speed at impact [m/s], applied along
            the -X direction of the world frame (ball-ward by convention).
        effort_weight: Running control-regularization weight.
        state_weight: Running state-regularization weight.
        limit_weight: Joint-limit quadratic-barrier weight.
        terminal_weight: Terminal clubhead-velocity residual weight.
        max_iterations: FDDP iteration cap.

    Returns:
        DdpSwingResult with state/control trajectories and diagnostics.

    Raises:
        CrocoddylNotAvailableError: When crocoddyl/pinocchio are absent.
        ValueError: On malformed arguments.
    """
    if horizon < 2:
        raise ValueError("horizon must be at least 2")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if target_speed <= 0:
        raise ValueError("target_speed must be positive")

    crocoddyl = require_crocoddyl()
    import pinocchio as pin

    healthy, reason = crocoddyl_stack_healthy()
    if not healthy:
        return DdpSwingResult(
            success=False,
            xs=np.zeros((0, 0)),
            us=np.zeros((0, 0)),
            terminal_speed=0.0,
            cost=float("nan"),
            iterations=0,
            message=reason,
        )

    golfer = golfer or GolferModel()
    club = club or ClubModel()
    model = build_pinocchio_model(golfer, club)
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFull(state)
    nu = actuation.nu

    clubhead_frame = model.getFrameId(rig_joint_link_name(_CLUBHEAD_JOINT))

    # Running costs: control effort, state regularization, joint-limit
    # barrier (quadratic outside the golfer's ROM box).
    run_costs = crocoddyl.CostModelSum(state, nu)
    run_costs.addCost(
        "effort",
        crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state, nu)),
        effort_weight,
    )
    run_costs.addCost(
        "state_reg",
        crocoddyl.CostModelResidual(
            state, crocoddyl.ResidualModelState(state, state.zero(), nu)
        ),
        state_weight,
    )
    limits = swing_joint_limits(golfer)
    lower = np.array([limits[j][0] for j in JOINTS]) * golfer.flexibility_factor
    upper = np.array([limits[j][1] for j in JOINTS]) * golfer.flexibility_factor
    velocity_bound = np.full(model.nv, 40.0)
    bounds = crocoddyl.ActivationBounds(
        np.concatenate([lower, -velocity_bound]),
        np.concatenate([upper, velocity_bound]),
    )
    run_costs.addCost(
        "joint_limits",
        crocoddyl.CostModelResidual(
            state,
            crocoddyl.ActivationModelQuadraticBarrier(bounds),
            crocoddyl.ResidualModelState(state, state.zero(), nu),
        ),
        limit_weight,
    )

    running_dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, run_costs
    )
    running = crocoddyl.IntegratedActionModelEuler(running_dam, dt)

    # Terminal cost: clubhead frame linear velocity toward the ball.
    terminal_costs = crocoddyl.CostModelSum(state, nu)
    target_motion = pin.Motion(np.array([-target_speed, 0.0, 0.0]), np.zeros(3))
    terminal_costs.addCost(
        "impact_speed",
        crocoddyl.CostModelResidual(
            state,
            crocoddyl.ResidualModelFrameVelocity(
                state,
                clubhead_frame,
                target_motion,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                nu,
            ),
        ),
        terminal_weight,
    )
    terminal_costs.addCost(
        "terminal_state_reg",
        crocoddyl.CostModelResidual(
            state, crocoddyl.ResidualModelState(state, state.zero(), nu)
        ),
        state_weight,
    )
    terminal_dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, terminal_costs
    )
    terminal = crocoddyl.IntegratedActionModelEuler(terminal_dam, 0.0)

    x0 = np.concatenate([pin.neutral(model), np.zeros(model.nv)])
    problem = crocoddyl.ShootingProblem(x0, [running] * horizon, terminal)
    solver = crocoddyl.SolverFDDP(problem)
    xs_init = [x0] * (horizon + 1)
    us_init = problem.quasiStatic([x0] * horizon)
    converged = solver.solve(xs_init, us_init, max_iterations)

    xs = np.asarray(solver.xs)
    us = np.asarray(solver.us)

    # Diagnostics: actual clubhead speed at the final knot.
    data = model.createData()
    q_final = xs[-1, : model.nq]
    v_final = xs[-1, model.nq :]
    pin.forwardKinematics(model, data, q_final, v_final)
    velocity = pin.getFrameVelocity(
        model, data, clubhead_frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    terminal_speed = float(np.linalg.norm(np.asarray(velocity.linear)))

    return DdpSwingResult(
        success=bool(converged),
        xs=xs,
        us=us,
        terminal_speed=terminal_speed,
        cost=float(solver.cost),
        iterations=int(solver.iter),
        message="FDDP converged" if converged else "FDDP did not converge",
    )
