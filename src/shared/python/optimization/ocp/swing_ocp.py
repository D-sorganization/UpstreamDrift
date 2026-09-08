"""Max-clubhead-speed swing OCP on bioptim (epic #9762, Phase 2.1).

The problem :func:`casadi_backend.solve_swing_casadi` solves -- drive the
terminal clubhead speed up under joint, velocity and torque limits, with an
effort integral and the smooth injury surrogate -- posed as a real
torque-driven optimal-control problem: multiple shooting (RK4) or direct
collocation, with the dynamics enforced on every interval by construction.

Two terminal objectives, because they are numerically very different:

- ``"track_speed"`` (default): minimise ``(||v_clubhead(T)|| - target)^2``.
  Convex in the terminal speed, so IPOPT converges; it is also the
  formulation :mod:`.crocoddyl_backend` already uses (it tracks a target
  impact motion) and what ``SwingOptimizationConfig.target_clubhead_velocity``
  means. Ask for the speed a golfer is trying to reach.
- ``"maximize_speed"``: minimise ``-||v_clubhead(T)||^2``, matching the
  CasADi backend's objective exactly. A negative-weight quadratic is
  **concave**, so the NLP is nonconvex in that direction: with the dynamics
  properly enforced the optimum sits on the velocity bound and IPOPT
  typically stops at the iteration cap with a good but not certified point.
  Use it for parity comparisons, not production runs.

The node grid equals the flagship optimizer's (``n_shooting = n_nodes - 1``,
``final_time = swing_duration``), so the decision vector comes back in the
``[angles.flatten(), velocities.flatten()]`` layout without resampling and
``SwingOptimizer`` can select ``solver="bioptim"`` through the backend
registry with no new plumbing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from src.shared.python.optimization._swing_kinematics import JOINTS
from src.shared.python.optimization._swing_models import (
    ClubModel,
    GolferModel,
    OptimizationConfig,
    OptimizationObjective,
)
from src.shared.python.optimization.casadi_backend import CasadiSwingResult
from src.shared.python.optimization.ocp._compat import require_bioptim
from src.shared.python.optimization.ocp.bioptim_model import make_swing_bio_model
from src.shared.python.optimization.ocp.result import (
    OcpSwingSolution,
    bioptim_provenance,
    solution_to_swing_result,
)

__all__ = [
    "MaxSpeedOcpOptions",
    "OdeKind",
    "build_max_speed_ocp",
    "solve_max_speed_ocp",
    "solve_max_speed_swing",
]

OdeKind = Literal["rk4", "collocation"]
TerminalObjective = Literal["track_speed", "maximize_speed"]

#: Target clubhead speed [m/s] when ``objective="track_speed"``. Mirrors
#: ``swing_bridge.SwingOptimizationConfig.target_clubhead_velocity``
#: (50 m/s ~ 112 mph, the PGA Tour average driver).
DEFAULT_TARGET_SPEED = 50.0

#: Velocity decision-variable bound [rad/s], mirroring ``casadi_backend``.
_VELOCITY_BOUND = 40.0
#: Objective scales mirroring ``casadi_backend.solve_swing_casadi``: the
#: speed term is normalised by 50 m/s, the injury surrogate by 100, effort by
#: 1000 plus the 1e-4 regulariser.
_SPEED_SCALE = 50.0
_INJURY_SCALE = 100.0
_EFFORT_SCALE = 1000.0
_EFFORT_REGULARISER = 1e-4
_VELOCITY_RISK_LIMIT = 20.0
_TORQUE_RISK_FRACTION = 0.8


@dataclass(frozen=True)
class MaxSpeedOcpOptions:
    """How the max-speed swing OCP is transcribed and scored.

    These travel together -- the objective choice, its target, the
    integrator and the grid all describe one problem statement -- so they
    are one object rather than eight loose keyword arguments. Validating
    them in ``__post_init__`` also means a malformed combination fails at
    the boundary, before any bioptim object is built.

    Attributes:
        objective: ``"track_speed"`` (default, convex) or
            ``"maximize_speed"`` (CasADi-backend parity, nonconvex). See
            the module docstring for why the default is the convex one.
        target_speed: Terminal clubhead speed [m/s] for ``"track_speed"``.
        ode: ``"rk4"`` multiple shooting or ``"collocation"`` (degree 3).
        n_integration_steps: RK4 substeps per shooting interval.
        n_shooting: Shooting intervals; ``None`` means
            ``config.n_nodes - 1`` so the grid matches the flagship
            optimizer's and no resampling is needed.
        parameters: Symbolic model parameters to expose (Phase 4).
        n_threads: bioptim worker threads.
    """

    objective: TerminalObjective = "track_speed"
    target_speed: float = DEFAULT_TARGET_SPEED
    ode: OdeKind = "rk4"
    n_integration_steps: int = 4
    n_shooting: int | None = None
    parameters: tuple[str, ...] = field(default_factory=tuple)
    n_threads: int = 1

    def __post_init__(self) -> None:
        if self.objective not in ("track_speed", "maximize_speed"):
            raise ValueError(
                "objective must be 'track_speed' or 'maximize_speed', "
                f"got {self.objective!r}"
            )
        if self.objective == "track_speed" and self.target_speed <= 0.0:
            raise ValueError("target_speed must be positive")
        if self.ode not in ("rk4", "collocation"):
            raise ValueError(f"ode must be 'rk4' or 'collocation', got {self.ode!r}")
        if self.n_integration_steps < 1:
            raise ValueError("n_integration_steps must be positive")
        if self.n_shooting is not None and self.n_shooting < 1:
            raise ValueError("n_shooting must be positive")
        if self.n_threads < 1:
            raise ValueError("n_threads must be positive")
        # Accept any sequence at the boundary, store the hashable form.
        object.__setattr__(self, "parameters", tuple(self.parameters))


def _ode_solver(bioptim: Any, kind: OdeKind, n_integration_steps: int) -> Any:
    if kind == "rk4":
        return bioptim.OdeSolver.RK4(n_integration_steps=n_integration_steps)
    if kind == "collocation":
        return bioptim.OdeSolver.COLLOCATION(polynomial_degree=3)
    raise ValueError(f"ode must be 'rk4' or 'collocation', got {kind!r}")


def clubhead_velocity_objective(controller: Any, model: Any) -> Any:
    """Custom Mayer objective: the clubhead velocity vector (quadratic)."""
    return model.symbolic.clubhead_velocity(
        controller.q, controller.qdot, controller.parameters.cx
    )


def clubhead_speed_error_objective(
    controller: Any, model: Any, target_speed: float
) -> Any:
    """Custom Mayer objective: terminal clubhead speed minus its target.

    Scalar and convex around the target, which is what makes the tracking
    formulation converge where pure maximisation does not.
    """
    import casadi as ca

    velocity = model.symbolic.clubhead_velocity(
        controller.q, controller.qdot, controller.parameters.cx
    )
    return ca.norm_2(velocity) - target_speed


def smooth_injury_risk_objective(controller: Any, limits: np.ndarray) -> Any:
    """Custom Lagrange objective mirroring ``casadi_backend``'s surrogate."""
    import casadi as ca

    qdot = controller.qdot
    tau = controller.tau
    risk: Any = 0.0
    for j in range(limits.shape[0]):
        risk = risk + 10.0 / (
            1 + ca.exp(-8.0 * (qdot[j] ** 2 - _VELOCITY_RISK_LIMIT**2) / 40.0)
        )
        risk = risk + 15.0 / (
            1
            + ca.exp(
                -8.0
                * (tau[j] ** 2 - (_TORQUE_RISK_FRACTION * limits[j]) ** 2)
                / (2 * limits[j])
            )
        )
    return risk


def _resample(values: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly resample ``n x n_in`` node values onto ``n_out`` nodes."""
    n_in = values.shape[1]
    if n_in == n_out:
        return np.asarray(values, dtype=float)
    src = np.linspace(0.0, 1.0, n_in)
    dst = np.linspace(0.0, 1.0, n_out)
    return np.vstack([np.interp(dst, src, row) for row in values])


def _resolve_grid(
    config: OptimizationConfig, options: MaxSpeedOcpOptions
) -> tuple[int, float]:
    """Return ``(n_shooting, final_time)`` for the requested node grid."""
    if config.n_nodes < 2:
        raise ValueError("config.n_nodes must be at least 2")
    final_time = float(config.swing_duration)
    if final_time <= 0.0:
        raise ValueError("config.swing_duration must be positive")
    return int(options.n_shooting or (config.n_nodes - 1)), final_time


def _initial_guess(
    x0: np.ndarray | None, n: int, n_nodes: int, n_shooting: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split a flagship-layout warm start onto the shooting grid."""
    if x0 is None:
        zeros = np.zeros((n, n_shooting + 1))
        return zeros, zeros.copy()
    flat = np.asarray(x0, dtype=float).reshape(-1)
    if flat.shape[0] != 2 * n * n_nodes:
        raise ValueError(f"x0 must have length {2 * n * n_nodes}, got {flat.shape[0]}")
    q_guess = _resample(flat[: n * n_nodes].reshape(n, n_nodes), n_shooting + 1)
    v_guess = _resample(flat[n * n_nodes :].reshape(n, n_nodes), n_shooting + 1)
    return q_guess, v_guess


def _build_objectives(
    bioptim: Any,
    model: Any,
    config: OptimizationConfig,
    limits: np.ndarray,
    options: MaxSpeedOcpOptions,
    final_time: float,
) -> Any:
    """Assemble the terminal-speed, effort and injury objective terms."""
    w_speed = config.objectives.get(OptimizationObjective.CLUBHEAD_VELOCITY, 1.0)
    w_injury = config.objectives.get(OptimizationObjective.INJURY_RISK, 0.0)
    w_energy = config.objectives.get(OptimizationObjective.ENERGY_EFFICIENCY, 0.0)

    mayer = bioptim.ObjectiveFcn.Mayer
    lagrange = bioptim.ObjectiveFcn.Lagrange
    objectives = bioptim.ObjectiveList()
    if options.objective == "track_speed":
        objectives.add(
            clubhead_speed_error_objective,
            custom_type=mayer,
            node=bioptim.Node.END,
            quadratic=True,
            weight=float(w_speed) / _SPEED_SCALE**2,
            model=model,
            target_speed=float(options.target_speed),
        )
    else:
        objectives.add(
            clubhead_velocity_objective,
            custom_type=mayer,
            node=bioptim.Node.END,
            quadratic=True,
            weight=-float(w_speed) / _SPEED_SCALE**2,
            model=model,
        )
    objectives.add(
        lagrange.MINIMIZE_CONTROL,
        key="tau",
        weight=float(w_energy) / _EFFORT_SCALE + _EFFORT_REGULARISER,
    )
    if w_injury:
        objectives.add(
            smooth_injury_risk_objective,
            custom_type=lagrange,
            node=bioptim.Node.ALL_SHOOTING,
            quadratic=False,
            weight=float(w_injury) / _INJURY_SCALE / final_time,
            limits=limits,
        )
    return objectives


def _build_bounds(
    bioptim: Any,
    golfer: GolferModel,
    joint_limits: dict[str, tuple[float, float]],
    limits: np.ndarray,
    address: np.ndarray,
) -> tuple[Any, Any]:
    """Joint/velocity/torque bounds, with the first node held at address."""
    n = len(JOINTS)
    flex = golfer.flexibility_factor
    q_lower = np.array([joint_limits[j][0] for j in JOINTS]) * flex
    q_upper = np.array([joint_limits[j][1] for j in JOINTS]) * flex
    x_bounds = bioptim.BoundsList()
    x_bounds["q"] = q_lower, q_upper
    x_bounds["qdot"] = np.full(n, -_VELOCITY_BOUND), np.full(n, _VELOCITY_BOUND)
    x_bounds["q"][:, 0] = np.clip(address, q_lower, q_upper)
    x_bounds["qdot"][:, 0] = 0.0
    u_bounds = bioptim.BoundsList()
    u_bounds["tau"] = -limits, limits
    return x_bounds, u_bounds


def _build_init_and_scaling(
    bioptim: Any, q_guess: np.ndarray, v_guess: np.ndarray, limits: np.ndarray
) -> tuple[Any, Any, Any, Any]:
    """Warm start plus the variable scaling IPOPT needs to make progress."""
    n = len(JOINTS)
    x_init = bioptim.InitialGuessList()
    x_init.add("q", q_guess, interpolation=bioptim.InterpolationType.EACH_FRAME)
    x_init.add("qdot", v_guess, interpolation=bioptim.InterpolationType.EACH_FRAME)
    u_init = bioptim.InitialGuessList()
    u_init["tau"] = np.zeros(n)

    # Scale torques to O(1): IPOPT stalls when tau is O(100) and q is O(1).
    u_scaling = bioptim.VariableScalingList()
    u_scaling.add("tau", scaling=limits)
    x_scaling = bioptim.VariableScalingList()
    x_scaling.add("q", scaling=np.ones(n))
    x_scaling.add("qdot", scaling=np.full(n, 10.0))
    return x_init, u_init, x_scaling, u_scaling


def build_max_speed_ocp(
    golfer: GolferModel,
    club: ClubModel,
    config: OptimizationConfig,
    torque_limits: dict[str, float],
    joint_limits: dict[str, tuple[float, float]],
    x0: np.ndarray | None = None,
    *,
    options: MaxSpeedOcpOptions | None = None,
) -> tuple[Any, Any]:
    """Build the max-clubhead-speed OCP. Returns ``(ocp, model)``.

    Args:
        golfer, club: The numeric model (anthropometric inertials, #9755).
        config: Node count, duration and objective weights. ``n_shooting``
            defaults to ``config.n_nodes - 1`` so the grids coincide.
        torque_limits, joint_limits: Per-joint bounds as in
            :func:`casadi_backend.solve_swing_casadi`.
        x0: Optional warm start in flagship layout; resampled to the
            shooting grid. Its first node fixes the address pose (start at
            rest), as the CasADi backend does. ``None`` starts from the
            neutral pose.
        options: Transcription and objective choices; see
            :class:`MaxSpeedOcpOptions`. ``None`` uses its defaults
            (convex target-speed objective, RK4 multiple shooting).

    Raises:
        BioptimNotAvailableError: When bioptim is not installed.
        ValueError: On malformed inputs.
    """
    bioptim = require_bioptim()
    options = options or MaxSpeedOcpOptions()
    n = len(JOINTS)
    n_shooting, final_time = _resolve_grid(config, options)

    model = make_swing_bio_model(golfer, club, parameters=options.parameters)
    limits = np.array([float(torque_limits.get(j, 100.0)) for j in JOINTS])
    q_guess, v_guess = _initial_guess(x0, n, config.n_nodes, n_shooting)

    dynamics = bioptim.DynamicsOptionsList()
    dynamics.add(
        bioptim.DynamicsOptions(
            ode_solver=_ode_solver(bioptim, options.ode, options.n_integration_steps),
            expand_dynamics=True,
            phase_dynamics=bioptim.PhaseDynamics.SHARED_DURING_THE_PHASE,
        )
    )
    x_bounds, u_bounds = _build_bounds(
        bioptim, golfer, joint_limits, limits, q_guess[:, 0]
    )
    x_init, u_init, x_scaling, u_scaling = _build_init_and_scaling(
        bioptim, q_guess, v_guess, limits
    )

    ocp = bioptim.OptimalControlProgram(
        model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=_build_objectives(
            bioptim, model, config, limits, options, final_time
        ),
        x_scaling=x_scaling,
        u_scaling=u_scaling,
        use_sx=False,
        n_threads=options.n_threads,
    )
    return ocp, model


def _solution_settings(
    config: OptimizationConfig, options: MaxSpeedOcpOptions, n_shooting: int
) -> dict[str, Any]:
    """Provenance record of exactly how this solve was posed."""
    return {
        "objective": options.objective,
        "target_speed": float(options.target_speed),
        "ode": options.ode,
        "n_integration_steps": options.n_integration_steps,
        "n_shooting": n_shooting,
        "final_time": float(config.swing_duration),
        "max_iterations": int(config.max_iterations),
        "objectives": {k.value: float(v) for k, v in config.objectives.items()},
    }


def solve_max_speed_ocp(
    golfer: GolferModel,
    club: ClubModel,
    config: OptimizationConfig,
    torque_limits: dict[str, float],
    joint_limits: dict[str, tuple[float, float]],
    x0: np.ndarray | None = None,
    *,
    options: MaxSpeedOcpOptions | None = None,
) -> OcpSwingSolution:
    """Build and solve the swing OCP; return arrays plus provenance."""
    bioptim = require_bioptim()
    options = options or MaxSpeedOcpOptions()
    ocp, model = build_max_speed_ocp(
        golfer, club, config, torque_limits, joint_limits, x0, options=options
    )
    solver = bioptim.Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(int(config.max_iterations))
    started = time.perf_counter()
    sol = ocp.solve(solver=solver)
    wall = time.perf_counter() - started

    n = len(JOINTS)
    fallback = (
        np.asarray(x0, dtype=float)
        if x0 is not None
        else np.zeros(2 * n * config.n_nodes)
    )
    result, q, qdot, tau = solution_to_swing_result(
        sol,
        bioptim,
        transcription=f"bioptim-{options.ode}",
        x_fallback=fallback,
        n_nodes=config.n_nodes,
    )
    empty = np.zeros(0)
    speed = float(
        np.linalg.norm(
            np.asarray(model.symbolic.clubhead_velocity(q[:, -1], qdot[:, -1], empty))
        )
    )
    return OcpSwingSolution(
        result=result,
        time=np.linspace(0.0, float(config.swing_duration), q.shape[1]),
        q=q,
        qdot=qdot,
        tau=tau,
        parameters={},
        clubhead_speed=speed,
        cost=result.fun,
        status=int(sol.status),
        iterations=result.iterations,
        wall_time_s=wall,
        provenance=bioptim_provenance(
            golfer, club, _solution_settings(config, options, q.shape[1] - 1)
        ),
    )


def solve_max_speed_swing(
    golfer: GolferModel,
    club: ClubModel,
    config: OptimizationConfig,
    torque_limits: dict[str, float],
    joint_limits: dict[str, tuple[float, float]],
    x0: np.ndarray,
) -> CasadiSwingResult:
    """Registry entry point (``solver="bioptim"``): flagship-layout result.

    Uses direct collocation and the convex target-speed objective: on this
    problem it is both better conditioned and an order of magnitude faster
    than RK4 multiple shooting (tens of IPOPT iterations against hundreds).
    Call :func:`solve_max_speed_ocp` directly to choose otherwise.
    """
    return solve_max_speed_ocp(
        golfer,
        club,
        config,
        torque_limits,
        joint_limits,
        x0,
        options=MaxSpeedOcpOptions(ode="collocation"),
    ).result
