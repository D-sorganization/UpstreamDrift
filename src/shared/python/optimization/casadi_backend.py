"""CasADi direct-transcription backend for swing optimization.

Part of epic #8390 (B3/#8398). Productionizes the pattern proven by
``examples/optimize_arm.py`` — symbolic dynamics + IPOPT — against the
shared seven-DOF swing model from :mod:`.model_provider` instead of a
hard-coded two-link arm.

Formulation (direct transcription on the ``SwingOptimizer`` node grid):

- Decision variables: joint angles ``Q`` and velocities ``V`` at the
  ``n_nodes`` waypoints, in the exact layout of the flagship optimizer's
  decision vector (``[angles.flatten(), velocities.flatten()]``), so warm
  starts and result extraction interoperate with the scipy path.
- Velocity consistency: central finite differences tie ``V`` to ``Q``;
  accelerations are finite-difference expressions of ``V``.
- Dynamics: joint torques come from a CasADi-symbolic recursive
  Newton-Euler (RNEA) over the same chain parameters the URDF bridge
  emits (single-axis revolute joints, generic segment inertials), bounded
  by the golfer's per-joint torque limits. The symbolic RNEA is validated
  against Pinocchio's ``pin.rnea`` in the live test suite.
- Objective: maximize terminal clubhead speed with an effort integral and
  the smooth injury surrogate's velocity/torque terms (B1/#8396) — all
  differentiable by construction.

``casadi`` is an opt-in dependency (the ``optimal-control`` extra); when
absent, :func:`require_casadi` raises with an install hint.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any

import numpy as np

from src.shared.python.optimization._swing_kinematics import JOINTS
from src.shared.python.optimization._swing_models import (
    ClubModel,
    GolferModel,
    OptimizationConfig,
    OptimizationObjective,
)
from src.shared.python.optimization.model_provider import build_swing_rig

_INSTALL_HINT = (
    "casadi is not installed. Install the optimal-control extra: "
    "pip install 'upstream-drift[optimal-control]'"
)

# Segment inertial constants mirroring motion_pipeline.model_bridge so the
# symbolic dynamics match the URDF the engine backends consume.
_LINK_MASS = 1.0
_LINK_INERTIA = 1e-2
_GRAVITY = np.array([0.0, 0.0, -9.81])

_AXES = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
}

# Velocity decision-variable bounds [rad/s]; generous but finite so IPOPT
# has a bounded feasible set (the smooth injury term discourages >20).
_VELOCITY_BOUND = 40.0


class CasadiNotAvailableError(RuntimeError):
    """Raised when casadi is required but not importable."""


def casadi_available() -> bool:
    """Whether the ``casadi`` module is importable (mock-tolerant)."""
    try:
        return find_spec("casadi") is not None
    except (ValueError, ModuleNotFoundError):
        return False


def require_casadi() -> Any:
    """Import and return ``casadi``, raising with an install hint if absent."""
    if not casadi_available():
        raise CasadiNotAvailableError(_INSTALL_HINT)
    return import_module("casadi")


@dataclass(frozen=True)
class CasadiSwingResult:
    """Solver outcome in the flagship optimizer's conventions."""

    success: bool
    x: np.ndarray
    fun: float
    message: str
    iterations: int


def _chain_parameters(
    golfer: GolferModel, club: ClubModel
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(offsets, axes) per DOF from the shared swing rig."""
    rig = build_swing_rig(golfer, club)
    offsets = []
    axes = []
    for name in JOINTS:
        joint = rig.joints[name]
        offsets.append(np.asarray(joint.tpose_offset, dtype=float))
        axes.append(_AXES[str(joint.axes[0])])
    return offsets, axes


def build_symbolic_rnea(
    golfer: GolferModel, club: ClubModel, ca: Any | None = None
) -> Any:
    """CasADi Function ``tau = rnea(q, v, a)`` for the swing chain.

    Classic recursive Newton-Euler for a fixed-base serial chain of
    single-axis revolute joints: forward pass propagates angular velocity,
    angular acceleration, and linear acceleration (including gravity);
    backward pass accumulates forces/moments and projects onto each joint
    axis. Each link's COM sits at its frame origin with the bridge's
    generic mass/inertia, matching the URDF inertials exactly.
    """
    ca = ca or require_casadi()
    offsets, axes = _chain_parameters(golfer, club)
    n = len(JOINTS)

    q = ca.SX.sym("q", n)
    v = ca.SX.sym("v", n)
    a = ca.SX.sym("a", n)

    def rot(axis: np.ndarray, angle: Any) -> Any:
        """Rodrigues rotation about a fixed unit axis."""
        k = ca.SX(axis)
        kx = ca.skew(k)
        return ca.SX.eye(3) + ca.sin(angle) * kx + (1 - ca.cos(angle)) * (kx @ kx)

    # Forward pass (all quantities in each link's own frame).
    omegas, domegas, accs = [], [], []
    coms_acc = []
    omega = ca.SX.zeros(3)
    domega = ca.SX.zeros(3)
    acc = ca.SX(-_GRAVITY)  # base acceleration trick: a0 = -g
    rotations = []
    for i in range(n):
        r_parent_child = rot(axes[i], q[i])
        r_child_parent = r_parent_child.T
        rotations.append(r_parent_child)
        z = ca.SX(axes[i])
        offset = ca.SX(offsets[i])
        omega_p, domega_p, acc_p = omega, domega, acc
        # Acceleration of the child-joint origin in the parent frame.
        acc_joint = (
            acc_p
            + ca.cross(domega_p, offset)
            + ca.cross(omega_p, ca.cross(omega_p, offset))
        )
        omega = r_child_parent @ omega_p + z * v[i]
        domega = (
            r_child_parent @ domega_p
            + ca.cross(r_child_parent @ omega_p, z * v[i])
            + z * a[i]
        )
        acc = r_child_parent @ acc_joint
        omegas.append(omega)
        domegas.append(domega)
        accs.append(acc)
        # URDF <inertial> without an origin places each link's COM at the
        # link frame origin — match that so pin.rnea on the bridge URDF
        # validates this function exactly.
        com = ca.SX.zeros(3)
        coms_acc.append((com, acc))

    # Backward pass.
    inertia = ca.SX.eye(3) * _LINK_INERTIA
    force = ca.SX.zeros(3)
    moment = ca.SX.zeros(3)
    taus = [None] * n
    for i in reversed(range(n)):
        com, acc_com = coms_acc[i]
        f_inertial = _LINK_MASS * acc_com
        m_inertial = inertia @ domegas[i] + ca.cross(omegas[i], inertia @ omegas[i])
        if i + 1 < n:
            r_child = rotations[i + 1]
            f_child = r_child @ force
            m_child = r_child @ moment + ca.cross(ca.SX(offsets[i + 1]), f_child)
        else:
            f_child = ca.SX.zeros(3)
            m_child = ca.SX.zeros(3)
        force = f_inertial + f_child
        moment = m_inertial + ca.cross(com, f_inertial) + m_child
        taus[i] = ca.dot(ca.SX(axes[i]), moment)

    tau = ca.vertcat(*taus)
    return ca.Function("swing_rnea", [q, v, a], [tau])


def build_clubhead_position(
    golfer: GolferModel, club: ClubModel, ca: Any | None = None
) -> Any:
    """CasADi Function ``p = clubhead(q)`` — terminal joint origin FK."""
    ca = ca or require_casadi()
    offsets, axes = _chain_parameters(golfer, club)

    q = ca.SX.sym("q", len(JOINTS))

    def rot(axis: np.ndarray, angle: Any) -> Any:
        k = ca.SX(axis)
        kx = ca.skew(k)
        return ca.SX.eye(3) + ca.sin(angle) * kx + (1 - ca.cos(angle)) * (kx @ kx)

    p = ca.SX.zeros(3)
    r_world = ca.SX.eye(3)
    for i in range(len(JOINTS)):
        p = p + r_world @ ca.SX(offsets[i])
        r_world = r_world @ rot(axes[i], q[i])
    return ca.Function("clubhead_position", [q], [p])


def solve_swing_casadi(
    golfer: GolferModel,
    club: ClubModel,
    config: OptimizationConfig,
    torque_limits: dict[str, float],
    joint_limits: dict[str, tuple[float, float]],
    x0: np.ndarray,
) -> CasadiSwingResult:
    """Solve the swing trajectory optimization with CasADi + IPOPT.

    Args:
        golfer: Anthropometrics (limits, flexibility).
        club: Club parameters (chain geometry).
        config: Node count, duration, objective weights, solver options.
        torque_limits: Per-joint torque bounds [N*m].
        joint_limits: Per-joint (lower, upper) angle bounds [rad].
        x0: Warm-start decision vector in flagship layout.

    Returns:
        CasadiSwingResult with the optimized decision vector.

    Raises:
        CasadiNotAvailableError: When casadi is not installed.
        ValueError: On malformed inputs.
    """
    ca = require_casadi()
    n_joints = len(JOINTS)
    n_nodes = config.n_nodes
    expected = 2 * n_joints * n_nodes
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.shape[0] != expected:
        raise ValueError(
            f"x0 must have length {expected} "
            f"(2 * {n_joints} joints * {n_nodes} nodes), got {x0.shape[0]}"
        )
    dt = config.swing_duration / max(n_nodes - 1, 1)

    rnea = build_symbolic_rnea(golfer, club, ca)
    clubhead = build_clubhead_position(golfer, club, ca)

    opti = ca.Opti()
    q_var = opti.variable(n_joints, n_nodes)
    v_var = opti.variable(n_joints, n_nodes)

    flex = golfer.flexibility_factor
    for j, joint in enumerate(JOINTS):
        lo, hi = joint_limits[joint]
        opti.subject_to(opti.bounded(lo * flex, q_var[j, :], hi * flex))
    opti.subject_to(opti.bounded(-_VELOCITY_BOUND, v_var, _VELOCITY_BOUND))

    # Velocity consistency (central differences; one-sided at endpoints).
    for k in range(n_nodes):
        if k == 0:
            dq = (q_var[:, 1] - q_var[:, 0]) / dt
        elif k == n_nodes - 1:
            dq = (q_var[:, -1] - q_var[:, -2]) / dt
        else:
            dq = (q_var[:, k + 1] - q_var[:, k - 1]) / (2 * dt)
        opti.subject_to(v_var[:, k] == dq)

    # Boundary: start at the warm-start address pose, at rest.
    q0_nodes = x0[: n_joints * n_nodes].reshape(n_joints, n_nodes)
    opti.subject_to(q_var[:, 0] == q0_nodes[:, 0])
    opti.subject_to(v_var[:, 0] == 0.0)

    limits_vec = np.array([torque_limits.get(j, 100.0) for j in JOINTS])
    effort = 0
    smooth_risk = 0
    for k in range(n_nodes):
        if k == 0:
            dv = (v_var[:, 1] - v_var[:, 0]) / dt
        elif k == n_nodes - 1:
            dv = (v_var[:, -1] - v_var[:, -2]) / dt
        else:
            dv = (v_var[:, k + 1] - v_var[:, k - 1]) / (2 * dt)
        tau_k = rnea(q_var[:, k], v_var[:, k], dv)
        opti.subject_to(opti.bounded(-limits_vec, tau_k, limits_vec))
        effort = effort + ca.sumsqr(tau_k) * dt
        # Smooth injury surrogate terms (velocity spike + torque
        # saturation), mirroring smooth_costs' logistic structure.
        for j in range(n_joints):
            smooth_risk = smooth_risk + 10.0 / (
                1 + ca.exp(-8.0 * (v_var[j, k] ** 2 - 20.0**2) / 40.0)
            )
            smooth_risk = smooth_risk + 15.0 / (
                1
                + ca.exp(
                    -8.0
                    * (tau_k[j] ** 2 - (0.8 * limits_vec[j]) ** 2)
                    / (2 * limits_vec[j])
                )
            )
    smooth_risk = smooth_risk / n_nodes

    # Terminal clubhead speed via FK jacobian-vector product.
    q_sym = ca.SX.sym("q", n_joints)
    v_sym = ca.SX.sym("v", n_joints)
    p_sym = clubhead(q_sym)
    speed_fn = ca.Function(
        "clubhead_speed",
        [q_sym, v_sym],
        [ca.norm_2(ca.jtimes(p_sym, q_sym, v_sym))],
    )
    terminal_speed = speed_fn(q_var[:, -1], v_var[:, -1])

    w_speed = config.objectives.get(OptimizationObjective.CLUBHEAD_VELOCITY, 1.0)
    w_injury = config.objectives.get(OptimizationObjective.INJURY_RISK, 0.0)
    w_energy = config.objectives.get(OptimizationObjective.ENERGY_EFFICIENCY, 0.0)
    objective = (
        -w_speed * terminal_speed / 50.0
        + w_injury * smooth_risk / 100.0
        + w_energy * effort / 1000.0
        + 1e-4 * effort  # regularizer keeps torques bounded when w_energy=0
    )
    opti.minimize(objective)

    opti.set_initial(q_var, q0_nodes)
    v0_nodes = x0[n_joints * n_nodes :].reshape(n_joints, n_nodes)
    opti.set_initial(v_var, v0_nodes)

    opti.solver(
        "ipopt",
        {"print_time": False},
        {"max_iter": int(config.max_iterations), "print_level": 0, "sb": "yes"},
    )
    try:
        solution = opti.solve()
    except RuntimeError as exc:
        return CasadiSwingResult(
            success=False,
            x=x0,
            fun=float("nan"),
            message=f"IPOPT failed: {exc}",
            iterations=0,
        )

    q_opt = np.asarray(solution.value(q_var), dtype=float).reshape(n_joints, n_nodes)
    v_opt = np.asarray(solution.value(v_var), dtype=float).reshape(n_joints, n_nodes)
    x_out = np.concatenate([q_opt.flatten(), v_opt.flatten()])
    stats = solution.stats()
    return CasadiSwingResult(
        success=bool(stats.get("success", True)),
        x=x_out,
        fun=float(solution.value(objective)),
        message=str(stats.get("return_status", "solved")),
        iterations=int(stats.get("iter_count", 0)),
    )
