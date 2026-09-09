"""CasADi direct-transcription backend for swing optimization.

Part of epic #8390 (B3/#8398). Productionizes the pattern proven by
``examples/optimize_arm.py`` — symbolic dynamics + IPOPT — against the
shared seven-DOF swing model from :mod:`.model_provider` instead of a
hard-coded two-link arm.

Two transcriptions share one formulation on the ``SwingOptimizer`` node
grid (decision layout ``[angles.flatten(), velocities.flatten()]`` so warm
starts and result extraction interoperate with the scipy path):

- ``"finite_difference"`` (legacy, #9756): central finite differences tie
  ``V`` to ``Q`` and accelerations to ``V``; torques are *evaluated* from
  RNEA at each node and bounded. The ODE is never enforced between nodes,
  so this is a smoothed kinematic fit with a torque check rather than a
  transcription. :func:`dynamics_defect` measures the gap.
- ``"multiple_shooting"``: torques are decision variables and RK4 shooting
  constraints ``x_{k+1} = Phi(x_k, tau_k, dt)`` enforce the forward
  dynamics on every interval. Same objective, bounds and result layout.

Dynamics come from a CasADi-symbolic recursive Newton-Euler (RNEA) over
the same chain the URDF bridge emits, with the **anthropometric** link
inertials of :func:`model_provider.swing_link_inertials` (#9755) so the
torque limits are physically meaningful. Forward dynamics is
``M(q)^-1 (tau - h(q, v))`` with ``M`` assembled from RNEA columns. Both
kernels are validated against Pinocchio's ``pin.rnea`` / ``pin.aba`` in the
live test suite.

Objective: maximize terminal clubhead speed with an effort integral and
the smooth injury surrogate's velocity/torque terms (B1/#8396) — all
differentiable by construction.

``casadi`` is an opt-in dependency (the ``optimal-control`` extra); when
absent, :func:`require_casadi` raises with an install hint.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Literal

import numpy as np

from src.shared.python.motion_pipeline.model_bridge import LinkInertial
from src.shared.python.optimization._swing_kinematics import JOINTS
from src.shared.python.optimization._swing_models import (
    ClubModel,
    GolferModel,
    OptimizationConfig,
    OptimizationObjective,
)
from src.shared.python.optimization._swing_reference import (
    ReferenceControls,
    ReferenceEndpoint,
    Rhs,
    _positive_real,
    finite_real_array,
    finite_vector,
    positive_integer,
    resolved_endpoint,
)
from src.shared.python.optimization.model_provider import (
    build_swing_rig,
    swing_link_inertials,
)

_INSTALL_HINT = (
    "casadi is not installed. Install the optimal-control extra: "
    "pip install 'upstream-drift[optimal-control]'"
)

Transcription = Literal["finite_difference", "multiple_shooting"]

_GRAVITY = np.array([0.0, 0.0, -9.81])

# Numerical reference defaults in rad / rad/s, never measurement uncertainty.
_REFERENCE_RTOL = 1e-8
_REFERENCE_POSITION_ATOL = 1e-10
_REFERENCE_VELOCITY_ATOL = 1e-10
_REFERENCE_STEPS_PER_INTERVAL = 32

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
    """Solver outcome in the flagship optimizer's conventions.

    ``torques`` (``n_joints x (n_nodes - 1)``) is populated by the
    multiple-shooting transcription and by the bioptim backend; the
    finite-difference path leaves it ``None`` because its torques are not
    decision variables.
    """

    success: bool
    x: np.ndarray
    fun: float
    message: str
    iterations: int
    torques: np.ndarray | None = None
    transcription: str = "finite_difference"


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


def _link_inertial_arrays(
    golfer: GolferModel,
    club: ClubModel,
    link_inertials: Mapping[str, LinkInertial] | None,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """``(mass, com, inertia_3x3)`` per DOF in ``JOINTS`` order."""
    inertials = (
        swing_link_inertials(golfer, club)
        if link_inertials is None
        else dict(link_inertials)
    )
    missing = [name for name in JOINTS if name not in inertials]
    if missing:
        raise ValueError(f"link_inertials missing entries for joints: {missing}")
    return [
        (
            float(inertials[name].mass),
            np.asarray(inertials[name].com, dtype=float),
            np.asarray(inertials[name].inertia_matrix, dtype=float),
        )
        for name in JOINTS
    ]


def _rotation(ca: Any, axis: np.ndarray, angle: Any) -> Any:
    """Rodrigues rotation about a fixed unit axis."""
    k = ca.SX(axis)
    kx = ca.skew(k)
    return ca.SX.eye(3) + ca.sin(angle) * kx + (1 - ca.cos(angle)) * (kx @ kx)


def build_symbolic_rnea(
    golfer: GolferModel,
    club: ClubModel,
    ca: Any | None = None,
    *,
    link_inertials: Mapping[str, LinkInertial] | None = None,
) -> Any:
    """CasADi Function ``tau = rnea(q, v, a)`` for the swing chain.

    Classic recursive Newton-Euler for a fixed-base serial chain of
    single-axis revolute joints: forward pass propagates angular velocity,
    angular acceleration, and linear acceleration (including gravity);
    backward pass accumulates forces/moments about each link origin and
    projects onto each joint axis.

    Args:
        golfer: Anthropometrics (chain geometry and, by default, inertials).
        club: Club parameters (chain geometry and clubhead / shaft masses).
        ca: The ``casadi`` module (imported on demand when ``None``).
        link_inertials: ``joint name -> LinkInertial`` in the link frame.
            Defaults to :func:`model_provider.swing_link_inertials`, the same
            inertials the URDF the engine backends consume carries (#9755).
            Pass :func:`model_provider.placeholder_link_inertials` to
            reproduce the pre-#9755 generic ``mass=1, I=1e-2`` chain.

    Postcondition: agrees with ``pin.rnea`` on
    ``model_provider.build_pinocchio_model`` built from the same inertials
    (live test).
    """
    ca = ca or require_casadi()
    offsets, axes = _chain_parameters(golfer, club)
    inertials = _link_inertial_arrays(golfer, club, link_inertials)
    n = len(JOINTS)

    q = ca.SX.sym("q", n)
    v = ca.SX.sym("v", n)
    a = ca.SX.sym("a", n)
    tau = rnea_expression(ca, q, v, a, offsets, axes, inertials)
    return ca.Function("swing_rnea", [q, v, a], [tau])


def rnea_expression(
    ca: Any,
    q: Any,
    v: Any,
    a: Any,
    offsets: list[np.ndarray],
    axes: list[np.ndarray],
    inertials: list[tuple[float, np.ndarray, np.ndarray]],
) -> Any:
    """Symbolic RNEA torques for the chain (all quantities in link frames)."""
    n = len(axes)
    omegas, domegas, accs = [], [], []
    omega = ca.SX.zeros(3)
    domega = ca.SX.zeros(3)
    acc = ca.SX(-_GRAVITY)  # base acceleration trick: a0 = -g
    rotations = []
    for i in range(n):
        r_parent_child = _rotation(ca, axes[i], q[i])
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

    # Backward pass: forces/moments about each link origin.
    force = ca.SX.zeros(3)
    moment = ca.SX.zeros(3)
    taus: list[Any] = [None] * n
    for i in reversed(range(n)):
        mass, com_np, inertia_np = inertials[i]
        com = ca.SX(com_np)
        inertia = ca.SX(inertia_np)
        # COM acceleration from the link-origin acceleration.
        acc_com = (
            accs[i]
            + ca.cross(domegas[i], com)
            + ca.cross(omegas[i], ca.cross(omegas[i], com))
        )
        f_inertial = mass * acc_com
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
    return ca.vertcat(*taus)


def build_mass_matrix(
    golfer: GolferModel,
    club: ClubModel,
    ca: Any | None = None,
    *,
    link_inertials: Mapping[str, LinkInertial] | None = None,
) -> Any:
    """CasADi Function ``M = mass_matrix(q)`` via CRBA-by-RNEA.

    Column ``i`` of ``M`` is ``rnea(q, 0, e_i) - rnea(q, 0, 0)``; with seven
    DOFs the dense assembly is cheap and exact (validated against
    ``pin.crba``).
    """
    ca = ca or require_casadi()
    rnea = build_symbolic_rnea(golfer, club, ca, link_inertials=link_inertials)
    n = len(JOINTS)
    q = ca.SX.sym("q", n)
    zero = ca.SX.zeros(n)
    bias = rnea(q, zero, zero)
    columns = [rnea(q, zero, ca.SX.eye(n)[:, i]) - bias for i in range(n)]
    return ca.Function("swing_mass_matrix", [q], [ca.horzcat(*columns)])


def build_forward_dynamics(
    golfer: GolferModel,
    club: ClubModel,
    ca: Any | None = None,
    *,
    link_inertials: Mapping[str, LinkInertial] | None = None,
) -> Any:
    """CasADi Function ``qddot = forward_dynamics(q, v, tau)``.

    Solves ``M(q) qddot = tau - h(q, v)`` with ``h = rnea(q, v, 0)``
    (validated against ``pin.aba``).
    """
    ca = ca or require_casadi()
    rnea = build_symbolic_rnea(golfer, club, ca, link_inertials=link_inertials)
    mass_matrix = build_mass_matrix(golfer, club, ca, link_inertials=link_inertials)
    n = len(JOINTS)
    q = ca.SX.sym("q", n)
    v = ca.SX.sym("v", n)
    tau = ca.SX.sym("tau", n)
    h = rnea(q, v, ca.SX.zeros(n))
    qddot = ca.solve(mass_matrix(q), tau - h)
    return ca.Function("swing_forward_dynamics", [q, v, tau], [qddot])


def _rk4_step(
    ca: Any, forward_dynamics: Any, q: Any, v: Any, tau: Any, dt: float
) -> tuple[Any, Any]:
    """One explicit RK4 step of ``(q, v)`` under constant torque ``tau``."""

    def rhs(qk: Any, vk: Any) -> tuple[Any, Any]:
        return vk, forward_dynamics(qk, vk, tau)

    k1q, k1v = rhs(q, v)
    k2q, k2v = rhs(q + 0.5 * dt * k1q, v + 0.5 * dt * k1v)
    k3q, k3v = rhs(q + 0.5 * dt * k2q, v + 0.5 * dt * k2v)
    k4q, k4v = rhs(q + dt * k3q, v + dt * k3v)
    q_next = q + dt / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
    v_next = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return q_next, v_next


def build_rk4_integrator(
    golfer: GolferModel,
    club: ClubModel,
    dt: float,
    ca: Any | None = None,
    *,
    n_substeps: int = 1,
    link_inertials: Mapping[str, LinkInertial] | None = None,
) -> Any:
    """CasADi Function ``(q1, v1) = step(q0, v0, tau)`` over one interval."""
    ca = ca or require_casadi()
    if dt <= 0:
        raise ValueError("dt must be positive")
    if n_substeps < 1:
        raise ValueError("n_substeps must be at least 1")
    fd = build_forward_dynamics(golfer, club, ca, link_inertials=link_inertials)
    n = len(JOINTS)
    q = ca.SX.sym("q", n)
    v = ca.SX.sym("v", n)
    tau = ca.SX.sym("tau", n)
    qk, vk = q, v
    for _ in range(n_substeps):
        qk, vk = _rk4_step(ca, fd, qk, vk, tau, dt / n_substeps)
    return ca.Function("swing_rk4_step", [q, v, tau], [qk, vk])


def _rollout(
    step: Any, q_start: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-torque forward rollout from rest at ``q_start`` (numeric)."""
    n = q_start.shape[0]
    q = np.zeros((n, n_nodes))
    v = np.zeros((n, n_nodes))
    q[:, 0] = q_start
    tau = np.zeros(n)
    for k in range(n_nodes - 1):
        q_next, v_next = step(q[:, k], v[:, k], tau)
        q[:, k + 1] = np.asarray(q_next).reshape(-1)
        v[:, k + 1] = np.asarray(v_next).reshape(-1)
    return q, v


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    """Central differences with one-sided endpoints (the FD path's stencil)."""
    out = np.empty_like(values)
    out[:, 0] = (values[:, 1] - values[:, 0]) / dt
    out[:, -1] = (values[:, -1] - values[:, -2]) / dt
    if values.shape[1] > 2:
        out[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2 * dt)
    return out


@dataclass(frozen=True)
class DynamicsDefectReport:
    """Local interval discrepancies in angular position and angular velocity.

    Each interval integrates the forward dynamics from ``(q_k, v_k)`` with
    the interval torque held constant and compares the result with
    ``(q_{k+1}, v_{k+1})``. Explicit fixed-grid residuals concern a discrete
    map only. Default adaptive results retain endpoint refinement diagnostics;
    neither path certifies physical realizability or whole-trajectory error.
    """

    position_defects: np.ndarray
    velocity_defects: np.ndarray
    torques: np.ndarray
    reference_resolution: tuple[ReferenceEndpoint, ...] | None = None

    @property
    def max_position_defect(self) -> float:
        return (
            float(np.max(self.position_defects)) if self.position_defects.size else 0.0
        )

    @property
    def max_velocity_defect(self) -> float:
        return (
            float(np.max(self.velocity_defects)) if self.velocity_defects.size else 0.0
        )

    @property
    def rms_position_defect(self) -> float:
        return _rms(self.position_defects)

    @property
    def rms_velocity_defect(self) -> float:
        return _rms(self.velocity_defects)

    @property
    def max_defect(self) -> float:
        """Legacy mixed-unit maximum; use separate component budgets for accuracy."""
        return max(self.max_position_defect, self.max_velocity_defect)

    def to_dict(self) -> dict[str, float]:
        return {
            "max_position_defect": self.max_position_defect,
            "max_velocity_defect": self.max_velocity_defect,
            "rms_position_defect": self.rms_position_defect,
            "rms_velocity_defect": self.rms_velocity_defect,
        }


def _rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def dynamics_defect(
    golfer: GolferModel,
    club: ClubModel,
    config: OptimizationConfig,
    x: np.ndarray,
    *,
    torques: np.ndarray | None = None,
    n_substeps: int | None = None,
    link_inertials: Mapping[str, LinkInertial] | None = None,
    reference_controls: ReferenceControls | None = None,
) -> DynamicsDefectReport:
    """Compare node endpoints with independently refined ODE integration.

    Args:
        golfer, club, config: The model and node grid the trajectory lives on.
        x: Decision vector ``[angles.flatten(), velocities.flatten()]``.
        torques: Optional ``n_joints x (n_nodes - 1)`` interval torques. When
            ``None`` (the finite-difference path) the torques are what the FD
            stencil implies: ``rnea(q_k, v_k, a_k)`` with ``a_k`` the central
            difference of ``v``.
        n_substeps: None selects the adaptive reference. A positive integer
            explicitly selects a fixed RK4 map for discrete-feasibility checks,
            without any implied reference resolution or continuous accuracy.
        reference_controls: Optional adaptive numerical settings. Incompatible
            with a fixed grid. Defaults are rtol=1e-8, position/velocity atol=
            1e-10 rad/rad/s and max_step=interval/32, then refined by the shared
            endpoint operation. Local solver tolerances are 10/100 times tighter
            than these endpoint budgets. Failure or unresolved refinement raises.

    Returns:
        Per-interval infinity-norm position and velocity defects.
    """
    if not isinstance(golfer, GolferModel) or not isinstance(club, ClubModel):
        raise TypeError("expected GolferModel and ClubModel")
    trajectory = _defect_trajectory(config, x)
    controls = _defect_reference_controls(n_substeps, reference_controls, trajectory[2])
    ca = require_casadi()
    applied = _defect_torque_values(
        (golfer, club, ca), trajectory, torques, link_inertials
    )
    if controls is not None:
        forward = build_forward_dynamics(
            golfer, club, ca, link_inertials=link_inertials
        )
        endpoints, resolution = _reference_endpoints(
            forward, trajectory, applied, controls
        )
    else:
        assert n_substeps is not None  # Validated explicit fixed-grid mode.
        step = build_rk4_integrator(
            golfer,
            club,
            trajectory[2],
            ca,
            n_substeps=n_substeps,
            link_inertials=link_inertials,
        )
        endpoints, resolution = _fixed_endpoints(step, trajectory, applied), None
    return _defect_report(endpoints, trajectory, applied, resolution)


def _defect_trajectory(
    config: OptimizationConfig, x: object
) -> tuple[np.ndarray, np.ndarray, float]:
    if not isinstance(config, OptimizationConfig):
        raise TypeError("config must be OptimizationConfig")
    nodes = positive_integer(config.n_nodes, "n_nodes")
    if nodes < 2:
        raise ValueError("config.n_nodes must be at least 2")
    duration = _positive_real(config.swing_duration, "swing_duration")
    values = finite_vector(x, "trajectory")
    count = len(JOINTS)
    if values.size != 2 * count * nodes:
        raise ValueError(f"x must have length {2 * count * nodes}, got {values.size}")
    return (
        values[: count * nodes].reshape(count, nodes),
        values[count * nodes :].reshape(count, nodes),
        duration / (nodes - 1),
    )


def _defect_reference_controls(
    n_substeps: int | None, controls: ReferenceControls | None, duration: float
) -> ReferenceControls | None:
    if n_substeps is not None:
        positive_integer(n_substeps, "n_substeps")
        if controls is not None:
            raise ValueError("reference_controls cannot accompany a fixed grid")
        return None
    if controls is not None and not isinstance(controls, ReferenceControls):
        raise TypeError("reference_controls must be ReferenceControls")
    return (
        controls
        if controls is not None
        else ReferenceControls(
            _REFERENCE_RTOL,
            _REFERENCE_POSITION_ATOL,
            _REFERENCE_VELOCITY_ATOL,
            duration / _REFERENCE_STEPS_PER_INTERVAL,
        )
    )


def _defect_torque_values(
    models: tuple[GolferModel, ClubModel, Any],
    trajectory: tuple[np.ndarray, np.ndarray, float],
    torques: object,
    inertials: Mapping[str, LinkInertial] | None,
) -> np.ndarray:
    q, v, dt = trajectory
    if torques is None:
        golfer, club, ca = models
        rnea = build_symbolic_rnea(golfer, club, ca, link_inertials=inertials)
        acceleration = _finite_difference(v, dt)
        torques = np.column_stack(
            [
                finite_real_array(
                    rnea(q[:, k], v[:, k], acceleration[:, k]), "derived torque"
                ).reshape(-1)
                for k in range(q.shape[1] - 1)
            ]
        )
    values = finite_real_array(torques, "torques")
    shape = (q.shape[0], q.shape[1] - 1)
    if values.shape != shape:
        raise ValueError(f"torques must have shape {shape}")
    return values


def _angular_rhs(forward: Any, torque: np.ndarray) -> Rhs:
    applied = torque.copy()
    applied.setflags(write=False)
    count = applied.size

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        acceleration = finite_real_array(
            forward(state[:count], state[count:], applied), "acceleration"
        ).reshape(-1)
        return np.r_[state[count:], acceleration]

    return rhs


def _reference_endpoints(
    forward: Any,
    trajectory: tuple[np.ndarray, np.ndarray, float],
    torques: np.ndarray,
    controls: ReferenceControls,
) -> tuple[np.ndarray, tuple[ReferenceEndpoint, ...]]:
    q, v, dt = trajectory
    results = tuple(
        resolved_endpoint(
            _angular_rhs(forward, torques[:, k]), np.r_[q[:, k], v[:, k]], dt, controls
        )
        for k in range(q.shape[1] - 1)
    )
    return np.column_stack([result.state for result in results]), results


def _fixed_endpoints(
    step: Any, trajectory: tuple[np.ndarray, np.ndarray, float], torques: np.ndarray
) -> np.ndarray:
    q, v, _ = trajectory
    endpoints = []
    for k in range(q.shape[1] - 1):
        q_next, v_next = step(q[:, k], v[:, k], torques[:, k])
        position = finite_real_array(q_next, "fixed-grid position").reshape(-1)
        velocity = finite_real_array(v_next, "fixed-grid velocity").reshape(-1)
        if position.shape != (q.shape[0],) or velocity.shape != (v.shape[0],):
            raise ValueError("fixed-grid endpoint shape differs from the state")
        endpoints.append(np.r_[position, velocity])
    return np.column_stack(endpoints)


def _defect_report(
    endpoints: np.ndarray,
    trajectory: tuple[np.ndarray, np.ndarray, float],
    torques: np.ndarray,
    resolution: tuple[ReferenceEndpoint, ...] | None,
) -> DynamicsDefectReport:
    q, v, _ = trajectory
    count = q.shape[0]
    with np.errstate(over="raise", invalid="raise"):
        position = np.max(np.abs(endpoints[:count] - q[:, 1:]), axis=0)
        velocity = np.max(np.abs(endpoints[count:] - v[:, 1:]), axis=0)
    return DynamicsDefectReport(position, velocity, torques, resolution)


def build_clubhead_position(
    golfer: GolferModel, club: ClubModel, ca: Any | None = None
) -> Any:
    """CasADi Function ``p = clubhead(q)`` — terminal joint origin FK."""
    ca = ca or require_casadi()
    offsets, axes = _chain_parameters(golfer, club)

    q = ca.SX.sym("q", len(JOINTS))
    p = ca.SX.zeros(3)
    r_world = ca.SX.eye(3)
    for i in range(len(JOINTS)):
        p = p + r_world @ ca.SX(offsets[i])
        r_world = r_world @ _rotation(ca, axes[i], q[i])
    return ca.Function("clubhead_position", [q], [p])


@dataclass(frozen=True)
class CasadiSolveOptions:
    """How :func:`solve_swing_casadi` discretises and models the swing.

    Grouped rather than passed loose so the entry point stays inside the
    architecture parameter budget, and so the CasADi backend takes its
    options the same way the bioptim one does
    (``ocp.swing_ocp.MaxSpeedOcpOptions``).

    Attributes:
        transcription: ``"finite_difference"`` (legacy kinematic fit with a
            torque check, deprecated -- it does not enforce the dynamics
            between nodes) or ``"multiple_shooting"`` (RK4 shooting
            constraints with torque decision variables, #9756).
        link_inertials: Override the anthropometric link inertials.
        n_substeps: RK4 substeps per shooting interval (multiple shooting
            only). Four keeps the integration error well below the node
            spacing at swing speeds; :func:`dynamics_defect` re-integrates
            on a finer grid to report the residual discretisation error.
    """

    transcription: Transcription = "finite_difference"
    link_inertials: Mapping[str, LinkInertial] | None = None
    n_substeps: int = 4

    def __post_init__(self) -> None:
        if self.transcription not in ("finite_difference", "multiple_shooting"):
            raise ValueError(
                "transcription must be 'finite_difference' or "
                f"'multiple_shooting', got {self.transcription!r}"
            )
        if self.n_substeps < 1:
            raise ValueError("n_substeps must be positive")


def _apply_decision_bounds(
    opti: Any,
    q_var: Any,
    v_var: Any,
    golfer: GolferModel,
    joint_limits: dict[str, tuple[float, float]],
) -> None:
    """Joint ROM (scaled by flexibility) and velocity bounds."""
    flex = golfer.flexibility_factor
    for j, joint in enumerate(JOINTS):
        lo, hi = joint_limits[joint]
        opti.subject_to(opti.bounded(lo * flex, q_var[j, :], hi * flex))
    opti.subject_to(opti.bounded(-_VELOCITY_BOUND, v_var, _VELOCITY_BOUND))


def _finite_difference_torques(
    opti: Any, q_var: Any, v_var: Any, dt: float, rnea: Any
) -> list[Any]:
    """Tie velocities to pose differences and read torques off via RNEA.

    Central differences in the interior, one-sided at the endpoints. This
    constrains the *kinematics* only: nothing here makes the trajectory
    dynamically feasible, which is why the path is deprecated (#9756).
    """
    n_nodes = q_var.shape[1]

    def _difference(var: Any, k: int) -> Any:
        if k == 0:
            return (var[:, 1] - var[:, 0]) / dt
        if k == n_nodes - 1:
            return (var[:, -1] - var[:, -2]) / dt
        return (var[:, k + 1] - var[:, k - 1]) / (2 * dt)

    for k in range(n_nodes):
        opti.subject_to(v_var[:, k] == _difference(q_var, k))
    return [
        rnea(q_var[:, k], v_var[:, k], _difference(v_var, k)) for k in range(n_nodes)
    ]


def _shooting_torques(
    opti: Any, q_var: Any, v_var: Any, tau_var: Any, step: Any
) -> list[Any]:
    """RK4 shooting constraints; torques are decision variables."""
    n_nodes = q_var.shape[1]
    for k in range(n_nodes - 1):
        q_next, v_next = step(q_var[:, k], v_var[:, k], tau_var[:, k])
        opti.subject_to(q_var[:, k + 1] == q_next)
        opti.subject_to(v_var[:, k + 1] == v_next)
    # The terminal node reuses the last interval torque for the smooth
    # risk term so every node contributes exactly once.
    torques = [tau_var[:, k] for k in range(n_nodes - 1)]
    torques.append(tau_var[:, n_nodes - 2])
    return torques


def _effort_and_risk(
    ca: Any,
    opti: Any,
    node_torques: list[Any],
    v_var: Any,
    limits_vec: np.ndarray,
    dt: float,
    *,
    terminal_node_has_effort: bool,
) -> tuple[Any, Any]:
    """Bound the node torques and accumulate the effort and risk terms."""
    n_nodes = len(node_torques)
    n_joints = limits_vec.shape[0]
    effort: Any = 0.0
    smooth_risk: Any = 0.0
    for k, tau_k in enumerate(node_torques):
        opti.subject_to(opti.bounded(-limits_vec, tau_k, limits_vec))
        if terminal_node_has_effort or k < n_nodes - 1:
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
    return effort, smooth_risk / n_nodes


def _terminal_speed(ca: Any, clubhead: Any, q_var: Any, v_var: Any) -> Any:
    """Clubhead speed at the last node, via an FK Jacobian-vector product."""
    n_joints = q_var.shape[0]
    q_sym = ca.SX.sym("q", n_joints)
    v_sym = ca.SX.sym("v", n_joints)
    p_sym = clubhead(q_sym)
    speed_fn = ca.Function(
        "clubhead_speed",
        [q_sym, v_sym],
        [ca.norm_2(ca.jtimes(p_sym, q_sym, v_sym))],
    )
    return speed_fn(q_var[:, -1], v_var[:, -1])


def _casadi_result(
    solution: Any,
    variables: tuple[Any, Any, Any | None],
    objective: Any,
    transcription: Transcription,
) -> CasadiSwingResult:
    """Read the solved values back into the flagship result shape."""
    q_var, v_var, tau_var = variables
    n_joints, n_nodes = q_var.shape
    q_opt = np.asarray(solution.value(q_var), dtype=float).reshape(n_joints, n_nodes)
    v_opt = np.asarray(solution.value(v_var), dtype=float).reshape(n_joints, n_nodes)
    torques = None
    if tau_var is not None:
        torques = np.asarray(solution.value(tau_var), dtype=float).reshape(
            n_joints, n_nodes - 1
        )
    stats = solution.stats()
    return CasadiSwingResult(
        success=bool(stats.get("success", True)),
        x=np.concatenate([q_opt.flatten(), v_opt.flatten()]),
        fun=float(solution.value(objective)),
        message=str(stats.get("return_status", "solved")),
        iterations=int(stats.get("iter_count", 0)),
        torques=torques,
        transcription=transcription,
    )


def _validate_warm_start(
    config: OptimizationConfig, x0: np.ndarray, n_joints: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split the flagship decision vector into pose and velocity nodes."""
    n_nodes = config.n_nodes
    if n_nodes < 2:
        raise ValueError("config.n_nodes must be at least 2")
    expected = 2 * n_joints * n_nodes
    flat = np.asarray(x0, dtype=float).reshape(-1)
    if flat.shape[0] != expected:
        raise ValueError(
            f"x0 must have length {expected} "
            f"(2 * {n_joints} joints * {n_nodes} nodes), got {flat.shape[0]}"
        )
    q0_nodes = flat[: n_joints * n_nodes].reshape(n_joints, n_nodes)
    v0_nodes = flat[n_joints * n_nodes :].reshape(n_joints, n_nodes)
    return q0_nodes, v0_nodes, config.swing_duration / (n_nodes - 1)


def _apply_warm_start(
    opti: Any,
    variables: tuple[Any, Any, Any | None],
    nodes: tuple[np.ndarray, np.ndarray],
    step: Any | None,
) -> None:
    """Seed the decision variables, rolling out the dynamics when shooting."""
    q_var, v_var, tau_var = variables
    q0_nodes, v0_nodes = nodes
    if step is None:
        opti.set_initial(q_var, q0_nodes)
        opti.set_initial(v_var, v0_nodes)
        return
    # A warm start that violates the shooting constraints (the flagship
    # guess is kinematic, not dynamic) sends IPOPT into a long restoration
    # phase. Roll the passive dynamics out from the address pose instead:
    # feasible by construction, and the optimizer then only has to shape
    # torques.
    n_joints, n_nodes = q_var.shape
    q_roll, v_roll = _rollout(step, q0_nodes[:, 0], n_nodes)
    opti.set_initial(q_var, q_roll)
    opti.set_initial(v_var, v_roll)
    opti.set_initial(tau_var, np.zeros((n_joints, n_nodes - 1)))


def _weighted_objective(
    config: OptimizationConfig, terminal_speed: Any, smooth_risk: Any, effort: Any
) -> Any:
    """Combine the speed, injury and effort terms at the configured weights."""
    w_speed = config.objectives.get(OptimizationObjective.CLUBHEAD_VELOCITY, 1.0)
    w_injury = config.objectives.get(OptimizationObjective.INJURY_RISK, 0.0)
    w_energy = config.objectives.get(OptimizationObjective.ENERGY_EFFICIENCY, 0.0)
    return (
        -w_speed * terminal_speed / 50.0
        + w_injury * smooth_risk / 100.0
        + w_energy * effort / 1000.0
        + 1e-4 * effort  # regularizer keeps torques bounded when w_energy=0
    )


def _build_transcription(
    opti: Any,
    variables: tuple[Any, Any],
    models: tuple[GolferModel, ClubModel, Any],
    dt: float,
    options: CasadiSolveOptions,
) -> tuple[list[Any], Any | None, Any | None]:
    """Constrain the trajectory and return ``(node_torques, tau_var, step)``.

    ``tau_var`` and ``step`` are ``None`` on the finite-difference path,
    where torques are read off by inverse dynamics rather than decided.
    """
    q_var, v_var = variables
    golfer, club, ca = models
    inertials = options.link_inertials
    if options.transcription == "finite_difference":
        rnea = build_symbolic_rnea(golfer, club, ca, link_inertials=inertials)
        return _finite_difference_torques(opti, q_var, v_var, dt, rnea), None, None
    n_joints, n_nodes = q_var.shape
    tau_var = opti.variable(n_joints, n_nodes - 1)
    step = build_rk4_integrator(
        golfer, club, dt, ca, n_substeps=options.n_substeps, link_inertials=inertials
    )
    return _shooting_torques(opti, q_var, v_var, tau_var, step), tau_var, step


def _configure_ipopt(opti: Any, config: OptimizationConfig) -> None:
    """Silence IPOPT and cap its iterations at the configured budget."""
    opti.solver(
        "ipopt",
        {"print_time": False, "expand": True},
        {"max_iter": int(config.max_iterations), "print_level": 0, "sb": "yes"},
    )


def solve_swing_casadi(
    golfer: GolferModel,
    club: ClubModel,
    config: OptimizationConfig,
    torque_limits: dict[str, float],
    joint_limits: dict[str, tuple[float, float]],
    x0: np.ndarray,
    *,
    options: CasadiSolveOptions | None = None,
) -> CasadiSwingResult:
    """Solve the swing trajectory optimization with CasADi + IPOPT.

    Args:
        golfer: Anthropometrics (limits, flexibility, inertials).
        club: Club parameters (chain geometry, masses).
        config: Node count, duration, objective weights, solver options.
        torque_limits: Per-joint torque bounds [N*m].
        joint_limits: Per-joint (lower, upper) angle bounds [rad].
        x0: Warm-start decision vector in flagship layout.
        options: Transcription and modelling choices; see
            :class:`CasadiSolveOptions`. ``None`` keeps the historical
            default, which is the deprecated finite-difference path.

    Returns:
        CasadiSwingResult with the optimized decision vector.

    Raises:
        CasadiNotAvailableError: When casadi is not installed.
        ValueError: On malformed inputs.
    """
    options = options or CasadiSolveOptions()
    transcription = options.transcription
    finite_difference = transcription == "finite_difference"
    if finite_difference:
        warnings.warn(
            "the finite-difference transcription does not enforce the swing "
            "dynamics between nodes, so its torques and clubhead speeds are "
            "not physically realisable (see docs/estimation/bioptim_parity.md "
            "and #9756). Use transcription='multiple_shooting', or the "
            "'bioptim' backend for constrained and tracking problems.",
            DeprecationWarning,
            stacklevel=2,
        )
    ca = require_casadi()
    n_joints = len(JOINTS)
    n_nodes = config.n_nodes
    q0_nodes, v0_nodes, dt = _validate_warm_start(config, x0, n_joints)

    clubhead = build_clubhead_position(golfer, club, ca)
    opti = ca.Opti()
    q_var = opti.variable(n_joints, n_nodes)
    v_var = opti.variable(n_joints, n_nodes)
    _apply_decision_bounds(opti, q_var, v_var, golfer, joint_limits)

    # Boundary: start at the warm-start address pose, at rest.
    opti.subject_to(q_var[:, 0] == q0_nodes[:, 0])
    opti.subject_to(v_var[:, 0] == 0.0)

    limits_vec = np.array([torque_limits.get(j, 100.0) for j in JOINTS])
    node_torques, tau_var, step = _build_transcription(
        opti, (q_var, v_var), (golfer, club, ca), dt, options
    )

    effort, smooth_risk = _effort_and_risk(
        ca,
        opti,
        node_torques,
        v_var,
        limits_vec,
        dt,
        terminal_node_has_effort=finite_difference,
    )
    objective = _weighted_objective(
        config, _terminal_speed(ca, clubhead, q_var, v_var), smooth_risk, effort
    )
    opti.minimize(objective)
    _apply_warm_start(opti, (q_var, v_var, tau_var), (q0_nodes, v0_nodes), step)

    _configure_ipopt(opti, config)
    try:
        solution = opti.solve()
    except RuntimeError as exc:
        return CasadiSwingResult(
            success=False,
            x=x0,
            fun=float("nan"),
            message=f"IPOPT failed: {exc}",
            iterations=0,
            transcription=transcription,
        )
    return _casadi_result(solution, (q_var, v_var, tau_var), objective, transcription)
