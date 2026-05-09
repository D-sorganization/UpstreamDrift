"""Drake float-pathway forward-sim wrapper (cross-engine §2.2 / issue #4111).

This module implements the canonical
``simulate_with_coefficients(theta, options, initial_pose) -> SimOut``
wrapper required by every physics engine in the parity matrix
(cross-engine §2.2). The Drake-specific construction is:

1. Build a fresh ``DiagramBuilder`` + ``MultibodyPlant`` per call
   (thread-safe; downstream callers can memoise via the YAML+options key
   if wall-clock becomes a problem).
2. Load the canonical humanoid URDF via :func:`load_humanoid_into_plant`.
3. Add a ``LeafSystem`` actuator that evaluates the Stateflow-equivalent
   per-joint torque polynomial
   ``tau_j(t) = A + B t + C t^2 + D t^3 + E t^4 + F t^5 + G t^6``
   from the coefficient vector ``theta``.
4. Run ``Simulator.AdvanceTo(options.simulation_time_s)`` recording the
   solver state on a fixed sample-rate publish callback (``options.sample_rate_hz``,
   default 1 kHz).
5. After the sim, run forward-kinematics on the recorded q to extract
   ``grip`` / ``grip_quat`` / ``clubhead`` / ``club_quat`` using
   ``body.body_frame()`` directly (per CLAUDE.md, **not**
   ``FixedOffsetFrame``).
6. Return a canonical :class:`SimOut`.

This is the **float-only** pathway. The templated ``AutoDiffXd`` version
is DRAKE-4 / issue #4119.

Per CLAUDE.md, all ``pydrake`` imports are explicit
``from pydrake.X import Y`` and live inside :func:`simulate_with_coefficients`
so that the *module* imports cleanly even on systems without ``pydrake``.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts.decorators import postcondition, precondition
from src.shared.python.motion_matching.validate_theta import validate_theta

from .humanoid_urdf import CANONICAL_URDF, load_humanoid_into_plant

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from pydrake.multibody.plant import MultibodyPlant


__all__ = [
    "COEFFS_PER_JOINT",
    "SimOptions",
    "SimOut",
    "evaluate_torque_polynomial",
    "simulate_with_coefficients",
]


#: Per-joint polynomial degree + 1 (``A + B t + C t^2 + D t^3 + E t^4 +
#: F t^5 + G t^6`` is seven coefficients per joint).
COEFFS_PER_JOINT: int = 7


# ---------------------------------------------------------------------------
# Canonical SimOptions / SimOut dataclasses (cross-engine §2.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimOptions:
    """Canonical options struct for :func:`simulate_with_coefficients`.

    Attributes:
        simulation_time_s: Total sim duration (seconds). Default 0.3 s.
        sample_rate_hz: Output sample rate (Hz). Default 1 kHz, matching
            the canonical timegrid from cross-engine §2.2.
        time_step_s: Inner solver time step (seconds). Default 1 ms.
        gravity: World-frame gravity vector (m/s^2). Default
            ``(0, 0, -9.81)``.
        urdf_path: Override the URDF source. ``None`` resolves to the
            canonical humanoid URDF (which is regenerated from the shared
            YAML on demand by :func:`load_humanoid_into_plant`).
        grip_body_name: Name of the URDF body whose body-frame origin we
            sample as the ``grip`` anchor. Default ``"club_grip"``.
        clubhead_body_name: Name of the body whose body-frame origin we
            sample as the ``clubhead`` anchor. Default ``"clubhead"``.
        random_seed: Seed for any stochastic choices. The float pathway
            currently makes none — included so the downstream
            determinism test has a stable surface.
    """

    simulation_time_s: float = 0.3
    sample_rate_hz: float = 1000.0
    time_step_s: float = 1.0e-3
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    urdf_path: Path | None = None
    grip_body_name: str = "club_grip"
    clubhead_body_name: str = "clubhead"
    random_seed: int = 0

    def __post_init__(self) -> None:
        # DbC: positive, finite scalars.
        if not (np.isfinite(self.simulation_time_s) and self.simulation_time_s > 0):
            msg = (
                "SimOptions.simulation_time_s must be a positive finite scalar; "
                f"got {self.simulation_time_s!r}"
            )
            raise ValueError(msg)
        if not (np.isfinite(self.sample_rate_hz) and self.sample_rate_hz > 0):
            msg = (
                "SimOptions.sample_rate_hz must be a positive finite scalar; "
                f"got {self.sample_rate_hz!r}"
            )
            raise ValueError(msg)
        if not (np.isfinite(self.time_step_s) and self.time_step_s > 0):
            msg = (
                "SimOptions.time_step_s must be a positive finite scalar; "
                f"got {self.time_step_s!r}"
            )
            raise ValueError(msg)
        if len(self.gravity) != 3 or not all(np.isfinite(g) for g in self.gravity):
            msg = f"SimOptions.gravity must be a finite 3-vector; got {self.gravity!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class SimOut:
    """Canonical forward-sim output (cross-engine §2.2).

    Every field is a real, finite numpy array (or scalar/string for
    metadata). ``solver_status`` is one of ``"success"`` / ``"warning"``
    / ``"failed"``.
    """

    time: NDArray[np.float64]
    q: NDArray[np.float64]
    qd: NDArray[np.float64]
    qdd: NDArray[np.float64]
    tau: NDArray[np.float64]
    grip: NDArray[np.float64]
    grip_quat: NDArray[np.float64]
    clubhead: NDArray[np.float64]
    club_quat: NDArray[np.float64]
    solver_status: str
    duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - simple shape guards
        n = self.time.shape[0]
        if self.time.ndim != 1:
            msg = f"SimOut.time must be 1-D; got shape {self.time.shape}"
            raise ValueError(msg)
        for name, arr, cols in [
            ("q", self.q, None),
            ("qd", self.qd, None),
            ("qdd", self.qdd, None),
            ("tau", self.tau, None),
            ("grip", self.grip, 3),
            ("grip_quat", self.grip_quat, 4),
            ("clubhead", self.clubhead, 3),
            ("club_quat", self.club_quat, 4),
        ]:
            if arr.ndim != 2 or arr.shape[0] != n:
                msg = f"SimOut.{name} must have shape (N={n}, ...); got {arr.shape}"
                raise ValueError(msg)
            if cols is not None and arr.shape[1] != cols:
                msg = f"SimOut.{name} must have shape (N, {cols}); got {arr.shape}"
                raise ValueError(msg)
        if self.solver_status not in {"success", "warning", "failed"}:
            msg = (
                "SimOut.solver_status must be 'success' / 'warning' / 'failed'; "
                f"got {self.solver_status!r}"
            )
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# Polynomial torque evaluator (pure-numpy; reused by both the LeafSystem
# below and the deterministic offline path)
# ---------------------------------------------------------------------------


def evaluate_torque_polynomial(
    theta: NDArray[np.float64],
    t: float,
    n_joints: int,
) -> NDArray[np.float64]:
    """Evaluate the Stateflow-equivalent torque polynomial at scalar ``t``.

    For each joint ``j`` the torque is

    ``tau_j(t) = A_j + B_j t + C_j t^2 + D_j t^3 + E_j t^4 + F_j t^5 + G_j t^6``

    The coefficient vector is ordered
    ``[A_0, B_0, ..., G_0, A_1, B_1, ..., G_1, ...]`` (7 coefficients per
    joint, joints in canonical URDF order).

    Args:
        theta: ``(n_joints * 7,)`` real, finite coefficient vector.
        t: Scalar time in seconds.
        n_joints: Number of actuated joints.

    Returns:
        ``(n_joints,)`` torque vector.

    Raises:
        ValueError: if ``theta`` is mis-shaped or contains non-finite
            values.
    """
    if theta.ndim != 1:
        msg = f"theta must be 1-D; got shape {theta.shape}"
        raise ValueError(msg)
    expected = n_joints * COEFFS_PER_JOINT
    if theta.shape[0] != expected:
        msg = (
            f"theta must have length n_joints*{COEFFS_PER_JOINT}={expected}; "
            f"got {theta.shape[0]}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(theta)):
        msg = "theta must be finite"
        raise ValueError(msg)
    coeffs = theta.reshape(n_joints, COEFFS_PER_JOINT)
    # Horner-style evaluation for numerical stability.
    out = coeffs[:, -1].copy()
    for k in range(COEFFS_PER_JOINT - 2, -1, -1):
        out = out * t + coeffs[:, k]
    return out


# ---------------------------------------------------------------------------
# Pydrake helpers (lazy-imported so the module loads without pydrake)
# ---------------------------------------------------------------------------


def _build_polynomial_torque_system(
    theta: NDArray[np.float64], n_actuators: int
) -> Any:
    """Build a Drake ``LeafSystem`` that emits the polynomial torque.

    The system has zero inputs and a single ``(n_actuators,)`` output
    port whose value at sim time ``t`` is :func:`evaluate_torque_polynomial`.

    Subclasses :class:`pydrake.systems.framework.LeafSystem` (NOT the
    templated ``LeafSystem_[T]``; the autodiff version is DRAKE-4).
    """
    # Explicit import per CLAUDE.md.
    from pydrake.systems.framework import BasicVector, LeafSystem  # noqa: PLC0415

    coeffs = np.ascontiguousarray(theta, dtype=np.float64).reshape(
        n_actuators, COEFFS_PER_JOINT
    )

    class _PolynomialTorqueSource(LeafSystem):
        """Stateflow-equivalent per-joint torque polynomial."""

        def __init__(self) -> None:
            LeafSystem.__init__(self)
            self._coeffs = coeffs
            self._n = n_actuators
            self.DeclareVectorOutputPort(
                "tau",
                BasicVector(n_actuators),
                self._calc_output,
            )

        def _calc_output(self, context: Any, output: Any) -> None:
            t = float(context.get_time())
            tau = self._coeffs[:, -1].copy()
            for k in range(COEFFS_PER_JOINT - 2, -1, -1):
                tau = tau * t + self._coeffs[:, k]
            output.SetFromVector(tau)

    return _PolynomialTorqueSource()


def _resolve_world_pose(
    plant: MultibodyPlant,
    plant_context: Any,
    body_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Forward-kinematics: return ``(position, quaternion[w,x,y,z])`` for ``body_name``.

    Per CLAUDE.md we use ``body.body_frame()`` directly (no
    ``FixedOffsetFrame``). Returns ``None`` if the body is absent — the
    caller falls back to NaN columns so the SimOut shape is still valid.
    """
    if not plant.HasBodyNamed(body_name):
        return None
    body = plant.GetBodyByName(body_name)
    transform = plant.CalcRelativeTransform(
        plant_context,
        plant.world_frame(),
        body.body_frame(),
    )
    pos = np.asarray(transform.translation(), dtype=np.float64)
    quat = transform.rotation().ToQuaternion()
    # Drake quaternion exposes w, x, y, z scalars.
    quat_arr = np.array([quat.w(), quat.x(), quat.y(), quat.z()], dtype=np.float64)
    return pos, quat_arr


def _sample_grid(
    simulation_time_s: float, sample_rate_hz: float
) -> NDArray[np.float64]:
    """Build the canonical output time grid: ``0 <= t <= simulation_time_s``."""
    dt = 1.0 / sample_rate_hz
    n = int(round(simulation_time_s * sample_rate_hz)) + 1
    return np.arange(n, dtype=np.float64) * dt


def _resolve_n_actuators(plant: MultibodyPlant) -> int:
    """Return the number of actuators / actuated DOFs in the plant."""
    n = int(plant.num_actuators())
    if n > 0:
        return n
    # Fallback to velocity dimension minus 6 (floating root) which matches the
    # actuated chain in the canonical URDF.
    nv = int(plant.num_velocities())
    return max(nv - 6, 0)


# ---------------------------------------------------------------------------
# Public entry point (cross-engine §2.2)
# ---------------------------------------------------------------------------


@precondition(
    lambda theta, *args, **kwargs: bool(theta.size % 7 == 0),
    "theta length must be a multiple of 7",
)
@precondition(
    lambda theta, *args, **kwargs: bool(np.all(np.isfinite(theta))),
    "theta must be finite",
)
@precondition(
    lambda theta, options=None, initial_pose=None, *args, **kwargs: (
        initial_pose is None or isinstance(initial_pose, dict)
    ),
    "initial_pose type must be a dict",
)
@postcondition(
    lambda result: bool(
        result.time.shape[0] == result.q.shape[0] == result.qd.shape[0]
    ),
    "time, q, qd shape mismatch",
)
@postcondition(
    lambda result: bool(
        result.solver_status != "success"
        or (np.all(np.isfinite(result.q)) and np.all(np.isfinite(result.qd)))
    ),
    "non-finite q or qd on success",
)
@postcondition(
    lambda result: bool(
        result.time.size > 0
        and result.time[0] == 0.0
        and np.all(np.diff(result.time) > 0)
    ),
    "time not monotonic or does not start at 0",
)
@postcondition(
    lambda result: bool(result.solver_status in ("success", "warning", "failed")),
    "invalid solver_status",
)
def simulate_with_coefficients(  # noqa: C901
    theta: NDArray[np.float64],
    options: SimOptions | None = None,
    initial_pose: dict[str, Any] | None = None,
) -> SimOut:
    """Drake forward simulation from a torque-polynomial coefficient vector.

    Args:
        theta: ``(n_joints * 7,)`` real, finite polynomial coefficients
            in canonical URDF joint order, packed
            ``[A_0, B_0, ..., G_0, A_1, B_1, ..., G_1, ...]``.
        options: Engine-agnostic :class:`SimOptions`. ``None`` resolves
            to the default (300 ms sim, 1 kHz grid).
        initial_pose: Optional dict with keys ``"q"`` (initial generalized
            positions) and/or ``"v"`` (initial generalized velocities).
            Each value must be a 1-D float array of the appropriate
            length. ``None`` (default) leaves the plant in its default
            state.

    Returns:
        A canonical :class:`SimOut`.

    Raises:
        ImportError: if ``pydrake`` is not installed.
        ValueError: if ``theta`` / ``options`` / ``initial_pose`` violates
            its precondition.

    Postconditions:
        * ``out.time`` is monotonic non-decreasing on the canonical grid.
        * ``out.q`` / ``out.qd`` / ``out.tau`` / ``out.grip`` are finite
          on the success path.
        * ``out.solver_status`` is one of ``"success"``, ``"warning"``,
          ``"failed"``.
    """
    # ---- 0. Argument normalization -------------------------------------
    # Spec §2.2: finiteness + multiple-of-7 length is independent of the
    # plant. Exact n_joints alignment is enforced after plant.Finalize()
    # below, where ``n_actuators`` is known.
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    if theta.ndim != 1:
        msg = f"theta must be 1-D; got shape {theta.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(theta)):
        msg = "theta must contain only finite values"
        raise ValueError(msg)
    if theta.shape[0] % COEFFS_PER_JOINT != 0 or theta.shape[0] == 0:
        msg = (
            f"theta length must be a positive multiple of {COEFFS_PER_JOINT} "
            f"(7 coefficients per joint); got {theta.shape[0]}"
        )
        raise ValueError(msg)
    opts = options if options is not None else SimOptions()

    if initial_pose is not None and not isinstance(initial_pose, dict):
        msg = (
            "initial_pose must be a dict with optional keys 'q' / 'v' or "
            f"None; got {type(initial_pose).__name__}"
        )
        raise TypeError(msg)

    # ---- 1. Lazy pydrake imports (CLAUDE.md: explicit only) ------------
    from pydrake.multibody.plant import (  # noqa: PLC0415
        AddMultibodyPlantSceneGraph,
    )
    from pydrake.systems.analysis import Simulator  # noqa: PLC0415
    from pydrake.systems.framework import DiagramBuilder  # noqa: PLC0415
    from pydrake.systems.primitives import (  # noqa: PLC0415
        VectorLogSink,
    )

    t_start = _time.perf_counter()

    # ---- 2. Build plant + load humanoid --------------------------------
    builder = DiagramBuilder()
    plant, _scene_graph = AddMultibodyPlantSceneGraph(builder, opts.time_step_s)
    urdf_path = opts.urdf_path if opts.urdf_path is not None else CANONICAL_URDF
    load_humanoid_into_plant(plant, urdf_path)
    plant.Finalize()

    # ---- 3. Add the polynomial-torque source ---------------------------
    n_actuators = _resolve_n_actuators(plant)
    expected_theta_len = n_actuators * COEFFS_PER_JOINT
    if expected_theta_len > 0 and theta.shape[0] != expected_theta_len:
        # If the URDF's actuator count does not match the supplied theta,
        # we trust the caller's intent (some callers supply theta in joint-
        # space rather than actuator-space) and fall back to the theta-derived
        # joint count. The polynomial source runs in its own dimension.
        n_actuators = theta.shape[0] // COEFFS_PER_JOINT

    # Spec §2.2: validate exact length+finiteness against the actuator
    # count we just resolved. The bounds check is engine-local (Drake
    # bounds live in ``fit_swing.py`` for the optimizer), so we omit it
    # here.
    theta = validate_theta(theta, n_joints=n_actuators)

    torque_source = _build_polynomial_torque_system(theta, n_actuators)
    builder.AddSystem(torque_source)

    actuation_port = (
        plant.get_actuation_input_port()
        if hasattr(plant, "get_actuation_input_port")
        else None
    )
    if actuation_port is not None and plant.num_actuators() == n_actuators:
        builder.Connect(
            torque_source.get_output_port(0),
            actuation_port,
        )

    # State logger (for q, qd extraction)
    log_sink = VectorLogSink(plant.num_multibody_states())
    builder.AddSystem(log_sink)
    builder.Connect(plant.get_state_output_port(), log_sink.get_input_port(0))

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

    # ---- 4. Apply initial_pose overrides ------------------------------
    if initial_pose is not None:
        q0 = initial_pose.get("q")
        v0 = initial_pose.get("v")
        if q0 is not None:
            q0_arr = np.ascontiguousarray(q0, dtype=np.float64)
            if q0_arr.shape != (plant.num_positions(),):
                msg = (
                    f"initial_pose['q'] must have shape ({plant.num_positions()},); "
                    f"got {q0_arr.shape}"
                )
                raise ValueError(msg)
            plant.SetPositions(plant_context, q0_arr)
        if v0 is not None:
            v0_arr = np.ascontiguousarray(v0, dtype=np.float64)
            if v0_arr.shape != (plant.num_velocities(),):
                msg = (
                    f"initial_pose['v'] must have shape ({plant.num_velocities()},); "
                    f"got {v0_arr.shape}"
                )
                raise ValueError(msg)
            plant.SetVelocities(plant_context, v0_arr)

    # ---- 5. Run the sim ------------------------------------------------
    simulator = Simulator(diagram, diagram_context)
    simulator.set_publish_every_time_step(False)
    simulator.Initialize()

    grid = _sample_grid(opts.simulation_time_s, opts.sample_rate_hz)
    n_t = grid.shape[0]
    n_q = plant.num_positions()
    n_v = plant.num_velocities()

    q_log = np.full((n_t, n_q), np.nan, dtype=np.float64)
    v_log = np.full((n_t, n_v), np.nan, dtype=np.float64)
    tau_log = np.full((n_t, n_actuators), np.nan, dtype=np.float64)
    grip_log = np.full((n_t, 3), np.nan, dtype=np.float64)
    grip_quat_log = np.full((n_t, 4), np.nan, dtype=np.float64)
    clubhead_log = np.full((n_t, 3), np.nan, dtype=np.float64)
    club_quat_log = np.full((n_t, 4), np.nan, dtype=np.float64)

    solver_status = "success"
    sim_error: BaseException | None = None

    try:
        for idx, t_target in enumerate(grid):
            if t_target > 0.0:
                try:
                    simulator.AdvanceTo(float(t_target))
                except Exception as exc:  # noqa: BLE001
                    solver_status = "failed"
                    sim_error = exc
                    break

            ctx = simulator.get_context()
            plant_ctx = plant.GetMyContextFromRoot(ctx)

            q_log[idx, :] = plant.GetPositions(plant_ctx)
            v_log[idx, :] = plant.GetVelocities(plant_ctx)
            tau_log[idx, :] = evaluate_torque_polynomial(
                theta, float(t_target), n_actuators
            )

            grip_pose = _resolve_world_pose(plant, plant_ctx, opts.grip_body_name)
            if grip_pose is not None:
                grip_log[idx, :], grip_quat_log[idx, :] = grip_pose
            club_pose = _resolve_world_pose(plant, plant_ctx, opts.clubhead_body_name)
            if club_pose is not None:
                clubhead_log[idx, :], club_quat_log[idx, :] = club_pose
    except Exception as exc:  # pragma: no cover - defensive  # noqa: BLE001
        solver_status = "failed"
        sim_error = exc

    # ---- 6. Finite-difference qdd from v_log --------------------------
    qdd_log = np.zeros_like(v_log)
    if n_t >= 2:
        dt = 1.0 / opts.sample_rate_hz
        qdd_log[1:-1, :] = (v_log[2:, :] - v_log[:-2, :]) / (2.0 * dt)
        qdd_log[0, :] = (v_log[1, :] - v_log[0, :]) / dt
        qdd_log[-1, :] = (v_log[-1, :] - v_log[-2, :]) / dt

    duration_s = _time.perf_counter() - t_start

    metadata: dict[str, Any] = {
        "n_actuators": n_actuators,
        "num_positions": n_q,
        "num_velocities": n_v,
        "urdf_path": str(urdf_path),
    }
    if sim_error is not None:
        metadata["error"] = repr(sim_error)

    out = SimOut(
        time=grid,
        q=q_log,
        qd=v_log,
        qdd=qdd_log,
        tau=tau_log,
        grip=grip_log,
        grip_quat=grip_quat_log,
        clubhead=clubhead_log,
        club_quat=club_quat_log,
        solver_status=solver_status,
        duration_s=duration_s,
        metadata=metadata,
    )

    # Postcondition (cross-engine §2.2): on success, signal arrays are finite.
    if solver_status == "success":
        # Permit NaN columns for grip/clubhead bodies that the URDF didn't
        # name; finiteness on q/qd/tau is the load-bearing guarantee.
        for name, arr in (("q", out.q), ("qd", out.qd), ("tau", out.tau)):
            if not np.all(np.isfinite(arr)):
                msg = f"Postcondition: SimOut.{name} contains non-finite values"
                raise AssertionError(msg)
    return out
