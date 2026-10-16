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

import logging
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts.decorators import postcondition, precondition
from src.shared.python.motion_matching.output_time_grid import build_output_grid
from src.shared.python.motion_matching.polynomial_torque import (
    COEFFS_PER_JOINT,
)
from src.shared.python.motion_matching.polynomial_torque import (
    POLY_DEGREE as _POLY_DEGREE,
)
from src.shared.python.motion_matching.polynomial_torque import (
    evaluate_polynomial_torque as _evaluate_polynomial_torque_matrix,
)
from src.shared.python.motion_matching.validate_theta import validate_theta

from ._simout_build import normalize_simout_args
from ._simulate_support import (
    RolloutConfig,
    allocate_logs,
    apply_initial_pose,
    assert_finite_on_success,
    check_initial_pose_type,
    check_theta_matches_actuators,
    evaluate_bernstein_torque,
    finite_difference_qdd,
    normalize_theta,
)
from ._simulate_support import (
    build_polynomial_torque_system as _build_polynomial_torque_system,
)
from ._simulate_support import record_rollout as _record_rollout
from .humanoid_urdf import CANONICAL_URDF, load_humanoid_into_plant

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from pydrake.multibody.plant import MultibodyPlant


logger = logging.getLogger(__name__)

POLY_DEGREE: int = _POLY_DEGREE

_CANONICAL_PINOCCHIO_URDF: Path = (
    Path(__file__).resolve().parents[2]
    / "pinocchio"
    / "models"
    / "generated"
    / "golfer.urdf"
)
DEFAULT_GOLFER_URDF: Path = (
    _CANONICAL_PINOCCHIO_URDF if _CANONICAL_PINOCCHIO_URDF.exists() else CANONICAL_URDF
)

GRIP_FRAME_NAME: str = "mid_hands"
CLUBHEAD_FRAME_NAME: str = "club_head"


__all__ = [
    "CLUBHEAD_FRAME_NAME",
    "COEFFS_PER_JOINT",
    "DEFAULT_GOLFER_URDF",
    "GRIP_FRAME_NAME",
    "POLY_DEGREE",
    "SimOptions",
    "SimOut",
    "evaluate_bernstein_torque",
    "evaluate_polynomial_torque",
    "evaluate_torque_polynomial",
    "is_drake_available",
    "simulate_with_coefficients",
]


# ---------------------------------------------------------------------------
# Runtime availability probe
# ---------------------------------------------------------------------------


def is_drake_available() -> bool:
    """Return whether a functional Drake C++ runtime (pydrake) is available."""
    try:
        from pydrake.systems.framework import DiagramBuilder  # noqa: PLC0415
    except ImportError:
        return False
    if type(DiagramBuilder).__module__ == "unittest.mock":
        return False
    return callable(DiagramBuilder)


# ---------------------------------------------------------------------------
# Polynomial & Quaternion helpers
# ---------------------------------------------------------------------------


def _quat_to_rotmat_series(quats: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert (N, 4) unit quaternions [w, x, y, z] to (N, 3, 3) rotation matrices."""
    from scipy.spatial.transform import Rotation as _Rotation

    q = np.asarray(quats, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quats must have shape (N, 4); got {q.shape}")
    return _Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


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
            ``(0, 0, -9.80665)``.
        urdf_path: Override the URDF source. ``None`` resolves to the
            canonical humanoid URDF.
        grip_body_name: Name of the URDF body whose body-frame origin we
            sample as the ``grip`` anchor. Default ``"club_grip"``.
        clubhead_body_name: Name of the body whose body-frame origin we
            sample as the ``clubhead`` anchor. Default ``"clubhead"``.
        random_seed: Seed for stochastic choices.
        basis: Basis used for torque parametrization: ``"power"`` or
            ``"bernstein"``. Default is ``"power"``.
        T_s: Horizon duration in seconds for Bernstein basis normalization.
        t0: Reference start time in seconds. Default 0.0.
        output_rate_hz: Optional output grid sample rate in Hz.
        compute_energy: Whether to populate kinetic and potential energy.
        compute_qdd: Whether to compute generalized accelerations. Default True.
        t_final: Horizon duration in seconds (alias for simulation_time_s / T_s).
        dt: Fixed integrator timestep in seconds (alias for time_step_s).
    """

    simulation_time_s: float = 0.3
    sample_rate_hz: float = 1000.0
    time_step_s: float = 1.0e-3
    gravity: tuple[float, float, float] | NDArray[np.float64] = (0.0, 0.0, -9.80665)
    urdf_path: Path | str | None = None
    grip_body_name: str = "club_grip"
    clubhead_body_name: str = "clubhead"
    random_seed: int = 0
    basis: Literal["power", "bernstein"] = "power"
    T_s: float | None = None
    t0: float = 0.0
    output_rate_hz: float | None = None
    compute_energy: bool = True
    compute_qdd: bool = True
    t_final: float | None = None
    dt: float | None = None

    def __post_init__(self) -> None:
        if self.t_final is not None:
            if not (np.isfinite(self.t_final) and self.t_final > 0):
                msg = f"SimOptions.t_final must be positive and finite; got {self.t_final!r}"
                raise ValueError(msg)
            object.__setattr__(self, "simulation_time_s", float(self.t_final))
        if self.T_s is not None:
            if not (np.isfinite(self.T_s) and self.T_s > 0):
                msg = f"SimOptions.T_s must be positive and finite; got {self.T_s!r}"
                raise ValueError(msg)
            object.__setattr__(self, "simulation_time_s", float(self.T_s))

        object.__setattr__(self, "T_s", float(self.simulation_time_s))
        object.__setattr__(self, "t_final", float(self.simulation_time_s))

        if self.dt is not None:
            if not (np.isfinite(self.dt) and self.dt > 0):
                msg = f"SimOptions.dt must be positive and finite; got {self.dt!r}"
                raise ValueError(msg)
            object.__setattr__(self, "time_step_s", float(self.dt))

        if self.output_rate_hz is not None:
            if not (np.isfinite(self.output_rate_hz) and self.output_rate_hz > 0):
                msg = f"SimOptions.output_rate_hz must be positive and finite; got {self.output_rate_hz!r}"
                raise ValueError(msg)
            object.__setattr__(self, "sample_rate_hz", float(self.output_rate_hz))

        object.__setattr__(self, "output_rate_hz", float(self.sample_rate_hz))
        object.__setattr__(self, "dt", float(self.time_step_s))

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
        if self.basis not in ("power", "bernstein"):
            msg = f"SimOptions.basis must be 'power' or 'bernstein'; got {self.basis!r}"
            raise ValueError(msg)
        g_arr = tuple(float(x) for x in self.gravity)
        if len(g_arr) != 3 or not all(np.isfinite(g) for g in g_arr):
            msg = f"SimOptions.gravity must be a finite 3-vector; got {self.gravity!r}"
            raise ValueError(msg)
        object.__setattr__(self, "gravity", g_arr)


@dataclass(frozen=True, init=False)
class SimOut:
    """Canonical forward-sim output (cross-engine §2.2 & Simscape contract).

    All time series are sampled at ``options.time_step_s`` from ``t=0`` to
    ``t=options.simulation_time_s`` inclusive.
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
    solver_status: str = "success"
    duration_s: float = 0.0
    kinetic_energy: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    potential_energy: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    meta: dict[str, Any] = field(default_factory=dict)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Accept the canonical fields plus the legacy keyword aliases.

        The accepted signature is ``_simout_build.SIMOUT_SIGNATURE``; omitted
        pose / energy arrays are NaN-filled and every array is shape-checked.
        """
        for name, value in normalize_simout_args(args, kwargs).items():
            object.__setattr__(self, name, value)

    @property
    def t(self) -> NDArray[np.float64]:
        return self.time

    @property
    def grip_position(self) -> NDArray[np.float64]:
        return self.grip

    @property
    def grip_rotation(self) -> NDArray[np.float64]:
        return _quat_to_rotmat_series(self.grip_quat)

    @property
    def clubhead_position(self) -> NDArray[np.float64]:
        return self.clubhead

    @property
    def clubhead_rotation(self) -> NDArray[np.float64]:
        return _quat_to_rotmat_series(self.club_quat)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.meta


# ---------------------------------------------------------------------------
# Polynomial torque evaluator (pure-numpy; reused by both the LeafSystem
# below and the deterministic offline path)
# ---------------------------------------------------------------------------


def evaluate_torque_polynomial(
    theta: NDArray[np.float64],
    t: float,
    n_joints: int,
    basis: str = "power",
    T_s: float = 1.0,
    t0: float = 0.0,
) -> NDArray[np.float64]:
    """Evaluate the continuous torque polynomial at scalar ``t``."""
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
    if basis == "bernstein":
        return evaluate_bernstein_torque(coeffs, t, T_s=T_s, t0=t0)
    return _evaluate_polynomial_torque_matrix(coeffs, t - t0)


evaluate_polynomial_torque = evaluate_torque_polynomial


# ---------------------------------------------------------------------------
# Pydrake helpers (lazy-imported so the module loads without pydrake)
# ---------------------------------------------------------------------------


def _sample_grid(
    simulation_time_s: float, sample_rate_hz: float
) -> NDArray[np.float64]:
    """Build the canonical output time grid: ``0 <= t <= simulation_time_s``."""
    return build_output_grid(simulation_time_s, sample_rate_hz)


def _resolve_n_actuators(plant: MultibodyPlant) -> int:
    """Return the number of actuators / actuated DOFs in the plant."""
    n = int(plant.num_actuators())
    if n > 0:
        return n
    # Fallback to velocity dimension minus 6 (floating root) which matches the
    # actuated chain in the canonical URDF.
    nv = int(plant.num_velocities())
    return max(nv - 6, 0)


@dataclass(frozen=True)
class _PydrakeApi:
    """The pydrake entry points ``simulate_with_coefficients`` needs."""

    add_plant: Any
    simulator_cls: Any
    builder_cls: Any
    log_sink_cls: Any


def _import_pydrake() -> _PydrakeApi:
    """Lazily import pydrake (explicit ``from pydrake.X import Y`` per CLAUDE.md)."""
    try:
        from pydrake.multibody.plant import (  # noqa: PLC0415
            AddMultibodyPlantSceneGraph,
        )
        from pydrake.systems.analysis import Simulator  # noqa: PLC0415
        from pydrake.systems.framework import DiagramBuilder  # noqa: PLC0415
        from pydrake.systems.primitives import (  # noqa: PLC0415
            VectorLogSink,
        )
    except ImportError as exc:
        msg = (
            "Drake runtime (pydrake) is required for "
            "simulate_with_coefficients. Install via pydrake or Drake container."
        )
        raise ImportError(msg) from exc
    return _PydrakeApi(
        add_plant=AddMultibodyPlantSceneGraph,
        simulator_cls=Simulator,
        builder_cls=DiagramBuilder,
        log_sink_cls=VectorLogSink,
    )


def _build_plant(api: _PydrakeApi, opts: SimOptions) -> tuple[Any, Any, Path]:
    """Create the builder + finalized plant with the humanoid loaded."""
    builder = api.builder_cls()
    plant, _scene_graph = api.add_plant(builder, opts.time_step_s)
    if opts.gravity is not None and hasattr(plant, "mutable_gravity_field"):
        plant.mutable_gravity_field().set_gravity_vector(
            np.asarray(opts.gravity, dtype=np.float64)
        )
    urdf_path = (
        opts.urdf_path
        if opts.urdf_path is not None
        else (DEFAULT_GOLFER_URDF if DEFAULT_GOLFER_URDF.exists() else CANONICAL_URDF)
    )
    load_humanoid_into_plant(plant, urdf_path)
    plant.Finalize()
    return builder, plant, urdf_path


def _wire_actuation_and_logger(
    api: _PydrakeApi,
    builder: Any,
    plant: Any,
    theta: NDArray[np.float64],
    opts: SimOptions,
) -> None:
    """Add the polynomial-torque source, connect it, and add the state logger."""
    n_actuators = _resolve_n_actuators(plant)
    torque_source = _build_polynomial_torque_system(
        theta,
        n_actuators,
        basis=opts.basis,
        T_s=opts.simulation_time_s,
        t0=opts.t0,
    )
    builder.AddSystem(torque_source)

    actuation_port = (
        plant.get_actuation_input_port()
        if hasattr(plant, "get_actuation_input_port")
        else None
    )
    if actuation_port is not None and plant.num_actuators() == n_actuators:
        builder.Connect(torque_source.get_output_port(0), actuation_port)

    # State logger (for q, qd extraction)
    log_sink = api.log_sink_cls(plant.num_multibody_states())
    builder.AddSystem(log_sink)
    builder.Connect(plant.get_state_output_port(), log_sink.get_input_port(0))


def _rollout_metadata(
    plant: Any,
    n_actuators: int,
    urdf_path: Path,
    opts: SimOptions,
    error: BaseException | None,
) -> dict[str, Any]:
    """Diagnostics dictionary attached to ``SimOut.meta``."""
    metadata: dict[str, Any] = {
        "n_actuators": n_actuators,
        "num_positions": plant.num_positions(),
        "num_velocities": plant.num_velocities(),
        "urdf_path": str(urdf_path),
        "basis": opts.basis,
    }
    if error is not None:
        metadata["error"] = repr(error)
    return metadata


# ---------------------------------------------------------------------------
# Public entry point (cross-engine §2.2)
# ---------------------------------------------------------------------------


@precondition(
    lambda theta, *args, **kwargs: bool(np.asarray(theta).size % 7 == 0),
    "theta length must be a multiple of 7",
)
@precondition(
    lambda theta, *args, **kwargs: bool(np.all(np.isfinite(np.asarray(theta)))),
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
    lambda result: bool(
        result.solver_status in ("success", "warning", "failed", "partial")
    ),
    "invalid solver_status",
)
def simulate_with_coefficients(
    theta: NDArray[np.float64],
    options: SimOptions | None = None,
    initial_pose: dict[str, Any] | None = None,
) -> SimOut:
    """Drake forward simulation from a torque-polynomial coefficient vector."""
    # ---- 0. Argument normalization -------------------------------------
    theta = normalize_theta(theta)
    opts = options if options is not None else SimOptions()
    check_initial_pose_type(initial_pose)
    api = _import_pydrake()

    t_start = _time.perf_counter()

    # ---- 2. Build plant + load humanoid --------------------------------
    builder, plant, urdf_path = _build_plant(api, opts)

    # ---- 3. Add the polynomial-torque source ---------------------------
    n_actuators = _resolve_n_actuators(plant)
    check_theta_matches_actuators(theta, n_actuators)
    theta = validate_theta(theta, n_joints=n_actuators)
    _wire_actuation_and_logger(api, builder, plant, theta, opts)

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

    # ---- 4. Apply initial_pose overrides ------------------------------
    apply_initial_pose(plant, plant_context, initial_pose)

    # ---- 5. Run the sim ------------------------------------------------
    simulator = api.simulator_cls(diagram, diagram_context)
    simulator.set_publish_every_time_step(False)
    simulator.Initialize()

    grid = _sample_grid(opts.simulation_time_s, opts.sample_rate_hz)
    logs = allocate_logs(
        grid.shape[0], plant.num_positions(), plant.num_velocities(), n_actuators
    )
    rollout = _record_rollout(
        simulator=simulator,
        plant=plant,
        grid=grid,
        theta=theta,
        logs=logs,
        config=RolloutConfig(
            n_actuators=n_actuators,
            grip_body_name=opts.grip_body_name,
            clubhead_body_name=opts.clubhead_body_name,
            basis=opts.basis,
            T_s=opts.simulation_time_s,
            t0=opts.t0,
            compute_energy=opts.compute_energy,
        ),
    )

    # ---- 6. Finite-difference qdd from v_log --------------------------
    qdd_log = finite_difference_qdd(logs.v, opts.sample_rate_hz, opts.compute_qdd)
    duration_s = _time.perf_counter() - t_start

    out = SimOut(
        time=grid,
        q=logs.q,
        qd=logs.v,
        qdd=qdd_log,
        tau=logs.tau,
        grip=logs.grip,
        grip_quat=logs.grip_quat,
        clubhead=logs.clubhead,
        club_quat=logs.club_quat,
        solver_status=rollout.solver_status,
        duration_s=duration_s,
        kinetic_energy=logs.kinetic_energy,
        potential_energy=logs.potential_energy,
        meta=_rollout_metadata(plant, n_actuators, urdf_path, opts, rollout.error),
    )
    assert_finite_on_success(rollout.solver_status, out)
    return out
