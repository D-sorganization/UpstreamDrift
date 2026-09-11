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
from math import comb
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts.decorators import postcondition, precondition
from src.shared.python.math_utils.quaternion import rotmat_to_quat
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


def evaluate_bernstein_torque(
    coeffs: NDArray[np.float64],
    t: float,
    T_s: float = 1.0,
    t0: float = 0.0,
) -> NDArray[np.float64]:
    """Evaluate per-joint degree-6 torques in Bernstein polynomial basis.

    tau_j(t) = sum_{k=0}^6 c_{j,k} B_{k,6}((t - t0) / T_s)
    where B_{k,6}(s) = comb(6, k) * s^k * (1 - s)^(6 - k).

    Args:
        coeffs: 2D array of shape ``(n_joints, 7)`` containing control points.
        t: Time in seconds.
        T_s: Horizon duration in seconds.
        t0: Time offset in seconds.

    Returns:
        1D array of shape ``(n_joints,)`` with applied torques at time ``t``.
    """
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    if coeffs_arr.ndim != 2:
        raise ValueError(f"coeffs must be 2D (n_joints, 7); got ndim={coeffs_arr.ndim}")
    if coeffs_arr.shape[1] != COEFFS_PER_JOINT:
        raise ValueError(
            f"coeffs must have {COEFFS_PER_JOINT} columns; got shape {coeffs_arr.shape}"
        )
    if not np.isfinite(t):
        raise ValueError(f"t must be finite, got {t!r}")
    if T_s <= 0.0:
        raise ValueError(f"T_s must be positive, got {T_s!r}")

    s = (t - t0) / T_s
    k_vals = np.arange(COEFFS_PER_JOINT, dtype=np.float64)
    b_weights = np.array(
        [comb(POLY_DEGREE, k) for k in range(COEFFS_PER_JOINT)], dtype=np.float64
    )
    basis = b_weights * (s**k_vals) * ((1.0 - s) ** (POLY_DEGREE - k_vals))
    return coeffs_arr @ basis


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


@dataclass(frozen=True)
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

    def __init__(
        self,
        time: NDArray[np.float64] | None = None,
        q: NDArray[np.float64] | None = None,
        qd: NDArray[np.float64] | None = None,
        qdd: NDArray[np.float64] | None = None,
        tau: NDArray[np.float64] | None = None,
        grip: NDArray[np.float64] | None = None,
        grip_quat: NDArray[np.float64] | None = None,
        clubhead: NDArray[np.float64] | None = None,
        club_quat: NDArray[np.float64] | None = None,
        solver_status: str = "success",
        duration_s: float = 0.0,
        kinetic_energy: NDArray[np.float64] | None = None,
        potential_energy: NDArray[np.float64] | None = None,
        meta: dict[str, Any] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        t: NDArray[np.float64] | None = None,
        grip_position: NDArray[np.float64] | None = None,
        grip_rotation: NDArray[np.float64] | None = None,
        clubhead_position: NDArray[np.float64] | None = None,
        clubhead_rotation: NDArray[np.float64] | None = None,
    ) -> None:
        t_arr = time if time is not None else t
        if t_arr is None:
            raise ValueError("Either 'time' or 't' must be provided")
        t_vec = np.asarray(t_arr, dtype=np.float64)
        if t_vec.ndim != 1:
            msg = f"SimOut.time must be 1-D; got shape {t_vec.shape}"
            raise ValueError(msg)
        n = t_vec.shape[0]

        def _as_2d_buffer(val: Any, default_cols: int) -> NDArray[np.float64]:
            if val is not None:
                return np.asarray(val, dtype=np.float64)
            return np.zeros((n, default_cols), dtype=np.float64)

        q_mat = _as_2d_buffer(q, 0)
        qd_mat = _as_2d_buffer(qd, 0)
        nv = qd_mat.shape[1] if qd_mat.ndim == 2 else 0
        qdd_mat = _as_2d_buffer(qdd, nv)
        tau_mat = _as_2d_buffer(tau, nv)

        grip_mat = grip if grip is not None else grip_position
        if grip_mat is None:
            grip_mat = np.zeros((n, 3), dtype=np.float64)
        else:
            grip_mat = np.asarray(grip_mat, dtype=np.float64)

        if grip_quat is not None:
            g_quat = np.asarray(grip_quat, dtype=np.float64)
        elif grip_rotation is not None:
            g_quat = rotmat_to_quat(np.asarray(grip_rotation, dtype=np.float64))
        else:
            g_quat = np.zeros((n, 4), dtype=np.float64)

        head_mat = clubhead if clubhead is not None else clubhead_position
        if head_mat is None:
            head_mat = np.zeros((n, 3), dtype=np.float64)
        else:
            head_mat = np.asarray(head_mat, dtype=np.float64)

        if club_quat is not None:
            c_quat = np.asarray(club_quat, dtype=np.float64)
        elif clubhead_rotation is not None:
            c_quat = rotmat_to_quat(np.asarray(clubhead_rotation, dtype=np.float64))
        else:
            c_quat = np.zeros((n, 4), dtype=np.float64)

        for name, arr, cols in [
            ("q", q_mat, None),
            ("qd", qd_mat, None),
            ("qdd", qdd_mat, None),
            ("tau", tau_mat, None),
            ("grip", grip_mat, 3),
            ("grip_quat", g_quat, 4),
            ("clubhead", head_mat, 3),
            ("club_quat", c_quat, 4),
        ]:
            if arr.ndim != 2 or arr.shape[0] != n:
                msg = f"SimOut.{name} must have shape (N={n}, ...); got {arr.shape}"
                raise ValueError(msg)
            if cols is not None and arr.shape[1] != cols:
                msg = f"SimOut.{name} must have shape (N, {cols}); got {arr.shape}"
                raise ValueError(msg)

        if solver_status not in {"success", "warning", "failed"}:
            msg = (
                "SimOut.solver_status must be 'success' / 'warning' / 'failed'; "
                f"got {solver_status!r}"
            )
            raise ValueError(msg)

        def _as_1d_energy(val: Any) -> NDArray[np.float64]:
            if val is not None:
                return np.asarray(val, dtype=np.float64)
            return np.zeros(n, dtype=np.float64)

        ke_vec = _as_1d_energy(kinetic_energy)
        pe_vec = _as_1d_energy(potential_energy)
        if ke_vec.ndim != 1 or ke_vec.shape[0] != n:
            raise ValueError(
                f"SimOut.kinetic_energy must have shape ({n},); got {ke_vec.shape}"
            )
        if pe_vec.ndim != 1 or pe_vec.shape[0] != n:
            raise ValueError(
                f"SimOut.potential_energy must have shape ({n},); got {pe_vec.shape}"
            )

        meta_dict = (
            meta if meta is not None else (metadata if metadata is not None else {})
        )

        sim_fields = {
            "time": t_vec,
            "q": q_mat,
            "qd": qd_mat,
            "qdd": qdd_mat,
            "tau": tau_mat,
            "grip": grip_mat,
            "grip_quat": g_quat,
            "clubhead": head_mat,
            "club_quat": c_quat,
            "solver_status": solver_status,
            "duration_s": float(duration_s),
            "kinetic_energy": ke_vec,
            "potential_energy": pe_vec,
            "meta": meta_dict,
        }
        for field_name, field_val in sim_fields.items():
            object.__setattr__(self, field_name, field_val)

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


def _build_polynomial_torque_system(
    theta: NDArray[np.float64],
    n_actuators: int,
    basis: str = "power",
    T_s: float = 1.0,
    t0: float = 0.0,
) -> Any:
    """Build a Drake ``LeafSystem`` that emits the polynomial torque."""
    # Explicit import per CLAUDE.md.
    from pydrake.systems.framework import BasicVector, LeafSystem  # noqa: PLC0415

    coeffs = np.ascontiguousarray(theta, dtype=np.float64).reshape(
        n_actuators, COEFFS_PER_JOINT
    )

    class _PolynomialTorqueSource(LeafSystem):
        """Stateflow-equivalent per-joint continuous torque polynomial."""

        def __init__(self) -> None:
            LeafSystem.__init__(self)
            self._coeffs = coeffs
            self._n = n_actuators
            self._basis = basis
            self._T_s = T_s
            self._t0 = t0
            self.DeclareVectorOutputPort(
                "tau",
                BasicVector(n_actuators),
                self._calc_output,
            )

        def _calc_output(self, context: Any, output: Any) -> None:
            t = float(context.get_time())
            if self._basis == "bernstein":
                tau = evaluate_bernstein_torque(
                    self._coeffs, t, T_s=self._T_s, t0=self._t0
                )
            else:
                tau = _evaluate_polynomial_torque_matrix(self._coeffs, t - self._t0)
            output.SetFromVector(tau)

    return _PolynomialTorqueSource()


def _resolve_world_pose(
    plant: MultibodyPlant,
    plant_context: Any,
    body_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Forward-kinematics: return ``(position, quaternion[w,x,y,z])`` for ``body_name``."""
    target_name = body_name
    if not hasattr(plant, "HasBodyNamed") or not bool(plant.HasBodyNamed(target_name)):
        candidates: list[str] = []
        if target_name in ("club_grip", "grip"):
            candidates = ["mid_hands", "grip", "right_hand"]
        elif target_name in ("clubhead", "club_head"):
            candidates = ["club_head", "club_shaft"]
        elif target_name in ("mid_hands",):
            candidates = ["club_grip", "grip"]

        found = False
        for c in candidates:
            if hasattr(plant, "HasBodyNamed") and bool(plant.HasBodyNamed(c)):
                target_name = c
                found = True
                break
        if not found:
            return None

    body = plant.GetBodyByName(target_name)
    transform = plant.CalcRelativeTransform(
        plant_context,
        plant.world_frame(),
        body.body_frame(),
    )
    pos = np.asarray(transform.translation(), dtype=np.float64)
    quat = transform.rotation().ToQuaternion()
    # Drake quaternion exposes w, x, y, z scalars.
    quat_arr = np.array([quat.w(), quat.x(), quat.y(), quat.z()], dtype=np.float64)
    if pos.shape != (3,) or quat_arr.shape != (4,):
        return None
    return pos, quat_arr


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


@dataclass
class _RolloutResult:
    """Outcome of stepping the Drake simulator over the output time grid."""

    solver_status: str
    error: BaseException | None


@dataclass
class _RolloutLogs:
    """Pre-allocated, NaN-filled output buffers mutated in place per step."""

    q: NDArray[np.float64]
    v: NDArray[np.float64]
    tau: NDArray[np.float64]
    grip: NDArray[np.float64]
    grip_quat: NDArray[np.float64]
    clubhead: NDArray[np.float64]
    club_quat: NDArray[np.float64]
    kinetic_energy: NDArray[np.float64]
    potential_energy: NDArray[np.float64]


def _record_rollout(
    *,
    simulator: Any,
    plant: MultibodyPlant,
    grid: NDArray[np.float64],
    theta: NDArray[np.float64],
    n_actuators: int,
    grip_body_name: str,
    clubhead_body_name: str,
    logs: _RolloutLogs,
    basis: str = "power",
    T_s: float = 1.0,
    t0: float = 0.0,
    compute_energy: bool = True,
) -> _RolloutResult:
    """Step the simulator over ``grid`` and record state / torque / FK logs."""
    coeffs = (
        theta.reshape(n_actuators, COEFFS_PER_JOINT)
        if n_actuators > 0
        else np.zeros((0, COEFFS_PER_JOINT), dtype=np.float64)
    )
    for idx, t_target in enumerate(grid):
        if t_target > 0.0:
            try:
                simulator.AdvanceTo(float(t_target))
            except RuntimeError as exc:
                logger.exception(
                    "Drake AdvanceTo diverged at t=%.6f s (step %d/%d)",
                    float(t_target),
                    idx,
                    len(grid),
                )
                return _RolloutResult(solver_status="failed", error=exc)

        ctx = simulator.get_context()
        plant_ctx = plant.GetMyContextFromRoot(ctx)

        logs.q[idx, :] = plant.GetPositions(plant_ctx)
        logs.v[idx, :] = plant.GetVelocities(plant_ctx)
        if n_actuators > 0:
            if basis == "bernstein":
                logs.tau[idx, :] = evaluate_bernstein_torque(
                    coeffs, float(t_target), T_s=T_s, t0=t0
                )
            else:
                logs.tau[idx, :] = _evaluate_polynomial_torque_matrix(
                    coeffs, float(t_target) - t0
                )

        grip_pose = _resolve_world_pose(plant, plant_ctx, grip_body_name)
        if grip_pose is not None:
            logs.grip[idx, :], logs.grip_quat[idx, :] = grip_pose
        club_pose = _resolve_world_pose(plant, plant_ctx, clubhead_body_name)
        if club_pose is not None:
            logs.clubhead[idx, :], logs.club_quat[idx, :] = club_pose

        if compute_energy:
            if hasattr(plant, "CalcKineticEnergy"):
                try:
                    logs.kinetic_energy[idx] = float(plant.CalcKineticEnergy(plant_ctx))
                except (TypeError, ValueError):
                    logs.kinetic_energy[idx] = 0.0
            if hasattr(plant, "CalcPotentialEnergy"):
                try:
                    logs.potential_energy[idx] = float(
                        plant.CalcPotentialEnergy(plant_ctx)
                    )
                except (TypeError, ValueError):
                    logs.potential_energy[idx] = 0.0

    return _RolloutResult(solver_status="success", error=None)


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
    lambda result: bool(result.solver_status in ("success", "warning", "failed")),
    "invalid solver_status",
)
def simulate_with_coefficients(  # noqa: C901
    theta: NDArray[np.float64],
    options: SimOptions | None = None,
    initial_pose: dict[str, Any] | None = None,
) -> SimOut:
    """Drake forward simulation from a torque-polynomial coefficient vector."""
    # ---- 0. Argument normalization -------------------------------------
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
    if not is_drake_available():
        msg = (
            "Drake runtime (pydrake) is required for "
            "simulate_with_coefficients. Install via pydrake or Drake container."
        )
        raise ImportError(msg)

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

    # ---- 3. Add the polynomial-torque source ---------------------------
    n_actuators = _resolve_n_actuators(plant)
    expected_theta_len = n_actuators * COEFFS_PER_JOINT
    if expected_theta_len > 0 and theta.shape[0] != expected_theta_len:
        msg = (
            "theta is sized for "
            f"{theta.shape[0] // COEFFS_PER_JOINT} joint(s) "
            f"(length {theta.shape[0]}) but the plant has {n_actuators} "
            f"actuator(s), so it expects theta of length {expected_theta_len} "
            f"({n_actuators} * {COEFFS_PER_JOINT}). Supplying a mismatched "
            "theta would leave the actuation port disconnected while tau_log "
            "still recorded nonzero torques (issue #7725). Resize theta to the "
            "plant's actuator count."
        )
        raise ValueError(msg)

    theta = validate_theta(theta, n_joints=n_actuators)

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

    logs = _RolloutLogs(
        q=np.full((n_t, n_q), np.nan, dtype=np.float64),
        v=np.full((n_t, n_v), np.nan, dtype=np.float64),
        tau=np.full((n_t, n_actuators), np.nan, dtype=np.float64),
        grip=np.full((n_t, 3), np.nan, dtype=np.float64),
        grip_quat=np.full((n_t, 4), np.nan, dtype=np.float64),
        clubhead=np.full((n_t, 3), np.nan, dtype=np.float64),
        club_quat=np.full((n_t, 4), np.nan, dtype=np.float64),
        kinetic_energy=np.zeros(n_t, dtype=np.float64),
        potential_energy=np.zeros(n_t, dtype=np.float64),
    )

    rollout = _record_rollout(
        simulator=simulator,
        plant=plant,
        grid=grid,
        theta=theta,
        n_actuators=n_actuators,
        grip_body_name=opts.grip_body_name,
        clubhead_body_name=opts.clubhead_body_name,
        logs=logs,
        basis=opts.basis,
        T_s=opts.simulation_time_s,
        t0=opts.t0,
        compute_energy=opts.compute_energy,
    )
    solver_status = rollout.solver_status
    sim_error = rollout.error

    # ---- 6. Finite-difference qdd from v_log --------------------------
    qdd_log = np.zeros_like(logs.v)
    if opts.compute_qdd and n_t >= 2:
        dt = 1.0 / opts.sample_rate_hz
        qdd_log[1:-1, :] = (logs.v[2:, :] - logs.v[:-2, :]) / (2.0 * dt)
        qdd_log[0, :] = (logs.v[1, :] - logs.v[0, :]) / dt
        qdd_log[-1, :] = (logs.v[-1, :] - logs.v[-2, :]) / dt

    duration_s = _time.perf_counter() - t_start

    metadata: dict[str, Any] = {
        "n_actuators": n_actuators,
        "num_positions": n_q,
        "num_velocities": n_v,
        "urdf_path": str(urdf_path),
        "basis": opts.basis,
    }
    if sim_error is not None:
        metadata["error"] = repr(sim_error)

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
        solver_status=solver_status,
        duration_s=duration_s,
        kinetic_energy=logs.kinetic_energy,
        potential_energy=logs.potential_energy,
        meta=metadata,
    )

    # Postcondition (cross-engine §2.2): on success, signal arrays are finite.
    if solver_status == "success":
        for name, arr in (("q", out.q), ("qd", out.qd), ("tau", out.tau)):
            if not np.all(np.isfinite(arr)):
                msg = f"Postcondition: SimOut.{name} contains non-finite values"
                raise AssertionError(msg)
    return out
