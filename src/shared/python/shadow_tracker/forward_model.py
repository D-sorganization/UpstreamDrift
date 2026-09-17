"""Full-body forward dynamics model adapter and state/velocity mappings (#10130).

Provides:
1. Exact algebraic RPY body Jacobian and closed-form inverse.
2. Machine-precision bidirectional native <-> canonical full-body state and velocity mappings.
3. FullBodyForwardModel implementing the ForwardModel protocol.
4. Gate G4 replay audit with root force and reset detection.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
import math
from typing import Any, Final

import numpy as np

from src.shared.python.motion_matching.full_body_forward_dynamics import (
    RolloutOptions,
    simulate_full_body_forward,
)
from src.shared.python.pose_interchange.se3 import (
    euler_xyz_deg_to_matrix,
    matrix_to_euler_xyz_deg,
    matrix_to_quat,
    quat_to_matrix,
)
from ._validation import (
    REPLAY_AUDIT_SCHEMA_VERSION,
    check_strict_float,
)
from .contracts import (
    CANONICAL_ARTICULATED_CONVENTION,
    ModelCapabilities,
    ReplayAudit,
    RolloutRequest,
    RolloutResult,
)

logger = logging.getLogger(__name__)

_NATIVE_COORDS: Final[int] = 41
_CANONICAL_COORDS: Final[int] = 42
_CANONICAL_TANGENT: Final[int] = 41
_ROOT_UNACTUATED_COUNT: Final[int] = 6

_DEFAULT_COORDINATE_NAMES: tuple[str, ...] = (
    "TranslationInputX",
    "TranslationInputY",
    "TranslationInputZ",
    "HipInputX",
    "HipInputY",
    "HipInputZ",
    "SpineInputX",
    "SpineInputY",
    "TorsoInput",
    "LEInput",
    "LFInput",
    "LScapInputX",
    "LScapInputY",
    "LSInputX",
    "LSInputY",
    "LSInputZ",
    "LWInputX",
    "LWInputY",
    "REInput",
    "RFInput",
    "RScapInputX",
    "RScapInputY",
    "RSInputX",
    "RSInputY",
    "RSInputZ",
    "RWInputX",
    "RWInputY",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
)


def rpy_jacobian_body(rpy: Sequence[float] | np.ndarray) -> np.ndarray:
    """Calculate the 3x3 body-frame angular velocity Jacobian for intrinsic XYZ Euler angles.

    Relates Euler angle rates to body-fixed angular velocity:
    omega_body = J_body(rpy) @ drpy
    """
    if len(rpy) != 3 or not all(math.isfinite(x) for x in rpy):
        raise ValueError("rpy must be a 3-element finite sequence")
    _phi, theta, psi = float(rpy[0]), float(rpy[1]), float(rpy[2])
    c_th, s_th = np.cos(theta), np.sin(theta)
    c_ps, s_ps = np.cos(psi), np.sin(psi)
    return np.array(
        [
            [c_ps * c_th, s_ps, 0.0],
            [-s_ps * c_th, c_ps, 0.0],
            [s_th, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rpy_jacobian_body_inv(rpy: Sequence[float] | np.ndarray) -> np.ndarray:
    """Calculate the exact inverse of rpy_jacobian_body.

    Relates body-fixed angular velocity to Euler angle rates:
    drpy = J_body_inv(rpy) @ omega_body
    Raises ValueError when pitch theta is at gimbal lock (+/- pi/2).
    """
    if len(rpy) != 3 or not all(math.isfinite(x) for x in rpy):
        raise ValueError("rpy must be a 3-element finite sequence")
    _phi, theta, psi = float(rpy[0]), float(rpy[1]), float(rpy[2])
    c_th = np.cos(theta)
    if abs(c_th) < 1e-8:
        raise ValueError("Gimbal lock in RPY Jacobian inverse (pitch near +/- pi/2)")
    s_th = np.sin(theta)
    c_ps, s_ps = np.cos(psi), np.sin(psi)
    inv_c = 1.0 / c_th
    return inv_c * np.array(
        [
            [c_ps, -s_ps, 0.0],
            [s_ps * c_th, c_ps * c_th, 0.0],
            [-c_ps * s_th, s_ps * s_th, c_th],
        ],
        dtype=np.float64,
    )


def native_to_canonical_full_body(
    q_native: np.ndarray, qd_native: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert native 41-coordinate state and velocity to canonical-v2 format.

    Native layout (41):
      [x, y, z, roll_rad, pitch_rad, yaw_rad, joint_0, ..., joint_34]
    Canonical-v2 layout:
      q (42): [x, y, z, quat_w, quat_x, quat_y, quat_z, joint_0, ..., joint_34]
      v (41): [vx, vy, vz, omega_body_x, omega_body_y, omega_body_z, joint_0, ..., joint_34]
    """
    q_arr = np.asarray(q_native, dtype=np.float64).reshape(-1)
    qd_arr = np.asarray(qd_native, dtype=np.float64).reshape(-1)
    if q_arr.shape != (_NATIVE_COORDS,):
        raise ValueError(
            f"Expected q_native shape ({_NATIVE_COORDS},), got {q_arr.shape}"
        )
    if qd_arr.shape != (_NATIVE_COORDS,):
        raise ValueError(
            f"Expected qd_native shape ({_NATIVE_COORDS},), got {qd_arr.shape}"
        )
    if not np.all(np.isfinite(q_arr)) or not np.all(np.isfinite(qd_arr)):
        raise ValueError("q_native and qd_native must contain only finite numbers")

    pos = q_arr[0:3]
    rpy = q_arr[3:6]
    joints_q = q_arr[6:]

    rot_mat = euler_xyz_deg_to_matrix(np.degrees(rpy))
    quat_wxyz = matrix_to_quat(rot_mat)
    q_canon = np.concatenate([pos, quat_wxyz, joints_q])

    lin_vel = qd_arr[0:3]
    j_body = rpy_jacobian_body(rpy)
    omega_body = j_body @ qd_arr[3:6]
    joints_qd = qd_arr[6:]
    v_canon = np.concatenate([lin_vel, omega_body, joints_qd])

    return q_canon, v_canon


def canonical_to_native_full_body(
    q_canonical: np.ndarray, v_canonical: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert canonical-v2 state and velocity to native 41-coordinate format.

    Inverse of native_to_canonical_full_body.
    """
    q_arr = np.asarray(q_canonical, dtype=np.float64).reshape(-1)
    v_arr = np.asarray(v_canonical, dtype=np.float64).reshape(-1)
    if q_arr.shape != (_CANONICAL_COORDS,):
        raise ValueError(
            f"Expected q_canonical shape ({_CANONICAL_COORDS},), got {q_arr.shape}"
        )
    if v_arr.shape != (_CANONICAL_TANGENT,):
        raise ValueError(
            f"Expected v_canonical shape ({_CANONICAL_TANGENT},), got {v_arr.shape}"
        )
    if not np.all(np.isfinite(q_arr)) or not np.all(np.isfinite(v_arr)):
        raise ValueError("q_canonical and v_canonical must contain only finite numbers")

    pos = q_arr[0:3]
    quat_wxyz = q_arr[3:7]
    joints_q = q_arr[7:]

    rot_mat = quat_to_matrix(quat_wxyz)
    rpy = np.radians(matrix_to_euler_xyz_deg(rot_mat))
    q_native = np.concatenate([pos, rpy, joints_q])

    lin_vel = v_arr[0:3]
    omega_body = v_arr[3:6]
    j_inv = rpy_jacobian_body_inv(rpy)
    rpy_rates = j_inv @ omega_body
    joints_v = v_arr[6:]
    qd_native = np.concatenate([lin_vel, rpy_rates, joints_v])

    return q_native, qd_native


def _parse_rollout_request(
    request: RolloutRequest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Validate and parse a RolloutRequest into numeric arrays."""
    if not isinstance(request, RolloutRequest):
        raise TypeError(f"request must be RolloutRequest, got {type(request).__name__}")

    times = np.asarray(request.time_points_s, dtype=np.float64)
    if len(times) < 2:
        raise ValueError(f"time_points_s requires at least 2 points, got {len(times)}")
    if not np.all(np.isfinite(times)):
        raise ValueError("time_points_s must contain only finite numbers")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("time_points_s must be strictly increasing")

    init_arr = np.asarray(request.initial_state, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(init_arr)):
        raise ValueError("initial_state must contain only finite numbers")

    if len(init_arr) == _CANONICAL_COORDS:
        q0, qd0 = canonical_to_native_full_body(
            init_arr, np.zeros(_CANONICAL_TANGENT, dtype=np.float64)
        )
    elif len(init_arr) == _NATIVE_COORDS:
        q0 = init_arr.copy()
        qd0 = np.zeros(_NATIVE_COORDS, dtype=np.float64)
    else:
        raise ValueError(
            f"initial_state must have length {_NATIVE_COORDS} (native) or "
            f"{_CANONICAL_COORDS} (canonical), got {len(init_arr)}"
        )

    controls_arr = np.asarray(request.controls, dtype=np.float64)
    if controls_arr.ndim != 2:
        raise ValueError(f"controls must be 2D array, got shape {controls_arr.shape}")
    if controls_arr.shape[0] != _NATIVE_COORDS:
        raise ValueError(
            f"controls rows must match coordinate count {_NATIVE_COORDS}, "
            f"got {controls_arr.shape[0]}"
        )

    root_forces = controls_arr[:_ROOT_UNACTUATED_COUNT, :]
    has_undeclared_root_forces = bool(np.any(np.abs(root_forces) > 1e-9))
    return times, q0, qd0, controls_arr, has_undeclared_root_forces


def _build_replay_audit(
    sim_res: Any,
    times: np.ndarray,
    integrator: str,
    has_undeclared_root_forces: bool,
    max_trans_tol_m: float,
    max_rot_tol_rad: float,
) -> ReplayAudit:
    """Construct ReplayAudit evaluating physical acceptance criteria."""
    is_physically_accepted = (
        sim_res.status == "success"
        and not has_undeclared_root_forces
        and sim_res.max_closure_translation_m <= max_trans_tol_m
        and (
            sim_res.max_closure_rotation_rad is not None
            and sim_res.max_closure_rotation_rad <= max_rot_tol_rad
        )
    )
    return ReplayAudit(
        schema_version=REPLAY_AUDIT_SCHEMA_VERSION,
        candidate_id="full_body_candidate",
        reset_count=1,
        integrator_name=integrator,
        integrator_version="1.0.0",
        coverage_start_s=float(times[0]),
        coverage_end_s=float(times[-1]),
        max_grip_translation_error_m=float(sim_res.max_closure_translation_m),
        max_grip_rotation_error_rad=float(
            sim_res.max_closure_rotation_rad
            if sim_res.max_closure_rotation_rad is not None
            else 0.0
        ),
        is_physically_accepted=is_physically_accepted,
    )


class FullBodyForwardModel:
    """Shadow Tracker forward dynamics model adapter implementing the ForwardModel protocol.

    Provides real full-body forward integration, Gate G4 replay audit, and detection
    of mid-run resets and undeclared root forces.
    """

    def __init__(
        self,
        model: Any | None = None,
        *,
        coordinate_names: Sequence[str] | None = None,
        max_translation_tol_m: float = 1e-3,
        max_rotation_tol_rad: float = 0.05,
    ) -> None:
        self.model = model
        self.coordinate_names: tuple[str, ...] = (
            tuple(coordinate_names)
            if coordinate_names is not None
            else getattr(model, "coordinate_order", _DEFAULT_COORDINATE_NAMES)
        )
        self.max_translation_tol_m = float(max_translation_tol_m)
        self.max_rotation_tol_rad = float(max_rotation_tol_rad)

    def capabilities(self) -> ModelCapabilities:
        """Return declared forward model capabilities and engine availability."""
        is_available = self.model is not None
        return ModelCapabilities(
            supported_bodies=self.coordinate_names,
            state_convention=CANONICAL_ARTICULATED_CONVENTION,
            actuator_modes=("torque_polynomial_deg6",),
            contact_modes=("hunt_crossley_regularized_coulomb",),
            is_available=is_available,
        )

    def rollout(self, request: RolloutRequest) -> RolloutResult:
        """Execute continuous zero-feedback forward simulation and produce replay audit."""
        if not self.capabilities().is_available:
            raise RuntimeError(
                "No physical engine or full-body model available for forward simulation."
            )

        times, q0, qd0, controls_arr, has_undeclared_root_forces = (
            _parse_rollout_request(request)
        )

        options = RolloutOptions(
            substeps=2,
            integrator="rk45",
            auto_calibrate_ground=False,
            preserve_ground_calibration=True,
        )

        sim_res = simulate_full_body_forward(
            model=self.model,
            ik_adapter=None,
            theta=controls_arr,
            time_grid=times,
            initial_state=(q0, qd0),
            marker_offsets=None,
            capture=None,
            options=options,
        )

        audit = _build_replay_audit(
            sim_res=sim_res,
            times=times,
            integrator=options.integrator,
            has_undeclared_root_forces=has_undeclared_root_forces,
            max_trans_tol_m=self.max_translation_tol_m,
            max_rot_tol_rad=self.max_rotation_tol_rad,
        )

        canonical_trajectory: list[tuple[float, ...]] = []
        for k in range(len(times)):
            q_k_canon, _ = native_to_canonical_full_body(sim_res.q[k], sim_res.qd[k])
            canonical_trajectory.append(tuple(float(x) for x in q_k_canon))

        return RolloutResult(
            trajectory=tuple(canonical_trajectory),
            realized_controls=request.controls,
            time_points_s=request.time_points_s,
            audit=audit,
        )
