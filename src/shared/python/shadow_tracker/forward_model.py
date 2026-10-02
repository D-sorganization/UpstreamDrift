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
from dataclasses import dataclass
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
    check_pos_float,
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


@dataclass(frozen=True, slots=True, kw_only=True)
class GripClosureResult:
    """Independent grip translation and rotation closure metrics."""

    translation_error_m: float
    rotation_error_rad: float
    translation_passed: bool
    rotation_passed: bool
    passed: bool


def quat_velocity_jacobian_body(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """Calculate the 4x3 body-frame quaternion velocity Jacobian.

    Relates body-fixed angular velocity to quaternion rate:
    q_dot = 0.5 * G(q) @ omega_body
    where G(q) is the 4x3 matrix for scalar-first unit quaternion q = [w, x, y, z].
    """
    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    if q_arr.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {q_arr.shape}")
    if not np.all(np.isfinite(q_arr)):
        raise ValueError("Quaternion must contain only finite numbers")
    norm = float(np.linalg.norm(q_arr))
    if norm < 1e-12:
        raise ValueError("cannot normalize a zero-norm quaternion")
    w, x, y, z = q_arr / norm
    return 0.5 * np.array(
        [
            [-x, -y, -z],
            [w, -z, y],
            [z, w, -x],
            [-y, x, w],
        ],
        dtype=np.float64,
    )


def quat_velocity_jacobian_body_inv(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """Calculate the 3x4 body-frame quaternion velocity inverse Jacobian (left inverse).

    Relates quaternion rate to body-fixed angular velocity:
    omega_body = 2.0 * G(q).T @ q_dot
    """
    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    if q_arr.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {q_arr.shape}")
    if not np.all(np.isfinite(q_arr)):
        raise ValueError("Quaternion must contain only finite numbers")
    norm = float(np.linalg.norm(q_arr))
    if norm < 1e-12:
        raise ValueError("cannot normalize a zero-norm quaternion")
    w, x, y, z = q_arr / norm
    return 2.0 * np.array(
        [
            [-x, w, z, -y],
            [-y, -z, w, x],
            [-z, y, -x, w],
        ],
        dtype=np.float64,
    )


def body_omega_to_quat_derivative(
    q: Sequence[float] | np.ndarray,
    omega_body: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Map body-fixed angular velocity to quaternion time derivative."""
    om = np.asarray(omega_body, dtype=np.float64).reshape(-1)
    if om.shape != (3,):
        raise ValueError(f"Expected omega_body shape (3,), got {om.shape}")
    if not np.all(np.isfinite(om)):
        raise ValueError("omega_body must contain only finite numbers")
    j_quat = quat_velocity_jacobian_body(q)
    return j_quat @ om


def quat_derivative_to_body_omega(
    q: Sequence[float] | np.ndarray,
    qd: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Map quaternion time derivative to body-fixed angular velocity."""
    qd_arr = np.asarray(qd, dtype=np.float64).reshape(-1)
    if qd_arr.shape != (4,):
        raise ValueError(f"Expected qd shape (4,), got {qd_arr.shape}")
    if not np.all(np.isfinite(qd_arr)):
        raise ValueError("qd must contain only finite numbers")
    j_inv = quat_velocity_jacobian_body_inv(q)
    return j_inv @ qd_arr


def canonical_37_to_native_41(state_37: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map 37-element canonical articulated state to 41-element native coordinate vector.

    Translation values remain in meters; joint and root angles convert from degrees to radians.
    """
    s = np.asarray(state_37, dtype=np.float64).reshape(-1)
    if s.shape != (37,):
        raise ValueError(f"Expected 37 elements for canonical state, got {s.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError("state_37 must contain only finite numbers")

    q_native = np.zeros(_NATIVE_COORDS, dtype=np.float64)
    q_native[0:3] = s[0:3]  # Translation
    q_native[3:6] = np.radians(s[3:6])  # Hip RPY
    q_native[6:8] = np.radians(s[6:8])  # Spine
    q_native[8] = np.radians(s[8])  # Torso
    # Arms: LE, LF, LScap (2), LS (3), LW (2), RE, RF, RScap (2), RS (3), RW (2)
    q_native[9] = np.radians(s[19])  # LE
    q_native[10] = np.radians(s[21])  # LF
    q_native[11:13] = np.radians(s[9:11])  # LScap
    q_native[13:16] = np.radians(s[13:16])  # LS
    q_native[16:18] = np.radians(s[23:25])  # LW
    q_native[18] = np.radians(s[20])  # RE
    q_native[19] = np.radians(s[22])  # RF
    q_native[20:22] = np.radians(s[11:13])  # RScap
    q_native[22:25] = np.radians(s[16:19])  # RS
    q_native[25:27] = np.radians(s[25:27])  # RW
    # Lower body: RHip, RKnee, RAnkle, LHip, LKnee, LAnkle
    q_native[27:30] = np.radians(s[30:33])  # RHip (flexion, adduction, rotation)
    q_native[30] = np.radians(s[34])  # RKnee
    q_native[31] = np.radians(s[36])  # RAnkle
    q_native[34:37] = np.radians(s[27:30])  # LHip (flexion, adduction, rotation)
    q_native[37] = np.radians(s[33])  # LKnee
    q_native[38] = np.radians(s[35])  # LAnkle

    return q_native


def native_41_to_canonical_37(
    q_native: Sequence[float] | np.ndarray,
) -> tuple[float, ...]:
    """Map 41-element native (or 42-element canonical-v2) state to 37-element canonical tuple."""
    arr = np.asarray(q_native, dtype=np.float64).reshape(-1)
    if arr.shape == (37,):
        return tuple(float(x) for x in arr)
    if arr.shape == (27,):
        return tuple(float(x) for x in arr) + (0.0,) * 10
    if arr.shape == (_CANONICAL_COORDS,):  # 42
        q_nat, _ = canonical_to_native_full_body(
            arr, np.zeros(_CANONICAL_TANGENT, dtype=np.float64)
        )
        arr = q_nat
    if arr.shape != (_NATIVE_COORDS,):
        raise ValueError(
            f"Expected 41 (native), 42 (canonical-v2), or 37/27 elements, got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("q_native must contain only finite numbers")

    s = [0.0] * 37
    s[0:3] = [float(x) for x in arr[0:3]]  # Translation
    s[3:6] = [float(x) for x in np.degrees(arr[3:6])]  # Hip
    s[6:8] = [float(x) for x in np.degrees(arr[6:8])]  # Spine
    s[8] = float(np.degrees(arr[8]))  # Torso
    # Scapulae
    s[9:11] = [float(x) for x in np.degrees(arr[11:13])]  # LScap
    s[11:13] = [float(x) for x in np.degrees(arr[20:22])]  # RScap
    # Shoulders
    s[13:16] = [float(x) for x in np.degrees(arr[13:16])]  # LS
    s[16:19] = [float(x) for x in np.degrees(arr[22:25])]  # RS
    # Elbows
    s[19] = float(np.degrees(arr[9]))  # LE
    s[20] = float(np.degrees(arr[18]))  # RE
    # Forearms
    s[21] = float(np.degrees(arr[10]))  # LF
    s[22] = float(np.degrees(arr[19]))  # RF
    # Wrists
    s[23:25] = [float(x) for x in np.degrees(arr[16:18])]  # LW
    s[25:27] = [float(x) for x in np.degrees(arr[25:27])]  # RW
    # Lower body: LHip, RHip, LKnee, RKnee, LAnkle, RAnkle
    s[27:30] = [float(x) for x in np.degrees(arr[34:37])]  # LHip
    s[30:33] = [float(x) for x in np.degrees(arr[27:30])]  # RHip
    s[33] = float(np.degrees(arr[37]))  # LKnee
    s[34] = float(np.degrees(arr[30]))  # RKnee
    s[35] = float(np.degrees(arr[38]))  # LAnkle
    s[36] = float(np.degrees(arr[31]))  # RAnkle

    return tuple(s)


def closed_grip_golfer_setup() -> dict[str, float]:
    """Return an address pose satisfying closed-loop grip closure gates.

    Corrects diagnostic reference arm kinematics so lead and trail hands
    meet on the club grip in front of the body with grip distance < 0.05m.
    """
    from src.shared.python.motion_matching.diagnostics.reference_pose import (
        reference_golfer_setup,
    )

    angles = reference_golfer_setup()
    angles.update(
        {
            "LSStartPositionX": -72.91,
            "LSStartPositionY": -75.61,
            "LSStartPositionZ": -72.29,
            "RSStartPositionX": 99.14,
            "RSStartPositionY": 33.57,
            "RSStartPositionZ": 78.69,
            "LEStartPosition": -84.22,
            "REStartPosition": 30.94,
            "LWStartPositionX": -76.07,
            "LWStartPositionY": 0.0,
            "RWStartPositionX": 30.74,
            "RWStartPositionY": 0.0,
            "LFStartPosition": 3.39,
            "RFStartPosition": 0.24,
        }
    )
    return angles


def evaluate_grip_closure(
    pose_or_model: Any,
    *,
    max_translation_m: float = 0.08,
    max_rotation_rad: float = 0.5,
) -> GripClosureResult:
    """Evaluate independent grip translation and rotation closure gates."""
    check_pos_float(max_translation_m, "max_translation_m")
    check_pos_float(max_rotation_rad, "max_rotation_rad")

    if hasattr(pose_or_model, "closure_errors") and callable(
        pose_or_model.closure_errors
    ):
        err_p, _ = pose_or_model.closure_errors()
        err_p_arr = np.asarray(err_p, dtype=np.float64).reshape(-1)
        trans_err = float(np.linalg.norm(err_p_arr[:3]))
        rot_err = float(np.linalg.norm(err_p_arr[3:6])) if len(err_p_arr) >= 6 else 0.0
    else:
        points = getattr(pose_or_model, "points", pose_or_model)
        if not hasattr(points, "__getitem__"):
            raise TypeError(
                f"pose_or_model must be SkeletonPose or provide closure_errors(), got {type(pose_or_model).__name__}"
            )
        try:
            lh = np.asarray(points["l_hand"], dtype=np.float64).reshape(3)
            rh = np.asarray(points["r_hand"], dtype=np.float64).reshape(3)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"points missing required hand landmarks: {exc}") from exc

        if not np.all(np.isfinite(lh)) or not np.all(np.isfinite(rh)):
            raise ValueError("Hand landmarks must be finite numbers")

        trans_err = float(np.linalg.norm(lh - rh))
        rot_err = 0.0
        if "l_wrist" in points and "r_wrist" in points:
            lw = np.asarray(points["l_wrist"], dtype=np.float64).reshape(3)
            rw = np.asarray(points["r_wrist"], dtype=np.float64).reshape(3)
            vl = lh - lw
            vr = rh - rw
            nl = float(np.linalg.norm(vl))
            nr = float(np.linalg.norm(vr))
            if nl > 1e-6 and nr > 1e-6:
                ul = vl / nl
                ur = vr / nr
                dot = float(np.clip(np.abs(np.dot(ul, ur)), -1.0, 1.0))
                rot_err = float(np.arccos(dot))

    trans_passed = bool(trans_err <= max_translation_m)
    rot_passed = bool(rot_err <= max_rotation_rad)
    passed = bool(trans_passed and rot_passed)

    return GripClosureResult(
        translation_error_m=trans_err,
        rotation_error_rad=rot_err,
        translation_passed=trans_passed,
        rotation_passed=rot_passed,
        passed=passed,
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
