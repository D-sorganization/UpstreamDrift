import numpy as np
import numpy.typing as npt

from src.shared.python.contracts import require

from ..spatial_vectors import skew

from typing import TypeAlias

Vec3: TypeAlias = npt.NDArray[np.float64]
Mat3: TypeAlias = npt.NDArray[np.float64]
Quat: TypeAlias = npt.NDArray[np.float64]


def euler_to_rotation_matrix(
    euler: Vec3 | list[float] | tuple[float, float, float],
) -> Mat3:
    euler = np.asarray(euler, dtype=np.float64)
    require(
        euler.shape == (3,),
        "euler must be a length-3 array [roll, pitch, yaw]",
        euler.shape,
    )
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    return R


def rotation_matrix_to_euler(R: Mat3) -> Vec3:
    R = np.asarray(R, dtype=np.float64)
    require(R.shape == (3, 3), "R must be a (3, 3) rotation matrix", R.shape)

    if np.abs(R[2, 0]) >= 1.0 - 1e-10:
        yaw = 0.0
        if R[2, 0] < 0:
            pitch = np.pi / 2
            roll = np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = -np.arcsin(R[2, 0])
        cp = np.cos(pitch)
        roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
        yaw = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


def euler_to_quaternion(
    euler: Vec3 | list[float] | tuple[float, float, float],
) -> Quat:
    euler = np.asarray(euler, dtype=np.float64)
    roll, pitch, yaw = euler[0] / 2, euler[1] / 2, euler[2] / 2

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z], dtype=np.float64)


def quaternion_to_euler(quat: Quat | list[float]) -> Vec3:
    quat = np.asarray(quat, dtype=np.float64)
    require(
        quat.shape == (4,), "quat must be a length-4 array [w, x, y, z]", quat.shape
    )
    require(
        float(np.linalg.norm(quat)) > 1e-10,
        "quat must not be a zero vector",
        float(np.linalg.norm(quat)),
    )
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2, sinp) if np.abs(sinp) >= 1 else np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


def quaternion_to_rotation_matrix(quat: Quat | list[float]) -> Mat3:
    quat = np.asarray(quat, dtype=np.float64)
    require(
        quat.shape == (4,), "quat must be a length-4 array [w, x, y, z]", quat.shape
    )
    require(
        float(np.linalg.norm(quat)) > 1e-10,
        "quat must not be a zero vector",
        float(np.linalg.norm(quat)),
    )
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def rotation_matrix_to_quaternion(R: Mat3) -> Quat:
    R = np.asarray(R, dtype=np.float64)
    require(R.shape == (3, 3), "R must be a (3, 3) rotation matrix", R.shape)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z], dtype=np.float64)


def axis_angle_to_rotation_matrix(axis: Vec3 | list[float], angle: float) -> Mat3:
    axis = np.asarray(axis, dtype=np.float64)
    require(axis.shape == (3,), "axis must be a (3,) vector", axis.shape)
    require(
        float(np.linalg.norm(axis)) > 1e-10,
        "axis must not be a zero vector",
        float(np.linalg.norm(axis)),
    )
    axis = axis / np.linalg.norm(axis)

    K = skew(axis)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def quaternion_multiply(q1: Quat | list[float], q2: Quat | list[float]) -> Quat:
    if q1 is None:
        raise ValueError("q1 must be provided")
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quaternion_inverse(q: Quat | list[float]) -> Quat:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64) / np.dot(q, q)


def slerp(q1: Quat, q2: Quat, t: float) -> Quat:
    if q1 is None:
        raise ValueError("q1 must be provided")
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    dot = np.dot(q1, q2)

    if dot < 0:
        q2 = -q2
        dot = -dot

    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return s1 * q1 + s2 * q2
