"""Floating-base manifold spline on SE(3) with local-coordinate retraction (#11536).

Provides continuous trajectory representation on SE(3) (translation in R^3,
rotation in SO(3) via unit quaternion or rotation matrix), local-coordinates
perturbations in se(3), exact retraction and local-coordinate Jacobians,
and the outer Gauss-Newton solver on the manifold with whiteness diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.interpolate import BSpline

from src.shared.python.core.contracts import ensure, require
from src.shared.python.estimation.mosaic.kinematic_basis import BSplineBasis

FloatArray: TypeAlias = npt.NDArray[np.float64]

_TOL_SMALL_ANGLE = 1e-6
_TOL_TAYLOR = 1e-4
H_FD_JACOBIAN = 1e-6


# ---------------------------------------------------------------------------
# Elementary spatial vector / matrix helpers
# ---------------------------------------------------------------------------


def skew(v: FloatArray) -> FloatArray:
    """Return 3x3 skew-symmetric matrix [v]x for a 3-vector."""
    require(v.shape == (3,), "v must be a 3-vector", v.shape)
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def unskew(matrix: FloatArray) -> FloatArray:
    """Return 3-vector from a 3x3 skew-symmetric matrix."""
    require(matrix.shape == (3, 3), "matrix must be 3x3", matrix.shape)
    return np.array(
        [matrix[2, 1], matrix[0, 2], matrix[1, 0]],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Pose representation conversions: 4x4 matrix <-> 7-vector [p(3), q_wxyz(4)]
# ---------------------------------------------------------------------------


def pose7_to_matrix(pose7: FloatArray) -> FloatArray:
    """Convert 7-vector [p_x, p_y, p_z, qw, qx, qy, qz] to a 4x4 SE(3) matrix."""
    require(pose7.ndim == 1 and pose7.size == 7, "pose7 must have shape (7,)")
    p = np.asarray(pose7[:3], dtype=np.float64)
    q = np.asarray(pose7[3:], dtype=np.float64)
    q_norm = float(np.linalg.norm(q))
    require(q_norm > 1e-12, "quaternion cannot have zero norm")
    qw, qx, qy, qz = q / q_norm

    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qw * qz)
    r02 = 2.0 * (qx * qz + qw * qy)

    r10 = 2.0 * (qx * qy + qw * qz)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qw * qx)

    r20 = 2.0 * (qx * qz - qw * qy)
    r21 = 2.0 * (qy * qz + qw * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    T: FloatArray = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=np.float64
    )
    T[:3, 3] = p
    return T


def matrix_to_pose7(T: FloatArray) -> FloatArray:
    """Convert 4x4 SE(3) matrix to 7-vector [p_x, p_y, p_z, qw, qx, qy, qz]."""
    require(T.shape == (4, 4), "T must have shape (4, 4)", T.shape)
    p = np.asarray(T[:3, 3], dtype=np.float64)
    R = np.asarray(T[:3, :3], dtype=np.float64)

    trace = float(np.trace(R))
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    if q[0] < 0.0:
        q = -q
    q /= float(np.linalg.norm(q))
    return np.concatenate([p, q])


# ---------------------------------------------------------------------------
# SO(3) Lie Group operations
# ---------------------------------------------------------------------------


def so3_exp(phi: FloatArray) -> FloatArray:
    """Matrix exponential on SO(3): Rodrigues formula for rotation vector phi."""
    require(phi.shape == (3,), "phi must be a 3-vector", phi.shape)
    theta = float(np.linalg.norm(phi))
    phi_x = skew(phi)
    if theta < _TOL_SMALL_ANGLE:
        return np.eye(3, dtype=np.float64) + phi_x + 0.5 * (phi_x @ phi_x)
    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    return np.eye(3, dtype=np.float64) + a * phi_x + b * (phi_x @ phi_x)


def so3_log(R: FloatArray) -> FloatArray:
    """Matrix logarithm on SO(3): returns rotation vector phi with ||phi|| <= pi."""
    require(R.shape == (3, 3), "R must be a 3x3 matrix", R.shape)
    cos_theta = float(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < _TOL_SMALL_ANGLE:
        return 0.5 * unskew(R - R.T)
    if math.pi - theta < 1e-6:
        d = np.diag(R)
        idx = int(np.argmax(d))
        axis: FloatArray = np.zeros(3, dtype=np.float64)
        axis[idx] = math.sqrt(max(0.0, (R[idx, idx] + 1.0) / 2.0))
        for j in range(3):
            if j != idx:
                denom = 2.0 * axis[idx]
                axis[j] = R[idx, j] if denom == 0.0 else R[idx, j] / denom
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 1e-12:
            axis /= axis_norm
        return theta * axis
    sin_theta = math.sin(theta)
    return (0.5 * theta / sin_theta) * unskew(R - R.T)


def so3_left_jacobian(phi: FloatArray) -> FloatArray:
    """Left Jacobian of SO(3): J_l(phi) (3x3 matrix)."""
    require(phi.shape == (3,), "phi must be a 3-vector", phi.shape)
    theta = float(np.linalg.norm(phi))
    phi_x = skew(phi)
    if theta < _TOL_SMALL_ANGLE:
        return np.eye(3, dtype=np.float64) + 0.5 * phi_x + (1.0 / 6.0) * (phi_x @ phi_x)
    a = (1.0 - math.cos(theta)) / (theta * theta)
    b = (theta - math.sin(theta)) / (theta * theta * theta)
    return np.eye(3, dtype=np.float64) + a * phi_x + b * (phi_x @ phi_x)


def so3_left_jacobian_inverse(phi: FloatArray) -> FloatArray:
    """Inverse left Jacobian of SO(3): J_l^{-1}(phi) (3x3 matrix)."""
    require(phi.shape == (3,), "phi must be a 3-vector", phi.shape)
    theta = float(np.linalg.norm(phi))
    require(theta <= math.pi + 1e-9, "||phi|| must be <= pi", theta)
    phi_x = skew(phi)
    if theta < _TOL_TAYLOR:
        coeff = 1.0 / 12.0 + (theta * theta) / 720.0
    else:
        half = 0.5 * theta
        coeff = 1.0 / (theta * theta) - math.cos(half) / (2.0 * theta * math.sin(half))
    return np.eye(3, dtype=np.float64) - 0.5 * phi_x + coeff * (phi_x @ phi_x)


# ---------------------------------------------------------------------------
# SE(3) Lie Group operations: Exp, Log, Retract, Local Coordinates
# ---------------------------------------------------------------------------


def se3_exp(xi: FloatArray) -> FloatArray:
    """Exponential map on SE(3) mapping twist xi = [rho, phi] to 4x4 matrix."""
    require(xi.ndim == 1 and xi.size == 6, "xi must have shape (6,)", xi.shape)
    rho = np.asarray(xi[:3], dtype=np.float64)
    phi = np.asarray(xi[3:], dtype=np.float64)
    R = so3_exp(phi)
    V = so3_left_jacobian(phi)
    p = V @ rho
    T: FloatArray = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def se3_log(T: FloatArray) -> FloatArray:
    """Logarithm map on SE(3) mapping 4x4 matrix T to twist xi = [rho, phi]."""
    require(T.shape == (4, 4), "T must have shape (4, 4)", T.shape)
    R = np.asarray(T[:3, :3], dtype=np.float64)
    p = np.asarray(T[:3, 3], dtype=np.float64)
    phi = so3_log(R)
    V_inv = so3_left_jacobian_inverse(phi)
    rho = V_inv @ p
    return np.concatenate([rho, phi])


def se3_retract(T: FloatArray, delta_xi: FloatArray) -> FloatArray:
    """Retraction on SE(3): Exp(delta_xi) @ T.

    Supports T as 4x4 matrix or 7-vector [p, q_wxyz].
    """
    require(delta_xi.shape == (6,), "delta_xi must have shape (6,)", delta_xi.shape)
    T_mat = pose7_to_matrix(T) if T.size == 7 else np.asarray(T, dtype=np.float64)
    require(T_mat.shape == (4, 4), "T must be 4x4 or 7-vector")
    T_pert = se3_exp(delta_xi)
    T_new = T_pert @ T_mat
    return T_new


def se3_local_coordinates(T0: FloatArray, T1: FloatArray) -> FloatArray:
    """Local coordinates on SE(3): Log(T1 @ inv(T0))."""
    T0_mat = pose7_to_matrix(T0) if T0.size == 7 else np.asarray(T0, dtype=np.float64)
    T1_mat = pose7_to_matrix(T1) if T1.size == 7 else np.asarray(T1, dtype=np.float64)
    require(
        T0_mat.shape == (4, 4) and T1_mat.shape == (4, 4), "T must be 4x4 or 7-vector"
    )

    # If quaternions are provided directly as 7-vectors, enforce sign equivalence
    if T0.size == 7 and T1.size == 7:
        if float(np.dot(T0[3:], T1[3:])) < 0.0:
            T1_mat = pose7_to_matrix(np.concatenate([T1[:3], -T1[3:]]))

    T0_inv: FloatArray = np.eye(4, dtype=np.float64)
    R0_T = T0_mat[:3, :3].T
    T0_inv[:3, :3] = R0_T
    T0_inv[:3, 3] = -R0_T @ T0_mat[:3, 3]

    rel_T = T1_mat @ T0_inv
    return se3_log(rel_T)


# ---------------------------------------------------------------------------
# Jacobians
# ---------------------------------------------------------------------------


def se3_left_jacobian(xi: FloatArray) -> FloatArray:
    """Full 6x6 left Jacobian of SE(3)."""
    require(xi.shape == (6,), "xi must have shape (6,)")
    rho = np.asarray(xi[:3], dtype=np.float64)
    phi = np.asarray(xi[3:], dtype=np.float64)
    Jl_phi = so3_left_jacobian(phi)

    h = 1e-6
    Q: FloatArray = np.zeros((3, 3), dtype=np.float64)
    for k in range(3):
        step: FloatArray = np.zeros(3, dtype=np.float64)
        step[k] = h
        V_plus = so3_left_jacobian(phi + step)
        V_minus = so3_left_jacobian(phi - step)
        Q[:, k] = (V_plus @ rho - V_minus @ rho) / (2.0 * h)

    jac: FloatArray = np.zeros((6, 6), dtype=np.float64)
    jac[:3, :3] = Jl_phi
    jac[:3, 3:] = Q
    jac[3:, 3:] = Jl_phi
    return jac


def se3_left_jacobian_inverse(xi: FloatArray) -> FloatArray:
    """Full 6x6 inverse left Jacobian of SE(3)."""
    require(xi.shape == (6,), "xi must have shape (6,)")
    Jl = se3_left_jacobian(xi)
    return np.asarray(np.linalg.inv(Jl), dtype=np.float64)


def se3_retract_jacobian(T: FloatArray, delta_xi: FloatArray) -> FloatArray:
    """Exact Jacobian d(matrix_to_pose7(Exp(delta_xi) @ T)) / d(delta_xi) of shape (7, 6)."""
    require(delta_xi.shape == (6,), "delta_xi must have shape (6,)")
    T_mat = pose7_to_matrix(T) if T.size == 7 else np.asarray(T, dtype=np.float64)
    require(T_mat.shape == (4, 4), "T must be 4x4 or 7-vector")

    h = H_FD_JACOBIAN
    jac: FloatArray = np.zeros((7, 6), dtype=np.float64)
    for k in range(6):
        step: FloatArray = np.zeros(6, dtype=np.float64)
        step[k] = h
        p7_plus = matrix_to_pose7(se3_retract(T_mat, delta_xi + step))
        p7_minus = matrix_to_pose7(se3_retract(T_mat, delta_xi - step))

        # Enforce quaternion hemisphere continuity for central difference
        if float(np.dot(p7_plus[3:], p7_minus[3:])) < 0.0:
            p7_minus[3:] = -p7_minus[3:]

        jac[:, k] = (p7_plus - p7_minus) / (2.0 * h)
    return jac


def se3_local_coordinates_jacobian(T0: FloatArray, T1: FloatArray) -> FloatArray:
    """Exact Jacobian d(se3_local_coordinates(T0, Exp(d) @ T1)) / d(d) at d=0 of shape (6, 6)."""
    T0_mat = pose7_to_matrix(T0) if T0.size == 7 else np.asarray(T0, dtype=np.float64)
    T1_mat = pose7_to_matrix(T1) if T1.size == 7 else np.asarray(T1, dtype=np.float64)
    require(
        T0_mat.shape == (4, 4) and T1_mat.shape == (4, 4), "T must be 4x4 or 7-vector"
    )

    h = H_FD_JACOBIAN
    jac: FloatArray = np.zeros((6, 6), dtype=np.float64)
    for k in range(6):
        step: FloatArray = np.zeros(6, dtype=np.float64)
        step[k] = h
        T1_plus = se3_retract(T1_mat, step)
        T1_minus = se3_retract(T1_mat, -step)
        jac[:, k] = (
            se3_local_coordinates(T0_mat, T1_plus)
            - se3_local_coordinates(T0_mat, T1_minus)
        ) / (2.0 * h)
    return jac


# ---------------------------------------------------------------------------
# SE3ManifoldSpline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SE3ManifoldSpline:
    """Floating-base manifold spline on SE(3) with local-coordinate B-spline perturbation.

    Pose at sample node n is given by:
        T(t_n) = Exp((P C)_n) @ T_ref(t_n)
    where P is the (N, K) B-spline basis matrix and C is the (K, 6) local-coordinates
    perturbation matrix.
    """

    basis: BSplineBasis
    reference_poses: FloatArray
    coefficients: FloatArray = field(default_factory=lambda: np.zeros((0, 6)))
    screw_twist: FloatArray | None = None

    def __post_init__(self) -> None:
        require(
            self.reference_poses.shape == (self.basis.n_samples, 4, 4),
            "reference_poses must have shape (N, 4, 4)",
            self.reference_poses.shape,
        )
        if self.coefficients.size == 0:
            object.__setattr__(
                self,
                "coefficients",
                np.zeros((self.basis.n_coefficients, 6), dtype=np.float64),
            )
        require(
            self.coefficients.shape == (self.basis.n_coefficients, 6),
            "coefficients must have shape (n_coefficients, 6)",
            self.coefficients.shape,
        )
        if self.screw_twist is not None:
            require(self.screw_twist.shape == (6,), "screw_twist must have shape (6,)")

    @classmethod
    def from_poses(
        cls,
        times: FloatArray,
        poses: FloatArray,
        n_coefficients: int,
        degree: int = 5,
    ) -> SE3ManifoldSpline:
        """Construct a manifold spline from a sequence of sample poses."""
        basis = BSplineBasis.uniform(times, n_coefficients, degree=degree)
        require(
            poses.shape == (basis.n_samples, 4, 4),
            "poses must have shape (N, 4, 4)",
            poses.shape,
        )
        return cls(basis=basis, reference_poses=np.asarray(poses, dtype=np.float64))

    @classmethod
    def from_screw_motion(
        cls,
        times: FloatArray,
        T0: FloatArray,
        twist: FloatArray,
        n_coefficients: int,
        degree: int = 5,
    ) -> SE3ManifoldSpline:
        """Construct a manifold spline representing an exact screw motion T(t) = Exp(t * twist) @ T0."""
        require(twist.shape == (6,), "twist must have shape (6,)")
        T0_mat = (
            pose7_to_matrix(T0) if T0.size == 7 else np.asarray(T0, dtype=np.float64)
        )
        require(T0_mat.shape == (4, 4), "T0 must be 4x4 or 7-vector")

        basis = BSplineBasis.uniform(times, n_coefficients, degree=degree)
        linear_xi = times[:, None] * twist[None, :]
        coeffs = basis.fit_least_squares(linear_xi)
        T_ref = np.repeat(T0_mat[None, ...], times.size, axis=0)

        return cls(
            basis=basis,
            reference_poses=T_ref,
            coefficients=coeffs,
            screw_twist=np.asarray(twist, dtype=np.float64),
        )

    def evaluate_at(self, t: float) -> FloatArray:
        """Evaluate continuous pose T(t) (4x4 matrix) at scalar time t."""
        t_val = float(t)
        t_arr = self.basis.times
        require(t_arr[0] - 1e-9 <= t_val <= t_arr[-1] + 1e-9, "t out of spline bounds")
        t_clamped = float(np.clip(t_val, t_arr[0], t_arr[-1]))

        if self.screw_twist is not None:
            return se3_exp(t_clamped * self.screw_twist) @ self.reference_poses[0]

        identity = np.eye(self.basis.n_coefficients)
        spline = BSpline(
            self.basis.knots, identity, self.basis.degree, extrapolate=False
        )
        p_row = np.nan_to_num(spline(t_clamped))
        xi_t = p_row @ self.coefficients

        if np.allclose(self.reference_poses, self.reference_poses[0], atol=1e-12):
            return se3_exp(xi_t) @ self.reference_poses[0]

        idx = int(np.searchsorted(t_arr, t_clamped))
        if idx == 0:
            T_ref_t = self.reference_poses[0]
        elif idx >= t_arr.size:
            T_ref_t = self.reference_poses[-1]
        else:
            t0, t1 = t_arr[idx - 1], t_arr[idx]
            dt = t1 - t0
            alpha = (t_clamped - t0) / dt if dt > 0.0 else 0.0
            rel_xi = se3_local_coordinates(
                self.reference_poses[idx - 1], self.reference_poses[idx]
            )
            T_ref_t = se3_exp(alpha * rel_xi) @ self.reference_poses[idx - 1]

        return se3_exp(xi_t) @ T_ref_t

    def evaluate_velocity_at(self, t: float, h: float = 1e-5) -> FloatArray:
        """Evaluate continuous spatial velocity (6-vector) at scalar time t via finite difference."""
        t_val = float(t)
        t_arr = self.basis.times
        require(t_arr[0] - 1e-9 <= t_val <= t_arr[-1] + 1e-9, "t out of spline bounds")
        t_clamped = float(np.clip(t_val, t_arr[0], t_arr[-1]))

        t_min, t_max = t_arr[0], t_arr[-1]
        t_plus = min(t_max, t_clamped + h)
        t_minus = max(t_min, t_clamped - h)
        dt = t_plus - t_minus
        T_plus = self.evaluate_at(t_plus)
        T_minus = self.evaluate_at(t_minus)
        rel_T = T_plus @ np.linalg.inv(T_minus)
        return se3_log(rel_T) / dt

    def evaluate_poses(self) -> FloatArray:
        """Evaluate poses at all basis sample nodes (shape (N, 4, 4))."""
        xi = self.basis.position @ self.coefficients  # (N, 6)
        poses = np.zeros_like(self.reference_poses)
        for i in range(self.basis.n_samples):
            poses[i] = se3_exp(xi[i]) @ self.reference_poses[i]
        return poses

    def evaluate_velocities(self, h: float = 1e-5) -> FloatArray:
        """Evaluate spatial velocities at all basis sample nodes (shape (N, 6))."""
        times = self.basis.times
        vels = np.zeros((times.size, 6), dtype=np.float64)
        for i, t in enumerate(times):
            vels[i] = self.evaluate_velocity_at(t, h=h)
        return vels

    def evaluate_accelerations(self, h: float = 1e-5) -> FloatArray:
        """Evaluate spatial accelerations at all basis sample nodes (shape (N, 6))."""
        times = self.basis.times
        if self.screw_twist is not None:
            return np.zeros((times.size, 6), dtype=np.float64)

        accs = np.zeros((times.size, 6), dtype=np.float64)
        for i, t in enumerate(times):
            t_min, t_max = times[0], times[-1]
            t_plus = min(t_max, t + h)
            t_minus = max(t_min, t - h)
            dt = t_plus - t_minus
            v_plus = self.evaluate_velocity_at(t_plus, h=h)
            v_minus = self.evaluate_velocity_at(t_minus, h=h)
            accs[i] = (v_plus - v_minus) / dt
        return accs

    def evaluate(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return (poses, velocities, accelerations) at all sample nodes."""
        poses = self.evaluate_poses()
        vels = self.evaluate_velocities()
        accs = self.evaluate_accelerations()
        return poses, vels, accs

    def retract(self, delta_C: FloatArray) -> SE3ManifoldSpline:
        """Retract spline by local-coordinate perturbation delta_C (K, 6).

        Returns new SE3ManifoldSpline with updated reference poses and zeroed coefficients.
        """
        require(
            delta_C.shape == self.coefficients.shape,
            "delta_C must have shape (n_coefficients, 6)",
            delta_C.shape,
        )
        total_C = self.coefficients + delta_C
        xi = self.basis.position @ total_C
        new_poses = np.zeros_like(self.reference_poses)
        for i in range(self.basis.n_samples):
            new_poses[i] = se3_exp(xi[i]) @ self.reference_poses[i]

        return SE3ManifoldSpline(
            basis=self.basis,
            reference_poses=new_poses,
            coefficients=np.zeros_like(self.coefficients),
        )


# ---------------------------------------------------------------------------
# Whiteness Diagnostics for Gate 1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WhitenessDiagnostics:
    """Whiteness diagnostics on marker residual for qualification gate 1."""

    rms: float
    lag1_autocorrelation: float
    passed: bool
    details: dict[str, float] = field(default_factory=dict)


def compute_whiteness_diagnostics(
    residuals: FloatArray, sigma: float
) -> WhitenessDiagnostics:
    """Compute basis-resolution whiteness diagnostics on residuals."""
    require(sigma > 0.0, "sigma must be positive", sigma)
    require(residuals.ndim >= 1 and residuals.shape[0] >= 2, "need >= 2 time samples")

    r_whitened = np.asarray(residuals, dtype=np.float64) / sigma
    rms = float(np.sqrt(np.mean(r_whitened * r_whitened)))

    # Lag-1 temporal autocorrelation across all channels
    t_len = residuals.shape[0]
    flat_channels = r_whitened.reshape(t_len, -1)
    mean_val = float(np.mean(flat_channels))
    var_val = float(np.var(flat_channels))

    if var_val < 1e-15:
        lag1_corr = 0.0
    else:
        centered = flat_channels - mean_val
        autocov = float(np.mean(centered[:-1] * centered[1:]))
        lag1_corr = autocov / var_val

    # Pass criteria: whitened RMS near 1.0 (within [0.75, 1.25]) and low temporal correlation
    passed = bool(0.75 <= rms <= 1.25 and abs(lag1_corr) < 0.25)
    details = {
        "rms": rms,
        "lag1_autocorrelation": lag1_corr,
        "mean": mean_val,
        "variance": var_val,
    }
    return WhitenessDiagnostics(
        rms=rms, lag1_autocorrelation=lag1_corr, passed=passed, details=details
    )


# ---------------------------------------------------------------------------
# Outer Solve on the Manifold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifoldFitResult:
    """Result of outer Gauss-Newton solve on the manifold."""

    spline: SE3ManifoldSpline
    iterations: int
    cost_history: tuple[float, ...]
    converged: bool
    whiteness: WhitenessDiagnostics
    final_cost: float


def _compute_residuals(
    spline: SE3ManifoldSpline,
    marker_positions: FloatArray,
    marker_offsets: FloatArray,
    marker_sigma: float,
) -> tuple[FloatArray, FloatArray, float]:
    """Evaluate poses, residuals and cost for marker observations."""
    n_samples = spline.basis.n_samples
    poses = spline.evaluate_poses()
    res = np.zeros((n_samples, marker_offsets.shape[0], 3), dtype=np.float64)
    for i in range(n_samples):
        R_i = poses[i, :3, :3]
        p_i = poses[i, :3, 3]
        pred_i = marker_offsets @ R_i.T + p_i
        res[i] = pred_i - marker_positions[i]
    cost = 0.5 * float(np.sum((res / marker_sigma) ** 2))
    return res, poses, cost


def _build_normal_equations(
    poses: FloatArray,
    res: FloatArray,
    marker_offsets: FloatArray,
    P: FloatArray,
    marker_sigma: float,
    K: int,
) -> tuple[FloatArray, FloatArray]:
    """Build Gauss-Newton approximate Hessian H and gradient g."""
    n_samples = poses.shape[0]
    n_markers = marker_offsets.shape[0]
    H = np.zeros((K * 6, K * 6), dtype=np.float64)
    g = np.zeros(K * 6, dtype=np.float64)

    for i in range(n_samples):
        R_i = poses[i, :3, :3]
        p_i = poses[i, :3, 3]
        P_i = P[i]

        for m in range(n_markers):
            x_im = R_i @ marker_offsets[m] + p_i
            r_im = res[i, m] / marker_sigma

            dY_dxi: FloatArray = np.zeros((3, 6), dtype=np.float64)
            dY_dxi[:3, :3] = np.eye(3) / marker_sigma
            dY_dxi[:3, 3:] = -skew(x_im) / marker_sigma

            J_im: FloatArray = np.zeros((3, K * 6), dtype=np.float64)
            for j in range(K):
                p_ij = float(P_i[j])
                if abs(p_ij) > 1e-12:
                    J_im[:, 6 * j : 6 * j + 6] = p_ij * dY_dxi

            H += J_im.T @ J_im
            g += J_im.T @ r_im

    return H, g


def fit_floating_base_manifold(
    spline: SE3ManifoldSpline,
    marker_positions: FloatArray,
    marker_offsets: FloatArray,
    marker_sigma: float,
    max_iterations: int = 30,
    relative_cost_tolerance: float = 1e-8,
) -> ManifoldFitResult:
    """Outer Gauss-Newton solve on the manifold fitting SE3ManifoldSpline to marker observations.

    Parameters:
        spline: Initial floating-base manifold spline.
        marker_positions: Observed 3D marker coordinates of shape (N, M, 3).
        marker_offsets: Body-frame marker offsets of shape (M, 3).
        marker_sigma: Observation standard deviation (metres).
        max_iterations: Maximum Gauss-Newton iterations.
        relative_cost_tolerance: Stopping tolerance on relative cost decrease.
    """
    require(marker_sigma > 0.0, "marker_sigma must be positive")
    require(max_iterations >= 1, "max_iterations must be >= 1")
    n_samples = spline.basis.n_samples
    n_markers = marker_offsets.shape[0]
    require(
        marker_positions.shape == (n_samples, n_markers, 3),
        "marker_positions shape mismatch",
    )
    require(marker_offsets.shape == (n_markers, 3), "marker_offsets shape mismatch")

    current_spline = spline
    cost_history: list[float] = []
    damping = 1e-4

    P = spline.basis.position  # (N, K)
    K = spline.basis.n_coefficients

    res, poses, current_cost = _compute_residuals(
        current_spline, marker_positions, marker_offsets, marker_sigma
    )
    cost_history.append(current_cost)
    converged = False

    for _ in range(max_iterations):
        H, g = _build_normal_equations(poses, res, marker_offsets, P, marker_sigma, K)

        diag_H = np.diag(H)
        H_damped = H + damping * np.diag(np.maximum(diag_H, 1e-6))
        delta_vec = np.asarray(-np.linalg.solve(H_damped, g), dtype=np.float64)
        delta_C = np.asarray(delta_vec.reshape(K, 6), dtype=np.float64)

        candidate_spline = current_spline.retract(delta_C)
        cand_res, cand_poses, candidate_cost = _compute_residuals(
            candidate_spline, marker_positions, marker_offsets, marker_sigma
        )

        if candidate_cost < current_cost:
            rel_decrease = (current_cost - candidate_cost) / max(current_cost, 1e-12)
            current_spline = candidate_spline
            res, poses, current_cost = cand_res, cand_poses, candidate_cost
            cost_history.append(current_cost)
            damping = max(1e-7, damping * 0.3)

            if rel_decrease < relative_cost_tolerance:
                converged = True
                break
        else:
            damping = min(1e7, damping * 3.0)

    ensure(len(cost_history) >= 1, "must record at least initial cost")
    whiteness = compute_whiteness_diagnostics(res, marker_sigma)

    return ManifoldFitResult(
        spline=current_spline,
        iterations=len(cost_history) - 1,
        cost_history=tuple(cost_history),
        converged=converged,
        whiteness=whiteness,
        final_cost=current_cost,
    )
