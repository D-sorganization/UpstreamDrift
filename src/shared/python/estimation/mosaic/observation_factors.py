"""Observation factors: 2-D keypoint projection, robust kernels, marker attachment offsets (MOSAIC-06).

Implements:
- CameraModel: pinhole camera projection with intrinsics, extrinsics, and analytic Jacobians.
- Robust loss kernels: Huber, Cauchy, Geman-McClure, Tukey / bisquare with weights and derivatives.
- MarkerAttachmentFactor: parameterizes marker attachment offsets delta_p on planar chains with
  analytic Jacobians and synthetic offset recovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require
from src.shared.python.estimation.mosaic.planar_chain import (
    PlanarChain,
    PlanarMarkerSet,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Rodrigues Rotation Map and Helpers
# ---------------------------------------------------------------------------


def skew3(v: FloatArray) -> FloatArray:
    """Skew-symmetric 3x3 matrix [v]_x from 3-vector."""
    require(v.shape == (3,), "v must be a 3-vector", v.shape)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def rotation_from_rodrigues(rvec: FloatArray) -> FloatArray:
    """Convert Rodrigues rotation vector (axis-angle) to 3x3 rotation matrix."""
    r = np.asarray(rvec, dtype=np.float64)
    require(r.shape == (3,), "rvec must be a 3-vector", r.shape)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + skew3(r) + 0.5 * (skew3(r) @ skew3(r))
    k = r / theta
    k_skew = skew3(k)
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(theta) * k_skew
        + (1.0 - math.cos(theta)) * (k_skew @ k_skew)
    )


def rodrigues_from_rotation(R: FloatArray) -> FloatArray:
    """Convert 3x3 rotation matrix to Rodrigues rotation vector."""
    rot = np.asarray(R, dtype=np.float64)
    require(rot.shape == (3, 3), "R must be (3, 3)", rot.shape)
    tr = float(np.trace(rot))
    cos_theta = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.array(
            [
                0.5 * (rot[2, 1] - rot[1, 2]),
                0.5 * (rot[0, 2] - rot[2, 0]),
                0.5 * (rot[1, 0] - rot[0, 1]),
            ],
            dtype=np.float64,
        )
    if math.isclose(theta, math.pi, rel_tol=1e-5):
        # Near pi: use diagonal terms to extract axis
        diag = np.maximum(0.0, (np.diag(rot) + 1.0) / 2.0)
        axis = np.sqrt(diag)
        if rot[0, 1] < 0:
            axis[1] = -axis[1]
        if rot[0, 2] < 0:
            axis[2] = -axis[2]
        norm = float(np.linalg.norm(axis))
        if norm > 0:
            axis /= norm
        return axis * theta
    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * math.sin(theta))
    return axis * theta


# ---------------------------------------------------------------------------
# Camera Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera model with intrinsics, extrinsics, and analytic Jacobians.

    Extrinsics transform world coordinates into camera frame:
        p_camera = R @ p_world + t
    """

    fx: float
    fy: float
    cx: float
    cy: float
    R: FloatArray
    t: FloatArray
    rvec: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self) -> None:
        require(self.fx > 0.0, "fx must be positive", self.fx)
        require(self.fy > 0.0, "fy must be positive", self.fy)
        require(self.R.shape == (3, 3), "R must be (3, 3)", self.R.shape)
        require(self.t.shape == (3,), "t must be (3,)", self.t.shape)
        r_ortho = self.R.T @ self.R
        require(
            bool(np.allclose(r_ortho, np.eye(3), atol=1e-4)),
            "R must be an orthogonal matrix",
        )
        det_r = float(np.linalg.det(self.R))
        require(bool(np.isclose(det_r, 1.0, atol=1e-4)), "det(R) must be +1", det_r)

        if np.all(self.rvec == 0.0) and not np.allclose(self.R, np.eye(3)):
            computed_rvec = rodrigues_from_rotation(self.R)
            object.__setattr__(self, "rvec", computed_rvec)

    @classmethod
    def from_camera_pose(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        R_cam_to_world: FloatArray,
        t_cam_to_world: FloatArray,
    ) -> CameraModel:
        """Construct CameraModel from camera pose in world coordinates."""
        R_cw = np.asarray(R_cam_to_world.T, dtype=np.float64)
        t_cw = np.asarray(-R_cw @ t_cam_to_world, dtype=np.float64)
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, R=R_cw, t=t_cw)

    @property
    def camera_params(self) -> FloatArray:
        """10-parameter vector: [fx, fy, cx, cy, rx, ry, rz, tx, ty, tz]."""
        return np.array(
            [self.fx, self.fy, self.cx, self.cy, *self.rvec, *self.t],
            dtype=np.float64,
        )

    def with_camera_params(self, params: FloatArray) -> CameraModel:
        """Construct a new CameraModel from a 10-parameter vector."""
        p = np.asarray(params, dtype=np.float64)
        require(p.shape == (10,), "params must be 10-vector", p.shape)
        fx, fy, cx, cy = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        rvec = p[4:7]
        t = p[7:10]
        R = rotation_from_rodrigues(rvec)
        return CameraModel(fx=fx, fy=fy, cx=cx, cy=cy, R=R, t=t, rvec=rvec)

    def _to_3d_points(self, p_world: FloatArray) -> tuple[FloatArray, bool]:
        """Convert input world points to shape (..., 3) and return is_planar flag."""
        pw = np.asarray(p_world, dtype=np.float64)
        require(pw.ndim >= 1, "p_world must have at least 1 dimension", pw.shape)
        last_dim = pw.shape[-1]
        require(last_dim in (2, 3), "p_world last dimension must be 2 or 3", pw.shape)
        if last_dim == 2:
            zeros = np.zeros((*pw.shape[:-1], 1), dtype=np.float64)
            return np.concatenate([pw, zeros], axis=-1), True
        return pw, False

    def _camera_frame_points(
        self, p_world: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, bool]:
        """Transform world points to camera coordinates (pw_3d, xc, yc, zc, is_planar)."""
        pw_3d, is_planar = self._to_3d_points(p_world)
        pc = pw_3d @ self.R.T + self.t
        zc = pc[..., 2]
        require(
            bool(np.all(zc > 0.0)),
            "all projected points must have positive depth (Z_c > 0)",
        )
        return pw_3d, pc[..., 0], pc[..., 1], zc, is_planar

    def _jacobian_camera_frame(
        self, xc: FloatArray, yc: FloatArray, zc: FloatArray
    ) -> FloatArray:
        """Analytic Jacobian d_proj / d_pc of shape (..., 2, 3)."""
        zc2 = zc * zc
        zeros = np.zeros_like(zc)
        j_cam_u = np.stack([self.fx / zc, zeros, -self.fx * xc / zc2], axis=-1)
        j_cam_v = np.stack([zeros, self.fy / zc, -self.fy * yc / zc2], axis=-1)
        return np.stack([j_cam_u, j_cam_v], axis=-2)

    def project(self, p_world: FloatArray) -> FloatArray:
        """Project world points into 2-D pixel coordinates (u, v)."""
        _, xc, yc, zc, _ = self._camera_frame_points(p_world)
        u = self.fx * (xc / zc) + self.cx
        v = self.fy * (yc / zc) + self.cy
        return np.stack([u, v], axis=-1)

    def jacobian_p_world(self, p_world: FloatArray) -> FloatArray:
        """Analytic Jacobian d_proj / d_p_world.

        Returns shape (..., 2, 3) for 3-D inputs or (..., 2, 2) for planar 2-D inputs.
        """
        _, xc, yc, zc, is_planar = self._camera_frame_points(p_world)
        j_cam = self._jacobian_camera_frame(xc, yc, zc)

        # d_proj / d_pw = j_cam @ R
        j_pw = j_cam @ self.R
        if is_planar:
            return j_pw[..., :2]
        return j_pw

    def jacobian_intrinsics(self, p_world: FloatArray) -> FloatArray:
        """Analytic Jacobian d_proj / d_intrinsics w.r.t [fx, fy, cx, cy].

        Returns shape (..., 2, 4).
        """
        _, xc, yc, zc, _ = self._camera_frame_points(p_world)
        zeros = np.zeros_like(zc)
        ones = np.ones_like(zc)

        row_u = np.stack([xc / zc, zeros, ones, zeros], axis=-1)
        row_v = np.stack([zeros, yc / zc, zeros, ones], axis=-1)
        return np.stack([row_u, row_v], axis=-2)

    def _d_rotation_d_rvec(self) -> FloatArray:
        """Analytic derivative dR / dr_k of Rodrigues map, returns shape (3, 3, 3) indexed by (k, row, col)."""
        r = self.rvec
        theta = float(np.linalg.norm(r))
        d_R = np.zeros((3, 3, 3), dtype=np.float64)
        if theta < 1e-10:
            for k in range(3):
                e_k = np.zeros(3, dtype=np.float64)
                e_k[k] = 1.0
                d_R[k] = skew3(e_k)
            return d_R

        a = math.cos(theta)
        b = (1.0 - math.cos(theta)) / (theta * theta)
        c = math.sin(theta) / theta
        k_skew = skew3(r)
        r_outer = np.outer(r, r)

        for k in range(3):
            e_k = np.zeros(3, dtype=np.float64)
            e_k[k] = 1.0
            r_k = r[k]
            da_k = -c * r_k
            db_k = ((c - 2.0 * b) / (theta * theta)) * r_k
            dc_k = ((a - c) / (theta * theta)) * r_k
            d_outer = np.outer(e_k, r) + np.outer(r, e_k)
            d_skew = skew3(e_k)

            d_R[k] = (
                da_k * np.eye(3, dtype=np.float64)
                + db_k * r_outer
                + b * d_outer
                + dc_k * k_skew
                + c * d_skew
            )
        return d_R

    def jacobian_extrinsics(self, p_world: FloatArray) -> FloatArray:
        """Analytic Jacobian d_proj / d_extrinsics w.r.t [rx, ry, rz, tx, ty, tz].

        Returns shape (..., 2, 6).
        """
        pw_3d, xc, yc, zc, _ = self._camera_frame_points(p_world)
        j_cam = self._jacobian_camera_frame(xc, yc, zc)

        d_R = self._d_rotation_d_rvec()  # (3, 3, 3)
        # d_pc / d_r_k = pw_3d @ d_R[k].T
        cols_rot: list[FloatArray] = []
        for k in range(3):
            d_pc_k = pw_3d @ d_R[k].T  # (..., 3)
            # j_cam @ d_pc_k: (..., 2)
            cols_rot.append(np.einsum("...ij,...j->...i", j_cam, d_pc_k))

        # d_pc / d_t_k is j_cam[:, :, k]
        cols_trans = [j_cam[..., :, k] for k in range(3)]
        all_cols = cols_rot + cols_trans
        return np.stack(all_cols, axis=-1)  # (..., 2, 6)

    def jacobian_camera_params(self, p_world: FloatArray) -> FloatArray:
        """Full analytic Jacobian w.r.t all 10 camera parameters [fx, fy, cx, cy, rx, ry, rz, tx, ty, tz].

        Returns shape (..., 2, 10).
        """
        j_int = self.jacobian_intrinsics(p_world)  # (..., 2, 4)
        j_ext = self.jacobian_extrinsics(p_world)  # (..., 2, 6)
        return np.concatenate([j_int, j_ext], axis=-1)


# ---------------------------------------------------------------------------
# Robust Loss Kernels
# ---------------------------------------------------------------------------


class RobustKernel(Protocol):
    """Protocol for M-estimator robust loss kernels."""

    def rho(self, s: FloatArray | float) -> FloatArray | float:
        """Cost function rho(r^2) where s = r^2."""
        ...

    def psi(self, r: FloatArray | float) -> FloatArray | float:
        """Influence function psi(r) = d/dr [rho(r^2)]."""
        ...

    def weight(self, r: FloatArray | float) -> FloatArray | float:
        """IRLS weight w(r) = psi(r)/r."""
        ...

    def psi_deriv(self, r: FloatArray | float) -> FloatArray | float:
        """Derivative of influence function psi'(r)."""
        ...

    def w(self, r: FloatArray | float) -> FloatArray | float:
        """Alias for weight(r)."""
        ...

    def psi_prime(self, r: FloatArray | float) -> FloatArray | float:
        """Alias for psi_deriv(r)."""
        ...


@dataclass(frozen=True)
class HuberKernel:
    """Huber robust loss kernel with threshold delta."""

    delta: float

    def __post_init__(self) -> None:
        require(self.delta > 0.0, "delta must be positive", self.delta)

    def rho(self, s: FloatArray | float) -> FloatArray | float:
        s_arr = np.asarray(s, dtype=np.float64)
        delta2 = self.delta * self.delta
        res = np.where(
            s_arr <= delta2,
            0.5 * s_arr,
            self.delta * np.sqrt(np.maximum(0.0, s_arr)) - 0.5 * delta2,
        )
        return float(res) if np.ndim(s) == 0 else res

    def psi(self, r: FloatArray | float) -> FloatArray | float:
        r_arr = np.asarray(r, dtype=np.float64)
        abs_r = np.abs(r_arr)
        res = np.where(
            abs_r <= self.delta,
            r_arr,
            self.delta * np.sign(r_arr),
        )
        return float(res) if np.ndim(r) == 0 else res

    def weight(self, r: FloatArray | float) -> FloatArray | float:
        r_arr = np.asarray(r, dtype=np.float64)
        abs_r = np.abs(r_arr)
        safe_r = np.maximum(abs_r, 1e-14)
        res = np.where(
            abs_r <= self.delta,
            1.0,
            self.delta / safe_r,
        )
        return float(res) if np.ndim(r) == 0 else res

    def psi_deriv(self, r: FloatArray | float) -> FloatArray | float:
        r_arr = np.asarray(r, dtype=np.float64)
        abs_r = np.abs(r_arr)
        res = np.where(abs_r <= self.delta, 1.0, 0.0)
        return float(res) if np.ndim(r) == 0 else res

    def w(self, r: FloatArray | float) -> FloatArray | float:
        return self.weight(r)

    def psi_prime(self, r: FloatArray | float) -> FloatArray | float:
        return self.psi_deriv(r)


@dataclass(frozen=True)
class CauchyKernel:
    """Cauchy (Lorentzian) robust loss kernel with scale c."""

    c: float

    def __post_init__(self) -> None:
        require(self.c > 0.0, "c must be positive", self.c)

    def rho(self, s: FloatArray | float) -> FloatArray | float:
        s_arr = np.asarray(s, dtype=np.float64)
        c2 = self.c * self.c
        res = 0.5 * c2 * np.log1p(s_arr / c2)
        return float(res) if np.ndim(s) == 0 else res

    def psi(self, r: FloatArray | float) -> FloatArray | float:
        r_arr = np.asarray(r, dtype=np.float64)
        c2 = self.c * self.c
        res = r_arr / (1.0 + (r_arr * r_arr) / c2)
        return float(res) if np.ndim(r) == 0 else res

    def weight(self, r: FloatArray | float) -> FloatArray | float:
        r_arr = np.asarray(r, dtype=np.float64)
        c2 = self.c * self.c
        res = 1.0 / (1.0 + (r_arr * r_arr) / c2)
        return float(res) if np.ndim(r) == 0 else res

    def psi_deriv(self, r: FloatArray | float) -> FloatArray | float:
        r_arr = np.asarray(r, dtype=np.float64)
        c2 = self.c * self.c
        r2_c2 = (r_arr * r_arr) / c2
        res = (1.0 - r2_c2) / ((1.0 + r2_c2) ** 2)
        return float(res) if np.ndim(r) == 0 else res

    def w(self, r: FloatArray | float) -> FloatArray | float:
        return self.weight(r)

    def psi_prime(self, r: FloatArray | float) -> FloatArray | float:
        return self.psi_deriv(r)


@dataclass(frozen=True)
class GemanMcClureKernel:
    """Geman-McClure robust loss kernel with scale c."""

    c: float

    def __post_init__(self) -> None:
        require(self.c > 0.0, "c must be positive", self.c)

    def rho(self, s: FloatArray | float) -> FloatArray | float:
        s_arr = np.asarray(s, dtype=np.float64)
        c2 = self.c * self.c
        res = 0.5 * s_arr / (1.0 + s_arr / c2)
        return float(res) if np.ndim(s) == 0 else res

    def psi(self, r: FloatArray | float) -> FloatArray | float:
        r_arr = np.asarray(r, dtype=np.float64)
        c2 = self.c * self.c
        res = r_arr / ((1.0 + (r_arr * r_arr) / c2) ** 2)
        return float(res) if np.ndim(r) == 0 else res

    def weight(self, r: FloatArray | float) -> FloatArray | float:
        r_arr = np.asarray(r, dtype=np.float64)
        c2 = self.c * self.c
        res = 1.0 / ((1.0 + (r_arr * r_arr) / c2) ** 2)
        return float(res) if np.ndim(r) == 0 else res

    def psi_deriv(self, r: FloatArray | float) -> FloatArray | float:
        r_arr = np.asarray(r, dtype=np.float64)
        c2 = self.c * self.c
        r2_c2 = (r_arr * r_arr) / c2
        res = (1.0 - 3.0 * r2_c2) / ((1.0 + r2_c2) ** 3)
        return float(res) if np.ndim(r) == 0 else res

    def w(self, r: FloatArray | float) -> FloatArray | float:
        return self.weight(r)

    def psi_prime(self, r: FloatArray | float) -> FloatArray | float:
        return self.psi_deriv(r)


@dataclass(frozen=True)
class TukeyKernel:
    """Tukey (bisquare) redescending robust loss kernel with cutoff c."""

    c: float

    def __post_init__(self) -> None:
        require(self.c > 0.0, "c must be positive", self.c)

    def rho(self, s: FloatArray | float) -> FloatArray | float:
        s_arr = np.asarray(s, dtype=np.float64)
        c2 = self.c * self.c
        res = np.where(
            s_arr <= c2,
            (c2 / 6.0) * (1.0 - (1.0 - s_arr / c2) ** 3),
            c2 / 6.0,
        )
        return float(res) if np.ndim(s) == 0 else res

    def psi(self, r: FloatArray | float) -> FloatArray | float:
        r_arr = np.asarray(r, dtype=np.float64)
        abs_r = np.abs(r_arr)
        c2 = self.c * self.c
        res = np.where(
            abs_r <= self.c,
            r_arr * ((1.0 - (r_arr * r_arr) / c2) ** 2),
            0.0,
        )
        return float(res) if np.ndim(r) == 0 else res

    def weight(self, r: FloatArray | float) -> FloatArray | float:
        r_arr = np.asarray(r, dtype=np.float64)
        abs_r = np.abs(r_arr)
        c2 = self.c * self.c
        res = np.where(
            abs_r <= self.c,
            (1.0 - (r_arr * r_arr) / c2) ** 2,
            0.0,
        )
        return float(res) if np.ndim(r) == 0 else res

    def psi_deriv(self, r: FloatArray | float) -> FloatArray | float:
        r_arr = np.asarray(r, dtype=np.float64)
        abs_r = np.abs(r_arr)
        c2 = self.c * self.c
        r2_c2 = (r_arr * r_arr) / c2
        res = np.where(
            abs_r <= self.c,
            (1.0 - r2_c2) * (1.0 - 5.0 * r2_c2),
            0.0,
        )
        return float(res) if np.ndim(r) == 0 else res

    def w(self, r: FloatArray | float) -> FloatArray | float:
        return self.weight(r)

    def psi_prime(self, r: FloatArray | float) -> FloatArray | float:
        return self.psi_deriv(r)


# ---------------------------------------------------------------------------
# Marker Attachment Offsets Factor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkerAttachmentFactor:
    """Rigid marker attachment offsets factor for planar kinematic chains.

    Local marker position is parameterized as:
        p_local = p_0 + delta_p
    where p_0 is nominal offset and delta_p is the refined geometry parameter.
    """

    chain: PlanarChain
    link_indices: npt.NDArray[np.int64]
    nominal_offsets: FloatArray

    def __post_init__(self) -> None:
        require(self.link_indices.ndim == 1, "link_indices must be 1-D")
        require(
            self.nominal_offsets.shape == (self.link_indices.size, 2),
            "nominal_offsets must be (M, 2)",
            self.nominal_offsets.shape,
        )
        require(
            bool(np.all(self.link_indices >= 0)),
            "link indices must be non-negative",
        )
        require(
            bool(np.all(self.link_indices < self.chain.n_links)),
            "link indices must be < n_links",
        )

    @property
    def n_markers(self) -> int:
        return int(self.link_indices.size)

    def effective_offsets(self, delta_offsets: FloatArray | None = None) -> FloatArray:
        """Effective local marker positions p_0 + delta_p."""
        if delta_offsets is None:
            return self.nominal_offsets
        delta = np.asarray(delta_offsets, dtype=np.float64)
        require(
            delta.shape == self.nominal_offsets.shape,
            "delta_offsets must match nominal shape (M, 2)",
            delta.shape,
        )
        return self.nominal_offsets + delta

    def to_marker_set(self, delta_offsets: FloatArray | None = None) -> PlanarMarkerSet:
        """Build PlanarMarkerSet for kinematic routines."""
        return PlanarMarkerSet(
            link_index=self.link_indices,
            offsets=self.effective_offsets(delta_offsets),
        )

    def marker_positions(
        self, q: FloatArray, delta_offsets: FloatArray | None = None
    ) -> FloatArray:
        """Batched world marker positions (N, M, 2)."""
        marker_set = self.to_marker_set(delta_offsets)
        return self.chain.marker_positions(q, marker_set)

    def jacobian_offsets(self, q: FloatArray) -> FloatArray:
        """Analytic Jacobian d(p_world) / d(delta_p).

        For marker m attached to link i with absolute orientation phi_i(q),
        d(p_world_m) / d(delta_p_m) is the 2x2 rotation matrix R_i = [e_i, n_i].
        Returns shape (N, M, 2, 2) where the last two axes are the (2, 2) block.
        """
        q_arr = np.asarray(q, dtype=np.float64)
        require(
            q_arr.ndim == 2 and q_arr.shape[1] == self.chain.n_links,
            "q must be (N, n_links)",
            q_arr.shape,
        )
        # Chain forward kinematics gives unit axes e and n per link
        kin = self.chain._kinematics(q_arr, np.zeros_like(q_arr), np.zeros_like(q_arr))
        # e and n have shape (N, n_links, 2)
        idx = self.link_indices
        e_markers = kin.e[:, idx]  # (N, M, 2)
        n_markers = kin.n[:, idx]  # (N, M, 2)
        # Stack as columns of 2x2 matrix: [e_markers, n_markers]
        return np.stack([e_markers, n_markers], axis=-1)  # (N, M, 2, 2)

    def jacobian_kinematics(
        self, q: FloatArray, delta_offsets: FloatArray | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """Analytic Jacobians w.r.t joint angles q and link lengths l.

        Returns (d/dq, d/dl) each of shape (N, M, 2, n_links).
        """
        marker_set = self.to_marker_set(delta_offsets)
        return self.chain.marker_jacobians(q, marker_set)


@dataclass(frozen=True)
class OffsetRecoveryResult:
    """Recovery diagnostic and result for marker attachment offsets."""

    recovered_delta: FloatArray
    observable_fraction: float
    n_observable: int
    singular_values: FloatArray
    residual_rms: float


def recover_marker_offsets(
    factor: MarkerAttachmentFactor,
    q: FloatArray,
    observed_positions: FloatArray,
    threshold: float = 1e-4,
) -> OffsetRecoveryResult:
    """Recover local marker offsets delta_p from kinematics observations and report observability.

    Solves the linear normal equations for each marker offset and computes singular-value
    spectrum to report the observable fraction of attachment degrees of freedom.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    obs = np.asarray(observed_positions, dtype=np.float64)
    n_nodes = q_arr.shape[0]
    n_markers = factor.n_markers
    require(
        obs.shape == (n_nodes, n_markers, 2),
        "observed_positions must be (N, M, 2)",
        obs.shape,
    )

    # Nominal positions at delta_p = 0
    pos_nom = factor.marker_positions(q_arr, None)  # (N, M, 2)
    # Jacobians: (N, M, 2, 2)
    j_delta = factor.jacobian_offsets(q_arr)

    recovered = np.zeros((n_markers, 2), dtype=np.float64)
    all_svals: list[float] = []
    total_dof = 2 * n_markers
    observable_dof = 0

    # Each marker's offset delta_p_m can be solved independently:
    # A_m delta_p_m = y_m - pos_nom_m
    for m in range(n_markers):
        # A_m: shape (2*N, 2)
        A_m = j_delta[:, m].reshape(-1, 2)
        # b_m: shape (2*N,)
        b_m = (obs[:, m] - pos_nom[:, m]).reshape(-1)

        # SVD analysis
        svals = np.linalg.svd(A_m, compute_uv=False)
        all_svals.extend(svals.tolist())
        obs_m = int(np.sum(svals > threshold))
        observable_dof += obs_m

        sol, _, _, _ = np.linalg.lstsq(A_m, b_m, rcond=threshold)
        recovered[m] = sol

    # Total residual RMS
    final_pos = factor.marker_positions(q_arr, recovered)
    residual_rms = float(np.sqrt(np.mean((final_pos - obs) ** 2)))
    observable_fraction = float(observable_dof / total_dof)

    return OffsetRecoveryResult(
        recovered_delta=recovered,
        observable_fraction=observable_fraction,
        n_observable=observable_dof,
        singular_values=np.array(sorted(all_svals, reverse=True), dtype=np.float64),
        residual_rms=residual_rms,
    )
