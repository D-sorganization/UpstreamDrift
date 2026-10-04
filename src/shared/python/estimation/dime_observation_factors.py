"""Robust Marker and Markerless Observation Factors (#11421, #11424).

Provides unified observation factors for dynamics-informed mocap matching (DIME):
- 3D marker factors with explicit metric units (meters) and rigid attachment semantics.
- 2D markerless factors with pixel units, pinhole camera projection, and distortion.
- Calibrated confidence and anisotropic noise covariance whitening (separate from detector score).
- Robust loss kernels (Huber, Tukey, Cauchy, Pseudo-Huber) separate from noise weighting.
- Clean partitioning of held-out validation observations and occlusion mask handling without zero-filling.
- Camera transform inversion and chirality validation (positive depth and right-handed SO(3)).
- Exact quaternion sign equivalence across orientation representations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.contracts import (
    ContractViolationError,
    PreconditionError,
    require,
)
from src.shared.python.estimation.dime_contracts import DimeCompleteState
from src.shared.python.estimation.dime_manifest import CANONICAL_DIME_UNITS
from src.shared.python.estimation.residuals import project_pinhole

RobustKernelType: TypeAlias = Literal[
    "linear", "huber", "cauchy", "tukey", "pseudo_huber"
]
ObservationFactorType: TypeAlias = Literal["marker_3d", "markerless_2d"]

_VALID_KERNELS: Final[tuple[str, ...]] = (
    "linear",
    "huber",
    "cauchy",
    "tukey",
    "pseudo_huber",
)


# ==============================================================================
# Domain Exceptions
# ==============================================================================


class ChiralityViolationError(PreconditionError):
    """Raised when coordinates are mirrored or points violate camera chirality."""


class TimingViolationError(PreconditionError):
    """Raised when observation timestamps are irregular, non-monotonic, or non-finite."""


class CovarianceValidationError(PreconditionError):
    """Raised when observation covariance is invalid, non-positive, or dimension-mismatched."""


# ==============================================================================
# Helper Utilities
# ==============================================================================


def _make_readonly_array(arr: np.ndarray) -> np.ndarray:
    """Return a read-only copy of a float64 numpy array."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix in SO(3).

    Guarantees quaternion sign equivalence: R(q) == R(-q) for all unit quaternions.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    require(q_arr.shape == (4,), "Quaternion must have shape (4,)")
    norm_sq = float(np.sum(q_arr**2))
    require(abs(norm_sq - 1.0) < 1.0e-3, "Quaternion must have unit norm", norm_sq)
    w, x, y, z = q_arr / np.sqrt(norm_sq)
    return np.array(
        [
            [1.0 - 2.0 * (y**2 + z**2), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x**2 + z**2), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x**2 + y**2)],
        ],
        dtype=np.float64,
    )


def invert_camera_extrinsics(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert rigid camera transform (R, t) -> (R^T, -t @ R) with SO(3) validation.

    Enforces right-handedness: det(R) must be +1 within tolerance. Mirror reflection
    transforms with negative determinant violate chirality and are rejected.
    """
    r_arr = np.asarray(rotation, dtype=np.float64)
    t_arr = np.asarray(translation, dtype=np.float64)
    require(r_arr.shape == (3, 3), "Rotation must have shape (3, 3)")
    require(t_arr.shape == (3,), "Translation must have shape (3,)")

    det = float(np.linalg.det(r_arr))
    if det < 0.0 or abs(det - 1.0) > 1.0e-3:
        raise ChiralityViolationError(
            f"Rotation matrix must be right-handed SO(3) with det=+1, got det={det:.6f}"
        )
    ortho_err = float(np.max(np.abs(r_arr.T @ r_arr - np.eye(3))))
    if ortho_err > 1.0e-3:
        raise ChiralityViolationError(
            f"Rotation matrix must be orthonormal, max error={ortho_err:.6e}"
        )

    r_inv = r_arr.T
    t_inv = -t_arr @ r_arr
    return r_inv, t_inv


# ==============================================================================
# Robust Loss Kernels
# ==============================================================================


@dataclass(frozen=True)
class RobustLossKernel:
    """Robust loss kernel separate from confidence and noise covariance weighting.

    Supported kernels:
    - linear: standard unweighted L2 loss.
    - huber: linear penalty for residuals exceeding tuning_constant delta.
    - cauchy: smooth sub-quadratic log penalty for heavier tails.
    - tukey: hard redescending biweight rejecting outliers beyond cutoff c.
    - pseudo_huber: smooth differentiable approximation to Huber.
    """

    kernel: RobustKernelType = "huber"
    tuning_constant: float = 1.345

    def __post_init__(self) -> None:
        require(
            self.tuning_constant > 0.0 and np.isfinite(self.tuning_constant),
            "tuning_constant must be positive and finite",
            self.tuning_constant,
        )
        require(
            self.kernel in _VALID_KERNELS,
            f"Invalid kernel '{self.kernel}', expected one of {_VALID_KERNELS}",
        )

    def loss(self, sq_residual: np.ndarray) -> np.ndarray:
        """Evaluate rho(s) for squared residual / Mahalanobis distance s."""
        s = np.asarray(sq_residual, dtype=np.float64)
        c = self.tuning_constant
        c_sq = c**2

        if self.kernel == "linear":
            return s
        if self.kernel == "huber":
            r = np.sqrt(np.maximum(s, 0.0))
            return np.where(s <= c_sq, s, 2.0 * c * r - c_sq)
        if self.kernel == "cauchy":
            return c_sq * np.log1p(s / c_sq)
        if self.kernel == "tukey":
            ratio = np.clip(s / c_sq, 0.0, 1.0)
            val = (c_sq / 6.0) * (1.0 - (1.0 - ratio) ** 3)
            return np.where(s <= c_sq, val, c_sq / 6.0)
        if self.kernel == "pseudo_huber":
            return 2.0 * c_sq * (np.sqrt(1.0 + s / c_sq) - 1.0)
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def weight(self, residual_norm: np.ndarray) -> np.ndarray:
        """IRLS weight w(r) such that rho'(r^2) = w(r)."""
        r = np.asarray(residual_norm, dtype=np.float64)
        c = self.tuning_constant
        r_safe = np.maximum(r, 1.0e-12)

        if self.kernel == "linear":
            return np.ones_like(r)
        if self.kernel == "huber":
            return np.where(r <= c, 1.0, c / r_safe)
        if self.kernel == "cauchy":
            return 1.0 / (1.0 + (r / c) ** 2)
        if self.kernel == "tukey":
            return np.where(r <= c, (1.0 - (r / c) ** 2) ** 2, 0.0)
        if self.kernel == "pseudo_huber":
            return 1.0 / np.sqrt(1.0 + (r / c) ** 2)
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def sqrt_weight(self, residual_norm: np.ndarray) -> np.ndarray:
        """Square root weight sqrt(w(r)) for residual scaling."""
        return np.sqrt(np.maximum(self.weight(residual_norm), 0.0))

    def apply_robust_weight(self, whitened: np.ndarray, point_dim: int) -> np.ndarray:
        """Scale whitened point residuals by square-root robust weights."""
        if whitened.size == 0:
            return whitened
        n_points = whitened.size // point_dim
        reshaped = whitened.reshape(n_points, point_dim)
        norms = np.linalg.norm(reshaped, axis=1)
        sqrt_w = self.sqrt_weight(norms)[:, None]
        return (reshaped * sqrt_w).reshape(-1)


# ==============================================================================
# Timing & Metadata Specifications
# ==============================================================================


@dataclass(frozen=True)
class ObservationTiming:
    """Validated timing metadata enforcing strict temporal monotonicity."""

    timestamps: np.ndarray
    dt: float | None = None
    frame_indices: np.ndarray | None = None
    sample_rate_hz: float | None = None

    def __post_init__(self) -> None:
        raw_t = np.asarray(self.timestamps, dtype=np.float64)
        if raw_t.ndim != 1 or len(raw_t) == 0:
            raise TimingViolationError("timestamps must be a non-empty 1D array")
        if not np.all(np.isfinite(raw_t)):
            raise TimingViolationError("timestamps must contain strictly finite values")
        if len(raw_t) > 1 and not np.all(np.diff(raw_t) > 0.0):
            raise TimingViolationError(
                "timestamps must be strictly monotonic (positive dt)"
            )

        if self.dt is not None:
            if not np.isfinite(self.dt) or self.dt <= 0.0:
                raise TimingViolationError(
                    f"Declared dt must be strictly positive and finite, got {self.dt}"
                )

        if self.sample_rate_hz is not None:
            if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
                raise TimingViolationError(
                    f"sample_rate_hz must be strictly positive, got {self.sample_rate_hz}"
                )

        if self.frame_indices is not None:
            raw_idx = np.asarray(self.frame_indices, dtype=np.int64)
            if raw_idx.shape != raw_t.shape:
                raise TimingViolationError(
                    "frame_indices length must match timestamps length"
                )
            if len(raw_idx) > 1 and not np.all(np.diff(raw_idx) > 0):
                raise TimingViolationError("frame_indices must be strictly increasing")

        object.__setattr__(self, "timestamps", _make_readonly_array(raw_t))
        if self.frame_indices is not None:
            arr_idx = np.array(self.frame_indices, dtype=np.int64, copy=True)
            arr_idx.flags.writeable = False
            object.__setattr__(self, "frame_indices", arr_idx)


@dataclass(frozen=True)
class MarkerAttachment:
    """Declared marker attachment semantics on a kinematic link or segment."""

    name: str
    body_or_joint: str
    offset_m: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        require(bool(self.name.strip()), "Marker attachment name must be non-empty")
        require(
            bool(self.body_or_joint.strip()),
            "body_or_joint identifier must be non-empty",
        )
        offset = np.asarray(self.offset_m, dtype=np.float64)
        require(offset.shape == (3,), "offset_m must have shape (3,)")
        require(bool(np.all(np.isfinite(offset))), "offset_m must be strictly finite")
        require(self.weight > 0.0, "weight must be strictly positive")
        object.__setattr__(self, "offset_m", _make_readonly_array(offset))


@dataclass(frozen=True)
class DimeCameraParameters:
    """Calibrated pinhole camera parameters with Brown-Conrady lens distortion."""

    camera_id: str
    matrix: np.ndarray
    rotation_world_to_camera: np.ndarray
    translation_world_to_camera: np.ndarray
    distortion: np.ndarray | None = None
    image_size_px: tuple[int, int] | None = None
    scale_gauge_fixed: bool = True

    def __post_init__(self) -> None:
        require(bool(self.camera_id.strip()), "camera_id must be non-empty")
        k_mat = np.asarray(self.matrix, dtype=np.float64)
        require(k_mat.shape == (3, 3), "Camera matrix must have shape (3, 3)")
        require(
            k_mat[0, 0] > 0.0 and k_mat[1, 1] > 0.0, "Focal lengths must be positive"
        )

        r_cw = np.asarray(self.rotation_world_to_camera, dtype=np.float64)
        t_cw = np.asarray(self.translation_world_to_camera, dtype=np.float64)
        require(r_cw.shape == (3, 3), "rotation_world_to_camera must have shape (3, 3)")
        require(t_cw.shape == (3,), "translation_world_to_camera must have shape (3,)")

        det = float(np.linalg.det(r_cw))
        if det < 0.0 or abs(det - 1.0) > 1.0e-3:
            raise ChiralityViolationError(
                f"Camera rotation matrix must be right-handed SO(3) with det=+1, got {det:.6f}"
            )
        ortho_err = float(np.max(np.abs(r_cw.T @ r_cw - np.eye(3))))
        if ortho_err > 1.0e-3:
            raise ChiralityViolationError(
                f"Camera rotation matrix must be orthonormal, max error={ortho_err:.6e}"
            )

        object.__setattr__(self, "matrix", _make_readonly_array(k_mat))
        object.__setattr__(self, "rotation_world_to_camera", _make_readonly_array(r_cw))
        object.__setattr__(
            self, "translation_world_to_camera", _make_readonly_array(t_cw)
        )

        if self.distortion is not None:
            dist = np.asarray(self.distortion, dtype=np.float64)
            require(
                dist.ndim == 1 and dist.shape[0] in (4, 5),
                "distortion must have shape (4,) or (5,)",
            )
            object.__setattr__(self, "distortion", _make_readonly_array(dist))

    @classmethod
    def from_world_from_camera(
        cls,
        camera_id: str,
        matrix: np.ndarray,
        rotation_world_from_camera: np.ndarray,
        translation_world_from_camera_m: np.ndarray,
        distortion: np.ndarray | None = None,
        image_size_px: tuple[int, int] | None = None,
        scale_gauge_fixed: bool = True,
    ) -> DimeCameraParameters:
        """Construct camera parameters by inverting world-from-camera extrinsics."""
        r_cw, t_cw = invert_camera_extrinsics(
            rotation_world_from_camera, translation_world_from_camera_m
        )
        return cls(
            camera_id=camera_id,
            matrix=matrix,
            rotation_world_to_camera=r_cw,
            translation_world_to_camera=t_cw,
            distortion=distortion,
            image_size_px=image_size_px,
            scale_gauge_fixed=scale_gauge_fixed,
        )

    def invert(self) -> DimeCameraParameters:
        """Return inverted camera parameters exchanging world and camera frames."""
        r_wc, t_wc = invert_camera_extrinsics(
            self.rotation_world_to_camera, self.translation_world_to_camera
        )
        return DimeCameraParameters(
            camera_id=f"{self.camera_id}_inv",
            matrix=self.matrix,
            rotation_world_to_camera=r_wc,
            translation_world_to_camera=t_wc,
            distortion=self.distortion,
            image_size_px=self.image_size_px,
            scale_gauge_fixed=self.scale_gauge_fixed,
        )

    def validate_chirality(
        self, points_world: np.ndarray, min_depth: float = 1.0e-4
    ) -> None:
        """Verify that world points project to positive depth (Z > 0) in front of the camera."""
        pts = np.asarray(points_world, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        pts_cam = (
            pts @ self.rotation_world_to_camera.T + self.translation_world_to_camera
        )
        depths = pts_cam[:, 2]
        if np.any(depths <= min_depth):
            min_z = float(np.min(depths))
            raise ChiralityViolationError(
                f"Chirality violation: point depth Z_cam <= {min_depth} (got min Z={min_z:.6e})"
            )

    def project(
        self, points_world: np.ndarray, check_chirality: bool = True
    ) -> np.ndarray:
        """Project world points into pixel coordinates using pinhole camera model."""
        if check_chirality:
            self.validate_chirality(points_world)
        return project_pinhole(
            points_world,
            self.matrix,
            rotation_world_to_camera=self.rotation_world_to_camera,
            translation_world_to_camera=self.translation_world_to_camera,
            distortion=self.distortion,
        )


@dataclass(frozen=True)
class HeldOutEvaluationReport:
    """Evaluation summary on held-out observations in canonical units."""

    num_held_out: int
    rms_error: float
    max_error: float
    canonical_unit: str
    raw_residuals: np.ndarray
    whitened_residuals: np.ndarray


# ==============================================================================
# Covariance Whitening Helper
# ==============================================================================


def _build_whitening_operators(
    covariance: float | np.ndarray, n_points: int, point_dim: int
) -> tuple[np.ndarray, bool]:
    """Parse covariance input and return (operator, is_diagonal).

    operator is:
    - 2D (n_points, point_dim) of scale factors (1 / sigma) if is_diagonal is True.
    - 3D (n_points, point_dim, point_dim) of Cholesky inverses L_inv if False.
    """
    raw_cov = np.asarray(covariance, dtype=np.float64)

    # Scalar variance
    if raw_cov.ndim == 0:
        val = float(raw_cov)
        if not np.isfinite(val) or val <= 0.0:
            raise CovarianceValidationError(
                f"Variance must be strictly positive and finite, got {val}"
            )
        inv_sigma = 1.0 / np.sqrt(val)
        return np.full((n_points, point_dim), inv_sigma, dtype=np.float64), True

    # 1D array of diagonal variances of shape (point_dim,)
    if raw_cov.ndim == 1 and raw_cov.shape == (point_dim,):
        if not np.all(np.isfinite(raw_cov)) or np.any(raw_cov <= 0.0):
            raise CovarianceValidationError("Variances must be strictly positive")
        inv_sigma = 1.0 / np.sqrt(raw_cov)
        return np.tile(inv_sigma, (n_points, 1)), True

    # 2D full covariance matrix of shape (point_dim, point_dim)
    if raw_cov.ndim == 2 and raw_cov.shape == (point_dim, point_dim):
        sym_err = float(np.max(np.abs(raw_cov - raw_cov.T)))
        if sym_err <= 1.0e-7:
            try:
                chol = np.linalg.cholesky(raw_cov)
                l_inv = np.linalg.inv(chol)
                return np.tile(l_inv, (n_points, 1, 1)), False
            except np.linalg.LinAlgError as exc:
                if n_points != point_dim:
                    raise CovarianceValidationError(
                        "Covariance matrix must be positive definite"
                    ) from exc

    # 2D array of per-point diagonal variances of shape (n_points, point_dim)
    if raw_cov.ndim == 2 and raw_cov.shape == (n_points, point_dim):
        if not np.all(np.isfinite(raw_cov)) or np.any(raw_cov <= 0.0):
            raise CovarianceValidationError("Variances must be strictly positive")
        return 1.0 / np.sqrt(raw_cov), True

    # 3D array of per-point full covariances (n_points, point_dim, point_dim)
    if raw_cov.ndim == 3 and raw_cov.shape == (n_points, point_dim, point_dim):
        l_invs = []
        for i in range(n_points):
            sub_cov = raw_cov[i]
            if np.max(np.abs(sub_cov - sub_cov.T)) > 1.0e-7:
                raise CovarianceValidationError(
                    f"Covariance matrix at index {i} must be symmetric"
                )
            try:
                chol = np.linalg.cholesky(sub_cov)
                l_invs.append(np.linalg.inv(chol))
            except np.linalg.LinAlgError as exc:
                raise CovarianceValidationError(
                    f"Covariance matrix at index {i} must be positive definite"
                ) from exc
        return np.array(l_invs, dtype=np.float64), False

    raise CovarianceValidationError(
        f"Incompatible covariance shape {raw_cov.shape} for point_dim={point_dim}"
    )


def _apply_whitening(
    diff: np.ndarray,
    operators: np.ndarray,
    is_diagonal: bool,
    active_indices: np.ndarray,
) -> np.ndarray:
    """Whiten raw differences using precomputed inverse covariance operators."""
    if diff.size == 0:
        return diff
    if is_diagonal:
        scales = operators[active_indices]
        return diff * scales
    # Full covariance whitening: z_k = L_inv_k @ diff_k
    sub_ops = operators[active_indices]
    return np.einsum("kij,kj->ki", sub_ops, diff)


def _setup_observation_masks(
    n_points: int,
    valid_mask: np.ndarray | None,
    held_out_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and partition observation masks into fit and held-out subsets."""
    if valid_mask is None:
        v_mask: np.ndarray = np.ones(n_points, dtype=bool)
    else:
        v_arr = np.asarray(valid_mask, dtype=bool)
        require(
            v_arr.shape == (n_points,),
            "valid_mask length must match observations",
        )
        v_mask = v_arr

    if held_out_mask is None:
        h_mask: np.ndarray = np.zeros(n_points, dtype=bool)
    else:
        h_arr = np.asarray(held_out_mask, dtype=bool)
        require(
            h_arr.shape == (n_points,),
            "held_out_mask length must match observations",
        )
        h_mask = h_arr

    fit_mask: np.ndarray = v_mask & (~h_mask)
    active_indices: np.ndarray = np.flatnonzero(fit_mask)
    held_out_indices: np.ndarray = np.flatnonzero(v_mask & h_mask)
    return v_mask, h_mask, fit_mask, active_indices, held_out_indices


def _validate_detector_scores(
    n_points: int,
    detector_scores: Sequence[float] | np.ndarray | None,
) -> np.ndarray | None:
    """Validate optional detector confidence scores in [0, 1]."""
    if detector_scores is None:
        return None
    scores = np.asarray(detector_scores, dtype=np.float64)
    require(
        scores.shape == (n_points,),
        "detector_scores shape must match observations",
    )
    require(
        bool(np.all((scores >= 0.0) & (scores <= 1.0))),
        "detector_scores entries must lie in [0, 1]",
    )
    return _make_readonly_array(scores)


def _finite_difference_jacobian(
    eval_fn: Callable[[np.ndarray], np.ndarray],
    q_vec: np.ndarray,
    step: float = 1.0e-6,
) -> np.ndarray:
    """Compute central finite-difference Jacobian of residual vector."""
    n_q = q_vec.size
    f0 = eval_fn(q_vec)
    jac = np.empty((len(f0), n_q), dtype=np.float64)
    for col in range(n_q):
        delta = np.zeros(n_q, dtype=np.float64)
        delta[col] = step
        f_plus = eval_fn(q_vec + delta)
        f_minus = eval_fn(q_vec - delta)
        jac[:, col] = (f_plus - f_minus) / (2.0 * step)
    return jac


def _build_held_out_report(
    diff: np.ndarray,
    operators: np.ndarray,
    is_diagonal: bool,
    held_out_indices: np.ndarray,
    canonical_unit: str,
) -> HeldOutEvaluationReport:
    """Build evaluation report on held-out residual differences."""
    norms = np.linalg.norm(diff, axis=1)
    rms = float(np.sqrt(np.mean(norms**2)))
    max_err = float(np.max(norms))
    whitened = _apply_whitening(diff, operators, is_diagonal, held_out_indices)
    return HeldOutEvaluationReport(
        num_held_out=len(held_out_indices),
        rms_error=rms,
        max_error=max_err,
        canonical_unit=canonical_unit,
        raw_residuals=diff,
        whitened_residuals=whitened,
    )


# ==============================================================================
# Unified Observation Factor Contract
# ==============================================================================


class DimeObservationFactor(ABC):
    """Abstract unified observation factor interface for DIME estimation."""

    @property
    @abstractmethod
    def factor_type(self) -> ObservationFactorType:
        """Declared observation modality ('marker_3d' or 'markerless_2d')."""

    @property
    @abstractmethod
    def canonical_unit(self) -> str:
        """Canonical physical unit ('m' for 3D markers, 'px' for 2D keypoints)."""

    @property
    @abstractmethod
    def residual_dimension(self) -> int:
        """Scalar dimension of the active fitting residual vector."""

    @property
    @abstractmethod
    def num_active_observations(self) -> int:
        """Number of active points contributing to fitting."""

    @abstractmethod
    def evaluate_raw_residuals(self, q: np.ndarray | DimeCompleteState) -> np.ndarray:
        """Return unwhitened residuals in canonical units for active observations."""

    @abstractmethod
    def evaluate_residuals(self, q: np.ndarray | DimeCompleteState) -> np.ndarray:
        """Return robust whitened residuals for solver optimization."""

    @abstractmethod
    def evaluate_jacobian(
        self,
        q: np.ndarray | DimeCompleteState,
        *,
        wrt: Literal["state", "calibration"] = "state",
        method: Literal["auto", "analytical", "finite"] = "auto",
        step: float = 1.0e-6,
    ) -> np.ndarray:
        """Compute residual Jacobian with respect to candidate state or parameters."""

    @abstractmethod
    def evaluate_held_out(
        self, q: np.ndarray | DimeCompleteState
    ) -> HeldOutEvaluationReport:
        """Evaluate raw and whitened errors specifically on held-out partition."""


# ==============================================================================
# Concrete 3D Marker Observation Factor
# ==============================================================================


class Marker3DObservationFactor(DimeObservationFactor):
    """Robust 3D marker observation factor with metric units and covariance whitening."""

    def __init__(
        self,
        observations_3d_m: np.ndarray,
        kinematics_fn: Callable[[np.ndarray], np.ndarray],
        covariance: float | np.ndarray = 1.0e-4,
        valid_mask: np.ndarray | None = None,
        held_out_mask: np.ndarray | None = None,
        robust_loss: RobustLossKernel | None = None,
        **kwargs: Any,
    ) -> None:
        timing: ObservationTiming | None = kwargs.pop("timing", None)
        attachments: Sequence[MarkerAttachment] | None = kwargs.pop("attachments", None)
        marker_names: Sequence[str] | None = kwargs.pop("marker_names", None)
        detector_scores: np.ndarray | None = kwargs.pop("detector_scores", None)
        kinematics_jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = kwargs.pop(
            "kinematics_jacobian_fn", None
        )
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")
        obs = np.asarray(observations_3d_m, dtype=np.float64)
        require(
            obs.ndim == 2 and obs.shape[1] == 3,
            f"observations_3d_m must have shape (N, 3), got {obs.shape}",
        )
        self._n_points = obs.shape[0]
        self._obs_m = _make_readonly_array(obs)
        self._kinematics_fn = kinematics_fn
        self._kinematics_jacobian_fn = kinematics_jacobian_fn

        # Masking and holdout partition
        (
            self._valid_mask,
            self._held_out_mask,
            self._fit_mask,
            self._active_indices,
            self._held_out_indices,
        ) = _setup_observation_masks(self._n_points, valid_mask, held_out_mask)

        # Covariance whitening
        self._operators, self._is_diagonal = _build_whitening_operators(
            covariance, self._n_points, 3
        )
        self._robust_loss = (
            robust_loss if robust_loss is not None else RobustLossKernel("huber")
        )

        # Detector scores (strictly decoupled from covariance)
        self._detector_scores: np.ndarray | None = _validate_detector_scores(
            self._n_points, detector_scores
        )

        self._timing = timing
        self._attachments = tuple(attachments) if attachments is not None else None
        self._marker_names = (
            tuple(str(m) for m in marker_names) if marker_names is not None else None
        )

    @property
    def factor_type(self) -> ObservationFactorType:
        return "marker_3d"

    @property
    def canonical_unit(self) -> str:
        return "m"

    @property
    def residual_dimension(self) -> int:
        return int(len(self._active_indices) * 3)

    @property
    def num_active_observations(self) -> int:
        return int(len(self._active_indices))

    def _extract_q(self, q: np.ndarray | DimeCompleteState) -> np.ndarray:
        if isinstance(q, DimeCompleteState):
            return q.q
        return np.asarray(q, dtype=np.float64)

    def evaluate_raw_residuals(self, q: np.ndarray | DimeCompleteState) -> np.ndarray:
        if len(self._active_indices) == 0:
            return np.empty(0, dtype=np.float64)
        q_vec = self._extract_q(q)
        pred = np.asarray(self._kinematics_fn(q_vec), dtype=np.float64)
        require(
            pred.shape == (self._n_points, 3),
            f"kinematics_fn must return shape ({self._n_points}, 3), got {pred.shape}",
        )
        diff = pred[self._active_indices] - self._obs_m[self._active_indices]
        return diff.reshape(-1)

    def evaluate_residuals(self, q: np.ndarray | DimeCompleteState) -> np.ndarray:
        if len(self._active_indices) == 0:
            return np.empty(0, dtype=np.float64)
        raw = self.evaluate_raw_residuals(q).reshape(-1, 3)
        whitened = _apply_whitening(
            raw, self._operators, self._is_diagonal, self._active_indices
        )
        return self._robust_loss.apply_robust_weight(whitened, 3)

    def evaluate_jacobian(
        self,
        q: np.ndarray | DimeCompleteState,
        *,
        wrt: Literal["state", "calibration"] = "state",
        method: Literal["auto", "analytical", "finite"] = "auto",
        step: float = 1.0e-6,
    ) -> np.ndarray:
        q_vec = self._extract_q(q)
        if len(self._active_indices) == 0:
            return np.empty((0, len(q_vec)), dtype=np.float64)

        if (
            wrt == "state"
            and method in ("auto", "analytical")
            and self._kinematics_jacobian_fn is not None
        ):
            j_pred = np.asarray(self._kinematics_jacobian_fn(q_vec), dtype=np.float64)
            require(
                j_pred.shape[:2] == (self._n_points, 3),
                "kinematics_jacobian_fn must return shape (N, 3, n_q)",
            )
            j_act = j_pred[self._active_indices]
            raw_diff = self.evaluate_raw_residuals(q_vec).reshape(-1, 3)
            whitened = _apply_whitening(
                raw_diff, self._operators, self._is_diagonal, self._active_indices
            )
            norms = np.linalg.norm(whitened, axis=1)
            sqrt_w = self._robust_loss.sqrt_weight(norms)

            if self._is_diagonal:
                scales = self._operators[self._active_indices]
                j_white = j_act * scales[:, :, None]
            else:
                l_inv = self._operators[self._active_indices]
                j_white = np.einsum("kij,kja->kia", l_inv, j_act)

            j_rob = j_white * sqrt_w[:, None, None]
            return j_rob.reshape(-1, q_vec.size)

        # Central finite difference
        return _finite_difference_jacobian(self.evaluate_residuals, q_vec, step=step)

    def evaluate_held_out(
        self, q: np.ndarray | DimeCompleteState
    ) -> HeldOutEvaluationReport:
        n_held = len(self._held_out_indices)
        if n_held == 0:
            return HeldOutEvaluationReport(
                num_held_out=0,
                rms_error=0.0,
                max_error=0.0,
                canonical_unit="m",
                raw_residuals=np.empty((0, 3), dtype=np.float64),
                whitened_residuals=np.empty((0, 3), dtype=np.float64),
            )
        q_vec = self._extract_q(q)
        pred = np.asarray(self._kinematics_fn(q_vec), dtype=np.float64)
        diff = pred[self._held_out_indices] - self._obs_m[self._held_out_indices]
        return _build_held_out_report(
            diff, self._operators, self._is_diagonal, self._held_out_indices, "m"
        )


# ==============================================================================
# Concrete 2D Markerless Observation Factor
# ==============================================================================


class Markerless2DObservationFactor(DimeObservationFactor):
    """Robust 2D markerless observation factor with pixel units and camera projection."""

    def __init__(
        self,
        observations_2d_px: np.ndarray,
        kinematics_fn: Callable[[np.ndarray], np.ndarray],
        camera: DimeCameraParameters,
        covariance: float | np.ndarray = 1.0,
        valid_mask: np.ndarray | None = None,
        held_out_mask: np.ndarray | None = None,
        robust_loss: RobustLossKernel | None = None,
        **kwargs: Any,
    ) -> None:
        timing: ObservationTiming | None = kwargs.pop("timing", None)
        keypoint_names: Sequence[str] | None = kwargs.pop("keypoint_names", None)
        detector_scores: np.ndarray | None = kwargs.pop("detector_scores", None)
        kinematics_jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = kwargs.pop(
            "kinematics_jacobian_fn", None
        )
        strict_chirality: bool = kwargs.pop("strict_chirality", True)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")
        obs = np.asarray(observations_2d_px, dtype=np.float64)
        require(
            obs.ndim == 2 and obs.shape[1] == 2,
            f"observations_2d_px must have shape (N, 2), got {obs.shape}",
        )
        self._n_points = obs.shape[0]
        self._obs_px = _make_readonly_array(obs)
        self._kinematics_fn = kinematics_fn
        self._kinematics_jacobian_fn = kinematics_jacobian_fn
        self._camera = camera
        self._strict_chirality = strict_chirality

        (
            self._valid_mask,
            self._held_out_mask,
            self._fit_mask,
            self._active_indices,
            self._held_out_indices,
        ) = _setup_observation_masks(self._n_points, valid_mask, held_out_mask)

        # Covariance whitening in pixel units
        self._operators, self._is_diagonal = _build_whitening_operators(
            covariance, self._n_points, 2
        )
        self._robust_loss = (
            robust_loss if robust_loss is not None else RobustLossKernel("huber")
        )

        self._detector_scores: np.ndarray | None = _validate_detector_scores(
            self._n_points, detector_scores
        )

        self._timing = timing
        self._keypoint_names = (
            tuple(str(k) for k in keypoint_names)
            if keypoint_names is not None
            else None
        )

    @property
    def factor_type(self) -> ObservationFactorType:
        return "markerless_2d"

    @property
    def canonical_unit(self) -> str:
        return "px"

    @property
    def residual_dimension(self) -> int:
        return int(len(self._active_indices) * 2)

    @property
    def num_active_observations(self) -> int:
        return int(len(self._active_indices))

    def _extract_q(self, q: np.ndarray | DimeCompleteState) -> np.ndarray:
        if isinstance(q, DimeCompleteState):
            return q.q
        return np.asarray(q, dtype=np.float64)

    def evaluate_raw_residuals(self, q: np.ndarray | DimeCompleteState) -> np.ndarray:
        if len(self._active_indices) == 0:
            return np.empty(0, dtype=np.float64)
        q_vec = self._extract_q(q)
        pts_3d = np.asarray(self._kinematics_fn(q_vec), dtype=np.float64)
        require(
            pts_3d.shape == (self._n_points, 3),
            f"kinematics_fn must return 3D points ({self._n_points}, 3), got {pts_3d.shape}",
        )
        proj_px = self._camera.project(pts_3d, check_chirality=self._strict_chirality)
        diff = proj_px[self._active_indices] - self._obs_px[self._active_indices]
        return diff.reshape(-1)

    def evaluate_residuals(self, q: np.ndarray | DimeCompleteState) -> np.ndarray:
        if len(self._active_indices) == 0:
            return np.empty(0, dtype=np.float64)
        raw = self.evaluate_raw_residuals(q).reshape(-1, 2)
        whitened = _apply_whitening(
            raw, self._operators, self._is_diagonal, self._active_indices
        )
        return self._robust_loss.apply_robust_weight(whitened, 2)

    def evaluate_jacobian(
        self,
        q: np.ndarray | DimeCompleteState,
        *,
        wrt: Literal["state", "calibration"] = "state",
        method: Literal["auto", "analytical", "finite"] = "auto",
        step: float = 1.0e-6,
    ) -> np.ndarray:
        q_vec = self._extract_q(q)
        if len(self._active_indices) == 0:
            return np.empty((0, len(q_vec)), dtype=np.float64)

        # Central finite difference
        return _finite_difference_jacobian(self.evaluate_residuals, q_vec, step=step)

    def evaluate_held_out(
        self, q: np.ndarray | DimeCompleteState
    ) -> HeldOutEvaluationReport:
        n_held = len(self._held_out_indices)
        if n_held == 0:
            return HeldOutEvaluationReport(
                num_held_out=0,
                rms_error=0.0,
                max_error=0.0,
                canonical_unit="px",
                raw_residuals=np.empty((0, 2), dtype=np.float64),
                whitened_residuals=np.empty((0, 2), dtype=np.float64),
            )
        q_vec = self._extract_q(q)
        pts_3d = np.asarray(self._kinematics_fn(q_vec), dtype=np.float64)
        proj_px = self._camera.project(pts_3d, check_chirality=self._strict_chirality)
        diff = proj_px[self._held_out_indices] - self._obs_px[self._held_out_indices]
        return _build_held_out_report(
            diff, self._operators, self._is_diagonal, self._held_out_indices, "px"
        )


__all__ = [
    "ChiralityViolationError",
    "CovarianceValidationError",
    "DimeCameraParameters",
    "DimeObservationFactor",
    "HeldOutEvaluationReport",
    "Marker3DObservationFactor",
    "MarkerAttachment",
    "Markerless2DObservationFactor",
    "ObservationFactorType",
    "ObservationTiming",
    "RobustKernelType",
    "RobustLossKernel",
    "TimingViolationError",
    "invert_camera_extrinsics",
    "quaternion_to_rotation_matrix",
]
