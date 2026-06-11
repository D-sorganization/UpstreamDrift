"""Motion Matching API Contracts: Canonical validation across Drake/OpenSim/MuJoCo/Pinocchio.

This module defines Design-by-Contract (DbC) specifications for the motion
matching pipeline's core data exchange types. It enforces preconditions,
postconditions, and invariants to ensure type safety and numerical correctness
across all physics engine backends.

Canonical Formats:
    theta (Joint Configuration):
        - Drake: (n_dof,) float64 array. Includes free-flyer (6 DOF) + joint angles.
        - OpenSim: (n_coords,) float64 array. Generalized coordinates.
        - MuJoCo: (n_dof,) float32 array. Normalized to [-pi, pi] for revolutes.
        - Pinocchio: (n_q,) float64 array. Configuration vector.
        Canonical: (n_dof,) float64 or float32, finite, shape consistency.

    initial_pose:
        - Root frame position (3D): x, y, z in meters.
        - Root frame orientation: (4,) quaternion [w, x, y, z] unit-norm.
        - Joint angles: remaining DOFs as theta[7:] (after free-flyer).
        - Canonical: Bundle with position, quaternion, and joint angles.

    FitResult:
        - coefficients: (n_coeffs,) polynomial or control coefficients.
        - final_loss: scalar float >= 0 (optimization loss at convergence).
        - metadata: engine, time-to-fit, gradient-norm-at-convergence.

Public API:
    ThetaContractValidator   -- Validates joint configuration vectors.
    InitialPoseValidator     -- Validates root frame + joints.
    FitResult                -- Frozen dataclass for optimization results.
    validate_theta_contract  -- Entry point for theta validation.
    validate_initial_pose    -- Entry point for initial pose validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Final, NamedTuple

import numpy as np

from src.shared.python.contracts import (
    postcondition,
    precondition,
)

__all__ = [
    "ApiFitResult",
    "FitResult",
    "InitialPose",
    "ThetaContractValidator",
    "InitialPoseValidator",
    "validate_initial_pose",
    "validate_theta_contract",
]

logger = logging.getLogger(__name__)

# Validation tolerances
QUAT_NORM_TOL: Final[float] = 1.0e-6
THETA_FINITE_TOL: Final[float] = 1.0e-10
MAX_THETA_VALUE: Final[float] = 1.0e3
MIN_THETA_VALUE: Final[float] = -1.0e3
POSITION_MAX_NORM: Final[float] = 50.0
POSITION_MIN_COMPONENT: Final[float] = -50.0
POSITION_MAX_COMPONENT: Final[float] = 50.0

# Engine-specific DOF counts
ENGINE_DOF_MAP: Final[dict[str, int]] = {
    "drake": 23,  # 6 (free-flyer) + 17 (joints)
    "opensim": 23,  # varies; canonical is 23
    "mujoco": 17,  # 0 (fixed root) + 17 (joints)
    "pinocchio": 23,  # 6 (free-flyer) + 17 (joints)
}


class InitialPose(NamedTuple):
    """Canonical initial pose bundle: root position, orientation, and joint angles.

    Attributes:
        root_position: (3,) position [x, y, z] in meters.
        root_quat: (4,) unit quaternion [w, x, y, z].
        joint_angles: (n_remaining,) remaining DOF values.
    """

    root_position: np.ndarray
    root_quat: np.ndarray
    joint_angles: np.ndarray


@dataclass(frozen=True)
class ApiFitResult:
    """Standardized return type for fit_swing_* optimizers across all engines.

    Serves as the canonical bundle for swing trajectory optimization results,
    ensuring type safety and numerical validation across Drake, OpenSim,
    MuJoCo, and Pinocchio backends.

    Design by Contract:
        Preconditions:
            - coefficients: (n_coeffs,) float array, all finite.
            - final_loss: scalar >= 0, finite.
            - metadata: dict with required keys ('engine', 'time_s').
        Postconditions:
            - self.coefficients is a numpy array (not a list or tensor).
            - self.final_loss >= 0 and is finite.
            - coefficients.dtype in (float32, float64).

    Attributes:
        coefficients: (n_coeffs,) control/polynomial coefficients.
        final_loss: Scalar loss value at convergence.
        metadata: Dict with engine, time_s, and optional grad_norm.
        trajectory: Optional predicted/final trajectory (engine-specific).
    """

    coefficients: np.ndarray
    final_loss: float
    metadata: dict[str, Any]
    trajectory: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate FitResult fields at construction."""
        self._validate_coefficients()
        self._validate_final_loss()
        self._validate_metadata()

    def _validate_coefficients(self) -> None:
        """Ensure coefficients is a finite numpy array."""
        if not isinstance(self.coefficients, np.ndarray):
            raise TypeError(
                f"coefficients must be np.ndarray, got {type(self.coefficients)}"
            )
        if self.coefficients.ndim != 1:
            raise ValueError(
                f"coefficients must be 1-D, got shape {self.coefficients.shape}"
            )
        if not np.all(np.isfinite(self.coefficients)):
            raise ValueError("coefficients contains NaN or Inf")
        if self.coefficients.dtype not in (np.float32, np.float64):
            raise TypeError(
                f"coefficients.dtype must be float32 or float64, "
                f"got {self.coefficients.dtype}"
            )

    def _validate_final_loss(self) -> None:
        """Ensure final_loss is a non-negative finite scalar."""
        if not isinstance(self.final_loss, (float, np.floating)):
            raise TypeError(f"final_loss must be float, got {type(self.final_loss)}")
        if not np.isfinite(float(self.final_loss)):
            raise ValueError(f"final_loss must be finite, got {self.final_loss}")
        if float(self.final_loss) < 0:
            raise ValueError(f"final_loss must be >= 0, got {self.final_loss}")

    def _validate_metadata(self) -> None:
        """Ensure metadata dict has required keys."""
        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be dict, got {type(self.metadata)}")
        required_keys = {"engine", "time_s"}
        missing = required_keys - set(self.metadata.keys())
        if missing:
            raise ValueError(f"metadata missing required keys: {missing}")
        if self.metadata["engine"] not in ENGINE_DOF_MAP:
            raise ValueError(
                f"metadata['engine'] must be one of {list(ENGINE_DOF_MAP.keys())}, "
                f"got {self.metadata['engine']!r}"
            )
        if not isinstance(self.metadata["time_s"], (float, int, np.floating)):
            raise TypeError(
                f"metadata['time_s'] must be numeric, "
                f"got {type(self.metadata['time_s'])}"
            )
        if float(self.metadata["time_s"]) < 0:
            raise ValueError(
                f"metadata['time_s'] must be >= 0, got {self.metadata['time_s']}"
            )


FitResult = ApiFitResult


class ThetaContractValidator:
    """Validates joint configuration (theta) vectors across all physics engines.

    Enforces canonical theta format: (n_dof,) finite array, values within
    physically plausible ranges, and consistency with engine expectations.

    Design by Contract:
        Invariants:
            - n_dof in {17, 23} for golf swing models.
            - theta shape is exactly (n_dof,).
            - All values are finite.
            - All values in [-1e3, 1e3] radians.
    """

    def __init__(self, engine: str, n_dof: int | None = None) -> None:
        """Initialize validator for a specific engine.

        Args:
            engine: Engine name ('drake', 'opensim', 'mujoco', 'pinocchio').
            n_dof: Expected DOF count. If None, inferred from ENGINE_DOF_MAP.

        Raises:
            ValueError: If engine is unknown or n_dof is inconsistent.
        """
        if engine not in ENGINE_DOF_MAP:
            raise ValueError(
                f"engine must be one of {list(ENGINE_DOF_MAP.keys())}, got {engine!r}"
            )
        self.engine = engine
        self.n_dof_canonical = ENGINE_DOF_MAP[engine]

        if n_dof is None:
            self.n_dof = self.n_dof_canonical
        else:
            if not isinstance(n_dof, int) or n_dof <= 0:
                raise ValueError(f"n_dof must be a positive int, got {n_dof}")
            self.n_dof = n_dof

    @precondition(
        lambda self, theta: isinstance(theta, np.ndarray),
        "theta must be a numpy array",
    )
    @postcondition(
        lambda result: result is None or isinstance(result, str),
        "return value must be None (valid) or error string",
    )
    def validate(self, theta: np.ndarray) -> str | None:
        """Check theta for conformance to the canonical format.

        Args:
            theta: Proposed joint configuration array.

        Returns:
            None if valid, otherwise a descriptive error message.
        """
        if theta.ndim != 1:
            return f"theta must be 1-D, got shape {theta.shape}"

        if theta.shape[0] != self.n_dof:
            return f"theta length mismatch: expected {self.n_dof}, got {theta.shape[0]}"

        if not np.all(np.isfinite(theta)):
            bad_mask = ~np.isfinite(theta)
            bad_indices = np.where(bad_mask)[0]
            return (
                f"theta contains {bad_indices.size} non-finite "
                f"values at indices {bad_indices[:5].tolist()}..."
            )

        if np.any(theta < MIN_THETA_VALUE) or np.any(theta > MAX_THETA_VALUE):
            min_v, max_v = float(theta.min()), float(theta.max())
            return (
                f"theta values out of range [{MIN_THETA_VALUE}, "
                f"{MAX_THETA_VALUE}], got min={min_v:.2e}, max={max_v:.2e}"
            )

        return None

    def validate_raise(self, theta: np.ndarray) -> None:
        """Check theta and raise ValueError if invalid (DbC style).

        Args:
            theta: Joint configuration array.

        Raises:
            ValueError: If any validation check fails.
        """
        error = self.validate(theta)
        if error:
            raise ValueError(error)


class InitialPoseValidator:
    """Validates initial pose bundles (root frame + joint angles).

    Enforces canonical format:
        - root_position: (3,) position in meters.
        - root_quat: (4,) unit quaternion [w, x, y, z].
        - joint_angles: (n_remaining,) DOF values.

    Design by Contract:
        Invariants:
            - position norm <= POSITION_MAX_NORM.
            - position components in [-50, 50] meters.
            - quaternion unit-norm to within QUAT_NORM_TOL.
            - joint_angles all finite.
    """

    def __init__(self, engine: str, n_dof: int | None = None) -> None:
        """Initialize pose validator for a specific engine.

        Args:
            engine: Engine name.
            n_dof: Total DOF (including free-flyer if present).
        """
        if engine not in ENGINE_DOF_MAP:
            raise ValueError(
                f"engine must be one of {list(ENGINE_DOF_MAP.keys())}, got {engine!r}"
            )
        self.engine = engine
        self.n_dof_total = n_dof or ENGINE_DOF_MAP[engine]
        self.n_joints = self.n_dof_total - 6  # free-flyer is 6 DOF

    @precondition(
        lambda self, pose: isinstance(pose, InitialPose),
        "pose must be InitialPose",
    )
    @postcondition(
        lambda result: result is None or isinstance(result, str),
        "return value must be None (valid) or error string",
    )
    def validate(self, pose: InitialPose) -> str | None:
        """Check pose for conformance to canonical format.

        Args:
            pose: InitialPose namedtuple.

        Returns:
            None if valid, otherwise a descriptive error message.
        """
        # Validate root_position
        error = self._validate_position(pose.root_position)
        if error:
            return f"root_position: {error}"

        # Validate root_quat
        error = self._validate_quaternion(pose.root_quat)
        if error:
            return f"root_quat: {error}"

        # Validate joint_angles
        expected_joints = self.n_joints
        if pose.joint_angles.shape[0] != expected_joints:
            return (
                f"joint_angles: expected {expected_joints} values, "
                f"got {pose.joint_angles.shape[0]}"
            )

        if not np.all(np.isfinite(pose.joint_angles)):
            bad_idx = np.where(~np.isfinite(pose.joint_angles))[0]
            return (
                f"joint_angles: contains NaN/Inf at indices {bad_idx[:3].tolist()}..."
            )

        return None

    def _validate_position(self, pos: np.ndarray) -> str | None:
        """Check position vector (3,) for plausibility."""
        if not isinstance(pos, np.ndarray):
            return f"must be np.ndarray, got {type(pos)}"
        if pos.shape != (3,):
            return f"must have shape (3,), got {pos.shape}"
        if not np.all(np.isfinite(pos)):
            return "contains NaN or Inf"
        if np.linalg.norm(pos) > POSITION_MAX_NORM:
            return f"norm {float(np.linalg.norm(pos)):.2f} exceeds {POSITION_MAX_NORM}"
        if np.any(pos < POSITION_MIN_COMPONENT) or np.any(pos > POSITION_MAX_COMPONENT):
            return f"components out of [{POSITION_MIN_COMPONENT}, {POSITION_MAX_COMPONENT}]"
        return None

    def _validate_quaternion(self, quat: np.ndarray) -> str | None:
        """Check quaternion (4,) for unit-norm validity."""
        if not isinstance(quat, np.ndarray):
            return f"must be np.ndarray, got {type(quat)}"
        if quat.shape != (4,):
            return f"must have shape (4,), got {quat.shape}"
        if not np.all(np.isfinite(quat)):
            return "contains NaN or Inf"
        norm = float(np.linalg.norm(quat))
        if abs(norm - 1.0) > QUAT_NORM_TOL:
            return f"norm is {norm:.6f}, not 1.0 (tolerance {QUAT_NORM_TOL})"
        return None

    def validate_raise(self, pose: InitialPose) -> None:
        """Check pose and raise ValueError if invalid (DbC style).

        Args:
            pose: InitialPose bundle.

        Raises:
            ValueError: If any validation check fails.
        """
        error = self.validate(pose)
        if error:
            raise ValueError(error)


# ============================================================================
# Module-level entry points
# ============================================================================


@precondition(
    lambda theta, engine: isinstance(theta, np.ndarray) and engine in ENGINE_DOF_MAP,
    "theta must be np.ndarray and engine must be known",
)
@postcondition(
    lambda result: result is None or isinstance(result, str),
    "result must be None or error message string",
)
def validate_theta_contract(
    theta: np.ndarray,
    engine: str,
    n_dof: int | None = None,
) -> str | None:
    """Validate joint configuration for a specific physics engine.

    Entry point for canonical theta validation across Drake, OpenSim,
    MuJoCo, and Pinocchio.

    Args:
        theta: Joint configuration array.
        engine: One of 'drake', 'opensim', 'mujoco', 'pinocchio'.
        n_dof: Expected DOF count (inferred if not provided).

    Returns:
        None if valid; error message string if invalid.

    Examples:
        >>> theta = np.array([0.1, 0.2, ...])  # 23 values
        >>> error = validate_theta_contract(theta, "drake")
        >>> if error:
        ...     print(f"Validation failed: {error}")
    """
    try:
        validator = ThetaContractValidator(engine, n_dof)
        return validator.validate(theta)
    except ValueError as e:
        return str(e)


@precondition(
    lambda pose, engine: isinstance(pose, InitialPose) and engine in ENGINE_DOF_MAP,
    "pose must be InitialPose and engine must be known",
)
@postcondition(
    lambda result: result is None or isinstance(result, str),
    "result must be None or error message string",
)
def validate_initial_pose(
    pose: InitialPose,
    engine: str,
    n_dof: int | None = None,
) -> str | None:
    """Validate initial pose bundle for a specific physics engine.

    Entry point for canonical initial pose validation.

    Args:
        pose: InitialPose (root_position, root_quat, joint_angles).
        engine: One of 'drake', 'opensim', 'mujoco', 'pinocchio'.
        n_dof: Total DOF count (inferred if not provided).

    Returns:
        None if valid; error message string if invalid.

    Examples:
        >>> pose = InitialPose(
        ...     root_position=np.array([0, 0, 0]),
        ...     root_quat=np.array([1, 0, 0, 0]),
        ...     joint_angles=np.zeros(17),
        ... )
        >>> error = validate_initial_pose(pose, "drake")
    """
    try:
        validator = InitialPoseValidator(engine, n_dof)
        return validator.validate(pose)
    except ValueError as e:
        return str(e)
