"""Fixed tool-to-model club calibration and grip/face consistency (CO-01 #10605).

Reuses the shared club catalog (:mod:`club_models`) and SO(3)-safe helpers.
Does not replace Simscape MachineLearning CSV calibration tools; those remain
engine-lane utilities. This module is the shared motion-matching facade named
in the CO-01 anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.math_utils.quaternion import rotmat_to_quat
from src.shared.python.motion_matching.club_models import CLUBS, ClubSpec
from src.shared.python.motion_matching.club_only.observation import (
    ClubObservation,
    ComponentStatus,
    DerivationMetadata,
)
from src.shared.python.motion_matching.club_only.workbook_identity import (
    ORIENTATION_AXIS_POLICY,
)

# Relative length tolerance versus catalog club length for rigidity checks.
LENGTH_REL_TOL = 0.08
# Absolute axis-parallelism threshold for derived Z qualification.
_DEGENERATE_CROSS_NORM = 1.0e-6


@dataclass(frozen=True)
class ToolToModelTransform:
    """Fixed SE(3) map from tool/mocap frame into the model frame."""

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    residual_rms_m: float

    def __post_init__(self) -> None:
        r = np.asarray(self.rotation, dtype=np.float64)
        t = np.asarray(self.translation, dtype=np.float64)
        if r.shape != (3, 3):
            raise ValueError(f"rotation must be (3, 3), got {r.shape}")
        if t.shape != (3,):
            raise ValueError(f"translation must be (3,), got {t.shape}")
        det = float(np.linalg.det(r))
        if abs(det - 1.0) > 1.0e-6:
            raise ValueError(f"rotation must be proper SO(3) (det={det})")
        object.__setattr__(self, "rotation", r)
        object.__setattr__(self, "translation", t)

    def inverse(self) -> ToolToModelTransform:
        r_inv = self.rotation.T
        t_inv = -r_inv @ self.translation
        return ToolToModelTransform(
            rotation=r_inv,
            translation=t_inv,
            residual_rms_m=self.residual_rms_m,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "residual_rms_m": self.residual_rms_m,
        }


@dataclass(frozen=True)
class ClubGeometryCalibration:
    """Catalog-backed geometry calibration for one observation."""

    club_spec: ClubSpec
    length_m: float
    tool_to_model: ToolToModelTransform
    measured_median_length_m: float

    @classmethod
    @precondition(
        lambda cls, obs, club_type: isinstance(obs, ClubObservation),
        "obs must be a ClubObservation",
    )
    @postcondition(
        lambda result: isinstance(result, ClubGeometryCalibration),
        "must return ClubGeometryCalibration",
    )
    def from_observation(
        cls, obs: ClubObservation, *, club_type: str
    ) -> ClubGeometryCalibration:
        """Bind an observation to a catalog club; reject length/type mismatch."""
        if club_type not in CLUBS:
            raise ValueError(f"unknown club type {club_type!r}")
        if obs.club_type != club_type:
            raise ValueError(
                f"club type mismatch: observation declares {obs.club_type!r}, "
                f"requested {club_type!r}"
            )
        spec = CLUBS[club_type]
        measured = _median_grip_face_length(obs.mid_hands_xyz, obs.face_xyz)
        if abs(measured - spec.length_m) / spec.length_m > LENGTH_REL_TOL:
            raise ValueError(
                f"measured grip-to-face length {measured:.4f} m disagrees with "
                f"catalog length {spec.length_m:.4f} m for {club_type}"
            )
        identity = ToolToModelTransform(
            rotation=np.eye(3),
            translation=np.zeros(3),
            residual_rms_m=0.0,
        )
        return cls(
            club_spec=spec,
            length_m=spec.length_m,
            tool_to_model=identity,
            measured_median_length_m=measured,
        )


@dataclass(frozen=True)
class GripFaceConsistencyReport:
    """Rigid grip-to-face length consistency versus catalog."""

    median_length_m: float
    catalog_length_m: float
    relative_error: float
    passed: bool
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "median_length_m": self.median_length_m,
            "catalog_length_m": self.catalog_length_m,
            "relative_error": self.relative_error,
            "passed": self.passed,
            "notes": list(self.notes),
        }


def _median_grip_face_length(mid: np.ndarray, face: np.ndarray) -> float:
    a = np.asarray(mid, dtype=np.float64)
    b = np.asarray(face, dtype=np.float64)
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if not np.any(valid):
        raise ValueError("no finite mid-hands/face pairs for length check")
    lengths = np.linalg.norm(b[valid] - a[valid], axis=1)
    return float(np.median(lengths))


@precondition(
    lambda tool_points, model_points: (
        np.asarray(tool_points).shape == np.asarray(model_points).shape
    ),
    "tool and model point clouds must share shape",
)
@postcondition(
    lambda result: isinstance(result, ToolToModelTransform),
    "calibration must return ToolToModelTransform",
)
def calibrate_tool_to_model(
    tool_points: np.ndarray, model_points: np.ndarray
) -> ToolToModelTransform:
    """Umeyama/Kabsch rigid fit of tool points into the model frame."""
    tool = np.asarray(tool_points, dtype=np.float64)
    model = np.asarray(model_points, dtype=np.float64)
    if tool.ndim != 2 or tool.shape[1] != 3 or tool.shape[0] < 3:
        raise ValueError("need at least 3 corresponding (N, 3) points")
    if not (np.isfinite(tool).all() and np.isfinite(model).all()):
        raise ValueError("tool/model points must be finite")
    mu_t = tool.mean(axis=0)
    mu_m = model.mean(axis=0)
    x = tool - mu_t
    y = model - mu_m
    cov = x.T @ y / float(tool.shape[0])
    u, _, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt = vt.copy()
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = mu_m - r @ mu_t
    fitted = (tool @ r.T) + t
    rms = float(np.sqrt(np.mean(np.sum((fitted - model) ** 2, axis=1))))
    # Touch rotmat_to_quat so improper matrices fail closed in downstream use.
    _ = rotmat_to_quat(r)
    return ToolToModelTransform(rotation=r, translation=t, residual_rms_m=rms)


@precondition(
    lambda transform, points: isinstance(transform, ToolToModelTransform),
    "transform must be ToolToModelTransform",
)
@postcondition(
    lambda result: isinstance(result, np.ndarray) and result.ndim == 2,
    "transformed points must be (N, 3)",
)
def transform_points(transform: ToolToModelTransform, points: np.ndarray) -> np.ndarray:
    """Apply a tool-to-model SE(3) transform to an ``(N, 3)`` point set."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    return (pts @ transform.rotation.T) + transform.translation


@precondition(
    lambda mid_hands_xyz, face_xyz, offset_m: offset_m is not None,
    "butt-end requires an explicit offset from mid-hands (metres)",
)
@postcondition(
    lambda result: isinstance(result, np.ndarray) and result.shape[-1] == 3,
    "butt-end positions must be (N, 3)",
)
def mid_hands_to_butt_end(
    mid_hands_xyz: np.ndarray,
    face_xyz: np.ndarray,
    offset_m: float | None,
) -> np.ndarray:
    """Map mid-hands to butt-end along the grip-to-face shaft axis.

    Refuses to treat mid-hands as butt-end without an explicit offset.
    """
    if offset_m is None:
        raise ValueError(
            "refusing to rename mid-hands as butt-end without an offset transform"
        )
    if not np.isfinite(offset_m):
        raise ValueError("offset_m must be finite")
    mid = np.asarray(mid_hands_xyz, dtype=np.float64)
    face = np.asarray(face_xyz, dtype=np.float64)
    if mid.shape != face.shape or mid.ndim != 2 or mid.shape[1] != 3:
        raise ValueError("mid_hands_xyz and face_xyz must share shape (N, 3)")
    shaft = face - mid
    norms = np.linalg.norm(shaft, axis=1, keepdims=True)
    if np.any(norms < 1.0e-9):
        raise ValueError("degenerate grip-to-face axis for butt-end offset")
    direction = shaft / norms
    # Butt-end lies opposite the face along the shaft from mid-hands.
    return mid - direction * float(offset_m)


@precondition(
    lambda x_axis, y_axis, degenerate: np.asarray(x_axis).shape == (3,),
    "x_axis must be length-3",
)
@postcondition(
    lambda result: isinstance(result, DerivationMetadata),
    "must return DerivationMetadata",
)
def qualify_derived_orientation_axes(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    degenerate: bool,
) -> DerivationMetadata:
    """Qualify a derived third axis; require an explicit degeneracy flag."""
    x = np.asarray(x_axis, dtype=np.float64).reshape(3)
    y = np.asarray(y_axis, dtype=np.float64).reshape(3)
    cross = np.linalg.norm(np.cross(x, y))
    is_degenerate = bool(cross < _DEGENERATE_CROSS_NORM)
    if is_degenerate and not degenerate:
        raise ValueError(
            "degenerate orientation axes require degeneracy flag "
            f"(policy requires {ORIENTATION_AXIS_POLICY.degeneracy_flag_required})"
        )
    notes: list[str] = []
    if (not is_degenerate) and degenerate:
        # Caller may mark conservatively; still accept with note.
        notes.append("degeneracy flag set without geometric degeneracy")
    return DerivationMetadata(
        orientation_axis_status=ORIENTATION_AXIS_POLICY.status_when_derived,
        degenerate_axes=bool(degenerate or is_degenerate),
        notes=tuple(notes),
    )


@precondition(
    lambda obs: isinstance(obs, ClubObservation),
    "obs must be a ClubObservation",
)
@postcondition(
    lambda result: isinstance(result, GripFaceConsistencyReport),
    "must return GripFaceConsistencyReport",
)
def cross_check_grip_face_consistency(
    obs: ClubObservation,
) -> GripFaceConsistencyReport:
    """Cross-check rigid grip-to-face length against the catalog club."""
    if (
        obs.mask.mid_hands_position is ComponentStatus.UNOBSERVED
        or obs.mask.face_position is ComponentStatus.UNOBSERVED
    ):
        raise ValueError("grip/face consistency requires measured positions")
    measured = _median_grip_face_length(obs.mid_hands_xyz, obs.face_xyz)
    catalog = float(obs.catalog_length_m)
    rel = abs(measured - catalog) / catalog
    passed = rel <= LENGTH_REL_TOL
    notes: list[str] = []
    if not passed:
        notes.append(
            f"median length {measured:.4f} m exceeds {LENGTH_REL_TOL:.0%} of "
            f"catalog {catalog:.4f} m"
        )
    return GripFaceConsistencyReport(
        median_length_m=measured,
        catalog_length_m=catalog,
        relative_error=rel,
        passed=passed,
        notes=tuple(notes),
    )
