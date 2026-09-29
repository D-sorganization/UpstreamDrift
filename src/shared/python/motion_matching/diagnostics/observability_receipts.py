"""Head, trunk, and grip observability diagnostic receipts (MMR-06-I, #11106).

Provides:
1. AttachmentTransform: rigid SE(3) transformation with explicit physical SI units,
   invertibility, and SHA-256 provenance hashes.
2. HeadDiagnosticCalculator: evaluates 3D marker residuals, head-centre displacement,
   and SO(3) orientation residuals with explicit attachment-transform compensation.
3. TrunkDiagnosticCalculator: models trunk centre-of-mass observables, distinguishing
   between joint-centre and C7-proxy reference frames.
4. GripAndClubfaceCalibration: subject-calibrated hand grip and clubface transforms.
5. ObservabilityDiagnosticReceipt: immutable receipt documenting diagnostics across
   head, trunk, and grip.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, postcondition, precondition, require

logger = logging.getLogger(__name__)

OBSERVABILITY_RECEIPT_SCHEMA = "observability-diagnostics/1.0.0"

Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AttachmentTransform:
    """Rigid SE(3) spatial transform between an anatomical body and an attachment/marker frame."""

    from_frame: str
    to_frame: str
    translation_m: tuple[float, float, float]
    rotation_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    def __post_init__(self) -> None:
        require(bool(self.from_frame.strip()), "from_frame required", self.from_frame)
        require(bool(self.to_frame.strip()), "to_frame required", self.to_frame)
        require(len(self.translation_m) == 3, "translation_m must have 3 elements")
        require(
            all(np.isfinite(self.translation_m)),
            "translation_m must have finite values",
        )
        r = np.asarray(self.rotation_matrix, dtype=np.float64)
        require(r.shape == (3, 3), "rotation_matrix must be (3, 3)")
        require(bool(np.all(np.isfinite(r))), "rotation_matrix must have finite values")
        # Check orthogonality: R @ R.T == I and det(R) == 1
        require(
            bool(np.allclose(r @ r.T, np.eye(3), atol=1e-5)),
            "rotation_matrix must be orthogonal",
        )
        require(
            bool(np.isclose(np.linalg.det(r), 1.0, atol=1e-4)),
            "rotation_matrix determinant must be +1",
        )

    @classmethod
    def from_matrix_and_translation(
        cls,
        *,
        from_frame: str,
        to_frame: str,
        rotation: Array | Sequence[Sequence[float]],
        translation: Array | Sequence[float],
    ) -> AttachmentTransform:
        r_arr = np.asarray(rotation, dtype=np.float64)
        t_arr = np.asarray(translation, dtype=np.float64)
        require(r_arr.shape == (3, 3), "rotation shape must be (3, 3)")
        require(t_arr.shape == (3,), "translation shape must be (3,)")
        rot_tuple = (
            (float(r_arr[0, 0]), float(r_arr[0, 1]), float(r_arr[0, 2])),
            (float(r_arr[1, 0]), float(r_arr[1, 1]), float(r_arr[1, 2])),
            (float(r_arr[2, 0]), float(r_arr[2, 1]), float(r_arr[2, 2])),
        )
        t_tuple = (float(t_arr[0]), float(t_arr[1]), float(t_arr[2]))
        return cls(
            from_frame=from_frame,
            to_frame=to_frame,
            translation_m=t_tuple,
            rotation_matrix=rot_tuple,
        )

    @property
    def transform_sha256(self) -> str:
        payload = {
            "from_frame": self.from_frame,
            "to_frame": self.to_frame,
            "translation_m": list(self.translation_m),
            "rotation_matrix": [list(row) for row in self.rotation_matrix],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def apply(self, point_3d: Array | Sequence[float]) -> Array:
        """Apply transform to a 3D point in the source frame: p_dest = R @ p_src + t."""
        p = np.asarray(point_3d, dtype=np.float64)
        require(p.shape == (3,), "point_3d must have shape (3,)")
        r = np.asarray(self.rotation_matrix, dtype=np.float64)
        t = np.asarray(self.translation_m, dtype=np.float64)
        return r @ p + t

    def inverse(self) -> AttachmentTransform:
        """Return the analytical inverse rigid transformation."""
        r = np.asarray(self.rotation_matrix, dtype=np.float64)
        t = np.asarray(self.translation_m, dtype=np.float64)
        r_inv = r.T
        t_inv = -r_inv @ t
        return AttachmentTransform.from_matrix_and_translation(
            from_frame=self.to_frame,
            to_frame=self.from_frame,
            rotation=r_inv,
            translation=t_inv,
        )

    def roundtrip_identity(
        self, point_3d: Array | Sequence[float], tol_m: float = 1e-11
    ) -> bool:
        """Verify T^-1(T(p)) recovers p within numerical tolerance."""
        p = np.asarray(point_3d, dtype=np.float64)
        p_recovered = self.inverse().apply(self.apply(p))
        return bool(np.allclose(p, p_recovered, atol=tol_m))

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_frame": self.from_frame,
            "to_frame": self.to_frame,
            "translation_m": list(self.translation_m),
            "rotation_matrix": [list(row) for row in self.rotation_matrix],
            "transform_sha256": self.transform_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AttachmentTransform:
        rot_data = data["rotation_matrix"]
        rot = (
            (float(rot_data[0][0]), float(rot_data[0][1]), float(rot_data[0][2])),
            (float(rot_data[1][0]), float(rot_data[1][1]), float(rot_data[1][2])),
            (float(rot_data[2][0]), float(rot_data[2][1]), float(rot_data[2][2])),
        )
        trans = (
            float(data["translation_m"][0]),
            float(data["translation_m"][1]),
            float(data["translation_m"][2]),
        )
        return cls(
            from_frame=str(data["from_frame"]),
            to_frame=str(data["to_frame"]),
            translation_m=trans,
            rotation_matrix=rot,
        )


@dataclass(frozen=True, slots=True)
class GripAndClubfaceCalibration:
    """Subject-calibrated hand grip and clubface transformations."""

    lead_hand_frame: str
    trail_hand_frame: str
    grip_frame: str
    face_frame: str
    lead_grip_transform: AttachmentTransform
    trail_grip_transform: AttachmentTransform
    clubface_transform: AttachmentTransform
    convention: str = "lead_left_trail_right"

    def __post_init__(self) -> None:
        # Declared frame endpoints each transform must connect. The grip is calibrated
        # from each hand; the face is calibrated from the club grip reference frame.
        expected_endpoints: tuple[tuple[str, str, str, AttachmentTransform], ...] = (
            (
                "lead_grip_transform",
                self.lead_hand_frame,
                self.grip_frame,
                self.lead_grip_transform,
            ),
            (
                "trail_grip_transform",
                self.trail_hand_frame,
                self.grip_frame,
                self.trail_grip_transform,
            ),
            (
                "clubface_transform",
                self.grip_frame,
                self.face_frame,
                self.clubface_transform,
            ),
        )
        for name, from_frame, to_frame, transform in expected_endpoints:
            if transform.from_frame != from_frame or transform.to_frame != to_frame:
                raise ValueError(
                    f"{name} must connect {from_frame!r} -> {to_frame!r}; "
                    f"got {transform.from_frame!r} -> {transform.to_frame!r}"
                )

    @property
    def calibration_sha256(self) -> str:
        payload = {
            "lead_hand_frame": self.lead_hand_frame,
            "trail_hand_frame": self.trail_hand_frame,
            "grip_frame": self.grip_frame,
            "face_frame": self.face_frame,
            "lead_grip_sha256": self.lead_grip_transform.transform_sha256,
            "trail_grip_sha256": self.trail_grip_transform.transform_sha256,
            "clubface_sha256": self.clubface_transform.transform_sha256,
            "convention": self.convention,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "lead_hand_frame": self.lead_hand_frame,
            "trail_hand_frame": self.trail_hand_frame,
            "grip_frame": self.grip_frame,
            "face_frame": self.face_frame,
            "lead_grip_transform": self.lead_grip_transform.as_dict(),
            "trail_grip_transform": self.trail_grip_transform.as_dict(),
            "clubface_transform": self.clubface_transform.as_dict(),
            "convention": self.convention,
            "calibration_sha256": self.calibration_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GripAndClubfaceCalibration:
        return cls(
            lead_hand_frame=str(data["lead_hand_frame"]),
            trail_hand_frame=str(data["trail_hand_frame"]),
            grip_frame=str(data["grip_frame"]),
            face_frame=str(data["face_frame"]),
            lead_grip_transform=AttachmentTransform.from_dict(
                data["lead_grip_transform"]
            ),
            trail_grip_transform=AttachmentTransform.from_dict(
                data["trail_grip_transform"]
            ),
            clubface_transform=AttachmentTransform.from_dict(
                data["clubface_transform"]
            ),
            convention=str(data.get("convention", "lead_left_trail_right")),
        )


class TrunkObservableKind(str, Enum):
    """Trunk centre of mass observable frame definitions."""

    JOINT_CENTRE = "joint_centre"
    C7_PROXY = "c7_proxy"


@dataclass(frozen=True, slots=True)
class HeadDiagnosticResult:
    """Outcome of head observability evaluation."""

    head_centre_error_m: float
    orientation_error_deg: float
    orientation_error_rad: float
    marker_residuals_m: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "head_centre_error_m": float(self.head_centre_error_m),
            "orientation_error_deg": float(self.orientation_error_deg),
            "orientation_error_rad": float(self.orientation_error_rad),
            "marker_residuals_m": {
                k: float(v) for k, v in self.marker_residuals_m.items()
            },
        }


DEFAULT_HEAD_NOMINAL_OFFSETS: dict[str, tuple[float, float, float]] = {
    "HeadFront": (0.10, 0.0, 0.0),
    "HeadTop": (0.0, 0.0, 0.12),
    "HeadSide": (0.0, 0.09, 0.0),
}


class HeadDiagnosticCalculator:
    """Evaluates head marker residuals, centre displacement, and SO(3) orientation residuals."""

    def __init__(
        self,
        attachment_transform: AttachmentTransform | None = None,
        nominal_marker_offsets: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self.attachment_transform = attachment_transform
        if nominal_marker_offsets is not None:
            self.nominal_marker_offsets: dict[str, tuple[float, float, float]] = {
                k: (float(v[0]), float(v[1]), float(v[2]))
                for k, v in nominal_marker_offsets.items()
            }
        else:
            self.nominal_marker_offsets = dict(DEFAULT_HEAD_NOMINAL_OFFSETS)

    def evaluate(
        self,
        *,
        model_head_centre: Array | Sequence[float],
        model_head_rotation: Array | Sequence[Sequence[float]],
        observed_markers: Mapping[str, Array | Sequence[float]],
    ) -> HeadDiagnosticResult:
        """Evaluate head observability residuals.

        Fails closed if attempting to compare marker centroid to body centre without an
        explicit attachment transform.
        """
        if self.attachment_transform is None:
            raise ValueError(
                "Cannot compare marker centroid to anatomical body centre without explicit attachment transform."
            )

        c_model = np.asarray(model_head_centre, dtype=np.float64)
        r_model = np.asarray(model_head_rotation, dtype=np.float64)
        require(c_model.shape == (3,), "model_head_centre must be a 3-vector")
        require(r_model.shape == (3, 3), "model_head_rotation must be (3, 3)")

        required_markers = set(self.nominal_marker_offsets.keys())
        missing = required_markers - set(observed_markers.keys())
        if missing:
            raise ValueError(f"Missing required head markers: {sorted(missing)}")

        marker_names = sorted(required_markers)
        nom_pts = np.array(
            [self.nominal_marker_offsets[k] for k in marker_names], dtype=np.float64
        )
        obs_pts = np.array(
            [observed_markers[k] for k in marker_names], dtype=np.float64
        )

        # Centroids
        nom_centroid = np.mean(nom_pts, axis=0)
        obs_centroid = np.mean(obs_pts, axis=0)

        # Centered points for Kabsch algorithm
        p_centered = nom_pts - nom_centroid
        q_centered = obs_pts - obs_centroid

        # Cross-covariance matrix H = P^T @ Q
        h_cov = p_centered.T @ q_centered
        u, _, vt = np.linalg.svd(h_cov)
        d = np.linalg.det(vt.T @ u.T)
        v_diag = np.eye(3)
        if d < 0:
            v_diag[2, 2] = -1.0
        r_obs = vt.T @ v_diag @ u.T

        # Estimated head centre using attachment transform
        # The attachment transform translation gives the nominal marker centroid offset
        t_offset = np.asarray(self.attachment_transform.translation_m, dtype=np.float64)
        est_head_centre = obs_centroid - r_obs @ t_offset
        centre_error = float(np.linalg.norm(est_head_centre - c_model))

        # SO(3) orientation residual between the model anatomical orientation and the
        # observed anatomical orientation. Kabsch's r_obs maps marker-frame nominal
        # geometry into the observed marker frame; compose it with the inverse of the
        # calibrated attachment rotation (declared anatomical -> marker frame) so the
        # residual compares anatomical frames on both sides.
        r_attach = np.asarray(
            self.attachment_transform.rotation_matrix, dtype=np.float64
        )
        r_obs_anatomical = r_obs @ r_attach.T
        r_err = r_model.T @ r_obs_anatomical
        trace_val = float(np.trace(r_err))
        clamped_cos = float(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0))
        angle_rad = float(np.arccos(clamped_cos))
        angle_deg = float(np.degrees(angle_rad))

        # Expected marker positions given model state: c_model + r_model @ nom_pts[i]
        residuals: dict[str, float] = {}
        for k in marker_names:
            expected_pos = c_model + r_model @ np.asarray(
                self.nominal_marker_offsets[k], dtype=np.float64
            )
            residuals[k] = float(
                np.linalg.norm(
                    np.asarray(observed_markers[k], dtype=np.float64) - expected_pos
                )
            )

        return HeadDiagnosticResult(
            head_centre_error_m=centre_error,
            orientation_error_deg=angle_deg,
            orientation_error_rad=angle_rad,
            marker_residuals_m=residuals,
        )


class TrunkDiagnosticCalculator:
    """Evaluates trunk centre of mass observable fidelity."""

    def __init__(
        self,
        observable_kind: TrunkObservableKind = TrunkObservableKind.JOINT_CENTRE,
        c7_nominal_offset_m: tuple[float, float, float] = (-0.1, 0.0, 0.15),
    ) -> None:
        self.observable_kind = observable_kind
        self.c7_nominal_offset_m = c7_nominal_offset_m

    def compute_residual(
        self,
        *,
        model_trunk_centre: Array | Sequence[float],
        observed_point: Array | Sequence[float],
        trunk_rotation: Array | Sequence[Sequence[float]] | None = None,
    ) -> float:
        """Compute trunk observable residual in metres."""
        m_centre = np.asarray(model_trunk_centre, dtype=np.float64)
        obs = np.asarray(observed_point, dtype=np.float64)
        require(m_centre.shape == (3,), "model_trunk_centre shape must be (3,)")
        require(obs.shape == (3,), "observed_point shape must be (3,)")

        if self.observable_kind == TrunkObservableKind.JOINT_CENTRE:
            return float(np.linalg.norm(obs - m_centre))

        # C7 proxy: if rotation is supplied, compensate for posterior offset; otherwise compute raw shift
        offset = np.asarray(self.c7_nominal_offset_m, dtype=np.float64)
        if trunk_rotation is not None:
            r = np.asarray(trunk_rotation, dtype=np.float64)
            expected_c7 = m_centre + r @ offset
            return float(np.linalg.norm(obs - expected_c7))

        expected_c7 = m_centre + offset
        return float(np.linalg.norm(obs - expected_c7))


@dataclass(frozen=True, slots=True)
class ObservabilityDiagnosticReceipt:
    """Immutable diagnostic receipt documenting head, trunk, and grip observability."""

    schema_version: str
    head_attachment: AttachmentTransform
    grip_calibration: GripAndClubfaceCalibration
    trunk_observable: TrunkObservableKind
    head_centre_error_m: float
    head_orientation_error_deg: float
    marker_residuals_mm: Mapping[str, float]
    trunk_com_residual_m: float

    def __post_init__(self) -> None:
        require(
            self.schema_version == OBSERVABILITY_RECEIPT_SCHEMA,
            f"Unsupported schema_version {self.schema_version!r} (expected {OBSERVABILITY_RECEIPT_SCHEMA!r})",
        )
        require(self.head_centre_error_m >= 0.0, "head_centre_error_m must be >= 0")
        require(
            self.head_orientation_error_deg >= 0.0,
            "head_orientation_error_deg must be >= 0",
        )
        require(self.trunk_com_residual_m >= 0.0, "trunk_com_residual_m must be >= 0")
        # Freeze the caller-owned mapping into an immutable proxy so neither the
        # original mapping nor the field can mutate recorded receipt values.
        object.__setattr__(
            self,
            "marker_residuals_mm",
            MappingProxyType(
                {str(k): float(v) for k, v in self.marker_residuals_mm.items()}
            ),
        )

    @property
    def receipt_sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "head_attachment_sha256": self.head_attachment.transform_sha256,
            "grip_calibration_sha256": self.grip_calibration.calibration_sha256,
            "trunk_observable": self.trunk_observable.value,
            "head_centre_error_m": round(self.head_centre_error_m, 6),
            "head_orientation_error_deg": round(self.head_orientation_error_deg, 4),
            "marker_residuals_mm": {
                k: round(v, 4) for k, v in sorted(self.marker_residuals_mm.items())
            },
            "trunk_com_residual_m": round(self.trunk_com_residual_m, 6),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def save_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "head_attachment": self.head_attachment.as_dict(),
            "grip_calibration": self.grip_calibration.as_dict(),
            "trunk_observable": self.trunk_observable.value,
            "head_centre_error_m": float(self.head_centre_error_m),
            "head_orientation_error_deg": float(self.head_orientation_error_deg),
            "marker_residuals_mm": {
                k: float(v) for k, v in self.marker_residuals_mm.items()
            },
            "trunk_com_residual_m": float(self.trunk_com_residual_m),
            "receipt_sha256": self.receipt_sha256,
        }
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load_json(cls, path: Path | str) -> ObservabilityDiagnosticReceipt:
        target = Path(path)
        require(target.is_file(), f"Receipt file does not exist: {target}")
        data = json.loads(target.read_text(encoding="utf-8"))
        receipt = cls(
            schema_version=str(data["schema_version"]),
            head_attachment=AttachmentTransform.from_dict(data["head_attachment"]),
            grip_calibration=GripAndClubfaceCalibration.from_dict(
                data["grip_calibration"]
            ),
            trunk_observable=TrunkObservableKind(data["trunk_observable"]),
            head_centre_error_m=float(data["head_centre_error_m"]),
            head_orientation_error_deg=float(data["head_orientation_error_deg"]),
            marker_residuals_mm={
                str(k): float(v) for k, v in data.get("marker_residuals_mm", {}).items()
            },
            trunk_com_residual_m=float(data["trunk_com_residual_m"]),
        )
        stored_sha = data.get("receipt_sha256")
        if stored_sha is None:
            raise ValueError(
                "Receipt is missing receipt_sha256; integrity cannot be verified "
                f"for schema {OBSERVABILITY_RECEIPT_SCHEMA}"
            )
        if stored_sha != receipt.receipt_sha256:
            raise ValueError(
                f"Receipt SHA256 mismatch on load: stored {stored_sha} != computed {receipt.receipt_sha256}"
            )
        return receipt
