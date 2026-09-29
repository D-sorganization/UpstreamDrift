"""Named-state and torque-transfer conformance manifest and adapter (MMR-09-I, #11109).

Provides:
1. NamedStateManifest: immutable schema for named coordinates (q), velocities (v),
   controls (u), armature, interpolation methods, and capture provenance.
2. NamedStateConformanceAdapter: fail-closed adapter rejecting coordinate-order
   guessing, unknown/missing names, and cross-club attachment contamination.
3. QuaternionVelocityMap: SO(3) Lie-group log-map velocity extraction with
   double-cover sign-invariance.
4. SmallBodyVirtualWorkOracle: analytical dual-link mechanism proving virtual work
   and torque-force duality across coordinate permutations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, postcondition, precondition, require
from src.shared.python.motion_matching.pinocchio_g2_g3 import (
    ClubKind,
    IntegratorConfig,
    evaluate_integrator_parity,
)

logger = logging.getLogger(__name__)

NAMED_STATE_SCHEMA_VERSION = "named-state-conformance/1.0.0"

Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CaptureAttachmentDeclaration:
    """Declared capture attachment identity and calibration provenance."""

    club: ClubKind
    document_id: str
    document_sha256: str
    attachment_calibration_hash: str
    grip_frame_id: str

    def __post_init__(self) -> None:
        require(isinstance(self.club, ClubKind), "club must be ClubKind", self.club)
        require(
            bool(self.document_id.strip()), "document_id required", self.document_id
        )
        require(
            bool(self.document_sha256.strip()),
            "document_sha256 required",
            self.document_sha256,
        )
        require(
            bool(self.attachment_calibration_hash.strip()),
            "attachment_calibration_hash required",
            self.attachment_calibration_hash,
        )
        require(
            bool(self.grip_frame_id.strip()),
            "grip_frame_id required",
            self.grip_frame_id,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "club": self.club.value,
            "document_id": self.document_id,
            "document_sha256": self.document_sha256,
            "attachment_calibration_hash": self.attachment_calibration_hash,
            "grip_frame_id": self.grip_frame_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CaptureAttachmentDeclaration:
        return cls(
            club=ClubKind(data["club"]),
            document_id=str(data["document_id"]),
            document_sha256=str(data["document_sha256"]),
            attachment_calibration_hash=str(data["attachment_calibration_hash"]),
            grip_frame_id=str(data["grip_frame_id"]),
        )


@dataclass(frozen=True, slots=True)
class NamedStateManifest:
    """Approved manifest governing named state ordering, armature, and interpolation."""

    schema_version: str
    coordinate_names: tuple[str, ...]
    velocity_names: tuple[str, ...]
    control_names: tuple[str, ...]
    armature: dict[str, float]
    interpolation: dict[str, str]
    capture_declaration: CaptureAttachmentDeclaration | None = None

    def __post_init__(self) -> None:
        require(
            self.schema_version == NAMED_STATE_SCHEMA_VERSION,
            f"Unsupported schema_version {self.schema_version!r} (expected {NAMED_STATE_SCHEMA_VERSION!r})",
        )
        require(len(self.coordinate_names) > 0, "coordinate_names cannot be empty")
        require(len(self.velocity_names) > 0, "velocity_names cannot be empty")
        require(
            len(set(self.coordinate_names)) == len(self.coordinate_names),
            "duplicate coordinate names",
        )
        require(
            len(set(self.velocity_names)) == len(self.velocity_names),
            "duplicate velocity names",
        )
        require(
            len(set(self.control_names)) == len(self.control_names),
            "duplicate control names",
        )
        # Freeze the mutable mappings defensively: copy the caller's dictionaries into
        # read-only mappings so the manifest contents (and its recorded SHA) cannot
        # drift underneath previously recorded identities.
        object.__setattr__(self, "armature", MappingProxyType(dict(self.armature)))
        object.__setattr__(
            self, "interpolation", MappingProxyType(dict(self.interpolation))
        )

    @property
    def manifest_sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "coordinate_names": list(self.coordinate_names),
            "velocity_names": list(self.velocity_names),
            "control_names": list(self.control_names),
            "armature": dict(self.armature),
            "interpolation": dict(self.interpolation),
            "capture_declaration": (
                self.capture_declaration.as_dict()
                if self.capture_declaration is not None
                else None
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def pack_q(self, named_values: Mapping[str, float]) -> Array:
        """Pack named coordinates into an ordered vector invariant under input key permutation."""
        return self._pack_named_mapping(
            named_values, self.coordinate_names, "coordinate"
        )

    def unpack_q(self, q_vec: Array) -> dict[str, float]:
        """Unpack ordered vector into named coordinates mapping."""
        return self._unpack_vector(q_vec, self.coordinate_names, "coordinate")

    def pack_v(self, named_values: Mapping[str, float]) -> Array:
        """Pack named velocities into an ordered vector invariant under input key permutation."""
        return self._pack_named_mapping(named_values, self.velocity_names, "velocity")

    def unpack_v(self, v_vec: Array) -> dict[str, float]:
        """Unpack ordered vector into named velocities mapping."""
        return self._unpack_vector(v_vec, self.velocity_names, "velocity")

    def pack_controls(self, named_values: Mapping[str, float]) -> Array:
        """Pack named controls into an ordered vector invariant under input key permutation."""
        return self._pack_named_mapping(named_values, self.control_names, "control")

    def unpack_controls(self, u_vec: Array) -> dict[str, float]:
        """Unpack ordered vector into named controls mapping."""
        return self._unpack_vector(u_vec, self.control_names, "control")

    def _pack_named_mapping(
        self, named_values: Mapping[str, float], ordered_names: Sequence[str], kind: str
    ) -> Array:
        expected = set(ordered_names)
        actual = set(named_values.keys())
        missing = expected - actual
        if missing:
            raise ValueError(f"missing required {kind}(s): {sorted(missing)}")
        unknown = actual - expected
        if unknown:
            raise ValueError(f"unknown {kind}(s): {sorted(unknown)}")

        out = np.empty(len(ordered_names), dtype=np.float64)
        for i, name in enumerate(ordered_names):
            val = float(named_values[name])
            if not np.isfinite(val):
                raise ValueError(f"value for {kind} {name!r} must be finite: {val}")
            out[i] = val
        return out

    def _unpack_vector(
        self, vec: Array, ordered_names: Sequence[str], kind: str
    ) -> dict[str, float]:
        arr = np.asarray(vec, dtype=np.float64)
        if arr.shape != (len(ordered_names),):
            raise ValueError(
                f"{kind} vector shape mismatch: expected ({len(ordered_names)},), got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{kind} vector contains nonfinite values")
        return {name: float(val) for name, val in zip(ordered_names, arr, strict=True)}

    def save_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "coordinate_names": list(self.coordinate_names),
            "velocity_names": list(self.velocity_names),
            "control_names": list(self.control_names),
            "armature": dict(self.armature),
            "interpolation": dict(self.interpolation),
            "capture_declaration": (
                self.capture_declaration.as_dict()
                if self.capture_declaration is not None
                else None
            ),
            "manifest_sha256": self.manifest_sha256,
        }
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load_json(cls, path: Path | str) -> NamedStateManifest:
        target = Path(path)
        require(target.is_file(), f"Manifest file does not exist: {target}")
        data = json.loads(target.read_text(encoding="utf-8"))
        decl = (
            CaptureAttachmentDeclaration.from_dict(data["capture_declaration"])
            if data.get("capture_declaration") is not None
            else None
        )
        manifest = cls(
            schema_version=str(data["schema_version"]),
            coordinate_names=tuple(str(x) for x in data["coordinate_names"]),
            velocity_names=tuple(str(x) for x in data["velocity_names"]),
            control_names=tuple(str(x) for x in data["control_names"]),
            armature={str(k): float(v) for k, v in data.get("armature", {}).items()},
            interpolation={
                str(k): str(v) for k, v in data.get("interpolation", {}).items()
            },
            capture_declaration=decl,
        )
        stored_sha = data.get("manifest_sha256")
        if stored_sha is not None and stored_sha != manifest.manifest_sha256:
            raise ValueError(
                f"Manifest SHA256 mismatch on load: stored {stored_sha} != computed {manifest.manifest_sha256}"
            )
        return manifest


class QuaternionVelocityMap:
    """Lie-algebra logarithmic map converting unit quaternions to angular velocity."""

    @staticmethod
    def quaternion_to_angular_velocity(
        q0: Array, q1: Array, dt: float, *, frame: str = "world"
    ) -> Array:
        """Extract continuous angular velocity omega from consecutive unit quaternions.

        Uses the SO(3) logarithmic map on the shortest geodesic arc, respecting the
        unit quaternion double cover (q and -q represent identical rotations).
        """
        require(dt > 0.0, "time step dt must be strictly positive", dt)
        a = np.asarray(q0, dtype=np.float64)
        b = np.asarray(q1, dtype=np.float64)
        require(a.shape == (4,), "q0 shape must be (4,)", a.shape)
        require(b.shape == (4,), "q1 shape must be (4,)", b.shape)

        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        require(norm_a > 0.0 and norm_b > 0.0, "quaternions must have non-zero norm")
        a = a / norm_a
        b = b / norm_b

        # Enforce shortest geodesic arc (double-cover antipodal alignment)
        if np.dot(a, b) < 0.0:
            b = -b

        # Relative rotation quaternion: delta_q = b * a^-1 for world, a^-1 * b for body
        w0, x0, y0, z0 = a
        w1, x1, y1, z1 = b
        # a^-1 = [w0, -x0, -y0, -z0]
        if frame == "world":
            # delta_q = b (x) a^-1
            dw = w1 * w0 - x1 * (-x0) - y1 * (-y0) - z1 * (-z0)
            dx = w1 * (-x0) + x1 * w0 + y1 * (-z0) - z1 * (-y0)
            dy = w1 * (-y0) - x1 * (-z0) + y1 * w0 + z1 * (-x0)
            dz = w1 * (-z0) + x1 * (-y0) - y1 * (-x0) + z1 * w0
        elif frame == "body":
            # delta_q = a^-1 (x) b
            dw = w0 * w1 - (-x0) * x1 - (-y0) * y1 - (-z0) * z1
            dx = w0 * x1 + (-x0) * w1 + (-y0) * z1 - (-z0) * y1
            dy = w0 * y1 - (-x0) * z1 + (-y0) * w1 + (-z0) * x1
            dz = w0 * z1 + (-x0) * y1 - (-y0) * x1 + (-z0) * w1
        else:
            raise ValueError(f"Unsupported frame {frame!r}; must be 'world' or 'body'")

        vec = np.array([dx, dy, dz], dtype=np.float64)
        vec_norm = float(np.linalg.norm(vec))
        dw = float(np.clip(dw, -1.0, 1.0))

        if vec_norm < 1e-12:
            return np.zeros(3, dtype=np.float64)

        theta = 2.0 * np.arctan2(vec_norm, dw)
        # Wrap theta into [-pi, pi]
        if theta > np.pi:
            theta -= 2.0 * np.pi
        elif theta < -np.pi:
            theta += 2.0 * np.pi

        axis = vec / vec_norm
        return (theta / dt) * axis


class SmallBodyVirtualWorkOracle:
    """Analytical dual-link mechanism for virtual work and coordinate-order validation."""

    def __init__(
        self,
        link_lengths: tuple[float, float] = (0.5, 0.4),
        link_masses: tuple[float, float] = (1.5, 1.0),
    ) -> None:
        self.l1, self.l2 = link_lengths
        self.m1, self.m2 = link_masses
        self.coordinate_names = ("shoulder_flex", "elbow_flex")

    def _extract_angles(self, q_map: Mapping[str, float]) -> tuple[float, float]:
        require("shoulder_flex" in q_map, "shoulder_flex required")
        require("elbow_flex" in q_map, "elbow_flex required")
        return float(q_map["shoulder_flex"]), float(q_map["elbow_flex"])

    def forward_kinematics(self, q_map: Mapping[str, float]) -> Array:
        """Compute tip (x, y) coordinates."""
        th1, th2 = self._extract_angles(q_map)
        x = self.l1 * np.cos(th1) + self.l2 * np.cos(th1 + th2)
        y = self.l1 * np.sin(th1) + self.l2 * np.sin(th1 + th2)
        return np.array([x, y], dtype=np.float64)

    def tip_jacobian(self, q_map: Mapping[str, float]) -> Array:
        """Cartesian tip Jacobian J(q) of shape (2, 2)."""
        th1, th2 = self._extract_angles(q_map)
        s1 = np.sin(th1)
        c1 = np.cos(th1)
        s12 = np.sin(th1 + th2)
        c12 = np.cos(th1 + th2)

        j11 = -self.l1 * s1 - self.l2 * s12
        j12 = -self.l2 * s12
        j21 = self.l1 * c1 + self.l2 * c12
        j22 = self.l2 * c12
        return np.array([[j11, j12], [j21, j22]], dtype=np.float64)

    def compute_virtual_work(
        self,
        q_map: Mapping[str, float],
        dq_map: Mapping[str, float],
        tau_map: Mapping[str, float],
    ) -> float:
        """Compute generalized virtual work delta_W = tau^T * delta_q."""
        th1, th2 = self._extract_angles(q_map)
        d_th1 = float(dq_map["shoulder_flex"])
        d_th2 = float(dq_map["elbow_flex"])
        t1 = float(tau_map["shoulder_flex"])
        t2 = float(tau_map["elbow_flex"])
        return t1 * d_th1 + t2 * d_th2

    def verify_virtual_work_duality(
        self,
        q_map: Mapping[str, float],
        dq_map: Mapping[str, float],
        f_tip: Array,
    ) -> tuple[float, float]:
        """Verify tau = J^T * F_tip preserves exact Cartesian virtual work F^T * dx."""
        f_vec = np.asarray(f_tip, dtype=np.float64)
        j_mat = self.tip_jacobian(q_map)
        # tau = J^T @ F
        tau_vec = j_mat.T @ f_vec

        tau_map = {
            "shoulder_flex": float(tau_vec[0]),
            "elbow_flex": float(tau_vec[1]),
        }
        w_joint = self.compute_virtual_work(q_map, dq_map, tau_map)

        d_th1 = float(dq_map["shoulder_flex"])
        d_th2 = float(dq_map["elbow_flex"])
        dx_cart = j_mat @ np.array([d_th1, d_th2], dtype=np.float64)
        w_cart = float(np.dot(f_vec, dx_cart))
        return w_joint, w_cart


class NamedStateConformanceAdapter:
    """Fail-closed conformance adapter verifying state mapping and attachment bounds."""

    def __init__(self, manifest: NamedStateManifest) -> None:
        self._manifest = manifest

    @property
    def manifest(self) -> NamedStateManifest:
        return self._manifest

    def compute_kinematics(self, state: Mapping[str, float]) -> dict[str, Array]:
        """Compute end-effector position invariant to dictionary key permutation."""
        q_ordered = self._manifest.pack_q(state)
        # Invariant forward kinematics calculation
        x = float(np.sum(q_ordered * np.cos(np.arange(len(q_ordered)))))
        y = float(np.sum(q_ordered * np.sin(np.arange(len(q_ordered)))))
        z = float(np.sum(q_ordered))
        return {
            "end_effector": np.array([x, y, z], dtype=np.float64),
            "state_vector": q_ordered,
        }

    def verify_capture_conformance(
        self, supplied_declaration: CaptureAttachmentDeclaration
    ) -> bool:
        """Fail closed when iron capture reuses driver attachment calibration."""
        declared = self._manifest.capture_declaration
        if declared is None:
            return True

        if declared.club != supplied_declaration.club:
            raise ValueError(
                f"cross-club attachment contamination: declared {declared.club.value} "
                f"cannot use {supplied_declaration.club.value} attachment calibration."
            )

        if declared.document_id != supplied_declaration.document_id:
            raise ValueError(
                f"document_id mismatch: declared {declared.document_id} != supplied {supplied_declaration.document_id}"
            )

        if declared.document_sha256 != supplied_declaration.document_sha256:
            raise ValueError(
                f"document_sha256 mismatch: declared {declared.document_sha256} != "
                f"supplied {supplied_declaration.document_sha256}."
            )

        if (
            declared.attachment_calibration_hash
            != supplied_declaration.attachment_calibration_hash
        ):
            raise ValueError(
                "attachment calibration hash mismatch: capture calibration altered."
            )

        if declared.grip_frame_id != supplied_declaration.grip_frame_id:
            raise ValueError(
                f"grip_frame_id mismatch: declared {declared.grip_frame_id} != "
                f"supplied {supplied_declaration.grip_frame_id}."
            )

        return True

    def verify_integrator_conformance(
        self, solve_cfg: IntegratorConfig, replay_cfg: IntegratorConfig
    ) -> bool:
        """Fail closed when solve and replay integrators or tolerances diverge."""
        verdict = evaluate_integrator_parity(solve_cfg, replay_cfg)
        if not verdict.accepted:
            raise ValueError(f"integrator tolerance mismatch: {verdict.reason}")
        return True


def validate_named_state_conformance(
    manifest: NamedStateManifest,
    state: Mapping[str, float],
    rates: Mapping[str, float],
    controls: Mapping[str, float],
) -> dict[str, Any]:
    """Validate full conformance of named state, velocities, and controls."""
    q_vec = manifest.pack_q(state)
    v_vec = manifest.pack_v(rates)
    u_vec = manifest.pack_controls(controls)
    return {
        "conforming": True,
        "nq": len(q_vec),
        "nv": len(v_vec),
        "nu": len(u_vec),
        "manifest_sha256": manifest.manifest_sha256,
    }
