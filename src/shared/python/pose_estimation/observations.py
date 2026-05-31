"""Canonical markerless observations for physics-matched estimation.

The records in this module preserve the input side of markerless pose
estimation: per-camera 2D keypoints, per-keypoint confidence, camera
calibration, and optional triangulated 3D keypoints. They intentionally keep
camera observations separate from any collapsed skeleton so downstream
estimators can weight residuals by source and confidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import check_finite, require
from src.shared.python.simulation_backends.protocol import Trace

CANONICAL_OBSERVATIONS_SCHEMA_VERSION: Final[str] = "1.0.0"
TRACE_META_OBSERVATIONS_JSON: Final[str] = "canonical_observations_json"
_ROTATION_TOL: Final[float] = 1.0e-6


def _as_float_array(value: object, *, name: str) -> npt.NDArray[np.float64]:
    arr = np.asarray(value, dtype=float).copy()
    require(check_finite(arr), f"{name} must contain only finite values", arr.shape)
    arr.setflags(write=False)
    return arr


def _require_shape(
    arr: npt.NDArray[np.float64], *, name: str, shape: tuple[int, ...]
) -> None:
    require(arr.shape == shape, f"{name} must have shape {shape}, got {arr.shape}")


def _readonly_tuple(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    require(isinstance(values, Sequence), f"{name} must be a sequence")
    converted = tuple(str(value) for value in values)
    require(len(converted) > 0, f"{name} must be non-empty")
    require(
        all(value.strip() for value in converted), f"{name} entries must be non-empty"
    )
    require(
        len(set(converted)) == len(converted),
        f"{name} entries must be unique",
        converted,
    )
    return converted


@dataclass(frozen=True, slots=True)
class DetectorLayout:
    """Declared detector keypoint order for every frame in an observation set."""

    name: str
    keypoint_names: Sequence[str]

    def __post_init__(self) -> None:
        require(isinstance(self.name, str), "detector layout name must be a string")
        require(self.name.strip() != "", "detector layout name must be non-empty")
        object.__setattr__(
            self,
            "keypoint_names",
            _readonly_tuple(self.keypoint_names, name="keypoint_names"),
        )

    @property
    def keypoint_count(self) -> int:
        """Number of declared keypoints."""
        return len(self.keypoint_names)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {"name": self.name, "keypoint_names": list(self.keypoint_names)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DetectorLayout:
        """Build a detector layout from a JSON-like mapping."""
        require(isinstance(payload, Mapping), "detector layout must be a mapping")
        return cls(
            name=str(payload["name"]),
            keypoint_names=tuple(payload["keypoint_names"]),
        )


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics and optional distortion coefficients."""

    matrix: npt.NDArray[np.float64]
    distortion: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        matrix = _as_float_array(self.matrix, name="intrinsics.matrix")
        _require_shape(matrix, name="intrinsics.matrix", shape=(3, 3))
        require(matrix[2, 2] != 0.0, "intrinsics.matrix bottom-right must be non-zero")
        require(matrix[0, 0] > 0.0, "intrinsics.matrix fx must be positive")
        require(matrix[1, 1] > 0.0, "intrinsics.matrix fy must be positive")
        object.__setattr__(self, "matrix", matrix)

        if self.distortion is None:
            object.__setattr__(self, "distortion", None)
            return
        distortion = _as_float_array(self.distortion, name="intrinsics.distortion")
        require(
            distortion.ndim == 1,
            f"intrinsics.distortion must be 1-D, got {distortion.shape}",
        )
        object.__setattr__(self, "distortion", distortion)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "matrix": self.matrix.tolist(),
            "distortion": None if self.distortion is None else self.distortion.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CameraIntrinsics:
        """Build intrinsics from a JSON-like mapping."""
        require(isinstance(payload, Mapping), "intrinsics must be a mapping")
        distortion = payload.get("distortion")
        return cls(
            matrix=np.asarray(payload["matrix"], dtype=float),
            distortion=None
            if distortion is None
            else np.asarray(distortion, dtype=float),
        )


@dataclass(frozen=True, slots=True)
class CameraExtrinsics:
    """Camera pose as a transform from camera coordinates to world coordinates."""

    rotation_world_from_camera: npt.NDArray[np.float64]
    translation_world_from_camera_m: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        rotation = _as_float_array(
            self.rotation_world_from_camera, name="rotation_world_from_camera"
        )
        _require_shape(rotation, name="rotation_world_from_camera", shape=(3, 3))
        should_be_identity = rotation.T @ rotation
        require(
            np.allclose(should_be_identity, np.eye(3), atol=_ROTATION_TOL),
            "rotation_world_from_camera must be orthonormal",
        )
        require(
            np.isclose(np.linalg.det(rotation), 1.0, atol=_ROTATION_TOL),
            "rotation_world_from_camera determinant must be +1",
        )
        translation = _as_float_array(
            self.translation_world_from_camera_m,
            name="translation_world_from_camera_m",
        )
        _require_shape(
            translation,
            name="translation_world_from_camera_m",
            shape=(3,),
        )
        object.__setattr__(self, "rotation_world_from_camera", rotation)
        object.__setattr__(self, "translation_world_from_camera_m", translation)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "rotation_world_from_camera": self.rotation_world_from_camera.tolist(),
            "translation_world_from_camera_m": (
                self.translation_world_from_camera_m.tolist()
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CameraExtrinsics:
        """Build extrinsics from a JSON-like mapping."""
        require(isinstance(payload, Mapping), "extrinsics must be a mapping")
        return cls(
            rotation_world_from_camera=np.asarray(
                payload["rotation_world_from_camera"], dtype=float
            ),
            translation_world_from_camera_m=np.asarray(
                payload["translation_world_from_camera_m"], dtype=float
            ),
        )


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Calibration for one camera contributing 2D observations."""

    camera_id: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    image_size_px: tuple[int, int]

    def __post_init__(self) -> None:
        require(isinstance(self.camera_id, str), "camera_id must be a string")
        require(self.camera_id.strip() != "", "camera_id must be non-empty")
        require(
            isinstance(self.intrinsics, CameraIntrinsics),
            "intrinsics must be a CameraIntrinsics",
        )
        require(
            isinstance(self.extrinsics, CameraExtrinsics),
            "extrinsics must be a CameraExtrinsics",
        )
        require(len(self.image_size_px) == 2, "image_size_px must be (width, height)")
        width, height = (int(self.image_size_px[0]), int(self.image_size_px[1]))
        require(width > 0 and height > 0, "image_size_px values must be positive")
        object.__setattr__(self, "image_size_px", (width, height))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "camera_id": self.camera_id,
            "image_size_px": list(self.image_size_px),
            "intrinsics": self.intrinsics.to_dict(),
            "extrinsics": self.extrinsics.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CameraCalibration:
        """Build calibration from a JSON-like mapping."""
        require(isinstance(payload, Mapping), "camera calibration must be a mapping")
        return cls(
            camera_id=str(payload["camera_id"]),
            image_size_px=tuple(payload["image_size_px"]),  # type: ignore[arg-type]
            intrinsics=CameraIntrinsics.from_dict(payload["intrinsics"]),
            extrinsics=CameraExtrinsics.from_dict(payload["extrinsics"]),
        )


@dataclass(frozen=True, slots=True)
class KeypointObservation:
    """A single camera's 2D keypoint observations at one timestamp."""

    camera_id: str
    time_s: float
    keypoints_px: npt.NDArray[np.float64]
    confidence: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        require(isinstance(self.camera_id, str), "camera_id must be a string")
        require(self.camera_id.strip() != "", "camera_id must be non-empty")
        time_s = float(self.time_s)
        require(np.isfinite(time_s), "time_s must be finite")
        object.__setattr__(self, "time_s", time_s)

        keypoints = _as_float_array(self.keypoints_px, name="keypoints_px")
        require(
            keypoints.ndim == 2 and keypoints.shape[1] == 2,
            f"keypoints_px must have shape (K, 2), got {keypoints.shape}",
        )
        confidence = _as_float_array(self.confidence, name="confidence")
        require(
            confidence.shape == (keypoints.shape[0],),
            "confidence must have shape (K,) matching keypoints_px",
        )
        require(
            bool(np.all((confidence >= 0.0) & (confidence <= 1.0))),
            "confidence entries must be in [0, 1]",
        )
        object.__setattr__(self, "keypoints_px", keypoints)
        object.__setattr__(self, "confidence", confidence)

    @property
    def keypoint_count(self) -> int:
        """Number of keypoints in this frame."""
        return int(self.keypoints_px.shape[0])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "camera_id": self.camera_id,
            "time_s": self.time_s,
            "keypoints_px": self.keypoints_px.tolist(),
            "confidence": self.confidence.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> KeypointObservation:
        """Build a keypoint observation from a JSON-like mapping."""
        require(isinstance(payload, Mapping), "keypoint observation must be a mapping")
        return cls(
            camera_id=str(payload["camera_id"]),
            time_s=float(payload["time_s"]),
            keypoints_px=np.asarray(payload["keypoints_px"], dtype=float),
            confidence=np.asarray(payload["confidence"], dtype=float),
        )


@dataclass(frozen=True, slots=True)
class CanonicalObservations:
    """Canonical markerless observations consumed by estimators.

    ``frames`` holds per-camera 2D observations. ``keypoints_3d_m`` is optional
    triangulated data with shape ``(T, K, 3)`` and must not replace the original
    camera observations.
    """

    detector_layout: DetectorLayout
    cameras: Sequence[CameraCalibration]
    frames: Sequence[KeypointObservation]
    keypoints_3d_m: npt.NDArray[np.float64] | None = None
    keypoints_3d_confidence: npt.NDArray[np.float64] | None = None
    provenance: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    schema_version: str = CANONICAL_OBSERVATIONS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require(
            isinstance(self.detector_layout, DetectorLayout),
            "detector_layout must be a DetectorLayout",
        )
        cameras = tuple(self.cameras)
        frames = tuple(self.frames)
        require(len(cameras) > 0, "cameras must be non-empty")
        require(len(frames) > 0, "frames must be non-empty")
        camera_ids = tuple(camera.camera_id for camera in cameras)
        require(
            len(set(camera_ids)) == len(camera_ids), "camera_id values must be unique"
        )
        camera_id_set = set(camera_ids)
        for frame in frames:
            require(
                frame.camera_id in camera_id_set,
                f"unknown camera_id {frame.camera_id!r} in observation frame",
            )
            require(
                frame.keypoint_count == self.detector_layout.keypoint_count,
                "frame keypoint count must match detector layout "
                f"({frame.keypoint_count} != {self.detector_layout.keypoint_count})",
            )
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "frames", frames)

        keypoints_3d = self._coerce_keypoints_3d()
        confidence_3d = self._coerce_3d_confidence(keypoints_3d)
        object.__setattr__(self, "keypoints_3d_m", keypoints_3d)
        object.__setattr__(self, "keypoints_3d_confidence", confidence_3d)
        object.__setattr__(self, "provenance", dict(self.provenance))
        require(
            self.schema_version == CANONICAL_OBSERVATIONS_SCHEMA_VERSION,
            f"unsupported CanonicalObservations schema_version {self.schema_version!r}",
        )

    def _coerce_keypoints_3d(self) -> npt.NDArray[np.float64] | None:
        if self.keypoints_3d_m is None:
            return None
        keypoints_3d = _as_float_array(self.keypoints_3d_m, name="keypoints_3d_m")
        require(
            keypoints_3d.ndim == 3 and keypoints_3d.shape[2] == 3,
            f"keypoints_3d_m must have shape (T, K, 3), got {keypoints_3d.shape}",
        )
        require(
            keypoints_3d.shape[1] == self.detector_layout.keypoint_count,
            "3D keypoint count must match detector layout",
        )
        return keypoints_3d

    def _coerce_3d_confidence(
        self, keypoints_3d: npt.NDArray[np.float64] | None
    ) -> npt.NDArray[np.float64] | None:
        if self.keypoints_3d_confidence is None:
            return None
        require(
            keypoints_3d is not None,
            "keypoints_3d_confidence requires keypoints_3d_m",
        )
        if keypoints_3d is None:
            raise ValueError("keypoints_3d_confidence requires keypoints_3d_m")
        confidence = _as_float_array(
            self.keypoints_3d_confidence, name="keypoints_3d_confidence"
        )
        expected_shape = keypoints_3d.shape[:2]
        require(
            confidence.shape == expected_shape,
            f"keypoints_3d_confidence must have shape {expected_shape}, "
            f"got {confidence.shape}",
        )
        require(
            bool(np.all((confidence >= 0.0) & (confidence <= 1.0))),
            "keypoints_3d_confidence entries must be in [0, 1]",
        )
        return confidence

    def camera(self, camera_id: str) -> CameraCalibration:
        """Return calibration for ``camera_id``."""
        for camera in self.cameras:
            if camera.camera_id == camera_id:
                return camera
        raise KeyError(f"unknown camera_id {camera_id!r}")

    def frames_for_camera(self, camera_id: str) -> tuple[KeypointObservation, ...]:
        """Return all frames emitted by one camera in recorded order."""
        self.camera(camera_id)
        return tuple(frame for frame in self.frames if frame.camera_id == camera_id)

    def to_trace(self) -> Trace:
        """Return a CC-4 ``Trace`` envelope carrying this observation payload.

        The optional triangulated 3D points map to ``Trace.markers``. The full
        2D, confidence, and calibration payload is stored as scalar provenance
        metadata so :func:`simulation_backends.trace_io.write_trace` preserves it
        without extending the result schema.
        """
        sample_times = sorted({frame.time_s for frame in self.frames})
        t = np.asarray(sample_times, dtype=float)
        empty_state = np.empty((t.shape[0], 0), dtype=float)
        meta: dict[str, str | int | float | bool] = dict(self.provenance)
        meta[TRACE_META_OBSERVATIONS_JSON] = json.dumps(
            self.to_dict(), separators=(",", ":")
        )
        return Trace(
            t=t,
            q=empty_state,
            v=empty_state.copy(),
            dt=_infer_dt(t),
            backend="canonical-observations",
            meta=meta,
            markers=self.keypoints_3d_m,
        )

    @classmethod
    def from_trace(cls, trace: Trace) -> CanonicalObservations:
        """Recover observations embedded by :meth:`to_trace`."""
        require(isinstance(trace, Trace), "trace must be a Trace")
        encoded = trace.meta.get(TRACE_META_OBSERVATIONS_JSON)
        require(
            isinstance(encoded, str),
            f"trace.meta must contain {TRACE_META_OBSERVATIONS_JSON!r}",
        )
        if not isinstance(encoded, str):
            raise ValueError(
                f"trace.meta must contain {TRACE_META_OBSERVATIONS_JSON!r}"
            )
        return cls.from_json(encoded)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "schema_version": self.schema_version,
            "detector_layout": self.detector_layout.to_dict(),
            "cameras": [camera.to_dict() for camera in self.cameras],
            "frames": [frame.to_dict() for frame in self.frames],
            "keypoints_3d_m": (
                None if self.keypoints_3d_m is None else self.keypoints_3d_m.tolist()
            ),
            "keypoints_3d_confidence": (
                None
                if self.keypoints_3d_confidence is None
                else self.keypoints_3d_confidence.tolist()
            ),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CanonicalObservations:
        """Build observations from a JSON-like mapping."""
        require(isinstance(payload, Mapping), "CanonicalObservations must be a mapping")
        keypoints_3d = payload.get("keypoints_3d_m")
        confidence_3d = payload.get("keypoints_3d_confidence")
        return cls(
            detector_layout=DetectorLayout.from_dict(payload["detector_layout"]),
            cameras=tuple(
                CameraCalibration.from_dict(item) for item in payload["cameras"]
            ),
            frames=tuple(
                KeypointObservation.from_dict(item) for item in payload["frames"]
            ),
            keypoints_3d_m=(
                None if keypoints_3d is None else np.asarray(keypoints_3d, dtype=float)
            ),
            keypoints_3d_confidence=(
                None
                if confidence_3d is None
                else np.asarray(confidence_3d, dtype=float)
            ),
            provenance=dict(payload.get("provenance", {})),
            schema_version=str(
                payload.get(
                    "schema_version",
                    CANONICAL_OBSERVATIONS_SCHEMA_VERSION,
                )
            ),
        )

    def to_json(self) -> str:
        """Serialise observations to stable JSON text."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> CanonicalObservations:
        """Inverse of :meth:`to_json`."""
        payload = json.loads(text)
        require(
            isinstance(payload, Mapping), "CanonicalObservations JSON must be an object"
        )
        return cls.from_dict(payload)

    def to_path(self, output_path: Path | str) -> None:
        """Write :meth:`to_json` to ``output_path`` using UTF-8."""
        path = Path(output_path)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_path(cls, input_path: Path | str) -> CanonicalObservations:
        """Load observations from a UTF-8 JSON file."""
        return cls.from_json(Path(input_path).read_text(encoding="utf-8"))


def _infer_dt(t: npt.NDArray[np.float64]) -> float:
    if t.shape[0] < 2:
        return 0.0
    return float(np.median(np.diff(t)))


__all__ = [
    "CANONICAL_OBSERVATIONS_SCHEMA_VERSION",
    "CameraCalibration",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "CanonicalObservations",
    "DetectorLayout",
    "KeypointObservation",
    "TRACE_META_OBSERVATIONS_JSON",
]
