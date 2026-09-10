"""Calibrated model bindings shared by live dynamics and imported kinematics.

An engine link is not an anatomical segment. A binding explicitly defines its
anatomical frame, mass membership and units before the shared metrics see it.
Calibration translations and local COM are metres; input translations use
``length_scale`` metres per source unit. Missing links remain missing samples.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from src.shared.python.pose_interchange.se3 import is_valid_se3


def _rigid(value: Any, name: str) -> np.ndarray:
    matrix = np.array(value, dtype=float, copy=True)
    if not is_valid_se3(matrix):
        raise ValueError(f"{name} must be SE(3) with a proper rotation")
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True)
class SegmentBinding:
    """Link-to-anatomy calibration with explicit mass and membership."""

    link: str
    calibration: np.ndarray = field(default_factory=lambda: np.eye(4))
    calibration_id: str = "model-link-frame"
    mass_kg: float | None = None
    local_com: tuple[float, float, float] | None = None
    membership: str = "other"

    def __post_init__(self) -> None:
        if not self.link or not self.calibration_id:
            raise ValueError("link and calibration_id must be nonempty")
        object.__setattr__(self, "calibration", _rigid(self.calibration, "calibration"))
        if self.membership not in {"body", "club", "other"}:
            raise ValueError("membership must be body, club or other")
        if self.mass_kg is not None and (
            not np.isfinite(self.mass_kg) or self.mass_kg <= 0
        ):
            raise ValueError("mass_kg must be positive and finite")
        if self.local_com is not None:
            com = np.asarray(self.local_com, dtype=float)
            if com.shape != (3,) or not np.all(np.isfinite(com)):
                raise ValueError("local_com must be a finite three-vector in metres")
            object.__setattr__(self, "local_com", tuple(float(v) for v in com))


@dataclass(frozen=True)
class ModelBinding:
    """One model's declared mapping into a right-handed analysis world."""

    source: str
    world_frame: str
    segments: Mapping[str, SegmentBinding]
    length_scale: float = 1.0
    world_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    expected_body_segments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source or not self.world_frame or not self.segments:
            raise ValueError("source, world_frame and segments must be nonempty")
        if not np.isfinite(self.length_scale) or self.length_scale <= 0:
            raise ValueError("length_scale must be finite and positive")
        if any(
            not k or not isinstance(v, SegmentBinding) for k, v in self.segments.items()
        ):
            raise ValueError("segments must map nonempty names to SegmentBinding")
        from types import MappingProxyType

        object.__setattr__(self, "segments", MappingProxyType(dict(self.segments)))
        object.__setattr__(
            self, "world_transform", _rigid(self.world_transform, "world_transform")
        )
        object.__setattr__(
            self, "expected_body_segments", tuple(self.expected_body_segments)
        )


class TrajectoryRecorder:
    """Record calibrated transforms without changing or stepping the source model.

    Each append is atomic and copies source data. This same entry point accepts
    measured/inverse-kinematic transforms and forward-dynamic engine transforms.
    """

    def __init__(self, binding: ModelBinding, max_samples: int = 100_000) -> None:
        if not isinstance(binding, ModelBinding):
            raise TypeError("binding must be ModelBinding")
        if (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, int)
            or max_samples < 2
        ):
            raise ValueError("max_samples must be an integer of at least two")
        self.binding = binding
        self.max_samples = max_samples
        self._times: list[float] = []
        self._frames: list[dict[str, np.ndarray]] = []

    def append(self, time: float, transforms: Mapping[str, np.ndarray]) -> None:
        """Append one source-unit snapshot; timestamps must strictly increase."""
        if not np.isfinite(time) or (self._times and time <= self._times[-1]):
            raise ValueError("timestamps must be finite and strictly increasing")
        if len(self._times) >= self.max_samples:
            raise ValueError("maximum recording samples reached")
        frame: dict[str, np.ndarray] = {}
        segments = self.binding.segments
        for name, segment in segments.items():
            if segment.link not in transforms:
                frame[name] = np.full((4, 4), np.nan)
                continue
            matrix = _rigid(transforms[segment.link], segment.link).copy()
            matrix[:3, 3] *= self.binding.length_scale
            frame[name] = self.binding.world_transform @ matrix @ segment.calibration
        self._frames.append(frame)
        self._times.append(float(time))

    def sample(self, time: float, source: Any) -> None:
        """Read the public LiveKinematicsService transform protocol once."""
        self.append(time, source.get_link_transforms())

    def to_payload(self, **analysis_options: Any) -> dict[str, Any]:
        """Return JSON-safe canonical metric input; unavailable samples are null."""
        if len(self._times) < 2:
            raise ValueError("at least two samples are required")
        segments: dict[str, Any] = {}
        bindings = self.binding.segments
        for name, binding in bindings.items():
            matrices = np.stack([frame[name] for frame in self._frames])
            segments[name] = {
                "positions": _nullable(matrices[:, :3, 3]),
                "rotations": _nullable(matrices[:, :3, :3]),
                "mass_kg": binding.mass_kg,
                "local_com": binding.local_com,
                "membership": binding.membership,
            }
        payload = {
            "times": list(self._times),
            "segments": segments,
            "source": self.binding.source,
            "world_frame": self.binding.world_frame,
            "expected_body_segments": list(self.binding.expected_body_segments),
        }
        if set(analysis_options) & set(payload):
            raise ValueError("analysis_options cannot replace recorded data")
        return {**payload, **analysis_options}


def _nullable(array: np.ndarray) -> list[Any]:
    finite = np.asarray(np.isfinite(array), dtype=bool)
    result = array.astype(object)
    result[~finite] = None
    return result.tolist()


def model_binding_from_dict(payload: Mapping[str, Any]) -> ModelBinding:
    """Parse one strict binding schema for desktop, web and scripting clients."""
    if not isinstance(payload, Mapping):
        raise TypeError("binding must be a mapping")
    unknown = set(payload) - {item.name for item in fields(ModelBinding)}
    if unknown:
        raise ValueError(f"Unknown binding fields: {sorted(unknown)}")
    values = dict(payload)
    raw_segments = values.get("segments", {})
    if not isinstance(raw_segments, Mapping):
        raise ValueError("segments must be a mapping")
    allowed = {item.name for item in fields(SegmentBinding)}
    segments = {}
    for name, segment in raw_segments.items():
        if not isinstance(segment, Mapping) or set(segment) - allowed:
            raise ValueError(f"Unknown or invalid segment binding fields: {name}")
        segments[name] = SegmentBinding(**segment)
    values["segments"] = segments
    return ModelBinding(**values)
