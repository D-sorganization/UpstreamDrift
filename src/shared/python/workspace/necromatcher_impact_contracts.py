"""Immutable authored point, face and impact-selection assumptions.

These inputs define a research extraction; no geometry, contact event or historical
clock calibration is inferred. All length/translation fields use metres.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np


def finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Require finite real values, excluding boolean and text coercions."""
    array = np.asarray(value)
    if (
        any(
            isinstance(v, (bool, np.bool_))
            for v in np.asarray(value, dtype=object).ravel()
        )
        or array.dtype.kind not in "iuf"
        or array.shape != shape
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} requires finite real numeric {shape}")
    return np.array(array, dtype=float, copy=True)


def nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} requires a nonempty trimmed declaration")


def strict_fields(record: Mapping[str, Any], cls: type) -> None:
    if not isinstance(record, Mapping) or set(record) != {f.name for f in fields(cls)}:
        raise ValueError("Impact declaration must contain exactly its typed fields")


@dataclass(frozen=True)
class ReplayImpactGeometry:
    """Explicit body-local head point and orthonormal face normal/up axes.

    Mass and scalar MOI are authored effective impact-model assumptions. The
    point is not implicitly a named Clubhead frame or solid COM. Detached tuples
    retain exact inputs; unit axes are required rather than silently normalized.
    """

    body: str
    local_head_point_m: tuple[float, float, float]
    local_face_normal: tuple[float, float, float]
    local_face_up: tuple[float, float, float]
    mass_kg: float
    moi_kg_m2: float
    assumption_description: str

    def __post_init__(self) -> None:
        nonempty(self.body, "Body")
        nonempty(self.assumption_description, "Geometry assumptions")
        for name in ("local_head_point_m", "local_face_normal", "local_face_up"):
            value = finite_array(getattr(self, name), (3,), name)
            object.__setattr__(self, name, tuple(float(v) for v in value))
        axes = np.array([self.local_face_normal, self.local_face_up])
        if not np.allclose(axes @ axes.T, np.eye(2), rtol=0, atol=1e-12):
            raise ValueError("Face normal/up must be unit and perpendicular")
        for name in ("mass_kg", "moi_kg_m2"):
            value = finite_array(getattr(self, name), (), name).item()
            if value <= 0:
                raise ValueError("Effective mass and MOI must be positive")
            object.__setattr__(self, name, float(value))

    def to_record(self) -> dict[str, Any]:
        """Return detached JSON-compatible declarations."""
        record = asdict(self)
        for key in ("local_head_point_m", "local_face_normal", "local_face_up"):
            record[key] = list(record[key])
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ReplayImpactGeometry:
        strict_fields(record, cls)
        return cls(**dict(record))


@dataclass(frozen=True)
class ReplayImpactSelection:
    """Recorded replay sample and authored proper rigid model-world→flight map.

    Translation locates positions only; vectors are rotated without translation.
    Sample index is an operator choice, not automatic collision detection.
    """

    recorded_sample_index: int
    world_to_flight_rotation: tuple[tuple[float, float, float], ...]
    world_to_flight_translation_m: tuple[float, float, float]
    selection_description: str

    def __post_init__(self) -> None:
        if (
            type(self.recorded_sample_index) is not int
            or self.recorded_sample_index < 0
        ):
            raise ValueError("Recorded sample must be a nonnegative integer")
        nonempty(self.selection_description, "Selection assumptions")
        rotation = finite_array(
            self.world_to_flight_rotation, (3, 3), "Flight rotation"
        )
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-12
        ) or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-12):
            raise ValueError("Flight transform requires a proper orthonormal rotation")
        translation = finite_array(
            self.world_to_flight_translation_m, (3,), "Flight translation"
        )
        object.__setattr__(
            self,
            "world_to_flight_rotation",
            tuple(tuple(float(v) for v in row) for row in rotation),
        )
        object.__setattr__(
            self, "world_to_flight_translation_m", tuple(float(v) for v in translation)
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["world_to_flight_rotation"] = [
            list(row) for row in self.world_to_flight_rotation
        ]
        record["world_to_flight_translation_m"] = list(
            self.world_to_flight_translation_m
        )
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ReplayImpactSelection:
        strict_fields(record, cls)
        return cls(**dict(record))
