"""Immutable authored camera/model recipes; validation is not admission."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Real
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from src.shared.python.motion_matching.historical_fit import CameraProjection
from .project_store import validate_workspace_id

HYPOTHESIS_REQUEST_SCHEMA = "necromatcher/native-hypothesis-request/1"


def _record(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly its declared fields")
    return value


def _hash(value: Any) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
    ):
        raise ValueError("Hypothesis identities require prefixed SHA-256 hashes")
    return value


def _names(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} requires nonempty named coordinates")
    names = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in names):
        raise ValueError(f"{label} requires nonempty named coordinates")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must be unique")
    return names


def _numbers(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} requires a finite {size}-vector")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        raise ValueError(f"{label} requires finite real numbers")
    values = tuple(float(item) for item in value)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} requires finite real numbers")
    return values


@dataclass(frozen=True)
class HypothesisParents:
    """Caller pins, independently authenticated against the library on admission."""

    source_fit_id: str
    source_fit_hash: str
    capture_id: str
    capture_hash: str
    source_sha256: str
    source_clock_sha256: str

    def __post_init__(self) -> None:
        validate_workspace_id(self.source_fit_id, "Source fit ID")
        validate_workspace_id(self.capture_id, "Capture ID")
        for item in (
            self.source_fit_hash,
            self.capture_hash,
            self.source_sha256,
            self.source_clock_sha256,
        ):
            _hash(item)

    @classmethod
    def from_record(cls, record: Any) -> HypothesisParents:
        values = _record(record, set(cls.__dataclass_fields__), "Hypothesis parents")
        return cls(**values)

    def to_record(self) -> dict[str, str]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass(frozen=True)
class HypothesisModel:
    """Exact authored definition bytes and immutable body-local marker offsets."""

    model_id: str
    model_hash: str
    definition_bytes: bytes
    attachments: Mapping[str, tuple[str, tuple[float, ...]]]

    def __post_init__(self) -> None:
        validate_workspace_id(self.model_id, "Model ID")
        _hash(self.model_hash)
        if not isinstance(self.definition_bytes, bytes) or not self.definition_bytes:
            raise ValueError("Hypothesis requires exact definition bytes")
        try:
            definition = json.loads(self.definition_bytes)
            canonical = json.dumps(definition, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("Hypothesis definition must be finite JSON") from exc
        if not isinstance(definition, dict):
            raise ValueError("Hypothesis definition must be an object")
        if self.definition_bytes != canonical:
            raise ValueError(
                "Hypothesis definition requires canonical JSON serialization; "
                "use HypothesisModel.from_record"
            )
        if not isinstance(self.attachments, Mapping) or not self.attachments:
            raise ValueError("Hypothesis requires named marker attachments")
        copied = {}
        for name, attachment in self.attachments.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(attachment, (list, tuple))
                or len(attachment) != 2
            ):
                raise ValueError(
                    "Hypothesis attachments require names and body offsets"
                )
            body, offset = attachment
            if not isinstance(body, str) or not body:
                raise ValueError("Hypothesis marker attachment requires a named body")
            copied[name] = (body, _numbers(offset, 3, "Marker offset"))
        object.__setattr__(self, "attachments", MappingProxyType(copied))

    @property
    def definition_sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.definition_bytes).hexdigest()

    @classmethod
    def from_record(cls, record: Any) -> HypothesisModel:
        values = _record(
            record,
            {"model_id", "model_hash", "definition", "attachments"},
            "Hypothesis model",
        )
        try:
            encoded = json.dumps(values["definition"], allow_nan=False).encode("utf-8")
        except (ValueError, TypeError) as exc:
            raise ValueError("Hypothesis definition must be finite JSON") from exc
        return cls(
            values["model_id"], values["model_hash"], encoded, values["attachments"]
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "definition": json.loads(self.definition_bytes),
            "attachments": {
                name: [body, list(offset)]
                for name, (body, offset) in self.attachments.items()
            },
        }


@dataclass(frozen=True)
class HypothesisCoordinateMap:
    """Explicit scalar-unit/order and locked-pose mapping; no inferred defaults."""

    coordinate_order: tuple[str, ...]
    coordinate_units: tuple[str, ...]
    free_coordinates: tuple[str, ...]
    reference_pose: tuple[float, ...]

    def __post_init__(self) -> None:
        order = _names(self.coordinate_order, "Coordinate order")
        free = _names(self.free_coordinates, "Free coordinates")
        if not isinstance(self.coordinate_units, (list, tuple)):
            raise ValueError("Hypothesis requires explicit coordinate units")
        units = tuple(self.coordinate_units)
        if len(units) != len(order) or any(item not in ("m", "rad") for item in units):
            raise ValueError(
                "Hypothesis coordinate units must be native metres/radians"
            )
        if any(item not in order for item in free):
            raise ValueError("Free coordinates must belong to the declared order")
        object.__setattr__(self, "coordinate_order", order)
        object.__setattr__(self, "coordinate_units", units)
        object.__setattr__(self, "free_coordinates", free)
        object.__setattr__(
            self,
            "reference_pose",
            _numbers(self.reference_pose, len(order), "Reference pose"),
        )

    @classmethod
    def from_record(cls, record: Any) -> HypothesisCoordinateMap:
        return cls(
            **_record(
                record, set(cls.__dataclass_fields__), "Hypothesis coordinate mapping"
            )
        )

    def to_record(self) -> dict[str, Any]:
        return {key: list(getattr(self, key)) for key in self.__dataclass_fields__}


@dataclass(frozen=True)
class HypothesisGauge:
    """Authored single-view scale/origin constraints, without calibration claims."""

    stature_m: float
    world_origin: str
    world_orientation: str
    stature_source: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.stature_m, bool)
            or not isinstance(self.stature_m, Real)
            or not np.isfinite(self.stature_m)
            or self.stature_m <= 0
        ):
            raise ValueError("Authored stature hypothesis must be finite and positive")
        for name in ("world_origin", "world_orientation", "stature_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "Hypothesis gauge requires explicit authored constraints/source"
                )
        object.__setattr__(self, "stature_m", float(self.stature_m))

    @classmethod
    def from_record(cls, record: Any) -> HypothesisGauge:
        return cls(**_record(record, set(cls.__dataclass_fields__), "Hypothesis gauge"))

    def to_record(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass(frozen=True)
class NativeHypothesisRequest:
    """Versioned authored recipe; native/source authentication is a separate seam."""

    parents: HypothesisParents
    model: HypothesisModel
    camera: CameraProjection
    mapping: HypothesisCoordinateMap
    gauge: HypothesisGauge

    def __post_init__(self) -> None:
        expected = (
            HypothesisParents,
            HypothesisModel,
            CameraProjection,
            HypothesisCoordinateMap,
            HypothesisGauge,
        )
        if any(
            not isinstance(getattr(self, key), kind)
            for key, kind in zip(self.__dataclass_fields__, expected, strict=True)
        ):
            raise ValueError(
                "Native hypothesis requires validated typed nested records"
            )

    @classmethod
    def from_record(cls, record: Any) -> NativeHypothesisRequest:
        values = _record(
            record,
            {"schema_version", *cls.__dataclass_fields__},
            "Native hypothesis request",
        )
        if values["schema_version"] != HYPOTHESIS_REQUEST_SCHEMA:
            raise ValueError("Unknown native hypothesis request schema")
        camera = _record(
            values["camera"],
            {"intrinsics", "rotation", "translation"},
            "Hypothesis camera",
        )
        for value in camera.values():
            try:
                items = np.asarray(value, dtype=object).flat
                if any(
                    isinstance(item, bool) or not isinstance(item, Real)
                    for item in items
                ):
                    raise ValueError(
                        "Hypothesis camera requires finite real numeric values"
                    )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Hypothesis camera requires finite real numeric values"
                ) from exc
        return cls(
            HypothesisParents.from_record(values["parents"]),
            HypothesisModel.from_record(values["model"]),
            CameraProjection(**camera),
            HypothesisCoordinateMap.from_record(values["mapping"]),
            HypothesisGauge.from_record(values["gauge"]),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": HYPOTHESIS_REQUEST_SCHEMA,
            "parents": self.parents.to_record(),
            "model": self.model.to_record(),
            "camera": {
                key: getattr(self.camera, key).tolist()
                for key in ("intrinsics", "rotation", "translation")
            },
            "mapping": self.mapping.to_record(),
            "gauge": self.gauge.to_record(),
        }
