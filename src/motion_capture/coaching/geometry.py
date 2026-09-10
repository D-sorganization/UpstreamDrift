"""Saved metric scene references shared by camera and model analysis.

Coordinates are world metres, bound to a scene identity. Screen drawings remain
in ``drawing.py``; no image pixel is implicitly promoted to a metric point.
"""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.motion_capture.rig.documents import write_document

Vector3 = tuple[float, float, float]
MAX_GEOMETRY_BYTES = 2_000_000


class SceneReference(BaseModel):
    """Immutable appearance and inclusive visibility on the scene clock."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=100)
    title: str = Field(default="Reference", min_length=1, max_length=200)
    colour: str = Field(default="#ffcc33", pattern=r"^#[0-9a-fA-F]{6}$")
    opacity: float = Field(default=0.3, ge=0, le=1)
    visible: bool = True
    first_s: float | None = None
    last_s: float | None = None

    @model_validator(mode="after")
    def valid_interval(self) -> Self:
        if (
            self.first_s is not None
            and self.last_s is not None
            and self.last_s < self.first_s
        ):
            raise ValueError("Visibility ends before it starts")
        return self

    def at(self, time_s: float) -> bool:
        """Return visibility at a finite scene time, including interval endpoints."""
        if not isfinite(time_s):
            raise ValueError("Scene time must be finite")
        return (
            self.visible
            and (self.first_s is None or time_s >= self.first_s)
            and (self.last_s is None or time_s <= self.last_s)
        )

    def changed(self, **values: object) -> Self:
        """Return a validated immutable edit, never bypassing geometry contracts."""
        return type(self).model_validate(self.model_dump() | values)


class ReferencePoint(SceneReference):
    """A named fixed point in the owning scene's world frame, in metres."""

    position_m: Vector3


class ReferencePlane(SceneReference):
    """An oriented plane through three non-collinear metric anchor points.

    The normal follows ``(along-origin) cross (across-origin)``. Extent controls
    only the displayed square; signed distances refer to the infinite plane.
    """

    origin_m: Vector3
    along_m: Vector3
    across_m: Vector3
    half_size_m: float = Field(default=1, gt=0, le=1000)

    @model_validator(mode="after")
    def valid_plane(self) -> Self:
        self._basis()
        return self

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        along = np.subtract(self.along_m, self.origin_m)
        across = np.subtract(self.across_m, self.origin_m)
        lengths = float(np.linalg.norm(along)), float(np.linalg.norm(across))
        if min(lengths) < 1e-9 or not np.isfinite(lengths).all():
            raise ValueError("Plane anchors must be distinct finite points")
        x = along / lengths[0]
        normal = np.cross(x, across / lengths[1])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            raise ValueError("Plane anchors must not be collinear")
        normal /= norm
        return x, np.cross(normal, x), normal

    def vertices(self) -> npt.NDArray[np.float64]:
        """Return four world-metre corners using the suite's shared plane mesh."""
        from src.shared.python.biomechanics.swing_plane_visualization import (
            generate_plane_vertices,
        )

        x, y, normal = self._basis()
        return generate_plane_vertices(
            np.asarray(self.origin_m), normal, x, y, self.half_size_m
        )

    def distances(self, points_m: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return signed metres for (..., 3) points; NaN gaps remain missing."""
        points = np.asarray(points_m, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3 or np.isinf(points).any():
            raise ValueError("Points must have shape (..., 3), with no infinities")
        _, _, normal = self._basis()
        return np.asarray((points - self.origin_m) @ normal, dtype=float)


class ReferenceGeometry(BaseModel):
    """A portable scene-bound geometry document, independent of any viewport."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    schema_version: Literal["analysis-geometry/1.0.0"] = "analysis-geometry/1.0.0"
    scene_id: str = Field(min_length=1, max_length=200)
    frame: Literal["world"] = "world"
    units: Literal["m"] = "m"
    planes: tuple[ReferencePlane, ...] = Field(default=(), max_length=200)
    points: tuple[ReferencePoint, ...] = Field(default=(), max_length=2000)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        ids = [item.id for item in (*self.planes, *self.points)]
        if len(ids) != len(set(ids)):
            raise ValueError("Scene reference IDs must be unique")
        return self

    def save(self, path: Path) -> None:
        """Atomically replace a bounded JSON sidecar after full validation."""
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
        if len(encoded.encode("utf-8")) + 1 > MAX_GEOMETRY_BYTES:
            raise ValueError("Reference geometry exceeds the document size limit")
        write_document(path, payload)

    @classmethod
    def load(cls, path: Path, *, scene_id: str) -> Self:
        """Load bounded data and reject geometry belonging to another scene."""
        with path.open("rb") as stream:
            content = stream.read(MAX_GEOMETRY_BYTES + 1)
        if len(content) > MAX_GEOMETRY_BYTES:
            raise ValueError("Reference geometry exceeds the document size limit")
        result = cls.model_validate_json(content)
        if result.scene_id != scene_id:
            raise ValueError("Reference geometry belongs to a different scene")
        return result
