"""Immutable calibration, source geometry and clock evidence for comparisons."""

import hashlib
import json
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.motion_capture.rig.alignment import reference_time_fields
from src.shared.python.pose_estimation import (
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
)
from .model import Asset, Point3

Matrix3 = tuple[Point3, Point3, Point3]


def fingerprint(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def asset_identity(asset: Asset) -> str:
    """Geometry, source and mapping identity; editable library notes are excluded."""
    values = asset.model_dump(mode="json", exclude={"title", "notes", "archived"})
    # Preserve bindings saved before optional club connectivity was introduced.
    if not values.get("club_edges"):
        values.pop("club_edges", None)
    return fingerprint(values)


class CameraSnapshot(BaseModel):
    """Exact calibration used for rendering; provenance is not an accuracy claim."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    camera_id: str = Field(min_length=1, max_length=200)
    matrix: Matrix3
    rotation_world_from_camera: Matrix3
    translation_world_from_camera_m: Point3
    image_size_px: tuple[int, int]
    distortion: tuple[float, ...] = ()
    provenance: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        self.record()
        if len(self.distortion) not in (0, 4, 5, 8, 12, 14):
            raise ValueError("Unsupported OpenCV distortion coefficient count")
        return self

    def record(self) -> CameraCalibration:
        return CameraCalibration(
            self.camera_id,
            CameraIntrinsics(
                np.asarray(self.matrix),
                np.asarray(self.distortion) if self.distortion else None,
            ),
            CameraExtrinsics(
                np.asarray(self.rotation_world_from_camera),
                np.asarray(self.translation_world_from_camera_m),
            ),
            self.image_size_px,
        )

    @property
    def identity(self) -> str:
        return fingerprint(self.record().to_dict())

    @classmethod
    def from_calibration(cls, record: CameraCalibration, *, provenance: str) -> Self:
        payload = record.to_dict()
        intrinsics = payload["intrinsics"]
        return cls.model_validate(
            {
                "camera_id": payload["camera_id"],
                "image_size_px": payload["image_size_px"],
                "matrix": intrinsics["matrix"],
                "distortion": intrinsics["distortion"] or (),
                **payload["extrinsics"],
                "provenance": provenance,
            }
        )


class ViewClock(BaseModel):
    """Original frame clock to scene clock, with its explicit evidence source."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    view: str = Field(min_length=1, max_length=200)
    offset_ns: int = Field(default=0, strict=True)
    uncertainty_ns: int = Field(default=0, ge=0, strict=True)
    source: str = Field(
        default="nominal-frame-rate-unverified", min_length=1, max_length=500
    )

    def player_time(self, original_time: float) -> float:
        if not np.isfinite(original_time):
            raise ValueError("Original frame time must be finite")
        return float(
            reference_time_fields(
                original_time, self.offset_ns, self.uncertainty_ns, self.source
            )["time_ref_s"]
        )
