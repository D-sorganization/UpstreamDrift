"""Versioned reference assets, separate from observed player landmarks."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

Axis = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
Point3 = tuple[float, float, float]
MAX_REFERENCE_BYTES = 64_000_000
MAX_SAMPLES = 1_000_000


class ReferenceSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    path: str = Field(min_length=1, max_length=4096)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["c3d", "body_target_json_v1", "video", "marker-trajectory/1.0.0"]


class ReferenceAsset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    schema_version: Literal["reference-asset/1.0.0"] = "reference-asset/1.0.0"
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(min_length=1, max_length=200)
    notes: str = Field(default="", max_length=50_000)
    archived: bool = Field(default=False, strict=True)
    source: ReferenceSource
    model_identity: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def valid_identity(self) -> Self:
        if str(UUID(self.id)) != self.id:
            raise ValueError("Reference identity must be a canonical UUID")
        if not self.title.strip() or any(ord(c) < 32 for c in self.title):
            raise ValueError("Reference title needs visible text on one line")
        return self

    def changed(self, **values: object) -> Self:
        return type(self).model_validate(self.model_dump() | values)

    @property
    def source_path(self) -> Path:
        return Path(self.source.path)


class ReferenceMotion(ReferenceAsset):
    """Metres in a right-handed Z-up frame; None is an unobserved marker.

    source_axes lists the signed source axes assigned to canonical X, Y, Z.
    source_names and joint_names retain the explicit one-to-one name mapping.
    Registration into a particular camera rig is a separate document.
    """

    kind: Literal["motion"] = "motion"
    units: Literal["m"] = "m"
    coordinate_frame: Literal["z_up_right_handed"] = "z_up_right_handed"
    source_units: Literal["m", "cm", "mm"]
    source_axes: tuple[Axis, Axis, Axis]
    source_names: tuple[str, ...] = Field(min_length=1, max_length=256)
    joint_names: tuple[str, ...] = Field(min_length=1, max_length=256)
    edges: tuple[tuple[int, int], ...] = Field(default=(), max_length=1024)
    club_edges: tuple[tuple[int, int], ...] = Field(default=(), max_length=64)
    time_s: tuple[float, ...] = Field(min_length=1, max_length=100_000)
    points_m: tuple[tuple[Point3 | None, ...], ...] = Field(
        min_length=1, max_length=100_000
    )

    @model_validator(mode="after")
    def valid_motion(self) -> Self:
        count = len(self.joint_names)
        if len(self.source_names) != count:
            raise ValueError("Each joint needs its original source name")
        for names in (self.source_names, self.joint_names):
            if len(set(names)) != len(names) or any(
                not n.strip() or len(n) > 200 for n in names
            ):
                raise ValueError(
                    "Joint/source names must be unique, non-empty and at most 200 characters"
                )
        if len({axis[-1] for axis in self.source_axes}) != 3:
            raise ValueError("Source axes must use X, Y and Z exactly once")
        if any(b <= a for a, b in pairwise(self.time_s)):
            raise ValueError("Reference timestamps must be strictly increasing")
        if len(self.time_s) != len(self.points_m) or any(
            len(row) != count for row in self.points_m
        ):
            raise ValueError(
                "Reference point dimensions must match timestamps and joint names"
            )
        if len(self.time_s) * count > MAX_SAMPLES:
            raise ValueError(
                "Reference exceeds one million joint samples; trim the source first"
            )
        if any(a == b or min(a, b) < 0 or max(a, b) >= count for a, b in self.edges):
            raise ValueError("Skeleton edges must connect two distinct existing joints")
        if len({tuple(sorted(edge)) for edge in self.edges}) != len(self.edges):
            raise ValueError("Skeleton edges must be unique")
        if not set(self.club_edges).issubset(self.edges) or len(
            set(self.club_edges)
        ) != len(self.club_edges):
            raise ValueError("Club edges must be unique members of skeleton edges")
        if not any(point is not None for row in self.points_m for point in row):
            raise ValueError("Reference contains no observed points")
        return self


class ReferenceVideo(ReferenceAsset):
    """A linked 2D recording; no camera viewpoint or 3D geometry is inferred."""

    kind: Literal["video"] = "video"
    clock: Literal["nominal-frame-rate"] = "nominal-frame-rate"
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    frames: int = Field(gt=0, strict=True)
    fps: float = Field(gt=0)


Asset = ReferenceMotion | ReferenceVideo
ASSET_ADAPTER: TypeAdapter[Asset] = TypeAdapter(
    Annotated[Asset, Field(discriminator="kind")]
)
