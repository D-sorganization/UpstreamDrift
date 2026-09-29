"""Immutable player placement revisions; geometry remains owned by Tools.

These are capture workspace records, not a competing camera model or numerical
qualification. A changed manual scene identifier means the cameras moved.
Signatures detect declared metadata changes, not unreported physical changes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from shared.python.sidekick.lab.mocap.reference_placements import (
    PlacementObservation,
    ReferenceTarget,
)

from src.motion_capture.rig.documents import write_document

from ..calibration_profiles import CalibrationProfile, CameraSetup

Label = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
MAX_REVISION_BYTES = 4 * 1024 * 1024


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class CameraSelection(_Record):
    """A rig view and its optional existing intrinsic profile selection."""

    view: Label
    setup: CameraSetup
    profile: CalibrationProfile | None = None

    @model_validator(mode="after")
    def _compatible(self) -> CameraSelection:
        if self.profile is not None and self.profile.setup != self.setup:
            raise ValueError("Selected intrinsic profile has different camera settings")
        return self

    @property
    def profile_key(self) -> str:
        return self.profile.profile_id if self.profile is not None else "unreviewed"

    def signature(self, scene_id: str) -> str:
        """Bind marked frames to declared optics and one stationary camera scene."""
        payload = {
            "view": self.view,
            "setup": self.setup.model_dump(mode="json"),
            "profile": self.profile_key,
            "calibration_sha256": self.profile.calibration_sha256
            if self.profile
            else None,
            "scene": scene_id,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ReferenceSample(_Record):
    """An operator-marked frame with unchanged canonical point identities."""

    observation: PlacementObservation
    camera_signature: Digest
    source_frame: str = Field(min_length=1, max_length=500)
    source_sha256: Digest
    enabled: bool = Field(default=True, strict=True)
    notes: str = Field(default="", max_length=4000)

    @field_validator("source_frame")
    @classmethod
    def _relative_frame(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or PureWindowsPath(value).drive
            or "\\" in value
            or ".." in path.parts
            or ":" in value
            or not path.name
        ):
            raise ValueError("Use a portable relative frame path within the workspace")
        return value


class ReferenceSession(_Record):
    """One editable placement set saved as a new immutable revision each time."""

    schema_version: Literal["capture-reference-session/1"] = (
        "capture-reference-session/1"
    )
    revision_id: UUID = Field(default_factory=uuid4)
    parent_revision_id: UUID | None = None
    capture_id: Label
    title: Label
    scene_id: Label
    created_utc: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    cameras: tuple[CameraSelection, ...] = Field(min_length=1, max_length=12)
    targets: tuple[ReferenceTarget, ...] = Field(default=(), max_length=32)
    samples: tuple[ReferenceSample, ...] = Field(default=(), max_length=256)
    anchor_placement_id: Label | None = None
    notes: str = Field(default="", max_length=8000)

    @model_validator(mode="after")
    def _bindings(self) -> ReferenceSession:
        cameras = {camera.view: camera for camera in self.cameras}
        identities = {camera.setup.camera_identity for camera in self.cameras}
        targets = {target.reference_id: target for target in self.targets}
        if len(cameras) != len(self.cameras) or len(identities) != len(self.cameras):
            raise ValueError("Each view must identify a different physical camera")
        if len(targets) != len(self.targets):
            raise ValueError("Reference target IDs must be unique")
        seen: set[tuple[str, str]] = set()
        placements: dict[str, str] = {}
        for sample in self.samples:
            self._validate_sample(sample, cameras, targets, seen, placements)
        if (
            self.anchor_placement_id is not None
            and self.anchor_placement_id not in placements
        ):
            raise ValueError("The world anchor must identify an observed placement")
        return self

    def _validate_sample(
        self,
        sample: ReferenceSample,
        cameras: dict[str, CameraSelection],
        targets: dict[str, ReferenceTarget],
        seen: set[tuple[str, str]],
        placements: dict[str, str],
    ) -> None:
        observation = sample.observation
        camera = cameras.get(observation.camera_key)
        target = targets.get(observation.reference_id)
        if camera is None or target is None:
            raise ValueError("Sample references an unknown camera or target")
        if sample.camera_signature != camera.signature(self.scene_id):
            raise ValueError(
                "Sample camera settings or scene changed; mark a new frame"
            )
        if observation.profile_id != camera.profile_key:
            raise ValueError("Sample belongs to a different intrinsic profile")
        pair = (observation.placement_id, observation.camera_key)
        if pair in seen:
            raise ValueError("A camera can have only one sample per physical placement")
        seen.add(pair)
        previous = placements.setdefault(
            observation.placement_id, observation.reference_id
        )
        if previous != observation.reference_id:
            raise ValueError("All cameras must see the same target at one placement")
        observation.calibration_observation(target)
        width, height = camera.setup.image_size_px
        if any(
            not (0 <= x < width and 0 <= y < height) for x, y in observation.pixels_px
        ):
            raise ValueError("Marked points must lie inside the original camera image")

    @property
    def unreviewed_views(self) -> tuple[str, ...]:
        """Views lacking a profile; selected profiles still require live validation."""
        return tuple(camera.view for camera in self.cameras if camera.profile is None)

    def revise(self, **changes: Any) -> ReferenceSession:
        """Validate edits as a new revision, retaining its immediate predecessor."""
        payload = self.model_dump()
        payload.update(changes)
        payload.update(
            revision_id=uuid4(),
            parent_revision_id=self.revision_id,
            created_utc=datetime.now(UTC),
        )
        return ReferenceSession.model_validate(payload)


def load_revision(path: Path, *, capture_id: str) -> ReferenceSession:
    """Bound reads and reject another capture's saved observations."""
    with path.open("rb") as stream:
        data = stream.read(MAX_REVISION_BYTES + 1)
    if len(data) > MAX_REVISION_BYTES:
        raise ValueError("Reference revision exceeds the workspace document limit")
    session = ReferenceSession.model_validate_json(data)
    if session.capture_id != capture_id:
        raise ValueError("Reference revision belongs to another capture")
    return session


def save_revision(root: Path, session: ReferenceSession, *, capture_id: str) -> Path:
    """Save a new UUID revision through the existing atomic sidecar writer.

    The operator dialog is a single writer. Earlier revision paths never change;
    repeated saves are idempotent and conflicting or corrupt files are retained.
    Frame bytes are verified separately before inspection or numerical use.
    """
    session = ReferenceSession.model_validate(session.model_dump())
    if session.capture_id != capture_id:
        raise ValueError("Cannot save reference observations to another capture")
    folder = root / "reference_calibration"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{session.revision_id}.json"
    if target.exists():
        if load_revision(target, capture_id=capture_id) != session:
            raise ValueError("Cannot replace an existing revision")
        return target
    payload = session.model_dump(mode="json")
    encoded = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode()
    if len(encoded) > MAX_REVISION_BYTES:
        raise ValueError("Reference revision exceeds the workspace document limit")
    write_document(target, payload)
    return target
