"""Player workspace references to existing intrinsic calibration artifacts.

These records select compatible lens settings; they neither solve geometry nor
certify extrinsics or physical accuracy. Calibration remains in its existing
authority. Manual settings require a fresh operator confirmation before reuse.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.motion_capture.reconstruct.intrinsics import IntrinsicsRecord
from src.motion_capture.rig.documents import write_document

_Label = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
]
_Pixel = Annotated[int, Field(strict=True, gt=0)]


class CameraSetup(BaseModel):
    """Player-declared optical configuration, distinct from a camera contract.

    Zoom and focus are opaque setting labels: manual lens markings are not
    converted into invented focal lengths. Sensor mode includes digital crop.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    camera_identity: _Label
    lens: _Label
    zoom: _Label
    focus: _Label
    image_size_px: tuple[_Pixel, _Pixel]
    sensor_mode: _Label


def _record(data: bytes, camera_id: str, size: tuple[int, int]) -> IntrinsicsRecord:
    payload = json.loads(data)
    if isinstance(payload, dict):
        payload = payload.get("cameras")
    if not isinstance(payload, list):
        raise ValueError("Expected the existing intrinsics calibration list")
    records = [IntrinsicsRecord.model_validate(item) for item in payload]
    matching = [record for record in records if record.camera_id == camera_id]
    if len(matching) != 1:
        raise ValueError("Expected exactly one matching camera in the calibration")
    record = matching[0]
    if record.image_size_px != size:
        raise ValueError("Calibration image size does not match the camera setup")
    if not record.ok or not np.isfinite(record.rms_px) or record.rms_px < 0:
        raise ValueError("Calibration does not meet the existing quality threshold")
    matrix = np.asarray(record.matrix, dtype=float)
    distortion = np.asarray(record.distortion, dtype=float)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[2], [0, 0, 1])
        or distortion.shape not in {(4,), (5,), (8,), (12,), (14,)}
        or not np.isfinite(distortion).all()
    ):
        raise ValueError("Calibration camera matrix or distortion is invalid")
    return record


class CalibrationProfile(BaseModel):
    """Immutable named reference to an existing per-view intrinsic result."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    profile_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    name: _Label
    setup: CameraSetup
    camera_id: _Label
    intrinsics_path: str = Field(min_length=1)
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_intrinsics_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def capture(
        cls,
        *,
        name: str,
        setup: CameraSetup,
        camera_id: str,
        intrinsics_path: Path,
    ) -> CalibrationProfile:
        """Bind settings to validated bytes, never to a mutable filename alone."""
        path = intrinsics_path.resolve(strict=True)
        data = path.read_bytes()
        _record(data, camera_id, setup.image_size_px)
        return cls(
            name=name,
            setup=setup,
            camera_id=camera_id,
            intrinsics_path=str(path),
            calibration_sha256=hashlib.sha256(data).hexdigest(),
            source_intrinsics_path=str(path),
            source_sha256=hashlib.sha256(data).hexdigest(),
        )


@dataclass(frozen=True)
class ProfileCompatibility:
    """Only intrinsic reuse compatibility; no scene-pose qualification claim."""

    reasons: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.reasons


def check_profile(
    profile: CalibrationProfile,
    current_setup: CameraSetup,
    *,
    settings_confirmed: bool,
) -> ProfileCompatibility:
    """Explain why a saved intrinsic profile cannot be reused for this setup."""
    reasons: list[str] = []
    if settings_confirmed is not True:
        reasons.append(
            "Confirm the current lens, zoom and focus settings before reuse."
        )
    labels = {
        "camera_identity": "Camera identity",
        "lens": "Lens",
        "zoom": "Optical zoom",
        "focus": "Focus",
        "image_size_px": "Image size",
        "sensor_mode": "Sensor or crop mode",
    }
    for field, label in labels.items():
        if getattr(profile.setup, field) != getattr(current_setup, field):
            reasons.append(
                f"{label} changed; select a compatible profile or recalibrate."
            )
    try:
        data = Path(profile.intrinsics_path).read_bytes()
    except OSError:
        reasons.append("Calibration file unavailable; locate it or recalibrate.")
    else:
        if hashlib.sha256(data).hexdigest() != profile.calibration_sha256:
            reasons.append("Calibration file changed; save a new profile after review.")
        else:
            try:
                _record(data, profile.camera_id, profile.setup.image_size_px)
            except (ValueError, TypeError):
                reasons.append("Calibration file is invalid; recalibrate before reuse.")
    return ProfileCompatibility(tuple(reasons))


class ProfileHistory(BaseModel):
    """Append revisions without silently rewriting a previous capture's setup."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["capture-calibration-profiles/1.0.0"] = (
        "capture-calibration-profiles/1.0.0"
    )
    profiles: tuple[CalibrationProfile, ...] = ()
    active_profile_id: str | None = None

    @model_validator(mode="after")
    def _valid_selection(self) -> ProfileHistory:
        identifiers = [profile.profile_id for profile in self.profiles]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Calibration profile IDs must be unique")
        if (
            self.active_profile_id is not None
            and self.active_profile_id not in identifiers
        ):
            raise ValueError("Unknown active calibration profile")
        return self

    def select(self, profile_id: str) -> ProfileHistory:
        """Restore a selection; consumers must still check its compatibility."""
        return ProfileHistory(profiles=self.profiles, active_profile_id=profile_id)


def _archive_profile(path: Path, profile: CalibrationProfile) -> CalibrationProfile:
    """Preserve the existing calibration record under the library's ownership."""
    data = Path(profile.intrinsics_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != profile.calibration_sha256:
        raise ValueError("Calibration changed before the profile could be saved")
    _record(data, profile.camera_id, profile.setup.image_size_px)
    payload = json.loads(data)
    # The existing reconstruction reader accepts this canonical camera wrapper.
    payload = {"cameras": payload["cameras"] if isinstance(payload, dict) else payload}
    archive = path.parent / "calibration_revisions"
    archive.mkdir(exist_ok=True)
    target = archive / f"{profile.source_sha256}.json"
    expected = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")
    if target.exists():
        if target.read_bytes() != expected:
            raise ValueError(
                "Saved calibration revision changed; refusing to overwrite it"
            )
    else:
        write_document(target, payload)
    return profile.model_copy(
        update={
            "intrinsics_path": str(target.resolve()),
            "calibration_sha256": hashlib.sha256(expected).hexdigest(),
        }
    )


def save_profile(path: Path, profile: CalibrationProfile) -> CalibrationProfile:
    """Atomically append/select a revision in the single-writer UI history.

    A malformed existing file is an error, never permission to replace history.
    Identical saves are idempotent; conflicting IDs cannot rewrite a revision.
    """
    history = (
        ProfileHistory.model_validate_json(path.read_bytes())
        if path.exists()
        else ProfileHistory()
    )
    existing = next(
        (item for item in history.profiles if item.profile_id == profile.profile_id),
        None,
    )
    if existing is not None:
        # Source provenance and identity remain fixed when the artifact is archived.
        excluded = {"intrinsics_path", "calibration_sha256"}
        if existing.model_dump(exclude=excluded) != profile.model_dump(
            exclude=excluded
        ):
            raise ValueError(
                "Cannot overwrite an existing calibration profile revision"
            )
        saved = existing
    else:
        saved = _archive_profile(path, profile)
    profiles = history.profiles if existing else (*history.profiles, saved)
    updated = ProfileHistory(profiles=profiles, active_profile_id=profile.profile_id)
    write_document(path, updated.model_dump(mode="json"))
    return saved


@dataclass(frozen=True)
class ProfileAssignment:
    """A reviewed intrinsic revision assigned to one current rig view."""

    view: str
    profile: CalibrationProfile
    setup: CameraSetup
    settings_confirmed: bool


def validate_profile_set(
    data: bytes, expected: Mapping[str, tuple[str, tuple[int, int]]]
) -> None:
    """Recheck a reviewed export against recorded camera identities and sizes.

    This validates intrinsic reuse metadata, not physical camera placement.
    The caller must still obtain current operator confirmation of lens settings.
    """
    payload = json.loads(data)
    selections = (
        payload.get("profile_selections") if isinstance(payload, dict) else None
    )
    if (
        not isinstance(selections, list)
        or len(selections) != len(expected)
        or not expected
    ):
        raise ValueError("Select a reviewed calibration profile for every camera")
    seen: set[str] = set()
    for item in selections:
        if not isinstance(item, dict):
            raise ValueError("Invalid calibration profile selection")
        view = item.get("view")
        if not isinstance(view, str) or view not in expected or view in seen:
            raise ValueError("Calibration profile views do not match this capture")
        seen.add(view)
        setup = CameraSetup.model_validate(item.get("setup"))
        if (setup.camera_identity, setup.image_size_px) != expected[view]:
            raise ValueError(f"{view}: camera identity or recorded image size changed")
        confirmed = item.get("confirmed_utc")
        if (
            not isinstance(confirmed, str)
            or datetime.fromisoformat(confirmed).utcoffset() is None
        ):
            raise ValueError(
                "Calibration selection needs a dated operator confirmation"
            )
        _record(data, view, setup.image_size_px)


def write_profile_set(
    path: Path,
    assignments: Sequence[ProfileAssignment],
    *,
    required_views: Sequence[str],
) -> None:
    """Export only verified cameras, atomically, after checking the whole rig."""
    views = [item.view for item in assignments]
    if (
        not required_views
        or any(not view.strip() for view in views)
        or len(views) != len(set(views))
        or set(views) != set(required_views)
    ):
        raise ValueError("Select one calibration for every view in the rig")
    identities = [item.setup.camera_identity for item in assignments]
    if len(identities) != len(set(identities)):
        raise ValueError("Each view must identify a different physical camera")
    cameras, selections = [], []
    for item in assignments:
        compatibility = check_profile(
            item.profile, item.setup, settings_confirmed=item.settings_confirmed
        )
        if not compatibility.compatible:
            raise ValueError(f"{item.view}: {' '.join(compatibility.reasons)}")
        data = Path(item.profile.intrinsics_path).read_bytes()
        if hashlib.sha256(data).hexdigest() != item.profile.calibration_sha256:
            raise ValueError("Calibration changed during export; review it again")
        record = _record(data, item.profile.camera_id, item.setup.image_size_px)
        camera = record.model_dump(mode="json")
        camera["camera_id"] = item.view
        cameras.append(camera)
        selections.append(
            {
                "view": item.view,
                "profile_id": item.profile.profile_id,
                "calibration_sha256": item.profile.calibration_sha256,
                "setup": item.setup.model_dump(mode="json"),
                "confirmed_utc": datetime.now(UTC).isoformat(),
            }
        )
    write_document(path, {"cameras": cameras, "profile_selections": selections})
