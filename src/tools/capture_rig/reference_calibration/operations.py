"""Worker-only adapters from capture media and point edits to canonical records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.python.sidekick.lab.mocap.reference_placements import (
    PlacementObservation,
    ReferenceTarget,
)
from src.motion_capture.rig.capture_notes import read_notes

from .frames import archive_frame, load_frame, read_frame_record, verify_frame
from .session import ReferenceSample, ReferenceSession


def verify_workspace(root: Path, capture_id: str) -> None:
    """Use the existing library identity as the capture ownership authority."""
    if read_notes(root).capture_id != capture_id:
        raise ValueError("Reference workspace belongs to another capture")


def add_target(request: dict[str, Any]) -> dict[str, Any]:
    """Retain measured dimensions without replacing earlier reference identities."""
    session = ReferenceSession.model_validate(request["session"])
    parameters = request["parameters"]
    key = parameters["reference_id"]
    if (
        not isinstance(key, str)
        or not key.startswith("measured-")
        or not key[9:].strip()
    ):
        raise ValueError("Give the measured reference a distinct name")
    if any(target.reference_id == key for target in session.targets):
        raise ValueError(
            "That reference name already exists; use a new name to preserve its observations"
        )
    if parameters["shape"] == "line":
        target = ReferenceTarget.line(key, parameters["length_m"])
    elif parameters["shape"] == "rectangle":
        target = ReferenceTarget.rectangle(
            key, parameters["width_m"], parameters["length_m"]
        )
    else:
        raise ValueError("Choose a measured line or rectangle")
    return session.revise(targets=(*session.targets, target)).model_dump(mode="json")


def extract_frame(request: dict[str, Any]) -> dict[str, Any]:
    """Decode the original capture recording off the UI thread and archive it."""
    from ..player import VideoReader
    from ..session import load_session

    root = Path(request["workspace"])
    session = ReferenceSession.model_validate(request["session"])
    verify_workspace(root, session.capture_id)
    params = request["parameters"]
    view = params["view"]
    selection = next(
        (camera for camera in session.cameras if camera.view == view), None
    )
    if selection is None:
        raise ValueError("Choose a camera from this reference session")
    media = load_session(root).view(view)
    if media.recording is None:
        raise ValueError(
            "Original camera recording is unavailable; restore it from the library"
        )
    index = params["frame_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Choose a nonnegative original frame number")
    with VideoReader(media.recording) as reader:
        if (reader.width, reader.height) != selection.setup.image_size_px:
            raise ValueError(
                "Original recording size differs from the declared camera setup"
            )
        if index >= reader.frame_count or reader.fps <= 0:
            raise ValueError(
                "Choose an available frame in a recording with known frame rate"
            )
        frame = reader.read(index)
        if frame is None:
            raise ValueError(
                "Could not decode this frame; choose another frame or repair the recording"
            )
        saved = archive_frame(
            root,
            frame,
            capture_id=session.capture_id,
            view=view,
            frame_index=index,
            timestamp_s=index / reader.fps,
            source_label=media.recording.name,
        )
        return {
            "frame": saved.model_dump(mode="json"),
            "frame_count": reader.frame_count,
        }


def mark_sample(request: dict[str, Any]) -> dict[str, Any]:
    """Replace one view's placement as a new revision, preserving physical IDs."""
    root = Path(request["workspace"])
    session = ReferenceSession.model_validate(request["session"])
    verify_workspace(root, session.capture_id)
    params = request["parameters"]
    record = read_frame_record(
        root, params["frame_path"], capture_id=session.capture_id
    )
    frame = load_frame(
        root, record.path, params["frame_sha256"], capture_id=session.capture_id
    )
    camera = next((item for item in session.cameras if item.view == record.view), None)
    if camera is None or camera.setup.image_size_px != (frame.shape[1], frame.shape[0]):
        raise ValueError("Archived frame does not match the selected camera")
    target = next(
        (
            item
            for item in session.targets
            if item.reference_id == params["reference_id"]
        ),
        None,
    )
    if target is None:
        raise ValueError("Choose a reference target from this session")
    points = params["points"]
    if (
        not isinstance(points, dict)
        or not points
        or set(points) - set(target.point_ids)
    ):
        raise ValueError("Mark at least one identified physical point")
    identifiers = tuple(identity for identity in target.point_ids if identity in points)
    observation = PlacementObservation(
        params["placement_id"],
        target.reference_id,
        camera.view,
        camera.profile_key,
        identifiers,
        tuple(tuple(points[identity]) for identity in identifiers),
        record.frame_index,
        round(record.timestamp_s * 1_000_000_000),
        held_out=params.get("held_out", False),
    )
    sample = ReferenceSample(
        observation=observation,
        camera_signature=camera.signature(session.scene_id),
        source_frame=record.path,
        source_sha256=record.sha256,
        notes=params.get("notes", ""),
    )
    retained = tuple(
        item
        for item in session.samples
        if (item.observation.placement_id, item.observation.camera_key)
        != (observation.placement_id, observation.camera_key)
    )
    return session.revise(samples=(*retained, sample)).model_dump(mode="json")


def verify_samples(root: Path, session: ReferenceSession) -> None:
    """Check original frames and metadata before saving or consuming observations."""
    verify_workspace(root, session.capture_id)
    for sample in session.samples:
        record = verify_frame(
            root,
            sample.source_frame,
            sample.source_sha256,
            capture_id=session.capture_id,
        )
        observation = sample.observation
        if (
            record.view != observation.camera_key
            or record.frame_index != observation.frame_sequence
            or round(record.timestamp_s * 1_000_000_000) != observation.timestamp_ns
        ):
            raise ValueError(
                "Observation no longer matches its original frame metadata"
            )
