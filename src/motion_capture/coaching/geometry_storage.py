"""Portable capture geometry with conservative scene-evidence binding."""

from pathlib import Path

from src.motion_capture.provenance import sha256_of
from src.motion_capture.reference.evidence import fingerprint
from src.motion_capture.rig.alignment import TIMING_REPORT_FILE

from .geometry import ReferenceGeometry


def geometry_path(root: Path) -> Path:
    """Return the one world-reference document shared by all capture views."""
    return root / "coaching" / "world.json"


def geometry_scene_id(root: Path) -> str:
    """Bind to recorded source, reconstruction, lens and clock evidence.

    Paths are relative so a byte-identical bundle remains portable. Updating
    reconstruction conservatively requires review, even when camera poses happen
    to remain equal. This function never creates a capture identity or recording.
    """
    recordings = root / "recordings.json"
    if not recordings.is_file():
        raise FileNotFoundError(
            "Capture recordings.json is required for scene references"
        )
    evidence: dict[str, str | None] = {"recordings.json": sha256_of(recordings)}
    for name in (
        "session_manifest.json",
        "intrinsics.json",
        TIMING_REPORT_FILE,
        "reconstruct/session_reconstruction.json",
    ):
        path = root / name
        evidence[name] = sha256_of(path) if path.is_file() else None
    # Camera matrices live in reconstruction.json, separate from the session
    # summary. Omit an absent new key to retain identities of camera-free scenes.
    camera_name = "reconstruct/reconstruction.json"
    camera_path = root / camera_name
    if camera_path.is_file():
        evidence[camera_name] = sha256_of(camera_path)
    return fingerprint(evidence)


def load_geometry(root: Path) -> ReferenceGeometry:
    """Load shared references, rejecting changed capture/calibration evidence."""
    identity = geometry_scene_id(root)
    path = geometry_path(root)
    return (
        ReferenceGeometry.load(path, scene_id=identity)
        if path.is_file()
        else ReferenceGeometry(scene_id=identity)
    )


def save_geometry(root: Path, geometry: ReferenceGeometry) -> None:
    """Save only geometry belonging to the current capture world and clock."""
    if geometry.scene_id != geometry_scene_id(root):
        raise ValueError("Reference geometry belongs to a different or changed scene")
    path = geometry_path(root)
    path.parent.mkdir(exist_ok=True)
    geometry.save(path)
