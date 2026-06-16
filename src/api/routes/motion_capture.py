"""Motion Capture tool API routes.

Provides REST endpoints for the Motion Capture tool page:
- Capture source enumeration (C3D, OpenPose, MediaPipe)
- Skeleton data retrieval
- Recording/playback control
- Frame-by-frame joint data

See issue #1206
"""

from __future__ import annotations

import importlib.util
import logging
import math
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.api.middleware.upload_limits import write_upload_file_to_path
from src.shared.python.core.contracts import precondition

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/motion-capture", tags=["motion-capture"])


# ── Request / Response Models ──


class CaptureSource(BaseModel):
    """Available motion capture source."""

    id: str
    name: str
    type: str = Field(description="c3d, openpose, or mediapipe")
    available: bool
    reason: str | None = Field(
        None, description="Why the source is unavailable (None when available)"
    )
    description: str


class JointData(BaseModel):
    """Single joint position and metadata."""

    name: str
    position: list[float] = Field(description="[x, y, z] position")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    parent: str | None = None


class SkeletonFrame(BaseModel):
    """One frame of skeleton data."""

    frame_index: int
    timestamp: float
    joints: list[JointData]


class RecordingInfo(BaseModel):
    """Metadata about a motion capture recording."""

    name: str
    source_type: str
    total_frames: int
    duration_seconds: float
    frame_rate: float
    joint_names: list[str]


class CaptureSessionRequest(BaseModel):
    """Request to start a capture session."""

    source_type: str = Field(
        "mediapipe", description="Capture source: c3d, openpose, mediapipe"
    )
    frame_rate: float = Field(30.0, description="Target frame rate", gt=0)


class CaptureSessionResponse(BaseModel):
    """Response after starting/stopping a capture session."""

    session_id: str
    status: str
    source_type: str
    message: str


class PlaybackRequest(BaseModel):
    """Request for recording playback control."""

    recording_name: str
    action: str = Field(description="play, pause, stop, seek")
    seek_frame: int | None = Field(None, description="Frame to seek to")


class PlaybackResponse(BaseModel):
    """Response with current playback state."""

    recording_name: str
    status: str
    current_frame: int
    total_frames: int


class C3DUploadResponse(BaseModel):
    """Metadata extracted from an uploaded C3D file.

    Marker positions are converted to meters server-side (mirroring the
    desktop C3D viewer's ``target_units="m"`` handling) so the web
    visualizer never has to guess mm-vs-m scaling.
    """

    recording_name: str
    marker_names: list[str]
    frame_rate: float
    total_frames: int
    duration_seconds: float
    native_units: str = Field(
        description="POINT units declared in the file ('' when absent)"
    )
    converted_units: str = Field(
        "m", description="Units of the stored marker positions"
    )


# ── Skeleton definitions ──

_MEDIAPIPE_SKELETON: list[dict[str, Any]] = [
    {"name": "nose", "parent": None},
    {"name": "left_eye", "parent": "nose"},
    {"name": "right_eye", "parent": "nose"},
    {"name": "left_ear", "parent": "left_eye"},
    {"name": "right_ear", "parent": "right_eye"},
    {"name": "left_shoulder", "parent": "nose"},
    {"name": "right_shoulder", "parent": "nose"},
    {"name": "left_elbow", "parent": "left_shoulder"},
    {"name": "right_elbow", "parent": "right_shoulder"},
    {"name": "left_wrist", "parent": "left_elbow"},
    {"name": "right_wrist", "parent": "right_elbow"},
    {"name": "left_hip", "parent": "left_shoulder"},
    {"name": "right_hip", "parent": "right_shoulder"},
    {"name": "left_knee", "parent": "left_hip"},
    {"name": "right_knee", "parent": "right_hip"},
    {"name": "left_ankle", "parent": "left_knee"},
    {"name": "right_ankle", "parent": "right_knee"},
]

_OPENPOSE_SKELETON: list[dict[str, Any]] = [
    {"name": "head", "parent": None},
    {"name": "neck", "parent": "head"},
    {"name": "right_shoulder", "parent": "neck"},
    {"name": "right_elbow", "parent": "right_shoulder"},
    {"name": "right_wrist", "parent": "right_elbow"},
    {"name": "left_shoulder", "parent": "neck"},
    {"name": "left_elbow", "parent": "left_shoulder"},
    {"name": "left_wrist", "parent": "left_elbow"},
    {"name": "mid_hip", "parent": "neck"},
    {"name": "right_hip", "parent": "mid_hip"},
    {"name": "right_knee", "parent": "right_hip"},
    {"name": "right_ankle", "parent": "right_knee"},
    {"name": "left_hip", "parent": "mid_hip"},
    {"name": "left_knee", "parent": "left_hip"},
    {"name": "left_ankle", "parent": "left_knee"},
]

# ── In-memory session state (mutable holder avoids 'global') ──

_sessions: dict[str, dict[str, Any]] = {}
_recordings: dict[str, dict[str, Any]] = {}
_session_state: dict[str, int] = {"counter": 0}


# ── Endpoints ──


@router.get("/sources", response_model=list[CaptureSource])
async def list_capture_sources() -> list[CaptureSource]:
    """List available motion capture sources with honest availability.

    Availability is probed server-side (importability of the backing
    package); unavailable sources carry a human-readable ``reason``.

    See issues #1206, #7454
    """
    sources = []
    for source_id, name, description in (
        (
            "mediapipe",
            "MediaPipe Pose",
            "Real-time pose estimation using Google MediaPipe",
        ),
        (
            "openpose",
            "OpenPose",
            "Multi-person pose estimation using OpenPose",
        ),
        (
            "c3d",
            "C3D File Import",
            "Import motion capture data from C3D files",
        ),
    ):
        available, reason = _source_availability(source_id)
        sources.append(
            CaptureSource(
                id=source_id,
                name=name,
                type=source_id,
                available=available,
                reason=reason,
                description=description,
            )
        )
    return sources


@router.get("/skeleton/{source_type}", response_model=list[JointData])
@precondition(
    lambda source_type: source_type is not None and len(source_type.strip()) > 0,
    "Source type must be a non-empty string",
)
async def get_skeleton_template(source_type: str) -> list[JointData]:
    """Get the skeleton joint template for a given source type.

    See issue #1206
    """
    if source_type == "mediapipe":
        skeleton = _MEDIAPIPE_SKELETON
    elif source_type == "openpose":
        skeleton = _OPENPOSE_SKELETON
    elif source_type == "c3d":
        # C3D has no fixed skeleton: marker sets are defined per-file and
        # become available after upload (issue #7454). An empty template is
        # honest — clients must not assume a MediaPipe-shaped joint set.
        skeleton = []
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source type: {source_type}. Use mediapipe, openpose, or c3d",
        )

    return [
        JointData(
            name=joint["name"],
            position=[0.0, 0.0, 0.0],
            confidence=1.0,
            parent=joint.get("parent"),
        )
        for joint in skeleton
    ]


@router.post("/session/start", response_model=CaptureSessionResponse)
async def start_capture_session(
    request: CaptureSessionRequest,
) -> CaptureSessionResponse:
    """Start a new motion capture session.

    See issue #1206
    """
    valid_sources = {"mediapipe", "openpose", "c3d"}
    if request.source_type not in valid_sources:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source type. Must be one of: {sorted(valid_sources)}",
        )

    available, reason = _source_availability(request.source_type)
    if not available:
        # No silent fallback to another estimator (issue #7454).
        raise HTTPException(
            status_code=409,
            detail=f"Capture source '{request.source_type}' is unavailable: {reason}",
        )

    _session_state["counter"] += 1
    session_id = f"session_{_session_state['counter']}"

    _sessions[session_id] = {
        "source_type": request.source_type,
        "frame_rate": request.frame_rate,
        "status": "recording",
        "frames": [],
    }

    return CaptureSessionResponse(
        session_id=session_id,
        status="recording",
        source_type=request.source_type,
        message=f"Capture session started with {request.source_type} at {request.frame_rate} fps",
    )


@router.post("/session/{session_id}/stop", response_model=CaptureSessionResponse)
@precondition(
    lambda session_id: session_id is not None and len(session_id.strip()) > 0,
    "Session ID must be a non-empty string",
)
async def stop_capture_session(session_id: str) -> CaptureSessionResponse:
    """Stop an active capture session and save the recording.

    See issue #1206
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    session = _sessions[session_id]
    session["status"] = "stopped"

    # Save as a recording
    recording_name = f"recording_{session_id}"
    _recordings[recording_name] = {
        "source_type": session["source_type"],
        "frame_rate": session["frame_rate"],
        "frames": session["frames"],
    }

    return CaptureSessionResponse(
        session_id=session_id,
        status="stopped",
        source_type=session["source_type"],
        message=f"Session stopped. Recording saved as '{recording_name}'",
    )


@router.get("/recordings", response_model=list[RecordingInfo])
async def list_recordings() -> list[RecordingInfo]:
    """List available recordings.

    See issue #1206
    """
    result = []
    for name, rec in _recordings.items():
        frames = rec.get("frames", [])
        frame_rate = rec.get("frame_rate", 30.0)
        total_frames = len(frames)
        duration = total_frames / frame_rate if frame_rate > 0 else 0.0

        joint_names = _recording_joint_names(rec)

        result.append(
            RecordingInfo(
                name=name,
                source_type=rec["source_type"],
                total_frames=total_frames,
                duration_seconds=duration,
                frame_rate=frame_rate,
                joint_names=joint_names,
            )
        )

    return result


@router.post("/playback", response_model=PlaybackResponse)
async def control_playback(request: PlaybackRequest) -> PlaybackResponse:
    """Control recording playback (play, pause, stop, seek).

    See issue #1206
    """
    if request.recording_name not in _recordings:
        raise HTTPException(
            status_code=404,
            detail=f"Recording '{request.recording_name}' not found",
        )

    recording = _recordings[request.recording_name]
    total_frames = len(recording.get("frames", []))

    valid_actions = {"play", "pause", "stop", "seek"}
    if request.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action. Must be one of: {sorted(valid_actions)}",
        )

    current_frame = 0
    if request.action == "seek" and request.seek_frame is not None:
        current_frame = max(0, min(request.seek_frame, total_frames - 1))

    status_map = {
        "play": "playing",
        "pause": "paused",
        "stop": "stopped",
        "seek": "playing",
    }

    return PlaybackResponse(
        recording_name=request.recording_name,
        status=status_map[request.action],
        current_frame=current_frame,
        total_frames=total_frames,
    )


@router.get("/frame/{recording_name}/{frame_index}", response_model=SkeletonFrame)
@precondition(
    lambda recording_name, frame_index: (
        recording_name is not None
        and len(recording_name.strip()) > 0
        and frame_index >= 0
    ),
    "Recording name must be non-empty and frame index must be non-negative",
)
async def get_frame(recording_name: str, frame_index: int) -> SkeletonFrame:
    """Get skeleton data for a specific frame.

    See issue #1206
    """
    if recording_name not in _recordings:
        raise HTTPException(
            status_code=404,
            detail=f"Recording '{recording_name}' not found",
        )

    recording = _recordings[recording_name]
    frames = recording.get("frames", [])

    if frame_index < 0 or frame_index >= len(frames):
        # Return a default rest-pose frame using the recording's joint set
        joints = [
            JointData(
                name=name,
                position=[0.0, 0.0, 0.0],
                confidence=0.0,
                parent=_skeleton_parent_for(recording["source_type"], name),
            )
            for name in _recording_joint_names(recording)
        ]

        return SkeletonFrame(
            frame_index=frame_index,
            timestamp=frame_index / recording.get("frame_rate", 30.0),
            joints=joints,
        )

    return SkeletonFrame(**frames[frame_index])


@router.post("/upload-c3d", response_model=C3DUploadResponse)
async def upload_c3d(file: UploadFile = File(...)) -> C3DUploadResponse:
    """Upload a C3D file and register it as a playback-ready recording.

    The file is parsed with the same ``C3DDataReader`` the desktop viewer
    uses; marker positions are converted to meters (``target_units="m"``)
    so mm-based files render at the correct scale. Returns marker metadata
    and the recording id usable with the playback/frame endpoints.

    See issue #7454
    """
    filename = file.filename or "upload.c3d"
    if not filename.lower().endswith(".c3d"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected a .c3d file, got '{filename}'",
        )

    available, reason = _source_availability("c3d")
    if not available:
        raise HTTPException(
            status_code=503,
            detail=f"C3D import is unavailable: {reason}",
        )

    from shared.python.sidekick.lab.bio.c3d_reader import C3DDataReader

    with tempfile.TemporaryDirectory(prefix="mocap_c3d_") as tmp_dir:
        tmp_path = Path(tmp_dir) / "upload.c3d"
        await write_upload_file_to_path(file, tmp_path)
        try:
            reader = C3DDataReader(tmp_path)
            metadata = reader.get_metadata()
            df_points = reader.points_dataframe(include_time=False, target_units="m")
        except (ValueError, KeyError, OSError, RuntimeError, IndexError) as exc:
            logger.exception("Failed to parse uploaded C3D file %s", filename)
            raise HTTPException(
                status_code=422,
                detail=f"Could not parse C3D file '{filename}': {exc}",
            ) from exc

    marker_names = list(metadata.marker_labels)
    frame_rate = float(metadata.frame_rate)
    frames = _frames_from_points(df_points, frame_rate)

    _session_state["counter"] += 1
    recording_name = f"c3d_{Path(filename).stem}_{_session_state['counter']}"
    _recordings[recording_name] = {
        "source_type": "c3d",
        "frame_rate": frame_rate,
        "frames": frames,
        "joint_names": marker_names,
    }

    duration = len(frames) / frame_rate if frame_rate > 0 else 0.0
    return C3DUploadResponse(
        recording_name=recording_name,
        marker_names=marker_names,
        frame_rate=frame_rate,
        total_frames=len(frames),
        duration_seconds=duration,
        native_units=str(metadata.units or ""),
        converted_units="m",
    )


# ── Helpers ──

_UNAVAILABLE_REASONS = {
    "mediapipe": (
        "mediapipe",
        "MediaPipe is not installed on the server (pip install mediapipe)",
    ),
    "openpose": (
        "openpose",
        "OpenPose Python bindings are not installed on the server",
    ),
    "c3d": (
        "ezc3d",
        "ezc3d is not installed on the server (pip install ezc3d)",
    ),
}


def _source_availability(source_id: str) -> tuple[bool, str | None]:
    """Probe whether a capture source's backing package is importable.

    Returns ``(available, reason)`` where ``reason`` is ``None`` when the
    source is available and a human-readable explanation otherwise.
    """
    module_name, reason = _UNAVAILABLE_REASONS[source_id]
    try:
        if importlib.util.find_spec(module_name) is not None:
            return True, None
    except (ImportError, ValueError):  # broken partial installs
        logger.warning("Probing importability of %s failed", module_name)
    return False, reason


def _recording_joint_names(recording: dict[str, Any]) -> list[str]:
    """Joint names for a recording: stored names (C3D markers) or skeleton."""
    stored = recording.get("joint_names")
    if stored:
        return list(stored)
    if recording["source_type"] == "openpose":
        skeleton = _OPENPOSE_SKELETON
    else:
        skeleton = _MEDIAPIPE_SKELETON
    return [j["name"] for j in skeleton]


def _skeleton_parent_for(source_type: str, joint_name: str) -> str | None:
    """Parent joint from the source's skeleton template (None for C3D markers)."""
    if source_type == "openpose":
        skeleton = _OPENPOSE_SKELETON
    elif source_type == "mediapipe":
        skeleton = _MEDIAPIPE_SKELETON
    else:
        return None
    for joint in skeleton:
        if joint["name"] == joint_name:
            return joint.get("parent")
    return None


def _frames_from_points(df_points: Any, frame_rate: float) -> list[dict[str, Any]]:
    """Convert a tidy C3D points DataFrame into playback frames.

    Non-finite coordinates (occluded markers) are zeroed with confidence 0
    so frames stay JSON-serializable; valid markers get confidence 1.
    """
    if df_points is None or df_points.empty:
        return []

    frames: list[dict[str, Any]] = []
    frame_ids = sorted(df_points["frame"].unique())
    grouped = dict(tuple(df_points.groupby("frame")))
    for out_index, frame_id in enumerate(frame_ids):
        group = grouped[frame_id]
        joints = []
        for row in group.itertuples(index=False):
            x, y, z = float(row.x), float(row.y), float(row.z)
            finite = all(math.isfinite(v) for v in (x, y, z))
            joints.append(
                {
                    "name": str(row.marker),
                    "position": [x, y, z] if finite else [0.0, 0.0, 0.0],
                    "confidence": 1.0 if finite else 0.0,
                    "parent": None,
                }
            )
        timestamp = out_index / frame_rate if frame_rate > 0 else float(out_index)
        frames.append(
            {
                "frame_index": out_index,
                "timestamp": timestamp,
                "joints": joints,
            }
        )
    return frames
