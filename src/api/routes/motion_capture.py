"""Motion Capture tool API routes.

Provides REST endpoints for the Motion Capture tool page:
- Capture source enumeration (C3D, OpenPose, MediaPipe)
- Skeleton data retrieval
- Recording/playback control
- Frame-by-frame joint data

See issue #1206
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/tools/motion-capture", tags=["motion-capture"])


# ── Request / Response Models ──


class CaptureSource(BaseModel):
    """Available motion capture source."""

    id: str
    name: str
    type: str = Field(description="c3d, openpose, or mediapipe")
    available: bool
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

# ── In-memory session state ──

_sessions: dict[str, dict[str, Any]] = {}
_recordings: dict[str, dict[str, Any]] = {}
_session_counter = 0


# ── Endpoints ──


@router.get("/sources", response_model=list[CaptureSource])
async def list_capture_sources() -> list[CaptureSource]:
    """List available motion capture sources.

    See issue #1206
    """
    sources = [
        CaptureSource(
            id="mediapipe",
            name="MediaPipe Pose",
            type="mediapipe",
            available=_check_mediapipe_available(),
            description="Real-time pose estimation using Google MediaPipe",
        ),
        CaptureSource(
            id="openpose",
            name="OpenPose",
            type="openpose",
            available=_check_openpose_available(),
            description="Multi-person pose estimation using OpenPose",
        ),
        CaptureSource(
            id="c3d",
            name="C3D File Import",
            type="c3d",
            available=True,
            description="Import motion capture data from C3D files",
        ),
    ]
    return sources


@router.get("/skeleton/{source_type}", response_model=list[JointData])
async def get_skeleton_template(source_type: str) -> list[JointData]:
    """Get the skeleton joint template for a given source type.

    See issue #1206
    """
    if source_type == "mediapipe":
        skeleton = _MEDIAPIPE_SKELETON
    elif source_type == "openpose":
        skeleton = _OPENPOSE_SKELETON
    elif source_type == "c3d":
        skeleton = _MEDIAPIPE_SKELETON  # Default to mediapipe layout for c3d
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
    global _session_counter  # noqa: PLW0603

    valid_sources = {"mediapipe", "openpose", "c3d"}
    if request.source_type not in valid_sources:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source type. Must be one of: {sorted(valid_sources)}",
        )

    _session_counter += 1
    session_id = f"session_{_session_counter}"

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

        # Get joint names from skeleton
        if rec["source_type"] == "openpose":
            skeleton = _OPENPOSE_SKELETON
        else:
            skeleton = _MEDIAPIPE_SKELETON
        joint_names = [j["name"] for j in skeleton]

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
        # Return a default skeleton frame with rest pose
        if recording["source_type"] == "openpose":
            skeleton = _OPENPOSE_SKELETON
        else:
            skeleton = _MEDIAPIPE_SKELETON

        joints = [
            JointData(
                name=j["name"],
                position=[0.0, 0.0, 0.0],
                confidence=0.0,
                parent=j.get("parent"),
            )
            for j in skeleton
        ]

        return SkeletonFrame(
            frame_index=frame_index,
            timestamp=frame_index / recording.get("frame_rate", 30.0),
            joints=joints,
        )

    return SkeletonFrame(**frames[frame_index])


# ── Helpers ──


def _check_mediapipe_available() -> bool:
    """Check if MediaPipe is installed."""
    try:
        import mediapipe  # noqa: F401

        return True
    except ImportError:
        return False


def _check_openpose_available() -> bool:
    """Check if OpenPose Python bindings are available."""
    try:
        import openpose  # noqa: F401

        return True
    except ImportError:
        return False
