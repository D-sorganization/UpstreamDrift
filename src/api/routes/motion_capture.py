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

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from src.api.middleware.upload_limits import write_upload_file_to_path
from src.shared.python.core.contracts import precondition
from src.shared.python.motion_pipeline.api import PipelineResponse
from src.shared.python.motion_pipeline.contracts import MarkerTrajectory
from src.shared.python.pose_estimation.registry import (
    capture_source_estimators,
    estimator_availability,
    get_estimator_info,
)

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

    Marker positions are converted to meters server-side by the motion
    pipeline's ``C3DAdapter`` so the web visualizer never has to guess
    mm-vs-m scaling.
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
    pipeline: PipelineResponse | None = Field(
        None,
        description=(
            "Tracked-motion result when ``run_pipeline=true`` was requested; "
            "None when the upload was playback-only"
        ),
    )


# ── Skeleton definitions ──
# Estimator skeletons are registry-driven (epic #8390, C2/#8402): a newly
# registered estimator surfaces in /sources, /skeleton, and recordings
# without edits here. The module-level aliases remain for internal helpers.


def _estimator_skeleton(source_type: str) -> list[dict[str, Any]] | None:
    try:
        return list(get_estimator_info(source_type).skeleton)
    except ValueError:
        return None


_MEDIAPIPE_SKELETON: list[dict[str, Any]] = _estimator_skeleton("mediapipe") or []
_OPENPOSE_SKELETON: list[dict[str, Any]] = _estimator_skeleton("openpose") or []

# ── In-memory session state (mutable holder avoids 'global') ──

_sessions: dict[str, dict[str, Any]] = {}
_recordings: dict[str, dict[str, Any]] = {}
_session_state: dict[str, int] = {"counter": 0}
_MAX_CACHE_SIZE = 50


# ── Endpoints ──


@router.get("/sources", response_model=list[CaptureSource])
async def list_capture_sources() -> list[CaptureSource]:
    """List available motion capture sources with honest availability.

    Availability is probed server-side (importability of the backing
    package); unavailable sources carry a human-readable ``reason``.

    See issues #1206, #7454
    """
    sources = []
    listing = [
        (info.name, info.display_name, info.description)
        for info in capture_source_estimators()
    ]
    listing.append(
        ("c3d", "C3D File Import", "Import motion capture data from C3D files")
    )
    for source_id, name, description in listing:
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
    registry_skeleton = _estimator_skeleton(source_type)
    if registry_skeleton is not None:
        skeleton = registry_skeleton
    elif source_type == "c3d":
        # C3D has no fixed skeleton: marker sets are defined per-file and
        # become available after upload (issue #7454). An empty template is
        # honest — clients must not assume a MediaPipe-shaped joint set.
        skeleton = []
    else:
        valid = ", ".join(
            sorted([info.name for info in capture_source_estimators()] + ["c3d"])
        )
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source type: {source_type}. Use one of: {valid}",
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
    if len(_sessions) > _MAX_CACHE_SIZE:
        _sessions.pop(next(iter(_sessions)))

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
    if len(_recordings) > _MAX_CACHE_SIZE:
        _recordings.pop(next(iter(_recordings)))

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
async def upload_c3d(
    file: UploadFile = File(...),
    run_pipeline: bool = Query(
        False,
        description=(
            "Also hand the parsed trajectory to the motion pipeline "
            "(adapter → preprocessing → scaling → IK → matching) and return "
            "the tracked-motion result in ``pipeline``"
        ),
    ),
    ik_backend: str = Query("geometric", description="IK backend for run_pipeline"),
    matching_backend: str = Query(
        "mujoco", description="Motion matching backend for run_pipeline"
    ),
    matching_model_urdf: str | None = Query(
        None, description="Production URDF path for matching backends that need one"
    ),
) -> C3DUploadResponse:
    """Upload a C3D file and register it as a playback-ready recording.

    The file is parsed once by the motion pipeline's ``C3DAdapter`` — the
    same reader ``POST /api/v1/motion-pipeline/run`` uses — so marker
    positions arrive in meters and format quirks are handled in one place
    (#8865). Returns marker metadata and the recording id usable with the
    playback/frame endpoints.

    With ``run_pipeline=true`` the parsed ``MarkerTrajectory`` is handed to
    ``MotionPipeline`` without re-parsing and the tracked-motion result is
    returned in ``pipeline``. A failed solve does not fail the upload: the
    recording is still registered and ``pipeline.success`` is ``False`` with
    the error message, so the playback half never depends on the solve.

    See issues #7454, #8865
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

    from src.shared.python.motion_pipeline.sources.base import AdapterContractError
    from src.shared.python.motion_pipeline.sources.c3d_adapter import C3DAdapter

    with tempfile.TemporaryDirectory(prefix="mocap_c3d_") as tmp_dir:
        tmp_path = Path(tmp_dir) / "upload.c3d"
        await write_upload_file_to_path(file, tmp_path)
        try:
            trajectory = C3DAdapter().load(tmp_path)
        except (
            AdapterContractError,
            ValueError,
            KeyError,
            OSError,
            RuntimeError,
            IndexError,
        ) as exc:
            logger.exception("Failed to parse uploaded C3D file %s", filename)
            raise HTTPException(
                status_code=422,
                detail=f"Could not parse C3D file '{filename}': {exc}",
            ) from exc

    marker_names = [str(label) for label in trajectory.metadata["source_labels"]]
    frame_rate = float(trajectory.metadata["fps"])
    frames = _frames_from_trajectory(trajectory, marker_names)

    _session_state["counter"] += 1
    recording_name = f"c3d_{Path(filename).stem}_{_session_state['counter']}"
    _recordings[recording_name] = {
        "source_type": "c3d",
        "frame_rate": frame_rate,
        "frames": frames,
        "joint_names": marker_names,
    }
    if len(_recordings) > _MAX_CACHE_SIZE:
        _recordings.pop(next(iter(_recordings)))

    pipeline_response = None
    if run_pipeline:
        pipeline_response = _hand_off_to_pipeline(
            trajectory,
            request_id=recording_name,
            ik_backend=ik_backend,
            matching_backend=matching_backend,
            matching_model_urdf=matching_model_urdf,
        )

    native_units = (
        str(trajectory.metadata["units"])
        if trajectory.metadata.get("units_declared")
        else ""
    )
    duration = len(frames) / frame_rate if frame_rate > 0 else 0.0
    return C3DUploadResponse(
        recording_name=recording_name,
        marker_names=marker_names,
        frame_rate=frame_rate,
        total_frames=len(frames),
        duration_seconds=duration,
        native_units=native_units,
        converted_units="m",
        pipeline=pipeline_response,
    )


# ── Helpers ──

# Non-estimator sources keep a local probe table; estimator sources are
# probed through the registry (C2/#8402). C3D is served by the motion
# pipeline's C3DAdapter, which accepts either of its backends (#8865).
_UNAVAILABLE_REASONS = {
    "c3d": (
        ("upstream_mocap_io", "ezc3d"),
        "no C3D backend is installed on the server "
        "(pip install ezc3d or upstream-mocap-io)",
    ),
}


def _source_availability(source_id: str) -> tuple[bool, str | None]:
    """Probe whether a capture source's backing package is importable.

    Returns ``(available, reason)`` where ``reason`` is ``None`` when the
    source is available and a human-readable explanation otherwise.
    """
    if source_id not in _UNAVAILABLE_REASONS:
        return estimator_availability(source_id)
    module_names, reason = _UNAVAILABLE_REASONS[source_id]
    for module_name in module_names:
        try:
            if importlib.util.find_spec(module_name) is not None:
                return True, None
        except (ImportError, ValueError):  # broken partial installs
            logger.warning("Probing importability of %s failed", module_name)
    return False, reason


def _hand_off_to_pipeline(
    trajectory: MarkerTrajectory,
    *,
    request_id: str,
    ik_backend: str,
    matching_backend: str,
    matching_model_urdf: str | None,
) -> PipelineResponse:
    """Run the motion pipeline on an already-parsed trajectory.

    Mirrors ``POST /api/v1/motion-pipeline/run`` but skips the adapter
    stage: the trajectory is passed through as canonical data so the C3D
    file is parsed exactly once. Failures are reported in the response
    rather than raised, because the caller's upload has already succeeded.
    """
    from src.shared.python.motion_pipeline.orchestrator import (
        AdapterOverride,
        MotionPipeline,
        PipelineConfig,
    )

    config = PipelineConfig(
        adapter=AdapterOverride(format="c3d"),
        ik_backend=ik_backend,
        matching_backend=matching_backend,
        matching_model_urdf=matching_model_urdf,
    )
    pipeline = MotionPipeline(config)
    try:
        result = pipeline.run(trajectory)
    except (ValueError, RuntimeError) as exc:
        logger.exception("Motion pipeline hand-off failed for %s", request_id)
        return PipelineResponse.from_error(request_id, str(exc))
    return PipelineResponse.from_result(result, pipeline.get_audit_log())


def _recording_joint_names(recording: dict[str, Any]) -> list[str]:
    """Joint names for a recording: stored names (C3D markers) or skeleton."""
    stored = recording.get("joint_names")
    if stored:
        return list(stored)
    skeleton = _estimator_skeleton(recording["source_type"]) or _MEDIAPIPE_SKELETON
    return [j["name"] for j in skeleton]


def _skeleton_parent_for(source_type: str, joint_name: str) -> str | None:
    """Parent joint from the source's skeleton template (None for C3D markers)."""
    skeleton = _estimator_skeleton(source_type)
    if skeleton is None:
        return None
    for joint in skeleton:
        if joint["name"] == joint_name:
            return joint.get("parent")
    return None


def _frames_from_trajectory(
    trajectory: MarkerTrajectory, marker_names: list[str]
) -> list[dict[str, Any]]:
    """Convert a canonical ``MarkerTrajectory`` into playback frames.

    The adapter drops occluded / non-finite markers from a frame; they are
    re-emitted here zeroed with confidence 0 so every frame lists the full
    marker set and stays JSON-serializable. Present markers get confidence 1.
    """
    frames: list[dict[str, Any]] = []
    for out_index, frame in enumerate(trajectory.frames):
        joints = []
        for name in marker_names:
            marker = frame.markers.get(name)
            position, confidence = [0.0, 0.0, 0.0], 0.0
            if marker is not None and all(
                math.isfinite(v) for v in (marker.x, marker.y, marker.z)
            ):
                position, confidence = [marker.x, marker.y, marker.z], 1.0
            joints.append(
                {
                    "name": name,
                    "position": position,
                    "confidence": confidence,
                    "parent": None,
                }
            )
        frames.append(
            {
                "frame_index": out_index,
                "timestamp": float(frame.timestamp),
                "joints": joints,
            }
        )
    return frames
