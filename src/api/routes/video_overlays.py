"""Video force/torque overlay routes projecting 2D glyphs (FTO-29, #11314).

Provides endpoints returning 2D projected force/torque glyphs (polyline shafts,
head polygons, moment arcs, and legend) for video frames through calibrated cameras.
Reuses FTO-8/FTO-22 projection and contracts (DRY).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence, cast

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from src.shared.python.core.contracts import require
from src.shared.python.force_overlay import (
    ForceGlyphStyle,
    GlyphSet,
    LegendSpec,
    WrenchKind,
    build_glyphs,
)
from src.shared.python.force_overlay.projection import (
    ProjectedGlyphSet,
    project_glyphs,
)
from src.shared.python.force_overlay.renderers.opencv_glyphs import (
    PinholeProjector,
)

if TYPE_CHECKING:
    from src.motion_capture.reconstruct.cameras import PinholeCamera
    from src.motion_capture.reference.registration import ReferenceRegistration
    from src.shared.python.force_overlay.contracts import ForceTorqueFrame
    from src.shared.python.force_overlay.series import ForceTorqueSeries
    from src.shared.python.pose_estimation.observations import CameraCalibration

router = APIRouter(tags=["overlays"])

_KIND_NAME_MAP: dict[str, WrenchKind] = {
    "actuator": WrenchKind.JOINT_ACTUATOR,
    "applied": WrenchKind.JOINT_ACTUATOR,
    "joint_actuator": WrenchKind.JOINT_ACTUATOR,
    "reaction": WrenchKind.JOINT_REACTION,
    "joint_reaction": WrenchKind.JOINT_REACTION,
    "contact": WrenchKind.CONTACT,
    "grip": WrenchKind.GRIP,
    "external": WrenchKind.EXTERNAL,
    "gravity": WrenchKind.GRAVITY,
    "muscle": WrenchKind.MUSCLE,
}


@dataclass
class VideoOverlaySource:
    """Registered video source with camera calibration, series, and timing."""

    camera: PinholeCamera | CameraCalibration | None
    series: ForceTorqueSeries | None = None
    fps: float = 30.0
    frame_times: list[float] | None = None
    registration: ReferenceRegistration | None = None
    total_frames: int = 0


class VideoOverlayStore:
    """Registry and storage for video overlay camera and series bindings."""

    def __init__(self) -> None:
        self._sources: dict[str, VideoOverlaySource] = {}

    def register_source(self, source_id: str, source: VideoOverlaySource) -> None:
        require(bool(source_id and source_id.strip()), "source_id cannot be blank")
        self._sources[source_id] = source

    def get_source(self, source_id: str) -> VideoOverlaySource:
        if source_id in self._sources:
            return self._sources[source_id]
        raise KeyError(f"Source '{source_id}' not found")


def get_video_overlay_store(request: Request) -> VideoOverlayStore:
    """Retrieve or initialize the VideoOverlayStore from app state."""
    state = request.app.state
    store = getattr(state, "video_overlay_store", None)
    if store is None:
        store = VideoOverlayStore()
        state.video_overlay_store = store
    return store


def _parse_kinds(kinds_str: str | None) -> frozenset[WrenchKind]:
    """Parse comma-separated kind names into a set of WrenchKinds."""
    if not kinds_str or not kinds_str.strip():
        return frozenset(WrenchKind)
    selected: set[WrenchKind] = set()
    for item in kinds_str.split(","):
        cleaned = item.strip().lower()
        if cleaned in ("all", "*"):
            return frozenset(WrenchKind)
        if cleaned in _KIND_NAME_MAP:
            selected.add(_KIND_NAME_MAP[cleaned])
        else:
            for k in WrenchKind:
                if k.value.lower() == cleaned:
                    selected.add(k)
                    break
    return frozenset(selected) if selected else frozenset(WrenchKind)


def _build_request_style(
    kinds_str: str | None, scale: float, show_labels: bool
) -> ForceGlyphStyle:
    """Construct ForceGlyphStyle honoring scale factor and kinds filter."""
    kinds = _parse_kinds(kinds_str)
    ratio = max(scale, 1e-6)
    return ForceGlyphStyle(
        force_scale_m_per_n=0.001 * ratio,
        torque_scale_m_per_nm=0.005 * ratio,
        kinds=kinds,
        show_labels=show_labels,
    )


def _resolve_frame_time(source: VideoOverlaySource, n: int) -> float:
    """Determine video timestamp for frame n."""
    if source.frame_times and 0 <= n < len(source.frame_times):
        return float(source.frame_times[n])
    fps = source.fps if source.fps > 0 else 30.0
    return float(n / fps)


def _resolve_force_frame(
    source: VideoOverlaySource, time_s: float
) -> ForceTorqueFrame | None:
    """Extract or align ForceTorqueFrame for timestamp."""
    if source.series is None:
        return None
    if source.registration is not None:
        from src.motion_capture.reference.force_alignment import (
            force_frame_for_video,
        )

        return force_frame_for_video(
            source.series, video_time_s=time_s, registration=source.registration
        )
    return cast("ForceTorqueFrame | None", source.series.frame_at(time_s))


def _build_empty_glyphs(time_s: float) -> GlyphSet:
    """Build an empty glyph set with unavailable label indicator."""
    legend = LegendSpec(
        engine="unknown",
        force_reference_n=100.0,
        torque_reference_nm=10.0,
        unavailable_labels=("no_frame_at_time",),
    )
    return GlyphSet(time_s=time_s, arrows=(), torque_arcs=(), legend=legend)


@router.get("/overlays/video/{source_id}/frames/{n}/glyphs")
async def get_video_frame_glyphs(
    source_id: str = Path(..., description="Unique video or capture source ID"),
    n: int = Path(..., ge=0, description="Video frame index (0-based)"),
    kinds: str | None = Query(None, description="Comma-separated wrench kinds"),
    scale: float = Query(1.0, gt=0, description="Arrow linear scaling factor"),
    show_labels: bool = Query(False, description="Whether to include text labels"),
    store: VideoOverlayStore = Depends(get_video_overlay_store),
) -> dict[str, Any]:
    """Retrieve 2D projected force/torque glyphs aligned with a video frame.

    Args:
        source_id: Video source identifier.
        n: 0-based frame index.
        kinds: Optional comma-separated kind filters.
        scale: Linear scaling factor (> 0).
        show_labels: Whether to attach text labels.
        store: Injected VideoOverlayStore.

    Returns:
        ProjectedGlyphSet dictionary payload with pixel coordinates.
    """
    try:
        source = store.get_source(source_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if source.camera is None:
        raise HTTPException(
            status_code=409,
            detail=f"No camera calibration available for source '{source_id}'",
        )

    if source.total_frames > 0 and n >= source.total_frames:
        raise HTTPException(
            status_code=404,
            detail=f"Frame index {n} out of range (total {source.total_frames})",
        )

    t = _resolve_frame_time(source, n)
    style = _build_request_style(kinds, scale, show_labels)
    frame = _resolve_force_frame(source, t)
    glyph_set = (
        build_glyphs(frame, style) if frame is not None else _build_empty_glyphs(t)
    )

    projector = PinholeProjector(source.camera)
    cam_size = getattr(source.camera, "image_size_px", (1920, 1080))
    projected = project_glyphs(glyph_set, projector, image_size_px=cam_size)
    return projected.to_dict()
