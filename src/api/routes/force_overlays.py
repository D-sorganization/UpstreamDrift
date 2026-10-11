"""Force/torque vector overlay routes (#1199, #11307, FTO-22).

Provides endpoints for streaming force/torque visualization data
from the active simulation engine provider. Emits renderer-neutral
GlyphSet payloads (glyph-set-v1) and ForceTorqueFrame inspection payloads.

No demo or fabricated vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, Depends

from src.api.middleware.error_handler import handle_api_errors
from src.api.services.force_overlay_service import (
    current_force_frame,
    force_overlay_payload,
    style_from_request_params,
)
from src.shared.python.core.contracts import precondition

from ..dependencies import get_engine_manager, get_logger
from ..models.requests import ForceOverlayRequest
from ..models.responses import ForceOverlayResponse

if TYPE_CHECKING:
    from src.shared.python.engine_core.engine_manager import EngineManager

router = APIRouter()


def _get_sim_time(engine_manager: Any) -> float:
    """Get current simulation time from active engine if available."""
    try:
        active = engine_manager.get_active_engine()
        if active is None:
            return 0.0
        if hasattr(active, "get_state"):
            state = active.get_state()
            if isinstance(state, dict):
                return float(state.get("time", 0.0))
        if hasattr(active, "time"):
            return float(active.time)
        return 0.0
    except (ValueError, RuntimeError, AttributeError, TypeError):
        return 0.0


def _build_overlay_response(
    engine_manager: Any,
    config: ForceOverlayRequest,
) -> ForceOverlayResponse:
    """Build ForceOverlayResponse from engine manager and request config."""
    if not config.enabled:
        return ForceOverlayResponse(
            sim_time=_get_sim_time(engine_manager),
            glyphs=None,
            frame=None,
            unavailable_reason="Force overlay disabled in request",
            total_force_magnitude=0.0,
            total_torque_magnitude=0.0,
            overlay_config={
                "force_types": config.force_types,
                "color_by_magnitude": config.color_by_magnitude,
                "scale_factor": config.scale_factor,
                "body_filter": config.body_filter,
                "show_labels": config.show_labels,
            },
        )

    try:
        engine = engine_manager.get_active_engine()
    except (AttributeError, RuntimeError):
        engine = None

    frame = current_force_frame(engine)
    style = style_from_request_params(
        force_types=config.force_types,
        scale_factor=config.scale_factor,
        show_labels=config.show_labels,
        scale_mode=config.scale_mode,
        reference_force_n=config.reference_force_n,
        reference_length_m=config.reference_length_m,
        kind_scale=config.kind_scale,
        groups=config.groups,
    )
    payload = force_overlay_payload(frame, style, body_filter=config.body_filter)

    if payload["glyphs"] is not None:
        sim_time = frame.time_s if frame is not None else 0.0
        total_force = sum(a["magnitude"] for a in payload["glyphs"].get("arrows", []))
        total_torque = sum(
            t["magnitude"] for t in payload["glyphs"].get("torque_arcs", [])
        )
    else:
        sim_time = _get_sim_time(engine_manager)
        total_force = 0.0
        total_torque = 0.0

    return ForceOverlayResponse(
        sim_time=sim_time,
        glyphs=payload["glyphs"],
        frame=payload["frame"],
        unavailable_reason=payload.get("unavailable_reason"),
        total_force_magnitude=total_force,
        total_torque_magnitude=total_torque,
        overlay_config={
            "force_types": config.force_types,
            "color_by_magnitude": config.color_by_magnitude,
            "scale_factor": config.scale_factor,
            "body_filter": config.body_filter,
            "show_labels": config.show_labels,
            "scale_mode": config.scale_mode,
        },
    )


@dataclass
class ForceOverlayQuery:
    """Query parameters of ``GET /simulation/forces`` (a FastAPI class dependency).

    Grouping the query surface keeps the route within the parameter budget and
    gives one place that converts the query into a ``ForceOverlayRequest``.
    """

    force_types: str = "applied"
    color_by_magnitude: bool = True
    body_filter: str | None = None
    show_labels: bool = False
    scale_factor: float = 0.01
    scale_mode: Literal["fixed", "body_weight", "peak"] = "fixed"
    reference_force_n: float | None = None
    reference_length_m: float = 0.5
    groups: str | None = None

    def to_request(self) -> ForceOverlayRequest:
        """Build the overlay request.

        Raises:
            ValueError: if ``force_types`` is blank or ``scale_factor`` <= 0.
        """
        if not self.force_types.strip():
            raise ValueError("force_types must be non-empty")
        if self.scale_factor <= 0:
            raise ValueError("Scale factor must be positive")
        return ForceOverlayRequest(
            enabled=True,
            force_types=self.force_types.split(","),
            color_by_magnitude=self.color_by_magnitude,
            body_filter=self.body_filter.split(",") if self.body_filter else None,
            show_labels=self.show_labels,
            scale_factor=self.scale_factor,
            scale_mode=self.scale_mode,
            reference_force_n=self.reference_force_n,
            reference_length_m=self.reference_length_m,
            kind_scale=None,
            groups=self.groups.split(",") if self.groups else None,
        )


@router.get(
    "/simulation/forces",
    response_model=ForceOverlayResponse,
)
@handle_api_errors
async def get_force_overlays(
    query: ForceOverlayQuery = Depends(),
    engine_manager: Any = Depends(get_engine_manager),
    logger: Any = Depends(get_logger),
) -> ForceOverlayResponse:
    """Get current force/torque vectors for 3D overlay rendering."""
    return _build_overlay_response(engine_manager, query.to_request())


@router.post(
    "/simulation/forces/config",
    response_model=ForceOverlayResponse,
)
@precondition(
    lambda config, engine_manager=None, logger=None: config.scale_factor > 0,
    "Scale factor must be positive",
)
@handle_api_errors
async def update_force_overlay_config(
    config: ForceOverlayRequest,
    engine_manager: Any = Depends(get_engine_manager),
    logger: Any = Depends(get_logger),
) -> ForceOverlayResponse:
    """Update force overlay configuration and return current vectors."""
    if not (config is not None):
        raise ValueError("config must be provided")
    return _build_overlay_response(engine_manager, config)
