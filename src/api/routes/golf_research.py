"""Authenticated authored-impact admission to explicit local research golf."""

from __future__ import annotations

from typing import Annotated, Any
from weakref import WeakKeyDictionary

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from src.api.routes.golf_simulator import get_current_session_service
from src.api.routes.matched_swings import require_local_client
from src.api.routes.necromatcher import get_library
from src.shared.python.golf_simulator import (
    AimContext,
    GolfSessionService,
    ShotMetadata,
    SourceKind,
)
from src.shared.python.workspace import (
    NecromatcherLibrary,
    load_research_impact_shot,
)

router = APIRouter(
    prefix="/tools/golf-simulator",
    tags=["golf-simulator"],
    dependencies=[Depends(require_local_client)],
)
Library = Annotated[NecromatcherLibrary, Depends(get_library)]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
_contexts: WeakKeyDictionary[GolfSessionService, dict[str, dict[str, Any]]] = (
    WeakKeyDictionary()
)
_CONTEXT_LIMIT = 32


class ResearchAim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    source_to_target_rotation: list[
        Annotated[list[float], Field(min_length=3, max_length=3)]
    ] = Field(min_length=3, max_length=3)
    revision: int = Field(ge=0)


class ResearchPrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    replay_id: Identifier
    run_id: Identifier
    shot_id: Identifier
    session_id: Identifier
    created_at_utc: str = Field(min_length=1, max_length=64)
    aim_context: ResearchAim
    context_revision: int = Field(ge=0)


def _local_session(session_id: str | None = None) -> GolfSessionService:
    service = get_current_session_service()
    if service is None or service.current_destination_id != "local_in_memory":
        raise HTTPException(400, "Connect the Local Reference Simulator first")
    if session_id is not None and service.session_id != session_id:
        raise HTTPException(400, "Research session identity differs")
    return service


@router.post("/shot/prepare-research-impact")
async def prepare_research_impact(
    request: ResearchPrepareRequest, library: Library
) -> dict[str, Any]:
    """Authenticate saved evidence off-loop; explicitly prepare without arming."""
    service = _local_session(request.session_id)
    if (
        request.shot_id in _contexts.get(service, {})
        or service.delivery_status(request.shot_id) is not None
    ):
        raise HTTPException(409, "Research shot identity has already been used")
    try:
        aim = AimContext(
            tuple(tuple(row) for row in request.aim_context.source_to_target_rotation),  # type: ignore[arg-type]
            request.aim_context.revision,
            "Explicit Local Research Aim",
        )
        metadata = ShotMetadata(
            request.shot_id,
            request.session_id,
            aim,
            request.created_at_utc,
            source_kind=SourceKind.MODEL_CONTACT,
        )
        admitted = await run_in_threadpool(
            load_research_impact_shot,
            library,
            request.replay_id,
            request.run_id,
            metadata,
        )
        if get_current_session_service() is not service:
            raise HTTPException(409, "Session changed during research admission")
        context = admitted.to_record()
        if (
            context.get("replay_id") != request.replay_id
            or context.get("run_id") != request.run_id
            or admitted.shot.shot_id != request.shot_id
            or admitted.shot.session_id != request.session_id
        ):
            raise ValueError("Admitted research identity differs from request")
        prepared = service.prepare_research_shot(
            admitted.shot, request.context_revision
        )
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    records = _contexts.setdefault(service, {})
    records[prepared.shot.shot_id] = context
    while len(records) > _CONTEXT_LIMIT:
        del records[next(iter(records))]
    return {
        "prepared_shot_id": prepared.prepared_shot_id,
        "shot_id": prepared.shot.shot_id,
        "context_revision": prepared.context_revision,
        "is_armed": prepared.is_armed,
        "created_at_utc": prepared.created_at_utc,
        "research": context,
    }


@router.get("/shot/{shot_id}/local-trajectory")
async def local_research_trajectory(shot_id: str) -> dict[str, Any]:
    """Return a new local simulation with its retained source research context."""
    service = _local_session()
    context = _contexts.get(service, {}).get(shot_id)
    if context is None:
        raise HTTPException(404, "No owned research context for this shot")
    try:
        record = service.get_local_trajectory_record(shot_id)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return {
        "shot_id": record.shot_id,
        "provenance": record.provenance,
        "simulated_at_utc": record.simulated_at_utc,
        "samples": [
            {
                "time_s": float(point.time),
                "position_m": [float(value) for value in point.position],
                "velocity_mps": [float(value) for value in point.velocity],
            }
            for point in record.points
        ],
        "research": context,
    }
