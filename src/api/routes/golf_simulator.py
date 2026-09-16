"""REST API endpoints for the Golf Simulator capability-aware integration.

Follows TDD, DbC, Law of Demeter, and DRY.
Acceptance criteria from Issue #10196 (GS-07).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.shared.python.golf_simulator.adapters.fake import FakeSimulatorAdapter
from src.shared.python.golf_simulator.adapters.local import LocalReferenceAdapter
from src.shared.python.golf_simulator.contracts import (
    AimContext,
    CapabilityDescriptor,
    CapabilityState,
    ConnectionStatus,
    ContactStatus,
    NumericalStatus,
    ScientificStatus,
    ReplayPlaybackState,
    SessionState,
    ShotEnvelope,
    ShotQualification,
    SimulatorAdapter,
    SimulatorCapabilities,
    SourceKind,
    SubmissionReceipt,
    SubmissionState,
)
from src.shared.python.golf_simulator.journal import ShotJournal
from src.shared.python.golf_simulator.replay import (
    MonotonicReplayClock,
    ReplaySubmissionCoordinator,
)
from src.shared.python.golf_simulator.session import GolfSessionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/golf-simulator", tags=["golf-simulator"])

# --- In-memory Service and Coordinator Singletons ---
_GLOBAL_JOURNAL: ShotJournal = ShotJournal()
_CURRENT_SERVICE: GolfSessionService | None = None
_CURRENT_COORDINATOR: ReplaySubmissionCoordinator | None = None
_CURRENT_CLOCK: MonotonicReplayClock | None = None


def get_current_session_service() -> GolfSessionService | None:
    """Accessor for the active session service singleton."""
    return _CURRENT_SERVICE


def reset_simulator_state() -> None:
    """Reset service and replay coordinator state (primarily for tests)."""
    global _CURRENT_SERVICE, _CURRENT_COORDINATOR, _CURRENT_CLOCK
    _CURRENT_SERVICE = None
    _CURRENT_COORDINATOR = None
    _CURRENT_CLOCK = None


def _get_or_create_service(
    session_id: str | None = None, destination_id: str = "local"
) -> GolfSessionService:
    global _CURRENT_SERVICE, _CURRENT_COORDINATOR, _CURRENT_CLOCK
    if _CURRENT_SERVICE is not None:
        if session_id is None or _CURRENT_SERVICE.session_id == session_id:
            return _CURRENT_SERVICE

    target_session_id = session_id or "default-session"
    adapter = _create_adapter(destination_id)
    _CURRENT_SERVICE = GolfSessionService(
        session_id=target_session_id,
        adapter=adapter,
        journal=_GLOBAL_JOURNAL,
    )
    _CURRENT_CLOCK = MonotonicReplayClock()
    _CURRENT_COORDINATOR = ReplaySubmissionCoordinator(
        session_service=_CURRENT_SERVICE,
        clock=_CURRENT_CLOCK,
    )
    return _CURRENT_SERVICE


def _create_adapter(destination_id: str) -> SimulatorAdapter:
    if destination_id == "local":
        return LocalReferenceAdapter()
    if destination_id == "gspro":
        # Return FakeSimulatorAdapter or configured GSPro adapter
        return FakeSimulatorAdapter(destination_id="gspro_tcp")
    if destination_id == "fake":
        return FakeSimulatorAdapter()
    # Fallback to local
    return LocalReferenceAdapter()


# --- Pydantic Schemas ---


class DestinationInfo(BaseModel):
    destination_id: str
    name: str
    description: str
    is_connected: bool
    capabilities: dict[str, Any]


class DestinationsResponse(BaseModel):
    destinations: list[DestinationInfo]


class CreateSessionRequest(BaseModel):
    destination_id: str = "local"
    session_id: str = "default-session"
    config: dict[str, Any] | None = None


class SessionStatusResponse(BaseModel):
    session_id: str
    destination_id: str
    state: str
    capabilities: dict[str, Any]


class AimContextSchema(BaseModel):
    source_to_target_rotation: list[list[float]] = Field(
        default_factory=lambda: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    revision: int = 1


class ShotQualificationSchema(BaseModel):
    contact: str = "qualified"
    numerical: str = "converged"
    scientific: str = "benchmarked"
    evidence_refs: list[str] = Field(default_factory=lambda: ["api"])


class ShotEnvelopeSchema(BaseModel):
    schema_version: int = 1
    shot_id: str
    session_id: str
    source_kind: str = "manual"
    ball_velocity_m_s: list[float]
    ball_angular_velocity_rad_s: list[float]
    aim_context: AimContextSchema
    qualification: ShotQualificationSchema
    created_at_utc: str
    model_run_id: str | None = None
    trace_digest: str | None = None
    impact_id: str | None = None
    impact_time_s: float | None = None


class PrepareShotRequest(BaseModel):
    shot: ShotEnvelopeSchema
    context_revision: int = 1


class PreparedShotResponse(BaseModel):
    prepared_shot_id: str
    shot_id: str
    context_revision: int
    is_armed: bool
    created_at_utc: str


class ArmShotRequest(BaseModel):
    prepared_shot_id: str
    context_revision: int


class ArmShotResponse(BaseModel):
    prepared_shot_id: str
    arm_token: str
    state: str


class DisarmShotRequest(BaseModel):
    prepared_shot_id: str


class CancelShotRequest(BaseModel):
    prepared_shot_id: str


class SubmitShotRequest(BaseModel):
    prepared_shot_id: str
    arm_token: str


class SubmissionReceiptResponse(BaseModel):
    shot_id: str
    session_id: str
    state: str
    destination_id: str
    attempt_id: str
    timestamp_utc: str
    detail: str


class ReplayActionRequest(BaseModel):
    action: Literal["play", "pause", "stop", "seek", "rate"]
    target_time_s: float | None = None
    playback_rate: float | None = None


class ReplayStatusResponse(BaseModel):
    playback_state: str
    current_time_s: float
    playback_rate: float


class ResolveUncertainRequest(BaseModel):
    operator_evidence: str
    confirmed: bool = True


# --- Endpoints ---


@router.get("/destinations", response_model=DestinationsResponse)
async def list_destinations() -> DestinationsResponse:
    """List available simulator destinations and their capability profiles."""
    dests = [
        DestinationInfo(
            destination_id="local",
            name="Local Reference Simulator",
            description="UpstreamDrift local aerodynamics ball-flight simulation",
            is_connected=True,
            capabilities={
                "shot_input": {
                    "state": "supported",
                    "evidence": "Direct in-process ball flight physics",
                },
                "club_data": {
                    "state": "supported",
                    "evidence": "Impact model reconstruction",
                },
                "local_trajectory_return": {
                    "state": "supported",
                    "evidence": "Full 3D ODE point series",
                },
            },
        ),
        DestinationInfo(
            destination_id="gspro",
            name="GSPro Open Connect v1",
            description="TCP socket connection to GSPro Connect (port 921)",
            is_connected=False,
            capabilities={
                "shot_input": {"state": "supported", "evidence": "Open Connect v1 API"},
                "club_data": {
                    "state": "supported",
                    "evidence": "Club speed, path, face angle",
                },
                "local_trajectory_return": {
                    "state": "unsupported",
                    "evidence": "GSPro simulates internally",
                },
            },
        ),
    ]
    return DestinationsResponse(destinations=dests)


def _capabilities_to_dict(caps: SimulatorCapabilities) -> dict[str, Any]:
    shot_input = caps.shot_input
    club_data = caps.club_data
    traj_return = caps.local_trajectory_return
    return {
        "shot_input": {
            "state": str(shot_input.state.value),
            "evidence": shot_input.evidence,
        },
        "club_data": {
            "state": str(club_data.state.value),
            "evidence": club_data.evidence,
        },
        "local_trajectory_return": {
            "state": str(traj_return.state.value),
            "evidence": traj_return.evidence,
        },
    }


def _receipt_to_response(receipt: SubmissionReceipt) -> SubmissionReceiptResponse:
    return SubmissionReceiptResponse(
        shot_id=receipt.shot_id,
        session_id=receipt.session_id,
        state=receipt.state.value,
        destination_id=receipt.destination_id,
        attempt_id=receipt.attempt_id,
        timestamp_utc=receipt.timestamp_utc,
        detail=receipt.detail,
    )


@router.post("/session", response_model=SessionStatusResponse)
async def create_or_update_session(
    req: CreateSessionRequest,
) -> SessionStatusResponse:
    """Create a new session or switch destination for an existing session."""
    global _CURRENT_SERVICE, _CURRENT_COORDINATOR, _CURRENT_CLOCK
    if _CURRENT_SERVICE is not None and _CURRENT_SERVICE.current_state in (
        SessionState.ARMED,
        SessionState.SUBMITTING,
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot switch destination while in state '{_CURRENT_SERVICE.current_state.value}'",
        )

    adapter = _create_adapter(req.destination_id)
    _CURRENT_SERVICE = GolfSessionService(
        session_id=req.session_id,
        adapter=adapter,
        journal=_GLOBAL_JOURNAL,
    )
    await _CURRENT_SERVICE.select_destination(adapter, req.config)

    _CURRENT_CLOCK = MonotonicReplayClock()
    _CURRENT_COORDINATOR = ReplaySubmissionCoordinator(
        session_service=_CURRENT_SERVICE,
        clock=_CURRENT_CLOCK,
    )

    caps = adapter.capabilities()
    return SessionStatusResponse(
        session_id=_CURRENT_SERVICE.session_id,
        destination_id=req.destination_id,
        state=_CURRENT_SERVICE.current_state.value,
        capabilities=_capabilities_to_dict(caps),
    )


@router.get("/session", response_model=SessionStatusResponse)
async def get_session() -> SessionStatusResponse:
    """Get status and capabilities of the active session."""
    service = _get_or_create_service()
    caps = service._adapter.capabilities()
    return SessionStatusResponse(
        session_id=service.session_id,
        destination_id=service.current_destination_id or "local",
        state=service.current_state.value,
        capabilities=_capabilities_to_dict(caps),
    )


@router.delete("/session")
async def close_session() -> dict[str, str]:
    """Close active session and reset state to idle."""
    reset_simulator_state()
    return {"status": "closed"}


@router.post("/shot/prepare", response_model=PreparedShotResponse)
async def prepare_shot(req: PrepareShotRequest) -> PreparedShotResponse:
    """Prepare a shot for arming and submission."""
    service = _get_or_create_service()
    if req.shot.session_id != service.session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Shot session_id '{req.shot.session_id}' mismatch with service session_id '{service.session_id}'",
        )

    # Convert Pydantic to domain model
    raw = req.shot
    aim = AimContext(
        source_to_target_rotation=tuple(
            tuple(r) for r in raw.aim_context.source_to_target_rotation
        ),  # type: ignore[arg-type]
        revision=raw.aim_context.revision,
    )
    qual = ShotQualification(
        contact=ContactStatus(raw.qualification.contact),
        numerical=NumericalStatus(raw.qualification.numerical),
        scientific=ScientificStatus(raw.qualification.scientific),
        evidence_refs=tuple(raw.qualification.evidence_refs),
    )
    domain_shot = ShotEnvelope(
        schema_version=raw.schema_version,
        shot_id=raw.shot_id,
        session_id=raw.session_id,
        source_kind=SourceKind(raw.source_kind),
        qualification=qual,
        ball_velocity_m_s=tuple(raw.ball_velocity_m_s),  # type: ignore[arg-type]
        ball_angular_velocity_rad_s=tuple(raw.ball_angular_velocity_rad_s),  # type: ignore[arg-type]
        aim_context=aim,
        created_at_utc=raw.created_at_utc,
        model_run_id=raw.model_run_id,
        trace_digest=raw.trace_digest,
        impact_id=raw.impact_id,
        impact_time_s=raw.impact_time_s,
    )

    try:
        prep = service.prepare(domain_shot, context_revision=req.context_revision)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return PreparedShotResponse(
        prepared_shot_id=prep.prepared_shot_id,
        shot_id=prep.shot.shot_id,
        context_revision=prep.context_revision,
        is_armed=prep.is_armed,
        created_at_utc=prep.created_at_utc,
    )


@router.post("/shot/arm", response_model=ArmShotResponse)
async def arm_shot(req: ArmShotRequest) -> ArmShotResponse:
    """Arm a prepared shot with context revision verification."""
    service = _get_or_create_service()
    try:
        token = service.arm(req.prepared_shot_id, context_revision=req.context_revision)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return ArmShotResponse(
        prepared_shot_id=req.prepared_shot_id,
        arm_token=token,
        state=service.current_state.value,
    )


@router.post("/shot/disarm")
async def disarm_shot(req: DisarmShotRequest) -> dict[str, str]:
    """Disarm an armed shot, returning session to PREPARED."""
    service = _get_or_create_service()
    try:
        service.disarm(req.prepared_shot_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return {"status": "disarmed", "state": service.current_state.value}


@router.post("/shot/cancel")
async def cancel_shot(req: CancelShotRequest) -> dict[str, str]:
    """Cancel a prepared or armed shot, returning session to IDLE."""
    service = _get_or_create_service()
    try:
        service.cancel(req.prepared_shot_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return {"status": "cancelled", "state": service.current_state.value}


@router.post("/shot/submit", response_model=SubmissionReceiptResponse)
async def submit_shot(req: SubmitShotRequest) -> SubmissionReceiptResponse:
    """Execute single-impact submission to the simulator destination."""
    service = _get_or_create_service()
    try:
        receipt = await service.submit_at_impact(req.prepared_shot_id, req.arm_token)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return _receipt_to_response(receipt)


@router.get("/shot/{shot_id}/status", response_model=SubmissionReceiptResponse)
async def get_shot_status(shot_id: str) -> SubmissionReceiptResponse:
    """Query submission status for a previously submitted shot."""
    service = _get_or_create_service()
    receipt = service.delivery_status(shot_id)
    if receipt is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Shot '{shot_id}' receipt not found",
        )

    return _receipt_to_response(receipt)


@router.post("/shot/{shot_id}/resolve", response_model=SubmissionReceiptResponse)
async def resolve_uncertain_delivery(
    shot_id: str, req: ResolveUncertainRequest
) -> SubmissionReceiptResponse:
    """Operator reconciliation for an UNCERTAIN submission."""
    service = _get_or_create_service()
    try:
        receipt = service.resolve_uncertain(
            shot_id=shot_id,
            operator_evidence=req.operator_evidence,
            confirmed=req.confirmed,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return _receipt_to_response(receipt)


@router.post("/replay/action", response_model=ReplayStatusResponse)
async def replay_action(req: ReplayActionRequest) -> ReplayStatusResponse:
    """Control monotonic replay clock playback."""
    _get_or_create_service()
    assert _CURRENT_CLOCK is not None
    clock = _CURRENT_CLOCK

    if req.action == "play":
        clock.play()
    elif req.action == "pause":
        clock.pause()
    elif req.action == "stop":
        clock.stop()
    elif req.action == "seek":
        if req.target_time_s is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="target_time_s required for seek",
            )
        clock.seek(req.target_time_s)
    elif req.action == "rate":
        if req.playback_rate is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="playback_rate required for rate",
            )
        clock.set_playback_rate(req.playback_rate)

    return ReplayStatusResponse(
        playback_state=clock.playback_state.value,
        current_time_s=clock.current_time_s,
        playback_rate=clock.playback_rate,
    )
