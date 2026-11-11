"""Shared local golf-simulator execution for REST and Agent API (#15021)."""

from __future__ import annotations

import asyncio
import datetime
import uuid
from typing import TYPE_CHECKING, Any

from src.shared.python.golf_simulator.contracts import (
    AimContext,
    ContactStatus,
    NumericalStatus,
    ScientificStatus,
    ShotEnvelope,
    ShotQualification,
    SourceKind,
    SubmissionReceipt,
    SubmissionState,
)
from src.shared.python.golf_simulator.launch_bridge import (
    shot_envelope_to_launch_conditions,
)

if TYPE_CHECKING:
    from src.shared.python.physics.swing_ball_flight_pipeline import (
        FlightSimulatorProtocol,
    )


def golden_shot_envelope() -> ShotEnvelope:
    """Pinned golden shot used for Agent API parity (matches API route tests)."""
    return ShotEnvelope(
        1,
        "agent-golden-shot-01",
        "agent-golden-session",
        SourceKind.MANUAL,
        ShotQualification(
            ContactStatus.QUALIFIED,
            NumericalStatus.CONVERGED,
            ScientificStatus.BENCHMARKED,
            ("agent_api_golden",),
        ),
        (60.0, 0.0, 15.0),
        (0.0, -250.0, 0.0),
        AimContext(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        "2026-09-15T12:00:00Z",
    )


def _resolve_flight_simulator(
    flight_simulator: FlightSimulatorProtocol | None,
) -> FlightSimulatorProtocol:
    if flight_simulator is not None:
        return flight_simulator
    from src.shared.python.physics.rust_kernel import is_rust_available

    if is_rust_available():
        from src.shared.python.physics.ball_simulator import BallFlightSimulator

        return BallFlightSimulator()
    from src.shared.python.physics.ball_enhanced_simulator import (
        EnhancedBallFlightSimulator,
    )

    return EnhancedBallFlightSimulator()


def trajectory_points_to_jsonable(points: list[Any]) -> list[dict[str, Any]]:
    """Serialise trajectory points for JSON/MCP without numpy leakage."""
    out: list[dict[str, Any]] = []
    for point in points:
        time = float(getattr(point, "time", 0.0))
        position = getattr(point, "position", None)
        velocity = getattr(point, "velocity", None)
        if position is None or velocity is None:
            raise TypeError("Trajectory point must expose time, position, and velocity")
        out.append(
            {
                "time": time,
                "position": [float(x) for x in position],
                "velocity": [float(x) for x in velocity],
            }
        )
    return out


async def execute_local_reference_shot(
    shot: ShotEnvelope,
    *,
    flight_simulator: FlightSimulatorProtocol | None = None,
    connected: bool = True,
) -> dict[str, Any]:
    """Run one local-reference shot; sole execution path for REST and MCP."""
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    attempt_id = str(uuid.uuid4())
    if not connected:
        receipt = SubmissionReceipt(
            shot_id=shot.shot_id,
            session_id=shot.session_id,
            state=SubmissionState.FAILED_BEFORE_SEND,
            destination_id="local_reference",
            attempt_id=attempt_id,
            timestamp_utc=now_utc,
            detail="Adapter not connected",
        )
        return {
            "receipt": _receipt_dict(receipt),
            "trajectory": [],
            "provenance": "local_reference:disconnected",
        }

    simulator = _resolve_flight_simulator(flight_simulator)
    launch = shot_envelope_to_launch_conditions(shot)
    points = await asyncio.to_thread(simulator.simulate_trajectory, launch)
    trajectory = trajectory_points_to_jsonable(list(points))
    provenance = (
        f"local_reference:{type(simulator).__name__}:"
        f"source={shot.source_kind.value}:schema={shot.schema_version}"
    )
    receipt = SubmissionReceipt(
        shot_id=shot.shot_id,
        session_id=shot.session_id,
        state=SubmissionState.CONFIRMED_ACCEPTED,
        destination_id="local_reference",
        attempt_id=attempt_id,
        timestamp_utc=now_utc,
        detail=f"Local simulation completed ({len(trajectory)} points)",
    )
    return {
        "receipt": _receipt_dict(receipt),
        "trajectory": trajectory,
        "trajectory_points": list(points),
        "provenance": provenance,
    }


def execute_golden_golf_shot(request: dict[str, Any] | None = None) -> dict[str, Any]:
    """Synchronous wrapper for the golden shot (async flight kernel under the hood)."""
    shot = golden_shot_envelope()
    padding = 0
    if request:
        raw = request.get("trajectory_padding", 0)
        if raw:
            padding = int(raw)
    body = asyncio.run(execute_local_reference_shot(shot))
    if padding > 0:
        body["trajectory"] = body["trajectory"] * padding
    public_sim = {k: v for k, v in body.items() if k != "trajectory_points"}
    return {
        "capability": "golf_simulator_golden_shot",
        "golden_case_id": "agent_api_golf_shot_v1",
        "qualification_banner": {
            "zvcf_definition": None,
            "qualification_state": {
                "destination": "local_reference",
                "contact": shot.qualification.contact.value,
                "numerical": shot.qualification.numerical.value,
                "scientific": shot.qualification.scientific.value,
            },
            "model_limits": [
                "Ball-flight ODE reference simulator; not GSPro or live launch monitor.",
            ],
        },
        "shot": {
            "shot_id": shot.shot_id,
            "session_id": shot.session_id,
            "ball_velocity_m_s": list(shot.ball_velocity_m_s),
            "ball_angular_velocity_rad_s": list(shot.ball_angular_velocity_rad_s),
        },
        "simulation": public_sim,
    }


def _receipt_dict(receipt: SubmissionReceipt) -> dict[str, Any]:
    return {
        "shot_id": receipt.shot_id,
        "session_id": receipt.session_id,
        "state": receipt.state.value,
        "destination_id": receipt.destination_id,
        "attempt_id": receipt.attempt_id,
        "timestamp_utc": receipt.timestamp_utc,
        "detail": receipt.detail,
    }
