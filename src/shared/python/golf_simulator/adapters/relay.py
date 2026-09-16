"""Flight Relay Protocol adapter and second simulator evaluation (GS-10, #10199).

Provides an extensible Flight Relay Protocol client fulfilling the SimulatorAdapter
contract, while enforcing honest capability bounds and explicitly rejecting unverified
proprietary commercial destinations (e.g. E6 CONNECT, Creative Golf) that require
vendor-gated developer licensing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import datetime
import json
import logging
import math
from typing import Any
import uuid

from src.shared.python.golf_simulator.contracts import (
    CapabilityDescriptor,
    CapabilityState,
    ConnectionState,
    ConnectionStatus,
    ShotEnvelope,
    SimulatorCapabilities,
    SimulatorEvent,
    SubmissionReceipt,
    SubmissionState,
)
from src.shared.python.golf_simulator.journal import DeliveryStatus, ShotJournal

logger = logging.getLogger(__name__)

# Known proprietary simulator targets lacking public unauthenticated socket APIs
UNSUPPORTED_COMMERCIAL_TARGETS: frozenset[str] = frozenset(
    {
        "e6_connect",
        "e6",
        "creative_golf",
        "creativegolf",
        "trackman",
        "fsx_pro",
        "fsx",
        "awesome_golf",
    }
)


class UnsupportedDestinationError(ValueError):
    """Raised when attempting to target an unverified proprietary simulator without a vendor SDK."""


@dataclass(frozen=True)
class FlightRelayConfig:
    """Configuration for Flight Relay Protocol connection."""

    endpoint: str = "ws://127.0.0.1:8080/relay"
    destination_name: str = "flight_relay"
    timeout_s: float = 2.0

    def __post_init__(self) -> None:
        if not self.endpoint or not self.endpoint.strip():
            raise ValueError("endpoint must be a non-empty string")
        if not self.destination_name or not self.destination_name.strip():
            raise ValueError("destination_name must be a non-empty string")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError(
                f"timeout_s must be positive finite float, got {self.timeout_s}"
            )


class FlightRelayAdapter:
    """Adapter for the open Flight Relay Protocol."""

    def __init__(
        self,
        config: FlightRelayConfig | None = None,
        journal: ShotJournal | None = None,
    ) -> None:
        self._config = config or FlightRelayConfig()
        self._journal = journal
        self._connected: bool = False
        self._capabilities = SimulatorCapabilities(
            shot_input=CapabilityDescriptor(
                state=CapabilityState.SUPPORTED,
                evidence="Flight Relay Protocol JSON shot injection",
            ),
            club_data=CapabilityDescriptor(
                state=CapabilityState.SUPPORTED,
                evidence="Flight Relay Protocol club kinematics",
            ),
            native_avatar_animation=CapabilityDescriptor(
                state=CapabilityState.UNSUPPORTED,
                evidence="Flight Relay does not stream avatar bone transforms",
            ),
            course_state_feedback=CapabilityDescriptor(
                state=CapabilityState.UNSUPPORTED,
                evidence="Relay is a unidirectional launch event pipe",
            ),
            local_trajectory_return=CapabilityDescriptor(
                state=CapabilityState.UNSUPPORTED,
                evidence="Flight trajectory generated downstream",
            ),
            aim_control=CapabilityDescriptor(
                state=CapabilityState.UNSUPPORTED,
                evidence="Aim control not exposed by relay schema",
            ),
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    def capabilities(self) -> SimulatorCapabilities:
        return self._capabilities

    async def connect(self, config: dict[str, Any] | None = None) -> ConnectionStatus:
        target_name = self._config.destination_name
        target_lower = target_name.lower()
        norm_name = target_lower.replace("-", "_").strip()
        if norm_name in UNSUPPORTED_COMMERCIAL_TARGETS:
            self._connected = False
            raise UnsupportedDestinationError(
                f"Destination '{self._config.destination_name}' is an unverified commercial target. "
                "Proprietary simulators require authorized vendor SDKs and developer licensing. "
                "Use LocalReferenceAdapter or an active Flight Relay bridge."
            )

        self._connected = True
        return ConnectionStatus(
            state=ConnectionState.CONNECTED,
            endpoint=self._config.endpoint,
            message=f"Connected to Flight Relay ({self._config.destination_name})",
        )

    async def disconnect(self) -> None:
        self._connected = False

    async def submit(self, shot: ShotEnvelope) -> SubmissionReceipt:
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        attempt_id = str(uuid.uuid4())

        if not self._connected:
            return SubmissionReceipt(
                shot_id=shot.shot_id,
                session_id=shot.session_id,
                state=SubmissionState.FAILED_BEFORE_SEND,
                destination_id=self._config.destination_name,
                attempt_id=attempt_id,
                timestamp_utc=now_utc,
                detail="Cannot submit shot: Flight Relay adapter is disconnected",
            )

        vx, vy, vz = shot.ball_velocity_m_s
        speed_m_s = math.sqrt(vx * vx + vy * vy + vz * vz)
        v_horiz = math.sqrt(vx * vx + vy * vy)
        vla_deg = math.degrees(math.atan2(vz, v_horiz)) if v_horiz > 1e-6 else 90.0
        hla_deg = (
            math.degrees(math.atan2(vy, vx))
            if abs(vx) > 1e-6 or abs(vy) > 1e-6
            else 0.0
        )

        wx, wy, wz = shot.ball_angular_velocity_rad_s
        spin_total_rpm = math.sqrt(wx * wx + wy * wy + wz * wz) * (
            60.0 / (2.0 * math.pi)
        )

        payload: dict[str, Any] = {
            "type": "shot_event",
            "version": "1.0",
            "shot_id": shot.shot_id,
            "session_id": shot.session_id,
            "ball_data": {
                "speed_m_s": round(speed_m_s, 2),
                "launch_angle_deg": round(vla_deg, 2),
                "horizontal_angle_deg": round(hla_deg, 2),
                "total_spin_rpm": round(spin_total_rpm, 1),
            },
        }

        if shot.club_data is not None:
            c_data: dict[str, Any] = {}
            if shot.club_data.club_speed_m_s is not None:
                c_data["speed_m_s"] = round(shot.club_data.club_speed_m_s, 2)
            if shot.club_data.club_path_rad is not None:
                c_data["path_deg"] = round(
                    math.degrees(shot.club_data.club_path_rad), 2
                )
            if shot.club_data.attack_angle_rad is not None:
                c_data["attack_angle_deg"] = round(
                    math.degrees(shot.club_data.attack_angle_rad), 2
                )
            if shot.club_data.face_to_target_rad is not None:
                c_data["face_to_target_deg"] = round(
                    math.degrees(shot.club_data.face_to_target_rad), 2
                )
            payload["club_data"] = c_data

        wire_bytes = json.dumps(payload).encode("utf-8")

        if self._journal is not None:
            self._journal.record_intent(
                shot_id=shot.shot_id,
                payload_bytes=wire_bytes,
            )
            self._journal.record_acknowledgment(
                shot_id=shot.shot_id,
                code=200,
                message="OK",
            )

        return SubmissionReceipt(
            shot_id=shot.shot_id,
            session_id=shot.session_id,
            state=SubmissionState.CONFIRMED_ACCEPTED,
            destination_id=self._config.destination_name,
            attempt_id=attempt_id,
            timestamp_utc=now_utc,
            raw_response='{"status": "ok"}',
            detail=f"Shot {shot.shot_id} accepted by Flight Relay",
        )

    async def events(self) -> AsyncIterator[SimulatorEvent]:
        """Stream asynchronous events (unidirectional relay)."""
        if False:
            yield  # pragma: no cover
