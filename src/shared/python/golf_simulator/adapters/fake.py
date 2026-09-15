"""Test spy and configurable fake simulator adapter for contract verification.

Follows TDD, DbC, and DRY.
"""

from __future__ import annotations

import asyncio
import datetime
import uuid
from collections.abc import AsyncIterator
from typing import Any

from src.shared.python.golf_simulator.contracts import (
    CapabilityDescriptor,
    CapabilityState,
    ConnectionState,
    ConnectionStatus,
    ShotEnvelope,
    SimulatorAdapter,
    SimulatorCapabilities,
    SimulatorEvent,
    SubmissionReceipt,
    SubmissionState,
)


class FakeSimulatorAdapter:
    """Configurable fake and spy simulator adapter for unit and contract testing."""

    def __init__(
        self,
        capabilities: SimulatorCapabilities | None = None,
        destination_id: str = "fake_destination",
        default_receipt_state: SubmissionState = SubmissionState.CONFIRMED_ACCEPTED,
        should_fail_connection: bool = False,
        fail_before_send: bool = False,
        simulate_delay_s: float = 0.0,
    ) -> None:
        self._capabilities = capabilities or SimulatorCapabilities(
            shot_input=CapabilityDescriptor(
                state=CapabilityState.SUPPORTED,
                evidence="Fake adapter for testing",
            ),
            club_data=CapabilityDescriptor(
                state=CapabilityState.SUPPORTED,
                evidence="Fake adapter club data support",
            ),
            local_trajectory_return=CapabilityDescriptor(
                state=CapabilityState.UNSUPPORTED,
                evidence="Fake adapter does not simulate flight",
            ),
            native_avatar_animation=CapabilityDescriptor(
                state=CapabilityState.UNSUPPORTED,
                evidence="Not supported",
            ),
            course_state_feedback=CapabilityDescriptor(
                state=CapabilityState.UNSUPPORTED,
                evidence="Not supported",
            ),
            aim_control=CapabilityDescriptor(
                state=CapabilityState.UNSUPPORTED,
                evidence="Not supported",
            ),
        )
        self._destination_id = destination_id
        self._default_receipt_state = default_receipt_state
        self._should_fail_connection = should_fail_connection
        self._fail_before_send = fail_before_send
        self._simulate_delay_s = simulate_delay_s

        # Spies
        self.connect_calls: list[dict[str, Any]] = []
        self.disconnect_calls: int = 0
        self.submitted_shots: list[ShotEnvelope] = []
        self.connected: bool = False

    def capabilities(self) -> SimulatorCapabilities:
        return self._capabilities

    async def connect(self, config: dict[str, Any]) -> ConnectionStatus:
        self.connect_calls.append(dict(config))
        if self._should_fail_connection:
            self.connected = False
            return ConnectionStatus(
                state=ConnectionState.FAILED,
                endpoint=self._destination_id,
                message="Simulated connection failure",
            )
        self.connected = True
        return ConnectionStatus(
            state=ConnectionState.CONNECTED,
            endpoint=self._destination_id,
            message="Fake adapter connected",
        )

    async def submit(self, shot: ShotEnvelope) -> SubmissionReceipt:
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        attempt_id = str(uuid.uuid4())

        if not self.connected or self._fail_before_send:
            return SubmissionReceipt(
                shot_id=shot.shot_id,
                session_id=shot.session_id,
                state=SubmissionState.FAILED_BEFORE_SEND,
                destination_id=self._destination_id,
                attempt_id=attempt_id,
                timestamp_utc=now_utc,
                detail="Fake adapter not connected or pre-send failure",
            )

        if self._simulate_delay_s > 0.0:
            await asyncio.sleep(self._simulate_delay_s)

        self.submitted_shots.append(shot)

        return SubmissionReceipt(
            shot_id=shot.shot_id,
            session_id=shot.session_id,
            state=self._default_receipt_state,
            destination_id=self._destination_id,
            attempt_id=attempt_id,
            timestamp_utc=now_utc,
            detail=f"Fake submission outcome: {self._default_receipt_state.value}",
        )

    async def events(self) -> AsyncIterator[SimulatorEvent]:
        if False:
            yield  # pragma: no cover

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False
