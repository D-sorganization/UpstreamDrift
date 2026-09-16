"""Tests for the Flight Relay Protocol adapter and second simulator evaluation (GS-10, #10199).

Verifies protocol conformance, honest capability declarations, simulated relay transport,
disconnect handling, and clear rejection of unsupported proprietary targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest

from src.shared.python.golf_simulator.adapters.relay import (
    FlightRelayAdapter,
    FlightRelayConfig,
    UnsupportedDestinationError,
)
from src.shared.python.golf_simulator.contracts import (
    AimContext,
    CapabilityState,
    ClubData,
    ConnectionState,
    ContactStatus,
    NumericalStatus,
    ScientificStatus,
    ShotEnvelope,
    ShotQualification,
    SimulatorAdapter,
    SourceKind,
    SubmissionState,
)
from src.shared.python.golf_simulator.journal import DeliveryStatus, ShotJournal

pytestmark = [pytest.mark.unit]


def _make_sample_shot(shot_id: str = "shot-relay-1") -> ShotEnvelope:
    identity_rot = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    aim = AimContext(source_to_target_rotation=identity_rot, revision=1)
    qual = ShotQualification(
        contact=ContactStatus.QUALIFIED,
        numerical=NumericalStatus.CONVERGED,
        scientific=ScientificStatus.BENCHMARKED,
        evidence_refs=("relay_test",),
    )
    club = ClubData(
        club_speed_m_s=45.0,
        club_path_rad=0.02,
        attack_angle_rad=-0.03,
        face_to_target_rad=0.01,
    )
    return ShotEnvelope(
        schema_version=1,
        shot_id=shot_id,
        session_id="session-relay-1",
        source_kind=SourceKind.MODEL_CONTACT,
        qualification=qual,
        ball_velocity_m_s=(68.0, 1.5, 14.0),
        ball_angular_velocity_rad_s=(0.0, -250.0, 30.0),
        club_data=club,
        aim_context=aim,
        created_at_utc="2026-09-16T12:00:00Z",
        model_run_id="run-relay-1",
        trace_digest="sha256-dummy-relay-trace",
        impact_id="impact-relay-1",
        impact_time_s=1.234,
    )


def test_flight_relay_adapter_satisfies_protocol() -> None:
    """Verify FlightRelayAdapter implements the SimulatorAdapter runtime protocol."""
    adapter = FlightRelayAdapter()
    assert isinstance(adapter, SimulatorAdapter)


def test_flight_relay_adapter_capabilities() -> None:
    """Verify honest capability declarations for Flight Relay Protocol."""
    adapter = FlightRelayAdapter()
    caps = adapter.capabilities()

    # Supported capabilities
    assert caps.shot_input.state == CapabilityState.SUPPORTED
    assert caps.club_data.state == CapabilityState.SUPPORTED

    # Truthfully declared unsupported capabilities
    assert caps.native_avatar_animation.state == CapabilityState.UNSUPPORTED
    assert caps.course_state_feedback.state == CapabilityState.UNSUPPORTED
    assert caps.local_trajectory_return.state == CapabilityState.UNSUPPORTED
    assert caps.aim_control.state == CapabilityState.UNSUPPORTED


@pytest.mark.asyncio
async def test_flight_relay_lifecycle() -> None:
    """Verify connect and disconnect state transitions."""
    adapter = FlightRelayAdapter()
    assert not adapter.is_connected

    status = await adapter.connect()
    assert status.state == ConnectionState.CONNECTED
    assert adapter.is_connected

    await adapter.disconnect()
    assert not adapter.is_connected


@pytest.mark.asyncio
async def test_flight_relay_submit_shot(tmp_path: Path) -> None:
    """Verify successful shot submission through Flight Relay adapter."""
    journal_path = tmp_path / "relay_journal.json"
    journal = ShotJournal(storage_path=journal_path)
    adapter = FlightRelayAdapter(journal=journal)

    await adapter.connect()
    shot = _make_sample_shot("shot-test-submit-1")

    receipt = await adapter.submit(shot)
    assert receipt.state == SubmissionState.CONFIRMED_ACCEPTED
    assert receipt.destination_id == "flight_relay"

    entry = journal.get_entry("shot-test-submit-1")
    assert entry.status == DeliveryStatus.ACKNOWLEDGED
    assert entry.response_code == 200

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_flight_relay_unsupported_destination_rejection() -> None:
    """Verify proprietary simulator targets without verified SDKs are explicitly rejected."""
    unsupported_targets = ["e6_connect", "creative_golf", "trackman", "fsx_pro"]

    for target in unsupported_targets:
        config = FlightRelayConfig(destination_name=target)
        adapter = FlightRelayAdapter(config=config)

        with pytest.raises(UnsupportedDestinationError) as exc_info:
            await adapter.connect()

        assert target in str(exc_info.value)
        assert not adapter.is_connected
