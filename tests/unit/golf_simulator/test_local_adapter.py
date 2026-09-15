"""Unit tests for LocalReferenceAdapter.

Follows TDD, DbC, Law of Demeter, and DRY.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from src.shared.python.golf_simulator.adapters.local import (
    LocalReferenceAdapter,
    TrajectoryRecord,
)
from src.shared.python.golf_simulator.contracts import (
    AimContext,
    CapabilityState,
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
from src.shared.python.physics.ball_launch_conditions import LaunchConditions
from src.shared.python.physics.swing_ball_flight_pipeline import FlightSimulatorProtocol

pytestmark = pytest.mark.unit


def _make_shot(shot_id: str = "shot-local-01") -> ShotEnvelope:
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
        evidence_refs=("unit_test",),
    )
    return ShotEnvelope(
        schema_version=1,
        shot_id=shot_id,
        session_id="session-local-01",
        source_kind=SourceKind.MANUAL,
        qualification=qual,
        ball_velocity_m_s=(65.0, 0.0, 15.0),
        ball_angular_velocity_rad_s=(0.0, -250.0, 0.0),
        aim_context=aim,
        created_at_utc="2026-09-15T12:00:00Z",
    )


@pytest.mark.asyncio
async def test_local_reference_adapter_satisfies_protocol() -> None:
    adapter = LocalReferenceAdapter()
    assert isinstance(adapter, SimulatorAdapter)

    caps = adapter.capabilities()
    assert caps.shot_input.state == CapabilityState.SUPPORTED
    assert caps.club_data.state == CapabilityState.SUPPORTED
    assert caps.local_trajectory_return.state == CapabilityState.SUPPORTED
    assert caps.native_avatar_animation.state == CapabilityState.UNSUPPORTED
    assert caps.course_state_feedback.state == CapabilityState.UNSUPPORTED


@pytest.mark.asyncio
async def test_local_reference_adapter_connect_disconnect() -> None:
    adapter = LocalReferenceAdapter()
    status = await adapter.connect({})
    assert status.state == ConnectionState.CONNECTED
    assert "local" in status.endpoint

    await adapter.disconnect()
    # Capabilities remain readable even when disconnected
    assert adapter.capabilities().shot_input.state == CapabilityState.SUPPORTED


@pytest.mark.asyncio
async def test_local_reference_adapter_submit_generates_trajectory_and_receipt() -> (
    None
):
    # Use real BallFlightSimulator or mock flight simulator
    mock_sim = MagicMock(spec=FlightSimulatorProtocol)
    dummy_point = MagicMock()
    mock_sim.simulate_trajectory.return_value = [dummy_point, dummy_point]

    adapter = LocalReferenceAdapter(flight_simulator=mock_sim)
    await adapter.connect({})

    shot = _make_shot("shot-test-42")
    receipt = await adapter.submit(shot)

    assert receipt.shot_id == "shot-test-42"
    assert receipt.session_id == "session-local-01"
    assert receipt.state == SubmissionState.CONFIRMED_ACCEPTED
    assert receipt.destination_id == "local_reference"

    # Verify simulate_trajectory was called with converted LaunchConditions
    mock_sim.simulate_trajectory.assert_called_once()
    launch_arg = mock_sim.simulate_trajectory.call_args[0][0]
    assert isinstance(launch_arg, LaunchConditions)
    expected_speed = math.sqrt(65.0**2 + 15.0**2)
    assert math.isclose(launch_arg.velocity, expected_speed, rel_tol=1e-5)

    # Verify distinct trajectory storage with provenance
    traj = adapter.get_last_trajectory("shot-test-42")
    assert traj == [dummy_point, dummy_point]

    record = adapter.get_trajectory_record("shot-test-42")
    assert record is not None
    assert record.shot_id == "shot-test-42"
    assert "local_reference" in record.provenance


@pytest.mark.asyncio
async def test_local_reference_adapter_not_connected_fails_before_send() -> None:
    adapter = LocalReferenceAdapter()
    shot = _make_shot()
    receipt = await adapter.submit(shot)
    assert receipt.state == SubmissionState.FAILED_BEFORE_SEND
    assert "not connected" in receipt.detail.lower()
