"""Unit tests for authenticated remote application bridge topology (GS-08, #10197).

Follows TDD, DbC, Law of Demeter, and DRY.
"""

from __future__ import annotations

import asyncio
import pytest

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
from src.shared.python.golf_simulator.remote_bridge import (
    AuthenticationError,
    BridgeSecurityError,
    CancellationToken,
    LocalBridgeServer,
    QueueCapacityExceededError,
    RemoteBridgeClient,
)

pytestmark = pytest.mark.unit


def _make_dummy_shot() -> ShotEnvelope:
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
        evidence_refs=("contract_test",),
    )
    return ShotEnvelope(
        schema_version=1,
        shot_id="shot-test-remote-1",
        session_id="session-test-remote",
        source_kind=SourceKind.MANUAL,
        qualification=qual,
        ball_velocity_m_s=(68.0, 1.0, 12.0),
        ball_angular_velocity_rad_s=(0.0, -280.0, 20.0),
        aim_context=aim,
        created_at_utc="2026-09-15T12:00:00Z",
    )


def test_bridge_server_blocks_non_loopback_vendor_binding() -> None:
    with pytest.raises(
        BridgeSecurityError,
        match="Vendor simulator socket must not be exposed outside loopback",
    ):
        LocalBridgeServer(
            bridge_host="127.0.0.1",
            bridge_port=9220,
            auth_token="valid-secret-token",
            vendor_host="0.0.0.0",
            vendor_port=921,
        )


def test_bridge_server_requires_auth_token() -> None:
    with pytest.raises(ValueError, match="auth_token must be a non-empty string"):
        LocalBridgeServer(
            bridge_host="127.0.0.1",
            bridge_port=9220,
            auth_token="",
        )


@pytest.mark.asyncio
async def test_bridge_rejects_unauthenticated_request() -> None:
    server = LocalBridgeServer(
        bridge_host="127.0.0.1",
        bridge_port=9220,
        auth_token="expected-secret-token",
    )
    shot = _make_dummy_shot()

    # Call handle_request directly with invalid token
    with pytest.raises(
        AuthenticationError, match="Invalid or missing authentication token"
    ):
        await server.handle_submit_request(auth_token="wrong-token", shot=shot)


@pytest.mark.asyncio
async def test_bridge_bounded_queue_backpressure() -> None:
    server = LocalBridgeServer(
        bridge_host="127.0.0.1",
        bridge_port=9220,
        auth_token="secret",
        queue_capacity=2,
    )
    shot = _make_dummy_shot()

    # Fill queue to capacity
    await server.enqueue_shot("secret", shot)
    await server.enqueue_shot("secret", shot)

    # Exceeding queue capacity must raise QueueCapacityExceededError
    with pytest.raises(QueueCapacityExceededError, match="Bounded queue is full"):
        await server.enqueue_shot("secret", shot)


@pytest.mark.asyncio
async def test_cancellation_token_aborts_request() -> None:
    server = LocalBridgeServer(
        bridge_host="127.0.0.1",
        bridge_port=9220,
        auth_token="secret",
    )
    token = CancellationToken()
    token.cancel()

    shot = _make_dummy_shot()
    with pytest.raises(asyncio.CancelledError, match="Operation cancelled by caller"):
        await server.handle_submit_request(
            auth_token="secret",
            shot=shot,
            cancellation_token=token,
        )


@pytest.mark.asyncio
async def test_remote_bridge_client_facade() -> None:
    # Client communicates with server facade
    server = LocalBridgeServer(
        bridge_host="127.0.0.1",
        bridge_port=9220,
        auth_token="secret-token-123",
    )
    client = RemoteBridgeClient(
        server_endpoint="127.0.0.1:9220",
        auth_token="secret-token-123",
        bridge_server=server,
    )
    shot = _make_dummy_shot()

    receipt = await client.submit(shot)
    assert isinstance(receipt, SubmissionReceipt)
    assert receipt.shot_id == shot.shot_id
    assert receipt.state == SubmissionState.CONFIRMED_ACCEPTED
    assert client.capabilities().shot_input.state.value == "supported"


@pytest.mark.asyncio
async def test_bridge_server_delegates_to_vendor_adapter() -> None:
    from unittest.mock import AsyncMock
    from src.shared.python.golf_simulator.contracts import SimulatorAdapter

    mock_adapter = AsyncMock(spec=SimulatorAdapter)
    now = "2026-09-15T12:00:01Z"
    expected_receipt = SubmissionReceipt(
        shot_id="shot-adapter-1",
        session_id="session-adapter-1",
        state=SubmissionState.CONFIRMED_ACCEPTED,
        destination_id="gspro_v1",
        attempt_id="att-1",
        timestamp_utc=now,
        detail="Vendor acknowledged",
    )
    mock_adapter.submit.return_value = expected_receipt

    server = LocalBridgeServer(
        bridge_host="127.0.0.1",
        bridge_port=9220,
        auth_token="token123",
        adapter=mock_adapter,
    )
    shot = _make_dummy_shot()
    receipt = await server.handle_submit_request(auth_token="token123", shot=shot)

    assert receipt == expected_receipt
    mock_adapter.submit.assert_awaited_once_with(shot)


@pytest.mark.asyncio
async def test_remote_bridge_client_unreachable_endpoint() -> None:
    from src.shared.python.golf_simulator.contracts import ConnectionState

    # Port 65432 is not listening on localhost
    client = RemoteBridgeClient(
        server_endpoint="127.0.0.1:65432",
        auth_token="tok",
        bridge_server=None,
    )
    status = await client.connect()
    assert status.state == ConnectionState.FAILED

    with pytest.raises(
        ConnectionError, match="Remote bridge connection not established"
    ):
        await client.submit(_make_dummy_shot())
