"""Opt-in live acceptance qualification test suite for golf simulator integration (GS-09, #10198).

Verifies the full end-to-end acceptance runbook sequence against an active simulator listener
(real licensed GSPro or simulated reference TCP peer in loopback):
1. Records and validates nonsecret environment, version, and license-mode receipt.
2. Demonstrates deterministic shot qualification: straight, left, right, chip, and putt.
3. Enforces honest capability boundaries: declares unsupported features (native avatar, course feedback).
4. Exercises ambiguous disconnect injection, verifying no automatic duplicate shots are submitted
   and durable journal retention is preserved.
5. Verifies single-producer active lease locking and conflict rejection.

Opt-in Contract:
This module is skipped cleanly by default unless GSPRO_LIVE_TEST=1 is present in the environment.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
from typing import Any
import pytest

pytestmark = [pytest.mark.integration]


@pytest.fixture(autouse=True)
def _require_live_test_opt_in() -> None:
    """Ensure tests are strictly opt-in and skip cleanly in standard CI."""
    if os.environ.get("GSPRO_LIVE_TEST") not in ("1", "true", "TRUE"):
        pytest.skip(
            "Opt-in live acceptance tests skipped. Set GSPRO_LIVE_TEST=1 to run."
        )


import json
from src.shared.python.golf_simulator.adapters.gspro.codec import (
    ResponseCategory,
    encode_shot_payload,
)
from src.shared.python.golf_simulator.adapters.gspro.profile import (
    DEFAULT_GSPRO_PROFILE,
)
from src.shared.python.golf_simulator.adapters.gspro.transport import (
    GSProTransport,
    TransportError,
)


from src.shared.python.golf_simulator.contracts import (
    AimContext,
    CapabilityState,
    ContactStatus,
    NumericalStatus,
    ScientificStatus,
    ShotEnvelope,
    ShotQualification,
    SourceKind,
    SubmissionState,
)
from src.shared.python.golf_simulator.discovery import (
    SupportReceipt,
    discover_simulator_installation,
)
from src.shared.python.golf_simulator.journal import (
    DeliveryStatus,
    ShotJournal,
)
from src.shared.python.golf_simulator.producer_lock import (
    ProducerConflictError,
    ProducerLockManager,
)
from tests.integration.golf_simulator.test_fake_gspro_peer import FakeGSProPeer

pytestmark = [pytest.mark.integration]


def _make_qualified_shot(
    shot_id: str,
    ball_velocity_m_s: tuple[float, float, float],
    ball_angular_velocity_rad_s: tuple[float, float, float],
) -> ShotEnvelope:
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
        evidence_refs=("acceptance_runbook_live_qualification",),
    )
    return ShotEnvelope(
        schema_version=1,
        shot_id=shot_id,
        session_id="session-live-acceptance",
        source_kind=SourceKind.MODEL_CONTACT,
        qualification=qual,
        ball_velocity_m_s=ball_velocity_m_s,
        ball_angular_velocity_rad_s=ball_angular_velocity_rad_s,
        aim_context=aim,
        created_at_utc="2026-09-16T12:00:00Z",
        model_run_id="run-live-acceptance-1",
        trace_digest="sha256-live-trace-1",
        impact_id="impact-1",
        impact_time_s=1.234,
    )


@pytest.mark.asyncio
async def test_live_environment_and_version_receipt() -> None:
    """Acceptance Step 1: Record and validate nonsecret environment, version, and license receipt."""
    config = discover_simulator_installation()
    receipt = config.build_support_receipt(
        app_version="2026.1.0",
        connector_version="1.0.0",
        optional_packages={"pywin32": True, "psutil": True},
    )
    assert isinstance(receipt, SupportReceipt)
    receipt_data = receipt.to_dict()

    # Invariants: no secrets in audit receipt
    secret_terms = ("token", "secret", "password", "key", "credential", "auth")
    for term in secret_terms:
        assert term not in receipt_data, (
            f"Secret term {term} leaked into support receipt"
        )

    assert receipt_data["platform"] == sys.platform
    assert receipt_data["connector_version"] == "1.0.0"
    assert receipt_data["profile_version"] == "v1"


@pytest.mark.asyncio
async def test_live_deterministic_shot_qualification(tmp_path: Path) -> None:
    """Acceptance Step 2: Qualify straight, left/hook, right/slice, chip, and putt."""

    port_str = os.environ.get("GSPRO_API_PORT")
    peer: FakeGSProPeer | None = None
    if port_str:
        host = os.environ.get("GSPRO_HOST", "127.0.0.1")
        port = int(port_str)
    else:
        peer = FakeGSProPeer(host="127.0.0.1")
        await peer.start()
        host = "127.0.0.1"
        port = peer.port

    journal_path = tmp_path / "live_acceptance_journal.json"
    journal = ShotJournal(storage_path=journal_path)
    transport = GSProTransport(host=host, port=port, journal=journal)

    try:
        await transport.connect()
        assert transport.is_connected

        # Scenario definitions: (label, velocity_xyz_m_s, angular_velocity_xyz_rad_s)
        # +x forward, +y left, +z up
        scenarios = [
            ("straight", (70.0, 0.0, 15.0), (0.0, -260.0, 0.0)),
            ("left_hook", (68.0, 5.0, 14.0), (0.0, -250.0, -60.0)),
            ("right_slice", (66.0, -6.0, 16.0), (0.0, -270.0, 80.0)),
            ("chip", (22.0, 0.0, 10.0), (0.0, -400.0, 0.0)),
            ("putt", (3.5, 0.0, 0.2), (0.0, -30.0, 0.0)),
        ]

        for label, vel, avel in scenarios:
            shot_id = f"shot-live-{label}"
            envelope = _make_qualified_shot(shot_id, vel, avel)
            payload = json.dumps(
                encode_shot_payload(envelope, DEFAULT_GSPRO_PROFILE)
            ).encode("utf-8")

            receipt = await transport.send_shot(shot_id, payload)
            assert receipt.category == ResponseCategory.CONFIRMED_ACCEPTED
            assert receipt.code == 200

            # Verify entry in durable journal
            entry = journal.get_entry(shot_id)
            assert entry.status == DeliveryStatus.ACKNOWLEDGED
            assert entry.response_code == 200
            assert entry.shot_id == shot_id
    finally:
        await transport.disconnect()
        if peer is not None:
            await peer.stop()


@pytest.mark.asyncio
async def test_live_unsupported_capability_declarations() -> None:
    """Acceptance Step 3: Explicitly declare unsupported cases (native avatar, course telemetry)."""
    from src.shared.python.golf_simulator.adapters.local import LocalReferenceAdapter

    adapter = LocalReferenceAdapter()
    caps = adapter.capabilities()

    # Honest declarations
    assert caps.shot_input.state == CapabilityState.SUPPORTED
    assert caps.club_data.state == CapabilityState.SUPPORTED
    assert caps.native_avatar_animation.state == CapabilityState.UNSUPPORTED
    assert caps.course_state_feedback.state == CapabilityState.UNSUPPORTED


@pytest.mark.asyncio
async def test_live_ambiguous_disconnect_recovery(tmp_path: Path) -> None:
    """Acceptance Step 4: Inject transport drop and assert AMBIGUOUS journal state with no duplicate shots."""
    peer = FakeGSProPeer(host="127.0.0.1")
    peer.behavior = "drop"
    await peer.start()

    journal_path = tmp_path / "live_disconnect_journal.json"
    journal = ShotJournal(storage_path=journal_path)
    transport = GSProTransport(host="127.0.0.1", port=peer.port, journal=journal)

    await transport.connect()
    shot_id = "shot-ambiguous-disconnect-1"
    envelope = _make_qualified_shot(shot_id, (70.0, 0.0, 15.0), (0.0, -260.0, 0.0))
    payload = json.dumps(encode_shot_payload(envelope, DEFAULT_GSPRO_PROFILE)).encode(
        "utf-8"
    )

    with pytest.raises(TransportError, match="Connection closed"):
        await transport.send_shot(shot_id, payload)

    entry = journal.get_entry(shot_id)
    assert entry.status == DeliveryStatus.AMBIGUOUS

    # Ensure no automatic retry occurred
    assert len(peer.received_payloads) == 1

    # Ensure recovery on reload maintains AMBIGUOUS status
    reloaded_journal = ShotJournal(storage_path=journal_path, auto_recover=True)
    assert reloaded_journal.get_entry(shot_id).status == DeliveryStatus.AMBIGUOUS

    # Ensure retention pruning preserves ambiguous record
    reloaded_journal.prune_retention(max_age_seconds=0.0, keep_unresolved=True)
    assert reloaded_journal.get_entry(shot_id) is not None

    await transport.disconnect()
    await peer.stop()


@pytest.mark.asyncio
async def test_live_single_producer_lock_prevents_collision() -> None:
    """Acceptance Step 5: Verify single-producer active lease locking prevents multiple competing callers."""
    manager = ProducerLockManager()
    lock = manager.acquire(
        producer_id="primary-live-runner",
        session_id="session-live-1",
        ttl_seconds=30.0,
    )
    assert manager.is_locked()
    assert manager.current_producer() == "primary-live-runner"

    # Competing producer must be rejected
    with pytest.raises(ProducerConflictError) as exc_info:
        manager.acquire(
            producer_id="secondary-launch-monitor",
            session_id="session-live-1",
            ttl_seconds=10.0,
        )

    assert exc_info.value.active_producer_id == "primary-live-runner"

    # Releasing active lock frees session
    manager.release(lock)
    assert not manager.is_locked()
