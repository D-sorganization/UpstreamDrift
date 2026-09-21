"""Unit tests for GolfSessionService.

Follows TDD, DbC, Law of Demeter, and DRY.
"""

from __future__ import annotations

import pytest

from src.shared.python.golf_simulator.adapters.fake import FakeSimulatorAdapter
from src.shared.python.golf_simulator.contracts import (
    AimContext,
    CapabilityDescriptor,
    CapabilityState,
    ContactStatus,
    NumericalStatus,
    ScientificStatus,
    SessionState,
    ShotEnvelope,
    ShotQualification,
    SimulatorCapabilities,
    SourceKind,
    SubmissionState,
)
from src.shared.python.golf_simulator.journal import ShotJournal
from src.shared.python.golf_simulator.session import GolfSessionService

pytestmark = pytest.mark.unit


def _make_shot(
    shot_id: str = "shot-001",
    session_id: str = "sess-001",
    source_kind: SourceKind = SourceKind.MANUAL,
    contact_status: ContactStatus = ContactStatus.QUALIFIED,
) -> ShotEnvelope:
    identity_rot = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    aim = AimContext(source_to_target_rotation=identity_rot, revision=1)
    qual = ShotQualification(
        contact=contact_status,
        numerical=NumericalStatus.CONVERGED,
        scientific=ScientificStatus.BENCHMARKED,
        evidence_refs=("unit_test",),
    )
    is_model = source_kind in (SourceKind.MODEL_CONTACT, SourceKind.DEMO_PEAK_SPEED)
    model_run_id = "run-42" if is_model else None
    trace_digest = "sha256:abcd" if is_model else None
    impact_id = "impact-001" if is_model else None
    impact_time_s = 0.25 if is_model else None

    return ShotEnvelope(
        schema_version=1,
        shot_id=shot_id,
        session_id=session_id,
        source_kind=source_kind,
        qualification=qual,
        ball_velocity_m_s=(60.0, 0.0, 15.0),
        ball_angular_velocity_rad_s=(0.0, -250.0, 0.0),
        aim_context=aim,
        created_at_utc="2026-09-15T12:00:00Z",
        model_run_id=model_run_id,
        trace_digest=trace_digest,
        impact_id=impact_id,
        impact_time_s=impact_time_s,
    )


@pytest.mark.asyncio
async def test_session_service_initial_state_idle() -> None:
    adapter = FakeSimulatorAdapter()
    service = GolfSessionService(session_id="sess-001", adapter=adapter)
    assert service.current_state == SessionState.IDLE
    assert service.session_id == "sess-001"


@pytest.mark.asyncio
async def test_session_service_prepare_arm_submit_happy_path() -> None:
    adapter = FakeSimulatorAdapter()
    service = GolfSessionService(session_id="sess-001", adapter=adapter)
    await service.select_destination(adapter)

    shot = _make_shot(shot_id="shot-01", session_id="sess-001")
    prep = service.prepare(shot, context_revision=1)
    assert service.current_state == SessionState.PREPARED
    assert prep.shot == shot

    token = service.arm(prep.prepared_shot_id, context_revision=1)
    assert service.current_state == SessionState.ARMED
    assert token is not None

    receipt = await service.submit_at_impact(prep.prepared_shot_id, arm_token=token)
    assert receipt.state == SubmissionState.CONFIRMED_ACCEPTED
    assert receipt.shot_id == "shot-01"
    assert service.current_state == SessionState.IDLE
    assert len(adapter.submitted_shots) == 1


@pytest.mark.asyncio
async def test_session_service_rejects_destination_switch_when_armed() -> None:
    adapter1 = FakeSimulatorAdapter(destination_id="dest1")
    adapter2 = FakeSimulatorAdapter(destination_id="dest2")
    service = GolfSessionService(session_id="sess-001", adapter=adapter1)
    await service.select_destination(adapter1)

    shot = _make_shot(session_id="sess-001")
    prep = service.prepare(shot, context_revision=1)
    service.arm(prep.prepared_shot_id, context_revision=1)

    # In-flight destination change must be rejected
    with pytest.raises(RuntimeError, match="Cannot switch destination"):
        await service.select_destination(adapter2)


@pytest.mark.asyncio
async def test_session_service_destination_switch_invalidates_prepared_shot() -> None:
    adapter1 = FakeSimulatorAdapter(destination_id="dest1")
    adapter2 = FakeSimulatorAdapter(destination_id="dest2")
    service = GolfSessionService(session_id="sess-001", adapter=adapter1)
    await service.select_destination(adapter1)

    shot = _make_shot(session_id="sess-001")
    prep = service.prepare(shot, context_revision=1)
    assert service.current_state == SessionState.PREPARED

    # Switching destination resets state to IDLE and invalidates preparation
    await service.select_destination(adapter2)
    assert service.current_state == SessionState.IDLE

    # Arming the old prepared shot must fail
    with pytest.raises(RuntimeError, match="No prepared shot"):
        service.arm(prep.prepared_shot_id, context_revision=1)


@pytest.mark.asyncio
async def test_session_service_arm_rejects_context_revision_mismatch() -> None:
    adapter = FakeSimulatorAdapter()
    service = GolfSessionService(session_id="sess-001", adapter=adapter)
    await service.select_destination(adapter)

    shot = _make_shot(session_id="sess-001")
    prep = service.prepare(shot, context_revision=1)

    # Attempting to arm with a different revision (e.g. user adjusted aim)
    with pytest.raises(ValueError, match="Context revision mismatch"):
        service.arm(prep.prepared_shot_id, context_revision=2)


@pytest.mark.asyncio
async def test_session_service_submit_rejects_invalid_arm_token() -> None:
    adapter = FakeSimulatorAdapter()
    service = GolfSessionService(session_id="sess-001", adapter=adapter)
    await service.select_destination(adapter)

    shot = _make_shot(session_id="sess-001")
    prep = service.prepare(shot, context_revision=1)
    service.arm(prep.prepared_shot_id, context_revision=1)

    with pytest.raises(ValueError, match="Invalid arm token"):
        await service.submit_at_impact(prep.prepared_shot_id, arm_token="wrong-token")

    # Spy proves no write was dispatched to the adapter on token failure
    assert len(adapter.submitted_shots) == 0


@pytest.mark.asyncio
async def test_session_service_rejects_unsupported_destination_capabilities() -> None:
    unsupported_caps = SimulatorCapabilities(
        shot_input=CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="Shot input not supported by this mock",
        ),
        club_data=CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="None",
        ),
        local_trajectory_return=CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="None",
        ),
        native_avatar_animation=CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="None",
        ),
        course_state_feedback=CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="None",
        ),
        aim_control=CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="None",
        ),
    )
    adapter = FakeSimulatorAdapter(capabilities=unsupported_caps)
    service = GolfSessionService(session_id="sess-001", adapter=adapter)
    await service.select_destination(adapter)

    shot = _make_shot(session_id="sess-001")
    with pytest.raises(ValueError, match="Destination does not support shot input"):
        service.prepare(shot, context_revision=1)


@pytest.mark.asyncio
async def test_session_service_model_qualification_enforcement() -> None:
    adapter = FakeSimulatorAdapter()
    service = GolfSessionService(session_id="sess-001", adapter=adapter)
    await service.select_destination(adapter)

    # MODEL_CONTACT with UNVERIFIED contact status must be rejected
    unverified_shot = _make_shot(
        session_id="sess-001",
        source_kind=SourceKind.MODEL_CONTACT,
        contact_status=ContactStatus.UNVERIFIED,
    )
    with pytest.raises(ValueError, match="Model shot requires qualified contact"):
        service.prepare(unverified_shot, context_revision=1)


@pytest.mark.asyncio
async def test_session_service_disarm_and_cancel() -> None:
    adapter = FakeSimulatorAdapter()
    service = GolfSessionService(session_id="sess-001", adapter=adapter)
    await service.select_destination(adapter)

    shot = _make_shot(session_id="sess-001")
    prep = service.prepare(shot, context_revision=1)
    token = service.arm(prep.prepared_shot_id, context_revision=1)
    assert service.current_state == SessionState.ARMED

    # Disarm transitions back to PREPARED
    service.disarm(prep.prepared_shot_id)
    assert service.current_state == SessionState.PREPARED

    # Submitting after disarm must fail
    with pytest.raises(RuntimeError, match="Session is not armed"):
        await service.submit_at_impact(prep.prepared_shot_id, arm_token=token)

    # Cancel transitions back to IDLE
    service.cancel(prep.prepared_shot_id)
    assert service.current_state == SessionState.IDLE


@pytest.mark.asyncio
async def test_session_service_uncertain_delivery_and_reconciliation() -> None:
    adapter = FakeSimulatorAdapter(
        default_receipt_state=SubmissionState.UNKNOWN_AMBIGUOUS
    )
    service = GolfSessionService(session_id="sess-001", adapter=adapter)
    await service.select_destination(adapter)

    shot = _make_shot(shot_id="shot-ambiguous", session_id="sess-001")
    prep = service.prepare(shot, context_revision=1)
    token = service.arm(prep.prepared_shot_id, context_revision=1)

    receipt = await service.submit_at_impact(prep.prepared_shot_id, arm_token=token)
    assert receipt.state == SubmissionState.UNKNOWN_AMBIGUOUS
    assert service.current_state == SessionState.UNCERTAIN

    # Subsequent preparation is blocked while uncertain
    shot2 = _make_shot(shot_id="shot-02", session_id="sess-001")
    with pytest.raises(RuntimeError, match="Session is in UNCERTAIN state"):
        service.prepare(shot2, context_revision=1)

    # Operator resolves uncertain submission
    reconciled = service.resolve_uncertain(
        shot_id="shot-ambiguous",
        operator_evidence="Operator observed flight onset on screen",
        confirmed=True,
    )
    assert reconciled.state == SubmissionState.CONFIRMED_ACCEPTED
    assert service.current_state == SessionState.IDLE


@pytest.mark.asyncio
async def test_session_service_journal_integration(tmp_path) -> None:
    journal = ShotJournal(storage_path=tmp_path / "journal.json")
    adapter = FakeSimulatorAdapter()
    service = GolfSessionService(
        session_id="sess-001",
        adapter=adapter,
        journal=journal,
    )
    await service.select_destination(adapter)

    shot = _make_shot(shot_id="shot-j-01", session_id="sess-001")
    prep = service.prepare(shot, context_revision=1)
    token = service.arm(prep.prepared_shot_id, context_revision=1)

    receipt = await service.submit_at_impact(prep.prepared_shot_id, arm_token=token)
    assert receipt.state == SubmissionState.CONFIRMED_ACCEPTED

    status = service.delivery_status("shot-j-01")
    assert status is not None
    assert status.shot_id == "shot-j-01"
    assert status.state == SubmissionState.CONFIRMED_ACCEPTED
