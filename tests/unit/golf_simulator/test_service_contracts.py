"""Unified service contract test suite across simulator adapter implementations.

Follows TDD, DbC, Law of Demeter, and DRY.
Verifies adapter substitution does not alter shot identity or delivery integrity.
"""

from __future__ import annotations

from typing import Callable

import pytest

from src.shared.python.golf_simulator.adapters.fake import FakeSimulatorAdapter
from src.shared.python.golf_simulator.adapters.local import LocalReferenceAdapter
from src.shared.python.golf_simulator.contracts import (
    AimContext,
    ContactStatus,
    NumericalStatus,
    ScientificStatus,
    SessionState,
    ShotEnvelope,
    ShotQualification,
    SimulatorAdapter,
    SourceKind,
    SubmissionState,
)
from src.shared.python.golf_simulator.session import GolfSessionService

pytestmark = pytest.mark.unit


def _create_fake_adapter() -> SimulatorAdapter:
    return FakeSimulatorAdapter(destination_id="test_fake")


def _create_local_adapter() -> SimulatorAdapter:
    return LocalReferenceAdapter()


@pytest.mark.parametrize(
    "adapter_factory",
    [
        _create_fake_adapter,
        _create_local_adapter,
    ],
    ids=["fake_adapter", "local_reference_adapter"],
)
@pytest.mark.asyncio
async def test_service_contract_happy_path(
    adapter_factory: Callable[[], SimulatorAdapter],
) -> None:
    adapter = adapter_factory()
    service = GolfSessionService(session_id="contract-session", adapter=adapter)

    conn = await service.select_destination(adapter)
    assert conn.state.value == "connected"

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
    shot = ShotEnvelope(
        schema_version=1,
        shot_id="contract-shot-001",
        session_id="contract-session",
        source_kind=SourceKind.MANUAL,
        qualification=qual,
        ball_velocity_m_s=(68.0, 1.0, 12.0),
        ball_angular_velocity_rad_s=(0.0, -280.0, 20.0),
        aim_context=aim,
        created_at_utc="2026-09-15T12:00:00Z",
    )

    # 1. Prepare
    prep = service.prepare(shot, context_revision=1)
    assert prep.shot.shot_id == "contract-shot-001"
    assert service.current_state == SessionState.PREPARED

    # 2. Arm
    token = service.arm(prep.prepared_shot_id, context_revision=1)
    assert token is not None
    assert service.current_state == SessionState.ARMED

    # 3. Submit at impact
    receipt = await service.submit_at_impact(prep.prepared_shot_id, arm_token=token)
    assert receipt.shot_id == "contract-shot-001"
    assert receipt.session_id == "contract-session"
    assert receipt.state == SubmissionState.CONFIRMED_ACCEPTED
    assert service.current_state == SessionState.IDLE


@pytest.mark.asyncio
async def test_adapter_substitution_preserves_shot_identity() -> None:
    fake_adapter = FakeSimulatorAdapter(destination_id="fake_01")
    local_adapter = LocalReferenceAdapter()

    service = GolfSessionService(session_id="sub-session", adapter=fake_adapter)
    await service.select_destination(fake_adapter)

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
    shot1 = ShotEnvelope(
        schema_version=1,
        shot_id="shot-id-preservation-1",
        session_id="sub-session",
        source_kind=SourceKind.MANUAL,
        qualification=qual,
        ball_velocity_m_s=(62.0, -1.0, 14.0),
        ball_angular_velocity_rad_s=(0.0, -220.0, -10.0),
        aim_context=aim,
        created_at_utc="2026-09-15T12:00:00Z",
    )

    # Shot 1 on fake
    p1 = service.prepare(shot1, context_revision=1)
    t1 = service.arm(p1.prepared_shot_id, context_revision=1)
    r1 = await service.submit_at_impact(p1.prepared_shot_id, arm_token=t1)
    assert r1.shot_id == "shot-id-preservation-1"

    # Switch to local reference adapter
    await service.select_destination(local_adapter)

    # Shot 2 on local reference
    shot2 = ShotEnvelope(
        schema_version=1,
        shot_id="shot-id-preservation-2",
        session_id="sub-session",
        source_kind=SourceKind.MANUAL,
        qualification=qual,
        ball_velocity_m_s=(62.0, -1.0, 14.0),
        ball_angular_velocity_rad_s=(0.0, -220.0, -10.0),
        aim_context=aim,
        created_at_utc="2026-09-15T12:05:00Z",
    )
    p2 = service.prepare(shot2, context_revision=1)
    t2 = service.arm(p2.prepared_shot_id, context_revision=1)
    r2 = await service.submit_at_impact(p2.prepared_shot_id, arm_token=t2)

    assert r2.shot_id == "shot-id-preservation-2"
    assert r2.destination_id == "local_reference"

    # Local adapter generated the trajectory without changing shot identity
    traj = local_adapter.get_last_trajectory("shot-id-preservation-2")
    assert traj is not None
    assert len(traj) > 0
