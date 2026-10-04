"""Explicit local research shots retain unqualified lineage and one-shot controls."""

from dataclasses import replace
import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from src.shared.python.golf_simulator import (
    AimContext,
    CapabilityDescriptor,
    CapabilityState,
    ContactStatus,
    GolfSessionService,
    LocalReferenceAdapter,
    NumericalStatus,
    ScientificStatus,
    SessionState,
    ShotEnvelope,
    ShotQualification,
    SourceKind,
    SubmissionState,
)
from src.shared.python.golf_simulator.adapters.fake import FakeSimulatorAdapter
from src.shared.python.golf_simulator.adapters.relay import FlightRelayAdapter

pytestmark = pytest.mark.unit


def shot() -> ShotEnvelope:
    return ShotEnvelope(
        1,
        "research-shot",
        "session",
        SourceKind.MODEL_CONTACT,
        ShotQualification(
            ContactStatus.UNVERIFIED,
            NumericalStatus.UNVERIFIED,
            ScientificStatus.UNVERIFIED,
            ("authenticated-impact-receipt",),
        ),
        (35.0, 0.0, 8.0),
        (0.0, -100.0, 0.0),
        AimContext(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        "2026-10-04T00:00:00Z",
        model_run_id="owned-run",
        trace_digest="sha256:" + "a" * 64,
        impact_id="sample-2",
        impact_time_s=0.2,
    )


def local() -> LocalReferenceAdapter:
    simulator = MagicMock()
    simulator.simulate_trajectory.return_value = [object(), object()]
    return LocalReferenceAdapter(simulator)


@pytest.mark.asyncio
async def test_explicit_research_keeps_default_gate_and_one_shot():
    adapter = local()
    service = GolfSessionService("session", adapter)
    await service.select_destination(adapter)
    original = shot()
    with pytest.raises(ValueError, match="qualified contact"):
        service.prepare(original)
    prepared = service.prepare_research_shot(original, 2)
    assert prepared.shot is original
    token = service.arm(prepared.prepared_shot_id, 2)
    receipt = await service.submit_at_impact(prepared.prepared_shot_id, token)
    assert receipt.state == SubmissionState.CONFIRMED_ACCEPTED
    assert receipt.destination_id == "local_reference"
    assert original.qualification.contact == ContactStatus.UNVERIFIED
    assert original.qualification.scientific == ScientificStatus.UNVERIFIED
    assert service.current_state == SessionState.IDLE
    assert len(adapter.get_last_trajectory(original.shot_id)) == 2
    with pytest.raises(RuntimeError, match="not armed"):
        await service.submit_at_impact(prepared.prepared_shot_id, token)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", [FakeSimulatorAdapter(), FlightRelayAdapter()])
async def test_nonlocal_adapter_rejected_even_with_local_endpoint(adapter):
    service = GolfSessionService("session", adapter)
    service._destination_id = "local_in_memory"
    with pytest.raises(ValueError, match="local"):
        service.prepare_research_shot(shot())
    assert service.current_state == SessionState.IDLE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"source_kind": SourceKind.MANUAL},
        {"session_id": "foreign"},
        {
            "qualification": ShotQualification(
                ContactStatus.QUALIFIED,
                NumericalStatus.CONVERGED,
                ScientificStatus.UNVERIFIED,
                ("proof",),
            )
        },
        {
            "qualification": ShotQualification(
                ContactStatus.UNVERIFIED,
                NumericalStatus.CONVERGED,
                ScientificStatus.BENCHMARKED,
                ("proof",),
            )
        },
        {
            "qualification": ShotQualification(
                ContactStatus.UNVERIFIED,
                NumericalStatus.CONVERGED,
                ScientificStatus.UNVERIFIED,
                ("proof",),
            )
        },
        {
            "qualification": ShotQualification(
                ContactStatus.UNVERIFIED,
                NumericalStatus.UNVERIFIED,
                ScientificStatus.UNVERIFIED,
            )
        },
        {
            "qualification": ShotQualification(
                ContactStatus.UNVERIFIED,
                NumericalStatus.UNVERIFIED,
                ScientificStatus.UNVERIFIED,
                (" ",),
            )
        },
        {
            "qualification": ShotQualification(
                "unverified",
                NumericalStatus.UNVERIFIED,
                ScientificStatus.UNVERIFIED,
                ("proof",),
            )
        },
    ],
)
async def test_missing_or_promoted_research_evidence_rejected(changes):
    adapter = local()
    service = GolfSessionService("session", adapter)
    await service.select_destination(adapter)
    with pytest.raises(ValueError):
        service.prepare_research_shot(replace(shot(), **changes))
    assert service.current_state == SessionState.IDLE


@pytest.mark.asyncio
async def test_cancel_and_destination_change_invalidate_research():
    adapter = local()
    service = GolfSessionService("session", adapter)
    await service.select_destination(adapter)
    prepared = service.prepare_research_shot(shot())
    token = service.arm(prepared.prepared_shot_id, 1)
    service.disarm(prepared.prepared_shot_id)
    service.cancel(prepared.prepared_shot_id)
    assert service.current_state == SessionState.IDLE
    prepared = service.prepare_research_shot(shot())
    await service.select_destination(FakeSimulatorAdapter())
    with pytest.raises(RuntimeError):
        service.arm(prepared.prepared_shot_id, 1)
    with pytest.raises(RuntimeError):
        await service.submit_at_impact(prepared.prepared_shot_id, token)


@pytest.mark.asyncio
async def test_submit_rechecks_research_destination_before_send():
    adapter = local()
    service = GolfSessionService("session", adapter)
    await service.select_destination(adapter)
    prepared = service.prepare_research_shot(shot())
    token = service.arm(prepared.prepared_shot_id, 1)
    forged = FakeSimulatorAdapter()
    service._adapter = forged
    with pytest.raises(ValueError, match="local"):
        await service.submit_at_impact(prepared.prepared_shot_id, token)
    assert not forged.submitted_shots
    assert service.current_state == SessionState.ARMED


@pytest.mark.asyncio
async def test_unconnected_local_rejected():
    service = GolfSessionService("session", local())
    with pytest.raises(ValueError, match="local"):
        service.prepare_research_shot(shot())


@pytest.mark.asyncio
async def test_capability_loss_rejected_before_local_submission(monkeypatch):
    adapter = local()
    service = GolfSessionService("session", adapter)
    await service.select_destination(adapter)
    prepared = service.prepare_research_shot(shot())
    token = service.arm(prepared.prepared_shot_id, 1)
    caps = replace(
        adapter.capabilities(),
        shot_input=CapabilityDescriptor(CapabilityState.UNSUPPORTED, "lost"),
    )
    monkeypatch.setattr(adapter, "capabilities", lambda: caps)
    with pytest.raises(ValueError, match="shot input"):
        await service.submit_at_impact(prepared.prepared_shot_id, token)
    assert adapter.get_trajectory_record(shot().shot_id) is None


@pytest.mark.asyncio
async def test_local_record_recall_requires_owned_delivery():
    adapter = local()
    service = GolfSessionService("session", adapter)
    await service.select_destination(adapter)
    with pytest.raises(ValueError, match="confirmed local"):
        service.get_local_trajectory_record(shot().shot_id)
    prepared = service.prepare_research_shot(shot())
    await service.submit_at_impact(
        prepared.prepared_shot_id, service.arm(prepared.prepared_shot_id, 1)
    )
    with pytest.raises(ValueError):
        service.get_local_trajectory_record("foreign-shot")
    await service.select_destination(FakeSimulatorAdapter())
    with pytest.raises(ValueError):
        service.get_local_trajectory_record(shot().shot_id)


@pytest.mark.asyncio
async def test_actual_local_flight_with_explicit_environment():
    from src.shared.python.physics.ball_launch_conditions import EnvironmentalConditions
    from src.shared.python.physics.ball_simulator import BallFlightSimulator

    adapter = LocalReferenceAdapter(
        BallFlightSimulator(environment=EnvironmentalConditions())
    )
    service = GolfSessionService("session", adapter)
    await service.select_destination(adapter)
    prepared = service.prepare_research_shot(shot())
    receipt = await service.submit_at_impact(
        prepared.prepared_shot_id, service.arm(prepared.prepared_shot_id, 1)
    )
    assert receipt.state == SubmissionState.CONFIRMED_ACCEPTED
    points = adapter.get_last_trajectory(shot().shot_id)
    assert points is not None and len(points) > 2
    assert max(point.position[0] for point in points) > 0
    record = service.get_local_trajectory_record(shot().shot_id)
    first_position = points[0].position.copy()
    record.points[0].position[:] = 999
    record.points.clear()
    retained = service.get_local_trajectory_record(shot().shot_id)
    assert len(retained.points) == len(points)
    assert (retained.points[0].position == first_position).all()


@pytest.mark.asyncio
async def test_local_simulation_allows_event_loop_heartbeat():
    release = threading.Event()
    observed = []
    simulator = MagicMock()

    def simulate(launch):
        observed.append(release.wait(1))
        return [object()]

    simulator.simulate_trajectory.side_effect = simulate
    adapter = LocalReferenceAdapter(simulator)
    service = GolfSessionService("session", adapter)
    await service.select_destination(adapter)
    prepared = service.prepare_research_shot(shot())
    token = service.arm(prepared.prepared_shot_id, 1)
    submission = asyncio.create_task(
        service.submit_at_impact(prepared.prepared_shot_id, token)
    )
    await asyncio.sleep(0)
    release.set()
    receipt = await submission
    assert observed == [True], "Local simulation blocked the event loop heartbeat"
    assert receipt.state == SubmissionState.CONFIRMED_ACCEPTED
