"""Unit tests for MonotonicReplayClock and ReplaySubmissionCoordinator.

Follows TDD, DbC, Law of Demeter, and DRY.
Acceptance criteria from Issue #10195 (GS-06).
"""

from __future__ import annotations

import pytest

from src.shared.python.golf_simulator.adapters.fake import FakeSimulatorAdapter
from src.shared.python.golf_simulator.contracts import (
    AimContext,
    ContactStatus,
    NumericalStatus,
    PreparedShot,
    ReplayPlaybackState,
    ReplayTimingRecord,
    ScientificStatus,
    SessionState,
    ShotEnvelope,
    ShotQualification,
    SourceKind,
    SubmissionState,
)
from src.shared.python.golf_simulator.journal import ShotJournal
from src.shared.python.golf_simulator.replay import (
    MonotonicReplayClock,
    ReplaySubmissionCoordinator,
)
from src.shared.python.golf_simulator.session import GolfSessionService

pytestmark = pytest.mark.unit


def _make_model_shot(
    shot_id: str = "shot-model-001",
    session_id: str = "sess-001",
    impact_time_s: float = 1.5,
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
        evidence_refs=("unit_test",),
    )
    return ShotEnvelope(
        schema_version=1,
        shot_id=shot_id,
        session_id=session_id,
        source_kind=SourceKind.MODEL_CONTACT,
        qualification=qual,
        ball_velocity_m_s=(65.0, 2.0, 15.0),
        ball_angular_velocity_rad_s=(10.0, 250.0, -15.0),
        aim_context=aim,
        created_at_utc="2026-09-15T12:00:00Z",
        model_run_id="run-42",
        trace_digest="sha256:fedcba9876543210",
        impact_id="impact-42",
        impact_time_s=impact_time_s,
    )


class FakeTimeSource:
    """Controllable monotonic time source for deterministic replay tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.current_time = start

    def monotonic(self) -> float:
        return self.current_time

    def advance(self, delta_s: float) -> None:
        self.current_time += delta_s


def test_monotonic_clock_normal_playback_and_speed_change() -> None:
    time_src = FakeTimeSource(start=100.0)
    clock = MonotonicReplayClock(time_fn=time_src.monotonic)

    assert clock.playback_state == ReplayPlaybackState.STOPPED
    assert clock.current_time_s == 0.0

    clock.play()
    assert clock.playback_state == ReplayPlaybackState.PLAYING

    # Advance 0.5s of real time at 1.0x
    time_src.advance(0.5)
    t_prev, t_curr, was_seek = clock.tick()
    assert t_prev == 0.0
    assert pytest.approx(t_curr, abs=1e-5) == 0.5
    assert not was_seek

    # Change playback rate to 0.5x slow motion
    clock.set_playback_rate(0.5)
    time_src.advance(1.0)
    t_prev, t_curr, was_seek = clock.tick()
    assert pytest.approx(t_prev, abs=1e-5) == 0.5
    assert pytest.approx(t_curr, abs=1e-5) == 1.0  # 0.5 + 1.0 * 0.5 = 1.0
    assert not was_seek


def test_monotonic_clock_pause_and_seek() -> None:
    time_src = FakeTimeSource(start=100.0)
    clock = MonotonicReplayClock(time_fn=time_src.monotonic)

    clock.play()
    time_src.advance(1.0)
    clock.tick()

    clock.pause()
    assert clock.playback_state == ReplayPlaybackState.PAUSED

    time_src.advance(1.0)
    t_prev, t_curr, was_seek = clock.tick()
    assert t_prev == 1.0
    assert t_curr == 1.0  # Paused: does not advance
    assert not was_seek

    # Explicit seek
    clock.seek(0.2)
    assert pytest.approx(clock.current_time_s, abs=1e-5) == 0.2
    t_prev, t_curr, was_seek = clock.tick()
    assert was_seek
    assert pytest.approx(t_curr, abs=1e-5) == 0.2


@pytest.mark.asyncio
async def test_replay_coordinator_submits_once_at_impact() -> None:
    time_src = FakeTimeSource(start=100.0)
    clock = MonotonicReplayClock(time_fn=time_src.monotonic)
    adapter = FakeSimulatorAdapter()
    await adapter.connect({})

    session_service = GolfSessionService(session_id="sess-001", adapter=adapter)
    shot = _make_model_shot(impact_time_s=1.0)
    prep = session_service.prepare(shot=shot, context_revision=1)
    arm_token = session_service.arm(
        prepared_shot_id=prep.prepared_shot_id, context_revision=1
    )

    coordinator = ReplaySubmissionCoordinator(
        session_service=session_service,
        clock=clock,
        time_fn=time_src.monotonic,
    )
    coordinator.arm_submission(
        prepared_shot=prep,
        arm_token=arm_token,
    )

    assert coordinator.is_armed
    assert not coordinator.has_triggered

    # Start playback
    clock.play()

    # Tick 1: t=0.0 -> 0.8 (before impact at 1.0)
    time_src.advance(0.8)
    receipt = await coordinator.tick()
    assert receipt is None
    assert not coordinator.has_triggered
    assert len(adapter.submitted_shots) == 0

    # Tick 2: t=0.8 -> 1.2 (crosses impact at 1.0)
    time_src.advance(0.4)
    receipt = await coordinator.tick()
    assert receipt is not None
    assert receipt.state == SubmissionState.CONFIRMED_ACCEPTED
    assert coordinator.has_triggered
    assert len(adapter.submitted_shots) == 1
    assert adapter.submitted_shots[0].shot_id == shot.shot_id

    # Tick 3: subsequent ticks after impact must NOT submit again
    time_src.advance(0.5)
    receipt2 = await coordinator.tick()
    assert receipt2 is None
    assert len(adapter.submitted_shots) == 1


@pytest.mark.asyncio
async def test_replay_coordinator_seeking_across_impact_disarms_without_submitting() -> (
    None
):
    time_src = FakeTimeSource(start=100.0)
    clock = MonotonicReplayClock(time_fn=time_src.monotonic)
    adapter = FakeSimulatorAdapter()
    await adapter.connect({})

    session_service = GolfSessionService(session_id="sess-001", adapter=adapter)
    shot = _make_model_shot(impact_time_s=1.0)
    prep = session_service.prepare(shot=shot, context_revision=1)
    arm_token = session_service.arm(
        prepared_shot_id=prep.prepared_shot_id, context_revision=1
    )

    coordinator = ReplaySubmissionCoordinator(
        session_service=session_service,
        clock=clock,
        time_fn=time_src.monotonic,
    )
    coordinator.arm_submission(
        prepared_shot=prep,
        arm_token=arm_token,
    )

    # User seeks across impact (from 0.0 directly to 1.5)
    clock.seek(1.5)
    receipt = await coordinator.tick()

    # Must disarm rather than submitting
    assert receipt is None
    assert not coordinator.is_armed
    assert not coordinator.has_triggered
    assert len(adapter.submitted_shots) == 0
    # Session state in session_service was disarmed back to PREPARED
    assert session_service.current_state == SessionState.PREPARED


@pytest.mark.asyncio
async def test_replay_coordinator_pause_and_rewind_safety() -> None:
    time_src = FakeTimeSource(start=100.0)
    clock = MonotonicReplayClock(time_fn=time_src.monotonic)
    adapter = FakeSimulatorAdapter()
    await adapter.connect({})

    session_service = GolfSessionService(session_id="sess-001", adapter=adapter)
    shot = _make_model_shot(impact_time_s=1.0)
    prep = session_service.prepare(shot=shot, context_revision=1)
    arm_token = session_service.arm(
        prepared_shot_id=prep.prepared_shot_id, context_revision=1
    )

    coordinator = ReplaySubmissionCoordinator(
        session_service=session_service,
        clock=clock,
        time_fn=time_src.monotonic,
    )
    coordinator.arm_submission(
        prepared_shot=prep,
        arm_token=arm_token,
    )

    # Play until 0.8s
    clock.play()
    time_src.advance(0.8)
    await coordinator.tick()

    # Pause playback
    clock.pause()
    time_src.advance(2.0)
    await coordinator.tick()
    assert not coordinator.has_triggered
    assert len(adapter.submitted_shots) == 0

    # Seek backward to 0.1s
    clock.seek(0.1)
    await coordinator.tick()
    assert not coordinator.has_triggered
    assert len(adapter.submitted_shots) == 0


@pytest.mark.asyncio
async def test_replay_coordinator_latency_recording_benchmark() -> None:
    """Benchmark p95 impact-to-send latency across 100 simulated samples."""
    adapter = FakeSimulatorAdapter()
    await adapter.connect({})

    timing_records: list[ReplayTimingRecord] = []

    for i in range(100):
        time_src = FakeTimeSource(start=100.0)
        clock = MonotonicReplayClock(time_fn=time_src.monotonic)
        session_id = f"sess-{i}"
        session_service = GolfSessionService(
            session_id=session_id,
            adapter=adapter,
        )
        shot = _make_model_shot(
            shot_id=f"shot-{i}",
            session_id=session_id,
            impact_time_s=1.0,
        )
        prep = session_service.prepare(shot=shot, context_revision=1)
        arm_token = session_service.arm(
            prepared_shot_id=prep.prepared_shot_id,
            context_revision=1,
        )

        coordinator = ReplaySubmissionCoordinator(
            session_service=session_service,
            clock=clock,
            time_fn=time_src.monotonic,
        )
        coordinator.arm_submission(
            prepared_shot=prep,
            arm_token=arm_token,
        )

        clock.play()
        time_src.advance(0.9)
        await coordinator.tick()

        # Step right across impact instant
        time_src.advance(0.15)
        receipt = await coordinator.tick()
        assert receipt is not None
        record = coordinator.last_timing_record
        assert record is not None
        timing_records.append(record)

    assert len(timing_records) == 100
    latencies = [r.impact_to_send_latency_ms for r in timing_records]
    latencies.sort()
    # In fake clock simulation, impact-to-send overhead is essentially 0 ms
    p95 = latencies[94]
    assert p95 <= 50.0, f"Expected p95 latency <= 50ms, got {p95}ms"
