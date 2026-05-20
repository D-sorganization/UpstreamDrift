"""Tests for GaitStateMachine."""

from __future__ import annotations

import pytest

from src.robotics.locomotion.gait_state_machine import (
    GaitEvent,
    GaitStateMachine,
)
from src.robotics.locomotion.gait_types import (
    GaitParameters,
    GaitPhase,
    GaitType,
    SupportState,
)


def make_params(step=0.2, ds_ratio=0.2) -> GaitParameters:
    return GaitParameters(
        gait_type=GaitType.WALK,
        step_length=0.3,
        step_width=0.2,
        step_height=0.05,
        step_duration=step,
        double_support_ratio=ds_ratio,
        com_height=0.9,
    )


def test_initial_state() -> None:
    sm = GaitStateMachine()
    assert not sm.is_walking
    assert sm.phase == GaitPhase.DOUBLE_SUPPORT
    assert sm.parameters is not None


def test_start_walking_idempotent() -> None:
    sm = GaitStateMachine(make_params())
    sm.start_walking()
    assert sm.is_walking
    sm.start_walking()  # no-op
    assert sm.is_walking


def test_stop_walking_only_in_double_support() -> None:
    sm = GaitStateMachine(make_params())
    sm.start_walking()
    # Force into swing by transitioning
    sm._transition_phase()
    assert sm.phase in (GaitPhase.LEFT_SWING, GaitPhase.RIGHT_SWING)
    sm.stop_walking()
    # Should not stop because in swing
    assert sm.is_walking


def test_stop_walking_when_double_support() -> None:
    sm = GaitStateMachine(make_params())
    sm.start_walking()
    assert sm.phase == GaitPhase.DOUBLE_SUPPORT
    sm.stop_walking()
    assert not sm.is_walking
    sm.stop_walking()  # idempotent


def test_emergency_stop() -> None:
    sm = GaitStateMachine(make_params())
    sm.start_walking()
    sm.emergency_stop()
    assert not sm.is_walking
    assert sm.phase == GaitPhase.DOUBLE_SUPPORT
    assert sm.state.support_state == SupportState.DOUBLE_SUPPORT_CENTERED


def test_set_parameters() -> None:
    sm = GaitStateMachine()
    p = make_params()
    sm.set_parameters(p)
    assert sm.parameters is p


def test_update_advances_time_and_transitions() -> None:
    sm = GaitStateMachine(make_params(step=0.1, ds_ratio=0.5))
    sm.start_walking()
    sm.update(0.2)  # advance through double support
    # Should have transitioned at least once
    assert sm.state.cycle_time > 0


def test_update_non_positive_dt_no_op() -> None:
    sm = GaitStateMachine(make_params())
    sm.start_walking()
    s = sm.update(0.0)
    assert s.phase_time == 0.0


def test_handle_event_routes() -> None:
    sm = GaitStateMachine(make_params())
    sm.handle_event(GaitEvent.START_REQUESTED)
    assert sm.is_walking
    sm.handle_event(GaitEvent.STOP_REQUESTED)
    sm.handle_event(GaitEvent.EMERGENCY_STOP)
    sm.handle_event(GaitEvent.FOOT_CONTACT)
    sm.handle_event(GaitEvent.FOOT_LIFTOFF)


def test_foot_contact_during_swing() -> None:
    sm = GaitStateMachine(make_params())
    sm.start_walking()
    sm._transition_phase()  # to swing
    pre_phase = sm.phase
    sm.handle_event(GaitEvent.FOOT_CONTACT)
    assert sm.phase != pre_phase or sm.phase == GaitPhase.DOUBLE_SUPPORT


def test_foot_liftoff_during_double_support() -> None:
    sm = GaitStateMachine(make_params())
    sm.start_walking()
    assert sm.phase == GaitPhase.DOUBLE_SUPPORT
    sm.handle_event(GaitEvent.FOOT_LIFTOFF)
    assert sm.phase != GaitPhase.DOUBLE_SUPPORT


def test_callbacks_invoked() -> None:
    sm = GaitStateMachine(make_params())
    calls: list[str] = []
    sm.register_callback("gait_change", lambda s, e: calls.append("gait"))
    sm.register_callback("phase_change", lambda s, e: calls.append("phase"))
    sm.register_callback("step_complete", lambda s, e: calls.append("step"))
    sm.start_walking()
    sm._transition_phase()  # to swing -> phase_change
    sm._transition_phase()  # back to double -> step_complete + phase_change
    assert "gait" in calls
    assert "phase" in calls
    assert "step" in calls


def test_callback_swallowed_on_exception() -> None:
    sm = GaitStateMachine(make_params())

    def bad(s, e) -> None:
        raise RuntimeError("nope")

    sm.register_callback("phase_change", bad)
    sm.start_walking()
    sm._transition_phase()  # should not raise


def test_get_foot_trajectory_phase_standing_returns_one() -> None:
    sm = GaitStateMachine()
    assert sm.get_foot_trajectory_phase("left") == 1.0


def test_get_foot_trajectory_phase_swing() -> None:
    sm = GaitStateMachine(make_params())
    sm.start_walking()
    # Force into LEFT_SWING
    sm._state.phase = GaitPhase.LEFT_SWING
    sm._state.phase_time = 0.05
    assert 0.0 <= sm.get_foot_trajectory_phase("left") <= 1.0
    sm._state.phase = GaitPhase.RIGHT_SWING
    assert 0.0 <= sm.get_foot_trajectory_phase("right") <= 1.0
    assert sm.get_foot_trajectory_phase("left") == 1.0  # in stance


def test_phase_progress_at_capacity_returns_one() -> None:
    sm = GaitStateMachine(make_params(step=0.1, ds_ratio=0.0))
    # double support duration is 0; phase_progress should saturate at 1.0
    assert sm.phase_progress == 1.0


def test_register_callback_unknown_event_silently_ignored() -> None:
    sm = GaitStateMachine()
    sm.register_callback("not_a_real_event", lambda s, e: None)
    # No exception raised


def test_walking_carries_excess_time() -> None:
    sm = GaitStateMachine(make_params(step=0.1, ds_ratio=0.5))
    sm.start_walking()
    # Step ds duration = 0.05; advance more
    sm.update(0.2)
    assert sm.state.cycle_time == pytest.approx(0.2, rel=1e-6)


def test_state_returns_copy() -> None:
    sm = GaitStateMachine()
    s1 = sm.state
    s1.step_count = 42
    assert sm.state.step_count == 0
