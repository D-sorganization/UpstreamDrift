"""Tests for :mod:`training.status`."""

from __future__ import annotations

import pytest

from training import (
    TERMINAL_STATUSES,
    InvalidStatusTransitionError,
    TrainingStatus,
    can_transition,
    validate_transition,
)

pytestmark = pytest.mark.unit


class TestTrainingStatusEnum:
    def test_values_are_stable_strings(self) -> None:
        assert TrainingStatus.PENDING.value == "pending"
        assert TrainingStatus.QUEUED.value == "queued"
        assert TrainingStatus.RUNNING.value == "running"
        assert TrainingStatus.PAUSED.value == "paused"
        assert TrainingStatus.COMPLETED.value == "completed"
        assert TrainingStatus.FAILED.value == "failed"
        assert TrainingStatus.CANCELLED.value == "cancelled"

    def test_terminal_set_membership(self) -> None:
        assert {
            TrainingStatus.COMPLETED,
            TrainingStatus.FAILED,
            TrainingStatus.CANCELLED,
        } == TERMINAL_STATUSES

    @pytest.mark.parametrize(
        "status",
        [
            TrainingStatus.COMPLETED,
            TrainingStatus.FAILED,
            TrainingStatus.CANCELLED,
        ],
    )
    def test_is_terminal_for_terminal_states(self, status: TrainingStatus) -> None:
        assert status.is_terminal is True

    @pytest.mark.parametrize(
        "status",
        [
            TrainingStatus.PENDING,
            TrainingStatus.QUEUED,
            TrainingStatus.RUNNING,
            TrainingStatus.PAUSED,
        ],
    )
    def test_is_terminal_for_active_states(self, status: TrainingStatus) -> None:
        assert status.is_terminal is False

    def test_is_active_only_for_running_and_paused(self) -> None:
        assert TrainingStatus.RUNNING.is_active is True
        assert TrainingStatus.PAUSED.is_active is True
        assert TrainingStatus.PENDING.is_active is False
        assert TrainingStatus.QUEUED.is_active is False
        assert TrainingStatus.COMPLETED.is_active is False


class TestCanTransition:
    @pytest.mark.parametrize(
        ("source", "destination"),
        [
            (TrainingStatus.PENDING, TrainingStatus.QUEUED),
            (TrainingStatus.PENDING, TrainingStatus.CANCELLED),
            (TrainingStatus.PENDING, TrainingStatus.FAILED),
            (TrainingStatus.QUEUED, TrainingStatus.RUNNING),
            (TrainingStatus.QUEUED, TrainingStatus.CANCELLED),
            (TrainingStatus.QUEUED, TrainingStatus.FAILED),
            (TrainingStatus.RUNNING, TrainingStatus.PAUSED),
            (TrainingStatus.RUNNING, TrainingStatus.COMPLETED),
            (TrainingStatus.RUNNING, TrainingStatus.FAILED),
            (TrainingStatus.RUNNING, TrainingStatus.CANCELLED),
            (TrainingStatus.PAUSED, TrainingStatus.RUNNING),
            (TrainingStatus.PAUSED, TrainingStatus.CANCELLED),
            (TrainingStatus.PAUSED, TrainingStatus.FAILED),
        ],
    )
    def test_permitted_edges(
        self, source: TrainingStatus, destination: TrainingStatus
    ) -> None:
        assert can_transition(source, destination) is True

    @pytest.mark.parametrize(
        ("source", "destination"),
        [
            (TrainingStatus.PENDING, TrainingStatus.RUNNING),  # must queue first
            (TrainingStatus.PENDING, TrainingStatus.COMPLETED),
            (TrainingStatus.PENDING, TrainingStatus.PAUSED),
            (TrainingStatus.QUEUED, TrainingStatus.COMPLETED),  # must run first
            (TrainingStatus.QUEUED, TrainingStatus.PAUSED),
            (TrainingStatus.RUNNING, TrainingStatus.QUEUED),  # no rewind
            (TrainingStatus.PAUSED, TrainingStatus.COMPLETED),  # resume then finish
            (TrainingStatus.COMPLETED, TrainingStatus.RUNNING),  # terminal
            (TrainingStatus.FAILED, TrainingStatus.QUEUED),
            (TrainingStatus.CANCELLED, TrainingStatus.PENDING),
        ],
    )
    def test_forbidden_edges(
        self, source: TrainingStatus, destination: TrainingStatus
    ) -> None:
        assert can_transition(source, destination) is False

    @pytest.mark.parametrize(
        "status",
        list(TrainingStatus),
    )
    def test_self_transition_forbidden(self, status: TrainingStatus) -> None:
        """A status cannot transition to itself — callers should not re-apply."""
        assert can_transition(status, status) is False

    def test_terminal_states_have_no_outgoing_edges(self) -> None:
        for terminal in TERMINAL_STATUSES:
            for other in TrainingStatus:
                assert can_transition(terminal, other) is False, (
                    f"{terminal} should have no outgoing edge to {other}"
                )

    def test_type_errors_for_non_enum_inputs(self) -> None:
        with pytest.raises(TypeError):
            can_transition("pending", TrainingStatus.QUEUED)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            can_transition(TrainingStatus.PENDING, "queued")  # type: ignore[arg-type]


class TestValidateTransition:
    def test_no_op_for_permitted_edge(self) -> None:
        validate_transition(TrainingStatus.PENDING, TrainingStatus.QUEUED)

    def test_raises_for_forbidden_edge(self) -> None:
        with pytest.raises(InvalidStatusTransitionError) as excinfo:
            validate_transition(TrainingStatus.COMPLETED, TrainingStatus.RUNNING)
        assert excinfo.value.source == TrainingStatus.COMPLETED
        assert excinfo.value.destination == TrainingStatus.RUNNING

    def test_error_message_contains_both_states(self) -> None:
        with pytest.raises(InvalidStatusTransitionError) as excinfo:
            validate_transition(TrainingStatus.PENDING, TrainingStatus.RUNNING)
        msg = str(excinfo.value)
        assert "PENDING" in msg or "pending" in msg
        assert "RUNNING" in msg or "running" in msg
