"""``step()`` on a kinematic-only service must raise :class:`CapabilityError`."""

from __future__ import annotations

import pytest

from src.shared.python.pose_interchange.live_kinematics import CapabilityError
from src.shared.python.pose_interchange.services._mock import (
    MockKinematicsService,
)

pytestmark = pytest.mark.unit


def test_mock_step_raises_capability_error() -> None:
    """A kinematic mock raises :class:`CapabilityError` from :meth:`step`."""
    svc = MockKinematicsService(engine_name="drake")
    assert svc.capabilities().supports_dynamics_step is False
    with pytest.raises(CapabilityError, match="does not support dynamics step"):
        svc.step(0.01)


def test_capability_error_is_runtime_error() -> None:
    """:class:`CapabilityError` must inherit :class:`RuntimeError`.

    Generic error handlers should still catch it via ``except RuntimeError``.
    """
    assert issubclass(CapabilityError, RuntimeError)


def test_mock_step_message_includes_engine_name() -> None:
    svc = MockKinematicsService(engine_name="mujoco")
    with pytest.raises(CapabilityError, match="mujoco"):
        svc.step(0.001)
