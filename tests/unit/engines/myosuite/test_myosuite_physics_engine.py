"""Tests for MyoSuitePhysicsEngine state-transition invariants (Issue #2483)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestIssue2483MyoSuiteTerminationHandling:
    """Issue #2483: step() must handle environment termination state."""

    def _make_engine_with_mock_env(self) -> tuple[Any, MagicMock]:
        """Create MyoSuitePhysicsEngine with a mocked Gym environment."""
        try:
            from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
                MyoSuitePhysicsEngine,
            )
        except ImportError:
            pytest.skip("MyoSuite not available")

        from unittest.mock import MagicMock

        engine = MyoSuitePhysicsEngine.__new__(MyoSuitePhysicsEngine)
        engine._dt = 0.002
        engine._last_action = np.zeros(3)
        engine._terminated = False
        engine.env_id = "mock_env"
        engine.sim = MagicMock()

        mock_env = MagicMock()
        mock_env.action_space.sample.return_value = np.zeros(3)
        engine.env = mock_env
        return engine, mock_env

    def test_step_records_termination_from_env(self) -> None:
        """step() must record termination state returned by env.step()."""
        engine, mock_env = self._make_engine_with_mock_env()
        mock_env.step.return_value = (
            np.zeros(5),  # obs
            1.0,  # reward
            True,  # terminated
            False,  # truncated
            {},  # info
        )

        engine.step()

        assert engine._terminated is True, (
            "step() must record terminated=True from env.step() return value"
        )

    def test_step_on_terminated_env_does_not_call_env_step(self) -> None:
        """step() must not call env.step() when the episode has already terminated."""
        engine, mock_env = self._make_engine_with_mock_env()
        engine._terminated = True

        engine.step()

        (
            mock_env.step.assert_not_called(),
            ("step() must not forward to env.step() after episode termination"),
        )
