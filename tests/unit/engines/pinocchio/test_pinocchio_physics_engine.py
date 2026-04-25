"""Tests for src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine."""

from __future__ import annotations

from typing import Any

import pytest


def test_import() -> None:
    """Verify the module can be imported."""
    try:
        import src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine

        assert (
            src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine
            is not None
        )
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Missing dependencies or import error: {e}")


class TestIssue2483PinocchioSetStateInvariants:
    """Issue #2483: set_state() must validate sizes and refresh derived kinematics."""

    def _make_engine_with_mock_model(self) -> Any:
        """Create a PinocchioPhysicsEngine with a mocked pinocchio model."""
        try:
            from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
                PinocchioPhysicsEngine,
            )
        except ImportError:
            pytest.skip("Pinocchio not available")

        from unittest.mock import MagicMock

        import numpy as np

        engine = PinocchioPhysicsEngine.__new__(PinocchioPhysicsEngine)
        engine._initialized = True
        engine.time = 0.0
        engine.tau = np.zeros(6)
        engine.a = np.zeros(6)

        mock_model = MagicMock()
        mock_model.nq = 7
        mock_model.nv = 6
        engine.model = mock_model

        mock_data = MagicMock()
        engine.data = mock_data
        engine.q = np.zeros(7)
        engine.v = np.zeros(6)

        return engine

    def test_set_state_raises_for_wrong_q_size(self) -> None:
        """set_state() must raise ValueError when q has wrong size."""
        import numpy as np

        engine = self._make_engine_with_mock_model()
        wrong_q = np.zeros(5)  # model.nq == 7
        correct_v = np.zeros(6)

        with pytest.raises(ValueError, match="(?i)size|shape|nq|dimension|q"):
            engine.set_state(wrong_q, correct_v)

    def test_set_state_raises_for_wrong_v_size(self) -> None:
        """set_state() must raise ValueError when v has wrong size."""
        import numpy as np

        engine = self._make_engine_with_mock_model()
        correct_q = np.zeros(7)
        wrong_v = np.zeros(3)  # model.nv == 6

        with pytest.raises(ValueError, match="(?i)size|shape|nv|dimension|v"):
            engine.set_state(correct_q, wrong_v)

    def test_set_state_calls_forward_kinematics_after_setting(self) -> None:
        """set_state() must refresh derived kinematics after updating q and v."""
        from unittest.mock import patch

        import numpy as np

        engine = self._make_engine_with_mock_model()
        new_q = np.ones(7) * 0.1
        new_v = np.ones(6) * 0.2

        with patch.object(engine, "forward") as mock_forward:
            engine.set_state(new_q, new_v)
            mock_forward.assert_called_once()

        np.testing.assert_array_almost_equal(engine.q, new_q)
        np.testing.assert_array_almost_equal(engine.v, new_v)
