"""Unit tests for contact dynamics module.

Tests cover:
    - ContactState creation and validation
    - FrictionCone operations
    - ContactManager functionality
    - Grasp analysis
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from src.robotics.contact.friction_cone import (
    FrictionCone,
    compute_friction_cone_constraint,
    linearize_friction_cone,
    project_to_friction_cone,
)
from src.robotics.contact.grasp_analysis import (
    check_force_closure,
    compute_grasp_matrix,
    compute_grasp_quality,
)
from src.robotics.core.types import ContactState


class TestIssue2499ContactManagerStateRestoration:
    """Issue #2499: detect_contacts(q) must restore the original engine state."""

    @pytest.fixture
    def mock_engine(self) -> object:
        """Engine mock that satisfies RoboticsCapable runtime check."""
        from unittest.mock import MagicMock

        from src.robotics.core.protocols import RoboticsCapable

        engine = MagicMock(spec=RoboticsCapable)
        # Track internal state
        state: dict[str, np.ndarray] = {
            "q": np.zeros(3),
            "v": np.zeros(3),
        }

        def get_state() -> tuple[np.ndarray, np.ndarray]:
            return state["q"].copy(), state["v"].copy()

        def set_state(q: np.ndarray, v: np.ndarray) -> None:
            state["q"] = q.copy()
            state["v"] = v.copy()

        engine.get_state.side_effect = get_state
        engine.set_state.side_effect = set_state
        engine._state = state  # expose for assertions
        return engine

    def test_detect_contacts_restores_q_after_speculative_call(
        self, mock_engine: object
    ) -> None:
        """After detect_contacts(q_new), engine must return to original q."""
        from src.robotics.contact.contact_manager import ContactManager

        q_orig = np.array([1.0, 2.0, 3.0])
        mock_engine.set_state(q_orig, np.zeros(3))

        manager = ContactManager(mock_engine)
        manager.detect_contacts(q=np.array([9.0, 9.0, 9.0]))

        q_after, _ = mock_engine.get_state()
        np.testing.assert_array_equal(q_after, q_orig)

    def test_detect_contacts_restores_v_after_speculative_call(
        self, mock_engine: object
    ) -> None:
        """After detect_contacts(q_new), engine must return to original v."""
        from src.robotics.contact.contact_manager import ContactManager

        v_orig = np.array([0.1, 0.2, 0.3])
        mock_engine.set_state(np.zeros(3), v_orig)

        manager = ContactManager(mock_engine)
        manager.detect_contacts(q=np.array([9.0, 9.0, 9.0]))

        _, v_after = mock_engine.get_state()
        np.testing.assert_array_equal(v_after, v_orig)

    def test_detect_contacts_without_q_does_not_change_state(
        self, mock_engine: object
    ) -> None:
        """detect_contacts() with no q argument must leave engine state unchanged."""
        from src.robotics.contact.contact_manager import ContactManager

        q_orig = np.array([5.0, 6.0, 7.0])
        mock_engine.set_state(q_orig, np.zeros(3))

        manager = ContactManager(mock_engine)
        manager.detect_contacts()

        q_after, _ = mock_engine.get_state()
        np.testing.assert_array_equal(q_after, q_orig)
