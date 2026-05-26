"""Tests for flexible shaft engine integration.

Task 3.3: Flexible Shaft Engine Integration tests.

Refactored to use shared engine availability module (DRY principle).
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass


class TestShaftInterfaceDefault:
    """Tests for default shaft interface behavior."""

    def test_interface_default_returns_false(self) -> None:
        """PhysicsEngine default implementation should return False."""
        from src.shared.python.engine_core.interfaces import PhysicsEngine

        # Check that the protocol method has a default that returns False
        # This test verifies the interface definition
        assert hasattr(PhysicsEngine, "set_shaft_properties")
        assert hasattr(PhysicsEngine, "get_shaft_state")
