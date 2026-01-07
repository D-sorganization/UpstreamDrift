"""Integration tests for backend consistency."""

from __future__ import annotations

from dtack.backends.backend_factory import BackendFactory, BackendType


class TestBackendConsistency:
    """Test that backends produce consistent results."""

    def test_backend_factory_creation(self) -> None:
        """Test backend factory can create backends."""
        # This is a placeholder - actual tests require model files
        factory = BackendFactory()
        assert factory is not None

    def test_backend_type_enum(self) -> None:
        """Test BackendType enum values."""
        assert BackendType.PINOCCHIO == "pinocchio"
        assert BackendType.MUJOCO == "mujoco"
        assert BackendType.PINK == "pink"
