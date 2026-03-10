"""Tests for dtack backend factory pattern.

Tests the BackendFactory.create() method and BackendType enum
without requiring actual physics engine dependencies (uses mocks
for backends that require pinocchio/mujoco/pink).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestBackendType:
    """Tests for BackendType enum."""

    def test_pinocchio_value(self) -> None:
        """BackendType.PINOCCHIO should have value 'pinocchio'."""
        # Import inside test to handle missing dependencies
        try:
            from dtack.backends.backend_factory import BackendType
        except ImportError:
            pytest.skip("dtack dependencies missing")

        assert BackendType.PINOCCHIO.value == "pinocchio"

    def test_mujoco_value(self) -> None:
        """BackendType.MUJOCO should have value 'mujoco'."""
        try:
            from dtack.backends.backend_factory import BackendType
        except ImportError:
            pytest.skip("dtack dependencies missing")

        assert BackendType.MUJOCO.value == "mujoco"

    def test_pink_value(self) -> None:
        """BackendType.PINK should have value 'pink'."""
        try:
            from dtack.backends.backend_factory import BackendType
        except ImportError:
            pytest.skip("dtack dependencies missing")

        assert BackendType.PINK.value == "pink"

    def test_backend_type_is_str_enum(self) -> None:
        """BackendType should be a string enum for easy comparison."""
        try:
            from dtack.backends.backend_factory import BackendType
        except ImportError:
            pytest.skip("dtack dependencies missing")

        assert isinstance(BackendType.PINOCCHIO, str)


class TestBackendFactory:
    """Tests for BackendFactory.create()."""

    def test_unsupported_backend_raises(self) -> None:
        """Creating with an unknown backend type should raise ValueError."""
        try:
            from dtack.backends.backend_factory import BackendFactory
        except ImportError:
            pytest.skip("dtack dependencies missing")

        with pytest.raises(ValueError, match="Unsupported backend type"):
            BackendFactory.create("nonexistent_backend", "/fake/path.urdf")

    def test_create_pinocchio_backend(self) -> None:
        """Factory should create PinocchioBackend for 'pinocchio' type."""
        try:
            from dtack.backends.backend_factory import BackendFactory
        except ImportError:
            pytest.skip("dtack dependencies missing")

        mock_backend = MagicMock()
        with patch(
            "dtack.backends.backend_factory.PinocchioBackend",
            return_value=mock_backend,
        ):
            result = BackendFactory.create("pinocchio", "/fake/model.urdf")
            assert result is mock_backend

    def test_create_mujoco_backend(self) -> None:
        """Factory should create MuJoCoBackend for 'mujoco' type."""
        try:
            from dtack.backends.backend_factory import BackendFactory
        except ImportError:
            pytest.skip("dtack dependencies missing")

        mock_backend = MagicMock()
        with patch(
            "dtack.backends.backend_factory.MuJoCoBackend",
            return_value=mock_backend,
        ):
            result = BackendFactory.create("mujoco", "/fake/model.xml")
            assert result is mock_backend

    def test_create_pink_backend(self) -> None:
        """Factory should create PINKBackend for 'pink' type."""
        try:
            from dtack.backends.backend_factory import BackendFactory
        except ImportError:
            pytest.skip("dtack dependencies missing")

        mock_backend = MagicMock()
        with patch(
            "dtack.backends.backend_factory.PINKBackend",
            return_value=mock_backend,
        ):
            result = BackendFactory.create("pink", "/fake/model.urdf")
            assert result is mock_backend

    def test_create_with_enum_type(self) -> None:
        """Factory should accept BackendType enum values."""
        try:
            from dtack.backends.backend_factory import BackendFactory, BackendType
        except ImportError:
            pytest.skip("dtack dependencies missing")

        mock_backend = MagicMock()
        with patch(
            "dtack.backends.backend_factory.PinocchioBackend",
            return_value=mock_backend,
        ):
            result = BackendFactory.create(BackendType.PINOCCHIO, "/fake/model.urdf")
            assert result is mock_backend

    def test_case_insensitive_type(self) -> None:
        """Factory should handle uppercase/mixed-case backend types."""
        try:
            from dtack.backends.backend_factory import BackendFactory
        except ImportError:
            pytest.skip("dtack dependencies missing")

        mock_backend = MagicMock()
        with patch(
            "dtack.backends.backend_factory.PinocchioBackend",
            return_value=mock_backend,
        ):
            result = BackendFactory.create("PINOCCHIO", "/fake/model.urdf")
            assert result is mock_backend
