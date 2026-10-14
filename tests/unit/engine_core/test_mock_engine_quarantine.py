"""Tests for MockPhysicsEngine quarantine guard functionality.

This module tests the quarantine mechanism that prevents MockPhysicsEngine
from being used in fail-closed scientific verification lanes.
"""

from __future__ import annotations

import unittest.mock

import pytest

from src.shared.python.engine_core.mock_engine import (
    MockEngineQuarantineError,
    MockPhysicsEngine,
    assert_not_mock_engine,
    get_mock_engine,
)

pytestmark = pytest.mark.unit


class TestMockEngineQuarantine:
    """Test MockPhysicsEngine quarantine behavior."""

    def test_default_instantiation_works_without_quarantine_env(self) -> None:
        """MockPhysicsEngine() succeeds when quarantine env vars are unset."""
        engine = MockPhysicsEngine()
        assert engine is not None
        assert engine.is_mock_engine is True

    def test_is_mock_engine_attribute_set_to_true(self) -> None:
        """MockPhysicsEngine.is_mock_engine is always True."""
        engine = MockPhysicsEngine()
        assert engine.is_mock_engine is True

    def test_assert_not_mock_engine_rejects_mock_instance(self) -> None:
        """assert_not_mock_engine raises for MockPhysicsEngine instances."""
        engine = MockPhysicsEngine()
        with pytest.raises(
            MockEngineQuarantineError,
            match="Mock or stub physics engine.*MockPhysicsEngine.*quarantined",
        ):
            assert_not_mock_engine(engine)

    def test_assert_not_mock_engine_rejects_unittest_mock(self) -> None:
        """assert_not_mock_engine raises for unittest.mock stubs."""
        mock_engine = unittest.mock.MagicMock()
        with pytest.raises(
            MockEngineQuarantineError,
            match="Mock or stub physics engine.*quarantined",
        ):
            assert_not_mock_engine(mock_engine)

    def test_assert_not_mock_engine_raises_on_none(self) -> None:
        """assert_not_mock_engine raises ValueError if engine is None."""
        with pytest.raises(ValueError, match="engine must not be None"):
            assert_not_mock_engine(None)

    def test_assert_not_mock_engine_passes_with_non_mock_engine(self) -> None:
        """assert_not_mock_engine passes for non-mock objects."""

        class RealEngineStub:
            is_mock_engine = False

        engine = RealEngineStub()
        assert_not_mock_engine(engine)  # Should not raise

    def test_assert_not_mock_engine_custom_context_message(self) -> None:
        """assert_not_mock_engine includes custom context in error message."""
        engine = MockPhysicsEngine()
        with pytest.raises(
            MockEngineQuarantineError,
            match="cannot be used for custom_context",
        ):
            assert_not_mock_engine(engine, context="custom_context")

    def test_forbid_env_var_raises_on_instantiation(self, monkeypatch) -> None:
        """Setting GOLF_FORBID_MOCK_ENGINE=1 causes MockPhysicsEngine() to raise."""
        monkeypatch.setenv("GOLF_FORBID_MOCK_ENGINE", "1")
        with pytest.raises(
            MockEngineQuarantineError,
            match="MockPhysicsEngine is forbidden when GOLF_FORBID_MOCK_ENGINE",
        ):
            MockPhysicsEngine()

    def test_forbid_env_var_true_raises_on_instantiation(self, monkeypatch) -> None:
        """Setting GOLF_FORBID_MOCK_ENGINE=true causes MockPhysicsEngine() to raise."""
        monkeypatch.setenv("GOLF_FORBID_MOCK_ENGINE", "true")
        with pytest.raises(MockEngineQuarantineError):
            MockPhysicsEngine()

    def test_forbid_env_var_yes_raises_on_instantiation(self, monkeypatch) -> None:
        """Setting GOLF_FORBID_MOCK_ENGINE=yes causes MockPhysicsEngine() to raise."""
        monkeypatch.setenv("GOLF_FORBID_MOCK_ENGINE", "yes")
        with pytest.raises(MockEngineQuarantineError):
            MockPhysicsEngine()

    def test_forbid_env_var_case_insensitive(self, monkeypatch) -> None:
        """GOLF_FORBID_MOCK_ENGINE env var is case-insensitive."""
        monkeypatch.setenv("GOLF_FORBID_MOCK_ENGINE", "TRUE")
        with pytest.raises(MockEngineQuarantineError):
            MockPhysicsEngine()

    def test_fail_closed_env_var_raises_on_instantiation(self, monkeypatch) -> None:
        """Setting GOLF_FAIL_CLOSED_PHYSICS=1 causes MockPhysicsEngine() to raise."""
        monkeypatch.setenv("GOLF_FAIL_CLOSED_PHYSICS", "1")
        with pytest.raises(
            MockEngineQuarantineError,
            match="MockPhysicsEngine is forbidden when.*GOLF_FAIL_CLOSED_PHYSICS",
        ):
            MockPhysicsEngine()

    def test_fail_closed_env_var_true_raises_on_instantiation(
        self, monkeypatch
    ) -> None:
        """Setting GOLF_FAIL_CLOSED_PHYSICS=true causes MockPhysicsEngine() to raise."""
        monkeypatch.setenv("GOLF_FAIL_CLOSED_PHYSICS", "true")
        with pytest.raises(MockEngineQuarantineError):
            MockPhysicsEngine()

    def test_fail_closed_env_var_case_insensitive(self, monkeypatch) -> None:
        """GOLF_FAIL_CLOSED_PHYSICS env var is case-insensitive."""
        monkeypatch.setenv("GOLF_FAIL_CLOSED_PHYSICS", "YES")
        with pytest.raises(MockEngineQuarantineError):
            MockPhysicsEngine()

    def test_get_mock_engine_works_without_quarantine(self) -> None:
        """get_mock_engine() succeeds when quarantine is not enabled."""
        engine = get_mock_engine()
        assert isinstance(engine, MockPhysicsEngine)
        assert engine.is_mock_engine is True

    def test_get_mock_engine_fails_with_forbid_env_var(self, monkeypatch) -> None:
        """get_mock_engine() raises when GOLF_FORBID_MOCK_ENGINE is set."""
        monkeypatch.setenv("GOLF_FORBID_MOCK_ENGINE", "1")
        with pytest.raises(MockEngineQuarantineError):
            get_mock_engine()

    def test_get_mock_engine_fails_with_fail_closed_env_var(self, monkeypatch) -> None:
        """get_mock_engine() raises when GOLF_FAIL_CLOSED_PHYSICS is set."""
        monkeypatch.setenv("GOLF_FAIL_CLOSED_PHYSICS", "1")
        with pytest.raises(MockEngineQuarantineError):
            get_mock_engine()

    def test_quarantine_env_var_with_whitespace(self, monkeypatch) -> None:
        """Env var values with leading/trailing whitespace are handled."""
        monkeypatch.setenv("GOLF_FORBID_MOCK_ENGINE", "  1  ")
        with pytest.raises(MockEngineQuarantineError):
            MockPhysicsEngine()

    def test_quarantine_env_var_invalid_value_does_not_forbid(
        self, monkeypatch
    ) -> None:
        """Invalid env var values (other than 1/true/yes) do not forbid instantiation."""
        monkeypatch.setenv("GOLF_FORBID_MOCK_ENGINE", "0")
        engine = MockPhysicsEngine()
        assert engine.is_mock_engine is True

    def test_quarantine_env_var_empty_does_not_forbid(self, monkeypatch) -> None:
        """Empty env var values do not forbid instantiation."""
        monkeypatch.setenv("GOLF_FORBID_MOCK_ENGINE", "")
        engine = MockPhysicsEngine()
        assert engine.is_mock_engine is True

    def test_assert_not_mock_engine_with_is_mock_engine_attribute(self) -> None:
        """assert_not_mock_engine detects is_mock_engine=True on non-MockPhysicsEngine objects."""

        class FakeEngine:
            is_mock_engine = True

        fake_engine = FakeEngine()
        with pytest.raises(MockEngineQuarantineError):
            assert_not_mock_engine(fake_engine)

    def test_quarantine_error_is_runtime_error(self, monkeypatch) -> None:
        """MockEngineQuarantineError is a RuntimeError subclass."""
        monkeypatch.setenv("GOLF_FORBID_MOCK_ENGINE", "1")
        try:
            MockPhysicsEngine()
        except RuntimeError as e:
            assert isinstance(e, RuntimeError)
            assert isinstance(e, MockEngineQuarantineError)

    def test_forbid_both_env_vars_raises(self, monkeypatch) -> None:
        """Setting both env vars to true still raises on instantiation."""
        monkeypatch.setenv("GOLF_FORBID_MOCK_ENGINE", "1")
        monkeypatch.setenv("GOLF_FAIL_CLOSED_PHYSICS", "1")
        with pytest.raises(MockEngineQuarantineError):
            MockPhysicsEngine()
