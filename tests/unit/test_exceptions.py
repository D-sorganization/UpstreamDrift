"""Tests for exceptions and core._core modules (Issues #1949, #1744)."""

from __future__ import annotations

import pytest

from src.shared.python.exceptions import EngineNotFoundError, GolfModelingError


class TestExceptionShim:
    def test_golf_modeling_error_raises(self) -> None:
        with pytest.raises(GolfModelingError):
            raise GolfModelingError("test error")

    def test_engine_not_found_error_raises(self) -> None:
        with pytest.raises(EngineNotFoundError):
            raise EngineNotFoundError("engine missing")

    def test_engine_not_found_is_subclass(self) -> None:
        assert issubclass(EngineNotFoundError, GolfModelingError)

    def test_exception_message(self) -> None:
        err = GolfModelingError("specific message")
        assert "specific message" in str(err)

    def test_both_are_exceptions(self) -> None:
        assert issubclass(GolfModelingError, Exception)
        assert issubclass(EngineNotFoundError, Exception)
