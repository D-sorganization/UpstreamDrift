# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for the structured error hierarchy in movement_optimizer.errors."""

from __future__ import annotations

import pytest

from movement_optimizer.errors import (
    FileIOError as MOptFileIOError,
)
from movement_optimizer.errors import (
    MovementOptimizerError,
    OptimizationError,
    PhysicsError,
    ValidationError,
)


class TestMovementOptimizerError:
    """Tests for the base MovementOptimizerError class."""

    def test_basic_construction(self) -> None:
        err = MovementOptimizerError("something went wrong")
        assert err.message == "something went wrong"
        assert err.error_code == "MOVEMENT_OPTIMIZER_ERROR"
        assert err.recoverable is True
        assert err.suggestion == ""

    def test_custom_fields(self) -> None:
        err = MovementOptimizerError(
            "bad input",
            error_code="MY_CODE",
            recoverable=False,
            suggestion="Do something different.",
        )
        assert err.error_code == "MY_CODE"
        assert err.recoverable is False
        assert err.suggestion == "Do something different."

    def test_str_with_suggestion(self) -> None:
        err = MovementOptimizerError("oops", suggestion="try again")
        assert "oops" in str(err)
        assert "try again" in str(err)

    def test_str_without_suggestion(self) -> None:
        err = MovementOptimizerError("oops")
        assert str(err) == "oops"

    def test_is_exception(self) -> None:
        err = MovementOptimizerError("oops")
        assert isinstance(err, Exception)

    def test_empty_message_raises(self) -> None:
        with pytest.raises(ValueError):
            MovementOptimizerError("")

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(MovementOptimizerError) as exc_info:
            raise MovementOptimizerError("test error")
        assert exc_info.value.message == "test error"


class TestOptimizationError:
    """Tests for OptimizationError."""

    def test_is_subclass(self) -> None:
        err = OptimizationError("optimizer diverged")
        assert isinstance(err, MovementOptimizerError)

    def test_default_error_code(self) -> None:
        err = OptimizationError("failed")
        assert err.error_code == "OPT_FAILED"

    def test_default_recoverable(self) -> None:
        err = OptimizationError("failed")
        assert err.recoverable is True

    def test_default_suggestion_non_empty(self) -> None:
        err = OptimizationError("failed")
        assert len(err.suggestion) > 0

    def test_custom_error_code(self) -> None:
        err = OptimizationError("nope", error_code="OPT_CUSTOM")
        assert err.error_code == "OPT_CUSTOM"

    def test_caught_as_base_class(self) -> None:
        with pytest.raises(MovementOptimizerError):
            raise OptimizationError("optimizer blew up")


class TestFileIOError:
    """Tests for FileIOError."""

    def test_is_subclass(self) -> None:
        err = MOptFileIOError("disk full")
        assert isinstance(err, MovementOptimizerError)

    def test_default_error_code(self) -> None:
        err = MOptFileIOError("cannot open file")
        assert err.error_code == "FILE_IO_ERROR"

    def test_default_recoverable(self) -> None:
        err = MOptFileIOError("cannot open file")
        assert err.recoverable is True

    def test_default_suggestion_non_empty(self) -> None:
        err = MOptFileIOError("cannot open file")
        assert len(err.suggestion) > 0

    def test_custom_suggestion(self) -> None:
        err = MOptFileIOError("nope", suggestion="Check your disk.")
        assert err.suggestion == "Check your disk."


class TestValidationError:
    """Tests for ValidationError."""

    def test_is_subclass(self) -> None:
        err = ValidationError("bad value")
        assert isinstance(err, MovementOptimizerError)

    def test_default_error_code(self) -> None:
        err = ValidationError("bad value")
        assert err.error_code == "VALIDATION_ERROR"

    def test_default_recoverable(self) -> None:
        err = ValidationError("bad value")
        assert err.recoverable is True

    def test_default_suggestion_non_empty(self) -> None:
        err = ValidationError("bad value")
        assert len(err.suggestion) > 0


class TestPhysicsError:
    """Tests for PhysicsError."""

    def test_is_subclass(self) -> None:
        err = PhysicsError("singular matrix")
        assert isinstance(err, MovementOptimizerError)

    def test_default_error_code(self) -> None:
        err = PhysicsError("singular matrix")
        assert err.error_code == "PHYSICS_ERROR"

    def test_default_recoverable_is_false(self) -> None:
        # Physics errors are unrecoverable by default
        err = PhysicsError("singular matrix")
        assert err.recoverable is False

    def test_default_suggestion_non_empty(self) -> None:
        err = PhysicsError("singular matrix")
        assert len(err.suggestion) > 0

    def test_override_recoverable(self) -> None:
        err = PhysicsError("weird but fixable", recoverable=True)
        assert err.recoverable is True


class TestErrorHierarchyCatchAll:
    """Tests that the base class can catch all subclass exceptions."""

    @pytest.mark.parametrize(
        "exc_class",
        [OptimizationError, MOptFileIOError, ValidationError, PhysicsError],
    )
    def test_all_subclasses_caught_by_base(self, exc_class: type) -> None:
        with pytest.raises(MovementOptimizerError):
            raise exc_class("test")
