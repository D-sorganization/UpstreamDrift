# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Structured error hierarchy for Movement Optimizer.

All public exceptions raised by this package inherit from
``MovementOptimizerError``, giving callers a single base class to catch
while still allowing fine-grained handling of specific failure modes.
"""

from __future__ import annotations


class MovementOptimizerError(Exception):
    """Base class for all Movement Optimizer exceptions.

    Parameters
    ----------
    message:
        Human-readable description of the error.
    error_code:
        Short machine-readable identifier (e.g. ``"OPT_DIVERGED"``).
    recoverable:
        ``True`` if the user can take action to resolve the problem
        (e.g. adjust parameters), ``False`` for unrecoverable failures.
    suggestion:
        Optional actionable suggestion displayed to the user.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "MOVEMENT_OPTIMIZER_ERROR",
        recoverable: bool = True,
        suggestion: str = "",
    ) -> None:
        if not message:
            raise ValueError("message must be a non-empty string")
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.recoverable = recoverable
        self.suggestion = suggestion

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message}  Suggestion: {self.suggestion}"
        return self.message


class OptimizationError(MovementOptimizerError):
    """Raised when the trajectory optimizer fails or diverges.

    Common causes include parameter ranges that produce an infeasible
    base-of-support constraint or a numerical singularity in the
    Lagrangian dynamics.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "OPT_FAILED",
        recoverable: bool = True,
        suggestion: str = (
            "Try reducing the range of motion, increasing the movement duration, "
            "or adjusting the smoothness weight."
        ),
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            recoverable=recoverable,
            suggestion=suggestion,
        )


class FileIOError(MovementOptimizerError):
    """Raised when a file read or write operation fails.

    This wraps lower-level ``OSError`` / ``IOError`` instances and
    provides a user-friendly message with actionable guidance.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "FILE_IO_ERROR",
        recoverable: bool = True,
        suggestion: str = (
            "Check that the file path is correct, that you have the necessary "
            "permissions, and that there is sufficient disk space."
        ),
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            recoverable=recoverable,
            suggestion=suggestion,
        )


class ValidationError(MovementOptimizerError):
    """Raised when user-supplied parameters fail validation.

    Follows the DBC (Design-by-Contract) pattern: callers are
    expected to supply valid inputs; this exception surfaces when they
    do not.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "VALIDATION_ERROR",
        recoverable: bool = True,
        suggestion: str = "Review the parameter values and ensure they are within the accepted ranges.",
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            recoverable=recoverable,
            suggestion=suggestion,
        )


class PhysicsError(MovementOptimizerError):
    """Raised when a physics computation produces an invalid result.

    Examples include singular mass matrices, non-finite torque values,
    or a body model whose segments form a degenerate configuration.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "PHYSICS_ERROR",
        recoverable: bool = False,
        suggestion: str = (
            "Verify that the body model parameters are physically plausible "
            "(e.g. positive segment lengths and masses)."
        ),
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            recoverable=recoverable,
            suggestion=suggestion,
        )
