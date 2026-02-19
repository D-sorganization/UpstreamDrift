"""Contract violation exception hierarchy.

All exception classes for Design by Contract violations:
- ContractViolationError: Base exception
- PreconditionError: Precondition violations
- PostconditionError: Postcondition violations
- InvariantError: Class invariant violations
- StateError: Invalid state for operation
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ContractViolationError(Exception):
    """Base exception for contract violations."""

    def __init__(
        self,
        contract_type: str,
        message: str,
        function_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.contract_type = contract_type
        self.function_name = function_name
        self.details = details or {}

        error_msg = f"{contract_type} violation"
        if function_name:
            error_msg += f" in {function_name}"
        error_msg += f": {message}"

        if details:
            error_msg += f"\n  Details: {details}"

        super().__init__(error_msg)


class PreconditionError(ContractViolationError):
    """Raised when a precondition is violated."""

    def __init__(
        self,
        message: str,
        function_name: str | None = None,
        parameter: str | None = None,
        value: Any = None,
    ) -> None:
        details = {}
        if parameter:
            details["parameter"] = parameter
        if value is not None:
            details["value"] = repr(value)

        super().__init__(
            contract_type="Precondition",
            message=message,
            function_name=function_name,
            details=details if details else None,
        )
        self.parameter = parameter
        self.value = value


class PostconditionError(ContractViolationError):
    """Raised when a postcondition is violated."""

    def __init__(
        self,
        message: str,
        function_name: str | None = None,
        result: Any = None,
    ) -> None:
        details: dict[str, Any] = {}
        if result is not None:
            # Avoid storing large arrays
            if isinstance(result, np.ndarray):
                details["result_shape"] = result.shape
                details["result_dtype"] = str(result.dtype)
            else:
                details["result"] = repr(result)[:100]

        super().__init__(
            contract_type="Postcondition",
            message=message,
            function_name=function_name,
            details=details if details else None,
        )
        self.result = result


class InvariantError(ContractViolationError):
    """Raised when a class invariant is violated."""

    def __init__(
        self,
        message: str,
        class_name: str | None = None,
        method_name: str | None = None,
    ) -> None:
        details = {}
        if class_name:
            details["class"] = class_name
        if method_name:
            details["after_method"] = method_name

        super().__init__(
            contract_type="Invariant",
            message=message,
            function_name=method_name,
            details=details if details else None,
        )
        self.class_name = class_name
        self.method_name = method_name


class StateError(ContractViolationError):
    """Raised when an operation is attempted in an invalid state."""

    def __init__(
        self,
        message: str,
        current_state: str | None = None,
        required_state: str | None = None,
        operation: str | None = None,
    ) -> None:
        details = {}
        if current_state:
            details["current_state"] = current_state
        if required_state:
            details["required_state"] = required_state

        super().__init__(
            contract_type="State",
            message=message,
            function_name=operation,
            details=details if details else None,
        )
        self.current_state = current_state
        self.required_state = required_state
        self.operation = operation
