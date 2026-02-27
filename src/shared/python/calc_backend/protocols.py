"""Protocol interfaces for the calculation backend.

These protocols define the structural typing contracts for calculator
routers and their request/response models. They enable type-safe
integration between FastAPI routers and calculation engines without
tight coupling.

Usage:
    def run_calculation(
        engine: CalculationEngine[ScrubberRequest, ScrubberResponse],
        request: ScrubberRequest,
    ) -> ScrubberResponse:
        return engine.calculate(request)
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

RequestT = TypeVar("RequestT", bound=BaseModel, contravariant=True)
ResponseT = TypeVar("ResponseT", bound=BaseModel, covariant=True)


@runtime_checkable
class CalculationEngine(Protocol[RequestT, ResponseT]):
    """Protocol for a calculation engine that transforms requests into responses.

    This covers the common pattern in calc_backend routers where a Pydantic
    request model is validated, computation is performed, and a Pydantic
    response model is returned.
    """

    def calculate(self, request: RequestT) -> ResponseT:
        """Execute the calculation and return a response model."""
        ...


@runtime_checkable
class ValidationMixin(Protocol):
    """Protocol for input validation before calculation.

    Implementors can validate request data and raise ``ValueError``
    or ``HTTPException`` for invalid inputs.
    """

    def validate_inputs(self, request: Any) -> None:
        """Validate inputs, raising ValueError for invalid data."""
        ...


@runtime_checkable
class ExpressionEvaluator(Protocol):
    """Protocol for safe mathematical expression evaluation.

    Implementations must validate and safely evaluate string expressions
    without arbitrary code execution.
    """

    def evaluate(self, expression: str, namespace: dict[str, Any]) -> Any:
        """Evaluate a mathematical expression in a restricted namespace."""
        ...

    def validate(self, expression: str) -> bool:
        """Check whether an expression is syntactically valid and safe."""
        ...
