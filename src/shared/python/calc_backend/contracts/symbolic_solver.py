"""Pydantic contracts for symbolic solver endpoints. See issue #2682."""

from pydantic import BaseModel, Field


class SymbolicSolveRequest(BaseModel):
    """Request model for symbolic equation solving."""

    equation: str = Field(..., description="Equation to solve (e.g., 'x**2 - 4 = 0')")
    variable: str = Field(..., description="Variable to solve for (e.g., 'x')")


class SymbolicSolveResponse(BaseModel):
    """Response model for symbolic equation solving."""

    solutions: list[str] = Field(default_factory=list)
    raw_result: str | None = None
    available: bool = Field(default=False)
    error: str | None = None


class SymbolicDerivativeRequest(BaseModel):
    """Request model for symbolic differentiation."""

    expression: str = Field(..., description="Expression to differentiate")
    variable: str = Field(..., description="Variable with respect to")


class SymbolicDerivativeResponse(BaseModel):
    """Response model for symbolic differentiation."""

    derivative: str | None = None
    available: bool = Field(default=False)
    error: str | None = None


class SymbolicSimplifyRequest(BaseModel):
    """Request model for symbolic simplification."""

    expression: str = Field(..., description="Expression to simplify")


class SymbolicSimplifyResponse(BaseModel):
    """Response model for symbolic simplification."""

    simplified: str | None = None
    available: bool = Field(default=False)
    error: str | None = None


class SymbolicHelpResponse(BaseModel):
    """Help metadata for symbolic calculator capabilities."""

    supported_operations: list[str]
    examples: list[dict[str, str]]
    available: bool = Field(default=False)
