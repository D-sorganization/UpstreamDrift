"""Shared utilities for API route modules."""

from __future__ import annotations

from pathlib import Path

from fastapi.responses import JSONResponse


def not_implemented_json(detail: str, tracking_issue: int) -> JSONResponse:
    """Build an honest 501 response for a not-yet-implemented capability.

    Web parity policy (issue #7448): endpoints must never fabricate output
    for capabilities that are not actually implemented. Instead they return
    HTTP 501 with a body of ``{"detail": ..., "tracking_issue": ...}`` that
    points at the parity issue implementing the feature.

    Args:
        detail: Human-readable explanation of what is not implemented.
        tracking_issue: GitHub issue number tracking the implementation.

    Returns:
        JSONResponse with status 501 and the honest-contract body.
    """
    if not detail:
        raise ValueError("detail must be a non-empty string")
    if tracking_issue <= 0:
        raise ValueError("tracking_issue must be a positive issue number")
    return JSONResponse(
        status_code=501,
        content={"detail": detail, "tracking_issue": tracking_issue},
    )


def find_project_root() -> Path:
    """Find the project root directory by looking for known markers."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src" / "shared" / "urdf").exists():
            return parent
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()
