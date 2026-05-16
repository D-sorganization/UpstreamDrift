"""DiagnosticEngine and DiagnosticResult for structured health checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

CheckStatus = Literal["pass", "fail", "warning"]


@dataclass
class DiagnosticResult:
    """Result of a single diagnostic check.

    Attributes:
        name: The check identifier.
        status: One of ``"pass"``, ``"fail"``, or ``"warning"``.
        message: Human-readable summary.
        details: Optional extra key-value context.
    """

    name: str
    status: CheckStatus
    message: str
    details: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.details is None:
            self.details = {}


class DiagnosticEngine:
    """Runs registered health checks and aggregates results."""

    def __init__(self) -> None:
        self._checks: list[tuple[str, Callable[[], DiagnosticResult]]] = []

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], DiagnosticResult],
    ) -> None:
        """Register a named check function."""
        self._checks.append((name, check_fn))

    def run_checks(self) -> list[DiagnosticResult]:
        """Run all registered checks and return their results."""
        results = []
        for name, fn in self._checks:
            result = self._run_single(name, fn)
            results.append(result)
        return results

    def _run_single(
        self,
        name: str,
        check_fn: Callable[[], DiagnosticResult],
    ) -> DiagnosticResult:
        try:
            return check_fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Diagnostic check %r raised unexpectedly: %s", name, exc)
            return DiagnosticResult(
                name=name,
                status="fail",
                message=f"Check raised unexpectedly: {exc}",
            )
