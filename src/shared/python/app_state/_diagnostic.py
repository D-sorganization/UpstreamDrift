"""DiagnosticEngine: self-diagnostic checks run on startup or on demand."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

CheckStatus = Literal["PASS", "FAIL", "SKIP"]


@dataclass
class DiagnosticResult:
    """The outcome of a single diagnostic check.

    Attributes:
        name: Human-readable check name.
        status: One of ``"PASS"``, ``"FAIL"``, or ``"SKIP"``.
        message: Optional detail string (e.g. error description).
    """

    name: str
    status: CheckStatus
    message: str = ""


class DiagnosticEngine:
    """Runs a battery of named boolean checks and collects results.

    Each registered check is a zero-argument callable that returns ``bool``.
    Exceptions inside a check are caught and converted to FAIL results so
    ``run_checks()`` never raises.

    Example::

        engine = DiagnosticEngine()
        engine.register_check("physics_import", lambda: _can_import_mujoco())
        results = engine.run_checks()
    """

    def __init__(self) -> None:
        self._checks: dict[str, Callable[[], bool]] = {}

    def register_check(self, name: str, check_fn: Callable[[], bool]) -> None:
        """Register a diagnostic check.

        Args:
            name: Unique, non-empty name for the check.
            check_fn: Zero-argument callable returning ``True`` on pass.

        Raises:
            ValueError: If *name* is empty.
        """
        if not name:
            raise ValueError("Diagnostic check name must be non-empty")
        self._checks[name] = check_fn

    def run_checks(self) -> list[DiagnosticResult]:
        """Execute all registered checks and return their results.

        Postcondition:
            Returns one :class:`DiagnosticResult` per registered check;
            never raises regardless of check behaviour.

        Returns:
            List of :class:`DiagnosticResult` in registration order.
        """
        results: list[DiagnosticResult] = []
        for name, fn in self._checks.items():
            results.append(self._run_single(name, fn))
        return results

    def _run_single(self, name: str, fn: Callable[[], bool]) -> DiagnosticResult:
        """Run one check and return a result, catching all exceptions.

        Args:
            name: Check name (for error reporting).
            fn: The check callable.

        Returns:
            :class:`DiagnosticResult` with status PASS, FAIL, or SKIP.
        """
        try:
            passed = bool(fn())
            status: CheckStatus = "PASS" if passed else "FAIL"
            message = "" if passed else "check returned False"
            return DiagnosticResult(name=name, status=status, message=message)
        except Exception as exc:  # noqa: BLE001 — intentional catch-all per spec
            logger.warning("Diagnostic check %r raised: %s", name, exc)
            return DiagnosticResult(
                name=name,
                status="FAIL",
                message=f"{type(exc).__name__}: {exc}",
            )
