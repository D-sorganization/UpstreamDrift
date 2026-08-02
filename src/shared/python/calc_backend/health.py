# ruff: noqa: E501
"""Health check endpoints for containerization and deployment.

Provides liveness and readiness probes for Kubernetes/Docker orchestration.
- /api/health: Liveness probe - is the app running?
- /api/ready: Readiness probe - is the app ready to serve requests?

See: https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum

logger = logging.getLogger(__name__)


class CheckStatus(StrEnum):
    """Health check status enum."""

    OK = "ok"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    name: str
    status: CheckStatus
    details: dict[str, Any] | None = None
    error: str | None = None


class HealthChecker:
    """Manages health checks for the calculation backend."""

    def __init__(self) -> None:
        """Initialize health checker."""
        self.checks: dict[str, Any] = {}

    def add_check(self, name: str, check_fn: Any) -> None:
        """Register a health check function.

        Args:
            name: Name of the check (e.g., 'database', 'redis')
            check_fn: Async or sync callable that returns HealthCheckResult

        Example:
            async def check_db():
                try:
                    # Test database connection
                    return HealthCheckResult('database', CheckStatus.OK)
                except Exception as e:
                    return HealthCheckResult('database', CheckStatus.UNHEALTHY, error=str(e))
        """  # noqa: E501
        self.checks[name] = check_fn

    async def run_checks(self) -> tuple[CheckStatus, dict[str, Any]]:
        """Run all registered health checks.

        Returns:
            Tuple of (overall_status, results_dict)
            - overall_status: OK, DEGRADED, or UNHEALTHY
            - results_dict: Detailed results for each check
        """
        results: dict[str, dict[str, Any]] = {}
        statuses = []

        for name, check_fn in self.checks.items():
            try:
                if callable(check_fn):
                    # Support both sync and async callables
                    import inspect

                    if inspect.iscoroutinefunction(check_fn):
                        result = await check_fn()
                    else:
                        result = check_fn()
                else:
                    result = check_fn

                results[name] = {
                    "status": result.status.value,
                    "details": result.details or {},
                }
                if result.error:
                    results[name]["error"] = result.error
                statuses.append(result.status)
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = {
                    "status": CheckStatus.UNHEALTHY.value,
                    "error": str(e),
                }
                statuses.append(CheckStatus.UNHEALTHY)

        # Determine overall status
        if any(s == CheckStatus.UNHEALTHY for s in statuses):
            overall_status = CheckStatus.UNHEALTHY
        elif any(s == CheckStatus.DEGRADED for s in statuses):
            overall_status = CheckStatus.DEGRADED
        else:
            overall_status = CheckStatus.OK

        return overall_status, results


# Global health checker instance
_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    return _health_checker


# ============================================================================
# Built-in Health Checks
# ============================================================================


def check_python_runtime() -> HealthCheckResult:
    """Check Python runtime is healthy."""
    try:
        return HealthCheckResult(
            "python_runtime",
            CheckStatus.OK,
            details={
                "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",  # noqa: E501
                "implementation": sys.implementation.name,
            },
        )
    except Exception as e:
        return HealthCheckResult("python_runtime", CheckStatus.UNHEALTHY, error=str(e))


async def check_dependencies() -> HealthCheckResult:
    """Check critical Python dependencies are importable.

    Verifies:
    - fastapi
    - pydantic
    - numpy
    - pandas
    """
    required_modules = [
        "fastapi",
        "pydantic",
        "numpy",
        "pandas",
        "scipy",
    ]
    missing = []

    for module_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        return HealthCheckResult(
            "dependencies",
            CheckStatus.UNHEALTHY,
            error=f"Missing required modules: {', '.join(missing)}",
        )

    return HealthCheckResult(
        "dependencies",
        CheckStatus.OK,
        details={"required_modules": required_modules},
    )


def check_application_state() -> HealthCheckResult:
    """Check application is properly initialized."""
    # This is a placeholder for application-specific state checks
    # Could check: configuration loaded, modules initialized, etc.
    try:
        return HealthCheckResult(
            "application_state",
            CheckStatus.OK,
            details={"initialized": True},
        )
    except Exception as e:
        return HealthCheckResult(
            "application_state",
            CheckStatus.UNHEALTHY,
            error=str(e),
        )


# Register default checks
_health_checker.add_check("python_runtime", check_python_runtime)
_health_checker.add_check("dependencies", check_dependencies)
_health_checker.add_check("application_state", check_application_state)
