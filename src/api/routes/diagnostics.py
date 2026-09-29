"""Diagnostics and integrations-health routes (issue #7458).

Exposes the desktop-grade launcher diagnostics report and the integrations
health probes to the web UI, so browser mode answers "which engines are
installed and at what version, and what's broken" identically to the PyQt6
launcher's Diagnostics dialog and Integrations Health panel.

Single-implementation guarantee: probe logic lives in
``src.launchers.launcher_diagnostics`` (``DIAGNOSTIC_CHECKS`` +
``LauncherDiagnostics``) and ``src.launchers.integrations_health_data``;
this module only schedules those probes and serializes their results.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, status

from src.api.utils.datetime_compat import iso_format, utc_now
from src.shared.python.config.environment import is_production
from src.shared.python.core.process_safety import safe_gather
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

#: Per-probe timeout (seconds). Some probes shell out to git or import heavy
#: physics engines on first touch; a slow probe is reported as a ``warning``
#: result instead of stalling the whole report.
PROBE_TIMEOUT_SECONDS = 15.0

#: Timeout (seconds) for the integrations probe batch (PATH lookups, env-var
#: checks, and one small JSON config read — normally well under a second).
INTEGRATIONS_TIMEOUT_SECONDS = 10.0


def _guard_production() -> None:
    """Hide diagnostics endpoints in production deployments (matches
    the existing ``GET /api/diagnostics`` behaviour in ``core.py``)."""
    if is_production():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


async def _run_probe(
    diag: Any,
    check_name: str,
    diagnostic_result_type: Any,
) -> Any:
    """Run one diagnostic probe in a worker thread with a timeout.

    Args:
        diag: Shared :class:`LauncherDiagnostics` instance.
        check_name: Entry from :data:`DIAGNOSTIC_CHECKS`; resolved to the
            ``check_<name>`` method.

    Returns:
        The probe's :class:`DiagnosticResult`, or a synthetic ``warning``
        result when the probe times out or raises.
    """
    method = getattr(diag, f"check_{check_name}")
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, method), timeout=PROBE_TIMEOUT_SECONDS
        )
    except TimeoutError:
        logger.warning(
            "Diagnostic probe %s timed out after %.1fs",
            check_name,
            PROBE_TIMEOUT_SECONDS,
        )
        return diagnostic_result_type(
            name=check_name,
            status="warning",
            message=f"Probe timed out after {PROBE_TIMEOUT_SECONDS:.0f}s",
            details={"timeout_seconds": PROBE_TIMEOUT_SECONDS},
            duration_ms=PROBE_TIMEOUT_SECONDS * 1000,
        )
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        logger.exception("Diagnostic probe %s raised", check_name)
        return diagnostic_result_type(
            name=check_name,
            status="warning",
            message=f"Probe raised {type(exc).__name__}: {exc}",
            details={"error": str(exc)},
        )


@router.get("/diagnostics/full", response_model=None)
async def get_full_diagnostics() -> dict[str, Any]:
    """Desktop-grade diagnostics report for the web UI (issue #7458).

    Runs every probe in :data:`DIAGNOSTIC_CHECKS` (engine availability and
    versions, dependency health, model registry and asset checks, git/vendor
    metadata, ...) concurrently in the default thread pool, each capped at
    ``PROBE_TIMEOUT_SECONDS``.

    Response-time expectations: typically 1-3 s when engines have already been
    imported; first call after server start can take longer because engine
    probes import heavy native modules. Worst case is bounded by
    ``PROBE_TIMEOUT_SECONDS`` (probes run concurrently), after which slow
    probes are reported as ``warning`` entries rather than blocking.

    Returns:
        The same report shape the desktop Diagnostics dialog renders:
        ``summary``, ``categories``, ``checks``, and ``recommendations``.
    """
    _guard_production()

    from src.launchers.launcher_diagnostics import (
        DIAGNOSTIC_CHECKS,
        DiagnosticResult,
        LauncherDiagnostics,
        build_report,
    )

    diag = LauncherDiagnostics()
    outcomes = await safe_gather(
        *(_run_probe(diag, name, DiagnosticResult) for name in DIAGNOSTIC_CHECKS)
    )
    results: list[DiagnosticResult] = []
    for name, outcome in zip(DIAGNOSTIC_CHECKS, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            results.append(
                DiagnosticResult(
                    name=name,
                    status="warning",
                    message=f"Probe failed: {outcome}",
                    details={"error": str(outcome)},
                )
            )
        else:
            results.append(outcome)

    return build_report(results, expected_tiles=len(diag.EXPECTED_TILE_IDS))


@router.get("/integrations/health", response_model=None)
async def get_integrations_health() -> dict[str, Any]:
    """Integrations Health probe results (MCP / CLI / API), issue #7458.

    Mirrors the desktop Integrations Health panel: one record per integration
    with the shared status taxonomy (``healthy`` / ``configured`` /
    ``warning`` / ``error`` / ``unconfigured`` / ``unknown``), plus the same
    copy-as-markdown rendering the desktop panel produces.

    Secret redaction is enforced server-side: record details and the markdown
    export pass through ``redact_secrets`` so Bearer tokens and configured
    provider key values never leave the process.

    Response-time expectations: probes are PATH lookups, env-var presence
    checks, and one small JSON config read — normally well under one second.
    The batch runs in a worker thread capped at
    ``INTEGRATIONS_TIMEOUT_SECONDS`` (503 on timeout).

    Returns:
        ``generated_at`` (ISO 8601), ``records`` (redacted), and ``markdown``
        (the desktop copy-diagnostics report).
    """
    _guard_production()

    from src.launchers.integrations_health_data import (
        collect_all,
        copy_diagnostics,
        record_to_redacted_dict,
    )

    loop = asyncio.get_running_loop()
    try:
        records = await asyncio.wait_for(
            loop.run_in_executor(None, collect_all),
            timeout=INTEGRATIONS_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Integrations health probes timed out after "
                f"{INTEGRATIONS_TIMEOUT_SECONDS:.0f}s"
            ),
        ) from exc

    return {
        "generated_at": iso_format(utc_now()),
        "records": [record_to_redacted_dict(record) for record in records],
        "markdown": copy_diagnostics(records),
    }
