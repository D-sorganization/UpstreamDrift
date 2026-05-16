"""Best-effort FSP telemetry for the app-state ring buffer (Phase 3, #5504).

Emits an ``"fsp.computed"`` event into the app-state :class:`StateLogger`
ring buffer when one is available, so the Sidekick chat agent can
include the latest FSP metrics in its context.

If the ``app_state`` package is not importable (older branches, headless
test contexts, partial installs), the function silently degrades to a
debug-log line -- callers can wire it into the engine without worrying
about availability.

Usage::

    from src.shared.python.biomechanics.fsp_integration import compute_swing_fsp
    from src.shared.python.biomechanics.fsp_telemetry import log_fsp_computed

    result = compute_swing_fsp(sim_frames)
    log_fsp_computed(result)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


_FSP_EVENT_TYPE: str = "fsp.computed"


def log_fsp_computed(fsp_result: Any) -> None:
    """Emit an ``fsp.computed`` event into the app-state logger.

    Args:
        fsp_result: Anything exposing ``slope_deg`` / ``direction_deg``
            attributes.  Missing attributes are tolerated -- they are
            omitted from the payload rather than raising.

    The function never raises; failures are logged at DEBUG level so the
    call site does not need a try/except guard.
    """
    payload = _build_payload(fsp_result)
    try:
        from src.shared.python.app_state import get_state_logger  # local import
    except ImportError:
        logger.debug(
            "fsp_telemetry: app_state not available; skipping %s payload=%s",
            _FSP_EVENT_TYPE,
            payload,
        )
        return

    try:
        get_state_logger().log_event(_FSP_EVENT_TYPE, payload)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("fsp_telemetry: state logger raised %r; payload=%s", exc, payload)


def _build_payload(fsp_result: Any) -> dict[str, float]:
    """Coerce optional numeric attributes into a JSON-safe payload dict."""
    payload: dict[str, float] = {}
    for key in ("slope_deg", "direction_deg"):
        value = getattr(fsp_result, key, None)
        if value is None:
            continue
        try:
            payload[key] = float(value)
        except (TypeError, ValueError):
            continue
    return payload


__all__ = ["log_fsp_computed"]
