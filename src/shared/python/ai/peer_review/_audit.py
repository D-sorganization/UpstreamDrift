"""Audit trail utilities for peer review."""

from __future__ import annotations

import time
from typing import Any


def _audit_event(
    kind: str,
    *,
    request_id: str,
    message: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single audit-trail event (DRY helper shared with chat layer)."""
    event: dict[str, Any] = {
        "kind": kind,
        "request_id": request_id,
        "timestamp": time.time(),
    }
    if message is not None:
        event["message"] = message
    if extra:
        event["extra"] = extra
    return event
