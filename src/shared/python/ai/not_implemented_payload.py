"""Shared honest ``not_implemented`` verdict payloads for AI tools and agent control."""

from __future__ import annotations

from typing import Any

NOT_IMPLEMENTED_STATUS = "not_implemented"
_RESERVED_VERDICT_KEYS = frozenset({"success", "error", "status"})


def build_not_implemented_payload(
    *,
    capability: str,
    message: str,
    **metadata: Any,
) -> dict[str, Any]:
    """Build the canonical honest refusal dict for a missing capability."""
    from src.shared.python.contracts import ensure, require

    require(
        not _RESERVED_VERDICT_KEYS.intersection(metadata),
        "Placeholder metadata may not override the verdict keys "
        f"{sorted(_RESERVED_VERDICT_KEYS)}; got "
        f"{sorted(_RESERVED_VERDICT_KEYS.intersection(metadata))}.",
    )
    payload: dict[str, Any] = {
        "success": False,
        "error": NOT_IMPLEMENTED_STATUS,
        "status": NOT_IMPLEMENTED_STATUS,
        "capability": capability,
        "message": message,
        **metadata,
    }
    ensure(
        payload["success"] is False
        and payload["status"] == NOT_IMPLEMENTED_STATUS
        and payload["error"] == NOT_IMPLEMENTED_STATUS,
        "A placeholder capability must report an honest failure.",
        value=payload,
    )
    return payload
