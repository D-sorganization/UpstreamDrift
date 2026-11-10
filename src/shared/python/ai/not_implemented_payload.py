"""Shared honest ``not_implemented`` verdict payloads for AI tools and agent control."""

from __future__ import annotations

from typing import Any

NOT_IMPLEMENTED_STATUS = "not_implemented"
_RESERVED_VERDICT_KEYS = frozenset({"success", "error", "status"})
_METADATA_RESERVED_KEYS_ERROR = (
    "Placeholder metadata may not override the verdict keys "
    "{reserved}; got {overlap}."
)


def _reject_reserved_metadata_keys(metadata: dict[str, Any]) -> None:
    """Reject metadata that tries to override honest-refusal verdict keys."""
    from src.shared.python.contracts import require

    overlap = _RESERVED_VERDICT_KEYS.intersection(metadata)
    require(
        not overlap,
        _METADATA_RESERVED_KEYS_ERROR.format(
            reserved=sorted(_RESERVED_VERDICT_KEYS),
            overlap=sorted(overlap),
        ),
    )


def build_not_implemented_payload(
    *,
    capability: str,
    message: str,
    **metadata: Any,
) -> dict[str, Any]:
    """Build the canonical honest refusal dict for a missing capability."""
    from src.shared.python.contracts import ensure

    _reject_reserved_metadata_keys(metadata)
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
