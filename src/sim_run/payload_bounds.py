"""Inline vs artifact delivery for Agent API capability payloads (#15021)."""

from __future__ import annotations

import json
from typing import Any

from .store import SimulationRunStore

INLINE_PAYLOAD_LIMIT_BYTES = 64 * 1024


def canonical_json(value: Any) -> str:
    """Deterministic JSON for hashing and size checks."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_size_bytes(payload: Any) -> int:
    """Return UTF-8 byte length of the canonical JSON serialisation."""
    return len(canonical_json(payload).encode("utf-8"))


def bound_payload(
    payload: dict[str, Any],
    *,
    store: SimulationRunStore | None,
    run_id: str,
    artifact_name: str = "result.json",
) -> dict[str, Any]:
    """Return inline JSON or an artifact handle when the payload exceeds 64 KiB."""
    size = payload_size_bytes(payload)
    if size <= INLINE_PAYLOAD_LIMIT_BYTES:
        return {"encoding": "inline", "size_bytes": size, "value": payload}
    if store is None:
        raise ValueError(
            f"Payload is {size} bytes (> {INLINE_PAYLOAD_LIMIT_BYTES}); "
            "a SimulationRunStore is required for artifact spill."
        )
    store.save_artifact(run_id, artifact_name, canonical_json(payload))
    return {
        "encoding": "artifact",
        "size_bytes": size,
        "artifact_ref": {"run_id": run_id, "artifact_name": artifact_name},
    }
