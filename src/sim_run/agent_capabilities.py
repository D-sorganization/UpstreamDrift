"""Agent API capability runs with v1 records — shared by REST and MCP (#15021)."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from .counterfactual_run import execute_golden_ztcf
from .golf_simulator_run import execute_golden_golf_shot
from .payload_bounds import bound_payload, canonical_json
from .store import SimulationRunStore
from .worker import get_repo_sha

CapabilityId = Literal["counterfactual_ztcf_golden", "golf_simulator_golden_shot"]

_EXECUTORS: dict[str, Any] = {
    "counterfactual_ztcf_golden": execute_golden_ztcf,
    "golf_simulator_golden_shot": execute_golden_golf_shot,
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def input_sha256(capability_id: str, request: dict[str, Any] | None) -> str:
    """Hash pinned golden inputs for parity checks."""
    payload = {
        "capability_id": capability_id,
        "request": request or {},
        "golden_binding": "v1",
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def run_agent_capability(
    capability_id: CapabilityId,
    request: dict[str, Any] | None,
    store: SimulationRunStore | None = None,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Execute one capability and return a v1 run envelope (REST/MCP shared)."""
    executor = _EXECUTORS.get(capability_id)
    if executor is None:
        raise ValueError(f"Unknown capability_id '{capability_id}'")
    assigned = run_id or f"agent_{uuid.uuid4().hex}"
    created = _iso_now()
    digest = input_sha256(capability_id, request)
    try:
        result_body = executor(request)
    except (RuntimeError, TypeError, ValueError) as exc:
        return {
            "schema": "ud-agent-capability-run/v1",
            "schema_version": "1.0.0",
            "run_id": assigned,
            "capability_id": capability_id,
            "status": "failed",
            "input_sha256": digest,
            "created_at": created,
            "completed_at": _iso_now(),
            "provenance": {"repo_sha": get_repo_sha()},
            "error": {"message": str(exc)},
        }

    delivery = bound_payload(
        result_body,
        store=store,
        run_id=assigned,
        artifact_name="capability_result.json",
    )
    return {
        "schema": "ud-agent-capability-run/v1",
        "schema_version": "1.0.0",
        "run_id": assigned,
        "capability_id": capability_id,
        "status": "succeeded",
        "input_sha256": digest,
        "created_at": created,
        "completed_at": _iso_now(),
        "provenance": {"repo_sha": get_repo_sha()},
        "delivery": delivery,
    }
