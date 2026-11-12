"""Fleet Agent API v1 helpers for ``ud-sim-mcp`` (UpstreamDrift #15022).

The conformance checks in ``scripts/config/fleet_agent_api_conformance/v1/template.json``
follow the Repository_Management template pattern (issue #2885); the template JSON
is vendored locally because the RM package is not readable from every agent host.
"""

from __future__ import annotations

from typing import Any, Literal

CONTRACT_VERSION: Literal["fleet-agent-api/v1"] = "fleet-agent-api/v1"

LocalRunStatus = Literal["queued", "running", "succeeded", "failed", "expired"]
FleetRunStatus = Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "EXPIRED"]

_STATUS_TO_FLEET: dict[str, FleetRunStatus] = {
    "queued": "QUEUED",
    "running": "RUNNING",
    "succeeded": "SUCCEEDED",
    "failed": "FAILED",
    "expired": "EXPIRED",
}


def fleet_run_status(local_status: str | None) -> FleetRunStatus | None:
    """Map sim_run lifecycle status strings to the Fleet v1 enum."""
    if not local_status:
        return None
    key = local_status.strip().lower()
    return _STATUS_TO_FLEET.get(key)


def map_sim_run_record_v1(record: dict[str, Any]) -> dict[str, Any]:
    """Project a sim_run JSON record into the Fleet Agent API v1 run shape."""
    local_status = record.get("status")
    fleet_status = fleet_run_status(
        str(local_status) if local_status is not None else ""
    )
    result = record.get("result") or {}
    prov = result.get("provenance") or record.get("provenance") or {}
    return {
        "contract_version": CONTRACT_VERSION,
        "status": fleet_status,
        "schema_version": record.get("schema_version"),
        "run_id": record.get("run_id"),
        "provenance": {
            "contract_version": CONTRACT_VERSION,
            "repo_sha": prov.get("repo_sha"),
            "engine_type": prov.get("engine_type")
            or record.get("request", {}).get("engine_type"),
        },
    }


def assert_run_record_conformance(record: dict[str, Any]) -> list[str]:
    """Validate a tool payload against Fleet v1 sim_run mapping rules."""
    failures: list[str] = []
    mapped = map_sim_run_record_v1(record)
    if mapped["contract_version"] != CONTRACT_VERSION:
        failures.append("contract_version mismatch")
    fleet_status = mapped["status"]
    if fleet_status is None:
        failures.append(f"unmapped status {record.get('status')!r}")
    elif fleet_status not in _STATUS_TO_FLEET.values():
        failures.append(f"invalid fleet status {fleet_status!r}")
    prov = mapped.get("provenance") or {}
    if prov.get("contract_version") != CONTRACT_VERSION:
        failures.append("missing provenance.contract_version")
    return failures
