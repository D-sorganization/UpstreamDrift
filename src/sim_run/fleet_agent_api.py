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


_FLEET_VALUES = frozenset(_STATUS_TO_FLEET.values())


def fleet_run_status(local_status: str | None) -> FleetRunStatus | None:
    """Map sim_run lifecycle status strings to the Fleet v1 enum."""
    if not local_status:
        return None
    raw = local_status.strip()
    upper = raw.upper()
    if upper in _FLEET_VALUES:
        return upper  # type: ignore[return-value]
    return _STATUS_TO_FLEET.get(raw.lower())


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


def client_run_record_v1(record: dict[str, Any]) -> dict[str, Any]:
    """Apply Fleet v1 public fields to a local run record for MCP clients."""
    mapped = map_sim_run_record_v1(record)
    fleet_status = mapped["status"]
    if fleet_status is None:
        raise ValueError(f"unmapped run status {record.get('status')!r}")
    out: dict[str, Any] = dict(record)
    out["contract_version"] = CONTRACT_VERSION
    out["status"] = fleet_status
    out["provenance"] = mapped["provenance"]
    result = out.get("result")
    if isinstance(result, dict):
        result_copy = dict(result)
        prov = result_copy.get("provenance")
        if isinstance(prov, dict):
            prov_copy = dict(prov)
            prov_copy["contract_version"] = CONTRACT_VERSION
            result_copy["provenance"] = prov_copy
        out["result"] = result_copy
    return out


def assert_run_record_conformance(record: dict[str, Any]) -> list[str]:
    """Validate a client-visible run record is already Fleet v1 shaped."""
    failures: list[str] = []
    if record.get("contract_version") != CONTRACT_VERSION:
        failures.append("contract_version mismatch")
    status = record.get("status")
    if status not in _FLEET_VALUES:
        failures.append(f"invalid fleet status {status!r}")
    prov = record.get("provenance") or {}
    if prov.get("contract_version") != CONTRACT_VERSION:
        nested = (record.get("result") or {}).get("provenance") or {}
        if nested.get("contract_version") != CONTRACT_VERSION:
            failures.append("missing provenance.contract_version")
    return failures
