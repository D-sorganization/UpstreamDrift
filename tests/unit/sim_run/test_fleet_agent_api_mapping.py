"""Fleet Agent API v1 mapping for sim_run records (#15022)."""

from __future__ import annotations

import pytest

from src.sim_run.fleet_agent_api import (
    CONTRACT_VERSION,
    assert_run_record_conformance,
    client_run_record_v1,
    fleet_run_status,
    map_sim_run_record_v1,
)

pytestmark = pytest.mark.unit


@pytest.mark.unit
def test_fleet_run_status_maps_sim_run_values() -> None:
    assert fleet_run_status("succeeded") == "SUCCEEDED"
    assert fleet_run_status("failed") == "FAILED"


@pytest.mark.unit
def test_map_sim_run_record_v1_projects_contract_and_status() -> None:
    payload = {
        "schema_version": "1.0.0",
        "status": "succeeded",
        "run_id": "run_123",
        "result": {"provenance": {"repo_sha": "abc", "engine_type": "pendulum"}},
    }
    mapped = map_sim_run_record_v1(payload)
    assert mapped["contract_version"] == CONTRACT_VERSION
    assert mapped["status"] == "SUCCEEDED"
    assert mapped["provenance"]["contract_version"] == CONTRACT_VERSION
    assert mapped["provenance"]["repo_sha"] == "abc"
    assert assert_run_record_conformance(payload) == [
        "contract_version mismatch",
        "invalid fleet status 'succeeded'",
        "missing provenance.contract_version",
    ]
    public = client_run_record_v1(payload)
    assert not assert_run_record_conformance(public)
