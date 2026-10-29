"""Tests for sim-run schemas and model contracts (Issue #14086 / Epic #14085)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

jsonschema = pytest.importorskip("jsonschema")

from src.api.models.sim_run import (
    Provenance as ApiProvenance,
    ResultEnvelope as ApiResultEnvelope,
    RunRecord as ApiRunRecord,
    RunRequest as ApiRunRequest,
)
from src.sim_run.models import (
    Provenance,
    ResultEnvelope,
    RunRecord,
    RunRequest,
    export_schema,
    export_schemas,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "sim-run-v1.json"


def test_exported_schema_equals_checked_in_schema() -> None:
    """Checked-in schema must be identical to programmatic export and valid draft 2020-12."""
    assert _SCHEMA_PATH.is_file(), f"Missing {_SCHEMA_PATH}"
    checked_in = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    exported = export_schema()
    jsonschema.Draft202012Validator.check_schema(checked_in)
    assert checked_in == exported


def test_individual_schemas_exportable() -> None:
    """Exported individual schemas must contain expected model definitions."""
    schemas = export_schemas()
    for key in ("RunRequest", "RunRecord", "ResultEnvelope"):
        assert key in schemas and schemas[key]["type"] == "object"


def test_run_request_valid_and_defaults() -> None:
    """Valid RunRequest must populate version, bounds, and defaults."""
    req = RunRequest(engine_type="mujoco", duration=2.0, timestep=0.002)
    assert req.schema_version == "1.0.0" and req.engine_type == "mujoco"
    assert (
        req.duration == 2.0 and req.timestep == 0.002 and req.timeout_seconds == 300.0
    )


def test_run_request_rejects_unknown_engine() -> None:
    """Request with unknown engine must be rejected."""
    with pytest.raises((ValueError, ValidationError), match="Unknown engine_type"):
        RunRequest(engine_type="unreal_unsupported_engine")


@pytest.mark.parametrize("bad_duration", [-1.0, 0.0, 301.0, float("inf"), float("nan")])
def test_run_request_rejects_out_of_bounds_duration(bad_duration: float) -> None:
    """Request with non-positive, >300s, or non-finite duration must be rejected."""
    with pytest.raises((ValueError, ValidationError)):
        RunRequest(engine_type="mujoco", duration=bad_duration)


@pytest.mark.parametrize(
    "bad_timestep", [-0.01, 0.0, 1e-8, 0.5, float("inf"), float("nan")]
)
def test_run_request_rejects_out_of_bounds_timestep(bad_timestep: float) -> None:
    """Request with out-of-range or non-finite timestep must be rejected."""
    with pytest.raises((ValueError, ValidationError)):
        RunRequest(engine_type="mujoco", duration=1.0, timestep=bad_timestep)


def test_run_record_lifecycle_states() -> None:
    """RunRecord must accept all lifecycle statuses and reject invalid ones."""
    req = RunRequest(engine_type="mujoco", duration=1.0)
    for s in ("queued", "running", "succeeded", "failed", "expired"):
        assert (
            RunRecord(
                run_id="r1", status=s, request=req, created_at="2026-10-29T21:00:00Z"
            ).status
            == s
        )
    with pytest.raises(ValidationError):
        RunRecord(
            run_id="r1", status="bad", request=req, created_at="2026-10-29T21:00:00Z"
        )  # type: ignore[arg-type]


def test_result_envelope_validation_status() -> None:
    """ResultEnvelope must support provenance and valid/invalid/unvalidated statuses."""
    prov = Provenance(
        repo_sha="08fde",
        engine_type="mujoco",
        input_sha256="abc",
        created_at="2026-10-29T21:00:00Z",
    )
    for s in ("valid", "invalid", "unvalidated"):
        assert (
            ResultEnvelope(
                run_id="r1", provenance=prov, validation_status=s, success=True
            ).validation_status
            == s
        )
    with pytest.raises(ValidationError):
        ResultEnvelope(
            run_id="r1", provenance=prov, validation_status="bad", success=True
        )  # type: ignore[arg-type]


def test_api_models_sim_run_re_export_parity() -> None:
    """src.api.models.sim_run must re-export identical classes to src.sim_run.models."""
    assert ApiRunRequest is RunRequest and ApiRunRecord is RunRecord
    assert ApiResultEnvelope is ResultEnvelope and ApiProvenance is Provenance
