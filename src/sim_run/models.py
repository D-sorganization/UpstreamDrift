"""Simulation run lifecycle and interchange models (ADR 0001 / Epic #14085 / #14086).

Defines versioned schemas for RunRequest, RunRecord, and ResultEnvelope,
derived directly from existing API models (src.api.models.requests and responses).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.api.models.requests import SimulationRequest
from src.api.models.responses import SimulationErrorInfo

RunStatus = Literal["queued", "running", "succeeded", "failed", "expired"]
ValidationStatus = Literal["valid", "invalid", "unvalidated"]
SCHEMA_VERSION_V1: Literal["1.0.0"] = "1.0.0"


class RunRequest(SimulationRequest):
    """Headless simulation run request specification inheriting engine and bound checks."""

    schema_version: Literal["1.0.0"] = Field(
        SCHEMA_VERSION_V1, description="RunRequest version"
    )
    tags: dict[str, str] = Field(
        default_factory=dict, description="Caller-provided tags"
    )
    timeout_seconds: float = Field(
        300.0, gt=0, le=600.0, description="Worker timeout in seconds"
    )


class Provenance(BaseModel):
    """Reproducibility and identity provenance for a completed simulation run."""

    repo_sha: str = Field(..., description="Git commit SHA of running codebase")
    engine_type: str = Field(..., description="Physics engine identifier")
    engine_version: str | None = Field(None, description="Engine version string")
    input_sha256: str = Field(..., description="SHA-256 digest of input JSON")
    created_at: str = Field(..., description="ISO 8601 submission timestamp")
    completed_at: str | None = Field(None, description="ISO 8601 completion timestamp")
    host: str | None = Field(None, description="Worker host or runner identifier")


class ResultEnvelope(BaseModel):
    """Output envelope holding simulation results, provenance, and validation status."""

    schema_version: Literal["1.0.0"] = Field(
        SCHEMA_VERSION_V1, description="Envelope version"
    )
    run_id: str = Field(..., description="Unique run identifier")
    provenance: Provenance = Field(..., description="Execution provenance metadata")
    validation_status: ValidationStatus = Field(
        "unvalidated", description="Validation status"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Validation errors"
    )
    success: bool = Field(..., description="Whether simulation completed successfully")
    data: dict[str, Any] = Field(
        default_factory=dict, description="Simulation trajectory data"
    )
    analysis_results: dict[str, Any] | None = Field(
        None, description="Analysis results"
    )
    artifacts: list[str] = Field(
        default_factory=list, description="Relative artifact paths"
    )
    error: SimulationErrorInfo | None = Field(
        None, description="Structured error details"
    )


class RunRecord(BaseModel):
    """Durable lifecycle record for a simulation run."""

    schema_version: Literal["1.0.0"] = Field(
        SCHEMA_VERSION_V1, description="RunRecord version"
    )
    run_id: str = Field(..., description="Unique durable run identifier")
    status: RunStatus = Field(..., description="Current lifecycle state")
    request: RunRequest = Field(..., description="Original run request")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    started_at: str | None = Field(None, description="ISO 8601 start timestamp")
    completed_at: str | None = Field(None, description="ISO 8601 completion timestamp")
    expires_at: str | None = Field(
        None, description="ISO 8601 quota expiration timestamp"
    )
    result: ResultEnvelope | None = Field(
        None, description="Result envelope if completed"
    )
    error: SimulationErrorInfo | None = Field(
        None, description="Structured error details"
    )


def export_schemas() -> dict[str, dict[str, Any]]:
    """Export versioned JSON Schemas for individual sim-run models."""
    return {
        "RunRequest": RunRequest.model_json_schema(),
        "RunRecord": RunRecord.model_json_schema(),
        "ResultEnvelope": ResultEnvelope.model_json_schema(),
    }


def export_schema() -> dict[str, Any]:
    """Export the unified versioned JSON Schema for the sim-run subsystem."""
    record_schema = RunRecord.model_json_schema()
    defs = dict(record_schema.pop("$defs"))
    defs["RunRecord"] = record_schema

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://upstreamdrift.dev/schemas/sim-run-v1.json",
        "title": "SimulationRunProtocol_v1",
        "description": "Versioned schemas for headless simulation runs (Epic #14085 / #14086).",
        "type": "object",
        "properties": {
            "request": {"$ref": "#/$defs/RunRequest"},
            "record": {"$ref": "#/$defs/RunRecord"},
            "result": {"$ref": "#/$defs/ResultEnvelope"},
        },
        "$defs": defs,
    }
