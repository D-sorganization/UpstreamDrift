"""Run manifest: the provenance that identifies one simulation run (Issue #14431 / Epic #14436).

A manifest pins everything needed to reproduce a run: the model and its
immutable revision, the parameters, the identity (never the content) of the
input data, the seed, the engine and its version, the UpstreamDrift commit
and the creation time. References must be immutable: a branch, tag or
abbreviated SHA is rejected because it can point at different content later.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

RUN_MANIFEST_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

_GIT_SHA = r"[0-9a-f]{40}"
_SHA256 = r"[0-9a-f]{64}"
GIT_SHA_PATTERN = rf"^{_GIT_SHA}$"
REVISION_PATTERN = rf"^(?:{_GIT_SHA}|sha256:{_SHA256})$"
SHA256_PATTERN = rf"^{_SHA256}$"
_NON_BLANK = r"\S"


class RunManifestError(ValueError):
    """Raised when a run manifest is missing provenance or carries a mutable ref."""


def _require_immutable(value: str, pattern: str, field: str, expected: str) -> str:
    if not re.fullmatch(pattern, value):
        raise ValueError(
            f"{field} is a mutable ref {value!r}; expected {expected} "
            "(branch names, tags and abbreviated SHAs can move)"
        )
    return value


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class ModelIdentity(_Strict):
    """The model a run used, pinned to an immutable revision."""

    model_id: str = Field(..., pattern=_NON_BLANK, description="Model identifier")
    revision: str = Field(
        ...,
        pattern=REVISION_PATTERN,
        description="40-char git commit SHA or 'sha256:<64 hex>' model-file hash",
    )

    @field_validator("revision", mode="before")
    @classmethod
    def _immutable_revision(cls, value: Any) -> Any:
        if isinstance(value, str):
            _require_immutable(
                value,
                REVISION_PATTERN,
                "revision",
                "a 40-char git commit SHA or 'sha256:<64 hex>' file hash",
            )
        return value


class InputDataIdentity(_Strict):
    """Identity of the run's input data. Holds a digest, never the data itself."""

    sha256: str = Field(
        ..., pattern=SHA256_PATTERN, description="SHA-256 of the input data"
    )
    label: str | None = Field(None, description="Optional non-sensitive label")


class EngineIdentity(_Strict):
    """Physics engine name and exact version."""

    name: str = Field(..., pattern=_NON_BLANK, description="Engine name")
    version: str = Field(..., pattern=_NON_BLANK, description="Engine version")


class RunManifest(_Strict):
    """Versioned provenance record for one simulation run."""

    schema_version: Literal["1.0.0"] = Field(..., description="Manifest version")
    model: ModelIdentity
    parameters: dict[str, Any] = Field(..., description="Run parameters")
    input_data: InputDataIdentity
    seed: int = Field(..., ge=0, strict=True, description="Random seed")
    engine: EngineIdentity
    ud_commit: str = Field(
        ...,
        pattern=GIT_SHA_PATTERN,
        description="40-char UpstreamDrift git commit SHA",
    )
    created_at: datetime = Field(
        ...,
        description="ISO 8601 creation time with a UTC offset",
        json_schema_extra={"pattern": r"(?:Z|[+-]\d{2}:\d{2})$"},
    )

    @field_validator("ud_commit", mode="before")
    @classmethod
    def _immutable_commit(cls, value: Any) -> Any:
        if isinstance(value, str):
            _require_immutable(
                value, GIT_SHA_PATTERN, "ud_commit", "a 40-char git commit SHA"
            )
        return value

    @field_validator("created_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("created_at must carry a UTC offset (e.g. 'Z')")
        return value


def validate_run_manifest(data: Mapping[str, Any]) -> RunManifest:
    """Validate a manifest mapping and return the parsed, immutable manifest.

    Raises:
        TypeError: if ``data`` is not a mapping.
        RunManifestError: if any provenance field is missing or invalid; the
            message names each offending field.
    """
    if not isinstance(data, Mapping):
        raise TypeError(f"run manifest must be a mapping, got {type(data).__name__}")
    try:
        return RunManifest.model_validate(dict(data))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        raise RunManifestError(f"invalid run manifest: {problems}") from exc


def export_run_manifest_schema() -> dict[str, Any]:
    """Export the versioned JSON Schema checked in at ``schemas/run-manifest-v1.json``."""
    schema = RunManifest.model_json_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://upstreamdrift.dev/schemas/run-manifest-v1.json",
        **schema,
        "description": "Provenance for one simulation run (Issue #14431 / Epic #14436).",
    }
