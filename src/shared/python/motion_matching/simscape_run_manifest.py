"""Simscape R2025b run-manifest schema for MS-60 (#10347).

Fail-closed validation: only the qualified R2025b release is accepted.
Native qualification receipts must name host, release, model/candidate SHAs,
and wall-clock; unlicensed default CI must not pretend a native pass.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.shared.python.contracts import postcondition, precondition

RUN_MANIFEST_SCHEMA_VERSION = "simscape-run-manifest/1"
REQUIRED_MATLAB_RELEASE = "2025b"

_REQUIRED_KEYS = (
    "schema_version",
    "issue",
    "run_id",
    "matlab_release",
    "matlab_version",
    "host",
    "model_sha256",
    "candidate_sha256",
    "replay_npz_sha256",
    "wall_clock_s",
    "qualification",
    "evidence_dir",
    "artifacts",
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value.lower())
    )


@precondition(
    lambda **kwargs: bool(str(kwargs.get("run_id", "")).strip()),
    "run_id must be non-empty",
)
@precondition(
    lambda **kwargs: bool(str(kwargs.get("host", "")).strip()),
    "host must be non-empty",
)
@precondition(
    lambda **kwargs: float(kwargs.get("wall_clock_s", -1.0)) >= 0.0,
    "wall_clock_s must be non-negative",
)
@postcondition(
    lambda r: r.get("schema_version") == RUN_MANIFEST_SCHEMA_VERSION,
    "manifest schema_version must match",
)
def build_simscape_run_manifest(
    *,
    run_id: str,
    matlab_release: str,
    matlab_version: str,
    host: str,
    model_sha256: str,
    candidate_sha256: str,
    replay_npz_sha256: str,
    wall_clock_s: float,
    qualification: str,
    evidence_dir: str,
    artifacts: Mapping[str, str],
    issue: str = "#10347",
    machine: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned Simscape run manifest payload."""
    payload: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "issue": issue,
        "run_id": run_id,
        "matlab_release": matlab_release,
        "matlab_version": matlab_version,
        "host": host,
        "model_sha256": model_sha256,
        "candidate_sha256": candidate_sha256,
        "replay_npz_sha256": replay_npz_sha256,
        "wall_clock_s": float(wall_clock_s),
        "qualification": qualification,
        "evidence_dir": evidence_dir,
        "artifacts": dict(artifacts),
    }
    if machine:
        payload["machine"] = machine
    if extra:
        payload["extra"] = dict(extra)
    validate_simscape_run_manifest(payload)
    return payload


@precondition(
    lambda manifest: isinstance(manifest, Mapping),
    "manifest must be a mapping",
)
def validate_simscape_run_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed if the manifest is incomplete or not R2025b-qualified."""
    missing = [key for key in _REQUIRED_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"Simscape run manifest missing keys: {missing}")

    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported run manifest schema: {manifest.get('schema_version')!r}"
        )

    release = str(manifest.get("matlab_release", "")).strip().lower().lstrip("r")
    if release != REQUIRED_MATLAB_RELEASE:
        raise ValueError(
            "Simscape run manifest requires MATLAB R2025b; "
            f"got {manifest.get('matlab_release')!r} (no R2026a substitution)"
        )

    for sha_key in ("model_sha256", "candidate_sha256", "replay_npz_sha256"):
        if not _is_sha256(manifest.get(sha_key)):
            raise ValueError(f"{sha_key} must be a 64-char hex SHA-256 digest")

    if not str(manifest.get("host", "")).strip():
        raise ValueError("host must name the licensed execution machine")

    if float(manifest["wall_clock_s"]) < 0.0:
        raise ValueError("wall_clock_s must be non-negative")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("artifacts must be a non-empty mapping of relative paths")
