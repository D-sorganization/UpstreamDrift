"""Simulation result validation and numerical integrity gate (Epic #14085 / #14089).

Guarantees that simulation outputs satisfy structural schema requirements,
provenance completeness, and numerical validity (detecting NaN, +Inf, -Inf)
before any run can transition to the 'succeeded' state.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import ValidationError

from src.sim_run.models import Provenance, ResultEnvelope

_HEX_CHARS = frozenset("0123456789abcdefABCDEF")


def validate_provenance(prov: Provenance | dict[str, Any]) -> list[str]:
    """Validate that execution provenance contains complete, non-stub identity metadata."""
    if isinstance(prov, dict):
        try:
            provenance = Provenance.model_validate(prov)
        except ValidationError as exc:
            return [
                f"Provenance schema error: {e['loc']} {e['msg']}" for e in exc.errors()
            ]
    else:
        provenance = prov

    errors: list[str] = []
    sha = (provenance.repo_sha or "").strip()
    if not sha or sha.lower() == "unknown":
        errors.append("Invalid provenance: repo_sha is empty or unknown")

    in_hash = (provenance.input_sha256 or "").strip()
    if len(in_hash) != 64 or not all(c in _HEX_CHARS for c in in_hash):
        errors.append(
            f"Invalid provenance: input_sha256 must be a 64-character hexadecimal digest, got '{in_hash}'"
        )

    if not (provenance.engine_type or "").strip():
        errors.append("Invalid provenance: engine_type cannot be empty")

    return errors


def validate_numerical_integrity(data: Any, path: str = "data") -> list[str]:
    """Recursively scan structured simulation payloads for NaN or Inf floating point values."""
    errors: list[str] = []
    if data is None:
        errors.append(f"NaN/null value detected at {path}")
    elif isinstance(data, float):
        if math.isnan(data):
            errors.append(f"NaN value detected at {path}")
        elif math.isinf(data):
            sign = "+" if data > 0 else "-"
            errors.append(f"{sign}Inf value detected at {path}")
    elif isinstance(data, str) and data.strip().lower() in (
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
    ):
        errors.append(f"NaN/Inf value detected at {path}")
    elif isinstance(data, dict):
        for key, val in data.items():
            errors.extend(validate_numerical_integrity(val, path=f"{path}.{key}"))
    elif isinstance(data, (list, tuple)):
        for idx, item in enumerate(data):
            errors.extend(validate_numerical_integrity(item, path=f"{path}[{idx}]"))
    return errors


def validate_result(
    result: ResultEnvelope | dict[str, Any],
) -> tuple[bool, ResultEnvelope | None, list[str]]:
    """Validate result envelope structure, provenance, and numerical integrity.

    Returns:
        tuple of (is_valid, validated_envelope, error_diagnostics)
    """
    if isinstance(result, dict):
        try:
            envelope = ResultEnvelope.model_validate(result)
        except ValidationError as exc:
            errs = [
                f"ResultEnvelope schema invalid: {e['loc']} - {e['msg']}"
                for e in exc.errors()
            ]
            return False, None, errs
    else:
        envelope = result

    errors: list[str] = []
    errors.extend(validate_provenance(envelope.provenance))
    errors.extend(validate_numerical_integrity(envelope.data, path="data"))
    if envelope.analysis_results:
        errors.extend(
            validate_numerical_integrity(
                envelope.analysis_results, path="analysis_results"
            )
        )

    if errors:
        envelope.validation_status = "invalid"
        envelope.validation_errors = errors
        return False, envelope, errors

    envelope.validation_status = "valid"
    envelope.validation_errors = []
    return True, envelope, []
