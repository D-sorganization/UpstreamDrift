"""Compare a qualified structural runtime with the actual campaign runner."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any

from scripts.research.proximal_distal_energy.articulated_structural_factorial_runtime_audit import (
    validate_runtime_audit,
)

AUDIT_SCHEMA = "articulated-structural-factorial-runtime-replay-audit/1.0.0"
DETERMINISTIC_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_STABLE_RUNTIME_SECTIONS = (
    "identity",
    "platform",
    "distributions",
    "engines",
    "source_checkout",
    "execution_modules",
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _canonical_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _host_record() -> dict[str, object]:
    return {
        "logical_cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_compiler": platform.python_compiler(),
    }


def audit_runtime_replay(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    qualified_audit: Mapping[str, object],
    observed_audit: Mapping[str, object],
    environment: Mapping[str, str | None],
    host: Mapping[str, object],
) -> dict[str, object]:
    """Return an outcome-blind exact replay gate for the actual runner."""

    qualified_identity = validate_runtime_audit(
        plan=plan, launch=launch, audit=qualified_audit
    )
    observed_identity = validate_runtime_audit(
        plan=plan, launch=launch, audit=observed_audit
    )
    mismatched_sections = [
        name
        for name in _STABLE_RUNTIME_SECTIONS
        if qualified_audit.get(name) != observed_audit.get(name)
    ]
    mismatched_environment = [
        name
        for name, expected in DETERMINISTIC_ENVIRONMENT.items()
        if environment.get(name) != expected
    ]
    stable_exact = not mismatched_sections
    environment_exact = not mismatched_environment
    passes = stable_exact and environment_exact
    identity_payload: dict[str, object] = {
        "plan_sha256": launch.get("plan_sha256"),
        "execution_revision": launch.get("execution_revision"),
        "qualified_runtime_identity_sha256": qualified_identity,
        "observed_runtime_identity_sha256": observed_identity,
        "qualified_audit_source_revision": _mapping(
            qualified_audit.get("audit_source_checkout"),
            name="qualified_audit.audit_source_checkout",
        ).get("revision"),
        "observed_audit_source_revision": _mapping(
            observed_audit.get("audit_source_checkout"),
            name="observed_audit.audit_source_checkout",
        ).get("revision"),
    }
    audit_identity_payload = {
        "identity": identity_payload,
        "mismatched_runtime_sections": mismatched_sections,
        "mismatched_environment_names": mismatched_environment,
        "host": dict(host),
    }
    return {
        "schema_version": AUDIT_SCHEMA,
        "classification": (
            "runtime_replay_contract_exact"
            if passes
            else "runtime_replay_contract_drift"
        ),
        "identity": {
            **identity_payload,
            "audit_identity_sha256": _canonical_sha256(audit_identity_payload),
        },
        "host": dict(host),
        "required_environment": dict(DETERMINISTIC_ENVIRONMENT),
        "mismatched_runtime_sections": mismatched_sections,
        "mismatched_environment_names": mismatched_environment,
        "gates": {
            "qualified_audit_valid": True,
            "observed_audit_valid": True,
            "stable_runtime_contract_exact": stable_exact,
            "deterministic_environment_exact": environment_exact,
            "passes": passes,
        },
        "claim_boundary": {
            "campaign_result_authority": False,
            "scientific_outcomes_inspected": False,
            "tolerance_relaxation_authorized": False,
            "human_or_coaching_inference": False,
        },
    }


def _read_mapping(path: Path) -> Mapping[str, object]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name=str(path))


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the actual-run runtime replay gate atomically."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--qualified-audit", type=Path, required=True)
    parser.add_argument("--observed-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_runtime_replay(
        plan=_read_mapping(args.plan),
        launch=_read_mapping(args.launch),
        qualified_audit=_read_mapping(args.qualified_audit),
        observed_audit=_read_mapping(args.observed_audit),
        environment={name: os.environ.get(name) for name in DETERMINISTIC_ENVIRONMENT},
        host=_host_record(),
    )
    _write_atomic(args.output, result)
    gates = _mapping(result.get("gates"), name="gates")
    return 0 if gates.get("passes") is True else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUDIT_SCHEMA",
    "DETERMINISTIC_ENVIRONMENT",
    "audit_runtime_replay",
    "main",
]
