"""Bind structural-factorial execution to a qualified native-engine runtime."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
from typing import Any

from scripts.research.proximal_distal_energy.articulated_structural_factorial_evaluator import (
    require_native_engine,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    NativeEngineUnavailable,
    plan_sha256,
)

_SCHEMA = "articulated-structural-factorial-runtime-audit/1.0.0"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_DISTRIBUTIONS = ("numpy", "scipy", "mujoco", "pin", "pinocchio")
EngineProbe = Callable[[str], Mapping[str, str]]


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _registered_engines(plan: Mapping[str, object]) -> tuple[str, ...]:
    design = _mapping(plan.get("design"), name="plan.design")
    raw = design.get("engines")
    if not isinstance(raw, list) or not raw:
        raise ValueError("plan.design.engines must be a nonempty list")
    engines = tuple(raw)
    if any(not isinstance(engine, str) or not engine for engine in engines):
        raise ValueError("registered engines must be nonempty strings")
    if len(set(engines)) != len(engines):
        raise ValueError("registered engines must be unique")
    return engines


def _distribution_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_structural_runtime(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    engine_probe: EngineProbe | None = None,
) -> dict[str, object]:
    """Return an identity-bound runtime audit without evaluating outcomes."""

    expected_plan_hash = plan_sha256(plan)
    if launch.get("plan_sha256") != expected_plan_hash:
        raise ValueError("launch plan identity does not match the supplied plan")
    revision = launch.get("execution_revision")
    if not isinstance(revision, str) or _SHA40.fullmatch(revision) is None:
        raise ValueError("launch execution_revision must be a lowercase SHA-1")
    probe = engine_probe or require_native_engine
    engines: dict[str, dict[str, object]] = {}
    for name in _registered_engines(plan):
        try:
            identity = dict(probe(name))
        except NativeEngineUnavailable as exc:
            if exc.engine != name:
                raise ValueError("runtime probe reported the wrong engine") from exc
            engines[name] = {
                "status": "unavailable",
                "failure": {
                    "code": "native_engine_unavailable",
                    "detail": exc.detail,
                },
            }
        else:
            if identity.get("name") != name or identity.get("operator") != "native":
                raise ValueError(
                    "qualified engine identity must name a native operator"
                )
            version = identity.get("version")
            if not isinstance(version, str) or not version:
                raise ValueError("qualified engine identity must declare a version")
            engines[name] = {"status": "qualified", "identity": identity}
    platform_record: dict[str, object] = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }
    distributions: dict[str, object] = _distribution_versions()
    identity_record: dict[str, object] = {
        "plan_sha256": expected_plan_hash,
        "execution_revision": revision,
    }
    qualified = all(row["status"] == "qualified" for row in engines.values())
    digest_payload: dict[str, object] = {
        "identity": identity_record,
        "platform": platform_record,
        "distributions": distributions,
        "engines": engines,
    }
    return {
        "schema_version": _SCHEMA,
        "classification": "runtime_qualification_not_scientific_evidence",
        "identity": identity_record,
        "platform": platform_record,
        "distributions": distributions,
        "engines": engines,
        "runtime_identity_sha256": _canonical_sha256(digest_payload),
        "qualified_for_registered_engines": qualified,
    }


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
    """Audit an explicit plan/launch pair and retain the result atomically."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = _mapping(json.loads(args.plan.read_text(encoding="utf-8")), name="plan")
    launch = _mapping(
        json.loads(args.launch.read_text(encoding="utf-8")), name="launch"
    )
    result = audit_structural_runtime(plan=plan, launch=launch)
    _write_atomic(args.output, result)
    return 0 if result["qualified_for_registered_engines"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["audit_structural_runtime"]
