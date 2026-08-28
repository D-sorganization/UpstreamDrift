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
import subprocess
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.articulated_forward_integration import (
    native_dynamics_operator,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_evaluator import (
    require_native_engine,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    NativeEngineUnavailable,
    plan_sha256,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

_SCHEMA = "articulated-structural-factorial-runtime-audit/1.2.0"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_DISTRIBUTIONS = ("numpy", "scipy", "mujoco", "pin", "pinocchio")
EngineProbe = Callable[[str], Mapping[str, str]]
OperatorProbe = Callable[[str], Mapping[str, object]]


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


def _native_operator_smoke(engine: str) -> dict[str, object]:
    model, _ = build_subject_scaled_model(default_synthetic_profiles()[0])
    operator = native_dynamics_operator(engine, model)
    q = np.zeros(model.nq, dtype=np.float64)
    qd = np.zeros(model.nq, dtype=np.float64)
    matrix, bias = operator(q, qd)
    matrix = np.asarray(matrix, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(matrix)))
    passes = bool(
        matrix.shape == (model.nq, model.nq)
        and bias.shape == (model.nq,)
        and np.all(np.isfinite(matrix))
        and np.all(np.isfinite(bias))
        and symmetry_error <= 1.0e-10
        and minimum_eigenvalue > 1.0e-12
    )
    return {
        "model_nq": model.nq,
        "mass_matrix_shape": list(matrix.shape),
        "bias_shape": list(bias.shape),
        "maximum_symmetry_error": symmetry_error,
        "minimum_mass_matrix_eigenvalue": minimum_eigenvalue,
        "passes": passes,
    }


def audit_structural_runtime(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    source_checkout: Mapping[str, object],
    engine_probe: EngineProbe | None = None,
    operator_probe: OperatorProbe | None = None,
) -> dict[str, object]:
    """Return an identity-bound runtime audit without evaluating outcomes."""

    expected_plan_hash = plan_sha256(plan)
    if launch.get("plan_sha256") != expected_plan_hash:
        raise ValueError("launch plan identity does not match the supplied plan")
    revision = launch.get("execution_revision")
    if not isinstance(revision, str) or _SHA40.fullmatch(revision) is None:
        raise ValueError("launch execution_revision must be a lowercase SHA-1")
    observed_revision = source_checkout.get("revision")
    tree_sha = source_checkout.get("tree_sha")
    tracked_clean = source_checkout.get("tracked_clean")
    if (
        not isinstance(observed_revision, str)
        or _SHA40.fullmatch(observed_revision) is None
        or not isinstance(tree_sha, str)
        or _SHA40.fullmatch(tree_sha) is None
        or not isinstance(tracked_clean, bool)
    ):
        raise ValueError(
            "source checkout must declare revision, tree SHA, and clean state"
        )
    source_matches = observed_revision == revision and tracked_clean
    probe = engine_probe or require_native_engine
    smoke_probe = operator_probe or _native_operator_smoke
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
            try:
                smoke = dict(smoke_probe(name))
            except (
                ImportError,
                RuntimeError,
                ValueError,
                TypeError,
                AttributeError,
            ) as exc:
                engines[name] = {
                    "status": "incompatible",
                    "identity": identity,
                    "failure": {
                        "code": "native_operator_smoke_error",
                        "detail": str(exc),
                    },
                }
            else:
                if smoke.get("passes") is not True:
                    engines[name] = {
                        "status": "incompatible",
                        "identity": identity,
                        "operator_smoke": smoke,
                        "failure": {"code": "native_operator_smoke_failed"},
                    }
                else:
                    engines[name] = {
                        "status": "qualified",
                        "identity": identity,
                        "operator_smoke": smoke,
                    }
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
    source_record: dict[str, object] = {
        "revision": observed_revision,
        "tree_sha": tree_sha,
        "tracked_clean": tracked_clean,
        "matches_launch_revision": source_matches,
    }
    digest_payload: dict[str, object] = {
        "identity": identity_record,
        "platform": platform_record,
        "distributions": distributions,
        "engines": engines,
        "source_checkout": source_record,
    }
    return {
        "schema_version": _SCHEMA,
        "classification": "runtime_qualification_not_scientific_evidence",
        "identity": identity_record,
        "platform": platform_record,
        "distributions": distributions,
        "engines": engines,
        "source_checkout": source_record,
        "runtime_identity_sha256": _canonical_sha256(digest_payload),
        "qualified_for_registered_engines": qualified,
        "qualified_for_execution": qualified and source_matches,
    }


def _source_checkout(root: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "tree_sha": git("rev-parse", "HEAD^{tree}"),
        "tracked_clean": not bool(git("status", "--porcelain", "--untracked-files=no")),
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
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    plan = _mapping(json.loads(args.plan.read_text(encoding="utf-8")), name="plan")
    launch = _mapping(
        json.loads(args.launch.read_text(encoding="utf-8")), name="launch"
    )
    result = audit_structural_runtime(
        plan=plan,
        launch=launch,
        source_checkout=_source_checkout(args.source_root.resolve()),
    )
    _write_atomic(args.output, result)
    return 0 if result["qualified_for_execution"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["audit_structural_runtime"]
