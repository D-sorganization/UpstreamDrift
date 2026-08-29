"""Bind structural-factorial execution to a qualified native-engine runtime."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
from importlib import metadata
import inspect
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
    evaluate_structural_case,
    require_native_engine,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_evidence import (
    validate_structural_evidence_arrays,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    NativeEngineUnavailable,
    plan_sha256,
    run_serial_cases,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

_SCHEMA = "articulated-structural-factorial-runtime-audit/1.4.0"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DISTRIBUTIONS = ("numpy", "scipy", "mujoco", "pin", "pinocchio")
ROOT = Path(__file__).resolve().parents[3]
EngineProbe = Callable[[str], Mapping[str, str]]
OperatorProbe = Callable[[str], Mapping[str, object]]
_REQUIRED_EXECUTION_MODULE_NAMES = (
    "native_dynamics_operator",
    "require_native_engine",
    "evaluate_structural_case",
    "run_serial_cases",
    "validate_structural_evidence_arrays",
    "build_subject_scaled_model",
    "default_synthetic_profiles",
)


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


def _audit_digest_payload(audit: Mapping[str, object]) -> dict[str, object]:
    return {
        name: audit[name]
        for name in (
            "identity",
            "platform",
            "distributions",
            "engines",
            "source_checkout",
            "audit_source_checkout",
            "execution_modules",
        )
    }


def _execution_module_provenance(source_root: Path) -> dict[str, object]:
    """Hash executed operator modules and require them to live under source_root."""

    root = source_root.resolve()
    functions = {
        "native_dynamics_operator": native_dynamics_operator,
        "require_native_engine": require_native_engine,
        "evaluate_structural_case": evaluate_structural_case,
        "run_serial_cases": run_serial_cases,
        "validate_structural_evidence_arrays": validate_structural_evidence_arrays,
        "build_subject_scaled_model": build_subject_scaled_model,
        "default_synthetic_profiles": default_synthetic_profiles,
    }
    provenance: dict[str, object] = {}
    for name, function in functions.items():
        raw_path = inspect.getsourcefile(function)
        if raw_path is None:
            raise ValueError(f"cannot resolve executed source for {name}")
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"executed module {name} is outside the declared execution source"
            ) from exc
        provenance[name] = {
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return provenance


def _validate_provenance_mapping(value: Mapping[str, object], *, name: str) -> None:
    if not value:
        raise ValueError(f"{name} must retain at least one executed module")
    for module_name, raw in value.items():
        row = _mapping(raw, name=f"{name}.{module_name}")
        path = row.get("path")
        digest = row.get("sha256")
        if (
            not isinstance(module_name, str)
            or not module_name
            or not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError(f"{name} contains invalid module provenance")


def _validate_execution_module_provenance(value: Mapping[str, object]) -> None:
    _validate_provenance_mapping(value, name="execution_modules")
    if set(value) != set(_REQUIRED_EXECUTION_MODULE_NAMES):
        raise ValueError(
            "execution_modules must retain exactly the required executed modules"
        )


def validate_runtime_audit(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    audit: Mapping[str, object],
) -> str:
    """Return the digest only for an intact audit qualified for this launch."""

    if audit.get("schema_version") != _SCHEMA:
        raise ValueError("runtime audit schema is invalid")
    if audit.get("classification") != "runtime_qualification_not_scientific_evidence":
        raise ValueError("runtime audit classification is invalid")
    expected_identity = {
        "plan_sha256": plan_sha256(plan),
        "execution_revision": launch.get("execution_revision"),
    }
    if audit.get("identity") != expected_identity:
        raise ValueError("runtime audit identity does not match the launch")
    source = _mapping(audit.get("source_checkout"), name="source_checkout")
    if (
        source.get("revision") != launch.get("execution_revision")
        or source.get("tracked_clean") is not True
        or source.get("matches_launch_revision") is not True
    ):
        raise ValueError("runtime audit source checkout is not launch-qualified")
    audit_source = _mapping(
        audit.get("audit_source_checkout"), name="audit_source_checkout"
    )
    if (
        not isinstance(audit_source.get("revision"), str)
        or _SHA40.fullmatch(str(audit_source.get("revision"))) is None
        or not isinstance(audit_source.get("tree_sha"), str)
        or _SHA40.fullmatch(str(audit_source.get("tree_sha"))) is None
        or audit_source.get("tracked_clean") is not True
    ):
        raise ValueError("runtime audit tool source is not clean and revision-bound")
    execution_modules = _mapping(
        audit.get("execution_modules"), name="execution_modules"
    )
    _validate_execution_module_provenance(execution_modules)
    engines = _mapping(audit.get("engines"), name="engines")
    registered = _registered_engines(plan)
    if set(engines) != set(registered) or any(
        _mapping(engines[name], name=f"engines.{name}").get("status") != "qualified"
        for name in registered
    ):
        raise ValueError("runtime audit does not qualify every registered engine")
    if audit.get("qualified_for_registered_engines") is not True:
        raise ValueError("runtime audit is not qualified for registered engines")
    if audit.get("qualified_for_execution") is not True:
        raise ValueError("runtime audit is not qualified for execution")
    digest = audit.get("runtime_identity_sha256")
    if not isinstance(digest, str) or digest != _canonical_sha256(
        _audit_digest_payload(audit)
    ):
        raise ValueError("runtime audit digest is invalid")
    return digest


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
    audit_source_checkout: Mapping[str, object],
    execution_modules: Mapping[str, object],
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
    audit_revision = audit_source_checkout.get("revision")
    audit_tree = audit_source_checkout.get("tree_sha")
    audit_clean = audit_source_checkout.get("tracked_clean")
    if (
        not isinstance(audit_revision, str)
        or _SHA40.fullmatch(audit_revision) is None
        or not isinstance(audit_tree, str)
        or _SHA40.fullmatch(audit_tree) is None
        or audit_clean is not True
    ):
        raise ValueError("audit source checkout must be clean and revision-bound")
    _validate_execution_module_provenance(execution_modules)
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
        "audit_source_checkout": dict(audit_source_checkout),
        "execution_modules": dict(execution_modules),
    }
    return {
        "schema_version": _SCHEMA,
        "classification": "runtime_qualification_not_scientific_evidence",
        "identity": identity_record,
        "platform": platform_record,
        "distributions": distributions,
        "engines": engines,
        "source_checkout": source_record,
        "audit_source_checkout": dict(audit_source_checkout),
        "execution_modules": dict(execution_modules),
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
    parser.add_argument("--audit-source-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    plan = _mapping(json.loads(args.plan.read_text(encoding="utf-8")), name="plan")
    launch = _mapping(
        json.loads(args.launch.read_text(encoding="utf-8")), name="launch"
    )
    source_root = args.source_root.resolve()
    result = audit_structural_runtime(
        plan=plan,
        launch=launch,
        source_checkout=_source_checkout(source_root),
        audit_source_checkout=_source_checkout(args.audit_source_root.resolve()),
        execution_modules=_execution_module_provenance(source_root),
    )
    _write_atomic(args.output, result)
    return 0 if result["qualified_for_execution"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["audit_structural_runtime", "validate_runtime_audit"]
