"""Run-local structural runtime replay gates are exact and fail closed."""

from __future__ import annotations

import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    plan_sha256,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runtime_audit import (
    _audit_digest_payload,
    _canonical_sha256,
    _REQUIRED_EXECUTION_MODULE_NAMES,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runtime_replay_audit import (
    AUDIT_SCHEMA,
    DETERMINISTIC_ENVIRONMENT,
    audit_runtime_replay,
)

pytestmark = pytest.mark.scientific


def _plan() -> dict[str, object]:
    return {"design": {"engines": ["mujoco"]}}


def _launch(plan: dict[str, object]) -> dict[str, object]:
    return {"plan_sha256": plan_sha256(plan), "execution_revision": "a" * 40}


def _runtime_audit(plan: dict[str, object]) -> dict[str, object]:
    audit: dict[str, object] = {
        "schema_version": "articulated-structural-factorial-runtime-audit/1.4.0",
        "classification": "runtime_qualification_not_scientific_evidence",
        "identity": {
            "plan_sha256": plan_sha256(plan),
            "execution_revision": "a" * 40,
        },
        "platform": {
            "python_implementation": "CPython",
            "python_version": "3.11.16",
            "system": "Linux",
            "release": "6.17.0",
            "machine": "x86_64",
        },
        "distributions": {
            "numpy": "2.4.6",
            "scipy": "1.17.1",
            "mujoco": "3.12.0",
            "pin": "4.1.0",
            "pinocchio": None,
        },
        "engines": {
            "mujoco": {
                "status": "qualified",
                "identity": {
                    "name": "mujoco",
                    "version": "3.12.0",
                    "operator": "native",
                },
                "operator_smoke": {
                    "model_nq": 20,
                    "mass_matrix_shape": [20, 20],
                    "bias_shape": [20],
                    "maximum_symmetry_error": 0.0,
                    "minimum_mass_matrix_eigenvalue": 1.0,
                    "passes": True,
                },
            }
        },
        "source_checkout": {
            "revision": "a" * 40,
            "tree_sha": "b" * 40,
            "tracked_clean": True,
            "matches_launch_revision": True,
        },
        "audit_source_checkout": {
            "revision": "c" * 40,
            "tree_sha": "d" * 40,
            "tracked_clean": True,
        },
        "execution_modules": {
            name: {
                "path": "scripts/research/operator.py",
                "sha256": "e" * 64,
            }
            for name in _REQUIRED_EXECUTION_MODULE_NAMES
        },
        "qualified_for_registered_engines": True,
        "qualified_for_execution": True,
    }
    audit["runtime_identity_sha256"] = _canonical_sha256(_audit_digest_payload(audit))
    return audit


def _environment() -> dict[str, str]:
    return dict(DETERMINISTIC_ENVIRONMENT)


def test_runtime_replay_accepts_only_exact_runtime_and_environment() -> None:
    plan = _plan()
    audit = _runtime_audit(plan)

    result = audit_runtime_replay(
        plan=plan,
        launch=_launch(plan),
        qualified_audit=audit,
        observed_audit=dict(audit),
        environment=_environment(),
        host={"processor": "fixture", "logical_cpu_count": 2},
    )

    assert result["schema_version"] == AUDIT_SCHEMA
    assert result["classification"] == "runtime_replay_contract_exact"
    assert result["gates"]["passes"] is True
    assert result["gates"]["stable_runtime_contract_exact"] is True
    assert result["gates"]["deterministic_environment_exact"] is True
    assert result["mismatched_runtime_sections"] == []
    assert result["claim_boundary"]["campaign_result_authority"] is False


def test_runtime_replay_rejects_distribution_drift_without_reporting_values() -> None:
    plan = _plan()
    qualified = _runtime_audit(plan)
    observed = _runtime_audit(plan)
    observed["distributions"] = {**observed["distributions"], "numpy": "2.5.0"}
    observed["runtime_identity_sha256"] = _canonical_sha256(
        _audit_digest_payload(observed)
    )

    result = audit_runtime_replay(
        plan=plan,
        launch=_launch(plan),
        qualified_audit=qualified,
        observed_audit=observed,
        environment=_environment(),
        host={},
    )

    assert result["classification"] == "runtime_replay_contract_drift"
    assert result["gates"]["passes"] is False
    assert result["mismatched_runtime_sections"] == ["distributions"]
    assert "2.5.0" not in str(result)


def test_runtime_replay_rejects_unspecified_thread_environment() -> None:
    plan = _plan()
    audit = _runtime_audit(plan)
    environment = _environment()
    environment["OPENBLAS_NUM_THREADS"] = "32"

    result = audit_runtime_replay(
        plan=plan,
        launch=_launch(plan),
        qualified_audit=audit,
        observed_audit=dict(audit),
        environment=environment,
        host={},
    )

    assert result["classification"] == "runtime_replay_contract_drift"
    assert result["gates"]["passes"] is False
    assert result["gates"]["deterministic_environment_exact"] is False
    assert result["mismatched_environment_names"] == ["OPENBLAS_NUM_THREADS"]


def test_runtime_replay_does_not_require_audit_tool_revision_equality() -> None:
    plan = _plan()
    qualified = _runtime_audit(plan)
    observed = _runtime_audit(plan)
    observed["audit_source_checkout"] = {
        "revision": "f" * 40,
        "tree_sha": "0" * 40,
        "tracked_clean": True,
    }
    observed["runtime_identity_sha256"] = _canonical_sha256(
        _audit_digest_payload(observed)
    )

    result = audit_runtime_replay(
        plan=plan,
        launch=_launch(plan),
        qualified_audit=qualified,
        observed_audit=observed,
        environment=_environment(),
        host={},
    )

    assert result["gates"]["passes"] is True
    assert result["identity"]["qualified_audit_source_revision"] == "c" * 40
    assert result["identity"]["observed_audit_source_revision"] == "f" * 40
