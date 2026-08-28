"""Runtime qualification is explicit, identity-bound, and fail-closed."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    NativeEngineUnavailable,
    plan_sha256,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runtime_audit import (
    _execution_module_provenance,
    audit_structural_runtime,
    validate_runtime_audit,
)

pytestmark = pytest.mark.scientific


def _plan() -> dict[str, object]:
    return {"design": {"engines": ["mujoco", "pinocchio"]}}


def _launch(plan: dict[str, object]) -> dict[str, object]:
    return {"plan_sha256": plan_sha256(plan), "execution_revision": "a" * 40}


def _source(*, revision: str = "a" * 40, clean: bool = True) -> dict[str, object]:
    return {"revision": revision, "tree_sha": "b" * 40, "tracked_clean": clean}


def _audit_source() -> dict[str, object]:
    return {"revision": "c" * 40, "tree_sha": "d" * 40, "tracked_clean": True}


def _execution_modules() -> dict[str, object]:
    return {
        "native_dynamics_operator": {
            "path": "scripts/research/operator.py",
            "sha256": "e" * 64,
        }
    }


def _smoke(_name: str) -> dict[str, object]:
    return {"model_nq": 4, "passes": True}


def test_runtime_audit_qualifies_every_registered_native_engine() -> None:
    plan = _plan()

    def probe(name: str) -> dict[str, str]:
        return {"name": name, "version": "3.0.0", "operator": "native"}

    result = audit_structural_runtime(
        plan=plan,
        launch=_launch(plan),
        source_checkout=_source(),
        audit_source_checkout=_audit_source(),
        execution_modules=_execution_modules(),
        engine_probe=probe,
        operator_probe=_smoke,
    )

    assert result["qualified_for_registered_engines"] is True
    assert result["qualified_for_execution"] is True
    assert result["source_checkout"] == {
        "revision": "a" * 40,
        "tree_sha": "b" * 40,
        "tracked_clean": True,
        "matches_launch_revision": True,
    }
    assert result["classification"] == "runtime_qualification_not_scientific_evidence"
    assert result["engines"] == {
        "mujoco": {
            "status": "qualified",
            "identity": {
                "name": "mujoco",
                "version": "3.0.0",
                "operator": "native",
            },
            "operator_smoke": {"model_nq": 4, "passes": True},
        },
        "pinocchio": {
            "status": "qualified",
            "identity": {
                "name": "pinocchio",
                "version": "3.0.0",
                "operator": "native",
            },
            "operator_smoke": {"model_nq": 4, "passes": True},
        },
    }
    assert len(str(result["runtime_identity_sha256"])) == 64


def test_runtime_audit_retains_typed_unavailability_without_qualification() -> None:
    plan = _plan()

    def probe(name: str) -> dict[str, str]:
        if name == "pinocchio":
            raise NativeEngineUnavailable(
                engine=name, detail="robotics package is unavailable"
            )
        return {"name": name, "version": "3.3.4", "operator": "native"}

    result = audit_structural_runtime(
        plan=plan,
        launch=_launch(plan),
        source_checkout=_source(),
        audit_source_checkout=_audit_source(),
        execution_modules=_execution_modules(),
        engine_probe=probe,
        operator_probe=_smoke,
    )

    assert result["qualified_for_registered_engines"] is False
    assert result["qualified_for_execution"] is False
    assert result["engines"]["pinocchio"] == {  # type: ignore[index]
        "status": "unavailable",
        "failure": {
            "code": "native_engine_unavailable",
            "detail": "robotics package is unavailable",
        },
    }


def test_runtime_audit_rejects_launch_identity_drift() -> None:
    plan = _plan()
    launch = _launch(plan)
    launch["plan_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="launch plan identity"):
        audit_structural_runtime(
            plan=plan,
            launch=launch,
            source_checkout=_source(),
            audit_source_checkout=_audit_source(),
            execution_modules=_execution_modules(),
            operator_probe=_smoke,
        )


@pytest.mark.parametrize(
    ("source", "matches"),
    [
        (_source(revision="c" * 40), False),
        (_source(clean=False), False),
    ],
)
def test_runtime_audit_rejects_source_checkout_drift(
    source: dict[str, object], matches: bool
) -> None:
    plan = _plan()

    def probe(name: str) -> dict[str, str]:
        return {"name": name, "version": "3.0.0", "operator": "native"}

    result = audit_structural_runtime(
        plan=plan,
        launch=_launch(plan),
        source_checkout=source,
        audit_source_checkout=_audit_source(),
        execution_modules=_execution_modules(),
        engine_probe=probe,
        operator_probe=_smoke,
    )

    assert result["source_checkout"]["matches_launch_revision"] is matches  # type: ignore[index]
    assert result["qualified_for_registered_engines"] is True
    assert result["qualified_for_execution"] is False


def test_runtime_audit_rejects_native_operator_smoke_failure() -> None:
    plan = _plan()

    def probe(name: str) -> dict[str, str]:
        return {"name": name, "version": "3.0.0", "operator": "native"}

    def smoke(name: str) -> dict[str, object]:
        return {"engine": name, "passes": name == "mujoco"}

    result = audit_structural_runtime(
        plan=plan,
        launch=_launch(plan),
        source_checkout=_source(),
        audit_source_checkout=_audit_source(),
        execution_modules=_execution_modules(),
        engine_probe=probe,
        operator_probe=smoke,
    )

    engines = result["engines"]
    assert isinstance(engines, dict)
    assert engines["mujoco"]["status"] == "qualified"
    assert engines["pinocchio"]["status"] == "incompatible"
    assert engines["pinocchio"]["failure"] == {"code": "native_operator_smoke_failed"}
    assert result["qualified_for_registered_engines"] is False
    assert result["qualified_for_execution"] is False


def test_runtime_audit_validator_accepts_only_intact_qualified_identity() -> None:
    plan = _plan()
    launch = _launch(plan)

    def probe(name: str) -> dict[str, str]:
        return {"name": name, "version": "3.0.0", "operator": "native"}

    audit = audit_structural_runtime(
        plan=plan,
        launch=launch,
        source_checkout=_source(),
        audit_source_checkout=_audit_source(),
        execution_modules=_execution_modules(),
        engine_probe=probe,
        operator_probe=_smoke,
    )

    assert (
        validate_runtime_audit(plan=plan, launch=launch, audit=audit)
        == audit["runtime_identity_sha256"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"qualified_for_execution": False}, "not qualified"),
        ({"runtime_identity_sha256": "0" * 64}, "digest"),
        ({"identity": {"plan_sha256": "0" * 64}}, "identity"),
    ],
)
def test_runtime_audit_validator_fails_closed_on_tampering(
    mutation: dict[str, object], message: str
) -> None:
    plan = _plan()
    launch = _launch(plan)

    def probe(name: str) -> dict[str, str]:
        return {"name": name, "version": "3.0.0", "operator": "native"}

    audit = audit_structural_runtime(
        plan=plan,
        launch=launch,
        source_checkout=_source(),
        audit_source_checkout=_audit_source(),
        execution_modules=_execution_modules(),
        engine_probe=probe,
        operator_probe=_smoke,
    )
    audit.update(mutation)

    with pytest.raises(ValueError, match=message):
        validate_runtime_audit(plan=plan, launch=launch, audit=audit)


def test_execution_module_provenance_rejects_a_different_source_root(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    provenance = _execution_module_provenance(repo_root)

    assert "native_dynamics_operator" in provenance
    assert all(len(str(row["sha256"])) == 64 for row in provenance.values())
    with pytest.raises(ValueError, match="outside the declared execution source"):
        _execution_module_provenance(tmp_path)
