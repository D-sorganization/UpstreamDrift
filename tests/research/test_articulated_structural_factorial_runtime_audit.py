"""Runtime qualification is explicit, identity-bound, and fail-closed."""

from __future__ import annotations

import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    NativeEngineUnavailable,
    plan_sha256,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runtime_audit import (
    audit_structural_runtime,
)

pytestmark = pytest.mark.scientific


def _plan() -> dict[str, object]:
    return {"design": {"engines": ["mujoco", "pinocchio"]}}


def _launch(plan: dict[str, object]) -> dict[str, object]:
    return {"plan_sha256": plan_sha256(plan), "execution_revision": "a" * 40}


def test_runtime_audit_qualifies_every_registered_native_engine() -> None:
    plan = _plan()

    def probe(name: str) -> dict[str, str]:
        return {"name": name, "version": "3.0.0", "operator": "native"}

    result = audit_structural_runtime(
        plan=plan, launch=_launch(plan), engine_probe=probe
    )

    assert result["qualified_for_registered_engines"] is True
    assert result["classification"] == "runtime_qualification_not_scientific_evidence"
    assert result["engines"] == {
        "mujoco": {
            "status": "qualified",
            "identity": {
                "name": "mujoco",
                "version": "3.0.0",
                "operator": "native",
            },
        },
        "pinocchio": {
            "status": "qualified",
            "identity": {
                "name": "pinocchio",
                "version": "3.0.0",
                "operator": "native",
            },
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
        plan=plan, launch=_launch(plan), engine_probe=probe
    )

    assert result["qualified_for_registered_engines"] is False
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
        audit_structural_runtime(plan=plan, launch=launch)
