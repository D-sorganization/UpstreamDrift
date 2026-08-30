"""Recovery replay registration is deterministic, dependency ordered, and fail closed."""

from __future__ import annotations

import copy
from collections.abc import Callable
import json
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_recovery_registration import (
    RECOVERY_SCHEMA,
    StructuralFactorialRecoveryRegistration,
    validate_recovery_registration,
    validate_registered_slice,
)

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).parents[2]
PLAN_PATH = (
    ROOT
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
    / "articulated_structural_factorial_plan.json"
)
LAUNCH_PATH = PLAN_PATH.with_name("articulated_structural_factorial_launch.json")
REGISTRATION_PATH = PLAN_PATH.with_name(
    "articulated_structural_factorial_recovery_registration.json"
)


def _registration() -> StructuralFactorialRecoveryRegistration:
    return StructuralFactorialRecoveryRegistration(
        qualified_runtime_audit_run_id=33297583257,
        qualified_runtime_identity_sha256=(
            "58b7cc58cffcc20e6736b662bd8f9119d8cd2374d9406d59d5c1dd44ee41dc7d"
        ),
        triggering_run_id=33286379004,
        triggering_dispatch_head="0824aa69321c576aeef5f69eee351cada5d4977c",
        repeat_run_ids=(33289155154, 33290346007, 33290812945),
        attested_repeat_run_ids=(33290346007, 33290812945),
        repeatability_audit_sha256=(
            "3021c3fc5d9dc392d1efa53aede5a0a022cce03503a3788f52cf7b8def96c71e"
        ),
        superseded_registration_sha256=(
            "124ac709ea9045fbf33b72b1929fccba509c15721474873c12841f978ec7453b"
        ),
        runner_conflict_run_id=33297882357,
        runner_conflict_dispatch_head=("693ffcf5b7678a746b82dd941fb2f0faeb5916b5"),
    )


def _inputs() -> tuple[dict[str, object], dict[str, object]]:
    return (
        json.loads(PLAN_PATH.read_text(encoding="utf-8")),
        json.loads(LAUNCH_PATH.read_text(encoding="utf-8")),
    )


def test_recovery_registration_is_canonical_and_dependency_ordered() -> None:
    plan, launch = _inputs()
    manifest = _registration().to_manifest(plan=plan, launch=launch)

    validate_recovery_registration(manifest=manifest, plan=plan, launch=launch)

    slices = cast(list[dict[str, object]], manifest["slices"])
    execution_policy = cast(dict[str, object], manifest["execution_policy"])
    evidence_policy = cast(dict[str, object], manifest["evidence_policy"])
    claim_boundary = cast(dict[str, object], manifest["claim_boundary"])
    assert manifest["schema_version"] == RECOVERY_SCHEMA
    assert [(item["case_start"], item["case_stop"]) for item in slices] == [
        (0, 20),
        (20, 40),
        (40, 60),
        (60, 80),
        (80, 100),
    ]
    assert slices[0]["depends_on_case_stop"] is None
    assert [item["depends_on_case_stop"] for item in slices[1:]] == [
        20,
        40,
        60,
        80,
    ]
    assert execution_policy["maximum_concurrent_structural_runs"] == 1
    assert execution_policy["runner_placement"] == {
        "workflow_runner_label": "ubuntu-latest",
        "runner_environment": "github-hosted",
        "runner_os": "Linux",
        "runner_arch": "X64",
        "image_os_prefix": "ubuntu",
    }
    amendment = cast(dict[str, object], manifest["operational_amendment"])
    assert amendment["timing"] == (
        "after_runner_placement_conflict_before_replacement_execution"
    )
    assert amendment["conflicting_run_promotable"] is False
    assert amendment["registered_design_or_numerical_gate_change"] is False
    assert evidence_policy["reuse_prior_checkpoint_bytes"] is False
    assert claim_boundary["scientific_outcomes_may_be_inspected"] is False


def test_tracked_recovery_registration_matches_canonical_manifest() -> None:
    plan, launch = _inputs()
    tracked = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))

    assert tracked == _registration().to_manifest(plan=plan, launch=launch)
    validate_recovery_registration(manifest=tracked, plan=plan, launch=launch)


def test_registered_slice_gate_accepts_only_exact_preregistered_slice() -> None:
    plan, launch = _inputs()
    manifest = _registration().to_manifest(plan=plan, launch=launch)

    record = validate_registered_slice(
        manifest=manifest,
        plan=plan,
        launch=launch,
        case_start=0,
        case_stop=20,
        runner_environment="github-hosted",
        runner_os="Linux",
        runner_arch="X64",
        runner_image_os="ubuntu24",
        runtime_audit_run_id=33297583257,
    )

    assert record["ordinal"] == 1
    for case_start, case_stop in ((0, 1), (10, 20), (100, 120)):
        with pytest.raises(ValueError, match="not an exact registered slice"):
            validate_registered_slice(
                manifest=manifest,
                plan=plan,
                launch=launch,
                case_start=case_start,
                case_stop=case_stop,
                runner_environment="github-hosted",
                runner_os="Linux",
                runner_arch="X64",
                runner_image_os="ubuntu24",
                runtime_audit_run_id=33297583257,
            )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"runner_environment": "self-hosted"}, "runner environment"),
        ({"runner_os": "Windows"}, "runner OS"),
        ({"runner_arch": "ARM64"}, "runner architecture"),
        ({"runner_image_os": "windows2025"}, "runner image"),
    ],
)
def test_registered_slice_gate_rejects_unregistered_runner_placement(
    changes: dict[str, str], message: str
) -> None:
    plan, launch = _inputs()
    manifest = _registration().to_manifest(plan=plan, launch=launch)
    placement = {
        "runner_environment": "github-hosted",
        "runner_os": "Linux",
        "runner_arch": "X64",
        "runner_image_os": "ubuntu24",
        "runtime_audit_run_id": 33297583257,
    }
    placement.update(changes)

    with pytest.raises(ValueError, match=message):
        validate_registered_slice(
            manifest=manifest,
            plan=plan,
            launch=launch,
            case_start=0,
            case_stop=20,
            **placement,
        )


def test_registered_slice_gate_rejects_unregistered_runtime_audit() -> None:
    plan, launch = _inputs()
    manifest = _registration().to_manifest(plan=plan, launch=launch)

    with pytest.raises(ValueError, match="runtime audit run"):
        validate_registered_slice(
            manifest=manifest,
            plan=plan,
            launch=launch,
            case_start=0,
            case_stop=20,
            runner_environment="github-hosted",
            runner_os="Linux",
            runner_arch="X64",
            runner_image_os="ubuntu24",
            runtime_audit_run_id=33277601263,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["slices"][1].update(case_start=21), "gap-free"),
        (
            lambda value: value["evidence_policy"].update(
                reuse_prior_checkpoint_bytes=True
            ),
            "reuse prior checkpoint",
        ),
        (
            lambda value: value["stop_conditions"].remove(
                "checkpoint_status_or_result_mismatch"
            ),
            "stop conditions",
        ),
    ],
)
def test_recovery_registration_rejects_weakened_contract(
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    plan, launch = _inputs()
    manifest = copy.deepcopy(_registration().to_manifest(plan=plan, launch=launch))
    mutation(manifest)

    with pytest.raises(ValueError, match=message):
        validate_recovery_registration(manifest=manifest, plan=plan, launch=launch)
