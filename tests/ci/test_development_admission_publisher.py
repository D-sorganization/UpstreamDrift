"""Public-consumer contracts for the protected admission publisher (#13618)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CALLER = ROOT / ".github/workflows/development-admission-report.yml"
CALLEE = ROOT / ".github/workflows/reusable-admission-report-publisher.yml"
PROVENANCE = ROOT / "scripts/config/development_admission_publisher_provenance.json"
POLICY = ROOT / ".github/workflows/fleet-policy-gate.yml"
pytestmark = pytest.mark.unit


def _workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True))


def test_local_copy_manifest_binds_the_published_provider_and_bytes() -> None:
    manifest = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["provider_repository"] == "D-sorganization/Repository_Management"
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["provider_commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["provider_bundle_sha256"])
    assert manifest["workflow_source_path"] == CALLEE.relative_to(ROOT).as_posix()
    assert (
        manifest["workflow_sha256"] == hashlib.sha256(CALLEE.read_bytes()).hexdigest()
    )
    assert manifest["source_ci_workflow_id"] == 216580685
    assert manifest["source_ci_workflow_path"] == ".github/workflows/ci-standard.yml"


def test_caller_uses_local_callee_with_explicit_pins_and_existing_secrets() -> None:
    caller = _workflow(CALLER)
    manifest = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert _triggers(caller) == {
        "workflow_run": {"workflows": ["CI Standard"], "types": ["completed"]}
    }
    assert caller["permissions"] == {"contents": "read"}
    assert set(caller["jobs"]) == {"publish-admission-report"}
    job = caller["jobs"]["publish-admission-report"]
    assert job["uses"] == "./.github/workflows/reusable-admission-report-publisher.yml"
    assert "steps" not in job and "runs-on" not in job
    assert job["with"] == {
        "ci_workflow_id": manifest["source_ci_workflow_id"],
        "trusted_provider_commit": manifest["provider_commit"],
        "trusted_bundle_sha256": manifest["provider_bundle_sha256"],
    }
    assert job["secrets"] == {
        "provider_app_id": "${{ secrets.JULES_APP_ID }}",
        "provider_app_private_key": "${{ secrets.JULES_APP_PRIVATE_KEY }}",
    }
    assert "conclusion == 'success'" in job["if"]
    assert "event == 'push'" in job["if"]
    callee = _workflow(CALLEE)
    assert set(job["secrets"]) == set(_triggers(callee)["workflow_call"]["secrets"])
    assert caller["concurrency"]["group"] != callee["concurrency"]["group"]
    assert caller["concurrency"]["cancel-in-progress"] is True


def _drift_guard() -> tuple[str, list[dict]]:
    steps = _workflow(POLICY)["jobs"]["fleet-policy-gate"]["steps"]
    step = next(
        step
        for step in steps
        if step.get("name") == "Verify canonical admission publisher workflow copy"
    )
    return step["run"], steps


def test_drift_guard_uses_the_existing_trusted_provider_before_mode_selection() -> None:
    script, steps = _drift_guard()
    names = [step.get("name") for step in steps]
    assert names.index(
        "Verify canonical admission publisher workflow copy"
    ) < names.index("Select approved development governance mode")
    assert "target/.github/workflows/reusable-admission-report-publisher.yml" in script
    assert "policy/.github/workflows/reusable-admission-report-publisher.yml" in script
    assert "|| true" not in script and "continue-on-error" not in script


@pytest.mark.parametrize(
    "case", ["matching", "changed", "missing_target", "missing_provider"]
)
def test_actual_drift_shell_rejects_changed_or_missing_canonical_copy(
    tmp_path: Path, case: str
) -> None:
    script, _ = _drift_guard()
    relative = ".github/workflows/reusable-admission-report-publisher.yml"
    for root in ("target", "policy"):
        path = tmp_path / root / relative
        path.parent.mkdir(parents=True)
        if case != f"missing_{root if root == 'target' else 'provider'}":
            path.write_bytes(b"trusted canonical bytes\n")
    if case == "changed":
        (tmp_path / "target" / relative).write_bytes(b"modified bytes\n")
        metadata = (
            tmp_path
            / "target/scripts/config/development_admission_publisher_provenance.json"
        )
        metadata.parent.mkdir(parents=True)
        metadata.write_text(
            json.dumps(
                {"workflow_sha256": hashlib.sha256(b"modified bytes\n").hexdigest()}
            )
        )
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        check=False,
    )
    assert (result.returncode == 0) is (case == "matching")
