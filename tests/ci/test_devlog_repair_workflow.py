"""Contracts for the consumer-side fleet development governance workflow."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/fleet-policy-gate.yml"
pytestmark = pytest.mark.unit


def _workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _step(workflow: dict, name: str) -> dict:
    return next(
        step
        for step in workflow["jobs"]["fleet-policy-gate"]["steps"]
        if step.get("name") == name
    )


def _run_shell(
    script: str, env: dict[str, str], *, cwd: Path
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=cwd,
        env={**os.environ, **env},
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )


def test_consumer_uses_protected_policy_checkout_and_trusted_event_base() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["fleet-policy-gate"]["steps"]
    checkouts = [
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert any(
        step.get("with", {}).get("repository")
        == "D-sorganization/Repository_Management"
        and step["with"].get("ref") == "main"
        and step["with"].get("path") == "policy"
        for step in checkouts
    )

    resolver = _step(workflow, "Resolve trusted development-log baseline")
    script = resolver["run"]
    assert resolver["env"]["PR_BASE_SHA"] == "${{ github.event.pull_request.base.sha }}"
    assert (
        resolver["env"]["MERGE_GROUP_BASE_SHA"]
        == "${{ github.event.merge_group.base_sha }}"
    )
    assert 'pull_request) base_sha="$PR_BASE_SHA"' in script
    assert 'merge_group) base_sha="$MERGE_GROUP_BASE_SHA"' in script
    assert "github.sha" not in script and "git -C target rev-parse HEAD" not in script

    base_file = _step(workflow, "Resolve trusted development-log baseline")["env"]
    assert "${{ github.sha }}" not in base_file


def test_policy_checkout_uses_fail_closed_repository_scoped_app_token() -> None:
    steps = _workflow()["jobs"]["fleet-policy-gate"]["steps"]
    policy_checkout_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/checkout@")
        and step.get("with", {}).get("repository")
        == "D-sorganization/Repository_Management"
    )
    token_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("id") == "policy-app-token"
    )
    token_step = steps[token_index]
    target_checkout = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/checkout@")
        and not step.get("with", {}).get("repository")
    )
    policy_checkout = steps[policy_checkout_index]

    assert token_index < policy_checkout_index
    assert token_step["uses"] == (
        "actions/create-github-app-token@64ab729a6294000f5aa4a4bc0b73d1f32bc45510"
    )
    assert token_step["with"]["app-id"] == "${{ secrets.JULES_APP_ID }}"
    assert token_step["with"]["private-key"] == "${{ secrets.JULES_APP_PRIVATE_KEY }}"
    assert token_step["with"]["owner"] == "${{ github.repository_owner }}"
    assert token_step["with"]["repositories"] == "Repository_Management"
    assert token_step["with"]["permission-contents"] == "read"
    assert "if" not in token_step and not token_step.get("continue-on-error", False)

    assert target_checkout["with"]["path"] == "target"
    assert target_checkout["with"]["token"] == "${{ github.token }}"
    assert (
        policy_checkout["with"]["token"]
        == "${{ steps.policy-app-token.outputs.token }}"
    )
    assert policy_checkout["with"]["persist-credentials"] == "false"


@pytest.mark.skipif(os.name == "nt", reason="the extracted workflow step requires Bash")
@pytest.mark.parametrize(
    ("event", "env_key"),
    [("pull_request", "PR_BASE_SHA"), ("merge_group", "MERGE_GROUP_BASE_SHA")],
)
def test_event_requiring_base_fails_when_its_base_sha_is_missing(
    tmp_path: Path, event: str, env_key: str
) -> None:
    resolver = _step(_workflow(), "Resolve trusted development-log baseline")
    result = _run_shell(
        resolver["run"],
        {
            "EVENT_NAME": event,
            "PR_BASE_SHA": "" if env_key == "PR_BASE_SHA" else "a" * 40,
            "MERGE_GROUP_BASE_SHA": ""
            if env_key == "MERGE_GROUP_BASE_SHA"
            else "a" * 40,
            "GITHUB_OUTPUT": str(tmp_path / "output"),
        },
        cwd=tmp_path,
    )
    assert result.returncode == 2
    assert "Required event base SHA is missing" in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="the extracted workflow step requires Bash")
@pytest.mark.parametrize(
    ("event", "base_env"),
    [("pull_request", "PR_BASE_SHA"), ("merge_group", "MERGE_GROUP_BASE_SHA")],
)
def test_resolver_exports_the_event_base_commit(
    tmp_path: Path, event: str, base_env: str
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=target, check=True, timeout=10)
    subprocess.run(
        ["git", "config", "user.email", "ci@example.invalid"], cwd=target, check=True
    )
    subprocess.run(["git", "config", "user.name", "CI test"], cwd=target, check=True)
    (target / "tracked").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked"], cwd=target, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture"], cwd=target, check=True, timeout=10
    )
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=target,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    resolver = _step(_workflow(), "Resolve trusted development-log baseline")
    result = _run_shell(
        resolver["run"],
        {
            "EVENT_NAME": event,
            "PR_BASE_SHA": base_sha if base_env == "PR_BASE_SHA" else "",
            "MERGE_GROUP_BASE_SHA": base_sha
            if base_env == "MERGE_GROUP_BASE_SHA"
            else "",
            "GITHUB_OUTPUT": str(tmp_path / "output"),
        },
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "output").read_text(encoding="utf-8") == f"sha={base_sha}\n"


def test_manual_and_schedule_default_to_strict_without_head_fallback() -> None:
    workflow = _workflow()
    selector = _step(workflow, "Select approved development governance mode")
    script = selector["run"]
    assert selector["env"]["EVENT_NAME"] == "${{ github.event_name }}"
    assert 'mode":"strict"' in script
    assert "Reconciliation or protected selector not yet published" in script
    assert "policy/scripts/devlog_repair_eligibility.py" in script
    assert '--base-sha "$DEVLOG_BASE_SHA" --head-sha "$head_sha"' in script
    assert '--event "$EVENT_NAME"' in script
    assert (
        "workflow_dispatch|schedule) ;;"
        in _step(workflow, "Resolve trusted development-log baseline")["run"]
    )
    assert (
        'mode="${DEVLOG_MODE:-strict}"'
        in _step(workflow, "Validate development log from central policy repository")[
            "run"
        ]
    )


def test_only_eligible_selector_result_can_choose_repair_and_result_stays_visible() -> (
    None
):
    workflow = _workflow()
    selector = _step(workflow, "Select approved development governance mode")["run"]
    assert 'case "$mode" in strict|repair)' in selector
    assert "mode=$(python -c" in selector

    gate = _step(workflow, "Validate development log from central policy repository")
    assert "policy/shared_scripts/development_log_gate.py" in gate["run"]
    assert '--base-ref "$DEVLOG_BASE_SHA"' in gate["run"]
    assert 'tee "$RUNNER_TEMP/devlog-result.json"' in gate["run"]

    summary = _step(workflow, "Publish development governance result")
    assert summary["if"] == "always()"
    assert "Reconciliation Required" in summary["run"]
    assert 'data.get("reconciliation_required") is True or not passed' in summary["run"]
    artifact = _step(workflow, "Preserve development governance receipts")
    assert artifact["if"] == "always()"
    assert "devlog-eligibility.json" in artifact["with"]["path"]
    assert "devlog-result.json" in artifact["with"]["path"]


@pytest.mark.skipif(os.name == "nt", reason="the extracted workflow step requires Bash")
@pytest.mark.parametrize(
    ("selected_mode", "expected_return"), [("repair", 0), ("anything", 2)]
)
def test_selector_accepts_only_repair_or_strict(
    tmp_path: Path, selected_mode: str, expected_return: int
) -> None:
    selector = _step(_workflow(), "Select approved development governance mode")
    helper = tmp_path / "policy/scripts/devlog_repair_eligibility.py"
    helper.parent.mkdir(parents=True)
    helper.touch()
    target = tmp_path / "target"
    target.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=target, check=True, timeout=10)
    subprocess.run(
        ["git", "config", "user.email", "ci@example.invalid"], cwd=target, check=True
    )
    subprocess.run(["git", "config", "user.name", "CI test"], cwd=target, check=True)
    (target / "tracked").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked"], cwd=target, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture"], cwd=target, check=True, timeout=10
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    real_python = (
        Path(
            os.environ.get(
                "VIRTUAL_ENV", "/home/dieterolson/Repositories/UpstreamDrift/.venv"
            )
        )
        / "bin/python"
    )
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *devlog_repair_eligibility.py*) printf '%s\\n' '{\"mode\":\""
        + selected_mode
        + "\"}' ;;\n"
        f'  *) exec {real_python} "$@" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    output = tmp_path / "output"
    result = _run_shell(
        selector["run"],
        {
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "DEVLOG_BASE_SHA": "a" * 40,
            "REPOSITORY": "D-sorganization/UpstreamDrift",
            "EVENT_NAME": "merge_group",
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_OUTPUT": str(output),
        },
        cwd=tmp_path,
    )
    assert result.returncode == expected_return
    if selected_mode == "repair":
        assert "mode=repair" in output.read_text(encoding="utf-8")
    else:
        assert "Invalid governance mode" in result.stderr


def test_manual_schedule_and_merge_group_have_correct_concurrency_behavior() -> None:
    workflow = _workflow()
    assert "merge_group" in workflow["on"]
    assert (
        workflow["concurrency"]["cancel-in-progress"]
        == "${{ github.event_name == 'pull_request' }}"
    )
    assert "workflow_dispatch" in workflow["on"]
    assert "schedule" in workflow["on"]
