"""Contracts for the consumer-side fleet development governance workflow."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import fork_pr_runner_guard


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


def _evaluate_fork_skip(job_if: str, event: str, head_repository: str | None) -> bool:
    """Evaluate this workflow's simple fork predicate against event contexts."""
    expression = job_if.removeprefix("${{").removesuffix("}}").strip()
    expression = expression.replace(
        "github.event.pull_request.head.repo.full_name",
        repr(head_repository),
    )
    expression = expression.replace(
        "github.event.pull_request", "True" if event == "pull_request" else "None"
    )
    expression = expression.replace(
        "github.repository", repr("D-sorganization/UpstreamDrift")
    )
    expression = expression.replace("||", "or").replace("&&", "and")
    expression = re.sub(r"!(?!=)", "not ", expression)
    return bool(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307


def _run_shell(
    script: str, env: dict[str, str], *, cwd: Path
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=cwd,
        env={
            **os.environ,
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
            "RUNNER_TEMP": str(cwd / "runner-temp"),
            **env,
        },
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

    target_checkout_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/checkout@")
        and not step.get("with", {}).get("repository")
    )
    token_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("id") == "policy-app-token"
    )
    policy_checkout_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/checkout@")
        and step.get("with", {}).get("repository")
        == "D-sorganization/Repository_Management"
    )
    setup_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/setup-python@")
    )
    resolver = _step(workflow, "Resolve trusted development-log baseline")
    resolver_index = steps.index(resolver)
    assert target_checkout_index < token_index < policy_checkout_index < setup_index
    assert setup_index + 1 == resolver_index
    assert steps[setup_index]["with"]["python-version"] == "3.12"
    script = resolver["run"]
    assert resolver["env"]["PR_BASE_SHA"] == "${{ github.event.pull_request.base.sha }}"
    assert resolver["env"]["PR_HEAD_SHA"] == "${{ github.event.pull_request.head.sha }}"
    assert resolver["env"]["GITHUB_SHA"] == "${{ github.sha }}"
    assert resolver["env"]["BASE_REF"] == "${{ github.event.pull_request.base.ref }}"
    assert resolver["env"]["REPOSITORY"] == "${{ github.repository }}"
    assert resolver["env"]["DEFAULT_BRANCH"] == (
        "${{ github.event.repository.default_branch }}"
    )
    assert (
        resolver["env"]["MERGE_GROUP_BASE_SHA"]
        == "${{ github.event.merge_group.base_sha }}"
    )
    assert "pull_request)" in script
    assert "merge_group)" in script
    assert "policy/scripts/devlog_trusted_base.py" in script
    assert "target/scripts/devlog_trusted_base.py" not in script
    assert "git -C target rev-parse HEAD" not in script
    for flag in (
        "--repo-root target",
        '--event "$EVENT_NAME"',
        '--event-base-sha "$PR_BASE_SHA"',
        '--pr-head-sha "$PR_HEAD_SHA"',
        '--expected-sha "$GITHUB_SHA"',
        '--base-ref "$BASE_REF"',
        '--repository "$REPOSITORY"',
        '--default-branch "$DEFAULT_BRANCH"',
    ):
        assert flag in script


def test_policy_job_skips_fork_prs_and_actual_guard_accepts_workflow(
    tmp_path: Path,
) -> None:
    workflow = _workflow()
    job_if = workflow["jobs"]["fleet-policy-gate"].get("if")
    assert isinstance(job_if, str)

    workflow_fixture = tmp_path / ".github/workflows"
    workflow_fixture.mkdir(parents=True)
    (workflow_fixture / WORKFLOW.name).write_text(
        WORKFLOW.read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert fork_pr_runner_guard.find_violations(tmp_path) == []

    cases = [
        ("pull_request", "D-sorganization/fork", False),
        ("pull_request", "D-sorganization/UpstreamDrift", True),
        ("merge_group", None, True),
        ("workflow_dispatch", None, True),
        ("schedule", None, True),
    ]
    for event, head_repository, expected in cases:
        assert _evaluate_fork_skip(job_if, event, head_repository) is expected

    assert workflow["jobs"]["fleet-policy-gate"]["runs-on"] == "d-sorg-fleet-docker"
    assert workflow["jobs"]["fleet-policy-gate"]["steps"]


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


def test_consumer_workflow_passes_actual_local_only_runner_guard(
    tmp_path: Path,
) -> None:
    guard_path = REPO_ROOT / ".github/workflows/local-only-runner-guard.yml"
    guard_workflow = yaml.load(
        guard_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    guard_step = next(
        step
        for step in guard_workflow["jobs"]["reject-hosted-runner-routing"]["steps"]
        if step.get("name", "").startswith(
            "Verify workflows use approved local routing"
        )
    )
    workflows_dir = tmp_path / ".github/workflows"
    workflows_dir.mkdir(parents=True)
    (workflows_dir / WORKFLOW.name).write_text(
        WORKFLOW.read_text(encoding="utf-8"), encoding="utf-8"
    )

    result = subprocess.run(
        [sys.executable, "-c", guard_step["run"]],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_consumer_passes_actual_pinning_and_inventory_guards(tmp_path: Path) -> None:
    fixture_root = tmp_path
    workflows_dir = fixture_root / ".github/workflows"
    workflows_dir.mkdir(parents=True)
    (workflows_dir / WORKFLOW.name).write_text(
        WORKFLOW.read_text(encoding="utf-8"), encoding="utf-8"
    )

    inventory_source = (REPO_ROOT / ".github/WORKFLOWS.md").read_text(encoding="utf-8")
    table_lines = [
        line for line in inventory_source.splitlines() if line.startswith("|")
    ]
    header_index = next(
        index for index, line in enumerate(table_lines) if line.startswith("| File ")
    )
    header = table_lines[header_index : header_index + 2]
    consumer_rows = [
        line
        for line in table_lines[header_index + 2 :]
        if line.split("|", maxsplit=2)[1].strip() == WORKFLOW.name
    ]
    inventory_path = fixture_root / ".github/WORKFLOWS.md"
    inventory_path.write_text(
        "\n".join([*header, *consumer_rows]) + "\n", encoding="utf-8"
    )

    pinning = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/check_github_actions_pinned.py"),
            "--workflow-dir",
            str(workflows_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    inventory = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/check_workflow_inventory.py"),
            "--repo-root",
            str(fixture_root),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )

    assert (pinning.returncode, inventory.returncode) == (0, 0), (
        f"pinning guard ({pinning.returncode}): {pinning.stderr}{pinning.stdout}\n"
        f"inventory guard ({inventory.returncode}): "
        f"{inventory.stderr}{inventory.stdout}"
    )

    assert consumer_rows, "fleet-policy-gate.yml inventory metadata is missing"
    row = consumer_rows[0].split("|")
    assert "pull_request" in row[2] and "merge_group" in row[2]
    assert "workflow_dispatch" in row[2] and "daily schedule" in row[2]
    assert "contents: read" in row[4]
    assert "Repository_Management contents: read (App)" in row[4]


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
def test_resolver_uses_protected_helper_result_not_event_anchor_or_head(
    tmp_path: Path,
) -> None:
    policy_helper = tmp_path / "policy/scripts/devlog_trusted_base.py"
    policy_helper.parent.mkdir(parents=True)
    helper_called = tmp_path / "policy-helper-called"
    trusted_base = "c" * 40
    helper_source = f"""
import argparse
import json
import os
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--repo-root", required=True)
parser.add_argument("--event", required=True)
parser.add_argument("--event-base-sha", required=True)
parser.add_argument("--pr-head-sha", required=True)
parser.add_argument("--expected-sha", required=True)
parser.add_argument("--base-ref", required=True)
parser.add_argument("--repository", required=True)
parser.add_argument("--default-branch", required=True)
args = parser.parse_args()
assert args.repo_root == "target"
assert args.event == "pull_request"
assert args.event_base_sha == {"a" * 40!r}
assert args.pr_head_sha == {"b" * 40!r}
assert args.expected_sha == {"d" * 40!r}
assert args.base_ref == "release/next"
assert args.repository == "D-sorganization/UpstreamDrift"
assert args.default_branch == "main"
pathlib.Path(os.environ["HELPER_CALL_MARKER"]).write_text("protected", encoding="utf-8")
print(json.dumps({{"verified": True, "proof_kind": "pull_request_protected_merge_parent", "baseline_sha": "{trusted_base}", "mode": "strict", "receipt": "complete"}}))
"""
    policy_helper.write_text(helper_source, encoding="utf-8")

    target_decoy = tmp_path / "target/scripts/devlog_trusted_base.py"
    target_decoy.parent.mkdir(parents=True)
    target_decoy.write_text(
        "from pathlib import Path\n"
        "Path('candidate-helper-called').write_text('bad')\n"
        f"print('{('e' * 40)}')\n",
        encoding="utf-8",
    )

    resolver = _step(_workflow(), "Resolve trusted development-log baseline")
    result = _run_shell(
        resolver["run"],
        {
            "EVENT_NAME": "pull_request",
            "PR_BASE_SHA": "a" * 40,
            "PR_HEAD_SHA": "b" * 40,
            "GITHUB_SHA": "d" * 40,
            "BASE_REF": "release/next",
            "REPOSITORY": "D-sorganization/UpstreamDrift",
            "DEFAULT_BRANCH": "main",
            "MERGE_GROUP_BASE_SHA": "",
            "HELPER_CALL_MARKER": str(helper_called),
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "RUNNER_TEMP": str(tmp_path / "runner-temp"),
        },
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert helper_called.read_text(encoding="utf-8") == "protected"
    assert not (tmp_path / "candidate-helper-called").exists()
    assert (tmp_path / "output").read_text(encoding="utf-8") == f"sha={trusted_base}\n"
    assert json.loads(
        (tmp_path / "runner-temp/devlog-base-proof.json").read_text(encoding="utf-8")
    ) == {
        "verified": True,
        "proof_kind": "pull_request_protected_merge_parent",
        "baseline_sha": trusted_base,
        "mode": "strict",
        "receipt": "complete",
    }


@pytest.mark.skipif(os.name == "nt", reason="the extracted workflow step requires Bash")
def test_resolver_fails_closed_when_protected_helper_is_missing(tmp_path: Path) -> None:
    target_decoy = tmp_path / "target/scripts/devlog_trusted_base.py"
    target_decoy.parent.mkdir(parents=True)
    target_decoy.write_text(
        "from pathlib import Path\n"
        "Path('candidate-helper-called').write_text('bad')\n"
        f'print(\'{{"baseline_sha":"{"e" * 40}"}}\')\n',
        encoding="utf-8",
    )
    resolver = _step(_workflow(), "Resolve trusted development-log baseline")
    result = _run_shell(
        resolver["run"],
        {
            "EVENT_NAME": "pull_request",
            "PR_BASE_SHA": "a" * 40,
            "PR_HEAD_SHA": "b" * 40,
            "GITHUB_SHA": "d" * 40,
            "BASE_REF": "main",
            "REPOSITORY": "D-sorganization/UpstreamDrift",
            "DEFAULT_BRANCH": "main",
            "MERGE_GROUP_BASE_SHA": "",
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "RUNNER_TEMP": str(tmp_path / "runner-temp"),
        },
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "output").read_text(encoding="utf-8") == "sha=\n"
    assert not (tmp_path / "candidate-helper-called").exists()
    saved_proof = json.loads(
        (tmp_path / "runner-temp/devlog-base-proof.json").read_text(encoding="utf-8")
    )
    assert saved_proof["verified"] is False
    assert saved_proof["proof_kind"] == "helper_unpublished"
    assert saved_proof["baseline_sha"] is None
    assert "not yet published" in saved_proof["reason"]

    selector = _step(_workflow(), "Select approved development governance mode")
    selected = _run_shell(
        selector["run"],
        {
            "DEVLOG_BASE_SHA": "",
            "REPOSITORY": "D-sorganization/UpstreamDrift",
            "EVENT_NAME": "pull_request",
            "RUNNER_TEMP": str(tmp_path / "runner-temp"),
            "GITHUB_OUTPUT": str(tmp_path / "selector-output"),
        },
        cwd=tmp_path,
    )
    assert selected.returncode == 0, selected.stdout + selected.stderr
    assert (tmp_path / "selector-output").read_text(encoding="utf-8") == "mode=strict\n"
    eligibility = json.loads(
        (tmp_path / "runner-temp/devlog-eligibility.json").read_text(encoding="utf-8")
    )
    assert eligibility["mode"] == "strict"


@pytest.mark.skipif(os.name == "nt", reason="the extracted workflow step requires Bash")
@pytest.mark.parametrize(
    ("helper_output", "helper_exit_code"),
    [
        ("not-json", 0),
        (
            '{"verified": true, "proof_kind": "pull_request_protected_merge_parent", "baseline_sha": null}',
            0,
        ),
        (
            '{"verified": false, "proof_kind": "pull_request_protected_merge_parent", "baseline_sha": "cccccccccccccccccccccccccccccccccccccccc"}',
            0,
        ),
        (
            '{"verified": true, "proof_kind": "wrong_proof", "baseline_sha": "cccccccccccccccccccccccccccccccccccccccc"}',
            0,
        ),
        (
            '{"verified": true, "proof_kind": "pull_request_protected_merge_parent", "baseline_sha": "not-a-commit-sha"}',
            0,
        ),
        (
            '{"verified": true, "proof_kind": "pull_request_protected_merge_parent", "baseline_sha": "cccccccccccccccccccccccccccccccccccccccc"}',
            1,
        ),
    ],
)
def test_present_trusted_helper_bad_proof_fails_resolver(
    tmp_path: Path, helper_output: str, helper_exit_code: int
) -> None:
    policy_helper = tmp_path / "policy/scripts/devlog_trusted_base.py"
    policy_helper.parent.mkdir(parents=True)
    policy_helper.write_text(
        "import argparse, os, sys\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--repo-root', required=True)\n"
        "parser.add_argument('--event', required=True)\n"
        "parser.add_argument('--event-base-sha', required=True)\n"
        "parser.add_argument('--pr-head-sha', required=True)\n"
        "parser.add_argument('--expected-sha', required=True)\n"
        "parser.add_argument('--base-ref', required=True)\n"
        "parser.add_argument('--repository', required=True)\n"
        "parser.add_argument('--default-branch', required=True)\n"
        "parser.parse_args()\n"
        "print(os.environ['HELPER_OUTPUT'])\n"
        "raise SystemExit(int(os.environ['HELPER_EXIT_CODE']))\n",
        encoding="utf-8",
    )

    resolver = _step(_workflow(), "Resolve trusted development-log baseline")
    result = _run_shell(
        resolver["run"],
        {
            "EVENT_NAME": "pull_request",
            "PR_BASE_SHA": "a" * 40,
            "PR_HEAD_SHA": "b" * 40,
            "GITHUB_SHA": "d" * 40,
            "BASE_REF": "main",
            "REPOSITORY": "D-sorganization/UpstreamDrift",
            "DEFAULT_BRANCH": "main",
            "MERGE_GROUP_BASE_SHA": "",
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "RUNNER_TEMP": str(tmp_path / "runner-temp"),
            "HELPER_OUTPUT": helper_output,
            "HELPER_EXIT_CODE": str(helper_exit_code),
        },
        cwd=tmp_path,
    )

    assert result.returncode == 2
    if helper_exit_code:
        assert "Protected trusted-base verification failed" in result.stderr
    else:
        assert "Protected trusted-base proof is invalid" in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.skipif(os.name == "nt", reason="the extracted workflow step requires Bash")
def test_merge_group_resolver_keeps_the_event_base_unchanged(tmp_path: Path) -> None:
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
            "EVENT_NAME": "merge_group",
            "PR_BASE_SHA": "",
            "PR_HEAD_SHA": "",
            "GITHUB_SHA": base_sha,
            "BASE_REF": "",
            "REPOSITORY": "D-sorganization/UpstreamDrift",
            "DEFAULT_BRANCH": "main",
            "MERGE_GROUP_BASE_SHA": base_sha,
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "RUNNER_TEMP": str(tmp_path / "runner-temp"),
        },
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "output").read_text(encoding="utf-8") == f"sha={base_sha}\n"
    assert (
        json.loads(
            (tmp_path / "runner-temp/devlog-base-proof.json").read_text(
                encoding="utf-8"
            )
        )["baseline_sha"]
        == base_sha
    )


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
        "workflow_dispatch|schedule)"
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
    assert "devlog-base-proof.json" in artifact["with"]["path"]
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
