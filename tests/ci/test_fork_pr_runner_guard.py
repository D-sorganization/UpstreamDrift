"""Fork pull-request code never reaches the self-hosted fleet (UpstreamDrift, RM#1989).

The first test runs ``scripts/fork_pr_runner_guard.py`` over this repository's
real workflows; the rest pin the checker's rules on small synthetic workflows
so a regression in the checker cannot silently pass the first test.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from scripts import fork_pr_runner_guard as guard

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

GUARD_IF = (
    "    if: >-\n"
    "      (!github.event.pull_request ||\n"
    "      github.event.pull_request.head.repo.full_name == github.repository)\n"
)


def _violations(tmp_path: Path, text: str) -> list[str]:
    (tmp_path / "wf.yml").write_text(dedent(text), encoding="utf-8")
    return guard.find_violations(tmp_path)


def test_no_workflow_runs_fork_pr_code_on_self_hosted() -> None:
    violations = guard.find_violations(WORKFLOWS)
    assert violations == [], "\n".join(violations)


def test_unguarded_fleet_job_on_pull_request_is_rejected(tmp_path: Path) -> None:
    text = "on: pull_request\njobs:\n  t:\n    runs-on: d-sorg-fleet\n"
    assert len(_violations(tmp_path, text)) == 1


@pytest.mark.parametrize(
    "runs_on",
    [
        "[self-hosted, Linux]",
        "${{ needs.pick-runner.outputs.runner }}",
        "${{ matrix.os }}",
        "${{ vars.X == 'local' && 'd-sorg-fleet' || 'ubuntu-latest' }}",
        "{group: fleet}",
    ],
)
def test_non_literal_or_self_hosted_runs_on_fails_closed(
    tmp_path: Path, runs_on: str
) -> None:
    text = f"on: [pull_request]\njobs:\n  t:\n    runs-on: {runs_on}\n"
    assert _violations(tmp_path, text)


def test_reusable_workflow_call_fails_closed(tmp_path: Path) -> None:
    text = "on: pull_request\njobs:\n  t:\n    uses: ./.github/workflows/x.yml\n"
    assert _violations(tmp_path, text)


def test_review_comment_and_workflow_call_triggers_count(tmp_path: Path) -> None:
    for event in ("pull_request_review_comment", "workflow_call"):
        text = f"on: {event}\njobs:\n  t:\n    runs-on: d-sorg-fleet\n"
        assert _violations(tmp_path, text), event


def test_hosted_only_job_is_allowed(tmp_path: Path) -> None:
    text = "on: pull_request\njobs:\n  t:\n    runs-on: ubuntu-24.04\n"
    assert _violations(tmp_path, text) == []


def test_push_only_workflow_is_out_of_scope(tmp_path: Path) -> None:
    text = "on: [push, schedule]\njobs:\n  t:\n    runs-on: d-sorg-fleet\n"
    assert _violations(tmp_path, text) == []


def test_guarded_job_is_allowed(tmp_path: Path) -> None:
    text = "on: pull_request\njobs:\n  t:\n" + GUARD_IF + "    runs-on: d-sorg-fleet\n"
    assert _violations(tmp_path, text) == []


def test_guard_anded_with_other_conditions_is_allowed(tmp_path: Path) -> None:
    text = (
        "on: pull_request\njobs:\n  t:\n"
        + GUARD_IF.rstrip("\n")
        + " &&\n      (always())\n"
        + "    runs-on: d-sorg-fleet\n"
    )
    assert _violations(tmp_path, text) == []


@pytest.mark.parametrize(
    "condition",
    [
        # A top-level || lets the other branch bypass the guard.
        "${{ github.actor == 'x' || (!github.event.pull_request || "
        "github.event.pull_request.head.repo.full_name == github.repository) }}",
        # The guard's two halves must stay together.
        "${{ github.event.pull_request.head.repo.full_name == github.repository }}",
        # The pre-#4464 p1am form misses the pull_request_review* events.
        "github.event_name != 'pull_request' || "
        "github.event.pull_request.head.repo.full_name == github.repository",
    ],
)
def test_bypassable_or_partial_guards_are_rejected(
    tmp_path: Path, condition: str
) -> None:
    text = (
        "on: pull_request\njobs:\n  t:\n"
        f'    if: "{condition}"\n    runs-on: d-sorg-fleet\n'
    )
    assert _violations(tmp_path, text)


def test_fork_routed_runs_on_is_allowed(tmp_path: Path) -> None:
    text = (
        "on: pull_request\njobs:\n  t:\n    runs-on: >-\n"
        "      ${{ github.event.pull_request &&\n"
        "      github.event.pull_request.head.repo.full_name != github.repository &&\n"
        "      'ubuntu-latest' || needs.pick-runner.outputs.runner }}\n"
    )
    assert _violations(tmp_path, text) == []


@pytest.mark.parametrize(
    "runs_on",
    [
        # Fork route not first: an earlier alternative can pick the fleet.
        "${{ vars.M == 'local' && 'd-sorg-fleet' || github.event.pull_request && "
        "github.event.pull_request.head.repo.full_name != github.repository && "
        "'ubuntu-latest' }}",
        # Routes forks to a self-hosted label.
        "${{ github.event.pull_request && "
        "github.event.pull_request.head.repo.full_name != github.repository && "
        "'d-sorg-fleet' || 'd-sorg-fleet' }}",
    ],
)
def test_misordered_or_self_hosted_fork_route_is_rejected(
    tmp_path: Path, runs_on: str
) -> None:
    text = f'on: pull_request\njobs:\n  t:\n    runs-on: "{runs_on}"\n'
    assert _violations(tmp_path, text)


def test_pull_request_target_head_checkout_on_fleet_is_rejected(
    tmp_path: Path,
) -> None:
    text = """\
        on: pull_request_target
        jobs:
          t:
            runs-on: d-sorg-fleet
            steps:
              - uses: actions/checkout@v7
                with:
                  ref: ${{ github.event.pull_request.head.sha }}
        """
    violations = _violations(tmp_path, text)
    assert len(violations) == 1
    assert "checks out PR head" in violations[0]


@pytest.mark.parametrize(
    "run",
    [
        "gh pr checkout 12",
        "git fetch origin pull/${{ github.event.issue.number }}/head",
    ],
)
def test_issue_comment_head_fetch_on_fleet_is_rejected(
    tmp_path: Path, run: str
) -> None:
    text = (
        "on: issue_comment\njobs:\n  t:\n    runs-on: d-sorg-fleet\n"
        f"    steps:\n      - run: {run}\n"
    )
    assert _violations(tmp_path, text)


def test_head_checkout_limited_to_pull_request_event_is_allowed(
    tmp_path: Path,
) -> None:
    text = (
        "on: [pull_request, pull_request_target]\njobs:\n  t:\n"
        + GUARD_IF
        + """\
    runs-on: d-sorg-fleet
    steps:
      - if: github.event_name == 'pull_request'
        uses: actions/checkout@v7
        with:
          ref: ${{ github.event.pull_request.head.sha }}
"""
    )
    assert _violations(tmp_path, text) == []


@pytest.mark.parametrize("level", ["workflow", "job"])
@pytest.mark.parametrize(
    "step",
    [
        "uses: actions/checkout@v7\n        with:\n          ref: ${{ env.PR_SHA }}",
        'run: git fetch origin "$PR_SHA"',
        "run: git checkout ${PR_SHA}",
        "run: git checkout ${{ env['PR_SHA'] }}",
    ],
)
def test_head_ref_aliased_through_env_is_rejected(
    tmp_path: Path, level: str, step: str
) -> None:
    env = "env:\n  PR_SHA: ${{ github.event.pull_request.head.sha }}\n"
    workflow_env = env if level == "workflow" else ""
    job_env = "    " + env.replace("\n  ", "\n      ") if level == "job" else ""
    text = (
        f"on: pull_request_target\n{workflow_env}jobs:\n  t:\n"
        f"    runs-on: d-sorg-fleet\n{job_env}"
        f"    steps:\n      - {step}\n"
    )
    violations = _violations(tmp_path, text)
    assert len(violations) == 1
    assert "checks out PR head" in violations[0]


def test_unreferenced_or_prefix_named_env_alias_is_allowed(tmp_path: Path) -> None:
    text = """\
        on: pull_request_target
        env:
          PR_SHA: ${{ github.event.pull_request.head.sha }}
        jobs:
          t:
            runs-on: d-sorg-fleet
            steps:
              - uses: actions/checkout@v7
              - run: echo "$PR_SHA_LABEL ${{ env.PR_SHAPE }}"
        """
    assert _violations(tmp_path, text) == []


@pytest.mark.parametrize(
    "ref",
    [
        "${{ github.event.pull_request.head['sha'] }}",
        "${{ github.event['pull_request']['head']['ref'] }}",
        "${{ github['head_ref'] }}",
        '${{ github.event.workflow_run["head_sha"] }}',
    ],
)
def test_bracket_form_head_reference_is_rejected(tmp_path: Path, ref: str) -> None:
    text = (
        "on: pull_request_target\njobs:\n  t:\n    runs-on: d-sorg-fleet\n"
        "    steps:\n      - uses: actions/checkout@v7\n"
        f"        with:\n          ref: {ref}\n"
    )
    violations = _violations(tmp_path, text)
    assert len(violations) == 1
    assert "checks out PR head" in violations[0]


def test_head_ref_passed_to_reusable_workflow_is_rejected(tmp_path: Path) -> None:
    text = """\
        on: workflow_run
        jobs:
          call:
            uses: ./.github/workflows/build.yml
            with:
              ref: ${{ github.event.workflow_run.head_sha }}
        """
    violations = _violations(tmp_path, text)
    assert len(violations) == 1
    assert "reusable workflow" in violations[0]


def test_reusable_workflow_reading_caller_head_is_rejected(tmp_path: Path) -> None:
    # A callee inherits its caller's event, so a privileged caller
    # (workflow_run) hands it head refs even though the job guard passes.
    text = (
        "on: workflow_call\njobs:\n  t:\n"
        + GUARD_IF
        + """\
    runs-on: d-sorg-fleet
    steps:
      - uses: actions/checkout@v7
        with:
          ref: ${{ github.event.workflow_run.head_sha }}
"""
    )
    violations = _violations(tmp_path, text)
    assert len(violations) == 1
    assert "checks out PR head" in violations[0]


def test_reusable_call_with_base_inputs_is_allowed(tmp_path: Path) -> None:
    text = """\
        on: workflow_run
        jobs:
          call:
            uses: ./.github/workflows/build.yml
            with:
              ref: ${{ github.event.workflow_run.head_repository.default_branch }}
        """
    assert _violations(tmp_path, text) == []


@pytest.mark.parametrize(
    "repository",
    [
        "${{ github.event.pull_request.head.repo.full_name }}",
        "${{ github.event.pull_request.head.repo.clone_url }}",
        "${{ github.event.workflow_run.head_repository.full_name }}",
    ],
)
def test_checkout_of_head_repository_is_rejected(
    tmp_path: Path, repository: str
) -> None:
    # Without a ref, actions/checkout takes the fork's default branch.
    text = (
        "on: pull_request_target\njobs:\n  t:\n    runs-on: d-sorg-fleet\n"
        "    steps:\n      - uses: actions/checkout@v7\n"
        f"        with:\n          repository: {repository}\n"
    )
    violations = _violations(tmp_path, text)
    assert len(violations) == 1
    assert "checks out PR head" in violations[0]


def test_base_context_job_without_head_checkout_is_allowed(tmp_path: Path) -> None:
    text = """\
        on: pull_request_target
        jobs:
          t:
            runs-on: d-sorg-fleet
            steps:
              - uses: actions/checkout@v7
              - run: gh api repos/x/y
        """
    assert _violations(tmp_path, text) == []


def test_main_reports_and_exits_nonzero(tmp_path: Path) -> None:
    (tmp_path / "wf.yml").write_text(
        "on: pull_request\njobs:\n  t:\n    runs-on: d-sorg-fleet\n", encoding="utf-8"
    )
    assert guard.main(["--workflows-dir", str(tmp_path)]) == 1
    (tmp_path / "wf.yml").write_text(
        "on: pull_request\njobs:\n  t:\n    runs-on: ubuntu-latest\n", encoding="utf-8"
    )
    assert guard.main(["--workflows-dir", str(tmp_path)]) == 0


# --- #1996: same-repo condition and head ref as data -------------------------

SAME_REPO_RUN = (
    "github.event.workflow_run.head_repository.full_name == github.repository"
)
RECOVERY_STEPS = """\
    runs-on: d-sorg-fleet
    steps:
      - uses: actions/checkout@v4
      - name: Recover
        run: |
          python3 scripts/self_hosted_runner_recovery.py \\
            --ref "${{ github.event.workflow_run.head_branch }}"
"""


def _workflow_run_job(condition: str | None, body: str) -> str:
    """Build a ``workflow_run`` workflow with one job from ``body``."""
    guard = f'    if: "{condition}"\n' if condition else ""
    return f"on: workflow_run\njobs:\n  t:\n{guard}{body}"


def test_head_ref_passed_to_script_as_data_is_allowed(tmp_path: Path) -> None:
    # reroute-runner-loss in util-self-hosted-runner-recovery.yml (RM#1995).
    assert _violations(tmp_path, _workflow_run_job(None, RECOVERY_STEPS)) == []


@pytest.mark.parametrize(
    "condition",
    [
        SAME_REPO_RUN,
        "github.repository == github.event.workflow_run.head_repository.full_name",
        f"github.event.workflow_run.conclusion == 'failure' && {SAME_REPO_RUN}",
        f"${{{{ (always()) && {SAME_REPO_RUN} }}}}",
        "github.event.workflow_run.head_repository['full_name'] == github.repository",
        "github.event.pull_request.head.repo.full_name == github.repository",
        "github.repository == github.event.pull_request.head.repo.full_name",
    ],
)
def test_anded_same_repo_condition_exempts_head_checkout(
    tmp_path: Path, condition: str
) -> None:
    body = (
        "    runs-on: d-sorg-fleet\n    steps:\n"
        "      - run: git checkout ${{ github.event.workflow_run.head_branch }}\n"
    )
    assert _violations(tmp_path, _workflow_run_job(condition, body)) == []


@pytest.mark.parametrize(
    "condition",
    [
        f"github.actor == 'x' || {SAME_REPO_RUN}",
        f"{SAME_REPO_RUN} || github.actor == 'x'",
        "github.event.workflow_run.head_repository.full_name != github.repository",
        "github.event.workflow_run.head_repository.full_name == 'other/repo'",
        f"!({SAME_REPO_RUN})",
    ],
)
def test_ored_or_unrelated_condition_does_not_exempt(
    tmp_path: Path, condition: str
) -> None:
    body = (
        "    runs-on: d-sorg-fleet\n    steps:\n"
        "      - run: git checkout ${{ github.event.workflow_run.head_branch }}\n"
    )
    assert _violations(tmp_path, _workflow_run_job(condition, body))


def test_run_step_git_checkout_of_head_branch_is_still_rejected(
    tmp_path: Path,
) -> None:
    body = (
        "    runs-on: d-sorg-fleet\n    steps:\n"
        "      - run: git checkout ${{ github.event.workflow_run.head_branch }}\n"
    )
    violations = _violations(tmp_path, _workflow_run_job(None, body))
    assert len(violations) == 1
    assert "checks out PR head" in violations[0]


def test_checkout_ref_input_is_still_rejected(tmp_path: Path) -> None:
    body = (
        "    runs-on: d-sorg-fleet\n    steps:\n"
        "      - uses: actions/checkout@v4\n        with:\n"
        "          ref: ${{ github.event.workflow_run.head_branch }}\n"
    )
    assert len(_violations(tmp_path, _workflow_run_job(None, body))) == 1


def test_head_ref_in_step_env_with_git_fetch_is_still_rejected(
    tmp_path: Path,
) -> None:
    body = (
        "    runs-on: d-sorg-fleet\n    steps:\n"
        "      - env:\n          REF: ${{ github.event.workflow_run.head_branch }}\n"
        '        run: git -C work fetch origin "$REF"\n'
    )
    assert len(_violations(tmp_path, _workflow_run_job(None, body))) == 1


def test_head_ref_to_non_checkout_action_input_is_data(tmp_path: Path) -> None:
    body = (
        "    runs-on: d-sorg-fleet\n    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: ./.github/actions/notify\n        with:\n"
        "          branch: ${{ github.event.workflow_run.head_branch }}\n"
    )
    assert _violations(tmp_path, _workflow_run_job(None, body)) == []


def test_same_repo_conditioned_ignores_non_strings() -> None:
    assert not guard.same_repo_conditioned(None)
    assert not guard.same_repo_conditioned(True)


@pytest.mark.parametrize(
    "run",
    [
        'git clone --branch "${{ github.event.workflow_run.head_branch }}" '
        '"${{ github.event.workflow_run.head_repository.clone_url }}"',
        "git pull origin ${{ github.event.workflow_run.head_branch }}",
        "git -C x reset --hard ${{ github.event.pull_request.head.sha }}",
        "git -c advice.detachedHead=false merge ${{ github.head_ref }}",
        "gh repo clone ${{ github.event.workflow_run.head_repository.full_name }}",
        "curl -L https://x/${{ github.event.workflow_run.head_sha }}.tar.gz",
        "wget https://x/${{ github.event.workflow_run.head_sha }}.zip",
    ],
)
def test_any_git_or_fetch_command_reading_head_is_rejected(
    tmp_path: Path, run: str
) -> None:
    body = f"    runs-on: d-sorg-fleet\n    steps:\n      - run: {run}\n"
    violations = _violations(tmp_path, _workflow_run_job(None, body))
    assert len(violations) == 1
    assert "checks out PR head" in violations[0]
