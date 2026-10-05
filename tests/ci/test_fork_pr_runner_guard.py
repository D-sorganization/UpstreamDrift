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
    assert guard.main(["--workflows", str(tmp_path)]) == 1
    (tmp_path / "wf.yml").write_text(
        "on: pull_request\njobs:\n  t:\n    runs-on: ubuntu-latest\n", encoding="utf-8"
    )
    assert guard.main(["--workflows", str(tmp_path)]) == 0
