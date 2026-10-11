#!/usr/bin/env python3
"""Reject workflow jobs that can run fork pull-request code on self-hosted runners.

Public fleet repositories run CI on ``d-sorg-fleet``, self-hosted runners on
maintainer hardware (Tools#4464). A fork pull request must never land on that
fleet. Canonical here (Repository_Management#1989), ported from Tools#5427 and
vendored to the other fleet repositories. This checker enforces two rules over
``.github/workflows/*.yml``:

1. **Fork PR code stays off the fleet.** In a workflow triggered by an event
   that runs the pull request's own code and workflow definition
   (``pull_request``, ``pull_request_review``, ``pull_request_review_comment``)
   or by ``workflow_call`` (the caller's trigger is unknown, so fail closed),
   every job that can reach a self-hosted runner must either

   * carry the canonical fork guard as a top-level ``&&`` conjunct of its
     ``if:`` (:data:`SAME_REPO_GUARD`), so a fork PR skips the job; or
   * route fork PRs to a GitHub-hosted runner as the first ``||`` alternative
     of its ``runs-on`` expression (:data:`FORK_ROUTE_CONJUNCTS`), so a fork PR
     still gets a result on a hosted runner. Use this only for lanes that must
     report on fork PRs, such as required checks.

2. **Privileged events never check out PR head code on the fleet.** In a
   workflow triggered by ``pull_request_target``, ``issue_comment`` or
   ``workflow_run`` (which run the base branch's workflow with base-repository
   credentials), a self-hosted job must not check out or fetch the PR head,
   unless that step is restricted to ``github.event_name == 'pull_request'``.
   ``workflow_call`` is privileged here too: a callee inherits its caller's
   event, and a ``workflow_run`` caller passes the job guard. A reusable-
   workflow call under these events must not pass a head ref in ``with:``.
   Bracket property access (``head['sha']``) is read as dotted access, and
   naming the head *repository* counts: checkout without a ref takes the
   fork's default branch.

A job counts as self-hosted unless every ``runs-on`` value is a literal
GitHub-hosted label. Expressions, matrix references, runner groups and
reusable-workflow calls are treated as self-hosted (fail closed).

The check reads expressions textually and does not evaluate them. It is a
defence in depth, not a sandbox: on ``pull_request`` a fork can edit the
workflow file itself, so the repository settings listed in Tools#4464
(approval for all outside collaborators, runner-group access) remain the
primary control.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.fork_pr_guard_analysis import (
        PULL_REQUEST_ONLY,
        SAME_REPO_CONDITIONS,
        dotted,
        head_checkout_steps,
        head_ref_aliases,
        normalize,
        passes_head_ref,
        requires_conjunct,
        same_repo_conditioned,
        split_top_level,
    )
except ImportError:  # executed as a file: scripts/ is on sys.path
    from fork_pr_guard_analysis import (
        PULL_REQUEST_ONLY,
        SAME_REPO_CONDITIONS,
        dotted,
        head_checkout_steps,
        head_ref_aliases,
        normalize,
        passes_head_ref,
        requires_conjunct,
        same_repo_conditioned,
        split_top_level,
    )

__all__ = [
    "PULL_REQUEST_ONLY",
    "SAME_REPO_CONDITIONS",
    "dotted",
    "find_violations",
    "main",
    "normalize",
    "requires_conjunct",
    "same_repo_conditioned",
    "split_top_level",
]

LOG = logging.getLogger("fork_pr_runner_guard")

#: Events whose workflow runs with the pull request's code and definition.
PR_CODE_TRIGGERS = frozenset(
    {
        "pull_request",
        "pull_request_review",
        "pull_request_review_comment",
        "workflow_call",
    }
)
#: Events whose workflow runs from the base branch with base credentials.
BASE_CONTEXT_TRIGGERS = frozenset(
    {"pull_request_target", "issue_comment", "workflow_run"}
)

#: The job-level ``if:`` guard: true off pull requests and for same-repo PRs,
#: false for a fork PR and for a PR whose head repository was deleted.
SAME_REPO_GUARD = (
    "!github.event.pull_request || "
    "github.event.pull_request.head.repo.full_name == github.repository"
)
#: The leading ``runs-on`` alternative that sends a fork PR to a hosted label.
FORK_ROUTE_CONJUNCTS = (
    "github.event.pull_request",
    "github.event.pull_request.head.repo.full_name != github.repository",
)
#: GitHub-hosted labels, kept in step with Repository_Management
#: ``scripts/runner_routing_guard.py``: ``ubuntu``/``macos``/``windows`` with at
#: least one suffix. A bare ``Linux``/``Windows`` is a self-hosted OS label.
_HOSTED = r"(ubuntu|macos|windows)(-[a-z0-9.]+)+"
HOSTED_LABEL = re.compile(rf"^{_HOSTED}$", re.IGNORECASE)
HOSTED_LITERAL = re.compile(rf"^'{_HOSTED}'$", re.IGNORECASE)


def _runs_on_values(job: dict[str, Any]) -> list[Any] | None:
    """Return the ``runs-on`` values, or None for a reusable-workflow call."""
    if "runs-on" not in job:
        return None
    runs_on = job["runs-on"]
    return runs_on if isinstance(runs_on, list) else [runs_on]


def hosted_only(job: dict[str, Any]) -> bool:
    """True when every ``runs-on`` value is a literal GitHub-hosted label."""
    values = _runs_on_values(job)
    if not values:
        return False
    return all(
        isinstance(value, str) and "${{" not in value and HOSTED_LABEL.match(value)
        for value in values
    )


def fork_routed(job: dict[str, Any]) -> bool:
    """True when ``runs-on`` sends fork PRs to a hosted label before anything."""
    values = _runs_on_values(job)
    if not values or len(values) != 1 or not isinstance(values[0], str):
        return False
    first = split_top_level(normalize(values[0]), "||")[0]
    conjuncts = [normalize(part) for part in split_top_level(first, "&&")]
    return (
        len(conjuncts) == len(FORK_ROUTE_CONJUNCTS) + 1
        and tuple(conjuncts[:-1]) == FORK_ROUTE_CONJUNCTS
        and bool(HOSTED_LITERAL.match(conjuncts[-1]))
    )


def triggers(workflow: dict[Any, Any]) -> set[str]:
    """Return the event names that trigger ``workflow``."""
    on = workflow.get("on", workflow.get(True))
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return {str(event) for event in on}
    if isinstance(on, dict):
        return {str(event) for event in on}
    return set()


def job_violations(
    wf_name: str,
    events: set[str],
    job_id: str,
    job: dict[str, Any],
    workflow: dict[Any, Any] | None = None,
) -> list[str]:
    """Return why one job can run fork PR code on a self-hosted runner.

    ``workflow`` supplies workflow-level ``env`` for alias resolution.
    """
    if hosted_only(job):
        return []
    violations: list[str] = []
    pr_events = sorted(events & PR_CODE_TRIGGERS)
    if pr_events and not (
        requires_conjunct(job.get("if"), SAME_REPO_GUARD) or fork_routed(job)
    ):
        violations.append(
            f"{wf_name}::{job_id}: runs on {', '.join(pr_events)} and can reach "
            f"a self-hosted runner without the fork guard; add "
            f"'({SAME_REPO_GUARD}) && ...' to the job if:, or start runs-on with "
            f"'{' && '.join(FORK_ROUTE_CONJUNCTS)} && <hosted label> || ...'"
        )
    privileged_events = events & (BASE_CONTEXT_TRIGGERS | {"workflow_call"})
    if privileged_events and not same_repo_conditioned(job.get("if")):
        aliases = head_ref_aliases(workflow, job)
        privileged = ", ".join(sorted(privileged_events))
        if passes_head_ref(job, aliases):
            violations.append(
                f"{wf_name}::{job_id}: passes a PR head ref to a reusable workflow "
                f"under {privileged}; the callee can check it out on a "
                f"self-hosted runner"
            )
        for step in head_checkout_steps(job, aliases):
            violations.append(
                f"{wf_name}::{job_id}: step '{step}' checks out PR head code on a "
                f"self-hosted runner under {privileged}; "
                f"restrict it to {PULL_REQUEST_ONLY} or drop the head ref"
            )
    return violations


def find_violations(workflow_dir: Path) -> list[str]:
    """Return one message per job in ``workflow_dir`` that breaks a rule."""
    violations: list[str] = []
    for path in sorted(workflow_dir.glob("*.y*ml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            violations.append(f"{path.name}: failed to parse YAML: {exc}")
            continue
        if not isinstance(data, dict) or not isinstance(data.get("jobs"), dict):
            continue
        events = triggers(data)
        for job_id, job in data["jobs"].items():
            if isinstance(job, dict):
                violations.extend(
                    job_violations(path.name, events, str(job_id), job, data)
                )
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checker as a CLI.

    Preconditions: --workflows-dir (alias --workflows) names a directory.
    Postconditions: returns 0 when no job breaks a rule, else 1 after logging
    one ::error:: line per violation.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workflows-dir",
        "--workflows",
        dest="workflows_dir",
        type=Path,
        default=Path(".github") / "workflows",
        help="Directory of workflow files (default: .github/workflows).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    violations = find_violations(args.workflows_dir)
    for violation in violations:
        LOG.error("::error::%s", violation)
    if violations:
        LOG.error("%d job(s) can run fork PR code on self-hosted.", len(violations))
        return 1
    LOG.info("No job can run fork PR code on a self-hosted runner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
