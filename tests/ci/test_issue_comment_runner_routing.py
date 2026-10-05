"""``issue_comment`` events never reach the self-hosted fleet (RM#1989).

For ``issue_comment`` GitHub exposes the PR marker as
``github.event.issue.pull_request``, not ``github.event.pull_request``, so the
canonical fork guard is TRUE for comments on fork PRs. Every execution of an
``issue_comment``-triggered workflow therefore stays on a GitHub-hosted runner.

Static contract per ``issue_comment`` workflow job:

* ``runs-on`` is a literal hosted label, or an expression whose first ``&&``
  arm routes ``github.event_name == 'issue_comment'`` to ``'ubuntu-latest'``;
* a ``pick-runner`` job never emits a fleet label for ``issue_comment``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

HOSTED = re.compile(r"^(ubuntu|windows|macos)-[\w.]+$")
ROUTE = re.compile(
    r"^\$\{\{\s*github\.event_name\s*==\s*'issue_comment'\s*&&\s*"
    r"'ubuntu-latest'\s*\|\|"
)
FLEET = re.compile(r"d-sorg-fleet|self-hosted")


def _triggers(doc: dict[Any, Any]) -> set[str]:
    on = doc.get("on", doc.get(True))
    if isinstance(on, str):
        return {on}
    return set(on or [])


def _issue_comment_workflows() -> list[Path]:
    found = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "issue_comment" in _triggers(doc):
            found.append(path)
    return found


def _routes_to_hosted(runs_on: object) -> bool:
    if isinstance(runs_on, str):
        return bool(HOSTED.match(runs_on) or ROUTE.match(runs_on.strip()))
    return False


def _picker_emits_fleet_for_issue_comment(job: dict[str, Any]) -> bool:
    for step in job.get("steps", []):
        run = step.get("run", "")
        if "runner=" in run and FLEET.search(run):
            if "issue_comment" not in run or "ubuntu-latest" not in run:
                return True
    return False


def test_issue_comment_workflows_discovered() -> None:
    assert _issue_comment_workflows(), "expected issue_comment workflows"


@pytest.mark.parametrize("path", _issue_comment_workflows(), ids=lambda p: p.name)
def test_issue_comment_jobs_never_reach_the_fleet(path: Path) -> None:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    bad = []
    for name, job in doc["jobs"].items():
        if not _routes_to_hosted(job.get("runs-on")):
            bad.append(f"{name}: runs-on {job.get('runs-on')!r} can reach the fleet")
        if name == "pick-runner" and _picker_emits_fleet_for_issue_comment(job):
            bad.append(f"{name}: emits a fleet label for issue_comment")
    assert not bad, f"{path.name}: " + "; ".join(bad)
