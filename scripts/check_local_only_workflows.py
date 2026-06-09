#!/usr/bin/env python3
"""Fail when GitHub Actions workflows can route jobs to hosted runners.

This is the *real* hosted-runner audit referenced by the ``CI Standard /
Reject hosted runner routing`` required status check (issue #7127). It parses
each workflow's ``runs-on`` (including ``strategy.matrix.os`` expansion) and
rejects hosted runner labels (``ubuntu``/``macos``/``windows``), while allowing:

- the two canary workflow files that *must* stay operable when the fleet is
  down (``local-only-runner-guard.yml``, ``runner-health-alert.yml``), and
- jobs literally named ``Reject hosted runner routing`` /
  ``Local-Only Workflow Runner Guard`` (the guard's own canary jobs), and
- ``runs-on`` expressions that resolve at runtime to fleet labels
  (``pick-runner`` outputs, ``d-sorg-fleet*``, ``self-hosted``).

The allowlist is what lets this guard run *itself* on ``ubuntu-latest`` without
self-tripping — the reason the previous ``echo "Bypass"`` stub existed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import yaml

WORKFLOW_DIR = Path(".github") / "workflows"

# Workflow files allowlisted to run on hosted runners. These are canaries that
# must remain operable when the self-hosted fleet itself is down. Kept in sync
# with .github/workflows/local-only-runner-guard.yml.
ALLOWLIST_FILES = frozenset(
    {
        "local-only-runner-guard.yml",
        "runner-health-alert.yml",
    }
)

# Job names allowlisted to run on hosted runners — the two human-facing names
# every repo's CI standard refers to.
ALLOWLIST_JOB_NAMES = frozenset(
    {
        "Reject hosted runner routing",
        "Local-Only Workflow Runner Guard",
    }
)

HOSTED = re.compile(r"^(ubuntu|macos|windows)(-latest|-\d+(\.\d+)?)?$")

# Substrings of a runs-on expression that resolve to fleet/self-hosted labels.
FLEET_TOKENS = ("pick-runner", "d-sorg-fleet", "self-hosted")


def _pull_request_base_ref() -> str | None:
    base_ref = os.environ.get("GITHUB_BASE_REF")
    if base_ref:
        return base_ref

    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return None

    try:
        payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    return payload.get("pull_request", {}).get("base", {}).get("ref")


def _changed_workflows_for_pull_request() -> list[Path] | None:
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        return None

    base_ref = _pull_request_base_ref()
    if not base_ref:
        return None

    subprocess.run(  # noqa: S603,S607 - fixed git argv, no shell
        ["git", "fetch", "--no-tags", "--depth=1", "origin", base_ref],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    result = subprocess.run(  # noqa: S603,S607 - fixed git argv, no shell
        [
            "git",
            "diff",
            "--name-only",
            f"origin/{base_ref}",
            "HEAD",
            "--",
            str(WORKFLOW_DIR),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None

    return [
        Path(line)
        for line in result.stdout.splitlines()
        if Path(line).suffix in {".yml", ".yaml"}
    ]


def _workflow_paths() -> list[Path]:
    changed_paths = _changed_workflows_for_pull_request()
    if changed_paths is not None:
        return sorted(path for path in changed_paths if path.exists())

    return sorted(
        path for path in WORKFLOW_DIR.rglob("*") if path.suffix in {".yml", ".yaml"}
    )


def collect_runs_on(job: dict) -> list[str]:
    """Return the candidate ``runs-on`` label strings for a job.

    Resolves ``strategy.matrix.os`` (and ``matrix.include[*].os``) when the
    ``runs-on`` references ``matrix.os``.
    """
    runs_on = job.get("runs-on")
    values: list[str] = []
    if isinstance(runs_on, str):
        values.append(runs_on)
    elif isinstance(runs_on, list):
        values.extend(v for v in runs_on if isinstance(v, str))

    if any("matrix.os" in v for v in values):
        values = []
        strategy = job.get("strategy") or {}
        matrix = strategy.get("matrix") or {}
        if isinstance(matrix, dict):
            os_axis = matrix.get("os")
            if isinstance(os_axis, list):
                values.extend(v for v in os_axis if isinstance(v, str))
            include = matrix.get("include")
            if isinstance(include, list):
                for item in include:
                    if isinstance(item, dict) and isinstance(item.get("os"), str):
                        values.append(item["os"])
    return values


def audit_workflow(path: Path) -> list[str]:
    """Return hosted-runner-routing violations for a single workflow file."""
    if path.name in ALLOWLIST_FILES:
        return []

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return [f"{path.name}: failed to parse YAML: {exc}"]

    if not isinstance(data, dict):
        return []
    jobs = data.get("jobs") or {}
    if not isinstance(jobs, dict):
        return []

    violations: list[str] = []
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue
        if job.get("name") in ALLOWLIST_JOB_NAMES:
            continue
        # Reusable workflow callers don't have runs-on.
        if "uses" in job and "runs-on" not in job:
            continue

        values = collect_runs_on(job)
        if not values:
            raw = job.get("runs-on")
            if isinstance(raw, str):
                values = [raw]

        for value in values:
            cleaned = value.strip()
            if any(token in cleaned for token in FLEET_TOKENS):
                continue
            if HOSTED.match(cleaned):
                violations.append(
                    f"{path.name}::{job_id}: runs-on '{cleaned}' is a hosted "
                    "runner; route to d-sorg-fleet-docker (or a pick-runner "
                    "expression)."
                )
    return violations


def main() -> int:
    if not WORKFLOW_DIR.exists():
        print("No .github/workflows directory; nothing to check.")
        return 0

    failures: list[str] = []
    for path in _workflow_paths():
        failures.extend(audit_workflow(path))

    if failures:
        print(
            "GitHub-hosted runner routing is forbidden. "
            "Use local self-hosted runners only."
        )
        print("\n".join(failures))
        print(
            "Allowed: local-only-runner-guard.yml, runner-health-alert.yml, "
            "and jobs literally named 'Reject hosted runner routing'."
        )
        return 1

    print("Workflow runner routing is local-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
