#!/usr/bin/env python3
"""Fail when GitHub Actions workflow jobs route to hosted runners."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in underprovisioned CI
    yaml = None  # type: ignore[assignment]

WORKFLOW_DIR = Path(".github") / "workflows"
HOSTED_RUNNER = re.compile(r"^(ubuntu|macos|windows)(-latest|-\d+)?$")
ALLOWLIST_FILES = frozenset(
    {
        "local-only-runner-guard.yml",
        "runner-health-alert.yml",
    }
)
ALLOWLIST_JOB_NAMES = frozenset(
    {
        "Reject hosted runner routing",
        "Local-Only Workflow Runner Guard",
    }
)


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


def _changed_workflows_for_pull_request(workflow_dir: Path) -> list[Path] | None:
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        return None

    base_ref = _pull_request_base_ref()
    if not base_ref:
        return None

    subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=1", "origin", base_ref],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            f"origin/{base_ref}",
            "HEAD",
            "--",
            str(workflow_dir),
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


def _workflow_paths(workflow_dir: Path) -> list[Path]:
    changed_paths = _changed_workflows_for_pull_request(workflow_dir)
    if changed_paths is not None:
        return sorted(path for path in changed_paths if path.exists())

    return sorted(
        path for path in workflow_dir.rglob("*") if path.suffix in {".yml", ".yaml"}
    )


def _runs_on_values(job: dict[str, Any]) -> list[str]:
    runs_on = job.get("runs-on")
    if isinstance(runs_on, str):
        values = [runs_on]
    elif isinstance(runs_on, list):
        values = [value for value in runs_on if isinstance(value, str)]
    else:
        return []

    if any("matrix.os" in value for value in values):
        values = []
        strategy = job.get("strategy") or {}
        matrix = strategy.get("matrix") if isinstance(strategy, dict) else {}
        if isinstance(matrix, dict):
            os_axis = matrix.get("os")
            if isinstance(os_axis, list):
                values.extend(value for value in os_axis if isinstance(value, str))
            include = matrix.get("include")
            if isinstance(include, list):
                values.extend(
                    item["os"]
                    for item in include
                    if isinstance(item, dict) and isinstance(item.get("os"), str)
                )
    return values


def _is_fleet_runner(value: str) -> bool:
    return "pick-runner" in value or "d-sorg-fleet" in value or "self-hosted" in value


def _workflow_failures(path: Path) -> list[str]:
    if yaml is None:
        return ["PyYAML is required to parse workflow routing."]
    if path.name in ALLOWLIST_FILES:
        return []

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"{path}: failed to parse YAML: {exc}"]

    if not isinstance(data, dict):
        return []
    jobs = data.get("jobs") or {}
    if not isinstance(jobs, dict):
        return []

    failures: list[str] = []
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue
        if job.get("name") in ALLOWLIST_JOB_NAMES:
            continue
        if "uses" in job and "runs-on" not in job:
            continue
        for value in _runs_on_values(job):
            runner = value.strip()
            if _is_fleet_runner(runner):
                continue
            if HOSTED_RUNNER.match(runner):
                failures.append(
                    f"{path}:{job_id}: runs-on {runner!r} is a hosted runner"
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-dir", type=Path, default=WORKFLOW_DIR)
    args = parser.parse_args(argv)

    workflow_dir = args.workflow_dir
    if not workflow_dir.exists():
        print("No .github/workflows directory; nothing to check.")
        return 0

    failures = [
        failure
        for path in _workflow_paths(workflow_dir)
        for failure in _workflow_failures(path)
    ]

    if failures:
        print(
            "GitHub-hosted runner routing is forbidden. "
            "Use local self-hosted runners only."
        )
        print("\n".join(failures))
        return 1

    print("Workflow runner routing is local-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
