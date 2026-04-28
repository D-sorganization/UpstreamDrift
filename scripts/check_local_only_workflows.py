#!/usr/bin/env python3
"""Fail when GitHub Actions workflows can route to hosted runners."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

WORKFLOW_DIR = Path(".github") / "workflows"
BANNED = (
    "ubuntu-latest",
    "windows-latest",
    "macos-latest",
    "force_cloud",
    "mode=cloud",
    "Routing to GitHub-hosted",
    "using GitHub-hosted",
    "runner=ubuntu-latest",
    "runner=windows-latest",
    "runner=macos-latest",
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


def _changed_workflows_for_pull_request() -> list[Path] | None:
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


def main() -> int:
    failures: list[str] = []
    if not WORKFLOW_DIR.exists():
        return 0

    for path in _workflow_paths():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8-sig")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for token in BANNED:
                if token in line:
                    failures.append(
                        f"{path}:{line_number}: banned hosted-runner token {token!r}"
                    )

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
