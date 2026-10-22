#!/usr/bin/env python3
# ruff: noqa
# fmt: off
# flake8: noqa
"""Git pre-push fallback: run ``scripts/agent-check`` on the push range."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path


def run_agent_check(from_ref: str, to_ref: str, root: Path) -> int:
    env = os.environ.copy()
    env["FLEET_HOOK_FROM_REF"] = from_ref
    env["FLEET_HOOK_TO_REF"] = to_ref
    script = root / "scripts" / "agent-check"
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        env=env,
        check=False,
    ).returncode


def main(argv: Sequence[str] | None = None, stdin: Iterable[str] | None = None) -> int:
    del argv
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.pre_push_scope import (  # noqa: PLC0415
        is_zero_sha,
        parse_push_refs,
        resolve_from_ref,
    )

    root = Path.cwd()
    status = 0
    for push_ref in parse_push_refs(stdin if stdin is not None else sys.stdin):
        if is_zero_sha(push_ref.local_sha):
            continue
        from_ref = resolve_from_ref(push_ref)
        if from_ref is None:
            print(
                "pre-push-agent-check: no diff base; skipping agent-check",
                file=sys.stderr,
            )
            continue
        status |= run_agent_check(from_ref, push_ref.local_sha, root)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
