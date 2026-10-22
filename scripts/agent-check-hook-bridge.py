#!/usr/bin/env python3
# ruff: noqa
# fmt: off
# flake8: noqa
"""Provider-specific stdin/stdout bridge around ``scripts/agent-check``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    start = Path.cwd()
    candidates = [start, *start.parents, Path(__file__).resolve().parent.parent]
    for root in candidates:
        if (root / "fleet-hooks" / "manifest.yml").is_file():
            return root
    return Path(__file__).resolve().parent.parent


def run_agent_check(root: Path) -> tuple[int, dict[str, object]]:
    script = root / "scripts" / "agent-check"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    payload: dict[str, object] = {}
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = {"raw": proc.stdout}
    return proc.returncode, payload


def summarize(report: dict[str, object]) -> str:
    checks = report.get("checks")
    if not isinstance(checks, list):
        return "agent-check failed; re-run scripts/agent-check locally."
    failures = [c for c in checks if isinstance(c, dict) and c.get("status") == "fail"]
    if not failures:
        return "agent-check reported failure; inspect scripts/agent-check output."
    lines = [
        f"- {item.get('id', 'check')}: {item.get('message', 'failed')}"
        for item in failures
        if isinstance(item, dict)
    ]
    return "Agent hook checks failed before handoff:\n" + "\n".join(lines[:12])


def cursor_stop() -> int:
    _ = sys.stdin.read()
    code, report = run_agent_check(repo_root())
    if code == 0:
        print("{}")
        return 0
    print(json.dumps({"followup_message": summarize(report)}))
    return 2


def claude_stop() -> int:
    _ = sys.stdin.read()
    return run_agent_check(repo_root())[0]


def codex_stop() -> int:
    raw = sys.stdin.read()
    stop_hook_active = False
    if raw.strip():
        try:
            payload = json.loads(raw)
            stop_hook_active = bool(payload.get("stop_hook_active"))
        except json.JSONDecodeError:
            stop_hook_active = False
    if stop_hook_active:
        print("{}")
        return 0
    code, report = run_agent_check(repo_root())
    if code == 0:
        print("{}")
        return 0
    print(json.dumps({"decision": "block", "reason": summarize(report)}))
    return 0


def antigravity_stop() -> int:
    _ = sys.stdin.read()
    code, report = run_agent_check(repo_root())
    if code == 0:
        print("{}")
        return 0
    print(json.dumps({"decision": "continue", "reason": summarize(report)}))
    return 0


def antigravity_post_tool() -> int:
    _ = sys.stdin.read()
    code, report = run_agent_check(repo_root())
    if code == 0:
        print("{}")
        return 0
    print(summarize(report), file=sys.stderr)
    return 1


def gemini_after_tool() -> int:
    _ = sys.stdin.read()
    code, report = run_agent_check(repo_root())
    if code == 0:
        print("{}")
        return 0
    print(summarize(report), file=sys.stderr)
    return 2


def gemini_after_agent() -> int:
    _ = sys.stdin.read()
    code, report = run_agent_check(repo_root())
    if code == 0:
        print("{}")
        return 0
    print(summarize(report), file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(
            "usage: agent-check-hook-bridge.py "
            "<cursor-stop|claude-stop|codex-stop|antigravity-stop|"
            "antigravity-post-tool|gemini-after-tool|gemini-after-agent>",
            file=sys.stderr,
        )
        return 2
    handlers = {
        "cursor-stop": cursor_stop,
        "claude-stop": claude_stop,
        "codex-stop": codex_stop,
        "antigravity-stop": antigravity_stop,
        "antigravity-post-tool": antigravity_post_tool,
        "gemini-after-tool": gemini_after_tool,
        "gemini-after-agent": gemini_after_agent,
    }
    try:
        return handlers[args[0]]()
    except KeyError:
        print(f"unknown mode: {args[0]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
