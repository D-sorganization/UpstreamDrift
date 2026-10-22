# ruff: noqa
# fmt: off
# flake8: noqa
"""Generate provider hook adapters that call ``scripts/agent-check``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

try:
    from src.contracts import require  # noqa: PLC0415
except ImportError:
    def require(condition: bool, message: str = "Precondition failed") -> None:
        if not condition:
            raise ValueError(message)

BRIDGE_REL = "scripts/agent-check-hook-bridge.py"
CURSOR_HOOK_VERSION = 1
CODEX_CLI_PIN = "0.139.0"
ANTIGRAVITY_HOOK_GROUP = "fleet-agent-check"
SUPPORTED_TARGETS = frozenset(
    {"cursor", "claude", "codex", "antigravity", "gemini", "pre-push"}
)


class AdapterError(ValueError):
    """Raised when manifest or adapter validation fails."""


@dataclass(frozen=True)
class WriteResult:
    path: str
    changed: bool


def load_manifest(root: Path) -> dict[str, Any]:
    path = root / "fleet-hooks" / "manifest.yml"
    require(path.is_file(), f"missing manifest: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "manifest must be a mapping")
    require(
        int(data.get("schema_version", 0)) == 1, "unsupported manifest schema_version"
    )
    return cast(dict[str, Any], data)


def default_timeout(manifest: dict[str, Any]) -> int:
    defaults = manifest.get("defaults") or {}
    return int(defaults.get("timeout_seconds") or 120)


def bridge_command(mode: str) -> str:
    return f"python3 {BRIDGE_REL} {mode}"


def bridge_command_git_root(mode: str) -> str:
    """Resolve the bridge from ``git rev-parse --show-toplevel`` (Codex cwd)."""
    return (
        'bash -lc \'root="$(git rev-parse --show-toplevel)"; '
        f'exec python3 "$root/{BRIDGE_REL}" {mode}\'"'
    )


def generate_cursor_hooks(_root: Path, *, timeout_seconds: int) -> dict[str, Any]:
    return {
        "version": CURSOR_HOOK_VERSION,
        "hooks": {
            "stop": [
                {**_command_hook("cursor-stop", timeout_seconds), "failClosed": True}
            ]
        },
    }


def _command_hook(mode: str, timeout_seconds: int) -> dict[str, Any]:
    return {
        "type": "command",
        "command": bridge_command(mode),
        "timeout": timeout_seconds,
    }


def fleet_claude_stop_hook(timeout_seconds: int) -> list[dict[str, Any]]:
    return [{"hooks": [_command_hook("claude-stop", timeout_seconds)]}]


def merge_claude_settings(
    existing: dict[str, Any], *, timeout_seconds: int
) -> dict[str, Any]:
    require(isinstance(existing, dict), "settings must be an object")
    merged = dict(existing)
    hooks = dict(merged.get("hooks") or {})
    stop_entries = list(hooks.get("Stop") or [])
    fleet_cmd = bridge_command("claude-stop")
    if not _claude_has_command(stop_entries, fleet_cmd):
        stop_entries.extend(fleet_claude_stop_hook(timeout_seconds))
    hooks["Stop"] = stop_entries
    merged["hooks"] = hooks
    return merged


def _claude_has_command(stop_entries: list[Any], command: str) -> bool:
    for entry in stop_entries:
        if not isinstance(entry, dict):
            continue
        for hook in entry.get("hooks") or []:
            if isinstance(hook, dict) and hook.get("command") == command:
                return True
    return False


def validate_cursor_hooks(doc: Any) -> None:
    require(isinstance(doc, dict), "cursor hooks must be an object")
    require(doc.get("version") == CURSOR_HOOK_VERSION, "cursor hooks.version must be 1")
    hooks = doc.get("hooks")
    require(isinstance(hooks, dict), "cursor hooks.hooks must be an object")
    stop = hooks.get("stop")
    require(isinstance(stop, list), "cursor hooks.stop must be a list")
    require(bool(stop), "cursor hooks.stop required")
    for item in stop:
        require(isinstance(item, dict), "cursor stop entry must be an object")
        require(
            item.get("type") in {None, "command"}, "cursor stop must be command type"
        )
        require(
            isinstance(item.get("command"), str) and item["command"], "command required"
        )


def _command_hook_git_root(mode: str, timeout_seconds: int) -> dict[str, Any]:
    return {
        "type": "command",
        "command": bridge_command_git_root(mode),
        "timeout": timeout_seconds,
    }


def generate_codex_hooks(_root: Path, *, timeout_seconds: int) -> dict[str, Any]:
    return {
        "description": f"Fleet agent-check hooks (Codex CLI pin {CODEX_CLI_PIN}).",
        "hooks": {
            "Stop": [{"hooks": [_command_hook_git_root("codex-stop", timeout_seconds)]}]
        },
    }


def generate_antigravity_hooks(_root: Path, *, timeout_seconds: int) -> dict[str, Any]:
    return {
        ANTIGRAVITY_HOOK_GROUP: {
            "PostToolUse": [
                {
                    "matcher": "*",
                    "hooks": [
                        _command_hook_git_root("antigravity-post-tool", timeout_seconds)
                    ],
                }
            ],
            "Stop": [_command_hook_git_root("antigravity-stop", timeout_seconds)],
        }
    }


def merge_gemini_settings(
    existing: dict[str, Any], *, timeout_seconds: int
) -> dict[str, Any]:
    require(isinstance(existing, dict), "gemini settings must be an object")
    merged = dict(existing)
    hooks = dict(merged.get("hooks") or {})
    hooks["AfterTool"] = [
        {
            "matcher": "*",
            "hooks": [
                _command_hook_git_root("gemini-after-tool", timeout_seconds),
            ],
        }
    ]
    hooks["AfterAgent"] = [
        {
            "hooks": [
                _command_hook_git_root("gemini-after-agent", timeout_seconds),
            ],
        }
    ]
    merged["hooks"] = hooks
    return merged


def generate_pre_push_sample() -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'root="$(git rev-parse --show-toplevel)"\n'
        'exec python3 "$root/scripts/pre_push_agent_check.py"\n'
    )


def validate_codex_hooks(doc: Any) -> None:
    require(isinstance(doc, dict), "codex hooks must be an object")
    hooks = doc.get("hooks")
    require(isinstance(hooks, dict), "codex hooks.hooks required")
    stop = hooks.get("Stop")
    require(isinstance(stop, list), "codex hooks.Stop must be a list")
    require(bool(stop), "codex hooks.Stop required")
    entry = stop[0]
    require(isinstance(entry, dict), "codex Stop entry must be an object")
    inner = entry.get("hooks")
    require(isinstance(inner, list), "codex Stop.hooks must be a list")
    require(bool(inner), "codex Stop.hooks required")


def validate_antigravity_hooks(doc: Any) -> None:
    require(isinstance(doc, dict), "antigravity hooks must be an object")
    group = doc.get(ANTIGRAVITY_HOOK_GROUP)
    require(isinstance(group, dict), "antigravity hook group required")
    require(isinstance(group.get("PostToolUse"), list), "PostToolUse required")
    stop = group.get("Stop")
    require(isinstance(stop, list) and bool(stop), "Stop required")
    handler = stop[0]
    require(isinstance(handler, dict), "Stop handler must be an object")
    require(
        isinstance(handler.get("command"), str) and handler["command"],
        "Stop command required",
    )


def validate_gemini_settings(doc: Any) -> None:
    require(isinstance(doc, dict), "gemini settings must be an object")
    hooks = doc.get("hooks")
    require(isinstance(hooks, dict), "gemini hooks required")
    after_tool = hooks.get("AfterTool")
    require(isinstance(after_tool, list) and bool(after_tool), "AfterTool required")
    after_agent = hooks.get("AfterAgent")
    require(isinstance(after_agent, list) and bool(after_agent), "AfterAgent required")


def validate_claude_settings(doc: Any) -> None:
    require(isinstance(doc, dict), "claude settings must be an object")
    hooks = doc.get("hooks")
    if hooks is None:
        return
    require(isinstance(hooks, dict), "claude hooks must be an object")
    stop = hooks.get("Stop")
    if stop is None:
        return
    require(isinstance(stop, list), "claude Stop must be a list")
    for entry in stop:
        require(isinstance(entry, dict), "claude Stop entry must be an object")
        inner = entry.get("hooks")
        require(isinstance(inner, list), "claude Stop.hooks must be a list")
        require(bool(inner), "claude Stop.hooks required")
        for hook in inner:
            require(isinstance(hook, dict), "claude hook must be an object")
            require(hook.get("type") == "command", "claude hook type must be command")
            require(
                isinstance(hook.get("command"), str), "claude hook command required"
            )


def _write_json(path: Path, doc: dict[str, Any]) -> WriteResult:
    text = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    changed = not path.is_file() or path.read_text(encoding="utf-8") != text
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return WriteResult(path.as_posix(), changed)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    parsed: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        require(False, f"{path} must be a JSON object")
    return cast(dict[str, Any], parsed)


def generate_adapters(  # noqa: C901, PLR0915
    root: Path,
    *,
    targets: frozenset[str],
    write: bool = True,
) -> dict[str, WriteResult]:
    unknown = targets - SUPPORTED_TARGETS
    require(not unknown, f"unknown adapter targets: {sorted(unknown)}")
    manifest = load_manifest(root)
    timeout = default_timeout(manifest)
    results: dict[str, WriteResult] = {}
    if "cursor" in targets:
        doc = generate_cursor_hooks(root, timeout_seconds=timeout)
        validate_cursor_hooks(doc)
        path = root / ".cursor/hooks.json"
        results["cursor"] = (
            _write_json(path, doc)
            if write
            else WriteResult(path.as_posix(), _differs(path, doc))
        )
    if "claude" in targets:
        settings_path = root / ".claude/settings.json"
        existing = _read_json_object(settings_path) if settings_path.is_file() else {}
        if not existing:
            existing = {"outputStyle": "Proactive"}
        merged = merge_claude_settings(existing, timeout_seconds=timeout)
        validate_claude_settings(merged)
        text = json.dumps(merged, indent=2, ensure_ascii=False) + "\n"
        changed = (
            not settings_path.is_file()
            or settings_path.read_text(encoding="utf-8") != text
        )
        if write and changed:
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings_path.write_text(text, encoding="utf-8")
        results["claude"] = WriteResult(settings_path.as_posix(), changed)
    if "codex" in targets:
        doc = generate_codex_hooks(root, timeout_seconds=timeout)
        validate_codex_hooks(doc)
        path = root / ".codex/hooks.json"
        results["codex"] = (
            _write_json(path, doc)
            if write
            else WriteResult(path.as_posix(), _differs(path, doc))
        )
    if "antigravity" in targets:
        doc = generate_antigravity_hooks(root, timeout_seconds=timeout)
        validate_antigravity_hooks(doc)
        path = root / ".agents/hooks.json"
        results["antigravity"] = (
            _write_json(path, doc)
            if write
            else WriteResult(path.as_posix(), _differs(path, doc))
        )
    if "gemini" in targets:
        settings_path = root / ".gemini/settings.json"
        existing = _read_json_object(settings_path) if settings_path.is_file() else {}
        merged = merge_gemini_settings(existing, timeout_seconds=timeout)
        validate_gemini_settings(merged)
        text = json.dumps(merged, indent=2, ensure_ascii=False) + "\n"
        changed = (
            not settings_path.is_file()
            or settings_path.read_text(encoding="utf-8") != text
        )
        if write and changed:
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings_path.write_text(text, encoding="utf-8")
        results["gemini"] = WriteResult(settings_path.as_posix(), changed)
    if "pre-push" in targets:
        path = root / "fleet-hooks/pre-push.sample"
        text = generate_pre_push_sample()
        changed = not path.is_file() or path.read_text(encoding="utf-8") != text
        if write and changed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        results["pre-push"] = WriteResult(path.as_posix(), changed)
    return results


def _differs(path: Path, doc: dict[str, Any]) -> bool:
    expected = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    return not path.is_file() or path.read_text(encoding="utf-8") != expected


def check_adapters(root: Path, *, targets: frozenset[str]) -> list[str]:  # noqa: C901
    manifest = load_manifest(root)
    timeout = default_timeout(manifest)
    drift: list[str] = []
    if "cursor" in targets:
        path = root / ".cursor/hooks.json"
        doc = generate_cursor_hooks(root, timeout_seconds=timeout)
        if _differs(path, doc):
            drift.append(path.as_posix())
    if "claude" in targets:
        path = root / ".claude/settings.json"
        existing = (
            _read_json_object(path) if path.is_file() else {"outputStyle": "Proactive"}
        )
        doc = merge_claude_settings(existing, timeout_seconds=timeout)
        expected = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            drift.append(path.as_posix())
    if "codex" in targets:
        path = root / ".codex/hooks.json"
        doc = generate_codex_hooks(root, timeout_seconds=timeout)
        if _differs(path, doc):
            drift.append(path.as_posix())
    if "antigravity" in targets:
        path = root / ".agents/hooks.json"
        doc = generate_antigravity_hooks(root, timeout_seconds=timeout)
        if _differs(path, doc):
            drift.append(path.as_posix())
    if "gemini" in targets:
        path = root / ".gemini/settings.json"
        existing = _read_json_object(path) if path.is_file() else {}
        doc = merge_gemini_settings(existing, timeout_seconds=timeout)
        expected = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            drift.append(path.as_posix())
    if "pre-push" in targets:
        path = root / "fleet-hooks/pre-push.sample"
        if (
            not path.is_file()
            or path.read_text(encoding="utf-8") != generate_pre_push_sample()
        ):
            drift.append(path.as_posix())
    return drift
