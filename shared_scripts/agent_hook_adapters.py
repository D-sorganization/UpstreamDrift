# ruff: noqa
# fmt: off
# flake8: noqa
# mypy: ignore-errors
"""Generate provider adapters for the installed trusted hook runtime."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Callable
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from shared_scripts.agent_hook_callback_ownership import provider_callback_commands
from shared_scripts.agent_hook_runtime import (
    ANTIGRAVITY_HOOK_GROUP,
    CURSOR_HOOK_VERSION,
    AdapterError,
    HookRuntime,
    TrustedRuntime,
    WriteResult,
    load_runtime_descriptor,
    legacy_managed_command,
    preserve_unmanaged_entries as _preserve_unmanaged_entries,
    require,
    validate_provider_paths,
    write_text_atomic as _write_text_atomic,
)
from shared_scripts.agent_hook_schemas import (
    validate_antigravity_hooks,
    validate_claude_settings,
    validate_codex_hooks,
    validate_cursor_hooks,
    validate_gemini_settings,
)

SUPPORTED_TARGETS = frozenset(
    {"cursor", "claude", "codex", "antigravity", "gemini", "pre-push"}
)


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


def bridge_command(mode: str, runtime: HookRuntime) -> str:
    return runtime.command(mode)


def _preserve(entries: Any, runtime: HookRuntime, *modes: str) -> list[Any]:
    owned = frozenset(bridge_command(mode, runtime) for mode in modes)
    return _preserve_unmanaged_entries(entries, owned_commands=owned)


def generate_cursor_hooks(
    _root: Path, *, timeout_seconds: int, runtime: HookRuntime,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hooks = dict((existing or {}).get("hooks") or {})
    stop = _preserve(hooks.get("stop", []), runtime, "cursor-stop")
    stop.append({
        **_command_hook("cursor-stop", timeout_seconds, runtime),
        "failClosed": True,
        "loop_limit": 2,
    })
    hooks["stop"] = stop
    return {**(existing or {}), "version": CURSOR_HOOK_VERSION, "hooks": hooks}


def _command_hook(mode: str, timeout_seconds: int, runtime: HookRuntime) -> dict[str, Any]:
    return {
        "type": "command",
        "command": bridge_command(mode, runtime),
        "timeout": timeout_seconds,
    }


def fleet_claude_stop_hook(timeout_seconds: int, runtime: HookRuntime) -> list[dict[str, Any]]:
    return [{"hooks": [_command_hook("claude-stop", timeout_seconds, runtime)]}]


def merge_claude_settings(
    existing: dict[str, Any], *, timeout_seconds: int, runtime: HookRuntime
) -> dict[str, Any]:
    require(isinstance(existing, dict), "settings must be an object")
    merged = dict(existing)
    hooks = dict(merged.get("hooks") or {})
    stop_entries = _preserve(hooks.get("Stop", []), runtime, "claude-stop")
    fleet_cmd = bridge_command("claude-stop", runtime)
    if not _claude_has_command(stop_entries, fleet_cmd):
        stop_entries.extend(fleet_claude_stop_hook(timeout_seconds, runtime))
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


def _command_hook_git_root(mode: str, timeout_seconds: int, runtime: HookRuntime) -> dict[str, Any]:
    return {
        "type": "command",
        "command": bridge_command(mode, runtime),
        "timeout": timeout_seconds,
    }


def generate_codex_hooks(
    _root: Path, *, timeout_seconds: int, runtime: HookRuntime,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc = dict(existing or {})
    hooks = dict(doc.get("hooks") or {})
    stop = _preserve(hooks.get("Stop", []), runtime, "codex-stop")
    stop.append({"hooks": [_command_hook_git_root("codex-stop", timeout_seconds, runtime)]})
    hooks["Stop"] = stop
    doc.setdefault("description", "Fleet trusted agent-check hooks.")
    doc["hooks"] = hooks
    return doc


def generate_antigravity_hooks(
    _root: Path, *, timeout_seconds: int, runtime: HookRuntime,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc = dict(existing or {})
    group = dict(doc.get(ANTIGRAVITY_HOOK_GROUP) or {})
    post_tool = _preserve(
        group.get("PostToolUse", []), runtime, "antigravity-post-tool"
    )
    post_tool.append(
        {
            "matcher": "*",
            "hooks": [
                _command_hook_git_root(
                    "antigravity-post-tool", timeout_seconds, runtime
                )
            ],
        }
    )
    stop = _preserve(group.get("Stop", []), runtime, "antigravity-stop")
    stop.append(_command_hook_git_root("antigravity-stop", timeout_seconds, runtime))
    group["PostToolUse"] = post_tool
    group["Stop"] = stop
    doc[ANTIGRAVITY_HOOK_GROUP] = group
    return doc


def merge_gemini_settings(
    existing: dict[str, Any], *, timeout_seconds: int, runtime: HookRuntime
) -> dict[str, Any]:
    require(isinstance(existing, dict), "gemini settings must be an object")
    merged = dict(existing)
    hooks = dict(merged.get("hooks") or {})
    after_tool = _preserve(hooks.get("AfterTool", []), runtime, "gemini-after-tool")
    after_agent = _preserve(
        hooks.get("AfterAgent", []), runtime, "gemini-after-agent"
    )
    after_tool.append(
        {
            "matcher": "*",
            "hooks": [
                _command_hook_git_root("gemini-after-tool", timeout_seconds, runtime),
            ],
        }
    )
    after_agent.append(
        {
            "hooks": [
                _command_hook_git_root("gemini-after-agent", timeout_seconds, runtime),
            ],
        }
    )
    hooks["AfterTool"] = after_tool
    hooks["AfterAgent"] = after_agent
    merged["hooks"] = hooks
    return merged


def generate_pre_push_sample() -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'root="$(git rev-parse --show-toplevel)"\n'
        'exec python3 "$root/scripts/pre_push_agent_check.py"\n'
    )


def _write_json(
    path: Path,
    doc: dict[str, Any],
    *,
    writer: Callable[[Path, str], None] = _write_text_atomic,
) -> WriteResult:
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise AdapterError(f"refusing symlinked provider config path: {path}")
    text = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    changed = not path.is_file() or path.read_text(encoding="utf-8") != text
    if changed:
        writer(path, text)
    return WriteResult(path.as_posix(), changed)


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise AdapterError(f"refusing symlinked provider config path: {path}")
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
    runtime: HookRuntime,
    write: bool = True,
    writer: Callable[[Path, str], None] = _write_text_atomic,
) -> dict[str, WriteResult]:
    unknown = targets - SUPPORTED_TARGETS
    require(not unknown, f"unknown adapter targets: {sorted(unknown)}")
    runtime.validate(root)
    timeout = 120
    results: dict[str, WriteResult] = {}
    if "cursor" in targets:
        path = root / ".cursor/hooks.json"
        existing = _read_json_object(path) if path.is_file() else {}
        doc = generate_cursor_hooks(root, timeout_seconds=timeout, runtime=runtime, existing=existing)
        validate_cursor_hooks(doc)
        results["cursor"] = (
            _write_json(path, doc, writer=writer)
            if write
            else WriteResult(path.as_posix(), _differs(path, doc))
        )
    if "claude" in targets:
        settings_path = root / ".claude/settings.json"
        existing = _read_json_object(settings_path) if settings_path.is_file() else {}
        if not existing:
            existing = {"outputStyle": "Proactive"}
        merged = merge_claude_settings(existing, timeout_seconds=timeout, runtime=runtime)
        validate_claude_settings(merged)
        text = json.dumps(merged, indent=2, ensure_ascii=False) + "\n"
        changed = (
            not settings_path.is_file()
            or settings_path.read_text(encoding="utf-8") != text
        )
        if write and changed:
            writer(settings_path, text)
        results["claude"] = WriteResult(settings_path.as_posix(), changed)
    if "codex" in targets:
        path = root / ".codex/hooks.json"
        existing = _read_json_object(path) if path.is_file() else {}
        doc = generate_codex_hooks(root, timeout_seconds=timeout, runtime=runtime, existing=existing)
        validate_codex_hooks(doc)
        results["codex"] = (
            _write_json(path, doc, writer=writer)
            if write
            else WriteResult(path.as_posix(), _differs(path, doc))
        )
    if "antigravity" in targets:
        path = root / ".agents/hooks.json"
        existing = _read_json_object(path) if path.is_file() else {}
        doc = generate_antigravity_hooks(root, timeout_seconds=timeout, runtime=runtime, existing=existing)
        validate_antigravity_hooks(doc)
        results["antigravity"] = (
            _write_json(path, doc, writer=writer)
            if write
            else WriteResult(path.as_posix(), _differs(path, doc))
        )
    if "gemini" in targets:
        settings_path = root / ".gemini/settings.json"
        existing = _read_json_object(settings_path) if settings_path.is_file() else {}
        merged = merge_gemini_settings(existing, timeout_seconds=timeout, runtime=runtime)
        validate_gemini_settings(merged)
        text = json.dumps(merged, indent=2, ensure_ascii=False) + "\n"
        changed = (
            not settings_path.is_file()
            or settings_path.read_text(encoding="utf-8") != text
        )
        if write and changed:
            writer(settings_path, text)
        results["gemini"] = WriteResult(settings_path.as_posix(), changed)
    if "pre-push" in targets:
        path = root / "fleet-hooks/pre-push.sample"
        text = generate_pre_push_sample()
        changed = not path.is_file() or path.read_text(encoding="utf-8") != text
        if write and changed:
            writer(path, text)
        results["pre-push"] = WriteResult(path.as_posix(), changed)
    return results


def _differs(path: Path, doc: dict[str, Any]) -> bool:
    expected = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    return not path.is_file() or path.read_text(encoding="utf-8") != expected


def check_adapters(
    root: Path, *, targets: frozenset[str], runtime: HookRuntime | None = None
) -> list[str]:
    """Report checkout-executing adapters unless given a trusted host runtime."""
    if runtime is None:
        legacy: list[str] = []
        paths = {
            "cursor": ".cursor/hooks.json",
            "claude": ".claude/settings.json",
            "codex": ".codex/hooks.json",
            "antigravity": ".agents/hooks.json",
            "gemini": ".gemini/settings.json",
        }
        for target, rel in paths.items():
            path = root / rel
            if target in targets and path.is_file():
                try:
                    document = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    legacy.append(path.as_posix())
                    continue
                commands = provider_callback_commands(target, document)
                if any(legacy_managed_command(command) for command in commands):
                    legacy.append(path.as_posix())
        return legacy
    runtime.validate(root)
    drift: list[str] = []
    managed = {
        "cursor": (".cursor/hooks.json", ("cursor-stop",)),
        "claude": (".claude/settings.json", ("claude-stop",)),
        "codex": (".codex/hooks.json", ("codex-stop",)),
        "antigravity": (".agents/hooks.json", ("antigravity-stop", "antigravity-post-tool")),
        "gemini": (".gemini/settings.json", ("gemini-after-tool", "gemini-after-agent")),
    }
    for target, (relative, modes) in managed.items():
        if target not in targets:
            continue
        path = root / relative
        try:
            document = _read_json_object(path) if path.is_file() else {}
        except (OSError, json.JSONDecodeError, AdapterError):
            document = {}
        commands = provider_callback_commands(target, document)
        if any(legacy_managed_command(command) for command in commands):
            drift.append(path.as_posix())
        elif any(runtime.command(mode) not in commands for mode in modes):
            drift.append(path.as_posix())
    if "pre-push" in targets:
        path = root / "fleet-hooks/pre-push.sample"
        if (
            not path.is_file()
            or path.read_text(encoding="utf-8") != generate_pre_push_sample()
        ):
            drift.append(path.as_posix())
    return drift
