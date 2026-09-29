#!/usr/bin/env python3
"""Validate explicit Tauri v2 IPC capabilities for backend commands."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path

HANDLER_RE = re.compile(r"generate_handler!\[(?P<body>.*?)\]", re.DOTALL)


def _kebab(command: str) -> str:
    return command.replace("_", "-")


def _registered_commands(lib_rs: Path) -> set[str]:
    text = lib_rs.read_text(encoding="utf-8")
    match = HANDLER_RE.search(text)
    if not match:
        raise ValueError("tauri::generate_handler![...] not found")
    return {
        token.strip()
        for token in match.group("body").replace("\n", " ").split(",")
        if token.strip()
    }


def _local_permission_commands(tauri_root: Path) -> dict[str, set[str]]:
    permissions_root = tauri_root / "permissions"
    if not permissions_root.exists():
        return {}

    permissions: dict[str, set[str]] = {}
    for path in sorted(permissions_root.glob("*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        for permission in data.get("permission", []):
            identifier = permission.get("identifier")
            if not isinstance(identifier, str):
                continue
            commands = permission.get("commands", {})
            allowed = commands.get("allow", []) if isinstance(commands, dict) else []
            permissions[identifier] = {
                command for command in allowed if isinstance(command, str)
            }
    return permissions


def tauri_capability_failures(root: Path = Path(".")) -> list[str]:
    tauri_root = root / "ui" / "src-tauri"
    commands = _registered_commands(tauri_root / "src" / "lib.rs")
    local_permissions = _local_permission_commands(tauri_root)
    capability_path = tauri_root / "capabilities" / "main.json"
    if not capability_path.exists():
        return [f"missing Tauri capability file: {capability_path}"]

    data = json.loads(capability_path.read_text(encoding="utf-8"))
    permissions = data.get("permissions")
    if not isinstance(permissions, list):
        return ["Tauri capability permissions must be a list"]
    permission_set = set(permissions)
    failures: list[str] = []
    if "*" in permission_set:
        failures.append("Tauri capabilities must not use wildcard permissions")
    for required in ("core:default", "log:default"):
        if required not in permission_set:
            failures.append(f"missing Tauri permission {required}")
    for command in sorted(commands):
        permission = f"allow-{_kebab(command)}"
        if permission not in permission_set:
            failures.append(f"missing Tauri permission {permission}")
        allowed_commands = local_permissions.get(permission)
        if allowed_commands is None:
            failures.append(f"missing Tauri permission definition {permission}")
        elif command not in allowed_commands:
            failures.append(
                f"Tauri permission {permission} must allow command {command}"
            )
    if data.get("windows") != ["main"]:
        failures.append("Tauri capability must be scoped to the main window")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    failures = tauri_capability_failures(args.root)
    if failures:
        print("\n".join(failures))
        return 1
    print("Tauri IPC capabilities are explicit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
