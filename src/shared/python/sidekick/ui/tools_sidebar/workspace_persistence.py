"""JSON persistence helpers for Sidekick workspace registries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .registry import WorkspaceRegistry, WorkspaceVariable

CALCULATOR_WORKSPACE_FORMAT_VERSION = 1


def validate_calculator_workspace_path(path: str | Path) -> Path:
    """Return a normalized JSON workspace path or raise a user-facing error."""
    if path is None:
        raise ValueError("workspace path is required")
    candidate = Path(path).expanduser()
    if candidate.exists() and candidate.is_dir():
        raise ValueError("workspace path must be a file, not a directory")
    if candidate.suffix.lower() != ".json":
        raise ValueError("calculator workspace files must use a .json suffix")
    return candidate


def save_workspace_registry(
    registry: WorkspaceRegistry,
    path: str | Path,
    *,
    scope: str,
) -> Path:
    target = validate_calculator_workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = registry.to_dict()
    payload["version"] = CALCULATOR_WORKSPACE_FORMAT_VERSION
    payload["scope"] = scope
    temp = target.with_name(f".{target.name}.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(target)
    return target


def load_workspace_registry(
    registry: WorkspaceRegistry,
    path: str | Path,
    *,
    expected_scope: str,
    replace: bool,
    confirm_replace: bool,
) -> tuple[WorkspaceVariable, ...]:
    if replace and not confirm_replace:
        raise PermissionError("replace load requires explicit confirmation")
    incoming = _registry_from_payload(
        validate_calculator_workspace_path(path),
        expected_scope=expected_scope,
    )
    imported = tuple(incoming.variables())
    registry.update_from(incoming, replace=replace)
    return imported


def _registry_from_payload(
    path: Path,
    *,
    expected_scope: str,
) -> WorkspaceRegistry:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("workspace file is not valid JSON") from exc
    _validate_workspace_payload(payload, expected_scope=expected_scope)
    try:
        return WorkspaceRegistry.load_json(path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("workspace file is not valid JSON") from exc


def _validate_workspace_payload(payload: Any, *, expected_scope: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError("workspace file must contain an object")
    if payload.get("version") != CALCULATOR_WORKSPACE_FORMAT_VERSION:
        raise ValueError("unsupported workspace version")
    if payload.get("scope") != expected_scope:
        raise ValueError(f"workspace scope must be {expected_scope}")
    variables = payload.get("variables")
    if not isinstance(variables, list):
        raise ValueError("workspace variables must be a list")
    for entry in variables:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError("workspace variables must contain names")
