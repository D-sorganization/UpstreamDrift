"""Provider-aware model source helpers for launcher integration.

These helpers centralize how launcher code resolves a model's source root,
working directory, artifact path, and supplemental PYTHONPATH entries.
They preserve the current repo-local behaviour by default while allowing
future external provider packs to opt into their own roots and execution
contexts.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _get_optional_string_attr(model: Any, attr_name: str) -> str | None:
    """Return a stripped string attribute when the model explicitly provides one."""
    value = getattr(model, attr_name, None)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def get_model_source_root(model: Any, default_root: Path) -> Path:
    """Resolve the repo/provider root for a model launch."""
    source_root = _get_optional_string_attr(model, "source_root")
    if source_root is None:
        return default_root

    root_path = Path(source_root)
    if root_path.is_absolute():
        return root_path
    return default_root / root_path


def resolve_model_artifact_path(model: Any, default_root: Path) -> Path:
    """Resolve a model's path relative to its source root."""
    model_path = _get_optional_string_attr(model, "path")
    if model_path is None:
        raise ValueError("model.path must be a non-empty string")

    artifact_path = Path(model_path)
    if artifact_path.is_absolute():
        return artifact_path
    return get_model_source_root(model, default_root) / artifact_path


def get_model_working_directory(
    model: Any,
    default_root: Path,
    fallback_relative: str | Path | None = None,
) -> Path:
    """Resolve the preferred working directory for launching a model."""
    source_root = get_model_source_root(model, default_root)
    working_dir = _get_optional_string_attr(model, "working_dir")
    if working_dir is not None:
        working_path = Path(working_dir)
        if working_path.is_absolute():
            return working_path
        return source_root / working_path

    if fallback_relative is None:
        return source_root

    fallback_path = Path(fallback_relative)
    if fallback_path.is_absolute():
        return fallback_path
    return source_root / fallback_path


def get_model_python_paths(model: Any, default_root: Path) -> tuple[Path, ...]:
    """Resolve extra PYTHONPATH entries declared by a model source."""
    raw_values = getattr(model, "python_paths", ())
    if not isinstance(raw_values, (list, tuple)):
        return ()

    source_root = get_model_source_root(model, default_root)
    resolved: list[Path] = []
    seen: set[Path] = set()

    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            continue
        item = raw_value.strip()
        if not item:
            continue
        path = Path(item)
        candidate = path if path.is_absolute() else source_root / path
        if candidate in seen:
            continue
        seen.add(candidate)
        resolved.append(candidate)

    return tuple(resolved)


def iter_unique_python_path_strings(
    base_paths: Iterable[str],
    extra_paths: Iterable[str],
) -> tuple[str, ...]:
    """Merge stringified python paths while preserving first-seen order."""
    seen: set[str] = set()
    merged: list[str] = []

    for item in [*base_paths, *extra_paths]:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)

    return tuple(merged)
