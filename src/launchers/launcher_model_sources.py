"""Compatibility facade for provider-aware launcher model source helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.shared.python.config.model_source_providers import (
    iter_unique_python_path_strings,
)
from src.shared.python.config.model_source_providers import (
    resolve_model_artifact as resolve_provider_model_artifact,
)
from src.shared.python.config.model_source_providers import (
    resolve_model_python_paths as resolve_provider_model_python_paths,
)
from src.shared.python.config.model_source_providers import (
    resolve_model_source_root as resolve_provider_model_source_root,
)
from src.shared.python.config.model_source_providers import (
    resolve_model_working_directory as resolve_provider_model_working_directory,
)


def get_model_source_root(model: Any, default_root: Path) -> Path:
    """Resolve the repo/provider root for a model launch."""
    return resolve_provider_model_source_root(model, default_root)


def resolve_model_artifact_path(model: Any, default_root: Path) -> Path:
    """Resolve a model's path relative to its source root."""
    return resolve_provider_model_artifact(model, default_root)


def get_model_working_directory(
    model: Any,
    default_root: Path,
    fallback_relative: str | Path | None = None,
) -> Path:
    """Resolve the preferred working directory for launching a model."""
    return resolve_provider_model_working_directory(
        model,
        default_root,
        fallback_relative=fallback_relative,
    )


def get_model_python_paths(model: Any, default_root: Path) -> tuple[Path, ...]:
    """Resolve extra PYTHONPATH entries declared by a model source."""
    return resolve_provider_model_python_paths(model, default_root)


__all__ = [
    "get_model_python_paths",
    "get_model_source_root",
    "get_model_working_directory",
    "iter_unique_python_path_strings",
    "resolve_model_artifact_path",
]
