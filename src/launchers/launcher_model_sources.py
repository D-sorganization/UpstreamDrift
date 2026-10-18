"""Compatibility facade for provider-aware launcher model source helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.shared.python.config.model_source_providers import (
    ModelSourcePathPolicy,
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


def require_model_source_root(model: Any, default_root: Path) -> Path:
    """Resolve and require the existence of a model's declared source root."""
    declared = getattr(model, "source_root", None)
    if declared:
        policy = ModelSourcePathPolicy(default_root)
        return policy.require_source_root(str(declared))
    return get_model_source_root(model, default_root)


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


def package_main_module_name(
    model_path: str,
    *,
    source_root: Path,
    launcher_root: Path,
) -> str | None:
    """Return the importable module for a package ``__main__`` launcher path.

    Local ``src/tools`` tiles need the ``src`` package prefix because the
    launcher places the repository root before ``repo/src`` on ``PYTHONPATH``
    and the repository root has another ``tools`` package.  Other packages
    retain their established ``repo/src`` import style.  Sibling providers
    expose their own ``src`` directory as an extra Python path, so their
    package modules deliberately omit that prefix.
    """
    normalized_path = model_path.replace("\\", "/")
    if not normalized_path.startswith("src/") or not normalized_path.endswith(
        "/__main__.py"
    ):
        return None
    module_parts = normalized_path.removesuffix("/__main__.py").split("/")
    is_local_tool_package = (
        source_root.resolve() == launcher_root.resolve()
        and module_parts[:2] == ["src", "tools"]
    )
    if not is_local_tool_package:
        module_parts = module_parts[1:]
    if not module_parts or not all(part.isidentifier() for part in module_parts):
        return None
    return ".".join(module_parts)


_package_main_module_name = package_main_module_name


__all__ = [
    "get_model_python_paths",
    "get_model_source_root",
    "get_model_working_directory",
    "iter_unique_python_path_strings",
    "package_main_module_name",
    "require_model_source_root",
    "resolve_model_artifact_path",
]
