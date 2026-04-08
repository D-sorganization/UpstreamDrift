"""Compatibility harness for launcher model providers.

This module validates that local and provider-backed model entries expose
enough metadata for the launcher to resolve assets and execution context
without relying on hardcoded repo-local assumptions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.launchers.launcher_model_sources import (
    get_model_python_paths,
    get_model_source_root,
    get_model_working_directory,
    resolve_model_artifact_path,
)


@dataclass(frozen=True)
class LauncherCompatibilityResult:
    """Resolved launcher compatibility state for a single model entry."""

    model_id: str
    provider: str
    source_root: Path
    artifact_path: Path | None
    working_dir: Path
    python_paths: tuple[Path, ...]
    issues: tuple[str, ...] = ()

    @property
    def is_compatible(self) -> bool:
        """Return True when the launcher can safely resolve this model."""
        return len(self.issues) == 0


def evaluate_launcher_model_compatibility(
    models: Iterable[Any],
    repo_root: Path,
) -> tuple[LauncherCompatibilityResult, ...]:
    """Resolve launcher compatibility for a sequence of local/provider models."""
    results: list[LauncherCompatibilityResult] = []

    for model in models:
        model_id = getattr(model, "id", "unknown")
        provider = getattr(model, "provider", None)
        provider_name = provider if isinstance(provider, str) and provider else "local"
        source_root = get_model_source_root(model, repo_root)
        working_dir = get_model_working_directory(model, repo_root)
        python_paths = get_model_python_paths(model, repo_root)

        issues: list[str] = []
        artifact_path: Path | None = None
        try:
            artifact_path = resolve_model_artifact_path(model, repo_root)
        except ValueError as exc:
            issues.append(str(exc))

        if not source_root.exists():
            issues.append(f"source root does not exist: {source_root}")
        if artifact_path is not None and not artifact_path.exists():
            issues.append(f"artifact path does not exist: {artifact_path}")
        if not working_dir.exists():
            issues.append(f"working directory does not exist: {working_dir}")

        missing_python_paths = [path for path in python_paths if not path.exists()]
        for path in missing_python_paths:
            issues.append(f"python path does not exist: {path}")

        results.append(
            LauncherCompatibilityResult(
                model_id=str(model_id),
                provider=provider_name,
                source_root=source_root,
                artifact_path=artifact_path,
                working_dir=working_dir,
                python_paths=python_paths,
                issues=tuple(issues),
            )
        )

    return tuple(results)


def assert_launcher_provider_compatibility(
    models: Iterable[Any],
    repo_root: Path,
) -> tuple[LauncherCompatibilityResult, ...]:
    """Validate model-provider compatibility and raise on the first failures."""
    results = evaluate_launcher_model_compatibility(models, repo_root)
    failures = [result for result in results if not result.is_compatible]
    if failures:
        details = "; ".join(
            f"{failure.model_id}: {', '.join(failure.issues)}" for failure in failures
        )
        raise ValueError(f"Launcher provider compatibility failed: {details}")
    return results
