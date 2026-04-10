"""Compatibility harness for launcher model providers.

This module validates that local and provider-backed model entries expose
enough metadata for the launcher to resolve assets and execution context
without relying on hardcoded repo-local assumptions.
"""

from __future__ import annotations

import importlib.util
import sys
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
from src.shared.python.config.model_pack_manifest import ModelPackManifest

_ENGINE_IMPORT_NAMES = {
    "drake": "pydrake",
    "mujoco": "mujoco",
    "opensim": "opensim",
    "pinocchio": "pinocchio",
    "myosuite": "myosuite",
}


@dataclass(frozen=True)
class CompatibilityIssue:
    """Machine-readable provider compatibility diagnostic."""

    code: str
    category: str
    message: str
    context: dict[str, str]


@dataclass(frozen=True)
class LauncherCompatibilityResult:
    """Resolved launcher compatibility state for a single model entry."""

    model_id: str
    provider: str
    canonical_id: str | None
    source_root: Path
    artifact_path: Path | None
    working_dir: Path
    python_paths: tuple[Path, ...]
    issues: tuple[CompatibilityIssue, ...] = ()

    @property
    def is_compatible(self) -> bool:
        """Return True when the launcher can safely resolve this model."""
        return len(self.issues) == 0


@dataclass(frozen=True)
class ProviderCompatibilityReport:
    """Manifest-level compatibility report for a provider pack."""

    provider: str
    manifest_path: Path
    pack_id: str | None
    results: tuple[LauncherCompatibilityResult, ...]
    issues: tuple[CompatibilityIssue, ...] = ()

    @property
    def is_compatible(self) -> bool:
        """Return True when the manifest and all models are compatible."""
        return len(self.issues) == 0 and all(
            result.is_compatible for result in self.results
        )


def is_engine_runtime_available(engine_type: str | None) -> bool:
    """Return whether the engine runtime backing a provider model is installed."""
    if engine_type is None:
        return True

    import_name = _ENGINE_IMPORT_NAMES.get(engine_type.strip().lower())
    if import_name is None:
        return True
    if import_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def _make_issue(
    code: str,
    category: str,
    message: str,
    **context: str | Path | None,
) -> CompatibilityIssue:
    """Create a normalized compatibility issue."""
    normalized_context = {
        key: str(value) for key, value in context.items() if value is not None
    }
    return CompatibilityIssue(
        code=code,
        category=category,
        message=message,
        context=normalized_context,
    )


def _requires_canonical_identity(model: Any) -> bool:
    """Return whether a provider model should expose cross-engine identity."""
    launcher = getattr(model, "launcher", None)
    launcher_category = getattr(launcher, "category", None)
    if isinstance(launcher_category, str) and launcher_category == "tool":
        return False

    engine_type = getattr(model, "engine_type", None)
    return isinstance(engine_type, str) and engine_type.strip() != ""


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
        identity = getattr(model, "identity", None)
        canonical_id = getattr(identity, "canonical_id", None)
        engine_type = getattr(model, "engine_type", None)
        source_root = get_model_source_root(model, repo_root)
        working_dir = get_model_working_directory(model, repo_root)
        python_paths = get_model_python_paths(model, repo_root)

        issues: list[CompatibilityIssue] = []
        artifact_path: Path | None = None
        try:
            artifact_path = resolve_model_artifact_path(model, repo_root)
        except ValueError as exc:
            issues.append(
                _make_issue(
                    "invalid_artifact_path",
                    "malformed_metadata",
                    str(exc),
                    model_id=str(model_id),
                    provider=provider_name,
                )
            )

        if not source_root.exists():
            issues.append(
                _make_issue(
                    "missing_source_root",
                    "malformed_metadata",
                    f"Source root does not exist: {source_root}",
                    model_id=str(model_id),
                    provider=provider_name,
                    source_root=source_root,
                )
            )
        if artifact_path is not None and not artifact_path.exists():
            issues.append(
                _make_issue(
                    "missing_artifact_path",
                    "malformed_metadata",
                    f"Artifact path does not exist: {artifact_path}",
                    model_id=str(model_id),
                    provider=provider_name,
                    artifact_path=artifact_path,
                )
            )
        if not working_dir.exists():
            issues.append(
                _make_issue(
                    "missing_working_directory",
                    "malformed_metadata",
                    f"Working directory does not exist: {working_dir}",
                    model_id=str(model_id),
                    provider=provider_name,
                    working_dir=working_dir,
                )
            )

        missing_python_paths = [path for path in python_paths if not path.exists()]
        for path in missing_python_paths:
            issues.append(
                _make_issue(
                    "missing_python_path",
                    "malformed_metadata",
                    f"Python path does not exist: {path}",
                    model_id=str(model_id),
                    provider=provider_name,
                    python_path=path,
                )
            )

        if not is_engine_runtime_available(
            engine_type if isinstance(engine_type, str) else None
        ):
            issues.append(
                _make_issue(
                    "runtime_unavailable",
                    "runtime_unavailable",
                    f"Engine runtime is unavailable for engine '{engine_type}'",
                    model_id=str(model_id),
                    provider=provider_name,
                    engine_type=engine_type if isinstance(engine_type, str) else None,
                )
            )

        results.append(
            LauncherCompatibilityResult(
                model_id=str(model_id),
                provider=provider_name,
                canonical_id=(
                    str(canonical_id)
                    if isinstance(canonical_id, str) and canonical_id.strip()
                    else None
                ),
                source_root=source_root,
                artifact_path=artifact_path,
                working_dir=working_dir,
                python_paths=python_paths,
                issues=tuple(issues),
            )
        )

    return tuple(results)


def validate_provider_manifest(
    manifest_path: str | Path,
    provider_root: Path | None = None,
) -> ProviderCompatibilityReport:
    """Validate a provider manifest and return a reusable compatibility report."""
    manifest_issues: list[CompatibilityIssue] = []
    manifest_file = Path(manifest_path)

    try:
        manifest = ModelPackManifest.load(manifest_file)
    except (OSError, ValueError, TypeError) as exc:
        manifest_issues.append(
            _make_issue(
                "invalid_manifest",
                "malformed_metadata",
                f"Manifest could not be loaded: {exc}",
                manifest_path=manifest_file,
            )
        )
        return ProviderCompatibilityReport(
            provider="unknown",
            manifest_path=manifest_file,
            pack_id=None,
            results=(),
            issues=tuple(manifest_issues),
        )

    root = provider_root if provider_root is not None else manifest_file.parent
    results = evaluate_launcher_model_compatibility(manifest.models, root)

    aggregated_issues: list[CompatibilityIssue] = []
    for result in results:
        aggregated_issues.extend(result.issues)
        model = next(
            (entry for entry in manifest.models if entry.id == result.model_id), None
        )
        if (
            model is not None
            and _requires_canonical_identity(model)
            and result.canonical_id is None
        ):
            aggregated_issues.append(
                _make_issue(
                    "missing_canonical_identity",
                    "malformed_metadata",
                    "Model is missing canonical cross-engine identity metadata",
                    model_id=result.model_id,
                    provider=result.provider,
                )
            )
        if model is not None and len(model.capabilities) == 0:
            aggregated_issues.append(
                _make_issue(
                    "missing_capabilities",
                    "malformed_metadata",
                    "Model must declare at least one capability",
                    model_id=result.model_id,
                    provider=result.provider,
                )
            )

    return ProviderCompatibilityReport(
        provider=manifest.provider,
        manifest_path=manifest_file,
        pack_id=manifest.pack_id,
        results=results,
        issues=(*manifest_issues, *aggregated_issues),
    )


def assert_launcher_provider_compatibility(
    models: Iterable[Any],
    repo_root: Path,
) -> tuple[LauncherCompatibilityResult, ...]:
    """Validate model-provider compatibility and raise on the first failures."""
    results = evaluate_launcher_model_compatibility(models, repo_root)
    failures = [result for result in results if not result.is_compatible]
    if failures:
        details = "; ".join(
            f"{failure.model_id}: {', '.join(issue.code for issue in failure.issues)}"
            for failure in failures
        )
        raise ValueError(f"Launcher provider compatibility failed: {details}")
    return results


def assert_provider_manifest_compatibility(
    manifest_path: str | Path,
    provider_root: Path | None = None,
) -> ProviderCompatibilityReport:
    """Validate a provider manifest and raise with machine-readable issue codes."""
    report = validate_provider_manifest(manifest_path, provider_root)
    if not report.is_compatible:
        issue_codes = [issue.code for issue in report.issues]
        for result in report.results:
            issue_codes.extend(
                f"{result.model_id}:{issue.code}" for issue in result.issues
            )
        raise ValueError(
            "Provider manifest compatibility failed: " + ", ".join(issue_codes)
        )
    return report
